from typing import Dict, Union, Sequence, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from cace.tools import scatter_sum

def build_mlp(layers,bias=False,n_out=1):
    mlp = []
    for l in layers:
        mlp += [nn.LazyLinear(l,bias=bias),nn.SiLU()]
    mlp += [nn.LazyLinear(n_out,bias=False)]
    return nn.Sequential(*mlp)
    
class AttentionAtomwise(nn.Module):
    def __init__(
        self,
        n_out: int = 1,
        n_hidden=[32,16],
        attention_hidden_nc=64,
        bias: bool = True,
        feature_key = 'node_feats',
        output_key = "pred_energy",
        avge0 = 0,
        sigma = 1,
    ):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.bias = bias
        self.feature_key = feature_key
        self.avge0 = avge0
        self.sigma = sigma

        self.weight_net = nn.Sequential(
            nn.LazyLinear(attention_hidden_nc,bias=bias),nn.SiLU(),
            nn.LazyLinear(attention_hidden_nc,bias=bias),nn.SiLU(),
            nn.LazyLinear(attention_hidden_nc,bias=bias)
        )
        self.energy_net = build_mlp(self.n_hidden,bias=bias,n_out=n_out)

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                training: bool = False,
                inference: bool = False,
                output_index = 0,
               ) -> Dict[str, torch.Tensor]:
        # check if self.feature_key exists, otherwise set default 
        if not hasattr(self, "feature_key") or self.feature_key is None: 
            self.feature_key = "node_feats"

        # reshape the feature vectors
        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key]
            features = features.reshape(features.shape[0], -1)
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)
        
        #Molecular representation
        X = self.weight_net(features)
        X = scatter_sum(src=X,index=data["batch"],dim=0)
        # if "molecular_rep" in self.model_outputs:
        #     data["molecular_rep"] = X
        #     return data
        
        # predict energies for each orbital
        # if "pred_energy" in self.model_outputs:
        y = self.energy_net(X).squeeze()
        if inference:
            y = (y*self.sigma) + self.avge0
        data[self.output_key] = y
        return data
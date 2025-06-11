import cace
import lightning as L
import torch
import tad_dftd3 as d3
import tad_mctc as mctc
from dispnet.util import MLP
from dispnet.tensornet import TensorProductLayer, TensorLinearMixing, TensorFeedForward
from dispnet.ceonet import CEONet

class DispNet2(L.LightningModule):
    def __init__(self,cutoff=8.5,nc=64,layers=0,lomax=2,compute_transdip=False):
        super().__init__()
        self.representation = CEONet(nc,layers=layers,lomax=lomax,cutoff=cutoff) #CACE + TensorAct
        self.register_buffer('COV_D3', d3.data.COV_D3)
        self.register_buffer('VDW_D3', d3.data.VDW_D3)
        self.register_buffer('R4R2', d3.data.R4R2)
        self.register_buffer('zeta', torch.linspace(0,10,200))
        self.register_buffer('e_bias',torch.tensor(3))

        self.register_buffer('a1',torch.tensor(0.49484001))
        self.register_buffer('s8',torch.tensor(0.78981345))
        self.register_buffer('a2',torch.tensor(5.73083694))

        self.compute_transdip = compute_transdip
        self.enet = MLP(hidden_sizes=[nc,nc], output_size=nc)
        if not compute_transdip:
            self.dyadnet = MLP(hidden_sizes=[nc,nc],output_size=nc)
        else:
            self.dyadnet = TensorFeedForward(nc,lomax)
    
    def calc_alpha(self,e,mu): #vectorized
        """
        zeta (G,)
        e (N,C)
        mu (N,C) or (N,C,3)
        """
        if mu.shape[-1] == 3:
            dyadic = mu[:,:,:,None] * mu[:,:,None,:] #(N,C,3,1)(N,C,1,3)->(N,C,3,3)
            dyadic_tr = dyadic.diagonal(dim1=-2,dim2=-1).sum(axis=-1) / 3 #(N,C)
        else:
            dyadic_tr = mu
        num = 2*e*dyadic_tr #(N,C)
        denom = e[:,:,None]**2 + self.zeta[None,None,:]**2 #(N,C,G)
        alpha = (num[:,:,None] / denom).sum(axis=1) #(N,G)
        return alpha
                
    def compute_disp(self,numbers,positions,c6):
        rcov = self.COV_D3[numbers] #for 3-body
        rvdw = self.VDW_D3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
        r4r2 = self.R4R2[numbers] #for 3-body
        param = {"a1":self.a1,"s8":self.s8,"a2":self.a2}

        #NOTE: EVERYTHING IN ATOMIC UNITS!!!
        energy = d3.disp.dispersion(
            numbers,
            positions,
            param,
            c6,
            rvdw,
            r4r2,
            d3.disp.rational_damping,
        )
        e_tot = torch.sum(energy, dim=-1)
        hartree_to_ev = 27.2114
        e_tot = e_tot * hartree_to_ev #convert to eV
        return e_tot

    def forward(self,data,training=False):
        data["positions"].requires_grad = True
        data = self.representation(data)

        #Predict alpha with energies/mu
        data["e"] = self.enet(data["node_feats"]) + self.e_bias
        if self.compute_transdip:
            mu = self.dyadnet(data["node_feats_A"])[1] # shape: (N, C, 3)
            dyadic = mu[:,:,:,None] * mu[:,:,None,:] #(N,C,3,1)(N,C,1,3)->(N,C,3,3)
            dyadic_tr = dyadic.diagonal(dim1=-2,dim2=-1).sum(axis=-1) / 3 #(N,C)
        else:
            dyadic_tr = self.dyadnet(data["node_feats"])
        data["dyadic_tr"] = dyadic_tr
        data["alpha"] = self.calc_alpha(data["e"],data["dyadic_tr"]) #(N,G)
        
        #Predict C6 from alphas
        alpha_prod = data["alpha"][:, None, :] * data["alpha"][None, :, :]  # shape: (N, N, G)
        c6all = 3 / torch.pi * torch.trapz(alpha_prod, self.zeta, dim=-1)

        results = []
        c6_results = []
        for i in torch.unique(data["batch"]):
            idx = torch.where(data["batch"] == i)[0]
            numbers = data["atomic_numbers"][idx]
            positions = data["positions"][idx]
            c6 = c6all[idx][:,idx]
            e_tot = self.compute_disp(numbers,positions,c6)
            results.append(e_tot)
            c6_results.append(c6.ravel())
            
        results = torch.stack(results)
        data["pred_energy"] = results
        data["pred_c6"] = torch.hstack(c6_results)
        
        grad_outputs = [torch.ones_like(data["pred_energy"])]
        gradients = torch.autograd.grad(
            outputs=[data["pred_energy"]],  # [n_graphs, ]
            inputs=[data["positions"]],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=training,  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=False,  # For complete dissociation turn to true
        )[0]
        data["pred_force"] = -gradients
        if data["pred_force"].isnan().any():
            print("NaN Force Predicted!")
        
        return data
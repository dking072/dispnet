import cace
import lightning as L
import torch
import tad_dftd3 as d3
import tad_mctc as mctc
from dispnet.ceonet import CEONet

class DispNet(L.LightningModule):
    def __init__(self,cutoff=8.5):
        super().__init__()
        self.representation = CEONet(layers=0,cutoff=cutoff) #CACE + TensorAct
        self.register_buffer('COV_D3', d3.data.COV_D3)
        self.register_buffer('VDW_D3', d3.data.VDW_D3)
        self.register_buffer('R4R2', d3.data.R4R2)

        self.register_buffer('a1',torch.tensor(0.49484001))
        self.register_buffer('s8',torch.tensor(0.78981345))
        self.register_buffer('a2',torch.tensor(5.73083694))
        
        from les.module.atomwise import Atomwise
        from dispnet.util import MLP
        self.c6q = MLP(output_size=16)
        
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

        #Predict c6 geometrically for all
        c6q = self.c6q(data["node_feats"])
        data["c6q"] = c6q
        c6all = torch.einsum('ai,ib->ab', c6q, c6q.T)

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
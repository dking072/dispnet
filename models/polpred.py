import cace
import lightning as L
import torch
import tad_dftd3 as d3
import tad_mctc as mctc
import e3nn
from e3nn import o3
import torch.nn.functional as F
from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
from mace.modules.utils import get_outputs
from les.module.ewald import Ewald
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from dispnet.mace.blocks import NonLinearDipolePolarReadoutBlock
from e3nn.io import CartesianTensor

from dispnet.util.pol import LRElec
from dispnet.util.calc_pol import calc_pol

def take_grad(vec,x):
    outs = []
    for y in vec:
        grad_outputs = [torch.ones_like(y)]
        gradients = torch.autograd.grad(
            outputs=[y],  # [n_graphs, ]
            inputs=[x],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=True,  # Make sure the graph is not destroyed during training
            create_graph=True,  # Create graph for second derivative
            allow_unused=False,  # For complete dissociation turn to true
        )[0]
        outs.append(gradients)
    outs = torch.stack(outs)
    return outs

class PolPred(L.LightningModule):
    def __init__(self,representation,qnet=None,sigma=1.0,freeze=True,twol=False,a_bias=12.56):
        super().__init__()
        self.cutoff = representation.r_max.item()
        self.zs = representation.atomic_numbers
        self.representation = representation
        self.twol = twol
        self.register_buffer('a_bias', torch.tensor([a_bias]).float())
        self.register_buffer('sigma', torch.tensor([sigma]).float())
        if freeze:
            self.representation.requires_grad_(False)

        #Charges
        irreps_in = o3.Irreps("192x0e + 192x1o + 192x0e")
        mlp_irreps = o3.Irreps(f"192x0e")
        irreps_out = o3.Irreps(f"1x0e")
        gate = e3nn.nn.Activation(mlp_irreps,[F.silu])
        self.enet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)
        self.qnet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)

        #Summed outer products for polarizabilities
        if self.twol:
            irreps_in = o3.Irreps('192x0e + 192x1o + 192x0e')
            irreps_out = o3.Irreps('192x0e + 192x1o + 192x2e')
            tp_irreps_out, instructions = tp_out_irreps_with_instructions(irreps_in,irreps_in,irreps_out)
            self.tp = o3.TensorProduct(irreps_in,irreps_in,tp_irreps_out,instructions)
            
            mlp_irreps = o3.Irreps('192x0e + 192x1o + 192x2e')
            dnet_irreps_out = o3.Irreps("1x0e + 1x1o + 1x2e")
            gate = e3nn.nn.Activation(o3.Irreps("192x0e"),[F.silu])
            self.dnet = NonLinearDipolePolarReadoutBlock(tp_irreps_out,mlp_irreps,gate,irreps_out=dnet_irreps_out)
            self.ct = CartesianTensor("ij=ji")
        else:
            irreps_in = o3.Irreps('192x0e + 192x1o + 192x0e')
            mlp_irreps = o3.Irreps('192x0e + 192x1o')
            gate = e3nn.nn.Activation(o3.Irreps("192x0e"),[F.silu])
            dnet_irreps_out = o3.Irreps("1x0e + 1x1o")
            self.dnet = NonLinearDipolePolarReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out=dnet_irreps_out)

    def calc_energy(self,atomic_e,q,positions,batch,a=None,monA=None):
        ees_lst = []
        atomic_lst = []
        efield_lst = []
        if a is not None:
            epol_lst = []
            
        unique_batches = torch.unique(batch)
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            atomic_e_now = atomic_e[mask]
            r_now, q_now = positions[mask], q[mask]
            monA_now = monA[mask] if (monA is not None) else None
            
            obj = LRElec(r_now,monA_now,sigma=self.sigma)
            e_es = obj.calc_qq(q_now)
            ees_lst.append(e_es)
            atomic_lst.append(atomic_e_now.sum())
            efield = obj.calc_efield(q_now)
            efield_lst.append(efield)
            if a is not None:
                a_now = a[mask]
                e_pol = obj.calc_qa(q_now,a_now)
                epol_lst.append(e_pol)
    
        ees_lst = torch.hstack(ees_lst)
        atomic_lst = torch.hstack(atomic_lst)
        efield_lst = torch.vstack(efield_lst)
        out = {"e_es":ees_lst,"e_atomic":atomic_lst,"efield":efield_lst}
        if a is not None:
            out["e_pol"] = torch.stack(epol_lst)
    
        return out

    def forward(self,data,training=False):
        efield = torch.tensor([0,0,0],device=data["positions"].device).float()
        efield.requires_grad = True
        data["cell"] = data["cell"].reshape(-1,3,3)
        data["alpha"] = data["alpha"].reshape(-1,3,3)

        #Calc original qs
        rep = self.representation.forward(data,compute_force=False)
        q = rep["latent_charges"]

        if self.twol:
            tp_out = self.tp(rep["node_feats"],rep["node_feats"])
            dnet_out = self.dnet(tp_out) #1 + 3 + 5
            qdiv2 = dnet_out[:,[0,4,5,6,7,8]]
            qdiv = dnet_out[:,[1,2,3]]
            qdiv2 = self.ct.to_cartesian(qdiv2)
            q_add = qdiv @ efield + efield @ qdiv2 @ efield
        else:
            dnet_out = self.dnet(rep["node_feats"]) #1 + 3
            qdiv = dnet_out[:,[1,2,3]]
            qdiv2 = 0
            q_add = qdiv @ efield
        q = q + q_add
        data["qdiv"] = qdiv
        data["qdiv2"] = qdiv2

        #Subtract mean r
        unique_batches = torch.unique(data["batch"])
        positions = []
        for i in unique_batches:
            mask = data["batch"] == i  # Create a mask for the i-th configuration
            r_now = data["positions"][mask]
            r_now = r_now - r_now.mean(axis=0)
            positions.append(r_now)
        positions = torch.vstack(positions)

        #Calc mu and e_lr
        mu = calc_pol(rep["latent_charges"],positions,data["cell"],data["batch"])
        e_lrs = self.representation.les.ewald(q,positions,data["cell"],data["batch"])
        e_lrs = e_lrs + mu @ efield

        #Predict polarization as ediv
        data["bare_mu"] = mu
        data["ediv"] = take_grad(e_lrs,efield)
        outs = []
        for ediv in data["ediv"]:
            outs.append(take_grad(ediv,efield))
        data["ediv2"] = torch.stack(outs)
        
        return data

# from les.util import grad
# return grad(y=e_lrs[:,None],x=efield)

# from les.util import grad
# edivs = []
# ediv2s_tot = []
# for e_lr in e_lrs:
#     edivs.append(grad(y=e_lr,x=efield))
#     ediv2s = []
#     for ediv in edivs:
#         ediv2s.append(grad(y=ediv,x=efield))
#     ediv2s = torch.vstack(ediv2s)
#     ediv2s_tot.append(ediv2s)
# edivs = torch.vstack(edivs)
# ediv2s_tot = torch.stack(ediv2s_tot)
    
# data["ediv"] = edivs
# data["ediv2"] = ediv2s_tot

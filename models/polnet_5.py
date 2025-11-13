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
import numpy as np

from cace.modules.cutoff import PolynomialCutoff
polycut = PolynomialCutoff(4.5)
def inv_cutoff(r):
    return 1 - polycut(r)

# from dispnet.util.pol import calc_elec
# from dispnet.util.pol import LRElec

class LRElec:
    def __init__(self,r_raw,monA=None,sigma=1,cutoff_fn=None):
        epsilon = 1e-6
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n, n, 3]
        torch.diagonal(r_ij).add_(epsilon)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        self.r_ij = r_ij
        self.r_ij_norm = r_ij_norm
        self.sigma = sigma
        if cutoff_fn is not None:
            self.damping = cutoff_fn(r_ij_norm)
        else:
            self.damping = None
        
        r_p_ij = 1/r_ij_norm
        if monA is not None:
            monA_idx = torch.where(monA)[0]
            monB_idx = torch.where(~monA)[0]
            r_p_ij[monA_idx[:,None],monA_idx] = 0
            r_p_ij[monB_idx[:,None],monB_idx] = 0
        else:
            ind = np.diag_indices(r_p_ij.shape[0])
            r_p_ij[ind[0],ind[1]] = torch.zeros(r_p_ij.shape[0],device=r_p_ij.device)
        self.r_p_ij = r_p_ij

        self.twopi =  2.0 * torch.pi
        self.c = 1/(self.sigma * (2.0 ** 0.5))
        self.erf_term = torch.special.erf(self.c*self.r_ij_norm)

    def calc_qq(self,q):
        q_pot = 1/self.twopi * q[:,None] * self.r_p_ij * self.erf_term * 1/2
        q_pot = q_pot.sum(axis=0)
        e_es = (q*q_pot).sum() * 90.0474 #Normalization
        return e_es

    def calc_qa(self,q,a,damping_denom=None):
        if damping_denom is not None:
            #E^2 -- should be positive
            dphi_dr = 1/self.twopi * 1/(self.r_ij_norm**4 + damping_denom**4)
        else:
            dphi_dr = 1/self.twopi * self.r_p_ij**4

        if self.damping is not None:
            dphi_dr = dphi_dr * self.damping

        epol = -0.5 * (q[None,:]**2 * dphi_dr * a[:,None]).sum() * 90.0474
        return epol

        # rhat = self.r_ij * self.r_p_ij[:,:,None]
        # E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * rhat
        # E_ij = E_ij_raw.sum(axis=1) #[N,3]

        # if len(a.shape) > 1:
        #     assert(len(a.shape) == 3) #[N,3,3]
        #     assert(a.shape[-1] == 3)
        #     assert(a.shape[-2] == 3)
        #     E_ij_prime = torch.einsum("iab,ib->ia",a,E_ij)
        # else:
        #     E_ij_prime = a[:,None] * E_ij
        # assert(E_ij_prime.shape == E_ij.shape)
        # epol = -0.5 * (E_ij * E_ij_prime).sum() * 90.0474
        # return epol

    # def calc_efield(self,q):
    #     dphi_dr = - 1/self.twopi * self.r_p_ij**2
    #     rhat = self.r_ij * self.r_p_ij[:,:,None]
    #     E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * rhat
    #     E_ij = E_ij_raw.sum(axis=1) #[N,3]
    #     return E_ij

class PolNet(L.LightningModule):
    def __init__(self,representation,qnet=None,sigma=1.0,freeze=True,anisotropy=False,a_bias=2):
        super().__init__()
        self.cutoff = representation.r_max.item()
        self.zs = representation.atomic_numbers
        self.representation = representation
        self.anisotropy = anisotropy
        self.register_buffer('a_bias', torch.tensor([a_bias]).float())
        self.register_buffer('sigma', torch.tensor([sigma]).float())
        self.damping_factor = torch.nn.Parameter(torch.tensor([7]).float())
        # if freeze:
        #     self.representation.requires_grad_(False)

        #Charges
        irreps_in = o3.Irreps("192x0e + 192x1o + 192x0e")
        mlp_irreps = o3.Irreps(f"192x0e")
        irreps_out = o3.Irreps(f"1x0e")
        gate = e3nn.nn.Activation(mlp_irreps,[F.silu])
        self.enet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)
        self.qnet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)

        #Summed outer products for polarizabilities
        if self.anisotropy:
            irreps_in = o3.Irreps('192x0e + 192x1o + 192x0e')
            irreps_out = o3.Irreps('192x0e + 192x2e')
            tp_irreps_out, instructions = tp_out_irreps_with_instructions(irreps_in,irreps_in,irreps_out)
            self.tp = o3.TensorProduct(irreps_in,irreps_in,tp_irreps_out,instructions)
            
            mlp_irreps = o3.Irreps('192x0e + 192x2e')
            dnet_irreps_out = o3.Irreps("1x0e + 1x2e")
            gate = e3nn.nn.Activation(o3.Irreps("192x0e"),[F.silu])
            self.dnet = NonLinearDipolePolarReadoutBlock(tp_irreps_out,mlp_irreps,gate,irreps_out=dnet_irreps_out)
            self.ct = CartesianTensor("ij=ji")
        else:
            mlp_irreps = o3.Irreps(f"192x0e")
            irreps_out = o3.Irreps(f"1x0e")
            self.anet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)

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
            
            obj = LRElec(r_now,monA_now,sigma=self.sigma,
                         cutoff_fn=inv_cutoff)
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

    def calc_ind(self,q,positions,batch,a,monA=None):
        # ees_lst = []
        # atomic_lst = []
        # efield_lst = []
        epol_lst = []
        # if a is not None:
        #     epol_lst = []
            
        unique_batches = torch.unique(batch)
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            # atomic_e_now = atomic_e[mask]
            r_now, q_now = positions[mask], q[mask]
            monA_now = monA[mask] if (monA is not None) else None
            
            # obj = LRElec(r_now,monA_now,sigma=self.sigma,cutoff_fn=inv_cutoff)
            obj = LRElec(r_now,monA_now,sigma=self.sigma)
            # e_es = obj.calc_qq(q_now)
            # ees_lst.append(e_es)
            # atomic_lst.append(atomic_e_now.sum())
            # efield = obj.calc_efield(q_now)
            # efield_lst.append(efield)
            a_now = a[mask]
            e_pol = obj.calc_qa(q_now,a_now,damping_denom=self.damping_factor)
            epol_lst.append(e_pol)

        return torch.hstack(epol_lst)
    
    def get_a(self,node_feats,sqrt3 = 1.7320508):
        if self.anisotropy:
            #bias trace
            tp_out = self.tp(node_feats,node_feats)
            dnet_out = self.dnet(tp_out)
            dnet_out[:,0] = F.relu(dnet_out[:,0] + self.a_bias)*sqrt3
            a = self.ct.to_cartesian(dnet_out)
        else:
            a = F.relu(self.anet(node_feats) + self.a_bias).squeeze()
        return a

    def forward(self,data,training=False,calc_qa=True,
                use_mace_q=False,use_mace_atomic=False):
        data["positions"].requires_grad = True
        
        rep = self.representation.forward(data,compute_force=False)
        # base_e = self.representation.atomic_energies_fn(data["node_attrs"])
        # q = rep["latent_charges"] if use_mace_q else self.qnet(rep["node_feats"]).squeeze()
        # if use_mace_atomic:
        #     atomic_e = rep["node_energy"]
        # else:
        #     atomic_e = self.enet(rep["node_feats"]).squeeze() + base_e.squeeze()
        # if calc_qa:
        #     a = self.get_a(rep["node_feats"])
        # else:
        #     a = None
        a = self.get_a(rep["node_feats"])
        q = rep["latent_charges"]
        data["pred_ind"] = self.calc_ind(q,data["positions"],data["batch"],a,monA=None)
        data["pred_energy"] = rep["energy"] + data["pred_ind"]
        data["pred_a"] = a
        data["a_ref"] = torch.zeros_like(a)

        # e_dct = self.calc_energy(atomic_e,q,data["positions"],data["batch"],a=a)
        # data["pred_es"] = e_dct["e_es"]
        # data["pred_atomic"] = e_dct["e_atomic"]
        # data["pred_q"] = q
        # data["pred_efield"] = e_dct["efield"]
        # if calc_qa:
        #     data["pred_ind"] = e_dct["e_pol"]
        #     data["pred_a"] = a
        #     data["pred_energy"] = data["pred_atomic"] + data["pred_es"] + data["pred_ind"]
        # else:
        #     data["pred_energy"] = data["pred_atomic"] + data["pred_es"]

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
        
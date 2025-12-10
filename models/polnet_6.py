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
from cace.modules.cutoff import SwitchFunction

def min_distance(cell,positions):
    inv = torch.linalg.inv(cell)
    rij = positions[:,None] - positions[None,:]
    s = torch.einsum("ab,ijb->ija",inv,rij)
    s = s - torch.round(s)
    s = torch.einsum("ab,ijb->ija",cell,s)
    return s #[N,N,3]

class LRElec:
    def __init__(self,r_raw,cell,monA=None,sigma=1):
        self.cell = cell
        self.twopi =  2.0 * torch.pi
        if cell is None:
            self.periodic = False
        else:
            self.periodic = torch.linalg.det(cell).any()

        if not self.periodic:
            epsilon = 1e-6
            r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n, n, 3]
            torch.diagonal(r_ij).add_(epsilon)
            r_ij_norm = torch.norm(r_ij, dim=-1)
            self.r_ij = r_ij
            self.r_ij_norm = r_ij_norm
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
        else:
            self.r_ij = min_distance(cell,r_raw)
            
        
    # self.c = 1/(self.sigma * (2.0 ** 0.5))
        # self.erf_term = torch.special.erf(self.c*self.r_ij_norm)

    # def calc_qq(self,q):
    #     q_pot = 1/self.twopi * q[:,None] * self.r_p_ij * self.erf_term * 1/2
    #     q_pot = q_pot.sum(axis=0)
    #     e_es = (q*q_pot).sum() * 90.0474 #Normalization
    #     return e_es

    def calc_qa(self,q,a):
        if not self.periodic:
            return self.calc_qa_real(q,a)
        else:
            return self.calc_qa_periodic(q,a)

    def calc_qa_real(self,q,a):
        if len(a.shape) == 1:
            emag = (q[None,:] * self.r_p_ij**2).pow(2)
            epol = -0.5 * 1/self.twopi * (a[:,None] * emag).sum() * 90.0474
        else:
            assert(len(a.shape) == 3) #[N,3,3]
            assert(a.shape[-1] == 3)
            assert(a.shape[-2] == 3)
            rhat = self.r_ij * self.r_p_ij[:,:,None] #[N,N,3]
            eij = (q[None,:] * self.r_p_ij**2)[:,:,None] * rhat
            eij_prime = torch.einsum("iab,ijb->ija",a,eij)
            epol = -0.5 * (eij * eij_prime).sum() * 90.0474
        return epol

    def calc_qa_periodic(self,q,a,cutoff=20):
        cutoff_fn = SwitchFunction(cutoff-3,cutoff)
        s = self.r_ij
        epol = 0
    
        #Compute min distance interaction
        rij = torch.linalg.norm(s,dim=-1)
        aq2 = a[None,:] * q[:,None]**2
        mask = ~torch.eye(rij.shape[0],dtype=torch.bool,device=a.device)
        epol = epol + (aq2[mask] * 1/(rij[mask]**4) * cutoff_fn(rij[mask])).sum()
    
        trans_vecs = []
        nvec = 5
        for i in range(0,nvec):
            for j in range(0,nvec):
                for k in range(0,nvec):
                    if i == j == k == 0:
                        continue
                    d = self.cell[:,0]*i + self.cell[:,1]*j + self.cell[:,2]*k
                    trans_vecs.append(d)
        trans_vecs = torch.vstack(trans_vecs)
        trans_ds = torch.linalg.norm(trans_vecs,dim=-1)
    
        lim = 0.5*torch.linalg.norm(self.cell.sum(axis=1)) + cutoff
        mask = trans_ds < lim
        assert(not mask.all()) #increase nvec if hit
        # print(len(trans_vecs[mask]))
        for d in trans_vecs[mask]:
            s_prime = s + d[None,None,:]
            rij = torch.linalg.norm(s_prime,dim=-1)
            epolp = (aq2 * 1/(rij**4) * cutoff_fn(rij)).sum()
            epol = epol + epolp
        epol = -0.5 * 1/self.twopi * epol * 90.0474
        return epol

class PolNet(L.LightningModule):
    def __init__(self,representation,qnet=None,sigma=1.0,freeze=True,anisotropy=False,a_bias=2):
        super().__init__()
        self.cutoff = representation.r_max.item()
        self.zs = representation.atomic_numbers
        self.representation = representation
        self.anisotropy = anisotropy
        self.register_buffer('a_bias', torch.tensor([a_bias]).float())
        self.register_buffer('sigma', torch.tensor([sigma]).float())
        # self.damping_factor = torch.nn.Parameter(torch.tensor([7]).float())
        self.damping_factor = None
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

    def calc_ind(self,q,positions,batch,a,cell,monA=None):
        epol_lst = []

        unique_batches = torch.unique(batch)
        cell = cell.reshape(len(unique_batches),3,3)
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            r_now, q_now = positions[mask], q[mask]
            cell_now = cell[i]
            monA_now = monA[mask] if (monA is not None) else None
            
            obj = LRElec(r_now,cell_now,monA_now,sigma=self.sigma)
            a_now = a[mask]
            e_pol = obj.calc_qa(q_now,a_now)
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
        a = self.get_a(rep["node_feats"])
        q = rep["latent_charges"]
        data["pred_ind"] = self.calc_ind(q,data["positions"],data["batch"],a,data["cell"],monA=None)
        data["pred_energy"] = rep["energy"] + data["pred_ind"]
        data["pred_a"] = a
        data["pred_q"] = q
        data["a_ref"] = torch.zeros_like(a)

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
        
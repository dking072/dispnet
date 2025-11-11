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

def make_batch(numA):
    arr = []
    for i,n in enumerate(numA):
        arr.append(torch.ones(n)*i)
    return torch.hstack(arr).to(numA.device).long()

def make_ptr(numA):
    arr1 = torch.zeros(1).to(numA.device)
    arr2 = torch.cumsum(numA,dim=0)
    return torch.hstack([arr1,arr2]).long()

def make_edge_ptr(numA_edges):
    arr1 = torch.zeros(1).to(numA_edges.device)
    arr2 = torch.cumsum(numA_edges,dim=0)
    return torch.hstack([arr1,arr2]).long()

def prep_batch(batch,og_ptr,typ="A"):
    ks = ["positions","node_attrs","edge_index","shifts","unit_shifts"]
    for k in ks:
        batch[k] = batch[f"{k}_mon{typ}"]
    batch["ptr"] = make_ptr(batch[f"num{typ}"])
    batch["batch"] = make_batch(batch[f"num{typ}"])
    batch["edge_ptr"] = make_edge_ptr(batch[f"num{typ}_edges"])
    #Correct the edge collation
    for i, p in enumerate(og_ptr[:-1]):
        edge_start, edge_stop = batch["edge_ptr"][i], batch["edge_ptr"][i+1]
        batch["edge_index"][:,edge_start:edge_stop] -= p
        batch["edge_index"][:,edge_start:edge_stop] += batch["ptr"][i]
    return batch

class LRElec:
    def __init__(self,r_raw,monA=None,sigma=1):
        epsilon = 1e-6
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n, n, 3]
        torch.diagonal(r_ij).add_(epsilon)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        self.r_ij = r_ij
        self.r_ij_norm = r_ij_norm
        self.sigma = sigma
        
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

    def calc_qa(self,q,a):
        dphi_dr = - 1/self.twopi * self.r_p_ij**2

        rhat = self.r_ij * self.r_p_ij[:,:,None]
        E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * rhat
        E_ij = E_ij_raw.sum(axis=1) #[N,3]

        if len(a.shape) > 1:
            assert(len(a.shape) == 3) #[N,3,3]
            assert(a.shape[-1] == 3)
            assert(a.shape[-2] == 3)
            E_ij_prime = torch.einsum("iab,ib->ia",a,E_ij)
        else:
            E_ij_prime = a[:,None] * E_ij
        assert(E_ij_prime.shape == E_ij.shape)
        epol = -0.5 * (E_ij * E_ij_prime).sum() * 90.0474
        return epol

    def calc_efield(self,q):
        dphi_dr = - 1/self.twopi * self.r_p_ij**2
        rhat = self.r_ij * self.r_p_ij[:,:,None]
        E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * rhat
        E_ij = E_ij_raw.sum(axis=1) #[N,3]
        return E_ij
        
class SaptNet(L.LightningModule):
    def __init__(self,representation,qnet=None,ewald_sigma=1.0,freeze=True,anisotropy=False,a_bias=12.56):
        super().__init__()
        cutoff = representation.r_max.item()
        zs = representation.atomic_numbers
        self.representation = representation
        self.anisotropy = anisotropy
        self.register_buffer('a_bias', torch.tensor([a_bias]).float())
        if freeze:
            self.representation.requires_grad_(False)

        #Charges
        irreps_in = o3.Irreps("192x0e + 192x1o + 192x0e")
        mlp_irreps = o3.Irreps(f"192x0e")
        irreps_out = o3.Irreps(f"1x0e")
        gate = e3nn.nn.Activation(mlp_irreps,[F.silu])
        if not qnet:
            self.qnet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)
        else:
            self.qnet = qnet
            # self.qnet.requires_grad_(False)

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

    def calc_lr(self,q,a,positions,batch,monA=None):
        ees_lst, epol_lst = [], []
        unique_batches = torch.unique(batch)
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            r_now, q_now, a_now = positions[mask], q[mask], a[mask]
            monA_now = monA[mask] if (monA is not None) else None
    
            #Calc pol data
            obj = LRElec(r_now,monA_now)
            e_es = obj.calc_qq(q_now)
            e_pol = obj.calc_qa(q_now,a_now)
            ees_lst.append(e_es)
            epol_lst.append(e_pol)
    
        ees_lst = torch.hstack(ees_lst)
        epol_lst = torch.stack(epol_lst)
    
        return ees_lst, epol_lst
    
    def get_qa(self,node_feats,sqrt3 = 1.7320508):
        q = self.qnet(node_feats).squeeze()
        if self.anisotropy:
            #bias trace
            tp_out = self.tp(node_feats,node_feats)
            dnet_out = self.dnet(tp_out)
            dnet_out[:,0] = F.relu(dnet_out[:,0] + self.a_bias)*sqrt3
            a = self.ct.to_cartesian(dnet_out)
        else:
            a = F.relu(self.anet(node_feats) + self.a_bias).squeeze()
        return q,a

    def forward(self,data,training=False):
        og_ptr = data["ptr"]
        dimer_pos = data["positions"]
        dimer_batch = data["batch"]
        
        data_A = prep_batch(data,og_ptr,typ="A")
        rep_A = self.representation.forward(data,compute_force=False)
        node_feats_A = rep_A["node_feats"]
        q_A, a_A = self.get_qa(node_feats_A)
        q_A = rep_A["latent_charges"]
        
        data_B = prep_batch(data,og_ptr,typ="B")
        rep_B = self.representation.forward(data,compute_force=False)
        node_feats_B = rep_B["node_feats"]
        q_B, a_B = self.get_qa(node_feats_B)
        q_B = rep_B["latent_charges"]
        
        idx_A, idx_B = 0,0
        qtot, atot = [], []
        monAtot = []
        for i_A, i_B in zip(data["numA"],data["numB"]):
            qA = q_A[idx_A:idx_A+i_A]
            qB = q_B[idx_B:idx_B+i_B]
            aA = a_A[idx_A:idx_A+i_A]
            aB = a_B[idx_B:idx_B+i_B]
            qtot.append(torch.hstack([qA,qB]))
            if len(aA.shape) == 3:
                atot.append(torch.vstack([aA,aB]))
            else:
                atot.append(torch.hstack([aA,aB]))
            monA = torch.hstack([torch.ones_like(qA),torch.zeros_like(qB)])
            monAtot.append(monA)
            idx_A += i_A
            idx_B += i_B
        qtot = torch.hstack(qtot)
        if len(aA.shape) == 3:
            atot = torch.vstack(atot)
        else:
            atot = torch.hstack(atot)
        monAtot = torch.hstack(monAtot).bool()

        e_es, e_ind = self.calc_lr(qtot,atot,dimer_pos,dimer_batch,monA=monAtot)
        data["pred_es"] = e_es
        data["pred_ind"] = e_ind
        data["pred_q"] = qtot
        data["pred_a"] = atot
        return data
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

# from dispnet.util.pol import calc_elec
from dispnet.util.pol import LRElec

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

class SaptNet(L.LightningModule):
    def __init__(self,representation,qnet=None,sigma=1.0,freeze=True,anisotropy=False,a_bias=12.56,calc_qa=False):
        super().__init__()
        self.cutoff = representation.r_max.item()
        self.zs = representation.atomic_numbers
        self.representation = representation
        self.anisotropy = anisotropy
        self.register_buffer('a_bias', torch.tensor([a_bias]).float())
        self.register_buffer('sigma', torch.tensor([sigma]).float())
        self.calc_qa = calc_qa
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
            #Don't freeze
            self.qnet.requires_grad_(True)

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

    def calc_lr(self,q,positions,batch,a=None,monA=None):
        ees_lst = []
        if a is not None:
            epol_lst = []
            
        unique_batches = torch.unique(batch)
        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration
            r_now, q_now = positions[mask], q[mask]
            monA_now = monA[mask] if (monA is not None) else None
            
            obj = LRElec(r_now,monA_now,sigma=self.sigma)
            e_es = obj.calc_qq(q_now)
            ees_lst.append(e_es)
            if a is not None:
                a_now = a[mask]
                e_pol = obj.calc_qa(q_now,a_now)
                epol_lst.append(e_pol)
    
        ees_lst = torch.hstack(ees_lst)
        out = {"e_es":ees_lst}
        if a is not None:
            out["e_pol"] = torch.stack(epol_lst)
    
        return out
    
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

    def forward(self,data,training=False,use_mace_charges=False):
        rep = self.representation.forward(data,compute_force=False)
        
        if not use_mace_charges:
            q = self.qnet(rep["node_feats"]).squeeze()
        else:
            q = rep["latent_charges"]
            
        if self.calc_qa:
            a = self.get_a(rep["node_feats"])
        else:
            a = None
        
        monAtot = []
        for i_A, i_B in zip(data["numA"].int(),data["numB"].int()):
            idxA = torch.ones(i_A,device=q.device)
            idxB = torch.zeros(i_B,device=q.device)
            monA = torch.hstack([idxA,idxB])
            monAtot.append(monA)
        monAtot = torch.hstack(monAtot).bool()

        # e_es, e_ind = self.calc_lr(q,a,dimer_pos,dimer_batch,monA=monAtot)
        e_lrs = self.calc_lr(q,data["positions"],data["batch"],a=a,monA=monAtot)
        data["pred_es"] = e_lrs["e_es"]
        data["pred_q"] = q
        if self.calc_qa:
            data["pred_ind"] = e_lrs["e_pol"]
            data["pred_a"] = a
        return data
        
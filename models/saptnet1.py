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
    def __init__(self,representation,ewald_sigma=1.0,freeze=True):
        super().__init__()
        cutoff = representation.r_max.item()
        zs = representation.atomic_numbers
        self.representation = representation
        self.ewald = Ewald(sigma=ewald_sigma)
        if freeze:
            self.representation.requires_grad_(False)

        irreps_in = o3.Irreps("192x0e + 192x1o + 192x0e")
        mlp_irreps = o3.Irreps(f"192x0e")
        irreps_out = o3.Irreps(f"1x0e")
        gate = e3nn.nn.Activation(mlp_irreps,[F.silu])
        self.qnet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)

    def calc_lr(self,data,out):
        q = self.qnet(out["node_feats"])
        e_lr = self.ewald.forward(q,data["positions"],None,data["batch"])
        return e_lr

    def pred_q(self,data):
        out = self.representation.forward(data,compute_force=False)
        data["les_q"] = out["latent_charges"]
        data["pred_q"] = self.qnet(out["node_feats"])
        return data
    
    def forward(self,data,training=False):
        og_ptr = data["ptr"]
        
        #Calc for dimer:
        lr_energies = {}
        dimer_out = self.representation.forward(data,compute_force=False)
        lr_energies["dimer"] = self.calc_lr(data,dimer_out)
        
        #Calc for monA:
        data = prep_batch(data,og_ptr,typ="A")
        monA_out = self.representation.forward(data,compute_force=False)
        lr_energies["monA"] = self.calc_lr(data,monA_out)
        
        #Calc for monB:
        data = prep_batch(data,og_ptr,typ="B")
        monB_out = self.representation.forward(data,compute_force=False)
        lr_energies["monB"] = self.calc_lr(data,monB_out)

        #Calc ediff:
        data["pred_es"] = lr_energies["dimer"] - (lr_energies["monA"] + lr_energies["monB"])
        for k in ["energy","les_energy"]:
            data[k] = dimer_out[k] - (monA_out[k] + monB_out[k])

        return data
        
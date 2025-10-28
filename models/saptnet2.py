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
from dispnet.util.pol import calc_efield, build_polarization_matrix

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
    def __init__(self,representation,qnet=None,ewald_sigma=1.0,freeze=True):
        super().__init__()
        cutoff = representation.r_max.item()
        zs = representation.atomic_numbers
        self.representation = representation
        self.ewald = Ewald(sigma=ewald_sigma)
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
            self.qnet.requires_grad_(False)

        #Summed outer products for polarizabilities
        irreps_out = o3.Irreps(f"192x0e")
        self.enet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)
        self.linmix = torch.nn.Linear(192,192, bias=False)

    def calc_lr(self,data,out,label="default"):
        #Calc both ES and pol
        q = self.qnet(out["node_feats"])
        e_es = self.ewald.forward(q,data["positions"],None,data["batch"])

        #Linearly mix
        evals = F.relu(self.enet(out["node_feats"]))
        n = out["node_feats"].shape[0]
        evecs = out["node_feats"][:,192:-192].reshape(n,192,3)
        evecs = self.linmix(evecs.transpose(-1,-2)).transpose(-1,-2)

        #Take outer products for polarizabilities
        alpha = evecs[:,:,:,None] * evecs[:,:,None,:]
        alpha = evals[:,:,None,None] * alpha
        I3 = torch.eye(3, device=q.device, dtype=q.dtype)
        atomic_pol = alpha.sum(axis=1) + I3[None,:,:] * 1e-4
        # data[f"atomic_pol_{label}"] = atomic_pol
        
        epols = [] #Try training on both? Huh...
        unique_batches = torch.unique(data["batch"])
        for i in unique_batches:
            mask = data["batch"] == i  # Create a mask for the i-th configuration
            r_now, q_now = data["positions"][mask], q[mask]
            alpha_now = atomic_pol[mask]

            E = calc_efield(r_now,q_now)
            A = build_polarization_matrix(r_now,alpha_now)
            mu = torch.linalg.solve(A, E.ravel())
            epol = -0.5 * torch.dot(E.ravel(),mu)
            epols.append(epol)
        e_ind = torch.stack(epols)

        return e_es, e_ind
            
    def forward(self,data,training=False):
        og_ptr = data["ptr"]
        
        #Calc for dimer:
        lr_energies = {"es":[],"ind":[]}
        dimer_out = self.representation.forward(data,compute_force=False)
        e_es, e_ind = self.calc_lr(data,dimer_out,label="dimer")
        lr_energies["es"].append(e_es)
        lr_energies["ind"].append(e_ind)
        
        #Calc for monA:
        data = prep_batch(data,og_ptr,typ="A")
        monA_out = self.representation.forward(data,compute_force=False)
        e_es, e_ind = self.calc_lr(data,monA_out,label="monA")
        lr_energies["es"].append(e_es)
        lr_energies["ind"].append(e_ind)
        
        #Calc for monB:
        data = prep_batch(data,og_ptr,typ="B")
        monB_out = self.representation.forward(data,compute_force=False)
        e_es, e_ind = self.calc_lr(data,monB_out,label="monB")
        lr_energies["es"].append(e_es)
        lr_energies["ind"].append(e_ind)

        #Calc ediff:
        for k in ["es","ind"]:
            data[f"pred_{k}"] = lr_energies[k][0] - (lr_energies[k][1] + lr_energies[k][2])
        for k in ["energy","les_energy"]:
            data[k] = dimer_out[k] - (monA_out[k] + monB_out[k])

        return data
        
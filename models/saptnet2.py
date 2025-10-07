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
    def __init__(self,representation,ewald_sigma=1.0,freeze=True):
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
        self.qnet = NonLinearReadoutBlock(irreps_in,mlp_irreps,gate,irreps_out)

        #Make 2e elements
        irreps_in = o3.Irreps('192x0e + 192x1o + 192x0e')
        irreps_out = o3.Irreps('192x0e + 192x2e')
        irreps_out, instructions = tp_out_irreps_with_instructions(irreps_in,irreps_in,irreps_out)
        self.tp = o3.TensorProduct(irreps_in,irreps_in,irreps_out,instructions)

        #Nonlinear readout for polarizabilities
        mlp_irreps = o3.Irreps('192x0e + 192x2e')
        gate = e3nn.nn.Activation(o3.Irreps("192x0e"),[F.silu])
        self.dnet = NonLinearDipolePolarReadoutBlock(irreps_out,mlp_irreps,gate)
        self.ct = CartesianTensor("ij=ji")

    def calc_lr(self,data,out):
        q = self.qnet(out["node_feats"])
        e_lr = self.ewald.forward(q,data["positions"],None,data["batch"])
        return e_lr

    def calc_pol(self,data,out):
        #Uses latent charges from MACE
        tp = self.tp(out["node_feats"],out["node_feats"])
        dnet = self.dnet(tp) # 1x0e + 1x2e
        
        #Calculate charges and polarizabilities
        # atomic_pol = self.ct.to_cartesian(dnet)
        s_matrix = self.ct.to_cartesian(dnet) #gosh ai is getting smart lol
        data["atomic_pol"] = torch.matrix_exp(s_matrix)

        epols = [] #Try training on both? Huh...
        unique_batches = torch.unique(data["batch"])
        for i in unique_batches:
            mask = data["batch"] == i  # Create a mask for the i-th configuration
            r_now, q_now = data["positions"][mask], out["latent_charges"][mask]
            alpha_now = data["atomic_pol"][mask]
            if (data["cell"] is not None) and (len(data["cell"].shape) == 3):
                box_now = data["cell"][i]  # Get the box for the i-th configuration
                if torch.linalg.det(box_now) < 1e-6:
                    box_now = None
            else:
                box_now = None

            # check if the box is periodic or not
            if box_now is None:
                E = calc_efield(r_now,q_now)
                A = build_polarization_matrix(r_now,alpha_now)
                mu = torch.linalg.solve(A, E.ravel())
                epol = -0.5 * torch.dot(E.ravel(),mu)
                epols.append(epol)
            else:
                pass
        return torch.stack(epols)
    
    def forward(self,data,training=False):
        og_ptr = data["ptr"]
        
        #Calc for dimer:
        lr_energies = {}
        dimer_out = self.representation.forward(data,compute_force=False)
        # lr_energies["dimer"] = self.calc_lr(data,dimer_out)
        lr_energies["dimer"] = self.calc_pol(data,dimer_out)
        
        #Calc for monA:
        data = prep_batch(data,og_ptr,typ="A")
        monA_out = self.representation.forward(data,compute_force=False)
        lr_energies["monA"] = self.calc_pol(data,monA_out)
        
        #Calc for monB:
        data = prep_batch(data,og_ptr,typ="B")
        monB_out = self.representation.forward(data,compute_force=False)
        lr_energies["monB"] = self.calc_pol(data,monB_out)

        #Calc ediff:
        data["pred_ind"] = lr_energies["dimer"] - (lr_energies["monA"] + lr_energies["monB"])
        for k in ["energy","les_energy"]:
            data[k] = dimer_out[k] - (monA_out[k] + monB_out[k])

        return data
        
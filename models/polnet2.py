import cace
import lightning as L
import torch
import tad_dftd3 as d3
import tad_mctc as mctc
import e3nn
from e3nn import o3
import torch.nn.functional as F
from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock, EquivariantProductBasisBlock
from mace.modules.blocks import NonLinearDipoleReadoutBlock
from dispnet.mace.blocks import NonLinearDipolePolarReadoutBlock
from cace.modules import build_mlp
from cace.tools import scatter_sum
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from e3nn.io import CartesianTensor
from mace.modules.utils import get_outputs

#Idea is to simplify model to try and get a(0) more correct
class PolNet(L.LightningModule):
    def __init__(self,representation,freeze=True):
        super().__init__()
        cutoff = representation.r_max.item()
        zs = representation.atomic_numbers
        self.representation = representation
        if freeze:
            self.representation.requires_grad_(False)
        self.e_field = torch.nn.Parameter(torch.tensor((0,0,0)).float())
        self.e_field.requires_grad_(True)

        #Make 2e elements
        irreps_in = o3.Irreps('192x0e + 192x1o + 192x0e')
        irreps_out = o3.Irreps('192x0e + 192x2e')
        irreps_out, instructions = tp_out_irreps_with_instructions(irreps_in,irreps_in,irreps_out)
        self.tp = o3.TensorProduct(irreps_in,irreps_in,irreps_out,instructions)

        #Nonlinear readout
        mlp_irreps = o3.Irreps('192x0e + 192x2e')
        gate = e3nn.nn.Activation(o3.Irreps("192x0e"),[F.silu])
        self.dnet = NonLinearDipolePolarReadoutBlock(irreps_out,mlp_irreps,gate)
        self.ct = CartesianTensor("ij=ji")

        # self.dnet = NonLinearDipoleReadoutBlock(irreps_in,irreps_out,gate)
        # self.dnet = NonLinearReadoutBlock(irreps_out,irreps_out,gate)
        
        # irreps_in = o3.Irreps("192x0e + 192x1o + 192x0e + 192x0e")
        # irreps_out = o3.Irreps("192x0e")
        # self.enet = EquivariantProductBasisBlock(irreps_in,irreps_out,correlation=2,use_sc=False,use_agnostic_product=False)
        
        #Readout
        # self.remove_mean = True
        # gate = e3nn.nn.Activation(irreps_out,[F.silu])
        # self.qnet = NonLinearReadoutBlock(irreps_out,irreps_out,gate)

    def forward(self,data,training=False,calc_force=False,calc_adiv=False):
        data["positions"].requires_grad = True
        data["atomic_numbers"] = (self.representation.atomic_numbers[None,:] * data["node_attrs"]).sum(axis=1).int()
        #["node_feats"] is 1 + 3 + 1
        rep = self.representation(data,compute_force=False)
        data["node_feats"] = rep["node_feats"]

        data["tp"] = self.tp(data["node_feats"],data["node_feats"])
        data["dnet"] = self.dnet(data["tp"]) # 1x0e + 1x2e

        #Cartesian tensor
        data["latent_charges"] = rep["latent_charges"]
        data["atomic_pol"] = self.ct.to_cartesian(data["dnet"])
        data["pred_polarizability"] = scatter_sum(src=data["atomic_pol"],index=data["batch"],dim=0)
        data["polarizability"] = data["alpha"].reshape(-1,3,3)
        return data
        
        # data["qadd"] = torch.norm(torch.einsum("nij,j->ni",data["atomic_pol"],self.e_field) + 1e-8,dim=-1)
        # data["q"] = (rep["latent_charges"] + data["qadd"])[:,None] #Add to q

        #Calculate polarization
        # all_P = []
        # all_phases = [] 
        # unique_batches = torch.unique(data["batch"])
        # for i in unique_batches:
        #     mask = data["batch"] == i  # Create a mask for the i-th configuration
        #     r_now, q_now = data["positions"][mask], data["q"][mask]
        #     if (data["cell"] is not None) and (len(data["cell"].shape) == 3):
        #         box_now = data["cell"][i]  # Get the box for the i-th configuration
        #         if torch.linalg.det(box_now) < 1e-6:
        #             box_now = None
        #     else:
        #         box_now = None

        #     # check if the box is periodic or not
        #     if box_now is None:
        #         # the box is not periodic, we use the direct sum
        #         polarization = torch.sum(q_now * r_now, dim=0)
        #         phase = torch.ones_like(r_now, dtype=torch.complex64)
        #     else:
        #         #reference here: https://github.com/ChengUCB/les/blob/main/src/les/module/bec.py
        #         r_frac = torch.matmul(r_now, torch.linalg.inv(box_now))
        #         phase = torch.exp(1j * 2.* torch.pi * r_frac)
        #         S = torch.sum(q_now * phase, dim=0)
        #         polarization = torch.matmul(box_now.to(S.dtype), 
        #                                     S.unsqueeze(1)) / (1j * 2.* torch.pi)
        #         polarization, phase = polarization.reshape(-1), phase

        #     normalization_factor = 1 #Future consideration
        #     all_P.append(polarization * normalization_factor)
        #     all_phases.append(phase)

        # P = torch.stack(all_P, dim=0)
        # phases = torch.cat(all_phases, dim=0)
        # data["polarization"] = P

        # alphas = []
        # for mu in data["polarization"]:
        #     alpha = []
        #     for v in mu:
        #         grad_outputs = [torch.ones_like(v)]
        #         g = torch.autograd.grad(
        #             outputs=v,
        #             inputs=self.e_field,
        #             grad_outputs=grad_outputs,
        #             retain_graph=True,
        #             create_graph=training,
        #         )[0]
        #         alpha.append(g)
        #     alpha = torch.vstack(alpha) #(3,3)
        #     alphas.append(alpha) 
        # alphas = torch.stack(alphas,dim=0) #(N,3,3)
        # data["pred_polarizability"] = alphas
        # data["polarizability"] = data["alpha"].reshape(-1,3,3)
        
        # return data
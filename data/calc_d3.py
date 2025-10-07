import pyscf
import os
import glob
import torch
from ase import io, Atoms
from dispnet.data.xyzdata import XYZData
from tqdm import tqdm
import torch
import tad_dftd3 as d3
import tad_mctc as mctc

def calc_d3(atm,rotate=False,calc_c6div=True):
    hartree_to_ev = 27.2114
    ang_to_bohr = 1/pyscf.lib.param.BOHR
    rotate = False
    numbers = atm.arrays["numbers"]
    positions = atm.arrays["positions"]
    data = {}
    if rotate:
        positions[:,[0,1,2]] = positions[:,[1,0,2]]
        positions[:,0] = -positions[:,0]
    data["numbers"] = torch.tensor(numbers)
    data["positions"] = torch.tensor(positions).float()
    data["positions"].requires_grad = True #So we get eV/A
    
    #Convert to Bohr for tad-dftd3
    positions = data["positions"] * ang_to_bohr 
    numbers = data["numbers"]
    
    #Ref values -- C6 for the different CN pairs
    ref = d3.reference.Reference()
    rcov = d3.data.COV_D3[numbers] #for 3-body
    rvdw = d3.data.VDW_D3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = d3.data.R4R2[numbers] #for 3-body
    param = {
    "a1": torch.tensor(0.49484001),
    "s8": torch.tensor(0.78981345),
    "a2": torch.tensor(5.73083694),
    }
    
    cn = mctc.ncoord.cn_d3(
        numbers, positions, counting_function=mctc.ncoord.exp_count, rcov=rcov
    )
    
    #### model will replace this part
    weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_weight)
    c6 = d3.model.atomic_c6(numbers, weights, ref)
    ####
    
    energy = d3.disp.dispersion(
        numbers,
        positions,
        param,
        c6,
        rvdw,
        r4r2,
        d3.disp.rational_damping,
    )
    energy = energy * hartree_to_ev
    tot_energy = torch.sum(energy, dim=-1)
    
    grad_outputs = [torch.ones_like(tot_energy)]
    gradients = torch.autograd.grad(
        outputs=[tot_energy],  # [n_graphs, ]
        inputs=[data["positions"]],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=True,  # Make sure the graph is not destroyed during training
        create_graph=False,  # Create graph for second derivative
        allow_unused=False,  # For complete dissociation turn to true
    )[0]
    force = -gradients #Ha/bohr

    n = data["positions"].shape[0]
    if calc_c6div:
        c6divs = []
        for c in c6.ravel():
            grad_outputs = torch.ones_like(c)
            c6div = torch.autograd.grad(
                outputs=[c],  # [n_graphs, ]
                inputs=[data["positions"]],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=True,  # Make sure the graph is not destroyed during training
                create_graph=False,  # Create graph for second derivative
                allow_unused=False,  # For complete dissociation turn to true
            )[0]
            c6divs.append(c6div)
        c6divs = torch.stack(c6divs).reshape(n,n,n,3)

    numbers = data["numbers"].cpu().detach().numpy()
    positions = data["positions"].cpu().detach().numpy()
        
    atm2 = Atoms(numbers,positions)
    atm2.info["energy"] = tot_energy.item()
    atm2.arrays["force"] = force.cpu().detach().numpy()
    atm2.arrays["c6"] = c6.cpu().detach().numpy()
    if calc_c6div:
        atm2.arrays["c6div"] = c6divs.cpu().detach().numpy().reshape(n,-1)
    return atm2
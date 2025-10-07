import cace
import lightning as L
import torch
import tad_dftd3 as d3
import tad_mctc as mctc
import e3nn
from e3nn import o3
import torch.nn.functional as F
from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock

#Idea is to simplify model to try and get a(0) more correct
class DispNet(L.LightningModule):
    def __init__(self,representation,nc=1,freeze=True,anisotropy=False):
        super().__init__()
        cutoff = representation.r_max.item()
        zs = representation.atomic_numbers
        self.representation = representation
        self.anisotropy = anisotropy
        if freeze:
            self.representation.requires_grad_(False)
        self.register_buffer('COV_D3', d3.data.COV_D3)
        self.register_buffer('VDW_D3', d3.data.VDW_D3)
        self.register_buffer('R4R2', d3.data.R4R2)
        self.register_buffer('zeta', torch.linspace(0,10,200))
        self.register_buffer('e_bias',torch.tensor(0.5))
        # self.e_bias = torch.nn.Parameter(torch.tensor(0.5))
        self.register_buffer('a_bias',torch.tensor(5.5))
        self.register_buffer('ang_to_bohr',torch.tensor(1.88973))
        self.register_buffer('min_e',torch.tensor(0.001))

        #For fit:
        self.register_buffer('a1',torch.tensor(0.49484001))
        self.register_buffer('s8',torch.tensor(0.78981345))
        self.register_buffer('a2',torch.tensor(5.73083694))

        irreps_in = o3.Irreps("192x0e + 192x1o + 192x0e")
        irreps_out = o3.Irreps(f"{nc}x0e")
        irreps_out_v = o3.Irreps(f"{nc}x1o")
        gate = e3nn.nn.Activation(irreps_out,[F.silu])
        self.enet = NonLinearReadoutBlock(irreps_in,irreps_out,gate,irreps_out)
        self.dyad_scalar = NonLinearReadoutBlock(irreps_in,irreps_out,gate,irreps_out)
        self.dyad_vector = LinearReadoutBlock(irreps_in,irreps_out_v)

    def compute_disp(self,numbers,positions,c6):
        rcov = self.COV_D3[numbers] #for 3-body
        rvdw = self.VDW_D3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
        r4r2 = self.R4R2[numbers] #for 3-body
        param = {"a1":self.a1,"s8":self.s8,"a2":self.a2}

        #NOTE: EVERYTHING IN ATOMIC UNITS!!!
        energy = d3.disp.dispersion(
            numbers,
            positions,
            param,
            c6,
            rvdw,
            r4r2,
            d3.disp.rational_damping,
        )
        e_tot = torch.sum(energy, dim=-1)
        hartree_to_ev = 27.2114
        e_tot = e_tot * hartree_to_ev #convert to eV
        return e_tot

    def forward(self,data,training=False,calc_force=False,calc_adiv=False):
        data["positions"].requires_grad = True
        data["atomic_numbers"] = (self.representation.atomic_numbers[None,:] * data["node_attrs"]).sum(axis=1).int()
        #["node_feats"] is 1 + 3 + 1
        data["node_feats"] = self.representation(data,compute_force=False)["node_feats"]

        #Predict alpha with energies/mu, in a.u.
        #Constrain e and mu to be positive with relu
        # data["e"] = F.relu(self.enet(data["node_feats"]) + self.e_bias) + self.min_e #(N,C)
        if self.anisotropy:
            data["mu"] = self.dyad_vector(data["node_feats"]).reshape(data["atomic_numbers"].shape[0],-1,3) #(N,C,3)

        #Scalar component of dyad (the trace)
        data["dyad_tr"] = F.relu(self.dyad_scalar(data["node_feats"]) + self.a_bias) + self.min_e #(N,C)
        
        #Vector component of dyad
        diag_mask = torch.eye(3, device=data["node_feats"].device) #(3,3)
        if self.anisotropy:
            sym = data["mu"][:,:,:,None] * data["mu"][:,:,None,:] #(N,C,3,3)
            tr = sym.diagonal(dim1=-2,dim2=-1).sum(axis=-1)/3 #(N,C)
            sym_notrace = sym - tr[:,:,None,None]*diag_mask[None,None,:,:] #(N,C,3,3)
        else:
            sym_notrace = 0

        #Calculate dyad from scalar and vector components
        dyad = (data["dyad_tr"][:,:,None,None] * diag_mask[None,None,:,:])/3 + sym_notrace #(N,C,3,3)

        #Calculate alpha(zeta)
        num = dyad * self.e_bias
        denom = self.e_bias + self.zeta[None,None,None,None,:]**2 #(N,C,3,3,G)
        alpha = num[...,None] / denom #(N,C,3,3,G)
        data["alpha"] = alpha.sum(axis=1).moveaxis(-1,1) #(N,3,3,G)
        data["alpha_avg"] = data["alpha"].diagonal(dim1=-2,dim2=-1).sum(axis=-1)/3 #(N,G)
        # print(data["alpha_avg"][:,0],data["alpha_avg"][:,0].shape )

        #Calculate safe pairwise distances
        r_ij = data["positions"][None,:,:] - data["positions"][:,None,:] # (N,N,3)
        torch.diagonal(r_ij).add_(0.1) # for safe derivatives
        r_ij_2 = torch.ones_like(r_ij) * 0.1 #Fixed distance between batches
        for i in range(data["ptr"][:-1].shape[0]):
            start, stop = data["ptr"][i], data["ptr"][i+1]
            r_ij_2[start:stop,start:stop,:] = r_ij[start:stop,start:stop,:]
        r_ij = r_ij_2
        norm = torch.norm(r_ij,dim=-1) # (N,N)
        rhat_ij = r_ij / norm[:,:,None] #normalize
        
        #Calculate dipole/dipole interaction tensor
        t = 3 * rhat_ij[:,:,None,:] * rhat_ij[:,:,:,None] - diag_mask[None,None,:,:] # (N,N,3,3)
        torch.diagonal(t).zero_() #Zero self-interaction
        
        #Interaction and integrate over frequencies for C6
        if t.isnan().any():
            print("NaN found!")
        atat = torch.einsum("agij,abjk,bgkl,abli->abg",data["alpha"],t,data["alpha"],t) #(N,N,G)
        c6all = (1/(2*torch.pi)) * torch.trapz(atat, self.zeta, dim=-1)
        
        results = []
        c6_results = []
        #Run through d3, scale the positions to Bohr
        for i in torch.unique(data["batch"]):
            idx = torch.where(data["batch"] == i)[0]
            numbers = data["atomic_numbers"][idx]
            positions = data["positions"][idx] * self.ang_to_bohr
            c6 = c6all[idx][:,idx]
            e_tot = self.compute_disp(numbers,positions,c6)
            results.append(e_tot)
            mask = ~torch.eye(c6.shape[0], dtype=torch.bool, device=c6.device)
            c6_results.append(c6[mask].ravel()) #remove diagonal
        d3_energy = torch.stack(results)
        data["pred_energy"] = d3_energy
        data["pred_c6"] = torch.hstack(c6_results)

        if calc_force:
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

        if calc_adiv: #isotropic j
            unique_batches = torch.unique(data["batch"])  # Get unique batch indices
            all_j = []
            assert(len(unique_batches) == 1) #Not tested for more
            for i in unique_batches:
                idx = torch.where(data["batch"] == i)[0]
                alpha_iso = data["alpha_avg"][idx,0].sum()
                grad_outputs = [torch.ones_like(alpha_iso)]
                gradients = torch.autograd.grad(
                    outputs=[alpha_iso],  # [n_graphs, ]
                    inputs=[data["positions"]],  # [n_nodes, 3]
                    grad_outputs=grad_outputs,
                    retain_graph=training,  # Make sure the graph is not destroyed during training
                    create_graph=training,  # Create graph for second derivative
                    allow_unused=False,  # For complete dissociation turn to true
                )[0]
                all_j.append(gradients[idx])
            all_j = torch.vstack(all_j)
            data["alpha_avg_div"] = all_j
        
        return data
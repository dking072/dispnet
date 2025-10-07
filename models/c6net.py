import cace
import pyscf
import lightning as L
import torch
import tad_dftd3 as d3
import tad_mctc as mctc
import e3nn
from e3nn import o3
import torch.nn.functional as F
from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock

class FakeRef:
    def __init__(self,c6,cn):
        self.c6 = c6
        self.cn = cn

#Idea is to simplify model to try and get a(0) more correct
class C6Net(L.LightningModule):
    def __init__(self,nc=1,fixed_e=0.5):
        super().__init__()
        self.register_buffer('zeta', torch.linspace(0,10,200))
        self.register_buffer('min_e',torch.tensor(0.001))

        self.register_buffer('COV_D3', d3.data.COV_D3)
        self.register_buffer('VDW_D3', d3.data.VDW_D3)
        self.register_buffer('R4R2', d3.data.R4R2)
        self.register_buffer('ang_to_bohr', torch.tensor([1/pyscf.lib.param.BOHR]))

        els = [1,6,7,8,9,15,16,17,35,53]
        self.els = els
        # self.register_buffer('cnref_og', d3.reference.Reference().cn) #(104,7)
        # self.register_buffer('c6ref_og', d3.reference.Reference().c6) #(104,104,7,7)
        self.register_buffer('cnref', d3.reference.Reference().cn[els]) #(Nel,7)
        self.register_buffer('c6ref', d3.reference.Reference().c6[els,:,:,:][:,els,:,:]) #(Nel,Nel,7,7)

        self.idx = torch.where(self.cnref.ravel() != -1)[0]
        c6ref_nonzero = self.c6ref.movedim(1,2).reshape(len(self.els)*7,len(self.els)*7)
        c6ref_nonzero = c6ref_nonzero[self.idx[:, None], self.idx]
        self.register_buffer('c6ref_nonzero',c6ref_nonzero)
        self.a_bias = torch.sqrt(self.c6ref_nonzero.mean())
        self.a0 = torch.nn.Parameter(torch.ones(len(self.idx),nc)*self.a_bias)
        if not fixed_e:
            self.e0 = torch.nn.Parameter(torch.ones(len(self.idx),nc)*0.5)
        else:
            self.register_buffer('e0',torch.tensor(fixed_e))
        self.fixed_e = fixed_e

    def calc_c6ref(self,data):
        #Calculate entire table
        a0 = F.relu(self.a0) + self.min_e
        if not self.fixed_e:
            e0 = F.relu(self.e0) + self.min_e
            num = a0 * e0 #(N,C)
            denom = e0[:,:,None] + self.zeta[None,None,:]**2 #(N,C,G)
            alpha = (num[:,:,None]/denom).sum(axis=1) #(N,G)
        else:
            num = a0 * self.e0 #(N,C)
            denom = self.e0 + self.zeta[None,None,:]**2 #(N,C,G)
            alpha = (num[:,:,None]/denom).sum(axis=1) #(N,G)

        #Calculate c6
        alpha_prod = alpha[:,None,:] * alpha[None,:,:] #(N,N,G)
        c6all = 3 / torch.pi * torch.trapz(alpha_prod, self.zeta, dim=-1) #(N,N)

        #Reformat
        # data["alpha_ref"] = torch.zeros(104*7,self.zeta.shape[0],device=data["positions"].device)
        # data["alpha_ref"][self.idx] = alpha
        # data["alpha_ref"] = data["alpha_ref"].reshape(104,7,-1)
        # data["alpha_ref"] = alpha #Ignore formatting for now
        el_idx = torch.hstack([torch.tensor(self.els)[:,None]]*7).ravel()[self.idx].to(data["positions"].device)
        cn_idx = torch.vstack([torch.arange(7)]*len(self.els)).ravel()[self.idx].to(data["positions"].device)
        data["alpha_ref"] = (torch.ones(104,7,self.zeta.shape[-1])*-1).to(data["positions"].device)
        data["alpha_ref"][el_idx,cn_idx,:] = alpha

        # data["pred_c6ref"] = torch.zeros_like(self.c6ref.movedim(1,2).reshape(self.cnref.ravel().shape[0],-1))
        # data["pred_c6ref"][self.idx[:, None], self.idx] = c6all
        # data["pred_c6ref"] = data["pred_c6ref"].reshape(len(self.els),7,len(self.els),7).movedim(1,2) #(104,104,7,7)
        data["pred_c6ref"] = c6all
        data["c6ref"] = self.c6ref_nonzero
        # data["c6ref"] = self.c6ref
        return data

    def forward(self,data,training=False,calc_adiv=False):
        data = self.calc_c6ref(data)

        #Calc alphas from interpolation
        data["positions"].requires_grad = True #So we get eV/A
        positions = data["positions"] * self.ang_to_bohr 
        numbers = data["numbers"].int()
        rcov = self.COV_D3[numbers]
        cn = mctc.ncoord.cn_d3(
            numbers, positions, counting_function=mctc.ncoord.exp_count, rcov=rcov
        )

        #Calc actual c6
        ref = d3.reference.Reference()
        ref.c6 = ref.c6.to(data["positions"].device)
        ref.cn = ref.cn.to(data["positions"].device)
        # print(ref.c6.device,ref.cn.device,numbers.device)
        # fake_ref = FakeRef(self.c6ref_og,self.cnref_og)
        weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_weight) #(N,7)
        c6_d3 = d3.model.atomic_c6(numbers, weights, ref)
        mask = ~torch.eye(c6_d3.shape[0], dtype=torch.bool, device=c6_d3.device) #Ignore diagonal components
        data["c6_d3"] = c6_d3[mask].ravel()
        data["weights"] = weights

        #Calc interpolated alpha / pred c6
        diag_mask = torch.eye(3, device=data["positions"].device) #(3,3)
        data["alpha"] = (data["alpha_ref"][numbers] * weights[:,:,None]).sum(dim=1) #(N,7,G)*(N,7) --> (N,G)
        data["alpha"] = data["alpha"][:,:,None,None] * diag_mask[None,None,:,:] #(N,G,3,3)
        
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
        c6all = (1/(2*torch.pi)) * torch.trapz(atat, self.zeta, dim=-1) #(N,N)
        data["pred_c6"] = c6all[mask].ravel()
        data["alpha_avg"] = data["alpha"].diagonal(dim1=-2, dim2=-1).sum(dim=-1)/3

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
import torch
import os
import glob
import torch
from dispnet.data.macedata_ediff import MaceXYZData
import numpy as np

def calc_efield(r_raw,q,sigma=1,calc_pot=False):
    epsilon = 1e-6
    r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n, n, 3]
    torch.diagonal(r_ij).add_(epsilon)
    r_ij_norm = torch.norm(r_ij, dim=-1)  # [n, n]

    # Screening function
    c = 1/(sigma * (2.0 ** 0.5))
    erf_term = torch.special.erf(c*r_ij_norm)
    r_p_ij = 1/r_ij_norm

    # Compute electrostatic potential
    # [n_node, 1] * [n_node, n_node,] * [n_node, n_node] --> [n_node,n_node]
    twopi =  2.0 * torch.pi
    if calc_pot:
        q_pot = 1/twopi * q[:,None] * r_p_ij * erf_term * 1/2 #idrk why 1/2, 1/4pie0?
        torch.diagonal(q_pot).zero_()
        q_pot = q_pot.sum(axis=0)
        pot = (q*q_pot).sum() * 90.0474 #Normalization
        
    # Derivative of φ(r) = erf(arg)/r, [N,N]
    exp_term = torch.exp(-(c*r_ij_norm)**2) # [n, n]
    dphi_dr = c * (2.0 / (torch.pi ** 0.5)) * exp_term / r_ij_norm - erf_term / (r_ij_norm ** 2)
    dphi_dr = 1/twopi * dphi_dr * 1/2
    
    # Electric field contributions: E_i = Σ_j q_j * (r_i - r_j)/r * (-dφ/dr)
    # [1,N,1] , [N,N,1] x [N,N,3] --> [N,N,3]
    E_ij = q[None,:,None] * dphi_dr[:,:,None] * (r_ij / r_ij_norm[:,:,None])
    
    # Sum over j to get total field at each i
    E = E_ij.sum(dim=1)  # [n, 3]

    if calc_pot:
        return q_pot, pot, E
    else:
        return E

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
        exp_term = torch.exp(-(self.c*self.r_ij_norm)**2) # [n, n]
        dphi_dr = self.c * (2.0 / (torch.pi ** 0.5)) * exp_term * self.r_p_ij - self.erf_term * self.r_p_ij**2
        dphi_dr = 1/self.twopi * dphi_dr * 1/2

        E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * self.r_ij * self.r_p_ij[:,:,None]
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
        exp_term = torch.exp(-(self.c*self.r_ij_norm)**2) # [n, n]
        dphi_dr = self.c * (2.0 / (torch.pi ** 0.5)) * exp_term * self.r_p_ij - self.erf_term * self.r_p_ij**2
        dphi_dr = 1/self.twopi * dphi_dr * 1/2

        E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * self.r_ij * self.r_p_ij[:,:,None]
        E_ij = E_ij_raw.sum(axis=1) #[N,3]
        return E_ij
        
def calc_elec(r_raw,q,alpha,monA=None,sigma=1,calc_pot=True):
    #Simplified polarization
    epsilon = 1e-6
    r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n, n, 3]
    torch.diagonal(r_ij).add_(epsilon)
    r_ij_norm = torch.norm(r_ij, dim=-1)  # [n, n]

    #Calc Efield
    twopi =  2.0 * torch.pi
    c = 1/(sigma * (2.0 ** 0.5))
    erf_term = torch.special.erf(c*r_ij_norm)

    r_p_ij = 1/r_ij_norm
    if monA is not None:
        monA_idx = torch.where(monA)[0]
        monB_idx = torch.where(~monA)[0]
        r_p_ij[monA_idx[:,None],monA_idx] = 0
        r_p_ij[monB_idx[:,None],monB_idx] = 0
    else:
        ind = np.diag_indices(r_p_ij.shape[0])
        r_p_ij[ind[0],ind[1]] = torch.zeros(r_p_ij.shape[0],device=r_p_ij.device)

    if calc_pot:
        q_pot = 1/twopi * q[:,None] * r_p_ij * erf_term * 1/2
        # torch.diagonal(q_pot).zero_()
        q_pot = q_pot.sum(axis=0)
        e_es = (q*q_pot).sum() * 90.0474 #Normalization
    else:
        e_es = torch.zeros(1,device=r_raw.device)
        
    # Derivative of φ(r) = erf(arg)/r, [N,N]
    exp_term = torch.exp(-(c*r_ij_norm)**2) # [n, n]
    dphi_dr = c * (2.0 / (torch.pi ** 0.5)) * exp_term * r_p_ij - erf_term * r_p_ij**2
    dphi_dr = 1/twopi * dphi_dr * 1/2
    
    # Field at i caused by j
    # [1,N,1] , [N,N,1] x [N,N,3] x [N,N,1] --> [N,N,3]
    E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * r_ij * r_p_ij[:,:,None]
    E_ij = E_ij_raw.sum(axis=1) #[N,3]
    
    #Calc E^T a E
    if len(alpha.shape) > 1:
        assert(len(alpha.shape) == 3) #[N,3,3]
        assert(alpha.shape[-1] == 3)
        assert(alpha.shape[-2] == 3)
        # E_ij_prime = torch.einsum("iab,ijb->ija",alpha,E_ij_raw)
        E_ij_prime = torch.einsum("iab,ib->ia",alpha,E_ij)
    else:
        # E_ij_prime = alpha[:,None] * E_ij_raw
        E_ij_prime = alpha[:,None] * E_ij
    assert(E_ij_prime.shape == E_ij.shape)
    epol = -0.5 * (E_ij * E_ij_prime).sum() * 90.0474
    return e_es, epol

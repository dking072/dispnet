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

def calc_elec(r_raw,q,alpha,sigma=1,calc_pot=True):
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

    if calc_pot:
        q_pot = 1/twopi * q[:,None] * r_p_ij * erf_term * 1/2 #idrk why 1/2, 1/4pie0?
        torch.diagonal(q_pot).zero_()
        q_pot = q_pot.sum(axis=0)
        e_es = (q*q_pot).sum() * 90.0474 #Normalization
        
    # Derivative of φ(r) = erf(arg)/r, [N,N]
    exp_term = torch.exp(-(c*r_ij_norm)**2) # [n, n]
    dphi_dr = c * (2.0 / (torch.pi ** 0.5)) * exp_term / r_ij_norm - erf_term / (r_ij_norm ** 2)
    dphi_dr = 1/twopi * dphi_dr * 1/2
    
    # Electric field contributions: E_i = Σ_j q_j * (r_i - r_j)/r * (-dφ/dr)
    # Field at i caused by j
    # [1,N,1] , [N,N,1] x [N,N,3] --> [N,N,3]
    E_ij_raw = q[None,:,None] * dphi_dr[:,:,None] * (r_ij / r_ij_norm[:,:,None])
    # g = 1 - torch.exp(-acd * u_ij**m)
    # E_ij = E_ij * g[:,:,None] #Thole damping
    
    #Calc E^T a E
    if len(alpha.shape) > 1:
        assert(len(alpha.squeeze().shape) == len(alpha.shape))
        assert(len(alpha.shape) == 3)
        E_ij_prime = torch.einsum("iab,ijb->ija",alpha,E_ij_raw)
    else:
        E_ij_prime = alpha[:,None,None] * E_ij_raw
    epol = -0.5 * (E_ij_raw * E_ij_prime).sum() * 90.0474
    return e_es, epol

def make_TE(r_raw,q,alpha,thole=0.055,acd=0.4,sigma=1,m=4,ind_scaling=1,calc_pot=True):
    epsilon = 1e-6
    r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n, n, 3]
    torch.diagonal(r_ij).add_(epsilon)
    r_ij_norm = torch.norm(r_ij, dim=-1)  # [n, n]

    #See following refs for thole damping:
    # http://dx.doi.org/10.1063/1.4807093 -- bad notation but more explained
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC2918242/ -- AMOEBA pff
    # https://tinkerdoc.readthedocs.io/en/latest/text/forcefield/polarize.html -- AMOEBA ref
    if len(alpha.shape) > 1:
        alpha_scalar = torch.linalg.det(alpha) ** (1/3)
    else:
        alpha_scalar = alpha
    pol_length = (alpha_scalar[:,None] * alpha_scalar[None,:])**(1/6) #[n,n]
    # thole_radius = thole * (alpha_scalar[:,None] * alpha_scalar[None,:])**(-1/2) #[n,n]

    #### DIPOLE-DIPOLE ####
    
    #Damped thole r #[n,n]
    u_ij = r_ij_norm / pol_length
    f = 1 - torch.exp(-thole * u_ij**m)
    dphi_dr = -1/(r_ij_norm**2) * f

    #2nd Derivative #[n,n]
    term1 = 2/(r_ij_norm**3) * f
    term2 = thole * m * u_ij**2 * torch.exp(-thole * u_ij**m) * 1/pol_length
    term2 = -1/(r_ij_norm**2) * term2
    dphi2_dr2 = term1 + term2

    I_coeff = -1/r_ij_norm * dphi_dr
    r_outer_coeff = -1/(r_ij_norm**2) * (dphi2_dr2 - 1/r_ij_norm * dphi_dr)

    #Make T
    I3 = torch.eye(3, device=r_raw.device, dtype=r_raw.dtype)
    r_outer = r_ij[:,:,None,:] * r_ij[:,:,:,None]
    T = I_coeff[:,:,None,None] * I3 - r_outer_coeff[:,:,None,None] * r_outer
    T.diagonal().zero_()

    #Scale by 1/4pi for efield consistency?
    twopi = 2.0 * torch.pi
    T = 1/2 * 1/twopi * T

    #Add diagonal alpha elements
    if len(alpha.shape) == 1: #[N]
        I3 = torch.eye(3, device=r_raw.device, dtype=r_raw.dtype)
        inv_alpha = 1/alpha[:,None,None] * I3[None,:,:]
    else:
        torch.linalg.inv(alpha) #[N,3,3]
    diag_idx = torch.arange(T.shape[0])
    T[diag_idx,diag_idx] = inv_alpha
    T = T.transpose(1,-1).transpose(-1,2)
    flat = r_raw.shape[0]*3
    T = T.reshape(flat,flat) #[Nx3,Nx3]

    #### E Field ####
    
    #Erf term
    c = 1/(sigma * (2.0 ** 0.5))
    erf_term = torch.special.erf(c*r_ij_norm)
    r_p_ij = 1/r_ij_norm

    if calc_pot:
        q_pot = 1/twopi * q[:,None] * r_p_ij * erf_term * 1/2 #idrk why 1/2, 1/4pie0?
        torch.diagonal(q_pot).zero_()
        q_pot = q_pot.sum(axis=0)
        e_es = (q*q_pot).sum() * 90.0474 #Normalization
        
    # Derivative of φ(r) = erf(arg)/r, [N,N]
    exp_term = torch.exp(-(c*r_ij_norm)**2) # [n, n]
    dphi_dr = c * (2.0 / (torch.pi ** 0.5)) * exp_term / r_ij_norm - erf_term / (r_ij_norm ** 2)
    dphi_dr = 1/twopi * dphi_dr * 1/2
    
    # Electric field contributions: E_i = Σ_j q_j * (r_i - r_j)/r * (-dφ/dr)
    # [1,N,1] , [N,N,1] x [N,N,3] --> [N,N,3]
    E_ij = q[None,:,None] * dphi_dr[:,:,None] * (r_ij / r_ij_norm[:,:,None])
    g = 1 - torch.exp(-acd * u_ij**m)
    E_ij = E_ij * g[:,:,None] #Thole damping
    
    # Sum over j to get total field at each i
    E = E_ij.sum(dim=1)  # [n, 3]
    E = E * 9.48933 * ind_scaling

    return T, E.ravel(), e_es

# def cg_pol(self,E,T,eps=1e-3):
#     #Initial guess and "preconditioner"
#     mu = E
#     M = T
    
#     #Calc pol
#     r = E - T @ mu
#     rho = M @ r
    
#     while r.abs().max() > eps:
#         rMr = r @ M @ r 
#         gamma = rMr / (rho @ T @ rho)
#         mu = mu + gamma * rho
#         r1 = r - gamma * T @ rho
#         beta = (r1 @ M @ r1) / rMr
#         rho = M @ r1 + beta * rho
#         r = r1

#     epol = -0.5 * mu @ E
#     return mu, epol


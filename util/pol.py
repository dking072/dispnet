#ML generated code
import torch

def calc_efield(r_raw,q,sigma=1):
    epsilon = 1e-6
    r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)  # [n, n, 3]
    torch.diagonal(r_ij).add_(epsilon)
    r_ij_norm = torch.norm(r_ij, dim=-1)  # [n, n]
    
    # Screening function
    arg = r_ij_norm / (sigma * (2.0 ** 0.5))
    erf_term = torch.special.erf(arg)
    exp_term = torch.exp(-arg**2) # [n, n]
    
    # Derivative of φ(r) = erf(arg)/r
    dphi_dr = (2.0 / (torch.pi ** 0.5)) * exp_term / (sigma * (2.0 ** 0.5) * r_ij_norm) \
              - erf_term / (r_ij_norm ** 2)
    
    # Electric field contributions: E_i = Σ_j q_j * (r_i - r_j)/r * (-dφ/dr)
    # Note: sign flipped because E = -∇φ
    E_ij = q.unsqueeze(0).unsqueeze(2) * (-dphi_dr.unsqueeze(2)) * (r_ij / r_ij_norm.unsqueeze(2))
    
    # Sum over j to get total field at each i
    E = E_ij.sum(dim=1)  # [n, 3]
    
    return E

def make_t(r,eps=1e-9):
    device = r.device

    N = r.shape[0]
    I3 = torch.eye(3, device=device, dtype=r.dtype)

    # Pairwise vectors and distances
    r_ij = r.unsqueeze(0) - r.unsqueeze(1)        # [N, N, 3]
    r2 = (r_ij**2).sum(dim=-1)                    # [N, N]
    r2 = r2 + torch.eye(N, device=device, dtype=r.dtype) * eps  # avoid exact zeros on diagonal
    norm = torch.sqrt(r2)                         # [N, N]

    # Precompute powers
    r3 = r2 * norm                                # r^3
    r5 = r3 * r2                                  # r^5

    # Outer product r_ij ⊗ r_ij -> [N, N, 3, 3]
    r_outer = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)

    # Undamped dipole-dipole tensor: T = 3 r⊗r / r^5 - I / r^3
    # Broadcast r5 and r3 to (N,N,1,1)
    denom_r5 = r5.unsqueeze(-1).unsqueeze(-1)
    denom_r3 = r3.unsqueeze(-1).unsqueeze(-1)

    T = 3.0 * r_outer / denom_r5 - I3.view(1, 1, 3, 3) / denom_r3  # [N, N, 3, 3]
    T.diagonal().zero_()
    
    return T, norm

def build_polarization_matrix(r, alpha, thole_a=0.4, eps=1e-9, device=None):
    #Make undamped interaction
    T, norm = make_t(r)

    #Thole dampoing
    alpha_scalar = torch.linalg.det(alpha) ** (1/3)
    alpha_pair_factor = (alpha_scalar.unsqueeze(0) * alpha_scalar.unsqueeze(1)) ** (1.0/6.0)
    alpha_pair_factor = alpha_pair_factor.clamp(min=eps)
    u = norm / (thole_a * alpha_pair_factor + eps) # [N,N]

    # Exponential Thole damping scalar f(u) = 1 - e^{-u}(1 + u + 0.5 u^2)
    # For numerical stability clamp u small/large appropriately
    exp_minus_u = torch.exp(-u)
    f = 1.0 - exp_minus_u * (1.0 + u + 0.5 * (u ** 2))   # [N, N]
    f = f.clamp(min=0.0, max=1.0)

    # Apply damping: T_damped = f * T
    f_block = f.unsqueeze(-1).unsqueeze(-1)              # [N,N,1,1]
    T_damped = T * f_block                               # [N,N,3,3]

    # Add inv polarizabilities on diagonal
    inv_alpha = torch.linalg.inv(alpha)
    diag_idx = torch.arange(T.shape[0])
    T[diag_idx,diag_idx] = inv_alpha

    T = T.transpose(1,-1).transpose(-1,2)
    flat = r.shape[0]*3
    T = T.reshape(flat,flat)

    return T
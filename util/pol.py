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

def build_polarization_matrix(r, alpha, thole_a=0.4, eps=1e-9, device=None):
    """
    Build the 3N x 3N polarization matrix A with:
      A_ii = inv(alpha_i)            (3x3 block)
      A_ij = - T_ij_damped           (3x3 block, i != j)

    Arguments
    ---------
    r : tensor (N,3)
        Cartesian positions.
    alpha : tensor
        Either shape (N,) for isotropic polarizabilities (scalars),
        or shape (N,3,3) for full anisotropic polarizability tensors.
    thole_a : float
        Thole damping parameter (typical ~0.3-0.5). Used in u = r/(a*(alpha_i*alpha_j)^{1/6}).
    eps : float
        Small regularizer to avoid division by zero.
    device : torch.device or None
        Device to use (defaults to r.device).

    Returns
    -------
    A : tensor (3N, 3N)
        Polarization matrix.
    """

    if device is None:
        device = r.device

    N = r.shape[0]
    I3 = torch.eye(3, device=device, dtype=r.dtype)

    # Pairwise vectors and distances
    r_ij = r.unsqueeze(0) - r.unsqueeze(1)        # [N, N, 3]
    r2 = (r_ij**2).sum(dim=-1)                    # [N, N]
    r2 = r2 + torch.eye(N, device=device, dtype=r.dtype) * eps  # avoid exact zeros on diagonal
    r = torch.sqrt(r2)                            # [N, N]

    # Precompute powers
    r3 = r2 * r                                   # r^3
    r5 = r3 * r2                                  # r^5

    # Outer product r_ij ⊗ r_ij -> [N, N, 3, 3]
    r_outer = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)

    # Undamped dipole-dipole tensor: T = 3 r⊗r / r^5 - I / r^3
    # Broadcast r5 and r3 to (N,N,1,1)
    denom_r5 = r5.unsqueeze(-1).unsqueeze(-1)
    denom_r3 = r3.unsqueeze(-1).unsqueeze(-1)

    T = 3.0 * r_outer / denom_r5 - I3.view(1, 1, 3, 3) / denom_r3  # [N, N, 3, 3]

    # Build Thole damping scalar f_ij for each pair (i,j)
    # Need alpha_i * alpha_j as scalar measure. Support isotropic and anisotropic cases.
    if alpha.dim() == 1:
        # isotropic scalars alpha_i
        alpha_vec = alpha.view(N)
        # pairwise geometric mean-like factor: (alpha_i * alpha_j)^(1/6)
        alpha_pair_factor = (alpha_vec.unsqueeze(0) * alpha_vec.unsqueeze(1)).clamp(min=eps) ** (1.0 / 6.0)
    elif alpha.dim() == 3 and alpha.shape[1:] == (3, 3):
        # anisotropic: define scalar measure as (det(alpha))^(1/3)  => cubic root of det gives volume-like measure
        # This is a common choice for building a scalar length-scale from anisotropic alpha.
        det_alpha = torch.linalg.det(alpha)                          # (N,)
        det_alpha = det_alpha.clamp(min=eps)
        alpha_scalar = det_alpha ** (1.0 / 3.0)                      # (N,)
        alpha_pair_factor = (alpha_scalar.unsqueeze(0) * alpha_scalar.unsqueeze(1)) ** (1.0/6.0)
    else:
        raise ValueError("alpha must be shape (N,) or (N,3,3)")

    # u = r / ( thole_a * alpha_pair_factor )
    alpha_pair_factor = alpha_pair_factor.to(device=device, dtype=r.dtype).clamp(min=eps)
    u = r / (thole_a * alpha_pair_factor + eps)

    # Exponential Thole damping scalar f(u) = 1 - e^{-u}(1 + u + 0.5 u^2)
    # For numerical stability clamp u small/large appropriately
    exp_minus_u = torch.exp(-u)
    f = 1.0 - exp_minus_u * (1.0 + u + 0.5 * (u ** 2))   # [N, N]
    f = f.clamp(min=0.0, max=1.0)

    # Apply damping: T_damped = f * T
    f_block = f.unsqueeze(-1).unsqueeze(-1)              # [N,N,1,1]
    T_damped = T * f_block

    # Assemble 3N x 3N matrix
    A = torch.zeros((3 * N, 3 * N), device=device, dtype=r.dtype)

    # Diagonal blocks: inverse polarizability
    if alpha.dim() == 1:
        # isotropic: inv_alpha * I_3 for each i
        inv_alpha = 1.0 / (alpha.view(N) + eps)
        for i in range(N):
            A[3*i:3*i+3, 3*i:3*i+3] = inv_alpha[i] * I3
    else:
        # anisotropic: invert each 3x3 alpha_i
        # ensure invertible; add small regularization if needed
        for i in range(N):
            Ai = alpha[i].to(device=device, dtype=r.dtype)
            # regularize if near-singular
            try:
                invAi = torch.linalg.inv(Ai)
            except RuntimeError:
                invAi = torch.linalg.inv(Ai + torch.eye(3, device=device, dtype=r.dtype) * eps)
            A[3*i:3*i+3, 3*i:3*i+3] = invAi

    # Off-diagonal blocks: -T_damped
    # Fill blocks (i != j)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            block = -T_damped[i, j]    # 3x3
            A[3*i:3*i+3, 3*j:3*j+3] = block

    return A
import torch
import torch.nn as nn
from typing import Dict, Optional

__all__ = ['BEC']

def compute_pol_pbc(r_now, q_now, box_now):
    r_frac = torch.matmul(r_now, torch.linalg.inv(box_now))
    phase = torch.exp(1j * 2.* torch.pi * r_frac)
    S = torch.sum(q_now * phase, dim=0)
    polarization = torch.matmul(box_now.to(S.dtype), 
                                S.unsqueeze(1)) / (1j * 2.* torch.pi)
    return polarization.reshape(-1), phase

def calc_pol(q,r,cell,batch,remove_mean=True,epsilon_factor=1):
    normalization_factor = epsilon_factor ** 0.5
    
    if q.dim() == 1:
        q = q.unsqueeze(1)

    # Check the input dimension
    n, d = r.shape
    assert d == 3, 'r dimension error'
    assert n == q.size(0), 'q dimension error'

    if batch is None:
        batch = torch.zeros(n, dtype=torch.int64, device=r.device)
    unique_batches = torch.unique(batch)  # Get unique batch indices

    # compute the polarization for each batch
    all_P = []
    all_phases = [] 
    for i in unique_batches:
        mask = batch == i  # Create a mask for the i-th configuration
        r_now, q_now = r[mask], q[mask]
        if remove_mean:
            q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)

        if cell is not None:
            box_now = cell[i]  # Get the box for the i-th configuration

        # check if the box is periodic or not
        if cell is None or torch.linalg.det(box_now) < 1e-6:
            # the box is not periodic, we use the direct sum
            polarization = torch.sum(q_now * r_now, dim=0)
            phase = torch.ones_like(r_now, dtype=torch.complex64)
        else:
            polarization, phase = compute_pol_pbc(r_now, q_now, box_now)
            print(polarization.shape)

        all_P.append(polarization * normalization_factor)
        all_phases.append(phase)
    P = torch.stack(all_P, dim=0)
    phases = torch.cat(all_phases, dim=0)
    result = P * phases.conj()
    return result.real

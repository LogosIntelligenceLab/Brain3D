"""Sampling weights for UMAP-projected high-level embeddings.

Paper equations (cosine similarity, then the two normalizations):

    s_j = (U_H · U_Hj) / (|U_H| |U_Hj|)
    v_j = s_j / |U_Hj|
    w_j = v_j / sum_k v_k

Negative similarities are clipped to zero so `w` is a probability vector.
If every weight is zero, the distribution falls back to uniform.
"""

import torch


def sampling_weights(u_h, bank_u, eps=1e-8):
    u_h = torch.as_tensor(u_h, dtype=torch.float32).flatten()
    bank = torch.as_tensor(bank_u, dtype=torch.float32)
    if bank.ndim != 2:
        raise ValueError(f"bank_u must be (N, D), got {tuple(bank.shape)}")
    bank_norm = bank.norm(dim=-1).clamp_min(eps)
    u_norm = u_h.norm().clamp_min(eps)
    cosine = (bank @ u_h) / (bank_norm * u_norm)
    raw = cosine / bank_norm
    raw = raw.clamp(min=0)
    total = raw.sum()
    if float(total) <= eps:
        return torch.full((bank.shape[0],), 1.0 / bank.shape[0])
    return raw / total

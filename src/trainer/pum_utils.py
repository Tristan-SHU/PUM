import hashlib
import json
import math
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple



# Utility Function
def construct_seed64(master: int, *keys) -> int:
    """
    Construct a 64-bit integer seed from (master, *keys), ensuring deterministic
    and consistent behavior across processes/devices.
    """
    payload = json.dumps([int(master)] + list(keys), separators=(",", ":"), ensure_ascii=False)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 64) - 1)


# Layer-wise Noise Scale Computation
@torch.no_grad()
def compute_layerwise_sigma(
    sd_ref: "OrderedDict[str, torch.Tensor]",
    layer_groups: "Dict[str, List[str]]",
    kappa: float,
    base_sd: "OrderedDict[str, torch.Tensor]" = None,
) -> "Dict[str, float]":
    """
    Compute the noise scale σ_l for each layer block. For each block l, define
    v_l = sd_ref - base_sd; if base_sd is None, then v_l = sd_ref.
    """
    sigma = {}
    for lk, names in layer_groups.items():
        sumsq = 0.0
        count = 0
        for n in names:
            t = sd_ref[n]
            if not torch.is_tensor(t):
                continue
            v = (t if base_sd is None else (t - base_sd[n])).detach().float()
            sumsq += (v * v).sum().item()
            count += v.numel()
        sigma[lk] = 0.0 if count == 0 else kappa * (sumsq / count) ** 0.5
    return sigma


# Per-layer zero-sum noise generation with α-scaling
@torch.no_grad()
def generate_zero_sum_noises(
    sd_ref: "OrderedDict[str, torch.Tensor]",
    layer_groups: "Dict[str, List[str]]",
    sigmas: "Dict[str, float]",
    m: int,
    alphas: "List[float]",
    seed_round: int,
    device: torch.device,
) -> "List[OrderedDict[str, torch.Tensor]]":
    """
    For each layer block l and replica k, sample z_{k,l} ~ N(0, σ_l^2 I). Center these samples to obtain zero-sum noises
    ε^0_{k,l}, then scale them as ε_{k,l} = α_k · ε^0_{k,l}. Returns m noise state_dicts with the same parameter shapes as sd_ref.
    """
    assert m >= 2  # m must be at least 2 to enable the zero-sum normalization procedure
    assert len(alphas) == m
    noises = [
        OrderedDict((k, torch.zeros_like(v, device=device, dtype=v.dtype)) for k, v in sd_ref.items())
        for _ in range(m)
    ]
    for lk, names in layer_groups.items():
        sigma = float(sigmas.get(lk, 0.0))
        if sigma == 0.0:
            continue

        # For each parameter name, draw m independent samples z, all allocated on the specified device.
        Z_per_name = {n: [] for n in names}
        for k in range(m):
            g = torch.Generator(device="cpu")
            g.manual_seed(construct_seed64(seed_round, "noise", lk, k))
            for n in names:
                t = sd_ref[n]
                z = torch.randn(t.shape, generator=g, device=device, dtype=t.dtype) * sigma
                Z_per_name[n].append(z)

        # Zero-sum normalization: compute ε^0_k = sqrt(m/(m-1)) * (z_k - mean) for each replica k, then scale by α_k
        scale = (m / (m - 1)) ** 0.5
        for n in names:
            stack = torch.stack(Z_per_name[n], dim=0)  # [m, ...]
            mean = stack.mean(dim=0)
            eps0 = (stack - mean) * scale
            for k in range(m):
                noises[k][n].add_(eps0[k] * float(alphas[k]))
    return noises


# Reparameterization: orthogonal / rotation blocks
def generate_rand_orth_matrix(d: int, gen: torch.Generator, device, dtype):
    M = torch.randn(d, d, generator=gen, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(M)
    s = torch.sign(torch.diag(R))
    return Q @ torch.diag_embed(s)


def construct_rotation_matrix(theta: torch.Tensor):  # [...]->[...,2,2]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2
    )  # [...,2,2]


# Sample T_{k,r} (attention components U and S_KV; without head permutation)
@torch.no_grad()
def sample_T_attention(
    H_Q: int,
    H_KV: int,
    d_h: int,
    rope_aware: bool,
    seed_round: int,
    k_copy: int,
    device="cpu",
    dtype=torch.float32,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """
    Return (U, S_KV), both orthogonal matrices:
      - Without RoPE: each KV head has an independent orthogonal block S_j ∈ O(d_h),
        and S_KV = block_diag(S_1, ..., S_{H_KV}). The i-th diagonal block of U
        is chosen as S_{π(i)} where π(i) = floor(i * H_KV / H_Q).
      - RoPE-aware: each RoPE plane is represented by a 2×2 rotation R(φ) that
        commutes with the RoPE phase operator Φ(p); the remaining assembly is
        analogous to the non-RoPE construction.
    """
    gS = torch.Generator(device="cpu")
    gS.manual_seed(construct_seed64(seed_round, "T", k_copy, "S"))

    blocks_kv = []
    if rope_aware:
        if d_h % 2 != 0:
            raise ValueError("RoPE-aware mode requires the per-head dimension d_h to be even.")
        for _ in range(H_KV):
            thetas = torch.rand(d_h // 2, generator=gS, device=device, dtype=dtype) * (2 * math.pi)
            R_planes = construct_rotation_matrix(thetas)            # [d_h//2, 2, 2]
            S_j = torch.block_diag(*R_planes.unbind(dim=0))         # [d_h, d_h]
            blocks_kv.append(S_j)
    else:
        for _ in range(H_KV):
            S_j = generate_rand_orth_matrix(d_h, gS, device=device, dtype=dtype)
            blocks_kv.append(S_j)

    S_KV = torch.block_diag(*blocks_kv).to(device=device, dtype=dtype)
    idx = [min((i * H_KV) // H_Q, H_KV - 1) for i in range(H_Q)]
    U_blocks = [blocks_kv[j] for j in idx]
    U = torch.block_diag(*U_blocks).to(device=device, dtype=dtype)
    return U, S_KV


# Forward transformation T: W_Q' = W_Q U, W_K' = W_K S, W_V' = W_V S, and W_O' = U^T W_O
@torch.no_grad()
def apply_T_attention_weights(
    W_Q: torch.Tensor,  # [H_Q*d_h, d_model]
    W_K: torch.Tensor,  # [H_KV*d_h, d_model]
    W_V: torch.Tensor,  # [H_KV*d_h, d_model]
    W_O: torch.Tensor,  # [d_model, H_Q*d_h]
    U: torch.Tensor,    # [H_Q*d_h, H_Q*d_h]   (orthogonal)
    S_KV: torch.Tensor  # [H_KV*d_h, H_KV*d_h] (orthogonal)
):
    """
    PyTorch Linear weight is [out, in] (column-vector/left-multiplication convention).
    This function applies the same transform as the paper's row-vector equations:
      W_Q' = W_Q U, W_K' = W_K S, W_V' = W_V S, W_O' = U^T W_O,
    but converted to PyTorch shapes.
    """
    # Left-multiply by U^T / S_KV^T; right-multiply by U
    W_Qp = U.T   @ W_Q
    W_Kp = S_KV.T @ W_K
    W_Vp = S_KV.T @ W_V
    W_Op = W_O @ U
    return W_Qp, W_Kp, W_Vp, W_Op


@torch.no_grad()
def inverse_T_on_update_attention(
    dW_Qp: torch.Tensor,  # [H_Q*d_h, d_model]
    dW_Kp: torch.Tensor,  # [H_KV*d_h, d_model]
    dW_Vp: torch.Tensor,  # [H_KV*d_h, d_model]
    dW_Op: torch.Tensor,  # [d_model, H_Q*d_h]
    U: torch.Tensor,      # [H_Q*d_h, H_Q*d_h]
    S_KV: torch.Tensor    # [H_KV*d_h, H_KV*d_h]
):
    """
    Apply T^{-1} to the parameter *updates* (Δ) that were measured in the published coordinates.
      Δ_Q = U   @ Δ_Q',  Δ_K = S @ Δ_K',  Δ_V = S @ Δ_V',  Δ_O = Δ_O' @ U^T
    """
    dW_Q = U    @ dW_Qp
    dW_K = S_KV @ dW_Kp
    dW_V = S_KV @ dW_Vp
    dW_O = dW_Op @ U.T
    return dW_Q, dW_K, dW_V, dW_O



# Forward transformation T for FFN
@torch.no_grad()
def apply_T_ffn_weights(
    W1_gate: torch.Tensor,   # [d_ff, d_model]
    W1_up:   torch.Tensor,   # [d_ff, d_model]
    W2:      torch.Tensor,   # [d_model, d_ff]
    P_ffn:   torch.Tensor,   # [d_ff, d_ff] (permutation / orthogonal)
    b1_gate: torch.Tensor = None,   # [d_ff] or None
    b1_up:   torch.Tensor = None,   # [d_ff] or None
    b2_down: torch.Tensor = None,   # [d_model] or None (unchanged)
):
    """
    Apply the FFN reparameterization T in PyTorch's [out, in] convention.
      gate/up: left-multiply P_ffn (row permutation); bias follows
      down   : right-multiply P_ffn^T (column permutation); bias unchanged
    """
    # shapes sanity (optional)
    d_ff = W1_gate.size(0)
    assert W1_gate.shape[0] == d_ff and W1_up.shape[0] == d_ff and W2.shape[1] == d_ff
    assert P_ffn.shape == (d_ff, d_ff)

    W1_gate_p = P_ffn @ W1_gate
    W1_up_p   = P_ffn @ W1_up
    W2_p      = W2 @ P_ffn.T

    b1_gate_p = (P_ffn @ b1_gate) if b1_gate is not None else None
    b1_up_p   = (P_ffn @ b1_up)   if b1_up   is not None else None
    b2_down_p = b2_down  # unchanged

    return W1_gate_p, W1_up_p, W2_p, b1_gate_p, b1_up_p, b2_down_p


# Inverse transform T^{-1} on FFN parameter updates (Δ)
@torch.no_grad()
def inverse_T_on_update_ffn(
    dW1_gate_p: torch.Tensor,   # [d_ff, d_model]
    dW1_up_p:   torch.Tensor,   # [d_ff, d_model]
    dW2_p:      torch.Tensor,   # [d_model, d_ff]
    P_ffn:      torch.Tensor,   # [d_ff, d_ff]
    db1_gate_p: torch.Tensor = None,   # [d_ff] or None
    db1_up_p:   torch.Tensor = None,   # [d_ff] or None
    db2_down_p: torch.Tensor = None,   # [d_model] or None (unchanged)
):
    """
    Apply T^{-1} to FFN updates measured in the published coordinates.
    """
    dW1_gate = P_ffn.T @ dW1_gate_p
    dW1_up   = P_ffn.T @ dW1_up_p
    dW2      = dW2_p   @ P_ffn

    db1_gate = (P_ffn.T @ db1_gate_p) if db1_gate_p is not None else None
    db1_up   = (P_ffn.T @ db1_up_p)   if db1_up_p   is not None else None
    db2_down = db2_down_p  # unchanged

    return dW1_gate, dW1_up, dW2, db1_gate, db1_up, db2_down

# Memory Saving Version
# @torch.no_grad()
# def apply_T_attention_weights(
#     q_w: torch.Tensor, k_w: torch.Tensor, v_w: torch.Tensor, o_w: torch.Tensor,
#     S_kv_blocks: List[torch.Tensor],   # len = H_KV, each [d_h, d_h]
#     H_Q: int, H_KV: int, d_h: int
# ):
#     # Q: Left multiplication U^T: multiply S_{π(i)}^T for each Q-head row block
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(q_w)
#         q_w[i*d_h:(i+1)*d_h, :] = S.T @ q_w[i*d_h:(i+1)*d_h, :]

#     # K/V: Left multiply by S_KV^T: Multiply by S_j^T for each row block of the KV-head
#     for j in range(H_KV):
#         S = S_kv_blocks[j].to(k_w)
#         k_w[j*d_h:(j+1)*d_h, :] = S.T @ k_w[j*d_h:(j+1)*d_h, :]
#         v_w[j*d_h:(j+1)*d_h, :] = S.T @ v_w[j*d_h:(j+1)*d_h, :]

#     # O: Right multiply by U: Multiply by U for each Q-head Column block multiplied by S_{π(i)}
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(o_w)
#         o_w[:, i*d_h:(i+1)*d_h] = o_w[:, i*d_h:(i+1)*d_h] @ S


# @torch.no_grad()
# def inverse_T_on_update_attention(
#     dq_p: torch.Tensor, dk_p: torch.Tensor, dv_p: torch.Tensor, do_p: torch.Tensor,
#     S_kv_blocks: List[torch.Tensor],
#     H_Q: int, H_KV: int, d_h: int
# ):
#     dq, dk, dv, do = dq_p.clone(), dk_p.clone(), dv_p.clone(), do_p.clone()

#     # Q: Right multiply by U: Row block multiplication S_{π(i)}
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(dq)
#         dq[i*d_h:(i+1)*d_h, :] = S @ dq[i*d_h:(i+1)*d_h, :]

#     # K/V: Right multiply by S_KV: Row block multiplication S_j
#     for j in range(H_KV):
#         S = S_kv_blocks[j].to(dk)
#         dk[j*d_h:(j+1)*d_h, :] = S @ dk[j*d_h:(j+1)*d_h, :]
#         dv[j*d_h:(j+1)*d_h, :] = S @ dv[j*d_h:(j+1)*d_h, :]

#     # O: Left multiplication U^T: block multiplication S_{π(i)}^T
#     for i in range(H_Q):
#         j = min(i * H_KV // H_Q, H_KV - 1)
#         S = S_kv_blocks[j].to(do)
#         do[:, i*d_h:(i+1)*d_h] = do[:, i*d_h:(i+1)*d_h] @ S.T

#     return dq, dk, dv, do


# Harmonic aggregation: aggregate parameter deltas using weights inversely proportional to the provided scaling factors (alphas).
@torch.no_grad()
def harmonic_aggregate(delta_list: List[OrderedDict], alphas: List[float]) -> OrderedDict:
    assert len(delta_list) == len(alphas)
    weights = [1.0 / float(a) for a in alphas]
    S0 = sum(weights)
    out = OrderedDict()
    keys = delta_list[0].keys()
    for k in keys:
        v0 = delta_list[0][k]
        if not torch.is_tensor(v0):
            out[k] = v0
            continue
        acc = None
        for i, d in enumerate(delta_list):
            term = d[k] * weights[i]
            acc = term if acc is None else (acc + term)
        out[k] = acc / S0
    return out
# -*- coding: utf-8 -*-
# src/trainer/unlearn/pum.py
from dataclasses import dataclass
from typing import List, Optional, Literal
from collections import OrderedDict
import copy
import torch

from trainer.pum_utils import (
    construct_seed64,
    compute_layerwise_sigma,
    generate_zero_sum_noises,
    sample_T_attention,
    apply_T_attention_weights,
    inverse_T_on_update_attention,
    sample_ffn_permutation,
    apply_T_ffn_weights,
    inverse_T_on_update_ffn,
    harmonic_aggregate,
)

from trainer.utils import (
    compute_gradascent, 
    compute_graddiff,
    compute_dpo,
    compute_npo,
    compute_simnpo,
    compute_satimp,
    compute_undial,
    compute_wga,
    prepare_ref_model,
)



@dataclass
class PUMConfig:
    R: int = 5
    m: int = 3
    alphas: Optional[List[float]] = None         # If None, the default value is [1,2,4,...]
    kappa: float = 0.1                           # σ_l = κ·RMS(v_l)
    eta_srv: float = 1.0                         # Server Step Size
    rope_aware: bool = False

    # Client Local Unlearning (First-order Small Steps)
    client_method: Literal["GrandAscent", "GradDiff", "DPO", "NPO", "SimNPO", "SatImp", "UnDIAL", "WGA"] = "GrandAscent"
    client_steps: int = 10
    client_lr: float = 1e-5

    # Deterministic Seeds
    s_noise: Optional[List[int]] = None          # Each round s_r
    t_reparam: Optional[List[int]] = None        # Each round t_r


class PUM:
    """
    Server-side orchestrator: implements the algorithm from
    For each round r: Construct m copies of zero-sum noise (including α scaling), sample T_{k,r,l} for each layer and publish copies
    → Client-side local forgetting → Apply T^{-1} to the increment and return to the common coordinate system → Harmonic aggregation → Server outer loop update.
    """
    def __init__(self, model, tokenizer, cfg: PUMConfig):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg

    def alphas(self) -> List[float]:
        if self.cfg.alphas is not None:
            assert len(self.cfg.alphas) == self.cfg.m
            return self.cfg.alphas
        xs = [1.0]
        for _ in range(1, self.cfg.m):
            xs.append(xs[-1] * 2.0)
        return xs

    @torch.no_grad()
    def layer_groups(self) -> dict:
        """
        Group the `state_dict` keys by transformer layer (parameters within the same layer share the same `σ_l`).
        If your repository already has a more refined layer enumeration function, you can replace this implementation.
        """
        sd = self.model.state_dict()
        groups = {}
        for k in sd.keys():
            if not k.startswith("model.layers."):
                # Embedding and lm_head can also be grouped into a separate group.
                continue
            parts = k.split(".")
            if len(parts) > 3:
                lid = parts[2]   # e.g., "0","1",...
                groups.setdefault(f"layer{lid}", []).append(k)
        return groups

    def client_unlearn(self, ul_model, ul_forget, ul_retain=None, ref_model=None, device="cuda"):
        """
        Unified client-side inner loop:
        - Format batches drawn from the dataloader into the input structures expected by the compute_* routines in utils.py.
        - Delegate loss computation to the compute_* functions in utils.py to avoid duplicating implementations.
        - Supported methods: ga / graddiff / dpo / npo / simnpo / satimp / undial / wga
        """
    
        def to_device_batch(batch, dev):
            return {k: v.to(dev) for k, v in batch.items()}

        def next_batch(it, dl):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            return b, it

        # Disable the key-value (KV) cache to avoid unnecessary memory consumption and inadvertent computation-graph interruptions
        orig_use_cache = getattr(getattr(ul_model, "config", None), "use_cache", None)
        if orig_use_cache is not None:
            ul_model.config.use_cache = False

        ul_model.to(device)
        ul_model.train()
        optim = torch.optim.AdamW(ul_model.parameters(), lr=self.cfg.client_lr)

        # dataloader 迭代器
        it_f = iter(ul_forget)
        it_r = iter(ul_retain) if ul_retain is not None else None

        # Hyperparameters (fall back to defaults defined in the utils functions when not specified in cfg) ----
        method = str(self.cfg.client_method).lower()
        retain_loss_type = getattr(self.cfg, "retain_loss_type", "NLL")

        alpha = getattr(self.cfg, "retain_alpha", 1.0)
        gamma = getattr(self.cfg, "forget_gamma", 1.0)

        dpo_beta    = getattr(self.cfg, "dpo_beta", 0.1)
        npo_beta    = getattr(self.cfg, "npo_beta", dpo_beta)
        undial_beta = getattr(self.cfg, "undial_beta", 10.0)

        simnpo_beta  = getattr(self.cfg, "simnpo_beta", 4.5)
        simnpo_delta = getattr(self.cfg, "simnpo_delta", 0.0)

        wga_beta     = getattr(self.cfg, "wga_beta", 1.0)
        satimp_beta1 = getattr(self.cfg, "satimp_beta1", 5.0)
        satimp_beta2 = getattr(self.cfg, "satimp_beta2", 1.0)

        # Lazily construct the teacher (reference) model if required, or reuse an externally provided ref_model ----
        need_teacher = (method in {"DPO", "NPO", "UnDIAL"}) or (
            retain_loss_type == "KL" and method in {"GradDiff", "WGA", "SimNPO", "SatImp"}
        )
        teacher = ref_model
        if teacher is None and need_teacher:
            teacher = prepare_ref_model(ul_model, device=device)

        last = None
        for _ in range(self.cfg.client_steps):
            bf, it_f = next_batch(it_f, ul_forget)

            if method == "DPO":
                assert isinstance(bf, dict) and ("original" in bf and "alternate" in bf), \
                    "DPO requires {'original': batch, 'alternate': batch}"
                bf_inputs = {
                    "original": to_device_batch(bf["original"], device),
                    "alternate": to_device_batch(bf["alternate"], device),
                }
                assert ul_retain is not None, "DPO requires retain dataloader"
                br, it_r = next_batch(it_r, ul_retain)
                br_inputs = to_device_batch(br, device)
                inputs = {"forget": bf_inputs, "retain": br_inputs}

            else:
                bf_inputs = to_device_batch(bf, device)
                if method in {"GradDiff", "NPO", "SimNPO", "SatImp", "UnDIAL", "WGA"}:
                    assert ul_retain is not None, f"{method} requires retain dataloader"
                    br, it_r = next_batch(it_r, ul_retain)
                    br_inputs = to_device_batch(br, device)
                    inputs = {"forget": bf_inputs, "retain": br_inputs}
                else:  # "GA"
                    inputs = {"forget": bf_inputs}

            if method == "GradAscent":
                loss, _ = compute_gradascent(ul_model, inputs)

            elif method == "GradDiff":
                loss, _ = compute_graddiff(
                    ul_model, inputs,
                    gamma=gamma, alpha=alpha,
                    retain_loss_type=retain_loss_type,
                )
            
            elif method == "DPO":
                loss, _ = compute_dpo(
                    ul_model, inputs,
                    alpha=alpha, beta=dpo_beta, gamma=gamma,
                    retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                )

            elif method == "SimNPO":
                loss, _ = compute_simnpo(
                    ul_model, inputs,
                    alpha=alpha, beta=simnpo_beta, delta=simnpo_delta, gamma=gamma,
                    retain_loss_type=retain_loss_type,
                )

            elif method == "NPO":
                loss, _ = compute_npo(
                    ul_model, inputs,
                    alpha=alpha, beta=npo_beta, gamma=gamma,
                    retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                )

            elif method == "UnDIAL":
                loss, _ = compute_undial(
                    ul_model, inputs,
                    alpha=alpha, beta=undial_beta, gamma=gamma,
                    retain_loss_type=retain_loss_type, ref_model=teacher, device=device,
                )

            elif method == "WGA":
                loss, _ = compute_wga(
                    ul_model, inputs,
                    alpha=alpha, beta=wga_beta, gamma=gamma,
                    retain_loss_type=retain_loss_type,
                )

            elif method == "SatImp":
                loss, _ = compute_satimp(
                    ul_model, inputs,
                    alpha=alpha, beta1=satimp_beta1, beta2=satimp_beta2, gamma=gamma,
                    retain_loss_type=retain_loss_type,
                )

            else:
                raise NotImplementedError(f"unknown client_method={self.cfg.client_method}")

            # Backpropagation and optimization ---
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ul_model.parameters(), 1.0)
            optim.step()
            last = float(loss.detach().cpu())

        # 恢复 use_cache
        if orig_use_cache is not None:
            ul_model.config.use_cache = orig_use_cache

        return last



    def run(self, ul_forget, ul_retain=None, ref_model=None, device="cuda", base_sd: OrderedDict = None):
        self.model.to(device).eval()

        # θ_{r-1}
        theta_prev: OrderedDict = copy.deepcopy(self.model.state_dict())
        alphas = self.alphas()
        layer_groups = self.layer_groups()

        # Read attention hyperparameters from config or model
        H_Q = getattr(self.model.config, "num_attention_heads", 32)
        H_KV = getattr(self.model.config, "num_key_value_heads", H_Q)
        d_h = getattr(self.model.config, "hidden_size", 4096) // H_Q
        L = getattr(self.model.config, "num_hidden_layers", 1)

        for r in range(1, self.cfg.R + 1):
            # ---- Line 1: σ_l; Lines 2–4: Layer-by-layer zero-sum noise + α scaling ----
            sig = compute_layerwise_sigma(theta_prev, layer_groups, self.cfg.kappa, base_sd=base_sd)
            s_r = (self.cfg.s_noise[r - 1] if self.cfg.s_noise else 101_000 + r)
            noises = generate_zero_sum_noises(
                theta_prev, layer_groups, sig, self.cfg.m, alphas, s_r, torch.device(device)
            )

            # -- Collect the parameter updates Δ for each published copy (note: aggregation must be performed after the per-copy loop) --
            delta_list: List[OrderedDict] = []
            alpha_list: List[float] = []

            # ---- Lines 6–11: Layer-wise copying ----
            for k in range(1, self.cfg.m + 1):
                # Line 8: Publish θ_pub^{(k,r)} = T(θ_{r-1} + ε_k)
                pub_sd = copy.deepcopy(theta_prev)

                # Add noise (add key by key)
                for name in pub_sd.keys():
                    if torch.is_tensor(pub_sd[name]) and torch.is_tensor(noises[k - 1][name]):
                        pub_sd[name] = pub_sd[name] + noises[k - 1][name]

                # === Apply T (Attention + FFN; no head permutation) to each layer ===
                for lid in range(L):
                    t_r = (self.cfg.t_reparam[r - 1] if self.cfg.t_reparam else 202_000 + r)
                    # ---- Attention Blocks ----
                    base_attn = f"model.layers.{lid}.self_attn"
                    qk = f"{base_attn}.q_proj.weight"; kk = f"{base_attn}.k_proj.weight"
                    vk = f"{base_attn}.v_proj.weight"; ok = f"{base_attn}.o_proj.weight"
                    W_Q = pub_sd.get(qk, None)
                    W_K = pub_sd.get(kk, None)
                    W_V = pub_sd.get(vk, None)
                    W_O = pub_sd.get(ok, None)

                    if all(t is not None for t in [W_Q, W_K, W_V, W_O]):
                        # Shape Assertions (Early Detection of Configuration Mismatches)
                        assert W_Q.shape[0] == H_Q * d_h and W_O.shape[1] == H_Q * d_h, \
                            f"Q/O shapes do not match H_Q*d_h: {W_Q.shape}, {W_O.shape}, H_Q*d_h={H_Q*d_h}"
                        assert W_K.shape[0] == H_KV * d_h and W_V.shape[0] == H_KV * d_h, \
                            f"K/V shapes do not match H_KV*d_h: {W_K.shape}, {W_V.shape}, H_KV*d_h={H_KV*d_h}"

                        # Construct seeds for T_{k,r,l} for this layer; generate U,S by weight dtype/device
                        seed_layer = construct_seed64(t_r, "T", k, lid)
                        U, S = sample_T_attention(
                            H_Q, H_KV, d_h, self.cfg.rope_aware,
                            seed_round=seed_layer, k_copy=k,
                            device=W_Q.device, dtype=W_Q.dtype
                        )
                        W_Qp, W_Kp, W_Vp, W_Op = apply_T_attention_weights(W_Q, W_K, W_V, W_O, U, S)
                        pub_sd[qk], pub_sd[kk], pub_sd[vk], pub_sd[ok] = W_Qp, W_Kp, W_Vp, W_Op

                    # ---- FFN (SwiGLU) Blocks: Same layer gate/up/down share a permutation ----
                    base_ffn = f"model.layers.{lid}.mlp"
                    gk = f"{base_ffn}.gate_proj.weight"; uk = f"{base_ffn}.up_proj.weight"
                    dk = f"{base_ffn}.down_proj.weight"
                    gb = f"{base_ffn}.gate_proj.bias";  ub = f"{base_ffn}.up_proj.bias";  db = f"{base_ffn}.down_proj.bias"

                    W1g = pub_sd.get(gk); W1u = pub_sd.get(uk); W2 = pub_sd.get(dk)
                    b1g = pub_sd.get(gb); b1u = pub_sd.get(ub); b2 = pub_sd.get(db)

                    if (W1g is not None) and (W1u is not None) and (W2 is not None):
                        d_ff = W1g.shape[0]
                        P_ffn = sample_ffn_permutation(
                            d_ff,
                            seed=construct_seed64(t_r, "FFN-P", k, lid),
                            device=W1g.device,
                            dtype=W1g.dtype,           # 与权重 dtype 对齐以减少显存/避免隐式 cast
                        )

                        W1g_p, W1u_p, W2_p, b1g_p, b1u_p, b2_p = apply_T_ffn_weights(
                            W1g, W1u, W2, P_ffn, b1_gate=b1g, b1_up=b1u, b2_down=b2
                        )
                        pub_sd[gk], pub_sd[uk], pub_sd[dk] = W1g_p, W1u_p, W2_p
                        if b1g is not None: pub_sd[gb] = b1g_p
                        if b1u is not None: pub_sd[ub] = b1u_p
                        if b2  is not None: pub_sd[db] = b2_p # According to the paper: the down bias remains unchanged, so we keep the original value here.

                # Start the client model with the published parameters
                model_k = copy.deepcopy(self.model).to(device)
                model_k.load_state_dict(pub_sd, strict=False)

                # Line 9–10: The client performs local unlearning steps on D_f (possibly with D_r/ref_model)
                _ = self.client_unlearn(model_k, ul_forget, ul_retain, ref_model, device=device)

                # Line 11: Δ' = θ_after - θ_pub
                after_sd = model_k.state_dict()
                delta_prime = OrderedDict()
                for name in pub_sd.keys():
                    if torch.is_tensor(pub_sd[name]):
                        delta_prime[name] = after_sd[name] - pub_sd[name]
                    else:
                        delta_prime[name] = pub_sd[name]

                # === Apply T^{-1} (Attention + FFN) to each layer ===
                for lid in range(L):
                    t_r = (self.cfg.t_reparam[r - 1] if self.cfg.t_reparam else 202_000 + r)
                    # ---- Attention Blocks ----
                    base_attn = f"model.layers.{lid}.self_attn"
                    qk = f"{base_attn}.q_proj.weight"; kk = f"{base_attn}.k_proj.weight"
                    vk = f"{base_attn}.v_proj.weight"; ok = f"{base_attn}.o_proj.weight"
                    dQp = delta_prime.get(qk, None)
                    dKp = delta_prime.get(kk, None)
                    dVp = delta_prime.get(vk, None)
                    dOp = delta_prime.get(ok, None)
                    if all(t is not None for t in [dQp, dKp, dVp, dOp]):
                        seed_layer = construct_seed64(t_r, "T", k, lid)
                        U, S = sample_T_attention(
                            H_Q, H_KV, d_h, self.cfg.rope_aware,
                            seed_round=seed_layer, k_copy=k,
                            device=dQp.device, dtype=dQp.dtype
                        )
                        dQ, dK, dV, dO = inverse_T_on_update_attention(dQp, dKp, dVp, dOp, U, S)
                        delta_prime[qk], delta_prime[kk], delta_prime[vk], delta_prime[ok] = dQ, dK, dV, dO

                    # ---- FFN Blocks ----
                    base_ffn = f"model.layers.{lid}.mlp"
                    gk = f"{base_ffn}.gate_proj.weight"; uk = f"{base_ffn}.up_proj.weight"
                    dk = f"{base_ffn}.down_proj.weight"
                    gb = f"{base_ffn}.gate_proj.bias";  ub = f"{base_ffn}.up_proj.bias";  db = f"{base_ffn}.down_proj.bias"

                    dW1g_p = delta_prime.get(gk); dW1u_p = delta_prime.get(uk); dW2_p = delta_prime.get(dk)
                    db1g_p = delta_prime.get(gb); db1u_p = delta_prime.get(ub); db2_p = delta_prime.get(db)

                    if (dW1g_p is not None) and (dW1u_p is not None) and (dW2_p is not None):
                        d_ff = dW1g_p.shape[0]
                        P_ffn = sample_ffn_permutation(
                            d_ff,
                            seed=construct_seed64(t_r, "FFN-P", k, lid),
                            device=dW1g_p.device,
                            dtype=dW1g_p.dtype,
                        )

                        dW1g, dW1u, dW2, db1g, db1u, db2 = inverse_T_on_update_ffn(
                            dW1g_p, dW1u_p, dW2_p, P_ffn, db1_gate_p=db1g_p, db1_up_p=db1u_p, db2_down_p=db2_p
                        )
                        delta_prime[gk], delta_prime[uk], delta_prime[dk] = dW1g, dW1u, dW2
                        if db1g_p is not None: delta_prime[gb] = db1g
                        if db1u_p is not None: delta_prime[ub] = db1u
                        if db2_p  is not None: delta_prime[db] = db2

                delta_list.append(delta_prime)
                alpha_list.append(alphas[k - 1])


            # Line 12: Harmonic aggregation Δ̄^{(r)} = Harmonic_Aggregate({Δ'^{(k,r)}}, {α_k})
            bar_delta = harmonic_aggregate(delta_list, alpha_list)
            total_sq = torch.stack([ (v.float()**2).sum() for v in bar_delta.values() if torch.is_tensor(v) ]).sum()
            print(f"[Debug] Round {r}: ||Δ̄||₂ = {total_sq.sqrt().item():.6f}")


            # Line 13: Server updates θ_r = θ_{r-1} + η_srv · \barΔ^{(r)}
            for n in theta_prev.keys():
                if torch.is_tensor(theta_prev[n]) and torch.is_tensor(bar_delta.get(n, None)):
                    theta_prev[n] = theta_prev[n] + self.cfg.eta_srv * bar_delta[n]

            self.model.load_state_dict(theta_prev, strict=False)

        # Line 14: Return θ_R
        return self.model
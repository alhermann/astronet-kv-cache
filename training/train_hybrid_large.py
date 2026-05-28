"""Memory-tight hybrid training for 70B+ models on 2x24 GiB.

Key differences from train_hybrid.py:
1. Sequential device map + per-GPU max_memory budget (22+23 GiB by default)
2. SDPA attention (not eager) — never materializes full attention tensors
3. Pre-hook on inject layers ONLY to capture hidden states for manual Q@K heuristic
   (instead of output_attentions=True for all 80 layers)
4. Hook on sense_layer to capture hidden state (instead of output_hidden_states for all layers)
5. Dequantized K/V projections stored in FP16 (not FP32) — half the memory

The "fact-window forward" cost drops from ~2-3 GB to ~150 MB transient,
making 72B / Llama 70B hybrid training feasible on 2x24 GiB hardware.

Heuristic = sum of per-token cross-window attention from inject layers only
(consistent with eval_large_model.py for the baselines).
"""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset


def _select_kv(all_kv, li, idx):
    K = torch.cat(all_kv[li][0], dim=2)
    V = torch.cat(all_kv[li][1], dim=2)
    li_idx = idx.to(K.device)
    return K[:, :, li_idx, :], V[:, :, li_idx, :]


class AstroHybridLarge(nn.Module):
    """AstroNet for large models — same architecture as AstroHybrid but
    stores dequantized projections in FP16 to halve memory."""
    def __init__(self, hidden_dim, n_mem_tokens=16, attn_dim=256,
                 n_kv_heads=4, head_dim=128, n_layers=28, inject_layers=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mem_tokens = n_mem_tokens
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.inject_layers = inject_layers or list(range(n_layers))
        self._kv_weights = {}

        self.queries = nn.Parameter(torch.randn(1, n_mem_tokens, attn_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attn_scale = 1.0 / math.sqrt(attn_dim)

        self.log_alpha = nn.Parameter(torch.tensor(-0.5))
        self.register_buffer('g', torch.zeros(1, n_mem_tokens, hidden_dim))

        bottleneck = 256
        self.layer_down = nn.ModuleDict()
        self.layer_up = nn.ModuleDict()
        for li in self.inject_layers:
            self.layer_down[str(li)] = nn.Linear(hidden_dim, bottleneck, bias=False)
            self.layer_up[str(li)] = nn.Linear(bottleneck, hidden_dim, bias=False)
        self.shared_down = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.shared_up = nn.Linear(bottleneck, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)
        for d in list(self.layer_down.values()) + [self.shared_down]:
            nn.init.xavier_uniform_(d.weight, gain=1.0)
        for u in list(self.layer_up.values()) + [self.shared_up]:
            nn.init.xavier_uniform_(u.weight, gain=0.5)

    def extract_model_weights(self, model):
        """Dequantize K/V projections to FP16, store on CPU to save GPU memory.
        Streamed to GPU per-layer during generate_kv() (~1 sec extra per step)."""
        print("  Extracting model KV projection weights (FP16 → CPU)...", flush=True)
        for li in range(self.n_layers):
            layer = model.model.layers[li].self_attn
            k_w = layer.k_proj.weight
            v_w = layer.v_proj.weight
            layer_device = k_w.device
            if hasattr(k_w, 'quant_state'):
                import bitsandbytes as bnb
                k_w = bnb.functional.dequantize_4bit(k_w.data, k_w.quant_state).half().cpu()
                v_w = bnb.functional.dequantize_4bit(v_w.data, v_w.quant_state).half().cpu()
            else:
                k_w = k_w.half().cpu()
                v_w = v_w.half().cpu()
            self._kv_weights[li] = {
                'k': k_w.detach(),
                'v': v_w.detach(),
                'device': layer_device,  # remember where the layer lives for streaming
            }
        total_params = sum(self._kv_weights[li]['k'].numel() + self._kv_weights[li]['v'].numel()
                           for li in range(self.n_layers))
        total_mb = total_params * 2 / 1024 / 1024
        print(f"  Extracted {len(self._kv_weights)} layers, "
              f"K shape: {self._kv_weights[0]['k'].shape}, "
              f"total ~{total_mb:.0f} MB FP16 on CPU", flush=True)

    @property
    def alpha(self):
        return torch.sigmoid(self.log_alpha)

    def sense(self, hidden_states):
        h = self.input_norm(hidden_states.float())
        B = h.shape[0]
        keys = self.key_proj(h)
        attn = torch.bmm(self.queries.expand(B, -1, -1),
                         keys.transpose(1, 2)) * self.attn_scale
        weights = torch.softmax(attn, dim=-1)
        return torch.bmm(weights, h)

    def update_state(self, sensed, keep_grad=False):
        alpha = self.alpha
        g_exp = self.g.expand(sensed.shape[0], -1, -1)
        new_g = (1 - alpha) * g_exp + alpha * sensed
        new_g = new_g.mean(dim=0, keepdim=True)
        self.g = new_g if keep_grad else new_g.detach()

    def generate_kv(self, layer_idx, real_kv_sample):
        """Generate K, V. Compute matmul on CPU (tiny: 16×8192 × 1024×8192),
        then move only the output (~32 KB) to GPU. Zero GPU memory pressure."""
        real_K, real_V = real_kv_sample
        li_str = str(layer_idx)
        if li_str in self.layer_down:
            virtual_hidden = self.layer_up[li_str](F.gelu(self.layer_down[li_str](self.g)))
        else:
            virtual_hidden = self.shared_up(F.gelu(self.shared_down(self.g)))
        virtual_hidden = virtual_hidden.clamp(-100, 100)

        # Compute projection on CPU (weights already there, move virtual_hidden to CPU)
        vh_cpu = virtual_hidden.cpu().half()
        k_flat = F.linear(vh_cpu, self._kv_weights[layer_idx]['k'])
        v_flat = F.linear(vh_cpu, self._kv_weights[layer_idx]['v'])

        # Reshape on CPU, then move tiny result to layer's GPU
        k_out = k_flat.view(1, self.n_mem_tokens, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v_out = v_flat.view(1, self.n_mem_tokens, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        layer_dev = self._kv_weights[layer_idx]['device']
        return k_out.to(device=layer_dev, dtype=real_K.dtype), \
               v_out.to(device=layer_dev, dtype=real_V.dtype)

    def reset_state(self):
        self.g.zero_()

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Pre-hook helpers for capturing hidden states (no full output_hidden_states) ===
class HookCapture:
    """Context manager that registers pre-hooks on inject + sense layers
    to capture hidden states for the SDPA forward path."""
    def __init__(self, model, inject_layers, sense_layer):
        self.model = model
        self.inject_layers = inject_layers
        self.sense_layer = sense_layer
        self.captured = {}  # layer_idx -> hidden_states tensor
        self.handles = []

    def __enter__(self):
        self.captured.clear()

        def make_hook(li):
            def hook(module, args, kwargs):
                h = args[0] if args else kwargs.get('hidden_states')
                if h is not None:
                    self.captured[li] = h.detach()
            return hook

        layers_to_hook = set(self.inject_layers) | {self.sense_layer}
        for li in layers_to_hook:
            handle = self.model.model.layers[li].self_attn.register_forward_pre_hook(
                make_hook(li), with_kwargs=True
            )
            self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def compute_heuristic_inject_only(captured, all_kv_window, inject_layers,
                                   nq, nkv, qpk, hd, model, embed_device, sl):
    """Compute multiplicative heuristic from inject layers only."""
    imp = torch.zeros(sl, device=embed_device)
    for li in inject_layers:
        if li not in captured:
            continue
        h = captured[li]
        layer_dev = model.model.layers[li].self_attn.q_proj.weight.device
        Q = model.model.layers[li].self_attn.q_proj(
            h.to(layer_dev).float()).half()
        Q = Q.view(1, sl, nq, hd).transpose(1, 2)[0]
        K = all_kv_window[li][0]  # last appended K for this window
        for hi in range(nkv):
            sc = torch.matmul(
                Q[hi*qpk:(hi+1)*qpk].float(),
                K[hi].float().T) / math.sqrt(hd)
            imp += sc.sum(dim=(0, 1)).to(embed_device)
        del Q, K
    return imp


def train_epoch(model, tokenizer, astro, optimizer, samples,
                inject_layers, sense_layer, k_real, embed_device, max_length=256):
    """One training epoch with memory-tight forward path."""
    astro.train()
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv

    total_loss = 0
    n_total = 0

    for si, s in enumerate(samples):
        astro.reset_state()
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []

        # Process fact windows with SDPA + pre-hooks
        with HookCapture(model, inject_layers, sense_layer) as cap:
            for wi in range(len(s.windows) - 1):
                ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=max_length,
                                truncation=True).to(embed_device)
                sl = ids['input_ids'].shape[1]
                cap.captured.clear()
                with torch.no_grad():
                    out = model(input_ids=ids['input_ids'], use_cache=True,
                                output_attentions=False)
                    # Per-window heuristic (last KV pair just appended)
                    window_kv = {li: (out.past_key_values[li][0][0], out.past_key_values[li][1][0])
                                 for li in inject_layers}
                    imp = compute_heuristic_inject_only(
                        cap.captured, window_kv, inject_layers,
                        nq, nkv, qpk, hd, model, embed_device, sl
                    )
                    imp[:4] = -1e9
                    all_attn.append(imp)
                    for li in range(nl):
                        all_kv[li][0].append(out.past_key_values[li][0].detach())
                        all_kv[li][1].append(out.past_key_values[li][1].detach())
                    # AstroNet sense from captured hidden state
                    if sense_layer in cap.captured:
                        hidden = cap.captured[sense_layer].detach()
                        sensed = astro.sense(hidden)
                        astro.update_state(sensed, keep_grad=(wi >= len(s.windows) - 3))
                    del out
                    torch.cuda.empty_cache()

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn).detach()

        # Cross-window scoring (no_grad)
        q_text = f'Question: {s.question}\nAnswer:'
        q_ids = tokenizer(q_text, return_tensors='pt', max_length=128, truncation=True).to(embed_device)
        # Capture inject layer hidden for Q
        with HookCapture(model, inject_layers, sense_layer) as cap:
            cap.captured.clear()
            with torch.no_grad():
                _ = model(input_ids=q_ids['input_ids'], output_attentions=False)
                cross = torch.zeros(total, device=embed_device)
                for li in inject_layers:
                    if li not in cap.captured:
                        continue
                    h_q = cap.captured[li]
                    layer_dev = model.model.layers[li].self_attn.q_proj.weight.device
                    Q = model.model.layers[li].self_attn.q_proj(
                        h_q.to(layer_dev).float()).half().view(-1, nq, hd)
                    K = torch.cat(all_kv[li][0], dim=2)[0]
                    for hi in range(nkv):
                        sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                          K[hi].float().T) / math.sqrt(hd)
                        cross += sc.sum(dim=(0, 1)).to(embed_device)
                    del Q, K
            torch.cuda.empty_cache()
        cross[:4] = -1e9
        mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult[:4] = -1e9

        k = min(k_real, total)
        _, idx = mult.topk(k)
        idx = idx.sort().values

        # Build hybrid cache: AstroNet KV + selected real KV
        n_mem = astro.n_mem_tokens
        cache = DynamicCache()
        for li in range(nl):
            K_real, V_real = _select_kv(all_kv, li, idx)
            K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
            K_combined = torch.cat([K_mem, K_real], dim=2)
            V_combined = torch.cat([V_mem, V_real], dim=2)
            cache.update(K_combined, V_combined, li)

        # FREE all_kv + intermediates BEFORE gradient forward to maximize headroom
        del all_kv, all_attn, heur, mult, cross, idx
        torch.cuda.empty_cache()

        # Gradient forward (small seq, only this is autograd-tracked).
        # CRITICAL: model.train() required for gradient checkpointing to activate.
        # CRITICAL: Monkey-patch DynamicCache.update to READ-ONLY mode.
        # Without this, update() appends new KV to the cache even with use_cache=False
        # (there's no use_cache guard in Qwen2Attention). During gradient checkpoint
        # recomputation, the cache would be double-updated → shape mismatch.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()

        # Monkey-patch: read the cache but don't mutate it
        _orig_update = DynamicCache.update
        def _readonly_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx < len(self.key_cache):
                key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return key_states, value_states
        DynamicCache.update = _readonly_update

        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer: {s.answer}'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(embed_device)
        input_ids = fq['input_ids']
        total_cache_len = n_mem + k
        pos = torch.arange(total_cache_len, total_cache_len + input_ids.shape[1],
                           device=embed_device).unsqueeze(0)
        out = model(input_ids=input_ids, past_key_values=cache, position_ids=pos,
                    use_cache=False)

        # Restore original update
        DynamicCache.update = _orig_update

        logits = out.logits[0, :-1]
        targets = input_ids[0, 1:]
        loss = F.cross_entropy(logits.float(), targets)

        # Free forward intermediates before backward to maximize headroom
        del out
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(astro.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_total += 1

        # Disable gradient checkpointing + re-enable cache + eval mode for fact-window forwards
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()

        del out, cache
        torch.cuda.empty_cache()

        if (si + 1) % 10 == 0:
            print(f'  train {si+1}/{len(samples)}: loss={total_loss/n_total:.3f} '
                  f'alpha={astro.alpha.item():.3f}', flush=True)

    return total_loss / max(n_total, 1)


def _build_hybrid_cache(astro, all_kv, idx, nl, k_real):
    """Helper: build a DynamicCache from AstroNet + selected real KVs.
    Called twice per zeroth-order step (once for current params, once for +ε)."""
    cache = DynamicCache()
    for li in range(nl):
        K_real, V_real = _select_kv(all_kv, li, idx)
        K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
        K_combined = torch.cat([K_mem, K_real], dim=2)
        V_combined = torch.cat([V_mem, V_real], dim=2)
        cache.update(K_combined, V_combined, li)
    return cache


def _forward_loss(model, input_ids, cache, n_mem, k, embed_device):
    """Compute cross-entropy loss for a query given a pre-built cache.
    Always runs under torch.no_grad() — caller is responsible for the context."""
    total_cache_len = n_mem + k
    pos = torch.arange(total_cache_len, total_cache_len + input_ids.shape[1],
                       device=embed_device).unsqueeze(0)
    out = model(input_ids=input_ids, past_key_values=cache,
                position_ids=pos, use_cache=False)
    logits = out.logits[0, :-1]
    targets = input_ids[0, 1:]
    return F.cross_entropy(logits.float(), targets)


def train_epoch_zeroth_order(model, tokenizer, astro, optimizer, samples,
                              inject_layers, sense_layer, k_real, embed_device,
                              max_length=256, epsilon=1e-3):
    """One training epoch using zeroth-order (forward-difference) gradient estimation.

    No model backward pass is needed — only two forward passes per sample:
      1. loss_current  — current AstroNet params
      2. loss_plus     — params + ε * random_direction

    Gradient estimate: g = (loss_plus - loss_current) / ε * random_direction
    The estimate is injected into .grad of each parameter so that the AdamW
    optimizer can apply its momentum and weight-decay correctly.

    Fact-window processing and score computation are identical to train_epoch.
    """
    astro.train()
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv

    total_loss = 0
    n_total = 0

    for si, s in enumerate(samples):
        astro.reset_state()
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []

        # ------------------------------------------------------------------ #
        # Step 1: Process fact windows — identical to train_epoch             #
        # ------------------------------------------------------------------ #
        with HookCapture(model, inject_layers, sense_layer) as cap:
            for wi in range(len(s.windows) - 1):
                ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=max_length,
                                truncation=True).to(embed_device)
                sl = ids['input_ids'].shape[1]
                cap.captured.clear()
                with torch.no_grad():
                    out = model(input_ids=ids['input_ids'], use_cache=True,
                                output_attentions=False)
                    window_kv = {li: (out.past_key_values[li][0][0], out.past_key_values[li][1][0])
                                 for li in inject_layers}
                    imp = compute_heuristic_inject_only(
                        cap.captured, window_kv, inject_layers,
                        nq, nkv, qpk, hd, model, embed_device, sl
                    )
                    imp[:4] = -1e9
                    all_attn.append(imp)
                    for li in range(nl):
                        all_kv[li][0].append(out.past_key_values[li][0].detach())
                        all_kv[li][1].append(out.past_key_values[li][1].detach())
                    if sense_layer in cap.captured:
                        hidden = cap.captured[sense_layer].detach()
                        sensed = astro.sense(hidden)
                        astro.update_state(sensed, keep_grad=False)
                    del out
                    torch.cuda.empty_cache()

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn).detach()

        # ------------------------------------------------------------------ #
        # Step 2: Cross-window multiplicative scoring — identical to          #
        #         train_epoch                                                  #
        # ------------------------------------------------------------------ #
        q_text = f'Question: {s.question}\nAnswer:'
        q_ids = tokenizer(q_text, return_tensors='pt', max_length=128, truncation=True).to(embed_device)
        with HookCapture(model, inject_layers, sense_layer) as cap:
            cap.captured.clear()
            with torch.no_grad():
                _ = model(input_ids=q_ids['input_ids'], output_attentions=False)
                cross = torch.zeros(total, device=embed_device)
                for li in inject_layers:
                    if li not in cap.captured:
                        continue
                    h_q = cap.captured[li]
                    layer_dev = model.model.layers[li].self_attn.q_proj.weight.device
                    Q = model.model.layers[li].self_attn.q_proj(
                        h_q.to(layer_dev).float()).half().view(-1, nq, hd)
                    K = torch.cat(all_kv[li][0], dim=2)[0]
                    for hi in range(nkv):
                        sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                          K[hi].float().T) / math.sqrt(hd)
                        cross += sc.sum(dim=(0, 1)).to(embed_device)
                    del Q, K
            torch.cuda.empty_cache()
        cross[:4] = -1e9
        mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult[:4] = -1e9

        k = min(k_real, total)
        _, idx = mult.topk(k)
        idx = idx.sort().values

        # ------------------------------------------------------------------ #
        # Step 3: Build query input_ids for loss computation                  #
        # ------------------------------------------------------------------ #
        query = (f'Based on what you read earlier, answer the following question.\n'
                 f'Question: {s.question}\nAnswer: {s.answer}')
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(embed_device)
        input_ids = fq['input_ids']
        n_mem = astro.n_mem_tokens

        # ------------------------------------------------------------------ #
        # Step 4: Zeroth-order gradient estimation                            #
        # ------------------------------------------------------------------ #

        # 4a. Collect current parameter data and generate perturbation direction.
        #     All tensors are kept on their native device to avoid extra transfers.
        params = list(astro.parameters())
        directions = [torch.randn_like(p) for p in params]

        # Normalize the direction to unit norm across the full parameter vector
        # so that epsilon is scale-invariant.
        total_norm = sum(d.norm().item() ** 2 for d in directions) ** 0.5
        if total_norm > 0:
            directions = [d / total_norm for d in directions]

        # 4b. Compute loss_current — no_grad, current params
        with torch.no_grad():
            cache_cur = _build_hybrid_cache(astro, all_kv, idx, nl, k)
            loss_current = _forward_loss(model, input_ids, cache_cur, n_mem, k, embed_device)
            del cache_cur
            torch.cuda.empty_cache()

        # 4c. Apply +ε perturbation in-place
        with torch.no_grad():
            for p, d in zip(params, directions):
                p.add_(epsilon * d)

        # 4d. Rebuild hybrid cache (AstroNet params changed) and compute loss_plus
        with torch.no_grad():
            cache_plus = _build_hybrid_cache(astro, all_kv, idx, nl, k)
            loss_plus = _forward_loss(model, input_ids, cache_plus, n_mem, k, embed_device)
            del cache_plus
            torch.cuda.empty_cache()

        # 4e. Restore original parameters
        with torch.no_grad():
            for p, d in zip(params, directions):
                p.sub_(epsilon * d)

        # 4f. Gradient estimate: g = (loss_plus - loss_current) / epsilon * direction
        fd_scalar = (loss_plus.item() - loss_current.item()) / epsilon

        # 4g. Assign estimated gradients and run optimizer step
        optimizer.zero_grad()
        with torch.no_grad():
            for p, d in zip(params, directions):
                if p.grad is None:
                    p.grad = fd_scalar * d
                else:
                    p.grad.copy_(fd_scalar * d)

        torch.nn.utils.clip_grad_norm_(astro.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_current.item()
        n_total += 1

        del all_kv, all_attn, heur, mult, cross, directions, params
        torch.cuda.empty_cache()

        if (si + 1) % 10 == 0:
            print(f'  train(zo) {si+1}/{len(samples)}: '
                  f'loss={total_loss/n_total:.3f}  '
                  f'fd_scalar={fd_scalar:.4f}  '
                  f'alpha={astro.alpha.item():.3f}', flush=True)

    return total_loss / max(n_total, 1)


def evaluate(model, tokenizer, astro, samples, inject_layers, sense_layer,
             k_real, embed_device, max_length=256):
    """Eval: compare hybrid (16+k_real) vs pure (300 real)."""
    astro.eval()
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    n_mem = astro.n_mem_tokens

    correct_hybrid = 0
    correct_pure = 0

    for si, s in enumerate(samples):
        astro.reset_state()
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []

        with HookCapture(model, inject_layers, sense_layer) as cap:
            for wi in range(len(s.windows) - 1):
                ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=max_length,
                                truncation=True).to(embed_device)
                sl = ids['input_ids'].shape[1]
                cap.captured.clear()
                with torch.no_grad():
                    out = model(input_ids=ids['input_ids'], use_cache=True,
                                output_attentions=False)
                    window_kv = {li: (out.past_key_values[li][0][0], out.past_key_values[li][1][0])
                                 for li in inject_layers}
                    imp = compute_heuristic_inject_only(
                        cap.captured, window_kv, inject_layers,
                        nq, nkv, qpk, hd, model, embed_device, sl
                    )
                    imp[:4] = -1e9
                    all_attn.append(imp)
                    for li in range(nl):
                        all_kv[li][0].append(out.past_key_values[li][0])
                        all_kv[li][1].append(out.past_key_values[li][1])
                    if sense_layer in cap.captured:
                        hidden = cap.captured[sense_layer].detach()
                        sensed = astro.sense(hidden)
                        astro.update_state(sensed)
                    del out
                    torch.cuda.empty_cache()

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        q_text = f'Question: {s.question}\nAnswer:'
        q_ids = tokenizer(q_text, return_tensors='pt', max_length=128, truncation=True).to(embed_device)
        with HookCapture(model, inject_layers, sense_layer) as cap:
            cap.captured.clear()
            with torch.no_grad():
                _ = model(input_ids=q_ids['input_ids'], output_attentions=False)
                cross = torch.zeros(total, device=embed_device)
                for li in inject_layers:
                    if li not in cap.captured:
                        continue
                    h_q = cap.captured[li]
                    layer_dev = model.model.layers[li].self_attn.q_proj.weight.device
                    Q = model.model.layers[li].self_attn.q_proj(
                        h_q.to(layer_dev).float()).half().view(-1, nq, hd)
                    K = torch.cat(all_kv[li][0], dim=2)[0]
                    for hi in range(nkv):
                        sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                          K[hi].float().T) / math.sqrt(hd)
                        cross += sc.sum(dim=(0, 1)).to(embed_device)
                    del Q, K
        cross[:4] = -1e9
        mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult[:4] = -1e9

        # Pure 300 eval
        k_pure = min(300, total)
        _, idx_pure = mult.topk(k_pure)
        idx_pure = idx_pure.sort().values
        cache_p = DynamicCache()
        for li in range(nl):
            K_p, V_p = _select_kv(all_kv, li, idx_pure)
            cache_p.update(K_p, V_p, li)
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(embed_device)
        pos_p = torch.arange(k_pure, k_pure + fq['input_ids'].shape[1], device=embed_device).unsqueeze(0)
        cur = fq['input_ids']; cc = cache_p; gen = []
        with torch.no_grad():
            for _ in range(20):
                o = model(input_ids=cur, past_key_values=cc, position_ids=pos_p)
                cc = o.past_key_values
                nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen.append(nxt[0, 0].item()); cur = nxt.to(embed_device)
                pos_p = torch.tensor([[k_pure + fq['input_ids'].shape[1] + len(gen) - 1]],
                                     device=embed_device)
                if nxt[0, 0].item() == tokenizer.eos_token_id: break
        if s.answer.lower() in tokenizer.decode(gen, skip_special_tokens=True).strip().lower():
            correct_pure += 1
        del cache_p, cc; torch.cuda.empty_cache()

        # Hybrid 16+k_real eval
        k_hyb = min(k_real, total)
        _, idx_hyb = mult.topk(k_hyb)
        idx_hyb = idx_hyb.sort().values
        cache_h = DynamicCache()
        with torch.no_grad():
            for li in range(nl):
                K_real, V_real = _select_kv(all_kv, li, idx_hyb)
                K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
                cache_h.update(torch.cat([K_mem, K_real], dim=2),
                               torch.cat([V_mem, V_real], dim=2), li)
        total_len = n_mem + k_hyb
        pos_h = torch.arange(total_len, total_len + fq['input_ids'].shape[1], device=embed_device).unsqueeze(0)
        cur = fq['input_ids']; cc = cache_h; gen = []
        with torch.no_grad():
            for _ in range(20):
                o = model(input_ids=cur, past_key_values=cc, position_ids=pos_h)
                cc = o.past_key_values
                nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen.append(nxt[0, 0].item()); cur = nxt.to(embed_device)
                pos_h = torch.tensor([[total_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                                     device=embed_device)
                if nxt[0, 0].item() == tokenizer.eos_token_id: break
        if s.answer.lower() in tokenizer.decode(gen, skip_special_tokens=True).strip().lower():
            correct_hybrid += 1
        del cache_h, cc, all_kv, all_attn, heur, mult, cross
        torch.cuda.empty_cache()

        if (si + 1) % 25 == 0:
            print(f'  eval {si+1}/{len(samples)}: pure300={correct_pure}/{si+1}  '
                  f'hybrid={correct_hybrid}/{si+1}', flush=True)

    n = len(samples)
    return correct_pure / n, correct_hybrid / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_eval', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--k_real', type=int, default=284)
    parser.add_argument('--n_mem', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256, help='Per-window max tokens')
    parser.add_argument('--gpu0_mem', default='22GiB')
    parser.add_argument('--gpu1_mem', default='23GiB')
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--eval_seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb,
        device_map='sequential',
        max_memory={0: args.gpu0_mem, 1: args.gpu1_mem},
        torch_dtype=torch.float16,
        attn_implementation='sdpa',
    )
    model.eval()
    embed_device = model.get_input_embeddings().weight.device

    hidden_dim = model.config.hidden_size
    nl = model.config.num_hidden_layers
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    sense_layer = nl // 2

    print(f"Model: {nl} layers, {hidden_dim} hidden, {nkv} KV heads, {hd} head_dim", flush=True)
    print(f"Inject layers: {inject_layers}, sense: {sense_layer}", flush=True)
    print(f"Embed device: {embed_device}", flush=True)

    astro = AstroHybridLarge(
        hidden_dim=hidden_dim,
        n_mem_tokens=args.n_mem,
        attn_dim=256,
        n_kv_heads=nkv,
        head_dim=hd,
        n_layers=nl,
        inject_layers=inject_layers,
    ).to(embed_device)
    astro.extract_model_weights(model)  # MUST happen before CPU offload (bnb needs GPU)
    print(f"AstroHybridLarge: {astro.parameter_count():,} params", flush=True)

    # CPU-offload layers AFTER extracting weights — frees ~2.75 GiB per GPU for backward.
    n_offload_per_gpu = 5  # 5 per GPU frees ~2.75 GiB/GPU — 4 was 16 MiB short
    print(f"CPU-offloading {n_offload_per_gpu} layers per GPU...", flush=True)
    from accelerate import cpu_offload_with_hook
    layer_devices = {li: str(next(model.model.layers[li].parameters()).device) for li in range(nl)}
    gpu0_layers = [li for li in range(nl) if 'cuda:0' in layer_devices[li]]
    gpu1_layers = [li for li in range(nl) if 'cuda:1' in layer_devices[li]]
    print(f"  GPU 0: {len(gpu0_layers)} layers ({gpu0_layers[0]}-{gpu0_layers[-1]})", flush=True)
    print(f"  GPU 1: {len(gpu1_layers)} layers ({gpu1_layers[0]}-{gpu1_layers[-1]})", flush=True)
    offload_hooks = []
    to_offload = gpu0_layers[:n_offload_per_gpu] + gpu1_layers[:n_offload_per_gpu]
    prev_hook = None
    for li in to_offload:
        exec_dev = torch.device(layer_devices[li])
        model.model.layers[li], hook = cpu_offload_with_hook(
            model.model.layers[li], execution_device=exec_dev, prev_module_hook=prev_hook)
        offload_hooks.append(hook)
        prev_hook = hook
    print(f"  Offloaded layers: {to_offload}", flush=True)
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3
        print(f"  After offload — cuda:{i}: {alloc:.1f} GiB alloc, ~{free:.1f} GiB free", flush=True)

    optimizer = torch.optim.AdamW(astro.parameters(), lr=args.lr, weight_decay=0.01)

    train_samples = generate_squad_dataset(n_samples=args.n_train, n_windows=5,
                                            vary_distance=True, seed=args.train_seed,
                                            split='train', include_answer=True)
    eval_samples = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                           vary_distance=True, seed=args.eval_seed, split='validation')
    print(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}", flush=True)

    print(f"\n{'='*50}\nEpoch 0 (before training)\n{'='*50}", flush=True)
    pure_acc, hybrid_acc = evaluate(model, tokenizer, astro, eval_samples,
                                     inject_layers, sense_layer, args.k_real, embed_device,
                                     max_length=args.max_length)
    print(f"  => pure300={pure_acc:.1%}  hybrid(16+{args.k_real})={hybrid_acc:.1%}", flush=True)

    results = [{'epoch': 0, 'pure': pure_acc, 'hybrid': hybrid_acc}]
    best_hybrid = hybrid_acc

    model_name = os.path.basename(args.model_path).replace('.', '_')
    unique_tag = f'_n{args.n_mem}_k{args.k_real}_t{args.n_train}_s{args.train_seed}'
    ckpt_path = f'checkpoints/astro_hybrid_{model_name}{unique_tag}.pt'

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}\nEpoch {epoch}\n{'='*50}", flush=True)
        train_loss = train_epoch(model, tokenizer, astro, optimizer, train_samples,
                                  inject_layers, sense_layer, args.k_real, embed_device,
                                  max_length=args.max_length)
        print(f"  train: loss={train_loss:.3f}", flush=True)
        pure_acc, hybrid_acc = evaluate(model, tokenizer, astro, eval_samples,
                                         inject_layers, sense_layer, args.k_real, embed_device,
                                         max_length=args.max_length)
        print(f"  eval: pure300={pure_acc:.1%}  hybrid(16+{args.k_real})={hybrid_acc:.1%}", flush=True)
        results.append({'epoch': epoch, 'pure': pure_acc, 'hybrid': hybrid_acc, 'train_loss': train_loss})

        if hybrid_acc > best_hybrid:
            best_hybrid = hybrid_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(astro.state_dict(), ckpt_path)
            print(f"  Saved to {ckpt_path}", flush=True)
            print(f"  NEW BEST: {hybrid_acc:.1%}", flush=True)

    print(f"\nBest hybrid: {best_hybrid:.1%} (pure300 baseline: {pure_acc:.1%})", flush=True)

    save_log = f'logs/results/hybrid_large_{model_name}{unique_tag}.json'
    os.makedirs(os.path.dirname(save_log), exist_ok=True)
    with open(save_log, 'w') as f:
        json.dump({'args': vars(args), 'results': results, 'best_hybrid': best_hybrid,
                   'checkpoint': ckpt_path}, f, indent=2)
    print(f"Log saved to {save_log}", flush=True)


if __name__ == '__main__':
    main()

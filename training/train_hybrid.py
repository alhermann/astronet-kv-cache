"""Direction 2: AstroNet hybrid KV cache — 16 learned + 284 selected real tokens.

AstroNet processes fact windows, accumulates Ca2+ memory state, and generates
16 KV pairs that capture cross-window context. These are prepended to 284
zero-shot selected real KV pairs for a total budget of 300.

Training signal: next-token cross-entropy on the query+answer window.
No task-specific labels — fully general.
"""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset


def _select_kv(all_kv, li, idx):
    """Select KV pairs by index, handling multi-GPU device placement."""
    K = torch.cat(all_kv[li][0], dim=2)
    V = torch.cat(all_kv[li][1], dim=2)
    li_idx = idx.to(K.device)
    return K[:, :, li_idx, :], V[:, :, li_idx, :]


class AstroHybrid(nn.Module):
    """AstroNet that produces KV pairs via the model's own projections.

    Sensing: cross-attention pooling over hidden states
    Memory: Ca2+ EMA dynamics
    Output: AstroNet produces "virtual hidden states" per layer, then the
            model's OWN k_proj/v_proj convert them to KV format.
            This guarantees the KV pairs are in the model's native format.
    """

    def __init__(self, hidden_dim, n_mem_tokens=16, attn_dim=256,
                 n_kv_heads=4, head_dim=128, n_layers=28, inject_layers=None,
                 model=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mem_tokens = n_mem_tokens
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.inject_layers = inject_layers or list(range(n_layers))

        # Will be populated by extract_model_weights()
        self._kv_weights = {}

        # === Sensing: cross-attention pooling ===
        self.queries = nn.Parameter(torch.randn(1, n_mem_tokens, attn_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attn_scale = 1.0 / math.sqrt(attn_dim)

        # === Ca2+ dynamics ===
        self.log_alpha = nn.Parameter(torch.tensor(-0.5))  # alpha ~ 0.38
        self.register_buffer('g', torch.zeros(1, n_mem_tokens, hidden_dim))
        # Memory bank mode: store per-window summaries, cross-attend at generation
        self.use_memory_bank = False
        self.memory_bank = []
        self._bank_attended = False
        # Gated mode: per-dimension forget/input gates (Ca2+ microdomain model)
        self.use_gated = False
        gate_bottleneck = 256
        self.gate_f_down = nn.Linear(hidden_dim * 2, gate_bottleneck, bias=False)
        self.gate_f_up = nn.Linear(gate_bottleneck, hidden_dim)
        self.gate_i_down = nn.Linear(hidden_dim * 2, gate_bottleneck, bias=False)
        self.gate_i_up = nn.Linear(gate_bottleneck, hidden_dim)
        # Forget gate bias +2: retain ~88% by default
        self.gate_f_up.bias.data.fill_(2.0)
        # Input gate bias -2: update ~12% by default
        self.gate_i_up.bias.data.fill_(-2.0)
        # Bank attention: cross-attend over stored summaries
        self.bank_queries = nn.Parameter(torch.randn(1, n_mem_tokens, attn_dim) * 0.02)
        self.bank_key_proj_out = nn.Linear(attn_dim, attn_dim, bias=False)

        # === Per-layer hidden state generators ===
        # Low-rank bottleneck: memory → 256 → hidden_dim per layer
        # Only for inject layers (4 layers), shared projection for the rest
        bottleneck = 256
        self.layer_down = nn.ModuleDict()
        self.layer_up = nn.ModuleDict()
        for li in self.inject_layers:
            self.layer_down[str(li)] = nn.Linear(hidden_dim, bottleneck, bias=False)
            self.layer_up[str(li)] = nn.Linear(bottleneck, hidden_dim, bias=False)
        # Shared for non-inject layers
        self.shared_down = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.shared_up = nn.Linear(bottleneck, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)
        for d in list(self.layer_down.values()) + [self.shared_down]:
            nn.init.xavier_uniform_(d.weight, gain=1.0)
        for u in list(self.layer_up.values()) + [self.shared_up]:
            nn.init.xavier_uniform_(u.weight, gain=0.5)

    def extract_model_weights(self, model, device):
        """Extract dequantized k_proj/v_proj weights from frozen 4-bit model.
        These are stored as frozen float32 buffers — no grad, no 4-bit issues."""
        print("  Extracting model KV projection weights...", flush=True)
        for li in range(self.n_layers):
            layer = model.model.layers[li].self_attn
            # Dequantize 4-bit weights to float32
            k_w = layer.k_proj.weight
            v_w = layer.v_proj.weight
            if hasattr(k_w, 'quant_state'):
                import bitsandbytes as bnb
                k_w = bnb.functional.dequantize_4bit(k_w.data, k_w.quant_state).float()
                v_w = bnb.functional.dequantize_4bit(v_w.data, v_w.quant_state).float()
            else:
                k_w = k_w.float()
                v_w = v_w.float()
            self._kv_weights[li] = {
                'k': k_w.to(device).detach(),
                'v': v_w.to(device).detach(),
            }
        print(f"  Extracted {len(self._kv_weights)} layers, "
              f"K shape: {self._kv_weights[0]['k'].shape}", flush=True)

    @property
    def alpha(self):
        return torch.sigmoid(self.log_alpha)

    @property
    def scale(self):
        return 1.0

    def sense(self, hidden_states):
        """Cross-attention pooling."""
        h = self.input_norm(hidden_states.float())
        B = h.shape[0]
        keys = self.key_proj(h)
        attn = torch.bmm(self.queries.expand(B, -1, -1),
                         keys.transpose(1, 2)) * self.attn_scale
        weights = torch.softmax(attn, dim=-1)
        return torch.bmm(weights, h)

    def update_state(self, sensed, keep_grad=False):
        """Ca2+ update: EMA, gated, or memory bank depending on mode."""
        if self.use_memory_bank:
            pooled = sensed.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
            self.memory_bank.append(pooled)
        elif self.use_gated:
            # Per-dimension gated update (Ca2+ microdomain model)
            g_exp = self.g.expand(sensed.shape[0], -1, -1)
            combined = torch.cat([g_exp, sensed], dim=-1)  # (B, K, 2*D)
            f = torch.sigmoid(self.gate_f_up(self.gate_f_down(combined)))
            i = torch.sigmoid(self.gate_i_up(self.gate_i_down(combined)))
            new_g = f * g_exp + i * sensed
            new_g = new_g.mean(dim=0, keepdim=True)
            self.g = new_g if keep_grad else new_g.detach()
        else:
            # Original EMA mode
            alpha = self.alpha
            g_exp = self.g.expand(sensed.shape[0], -1, -1)
            new_g = (1 - alpha) * g_exp + alpha * sensed
            new_g = new_g.mean(dim=0, keepdim=True)
            self.g = new_g if keep_grad else new_g.detach()

    def attend_bank(self):
        """Cross-attend over stored per-window summaries to produce memory state.
        Biologically: astrocyte integrates spatially distributed Ca2+ stores."""
        if not self.memory_bank:
            return
        # Stack bank: (1, T, hidden_dim) where T = number of windows
        bank = torch.cat(self.memory_bank, dim=1)  # (1, T, hidden_dim)
        # Project bank to attention key space
        bank_normed = self.input_norm(bank)
        bank_keys = self.bank_key_proj_out(self.key_proj(bank_normed))  # (1, T, attn_dim)
        # Cross-attend: bank_queries (1, K, attn_dim) @ bank_keys (1, T, attn_dim)
        attn = torch.bmm(self.bank_queries, bank_keys.transpose(1, 2)) * self.attn_scale
        weights = torch.softmax(attn, dim=-1)  # (1, K, T)
        # Weighted sum over bank values in hidden_dim space: (1, K, hidden_dim)
        self.g = torch.bmm(weights, bank.float())

    def generate_kv(self, layer_idx, real_kv_sample):
        """Generate K,V pair using the MODEL'S OWN k_proj/v_proj.

        In memory bank mode, first attends over stored summaries to produce
        the memory state g (called once, before the first layer).

        Args:
            layer_idx: which transformer layer
            real_kv_sample: (K, V) tuple for dtype/device reference
        Returns:
            mem_K: [1, n_kv_heads, n_mem_tokens, head_dim]
            mem_V: [1, n_kv_heads, n_mem_tokens, head_dim]
        """
        # In bank mode, attend over bank once (first generate_kv call per sample)
        if self.use_memory_bank and self.memory_bank and not self._bank_attended:
            self.attend_bank()
            self._bank_attended = True

        real_K, real_V = real_kv_sample

        # Transform memory state to this layer's hidden space via bottleneck
        li_str = str(layer_idx)
        if li_str in self.layer_down:
            virtual_hidden = self.layer_up[li_str](F.gelu(self.layer_down[li_str](self.g)))
        else:
            virtual_hidden = self.shared_up(F.gelu(self.shared_down(self.g)))

        # Clamp to prevent NaN from bitsandbytes dequantization
        virtual_hidden = virtual_hidden.clamp(-100, 100)

        # Use DEQUANTIZED copies of model's k_proj/v_proj (float32, gradient-safe)
        vh = virtual_hidden  # already float32 from AstroNet
        k_flat = F.linear(vh, self._kv_weights[layer_idx]['k'])  # [1, n_mem, n_kv * hd]
        v_flat = F.linear(vh, self._kv_weights[layer_idx]['v'])

        # Reshape to [1, n_kv_heads, n_mem_tokens, head_dim]
        k_out = k_flat.view(1, self.n_mem_tokens, self.n_kv_heads, self.head_dim)
        k_out = k_out.permute(0, 2, 1, 3)
        v_out = v_flat.view(1, self.n_mem_tokens, self.n_kv_heads, self.head_dim)
        v_out = v_out.permute(0, 2, 1, 3)

        return k_out.to(real_K.dtype), v_out.to(real_V.dtype)

    def reset_state(self):
        self.g = self.g.detach().zero_()
        self.memory_bank = []
        self._bank_attended = False

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AstroModulator(nn.Module):
    """Memory-modulated token selection (S2-mod).

    Instead of generating virtual KV pairs, the memory state modulates
    which real tokens are selected. This is biologically closer to
    astrocytic modulation of synaptic conductance.

    score_i = clamp(h_i) * clamp(c_i) * clamp(m_i)
    where m_i = f(memory_state, key_i) is the memory-derived relevance.

    All 300 tokens are real — memory only improves selection.
    """
    def __init__(self, hidden_dim, n_queries=16, attn_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        # Sensing: cross-attention pooling (same as AstroHybrid)
        self.queries = nn.Parameter(torch.randn(1, n_queries, attn_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.attn_scale = 1.0 / math.sqrt(attn_dim)

        # Ca2+ EMA dynamics
        self.log_alpha = nn.Parameter(torch.tensor(-0.5))  # alpha ~ 0.38
        self.register_buffer('g', torch.zeros(1, n_queries, hidden_dim))

        # Memory bank
        self.use_memory_bank = False
        self.memory_bank = []
        self._bank_attended = False
        self.bank_queries = nn.Parameter(torch.randn(1, n_queries, attn_dim) * 0.02)
        self.bank_key_proj = nn.Linear(attn_dim, attn_dim, bias=False)

        # Modulation: project memory state to scoring space
        # memory_state (hidden_dim) → attn_dim, then dot with cached keys
        self.mod_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        # Bias for neutral initialization: sigmoid(3) ≈ 0.95 (near-transparent)
        self.mod_bias = nn.Parameter(torch.tensor(3.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.mod_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.bank_key_proj.weight, gain=1.0)

    @property
    def alpha(self):
        return torch.sigmoid(self.log_alpha)

    def sense(self, hidden_states):
        """Cross-attention pooling over hidden states."""
        h = self.input_norm(hidden_states.float())
        B = h.shape[0]
        keys = self.key_proj(h)
        attn = torch.bmm(self.queries.expand(B, -1, -1),
                         keys.transpose(1, 2)) * self.attn_scale
        weights = torch.softmax(attn, dim=-1)
        return torch.bmm(weights, h)

    def update_state(self, sensed, keep_grad=False):
        if self.use_memory_bank:
            pooled = sensed.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
            self.memory_bank.append(pooled)
        else:
            alpha = self.alpha
            g_exp = self.g.expand(sensed.shape[0], -1, -1)
            new_g = (1 - alpha) * g_exp + alpha * sensed
            new_g = new_g.mean(dim=0, keepdim=True)
            self.g = new_g if keep_grad else new_g.detach()

    def attend_bank(self):
        if not self.memory_bank:
            return
        bank = torch.cat(self.memory_bank, dim=1)
        bank_normed = self.input_norm(bank)
        bank_keys = self.bank_key_proj(self.key_proj(bank_normed))
        attn = torch.bmm(self.bank_queries, bank_keys.transpose(1, 2)) * self.attn_scale
        weights = torch.softmax(attn, dim=-1)
        self.g = torch.bmm(weights, bank.float())

    def compute_memory_scores(self, all_hidden, device):
        """Compute memory-derived relevance scores for each cached token.

        Uses stored per-token hidden states (from sense layer) and the
        accumulated memory state to score each token's relevance.

        Args:
            all_hidden: list of (seq_len, hidden_dim) tensors per window
            device: target device
        Returns:
            tensor of shape (total_tokens,) with memory modulation scores.
        """
        if self.use_memory_bank and self.memory_bank and not self._bank_attended:
            self.attend_bank()
            self._bank_attended = True

        # Pool memory state to single vector: (1, attn_dim)
        g_pooled = self.g.mean(dim=1)  # (1, hidden_dim)
        g_key = self.mod_proj(g_pooled)  # (1, attn_dim)

        # Concatenate all hidden states and project to attn_dim
        H = torch.cat(all_hidden, dim=0).float().to(device)  # (total, hidden_dim)
        H_normed = self.input_norm(H)
        H_proj = self.key_proj(H_normed)  # (total, attn_dim)

        # Memory relevance: dot product + bias for near-neutral init
        mem_scores = (H_proj @ g_key.squeeze(0)) / math.sqrt(self.attn_dim) + self.mod_bias

        return mem_scores

    def reset_state(self):
        self.g = self.g.detach().zero_()
        self.memory_bank = []
        self._bank_attended = False

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, tokenizer, astro, optimizer, samples,
                inject_layers, sense_layer, k_real, device):
    """Train one epoch. Loss = perplexity on query+answer with hybrid cache."""
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

        # Process fact windows
        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384,
                            truncation=True).to(device)
            with torch.no_grad():
                out = model(input_ids=ids['input_ids'], use_cache=True,
                            output_attentions=True, output_hidden_states=True)

            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=device)
            for la in out.attentions:
                a = la[0].to(device)
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:4] = -1e9
            all_attn.append(imp)

            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0].detach())
                all_kv[li][1].append(out.past_key_values[li][1].detach())

            # AstroNet senses
            hidden = out.hidden_states[sense_layer].detach()
            sensed = astro.sense(hidden)
            # For long-context training, allow gradient from more windows
            # Default: last 2 windows for 5-window, but scale with n_windows
            n_grad_windows = max(2, len(s.windows) // 2)
            astro.update_state(sensed, keep_grad=(wi >= len(s.windows) - 1 - n_grad_windows))

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn).detach()

        # Zero-shot multiplicative scoring for real token selection
        q_text = f'Question: {s.question}\nAnswer:'
        q_ids = tokenizer(q_text, return_tensors='pt', max_length=128, truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)

        cross = torch.zeros(total, device=device)
        for li in inject_layers:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                  K[hi].float().T) / math.sqrt(hd)
                # Position-robust formulation: softmax over cache positions
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)

        # Avg-pool smoothing
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(
                cross.unsqueeze(0).unsqueeze(0),
                kernel_size=5, padding=2, stride=1,
            ).squeeze()

        cross[:4] = -1e9

        # Recent-window keep + sink + cross top-k (S1 selection)
        # Mask the entire last (query) window from cross-scoring; keep n_recent
        # tokens at its start as an observation-window analogue.
        n_sink = 4
        last_window_size = all_kv[inject_layers[0]][0][-1].shape[2]
        last_start = total - last_window_size
        n_recent = min(int(k_real * 0.2), last_window_size)
        recent = torch.arange(last_start, last_start + n_recent, device=device)
        scores = cross.clone()
        scores[last_start:total] = -1e9
        n_select = max(k_real - n_sink - len(recent), 0)
        n_avail = (scores > -1e8).sum().item()
        n_select = min(n_select, n_avail, scores.shape[0])
        _, top = scores.topk(n_select) if n_select > 0 else (None, torch.empty(0, dtype=torch.long, device=device))
        sink_idx = torch.arange(n_sink, device=device)
        idx = torch.cat([sink_idx, top, recent]).unique().sort().values
        idx = idx[:k_real]

        # Build hybrid cache: [AstroNet KV | Real selected KV]
        n_mem = astro.n_mem_tokens
        cache = DynamicCache()
        for li in range(nl):
            K_real, V_real = _select_kv(all_kv, li, idx)

            # AstroNet generates memory KV (gradient flows here!)
            K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
            K_mem, V_mem = K_mem.to(K_real.device), V_mem.to(V_real.device)

            # Prepend memory tokens: [mem | real]
            K_combined = torch.cat([K_mem, K_real], dim=2)
            V_combined = torch.cat([V_mem, V_real], dim=2)
            cache.update(K_combined, V_combined, li)

        # Forward pass with hybrid cache — teacher forcing
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer: {s.answer}'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        input_ids = fq['input_ids']
        total_cache_len = n_mem + len(idx)
        pos = torch.arange(total_cache_len, total_cache_len + input_ids.shape[1],
                           device=device).unsqueeze(0)

        # Forward WITH gradients through cache (AstroNet KV has grad)
        out = model(input_ids=input_ids, past_key_values=cache, position_ids=pos)

        # Loss on ALL tokens (general perplexity, not task-specific)
        logits = out.logits[0, :-1]
        targets = input_ids[0, 1:]
        loss = F.cross_entropy(logits.float(), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(astro.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_total += 1

        if (si + 1) % 10 == 0:
            print(f'  train {si+1}/{len(samples)}: loss={total_loss/n_total:.3f} '
                  f'alpha={astro.alpha.item():.3f}',
                  flush=True)

    return total_loss / max(n_total, 1)


def evaluate(model, tokenizer, astro, samples, inject_layers, sense_layer,
             k_real, device):
    """Evaluate: compare hybrid (16 mem + 284 real) vs pure (300 real)."""
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

        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384,
                            truncation=True).to(device)
            with torch.no_grad():
                out = model(input_ids=ids['input_ids'], use_cache=True,
                            output_attentions=True, output_hidden_states=True)
            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=device)
            for la in out.attentions:
                a = la[0].to(device)
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:4] = -1e9
            all_attn.append(imp)
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])
            with torch.no_grad():
                hidden = out.hidden_states[sense_layer]
                sensed = astro.sense(hidden)
                astro.update_state(sensed)

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        # Zero-shot multiplicative
        q_ids = tokenizer(f'Question: {s.question}\nAnswer:', return_tensors='pt',
                          max_length=128, truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
        cross = torch.zeros(total, device=device)
        for li in inject_layers:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                  K[hi].float().T) / math.sqrt(hd)
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)

        # Avg-pool smoothing
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(
                cross.unsqueeze(0).unsqueeze(0),
                kernel_size=5, padding=2, stride=1,
            ).squeeze()
        cross[:4] = -1e9

        # === Pure: 300 real tokens (position-robust S1: cross + smoothing + recent) ===
        k_pure = min(300, total)
        n_sink_p = 4
        last_window_size = all_kv[inject_layers[0]][0][-1].shape[2]
        last_start = total - last_window_size
        n_recent_p = min(int(k_pure * 0.2), last_window_size)
        recent_p = torch.arange(last_start, last_start + n_recent_p, device=device)
        scores_p = cross.clone()
        scores_p[last_start:total] = -1e9
        n_select_p = max(k_pure - n_sink_p - len(recent_p), 0)
        n_avail_p = (scores_p > -1e8).sum().item()
        n_select_p = min(n_select_p, n_avail_p, scores_p.shape[0])
        if n_select_p > 0:
            _, top_p = scores_p.topk(n_select_p)
        else:
            top_p = torch.empty(0, dtype=torch.long, device=device)
        sink_idx_p = torch.arange(n_sink_p, device=device)
        idx_pure = torch.cat([sink_idx_p, top_p, recent_p]).unique().sort().values
        idx_pure = idx_pure[:k_pure]
        k_pure = len(idx_pure)
        cache_pure = DynamicCache()
        for li in range(nl):
            K_p, V_p = _select_kv(all_kv, li, idx_pure)
            cache_pure.update(K_p, V_p, li)
        query_pure = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq_p = tokenizer(query_pure, return_tensors='pt', max_length=384, truncation=True).to(device)
        pos_p = torch.arange(k_pure, k_pure + fq_p['input_ids'].shape[1], device=device).unsqueeze(0)
        cur_p = fq_p['input_ids']; cc_p = cache_pure; gen_p = []
        with torch.no_grad():
            for _ in range(20):
                o_p = model(input_ids=cur_p, past_key_values=cc_p, position_ids=pos_p)
                cc_p = o_p.past_key_values
                nxt_p = o_p.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen_p.append(nxt_p[0, 0].item()); cur_p = nxt_p
                pos_p = torch.tensor([[k_pure + fq_p['input_ids'].shape[1] + len(gen_p) - 1]], device=device)
                if nxt_p[0, 0].item() == tokenizer.eos_token_id: break
        text_pure = tokenizer.decode(gen_p, skip_special_tokens=True).strip()
        if s.answer.lower() in text_pure.lower():
            correct_pure += 1

        # === Hybrid: 16 mem + 284 real (position-robust S1) ===
        k_hyb_target = min(k_real, total)
        n_sink_h = 4
        n_recent_h = min(int(k_hyb_target * 0.2), last_window_size)
        recent_h = torch.arange(last_start, last_start + n_recent_h, device=device)
        scores_h = cross.clone()
        scores_h[last_start:total] = -1e9
        n_select_h = max(k_hyb_target - n_sink_h - len(recent_h), 0)
        n_avail_h = (scores_h > -1e8).sum().item()
        n_select_h = min(n_select_h, n_avail_h, scores_h.shape[0])
        if n_select_h > 0:
            _, top_h = scores_h.topk(n_select_h)
        else:
            top_h = torch.empty(0, dtype=torch.long, device=device)
        sink_idx_h = torch.arange(n_sink_h, device=device)
        idx_hyb = torch.cat([sink_idx_h, top_h, recent_h]).unique().sort().values
        idx_hyb = idx_hyb[:k_hyb_target]
        k_hyb = len(idx_hyb)

        cache_hyb = DynamicCache()
        with torch.no_grad():
            for li in range(nl):
                K_real, V_real = _select_kv(all_kv, li, idx_hyb)
                K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
                K_mem, V_mem = K_mem.to(K_real.device), V_mem.to(V_real.device)
                cache_hyb.update(torch.cat([K_mem, K_real], dim=2),
                                 torch.cat([V_mem, V_real], dim=2), li)

        total_len = n_mem + k_hyb
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        pos = torch.arange(total_len, total_len + fq['input_ids'].shape[1],
                           device=device).unsqueeze(0)
        cur = fq['input_ids']; cc = cache_hyb; gen = []
        with torch.no_grad():
            for _ in range(20):
                o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
                cc = o.past_key_values
                nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen.append(nxt[0, 0].item()); cur = nxt
                pos = torch.tensor([[total_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                                   device=device)
                if nxt[0, 0].item() == tokenizer.eos_token_id:
                    break
        text_hyb = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if s.answer.lower() in text_hyb.lower():
            correct_hybrid += 1

        if (si + 1) % 25 == 0:
            print(f'  eval {si+1}/{len(samples)}: pure300={correct_pure}/{si+1}  '
                  f'hybrid={correct_hybrid}/{si+1}', flush=True)

    n = len(samples)
    return correct_pure / n, correct_hybrid / n


def _generate(model, tokenizer, all_kv, idx, k, n_mem, nl, device):
    """Generate with pure selected cache (no AstroNet)."""
    cache = DynamicCache()
    for li in range(nl):
        K, V = _select_kv(all_kv, li, idx)
        cache.update(K, V, li)

    query = 'Based on what you read earlier, answer the following question.\nQuestion: '
    # We need the actual question but don't have it here — use a placeholder approach
    # Actually this function is called from evaluate which has access to s.question
    # Let me fix this by passing the full query text
    return ""  # placeholder — will be handled in evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./models/qwen2.5-7b')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_eval', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--k_real', type=int, default=284, help='Real tokens (300 - 16 mem)')
    parser.add_argument('--n_mem', type=int, default=16)
    parser.add_argument('--sense_layer', type=int, default=14)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--diverse_training', type=str, default=None,
                        help='Diverse training sources, e.g. "squad+hotpotqa" (default: SQuAD only)')
    parser.add_argument('--n_windows', type=int, default=5, help='Windows per training sample')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Resume training from existing checkpoint')
    parser.add_argument('--alpha_init', type=float, default=None,
                        help='Override initial alpha (default: architecture default 0.38)')
    parser.add_argument('--memory_bank', action='store_true',
                        help='Use memory bank instead of EMA (per-window summaries + cross-attention)')
    parser.add_argument('--gated', action='store_true',
                        help='Use gated Ca2+ update (per-dimension forget/input gates)')
    parser.add_argument('--attn_dim', type=int, default=256, help='Bottleneck/attention dimension')
    args = parser.parse_args()

    print(f"Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type='nf4')
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    # Override device to embed device for multi-GPU compatibility
    args.device = str(model.get_input_embeddings().weight.device)

    hidden_dim = model.config.hidden_size
    nl = model.config.num_hidden_layers
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    print(f"Model: {nl} layers, {hidden_dim} hidden, {nkv} KV heads, {hd} head_dim", flush=True)

    astro = AstroHybrid(
        hidden_dim=hidden_dim,
        n_mem_tokens=args.n_mem,
        attn_dim=args.attn_dim,
        n_kv_heads=nkv,
        head_dim=hd,
        n_layers=nl,
        inject_layers=inject_layers,
    ).to(args.device)
    astro.extract_model_weights(model, args.device)
    if args.memory_bank:
        astro.use_memory_bank = True
        print("Memory bank mode ENABLED (per-window summaries + cross-attention)", flush=True)
    if args.gated:
        astro.use_gated = True
        print("Gated Ca2+ mode ENABLED (per-dimension forget/input gates)", flush=True)
    if args.resume_checkpoint:
        astro.load_state_dict(torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False), strict=False)
        print(f"Resumed from {args.resume_checkpoint}", flush=True)
    if args.alpha_init is not None:
        import math
        with torch.no_grad():
            # alpha = sigmoid(log_alpha), so log_alpha = logit(alpha)
            logit_val = math.log(args.alpha_init / (1 - args.alpha_init))
            astro.log_alpha.fill_(logit_val)
        print(f"Alpha initialized to {astro.alpha.item():.4f} (log_alpha={logit_val:.3f})", flush=True)
    print(f"AstroHybrid: {astro.parameter_count():,} params", flush=True)

    optimizer = torch.optim.AdamW(astro.parameters(), lr=args.lr, weight_decay=0.01)

    if args.diverse_training == 'longctx':
        from data.longctx_qa import generate_longctx_dataset
        # Support variable windows: --n_windows 12 or --diverse_training longctx with default variable
        win_arg = 'variable' if args.n_windows == 0 else args.n_windows
        train_samples = generate_longctx_dataset(tokenizer, n_samples=args.n_train,
                                                  n_windows=win_arg, chunk_size=384,
                                                  seed=args.train_seed, split='train',
                                                  include_answer=True)
    elif args.diverse_training:
        from data.diverse_qa import generate_diverse_dataset
        train_samples = generate_diverse_dataset(n_samples=args.n_train, n_windows=args.n_windows,
                                                  seed=args.train_seed, include_answer=True,
                                                  sources=args.diverse_training)
    else:
        train_samples = generate_squad_dataset(n_samples=args.n_train, n_windows=args.n_windows,
                                                vary_distance=True, seed=args.train_seed,
                                                split='train', include_answer=True)
    eval_samples = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                           vary_distance=True, seed=args.eval_seed, split='validation')
    print(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}", flush=True)

    # Fix the _generate function for pure eval
    def generate_pure(all_kv_local, idx_local, k_local, question):
        cache = DynamicCache()
        for li in range(nl):
            K = torch.cat(all_kv_local[li][0], dim=2)[:, :, idx_local, :]
            V = torch.cat(all_kv_local[li][1], dim=2)[:, :, idx_local, :]
            cache.update(K, V, li)
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(args.device)
        pos = torch.arange(k_local, k_local + fq['input_ids'].shape[1],
                           device=args.device).unsqueeze(0)
        cur = fq['input_ids']; cc = cache; gen = []
        with torch.no_grad():
            for _ in range(20):
                o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
                cc = o.past_key_values
                nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen.append(nxt[0, 0].item()); cur = nxt
                pos = torch.tensor([[k_local + fq['input_ids'].shape[1] + len(gen) - 1]],
                                   device=args.device)
                if nxt[0, 0].item() == tokenizer.eos_token_id:
                    break
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    # Monkey-patch the evaluate pure generation
    import types
    # Actually, let me just inline the eval properly

    # Initial eval
    print(f"\n{'='*50}\nEpoch 0 (before training)\n{'='*50}", flush=True)
    pure_acc, hybrid_acc = evaluate(model, tokenizer, astro, eval_samples,
                                     inject_layers, args.sense_layer, args.k_real, args.device)
    print(f"  => pure300={pure_acc:.1%}  hybrid({args.n_mem}+{args.k_real})={hybrid_acc:.1%}", flush=True)

    results = [{'epoch': 0, 'pure': pure_acc, 'hybrid': hybrid_acc}]
    best_hybrid = hybrid_acc

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}\nEpoch {epoch}\n{'='*50}", flush=True)
        t0 = time.time()
        train_loss = train_epoch(model, tokenizer, astro, optimizer, train_samples,
                                  inject_layers, args.sense_layer, args.k_real, args.device)
        print(f"  train: loss={train_loss:.3f} ({time.time()-t0:.0f}s)", flush=True)

        pure_acc, hybrid_acc = evaluate(model, tokenizer, astro, eval_samples,
                                         inject_layers, args.sense_layer, args.k_real, args.device)
        print(f"  eval: pure300={pure_acc:.1%}  hybrid({args.n_mem}+{args.k_real})={hybrid_acc:.1%}", flush=True)

        results.append({'epoch': epoch, 'train_loss': train_loss,
                         'pure': pure_acc, 'hybrid': hybrid_acc})

        if hybrid_acc > best_hybrid:
            best_hybrid = hybrid_acc
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_name = os.path.basename(args.model_path).replace('.', '_')
            diverse_tag = f'_diverse' if args.diverse_training else ''
            windows_tag = f'_w{args.n_windows}' if args.n_windows != 5 else ''
            resume_tag = '_ft' if args.resume_checkpoint else ''
            alpha_tag = f'_a{args.alpha_init:.2f}' if args.alpha_init is not None else ''
            bank_tag = '_bank' if args.memory_bank else ''
            gated_tag = '_gated' if args.gated else ''
            unique_tag = f'_n{args.n_mem}_k{args.k_real}_t{args.n_train}_s{args.train_seed}{windows_tag}{diverse_tag}{alpha_tag}{bank_tag}{gated_tag}{resume_tag}'
            ckpt_path = f'checkpoints/astro_hybrid_{ckpt_name}{unique_tag}.pt'
            torch.save(astro.state_dict(), ckpt_path)
            print(f"  Saved to {ckpt_path}", flush=True)
            print(f"  NEW BEST: {best_hybrid:.1%}", flush=True)

    save_data = {'args': vars(args), 'results': results, 'best_hybrid': best_hybrid,
                  'params': astro.parameter_count()}
    os.makedirs('logs', exist_ok=True)
    with open('logs/astro_hybrid_training.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nBest hybrid: {best_hybrid:.1%} (pure300 baseline: {results[0]['pure']:.1%})")


if __name__ == '__main__':
    main()

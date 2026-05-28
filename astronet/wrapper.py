"""Deployable AstroNet wrapper for frozen LLMs.

Minimal interface for using a trained AstroHybrid checkpoint at inference time.

Example:
    >>> from astronet.wrapper import AstroNetWrapper
    >>> wrapper = AstroNetWrapper.from_pretrained(
    ...     'Qwen/Qwen2.5-7B', astro_ckpt='checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt'
    ... )
    >>> answer = wrapper.answer(windows=[passage_1, passage_2, ...], question='...', k=300)
"""
import sys, os, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache


_SENSE_LAYER_BY_LAYERS = {28: 14, 32: 16, 40: 20, 48: 24, 64: 32}
_ATTN_DIM_BY_HIDDEN = {3584: 512, 4096: 256, 5120: 256}  # Qwen 7B uses larger attn_dim


class AstroNetWrapper:
    """Drop-in wrapper: load a frozen LLM + AstroNet hybrid checkpoint, answer multi-window questions.

    Public methods:
        from_pretrained(model_path_or_hf_id, astro_ckpt=...)  - factory
        answer(windows, question, k=300)                       - run hybrid pipeline
        answer_with_method(windows, question, method, k=300)   - explicit method choice

    Methods supported: 'hybrid' (S1+S2, default), 'mult' (S1-only), 'snapkv', 'h2o', 'streaming'.
    """

    def __init__(self, model, tokenizer, astro=None, sense_layer=None, n_mem=16, attn_dim=256):
        self.model = model
        self.tokenizer = tokenizer
        self.astro = astro
        self.n_mem = n_mem
        self.attn_dim = attn_dim
        self.device = model.get_input_embeddings().weight.device
        nl = model.config.num_hidden_layers
        self.sense_layer = sense_layer or _SENSE_LAYER_BY_LAYERS.get(nl, nl // 2)
        self.inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    @classmethod
    def from_pretrained(cls, model_path_or_hf_id, astro_ckpt=None, device_map=None,
                        quantization=True, attn_dim=None):
        """Load backbone + optional AstroNet hybrid checkpoint.

        Args:
            model_path_or_hf_id: local path or HuggingFace model ID
            astro_ckpt: path to AstroHybrid checkpoint (.pt); if None, returns wrapper without S2
            device_map: device_map for from_pretrained (default: {'':'cuda:0'} or 'auto' if multi-GPU)
            quantization: enable 4-bit nf4 quantization (default True)
            attn_dim: override attn_dim (default auto from hidden_size)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = dict(torch_dtype=torch.float16)
        if quantization:
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16)
        # Auto-detect when the model is too large for a single 24 GiB GPU
        # and fall back to multi-GPU device_map. Heuristic: if the user passed
        # an explicit device_map, honour it; otherwise estimate NF4 footprint
        # from config and switch to 'auto' when above ~22 GiB.
        if device_map is None:
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_path_or_hf_id)
                # Rough NF4 param-byte estimate (4 bits/param + scales)
                nparams_b = (cfg.num_hidden_layers
                              * (4 * cfg.hidden_size * cfg.hidden_size  # qkvo
                                 + 3 * cfg.hidden_size * getattr(cfg, 'intermediate_size', 4 * cfg.hidden_size)))
                nf4_gib = nparams_b * 0.5 / (1024 ** 3)  # 0.5 bytes/param at nf4
                device_map = 'auto' if nf4_gib > 22 else {'': 'cuda:0'}
            except Exception:
                device_map = {'': 'cuda:0'}
        load_kwargs['device_map'] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_path_or_hf_id, **load_kwargs)
        model.eval()

        astro = None
        if astro_ckpt:
            # Late import to avoid pulling training deps in inference-only use
            from training.train_hybrid import AstroHybrid
            nl = model.config.num_hidden_layers
            hidden_dim = model.config.hidden_size
            nkv = model.config.num_key_value_heads
            hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
            ad = attn_dim or _ATTN_DIM_BY_HIDDEN.get(hidden_dim, 256)
            astro = AstroHybrid(
                hidden_dim=hidden_dim, n_mem_tokens=16, attn_dim=ad,
                n_kv_heads=nkv, head_dim=hd, n_layers=nl,
                inject_layers=[nl // 4, nl // 2, 3 * nl // 4, nl - 2],
            ).to(model.get_input_embeddings().weight.device)
            astro.extract_model_weights(model, model.get_input_embeddings().weight.device)
            astro.load_state_dict(torch.load(astro_ckpt, map_location=astro.g.device), strict=False)
            astro.eval()

        return cls(model, tokenizer, astro=astro,
                   attn_dim=attn_dim or _ATTN_DIM_BY_HIDDEN.get(model.config.hidden_size, 256))

    def _process_windows(self, windows, max_window_len=384):
        """Run windows through the model, gather KV cache + sensed state.

        Returns:
            all_kv: {layer_idx: (list_of_K, list_of_V)}
            total: total cached tokens
            last_start: position where the last window begins
        """
        nl = self.model.config.num_hidden_layers
        all_kv = {li: ([], []) for li in range(nl)}
        boundaries = []
        offset = 0
        if self.astro is not None:
            self.astro.reset_state()
        for w in windows:
            ids = self.tokenizer(w, return_tensors='pt', max_length=max_window_len,
                                 truncation=True).to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=ids['input_ids'], use_cache=True,
                                 output_hidden_states=(self.astro is not None))
            sl = ids['input_ids'].shape[1]
            boundaries.append((offset, offset + sl))
            offset += sl
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])
            if self.astro is not None:
                with torch.no_grad():
                    hidden = out.hidden_states[self.sense_layer]
                    sensed = self.astro.sense(hidden)
                    self.astro.update_state(sensed)
        return all_kv, offset, boundaries[-1][0]

    def _compute_cross(self, all_kv, total, question):
        """Cross-window multiplicative attention scoring (the patched S1).

        Llama/Mistral q_proj is a bitsandbytes Linear4bit that expects fp16 input.
        Calling .float() before q_proj triggered 'expected Float but found Half'
        on those backbones (Qwen tolerates either). Keep input in fp16 through
        the projection; cast to fp32 only for the matmul.
        """
        nq = self.model.config.num_attention_heads
        nkv = self.model.config.num_key_value_heads
        hd = getattr(self.model.config, 'head_dim', self.model.config.hidden_size // nq)
        qpk = nq // nkv
        q_ids = self.tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                                max_length=128, truncation=True).to(self.device)
        with torch.no_grad():
            q_out = self.model(input_ids=q_ids['input_ids'], output_hidden_states=True)
        cross = torch.zeros(total, device=self.device)
        for li in self.inject_layers:
            layer_dev = all_kv[li][0][0].device
            # Keep hidden state in its native dtype (fp16) through q_proj; cast to fp32 after.
            hidden_in = q_out.hidden_states[li][0].to(layer_dev)  # fp16 from quantised model
            Q = self.model.model.layers[li].self_attn.q_proj(hidden_in).float().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :],
                                  K[hi].float().T) / math.sqrt(hd)
                cross += torch.softmax(sc, dim=-1).sum(dim=(0, 1)).to(self.device)
        if total > 5:
            cross = F.avg_pool1d(cross.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1).squeeze()
        cross[:4] = -1e9
        return cross

    def _select_kv(self, all_kv, idx, li):
        K = torch.cat(all_kv[li][0], dim=2)
        V = torch.cat(all_kv[li][1], dim=2)
        return K[:, :, idx.to(K.device), :], V[:, :, idx.to(V.device), :]

    def answer(self, windows, question, k=300, method='hybrid', max_new_tokens=64):
        """Answer `question` using `windows` of context with the chosen method.

        Args:
            windows: list of text chunks (typically ~384 tokens each)
            question: the question to answer
            k: total KV budget (default 300)
            method: 'hybrid' | 'mult' | 'snapkv' | 'h2o' | 'streaming'
            max_new_tokens: generation length

        Returns: generated answer string.
        """
        if method == 'hybrid' and self.astro is None:
            raise ValueError("method='hybrid' requires AstroNet checkpoint; load with astro_ckpt=...")
        nl = self.model.config.num_hidden_layers
        all_kv, total, last_start = self._process_windows(windows)
        last_window_size = all_kv[self.inject_layers[0]][0][-1].shape[2]
        n_sink = 4

        if method in ('hybrid', 'mult', 'snapkv'):
            cross = self._compute_cross(all_kv, total, question)
            k_real = k - (self.n_mem if method == 'hybrid' else 0)
            n_recent = min(int(k_real * 0.2), last_window_size)
            recent_idx = torch.arange(last_start, last_start + n_recent, device=self.device)
            scores = cross.clone()
            scores[last_start:total] = -1e9
            n_select = max(k_real - n_sink - n_recent, 0)
            n_avail = (scores > -1e8).sum().item()
            n_select = min(n_select, n_avail, scores.shape[0])
            _, top = scores.topk(n_select) if n_select > 0 else (None, torch.empty(0, dtype=torch.long, device=self.device))
            sink_idx = torch.arange(n_sink, device=self.device)
            idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values[:k_real]
        elif method == 'streaming':
            sink = torch.arange(n_sink, device=self.device)
            recent = torch.arange(total - (k - n_sink), total, device=self.device)
            idx = torch.cat([sink, recent])
        elif method == 'h2o':
            # Equal-attention heuristic (deprecated baseline; included for compatibility)
            raise NotImplementedError("h2o requires attention sums; use main eval_needle.py for h2o.")
        else:
            raise ValueError(f"unknown method: {method}")

        cache = DynamicCache()
        for li in range(nl):
            K_real, V_real = self._select_kv(all_kv, idx, li)
            if method == 'hybrid':
                K_mem, V_mem = self.astro.generate_kv(li, (K_real.to(self.device), V_real.to(self.device)))
                cache.update(torch.cat([K_mem.to(K_real.device), K_real], dim=2),
                             torch.cat([V_mem.to(V_real.device), V_real], dim=2), li)
            else:
                cache.update(K_real, V_real, li)
        prefix_len = (self.n_mem if method == 'hybrid' else 0) + len(idx)

        # Generate
        q = f'Based on what you read earlier, answer the following question.\nQuestion: {question}\nAnswer:'
        fq = self.tokenizer(q, return_tensors='pt', max_length=256, truncation=True).to(self.device)
        pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1],
                           device=self.device).unsqueeze(0)
        cur, cc, gen = fq['input_ids'], cache, []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                o = self.model(input_ids=cur, past_key_values=cc, position_ids=pos)
                cc = o.past_key_values
                nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen.append(nxt[0, 0].item())
                cur = nxt
                pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                                   device=self.device)
                if nxt[0, 0].item() == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def cache_bytes(self, k=300, dtype='fp16'):
        """Analytical KV-cache size in bytes for budget `k`."""
        a = self.model.config
        nl = a.num_hidden_layers
        nkv = a.num_key_value_heads
        hd = getattr(a, 'head_dim', a.hidden_size // a.num_attention_heads)
        per_token_per_layer = {
            'fp16': 2 * nkv * hd * 2,
            'k8v4': (1 + 0.5) * nkv * hd,
            'k4v4': (0.5 + 0.5) * nkv * hd,
        }[dtype]
        return k * nl * per_token_per_layer

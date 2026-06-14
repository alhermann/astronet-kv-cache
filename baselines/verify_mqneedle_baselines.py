"""Sanity-check that SnapKV and SnapKV-oracle in multiquery NIAH are
faithfully implemented. We:
  1. Build one multi-needle haystack and tokenize it.
  2. Identify the exact token positions where each needle's distinctive
     keyword lands.
  3. Run SnapKV's per-layer selection at compress_once.
  4. For each layer, report (a) what fraction of selected k positions are
     in the last OBS tokens, (b) whether any needle keyword positions are
     among the selected indices.
  5. Repeat with SnapKV oracle per-question; expect needle positions to be
     in the union of selected indices.

If SnapKV's selected positions are dominated by the obs-window tail and
miss needle positions, the 11% accuracy is a structural feature of the
algorithm, not a bug in our eval.
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from baselines.streaming_kv_core import AttentionCapture
from baselines.eval_faithful_streaming_needle import (
    score_per_layer_snapkv_layerwise, select_topk_global, OBS_WINDOW)
from baselines.eval_needle import NEEDLES
from baselines.eval_multiquery_needle import build_multineedle_haystack


def find_keyword_positions(tokenizer, ctx_text, keyword, device):
    """Return token positions where `keyword` appears (as a substring of
    the joined decoded tokens). Approximation but good enough for sanity."""
    ids = tokenizer(ctx_text, return_tensors='pt',
                     return_offsets_mapping=True, max_length=131072,
                     truncation=True).to(device)
    offsets = ids['offset_mapping'][0].tolist()
    starts = []
    pos = 0
    while True:
        idx = ctx_text.find(keyword, pos)
        if idx < 0: break
        starts.append(idx); pos = idx + len(keyword)
    out = []
    for s in starts:
        for ti, (a, b) in enumerate(offsets):
            if a <= s < b:
                out.append(ti); break
    return out, ids['input_ids']


def main():
    model_path = './models/llama-3.1-8b'
    device = 'cuda:1'
    n_windows, n_needles, k = 50, 4, 300
    seed = 42

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb,
        device_map={'': device}, torch_dtype=torch.float16)
    model.eval()
    n_layers = model.config.num_hidden_layers
    capture = AttentionCapture(model)

    windows, qa = build_multineedle_haystack(n_windows, n_needles, seed=seed)
    ctx_text = '\n\n'.join(windows) + '\n\n'

    # Find needle keyword positions for each of the 4 needles.
    print('Needles in this haystack:')
    keyword_map = {
        'ALPHA-7829':     'ALPHA-7829',
        'March 15th':     'March 15th',
        'Henderson':      'Henderson',
        '4,231,567':      '4,231,567',
        'blue serum':     'blue serum',
    }
    needle_positions = []
    full_ids = None
    for needle_text, q, a in NEEDLES:
        # Pick a distinctive keyword from the answer.
        kw = a.split()[0]
        if kw not in keyword_map: kw = a
        if kw not in ctx_text: continue
        positions, full_ids = find_keyword_positions(tokenizer, ctx_text, kw, device)
        if positions:
            print(f'  "{kw}" → token positions {positions}')
            needle_positions.append((kw, positions))

    ctx_len = full_ids.shape[1]
    obs = min(OBS_WINDOW, ctx_len - 1)
    print(f'\nctx_len={ctx_len}, obs_window=last {obs} tokens, k={k}')
    print(f'Tail region (obs window) = positions [{ctx_len-obs}, {ctx_len})\n')

    # --- SnapKV query-agnostic selection ---
    capture.clear(); capture.enabled = False
    out1 = model(input_ids=full_ids[:, :-obs], past_key_values=DynamicCache(),
                  use_cache=True)
    capture.enabled = True
    out2 = model(input_ids=full_ids[:, -obs:],
                  past_key_values=out1.past_key_values, use_cache=True)
    ctx_cache = out2.past_key_values

    scores = score_per_layer_snapkv_layerwise(
        capture, ctx_cache, ctx_len, obs, n_layers)
    per_layer_idx = [select_topk_global(s, k) for s in scores]

    # Aggregate selection: a token is "selected by SnapKV" if any layer picks it.
    union_snapkv = set()
    for i in per_layer_idx:
        union_snapkv.update(i.tolist())
    in_tail = sum(1 for p in union_snapkv if p >= ctx_len - obs)
    print(f'SnapKV (query-agnostic) — union across {n_layers} layers:')
    print(f'  |union|        = {len(union_snapkv)}')
    print(f'  in tail/obs    = {in_tail} ({100*in_tail/max(1,len(union_snapkv)):.1f}%)')
    for kw, positions in needle_positions:
        hits = sum(1 for p in positions if p in union_snapkv)
        print(f'  "{kw}" hit     = {hits}/{len(positions)}')

    # --- SnapKV oracle for one question (the first needle's question) ---
    if qa:
        q_text = f'Question: {qa[0][0]}\nAnswer:'
        q_ids = tokenizer(q_text, return_tensors='pt',
                          add_special_tokens=False).input_ids.to(device)
        q_pos = torch.arange(ctx_len, ctx_len + q_ids.shape[1],
                              device=device).unsqueeze(0)
        cloned = DynamicCache()
        for li in range(n_layers):
            K, V = ctx_cache[li]
            cloned.update(K.clone(), V.clone(), li)
        capture.clear(); capture.enabled = True
        out_q = model(input_ids=q_ids, past_key_values=cloned,
                       position_ids=q_pos, use_cache=True)
        full_q = out_q.past_key_values
        scores_o = score_per_layer_snapkv_layerwise(
            capture, full_q, ctx_len, q_ids.shape[1], n_layers)
        per_layer_idx_o = [select_topk_global(s, k) for s in scores_o]
        union_oracle = set()
        for i in per_layer_idx_o:
            union_oracle.update(i.tolist())
        in_tail_o = sum(1 for p in union_oracle if p >= ctx_len - obs)
        print(f'\nSnapKV ORACLE (Q-aware) for "{qa[0][0][:60]}...":')
        print(f'  |union|        = {len(union_oracle)}')
        print(f'  in tail/obs    = {in_tail_o} ({100*in_tail_o/max(1,len(union_oracle)):.1f}%)')
        for kw, positions in needle_positions:
            hits = sum(1 for p in positions if p in union_oracle)
            print(f'  "{kw}" hit     = {hits}/{len(positions)}')


if __name__ == '__main__':
    main()

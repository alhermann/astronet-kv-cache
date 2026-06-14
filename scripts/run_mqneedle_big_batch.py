"""Batched MQ-NIAH runner for 32B/24B backbones.

Loads the model ONCE and runs all anchor + λ-sweep cells in a single
process. Saves results to the same paths as the per-cell runner so the
4-model aggregator picks them up automatically.

Why: a 32B 4-bit load reads 17 shards (~20 min cold cache). The original
per-cell queue ran 11 separate Python invocations = ~3.7 h per model
just to load shards. Batching reduces that to a single ~20 min load.

Usage:
    python scripts/run_mqneedle_big_batch.py --model qwen32b   --multi_gpu
    python scripts/run_mqneedle_big_batch.py --model mistral24b --multi_gpu
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from baselines.streaming_kv_core import AttentionCapture, SenseCapture
from baselines.eval_multiquery_needle import (
    build_multineedle_haystack, compress_once, answer_question)
from baselines.eval_multiquery_squad import _find_answer
from astronet.astro_gate import AstroGate

MODELS = {
    'qwen32b':    ('./models/qwen2.5-32b',      'checkpoints/astro_gate_qwen32b.pt'),
    'mistral24b': ('./models/mistral-small-24b', 'checkpoints/astro_gate_mistral24b.pt'),
}

LAMBDA_SWEEP = [0.13, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]

# Cells to run: (method, n_windows, n_needles, k, seed, n_trials, tag, lam_or_None)
ANCHOR_CELLS = [
    ('full',          100, 4, 300, 42, 20, 'anchor', None),
    ('snapkv_oracle', 100, 4, 300, 42, 20, 'anchor', None),
    ('snapkv',        100, 4, 300, 42, 20, 'anchor', None),
    ('astrogate',     100, 4, 300, 42, 20, 'anchor', 0.5),
]


def _save_path(model_tag, method, w, n, k, seed, t, tag):
    return f'logs/results/mqneedle_{method}_{model_tag}_w{w}_n{n}_k{k}_s{seed}_t{t}_{tag}.json'


def _set_lambda(astro, lam):
    astro.log_lambda.data.fill_(math.log(math.expm1(lam)))


@torch.no_grad()
def run_cell(model, tokenizer, capture, sense_cap, astro, method,
             windows_per_trial_iter, k, n_layers, device, n_trials,
             n_windows, n_needles, seed_base):
    correct, total, n_err = 0, 0, 0
    for t in range(n_trials):
        try:
            windows, qa = build_multineedle_haystack(
                n_windows, n_needles, seed=seed_base * 1000 + t)
            compressed, full_layers, ctx_len = compress_once(
                model, tokenizer, capture, sense_cap, astro, method,
                windows, k, 16, n_layers, device)
            for q, a in qa:
                gen = answer_question(model, tokenizer, capture, method,
                                       compressed, full_layers, ctx_len, q,
                                       k, 16, n_layers, device)
                if _find_answer(gen, a): correct += 1
                total += 1
        except Exception as e:
            n_err += 1
            total += n_needles
            print(f'  trial {t} ERR: {type(e).__name__}: {str(e)[:400]}',
                   flush=True)
            if t == 0:
                import traceback
                traceback.print_exc()
    return correct, total, n_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(MODELS.keys()))
    p.add_argument('--multi_gpu', action='store_true')
    args = p.parse_args()

    model_path, ckpt_path = MODELS[args.model]
    if not os.path.exists(ckpt_path):
        print(f'Missing {ckpt_path}'); return

    print(f'[batch] model={args.model} multi_gpu={args.multi_gpu}', flush=True)
    print(f'[batch] loading model from {model_path}...', flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    if args.multi_gpu:
        from transformers import AutoConfig as _AC
        _cfg = _AC.from_pretrained(model_path)
        nL = _cfg.num_hidden_layers
        half = nL // 2
        device_map = {'model.embed_tokens': 0, 'model.norm': 1,
                      'model.rotary_emb': 0, 'lm_head': 1}
        for i in range(nL):
            device_map[f'model.layers.{i}'] = 0 if i < half else 1
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb,
            device_map=device_map, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb,
            device_map={'': 'cuda:0'}, torch_dtype=torch.float16)
    model.eval()
    print(f'[batch] model loaded ({time.time()-t0:.0f}s)', flush=True)

    input_device = str(model.get_input_embeddings().weight.device)
    capture = AttentionCapture(model)
    cfg = model.config
    n_layers = cfg.num_hidden_layers

    # Load AstroGate once. Critical: place astro on the device where the
    # sense layer lives — in multi-GPU split, the sense layer (=n_layers//2)
    # is on cuda:1, but the embedding is on cuda:0. Hidden states captured
    # at the sense layer are on cuda:1, so astro must be there too or the
    # gate_logits path hits a device mismatch.
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_cfg = raw.get('config', {}) if isinstance(raw, dict) else {}
    sl = raw.get('sense_layer', n_layers // 2) if isinstance(raw, dict) else n_layers // 2
    sense_dev = next(model.model.layers[sl].parameters()).device
    astro = AstroGate(
        hidden_dim=cfg.hidden_size,
        n_astro=saved_cfg.get('n_astro', 16),
        attn_dim=saved_cfg.get('attn_dim', 256),
        n_patterns=saved_cfg.get('n_patterns', 8),
    ).to(sense_dev)
    astro.load_state_dict(
        raw['astro'] if isinstance(raw, dict) and 'astro' in raw else raw,
        strict=False)
    astro.eval()
    sense_cap = SenseCapture(model, sl)
    device = input_device   # input ids still go to the embed device
    print(f'[batch] astro loaded on {sense_dev}; sense_layer={sl}; '
          f'input_device={input_device}', flush=True)

    cells = list(ANCHOR_CELLS)
    for lam_sweep in LAMBDA_SWEEP:
        if lam_sweep == 0.5: continue   # anchor cell already covers it
        ltag = f'lam{str(lam_sweep).replace(".","p")}'
        cells.append(('astrogate', 100, 4, 300, 42, 20, ltag, lam_sweep))

    cell_t0 = time.time()
    for ci, (method, nw, nn, k, seed, nt, tag, lam) in enumerate(cells):
        save_path = _save_path(args.model, method, nw, nn, k, seed, nt, tag)
        if os.path.exists(save_path):
            print(f'  [{ci+1}/{len(cells)}] SKIP {save_path}', flush=True)
            continue
        if method == 'astrogate' and lam is not None:
            _set_lambda(astro, lam)
            print(f'  [{ci+1}/{len(cells)}] {method} tag={tag} λ={lam}',
                   flush=True)
        else:
            print(f'  [{ci+1}/{len(cells)}] {method} tag={tag}', flush=True)

        ct0 = time.time()
        correct, total, n_err = run_cell(
            model, tokenizer, capture, sense_cap,
            astro if method in ('astrogate',) else None,
            method, None, k, n_layers, device, nt, nw, nn, seed)
        acc = correct / max(1, total)
        print(f'    ACC: {correct}/{total} = {acc*100:.2f}%  errors={n_err}  '
              f'({time.time()-ct0:.0f}s)', flush=True)

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({
                'protocol': 'multiquery_needle',
                'method': method,
                'model': os.path.basename(model_path),
                'checkpoint': ckpt_path if method == 'astrogate' else None,
                'ckpt_tag': tag,
                'k': k, 'n_windows': nw, 'n_needles': nn, 'n_trials': nt,
                'n_questions': total, 'seed': seed,
                'correct': correct, 'accuracy': acc, 'n_trial_errors': n_err,
                'lambda_used': lam if method == 'astrogate' else None,
            }, f, indent=2)
    print(f'[batch] all cells done ({time.time()-cell_t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()

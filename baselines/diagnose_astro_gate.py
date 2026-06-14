"""Diagnostic: does the trained AstroGate predict the query-aware oracle?

For each held-out multi-query SQuAD paragraph:
  - Compute oracle target (per-token relevance from query-aware SnapKV).
  - Compute astro gate_logits from ctx-body alone (no question).
  - Report: ROC-AUC, top-k precision, and Spearman ρ between gate and target.

Random-init AstroGate should be at AUC ≈ 0.5; a trained one should clearly
exceed that. This is the cheap signal before the full multiquery eval.
"""
from __future__ import annotations
import argparse, os, sys, json, statistics
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from baselines.streaming_kv_core import AttentionCapture, SenseCapture
from baselines.eval_faithful_streaming_needle import (
    score_per_layer_snapkv_layerwise, OBS_WINDOW)
from astronet.astro_gate import AstroGate
from training.train_astro_gate import (
    build_multiquery_samples_split, compute_oracle_target, _select_topk_idx)


def roc_auc(scores, labels):
    n_pos = int(labels.sum().item())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = scores.argsort(descending=True)
    rank = labels[order]
    tp = torch.cumsum(rank, dim=0).float()
    fp = torch.cumsum(1.0 - rank, dim=0)
    # Trapezoidal AUC ≈ Σ TPR · ΔFPR using ranks.
    auc = (tp * (1 - rank)).sum() / (n_pos * n_neg)
    return float(auc.item())


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = (ra.norm() * rb.norm()).item() + 1e-9
    return float((ra * rb).sum().item() / den)


@torch.no_grad()
def diagnose(model, tokenizer, capture, sense_cap, astro,
             samples, k, sense_layer, n_layers, device):
    aucs, prec_k, sprs, gate_means, tgt_pos = [], [], [], [], []
    for si, s in enumerate(samples):
        ctx_text = '\n\n'.join(s['windows']) + '\n\n'
        ctx_ids = tokenizer(ctx_text, return_tensors='pt',
                             max_length=8192, truncation=True
                             ).input_ids.to(device)
        ctx_len = ctx_ids.shape[1]
        obs = min(OBS_WINDOW, ctx_len - 1)
        capture.clear(); capture.enabled = False
        sense_cap.clear()
        h_chunks = []
        out1 = model(input_ids=ctx_ids[:, :-obs], past_key_values=DynamicCache(),
                     use_cache=True)
        h_chunks.append(sense_cap.hidden.detach()); sense_cap.clear()
        capture.enabled = True
        out2 = model(input_ids=ctx_ids[:, -obs:],
                     past_key_values=out1.past_key_values, use_cache=True)
        h_chunks.append(sense_cap.hidden.detach())
        ctx_cache = out2.past_key_values
        ctx_hidden = torch.cat(h_chunks, dim=1)
        astro.reset_state()
        astro.update_state(astro.sense(ctx_hidden))

        targets = []
        for q, _ in s['qa']:
            tgt = compute_oracle_target(model, tokenizer, capture, ctx_cache,
                                         ctx_len, q, k, n_layers, device)
            targets.append(tgt)
        target = torch.stack(targets, dim=0).mean(dim=0)
        gate = astro.gate_logits(ctx_hidden).to(target.device)

        # ROC-AUC against binarized target (any layer picked token).
        bin_tgt = (target > 0).float()
        aucs.append(roc_auc(gate.float(), bin_tgt))
        sprs.append(spearman(gate.float().cpu(), target.float().cpu()))

        # Top-k precision: among gate-top-k positions, how many appear in
        # the union of layer top-k (target > 0)?
        idx = gate.topk(min(k, gate.numel())).indices
        prec_k.append(float(bin_tgt[idx].mean().item()))
        gate_means.append(gate.mean().item())
        tgt_pos.append(int(bin_tgt.sum().item()))
    return {
        'n': len(samples),
        'mean_auc': statistics.mean(aucs),
        'mean_precision_at_k': statistics.mean(prec_k),
        'mean_spearman': statistics.mean(sprs),
        'mean_gate_value': statistics.mean(gate_means),
        'mean_oracle_positives': statistics.mean(tgt_pos),
        'per_sample': {'auc': aucs, 'prec_k': prec_k, 'spearman': sprs},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_eval', type=int, default=20)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--save_path', default=None)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)
    capture = AttentionCapture(model)
    n_layers = model.config.num_hidden_layers

    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = raw.get('config', {})
    astro = AstroGate(
        hidden_dim=model.config.hidden_size,
        n_astro=cfg.get('n_astro', 16),
        attn_dim=cfg.get('attn_dim', 256),
        n_patterns=cfg.get('n_patterns', 8),
    ).to(device)
    astro.load_state_dict(raw['astro'], strict=False)
    astro.eval()
    sense_layer = raw.get('sense_layer', n_layers // 2)
    sense_cap = SenseCapture(model, sense_layer)
    print(f'Loaded {args.checkpoint}: λ={astro.lam.item():.3f} '
          f'αf={astro.alpha_fast.item():.3f} αs={astro.alpha_slow.item():.3f}',
          flush=True)

    samples = build_multiquery_samples_split(
        'validation', args.n_eval, 4, 4, args.seed)
    out = diagnose(model, tokenizer, capture, sense_cap, astro,
                   samples, args.k, sense_layer, n_layers, device)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_sample'},
                     indent=2), flush=True)
    if args.save_path:
        with open(args.save_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Saved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()

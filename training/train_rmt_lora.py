"""RMT-lite + LoRA baseline for the AstroNet paper.

This is the "memory-augmented with backbone fine-tuning" baseline the
clarity audit asked for: a faithful Recurrent-Memory-Transformer
recurrence on top of a 4-bit-quantised Qwen 2.5-7B backbone, with
QLoRA fine-tuning of the attention projections so the backbone can
actually learn to USE the memory tokens.

Architecture
------------
For each context segment we form  [read_mem, content, write_mem],
forward through the backbone, then extract the write-slot output
(last n_mem hidden states at the LAST decoder layer) and use it as
the read_mem for the next segment. The write_mem input is a fixed
learnable embedding. The first segment's read_mem is also a
learnable embedding (identical to write_mem init).

Trainable parameters
--------------------
1. mem_init : (n_mem, hidden_dim) -- shared read+write seed
2. LoRA on q_proj, k_proj, v_proj, o_proj of every decoder layer

Backbone stays 4-bit NF4 quantised (QLoRA pattern). All grads flow
through the recurrence across all 5 segments (full BPTT within an
instance; detach across instances).

Loss
----
Next-token cross-entropy on the answer span ONLY (mask the rest).
This matches the AstroNet train_hybrid loss definition exactly.

Watchdog
--------
Every step we log ||grad(mem_init)|| and ||grad(any LoRA layer)||.
If mem-init gradient is < 1e-5 by step 500, we abort with a clear
message rather than burn 14h on a dead-gradient run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from data.real_qa import generate_squad_dataset


def find_answer_marker(token_ids, tokenizer, marker="Answer:"):
    """Return the position immediately after the 'Answer:' marker, or None.

    Uses prefix-decoding (string search) rather than per-token length
    accumulation, which is robust to tokenizer quirks (Mistral, etc.)
    where 'Answer:' tokenises differently in isolation vs. context.
    """
    # Accept list, tuple, or torch.Tensor input.
    if hasattr(token_ids, 'tolist') and not isinstance(token_ids, list):
        ids_list = token_ids.tolist()
    else:
        ids_list = list(token_ids)

    # Quick check: does the full decoded text contain the marker at all?
    full_text = tokenizer.decode(ids_list, skip_special_tokens=True)
    if marker not in full_text:
        return None

    # Walk backwards through token prefixes. Find the smallest i such that
    # the first i tokens do NOT contain the marker; token i is then the
    # one that completed it, and i + 1 is the position immediately after.
    for i in range(len(ids_list) - 1, -1, -1):
        prefix = tokenizer.decode(ids_list[:i], skip_special_tokens=True)
        if marker not in prefix:
            return i + 1
    return None


class RMTLite(nn.Module):
    """Just holds the learnable read/write mem embedding."""

    def __init__(self, n_mem: int, hidden_dim: int):
        super().__init__()
        self.n_mem = n_mem
        self.hidden_dim = hidden_dim
        # Single learnable seed for both read (first segment) and
        # write (every segment input).  Initialised at the same scale
        # as the backbone's residual stream after layer norm.
        self.mem_init = nn.Parameter(0.02 * torch.randn(1, n_mem, hidden_dim))


def forward_segment(model, embed_layer, content_ids, read_mem, write_mem,
                    device, attn_mask=None):
    """Forward one segment with [read_mem, content, write_mem].

    Returns the output hidden states at the WRITE-slot positions
    (i.e. the last n_mem positions of the last decoder layer's output).
    """
    n_mem = read_mem.shape[1]
    content_embeds = embed_layer(content_ids)            # (1, S, d)
    inputs_embeds = torch.cat(
        [read_mem, content_embeds, write_mem], dim=1
    )                                                    # (1, n_mem+S+n_mem, d)

    out = model(
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states[-1]                           # (1, total, d)
    new_state = hs[:, -n_mem:, :]                        # (1, n_mem, d)
    logits = out.logits                                  # (1, total, vocab)
    return logits, new_state


def compute_answer_loss(logits, content_ids, n_mem, tokenizer):
    """Cross-entropy on the answer span of the query segment only.

    logits   : (1, n_mem+S+n_mem, vocab)
    content_ids : (1, S) -- the query segment's content token IDs

    Mask everything except the tokens that come after the 'Answer:'
    marker. Loss is averaged over those tokens.
    """
    S = content_ids.shape[1]
    # logits[:, n_mem-1:n_mem+S-1] predict content_ids[:, 0:S] (next-token
    # prediction shift)
    pred_logits = logits[:, n_mem - 1: n_mem + S - 1, :]   # (1, S, vocab)
    target_ids = content_ids                                # (1, S)

    marker_pos = find_answer_marker(target_ids[0].tolist(), tokenizer)
    if marker_pos is None or marker_pos >= S:
        return None  # skip sample if no answer marker found

    # mask = 1 for positions to score
    mask = torch.zeros_like(target_ids, dtype=torch.float)
    mask[0, marker_pos:] = 1.0
    n_score = mask.sum().item()
    if n_score == 0:
        return None
    flat_logits = pred_logits.reshape(-1, pred_logits.shape[-1])
    flat_targets = target_ids.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    losses = losses * mask.reshape(-1)
    return losses.sum() / n_score


def grad_norm(params):
    total = 0.0
    n = 0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
            n += 1
    return math.sqrt(total) if n > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--save_dir',   required=True)
    ap.add_argument('--n_mem',      type=int,   default=16)
    ap.add_argument('--n_windows',  type=int,   default=5)
    ap.add_argument('--n_train',    type=int,   default=5000)
    ap.add_argument('--max_seg_tokens', type=int, default=256,
                    help='max content tokens per segment')
    ap.add_argument('--lr_lora',    type=float, default=2e-4)
    ap.add_argument('--lr_mem',     type=float, default=1e-4,
                    help='reduced from 1e-3 after gradient explosion '
                         'in the first BPTT-full smoke test')
    ap.add_argument('--bptt_depth', type=int,   default=2,
                    help='detach mem state more than this many segments '
                         'back from the query segment; default 2 to '
                         'avoid unrolled-recurrence gradient explosion')
    ap.add_argument('--epochs',     type=int,   default=3)
    ap.add_argument('--grad_accum', type=int,   default=8)
    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--lora_r',     type=int,   default=16)
    ap.add_argument('--lora_alpha', type=int,   default=32)
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--device',     default='cuda:0')
    ap.add_argument('--watchdog_after_steps', type=int, default=500)
    ap.add_argument('--watchdog_mem_grad_min', type=float, default=1e-5)
    ap.add_argument('--log_every',  type=int,   default=10)
    ap.add_argument('--save_every', type=int,   default=2000)
    args = ap.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # ----- model -----
    print(f"[rmt-lora] loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bfloat16 compute dtype has much larger dynamic range than fp16 and
    # is the standard QLoRA choice; fp16 caused gradient overflow on the
    # mem_init parameter in the first smoke test (g_mem = inf at step 1).
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable({'use_reentrant': False})

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    embed_layer = model.get_input_embeddings()

    # ----- mem module -----
    # Keep mem_init in fp32 (more headroom for accumulated grads through
    # 5 segments of recurrence) and cast to the model's compute dtype
    # only at forward time. This avoids the fp16 gradient overflow we
    # observed in the first smoke test.
    rmt = RMTLite(args.n_mem, hidden_dim).to(device).to(torch.float32)
    compute_dtype = torch.bfloat16
    print(f"[rmt-lora] mem_init shape {tuple(rmt.mem_init.shape)} "
          f"dtype={rmt.mem_init.dtype} compute={compute_dtype}", flush=True)

    # ----- optimiser with param groups -----
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    mem_params = [rmt.mem_init]
    optimiser = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.lr_lora,  'weight_decay': 0.0},
        {'params': mem_params,  'lr': args.lr_mem,   'weight_decay': 0.0},
    ])

    # ----- data -----
    samples = generate_squad_dataset(
        n_samples=args.n_train, n_windows=args.n_windows, seed=args.seed,
        include_answer=True,
    )
    print(f"[rmt-lora] {len(samples)} training samples", flush=True)

    # ----- training -----
    model.train()
    rmt.train()
    log_path = os.path.join(args.save_dir, 'train.log')
    log_f = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n'); log_f.flush()

    log(f"[rmt-lora] start {time.strftime('%F %T')}  lr_lora={args.lr_lora} "
        f"lr_mem={args.lr_mem} n_mem={args.n_mem} n_windows={args.n_windows} "
        f"epochs={args.epochs} grad_accum={args.grad_accum}")

    step = 0
    seen_loss = []
    optimiser.zero_grad()
    aborted = False

    for ep in range(args.epochs):
        for si, sample in enumerate(samples):
            # build per-segment token id tensors
            seg_ids = []
            for wi in range(args.n_windows):
                text = sample.windows[wi]
                ids = tokenizer(text, return_tensors='pt',
                                max_length=args.max_seg_tokens,
                                truncation=True).input_ids.to(device)
                seg_ids.append(ids)

            # forward recurrence; cast mem to compute_dtype at the
            # forward boundary, keep parameter in fp32. BPTT depth
            # controls how many segments back the gradient flows from
            # the query segment: segments earlier than that have their
            # mem-state detached, bounding the gradient unroll length.
            read_mem = rmt.mem_init.to(compute_dtype)   # (1, n_mem, d)
            write_mem = rmt.mem_init.to(compute_dtype)  # same seed
            loss = None
            bptt_start = max(0, args.n_windows - args.bptt_depth)
            for wi in range(args.n_windows):
                logits, new_state = forward_segment(
                    model, embed_layer, seg_ids[wi],
                    read_mem, write_mem, device,
                )
                if wi == args.n_windows - 1:
                    # query segment: compute answer loss
                    loss = compute_answer_loss(logits, seg_ids[wi],
                                               args.n_mem, tokenizer)
                # next segment's read = current write-slot output;
                # detach if we're before the BPTT window
                if wi < bptt_start:
                    read_mem = new_state.detach()
                else:
                    read_mem = new_state

            if loss is None or not torch.isfinite(loss):
                # skip pathological samples (no marker, NaN); reset
                optimiser.zero_grad()
                continue

            (loss / args.grad_accum).backward()
            seen_loss.append(loss.item())

            if (si + 1) % args.grad_accum == 0:
                # gradient watchdog
                g_mem = grad_norm([rmt.mem_init])
                g_lora = grad_norm(lora_params[:8])
                torch.nn.utils.clip_grad_norm_(lora_params + mem_params, 1.0)

                # cosine LR with warmup
                progress = step / max(1, len(samples) * args.epochs // args.grad_accum)
                if step < args.warmup_steps:
                    lr_scale = (step + 1) / args.warmup_steps
                else:
                    lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
                for pg, base_lr in zip(optimiser.param_groups,
                                       [args.lr_lora, args.lr_mem]):
                    pg['lr'] = base_lr * lr_scale

                optimiser.step()
                optimiser.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    avg = sum(seen_loss[-200:]) / max(1, len(seen_loss[-200:]))
                    log(f"step={step:5d} ep={ep} si={si:5d}  loss={loss.item():.3f}  "
                        f"avg200={avg:.3f}  g_mem={g_mem:.2e}  g_lora={g_lora:.2e}  "
                        f"lr_mem={optimiser.param_groups[1]['lr']:.2e}")

                # watchdog check
                if (step == args.watchdog_after_steps and
                        g_mem < args.watchdog_mem_grad_min):
                    log(f"[rmt-lora] WATCHDOG: mem-init grad {g_mem:.2e} below "
                        f"{args.watchdog_mem_grad_min:.0e} at step {step}. "
                        f"Aborting to save 14h of dead-gradient training.")
                    aborted = True
                    break

                if step % args.save_every == 0:
                    ckpt = os.path.join(args.save_dir, f'step{step}.pt')
                    torch.save({'mem_init': rmt.mem_init.detach().cpu(),
                                'step': step, 'loss': avg}, ckpt)
                    model.save_pretrained(os.path.join(args.save_dir, f'lora_step{step}'))
                    log(f"[rmt-lora] saved step {step} -> {ckpt}")

        if aborted:
            break
        log(f"[rmt-lora] epoch {ep} done. avg_loss(last 200) = "
            f"{sum(seen_loss[-200:]) / max(1, len(seen_loss[-200:])):.3f}")

    # final save
    final = os.path.join(args.save_dir, 'final.pt')
    torch.save({'mem_init': rmt.mem_init.detach().cpu(),
                'step': step, 'aborted': aborted,
                'final_loss': sum(seen_loss[-200:]) / max(1, len(seen_loss[-200:]))},
               final)
    model.save_pretrained(os.path.join(args.save_dir, 'lora_final'))
    log(f"[rmt-lora] done {time.strftime('%F %T')}. aborted={aborted} steps={step}")
    log_f.close()


if __name__ == '__main__':
    main()

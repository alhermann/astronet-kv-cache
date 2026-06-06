"""Soft-prompt baseline — defuses the strongest reviewer attack against Stage 2.

Reviewer attack: "Stage 2's 16 virtual KV tokens are just learned soft prompts
in KV space.  Soft prompting is well-known.  What's new?"

Defense: we train a *matched* soft-prompt baseline and show that AstroHybrid
beats it on the same multi-segment SQuAD task.

What this script does:
  1. Loads the same 4-bit quantised backbone used elsewhere.
  2. Defines a trainable `nn.Parameter` of shape (n_prompt, hidden_dim).
  3. Concatenates the soft-prompt embeddings IN FRONT of the token
     embeddings of each input segment (and of the query).  This is the
     canonical "prefix tuning" / "P-tuning" / "gist" formulation.
  4. Trains on the same 5000-sample SQuAD corpus used for Stage 2, same
     optimiser, same n_epochs.  Backbone frozen throughout.
  5. Saves the trained soft prompt for eval.

Two operating points are reported in the paper:
  --n_prompt 16    : token-count-matched comparison (16 prompt tokens vs
                     Stage 2's 16 virtual KV tokens).  ~57K trainable
                     params for Qwen 7B at hidden_dim=3584.
  --n_prompt 3900  : parameter-count-matched comparison (14M trainable
                     params, matching Stage 2's auxiliary module).  This
                     is large enough that the soft prompt occupies more
                     of the input than any real practitioner would.

Eval is via `training/eval_softprompt.py` which reuses
`generate_squad_dataset` and the same position-robust protocol as
`tab:squad_main`.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig)


class SoftPrompt(nn.Module):
    """Trainable prefix-tuning style soft prompt.

    Forward signature mirrors what callers expect from a tiny adaptor:
    given a tensor of token embeddings (B, T, H), it returns a tensor
    of shape (B, n_prompt + T, H) with the soft-prompt embeddings
    concatenated in front.
    """

    def __init__(self, n_prompt: int, hidden_dim: int):
        super().__init__()
        # Init scale matches typical input embedding norm (Llama/Qwen
        # embeddings have norms in 0.5--2.0 range).  Larger init makes
        # the prompt visible at the start of training; smaller would
        # make it an identity until trained.
        self.prompt = nn.Parameter(
            torch.randn(n_prompt, hidden_dim) * 0.5)
        self.n_prompt = n_prompt
        self.hidden_dim = hidden_dim

    def prepend(self, token_embeds: torch.Tensor) -> torch.Tensor:
        """token_embeds: (B, T, H) -> (B, n_prompt + T, H)."""
        B = token_embeds.shape[0]
        prompt = self.prompt.unsqueeze(0).expand(B, -1, -1).to(token_embeds.dtype)
        return torch.cat([prompt, token_embeds], dim=1)

    def parameter_count(self) -> int:
        return self.prompt.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_prompt', type=int, default=16,
                   help='Number of prompt tokens.  16 = token-count-matched '
                        'to Stage 2; ~3900 = param-count-matched on Qwen 7B.')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Soft prompts typically need a larger LR than '
                        'Stage 2 because we are only training one matrix.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[softprompt] model={args.model_path}  n_prompt={args.n_prompt}  '
          f'lr={args.lr}  epochs={args.epochs}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad = False

    hidden_dim = model.config.hidden_size
    sp = SoftPrompt(args.n_prompt, hidden_dim).to(args.device).float()
    print(f'[softprompt] trainable: {sp.parameter_count():,} params', flush=True)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.real_qa import generate_squad_dataset
    samples = generate_squad_dataset(
        n_samples=args.n_train, n_windows=5, vary_distance=True,
        seed=42, split='train')
    print(f'[softprompt] generated {len(samples)} training samples', flush=True)

    optimizer = torch.optim.AdamW([sp.prompt], lr=args.lr, weight_decay=0.01)

    embed = model.get_input_embeddings()
    total_loss = 0.0
    n_step = 0
    for epoch in range(args.epochs):
        sp.train()
        t0 = time.time()
        for si, s in enumerate(samples):
            # Build the prompt: all context segments concatenated, then
            # the query.  Same prompt structure as the AstroHybrid eval
            # uses; soft prompt prepended in front of EVERYTHING.
            full_text = '\n\n'.join(s.windows[:-1]) + '\n\n' + s.windows[-1]
            ids = tokenizer(full_text, return_tensors='pt', max_length=2048,
                              truncation=True).to(args.device)
            # Append the answer for teacher-forced LM loss.
            ans_ids = tokenizer(' ' + s.answer, return_tensors='pt',
                                  add_special_tokens=False).to(args.device)
            full_ids = torch.cat([ids['input_ids'], ans_ids['input_ids']], dim=1)

            with torch.no_grad():
                tok_embeds = embed(full_ids).float()
            inputs_embeds = sp.prepend(tok_embeds)
            # The model expects half-precision embeddings under bnb.
            out = model(inputs_embeds=inputs_embeds.half(), use_cache=False)
            logits = out.logits.float()
            # NLL on the answer span only (last ans_len tokens of the
            # post-prompt portion).  Loss positions = logits aligned
            # to predict each answer token from the one before.
            ans_len = ans_ids['input_ids'].shape[1]
            target = full_ids[0, -ans_len:]
            pred = logits[0, -ans_len-1:-1]
            loss = F.cross_entropy(pred, target)
            (loss / 4).backward()
            total_loss += loss.item(); n_step += 1
            if (si + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_([sp.prompt], 1.0)
                optimizer.step(); optimizer.zero_grad()
            if (si + 1) % 50 == 0:
                print(f'  train {si+1}/{len(samples)}: '
                       f'mean loss = {total_loss / n_step:.3f}', flush=True)
        if n_step:
            optimizer.step(); optimizer.zero_grad()
        print(f'epoch {epoch+1}/{args.epochs}: mean loss = '
               f'{total_loss / max(1, n_step):.4f} ({time.time() - t0:.0f}s)',
              flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    torch.save({
        'prompt': sp.prompt.detach().cpu(),
        'n_prompt': args.n_prompt,
        'hidden_dim': hidden_dim,
        'args': vars(args),
        'param_count': sp.parameter_count(),
    }, args.save_path)
    print(f'[softprompt] saved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()

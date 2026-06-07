"""P2 (multi-position) retraining variant.

Wraps train_hybrid.py's main loop but RE-SHUFFLES THE FACT POSITION of
each training sample randomly to {0, 1, 2, 3} per epoch.  The existing
training script always places the fact at position 0 (windows[0]),
which the position-robust eval shows manifests as Stage 2 helping less
at the harder fact-at-end positions.

This is a small but principled fix: train and eval distributions match.

Usage:
    python training/train_hybrid_multipos.py \
        --model_path ./models/qwen2.5-7b \
        --n_train 5000 --n_eval 100 --epochs 2 \
        --n_mem 16 --attn_dim 256 \
        --device cuda:0
        # save_path is auto-generated to include _multipos tag

We delegate to train_hybrid.py for the heavy lifting --- only the dataset
construction differs.  This keeps the model + training-loop code paths
identical and ensures any improvement is attributable to the data
distribution change, not to a parallel code change.
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reshuffle_dataset(samples, rng):
    """Reassign each sample's fact_window uniformly at random across the
    non-query candidate windows.  Returns a NEW list of samples; original
    list is unchanged."""
    from training.eval_hybrid_position_robust import shuffle_fact_position
    from data.synthetic import CrossContextSample
    out = []
    for s in samples:
        n_candidates = len(s.windows) - 1  # exclude query window
        position = rng.randint(0, max(0, n_candidates - 1))
        moved = shuffle_fact_position([s], position)[0]
        out.append(moved)
    return out


def main():
    # First parse our own args; then forward to train_hybrid main.
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--k_real', type=int, default=284)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--sense_layer', type=int, default=14)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--train_seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_windows', type=int, default=5)
    args = p.parse_args()

    # Generate the SQuAD training samples, then re-shuffle their fact
    # positions uniformly at random.  Use a fixed RNG seeded on the
    # train_seed so the shuffle is reproducible.
    from data.real_qa import generate_squad_dataset
    base_train = generate_squad_dataset(
        n_samples=args.n_train, n_windows=args.n_windows,
        vary_distance=True, seed=args.train_seed, split='train')
    rng = random.Random(args.train_seed + 1)
    shuffled = reshuffle_dataset(base_train, rng)
    # Sanity check distribution.
    from collections import Counter
    pos_counts = Counter(s.fact_window for s in shuffled)
    print(f'[multipos] fact-position distribution after shuffle: '
          f'{dict(sorted(pos_counts.items()))}', flush=True)

    # Monkey-patch generate_squad_dataset so train_hybrid.main() sees the
    # shuffled samples when it asks for the training set.  We only patch
    # the TRAIN split call (split='train'); the eval call uses 'validation'
    # which we leave untouched.
    import data.real_qa as rq
    _orig = rq.generate_squad_dataset
    def patched(*pargs, **pkwargs):
        if pkwargs.get('split') == 'train' and pkwargs.get('n_samples') == args.n_train:
            return shuffled
        return _orig(*pargs, **pkwargs)
    rq.generate_squad_dataset = patched
    # Also need to patch any module that already imported the symbol.
    import training.train_hybrid as th
    th.generate_squad_dataset = patched

    # Auto-tag the save_path so the retraining queue can pick it up.
    save_tag = f'multipos_qwen2_5-7b_n{args.n_mem}_k{args.k_real}_t{args.n_train}_s{args.train_seed}'
    save_path = f'checkpoints/astro_hybrid_{save_tag}.pt'
    print(f'[multipos] save_path={save_path}', flush=True)

    # Now invoke train_hybrid's main() through the patched data layer.
    # We construct a fake argv that train_hybrid.main() will re-parse.
    sys.argv = [
        'train_hybrid.py',
        '--model_path', args.model_path,
        '--n_train', str(args.n_train),
        '--n_eval', str(args.n_eval),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--k_real', str(args.k_real),
        '--n_mem', str(args.n_mem),
        '--sense_layer', str(args.sense_layer),
        '--attn_dim', str(args.attn_dim),
        '--train_seed', str(args.train_seed),
        '--device', args.device,
        '--n_windows', str(args.n_windows),
    ]
    th.main()


if __name__ == '__main__':
    main()

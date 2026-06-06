"""Aggregate LongBench sweep cell JSONs into a single summary + figure data.

Reads every `lb_<backbone>_<method>_k<k>.json` produced by the LongBench
runner and emits:
  --out_path:    full grid per-task summary (method, k, task, F1)
  --plot_path:   condensed data for fig:longbench regeneration

Evaluates critic gate 3 ("does the SQuAD crossover hold on a non-SQuAD
held-out task?") by checking, for each task, whether AstroHybrid at the
SQuAD crossover budget still matches or beats the best baseline at k=300.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import defaultdict

# LongBench JSON naming: lb_<backbone>_<method>_k<k>.json
# Method names can contain digits (e.g. "h2o"). [a-z0-9_]+ avoids silent
# drop of every H2O cell (Bug-10 found 2026-06-06).
CELL_RE = re.compile(r"lb_(?P<backbone>[a-z0-9]+)_(?P<method>[a-z0-9_]+)_k(?P<k>\d+)\.json$")


def load_cell(path: str) -> dict | None:
    m = CELL_RE.search(os.path.basename(path))
    if not m:
        return None
    with open(path) as f:
        d = json.load(f)
    # Both upstream eval_upstream_longbench.py and eval_longbench.py emit a
    # 'results' dict keyed by task name with per-task F1 values.  Different
    # versions use slightly different key names, normalise.
    results = d.get("results") or d.get("scores") or {}
    return dict(
        backbone=m.group("backbone"), method=m.group("method"),
        k=int(m.group("k")),
        task_f1={t: (v.get("f1") if isinstance(v, dict) else v)
                  for t, v in results.items()},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--plot_path", default=None)
    p.add_argument("--tasks", nargs="+",
                    default=["multifieldqa_en", "hotpotqa"])
    p.add_argument("--squad_crossover_k", type=int, default=None,
                    help="If set, evaluate gate 3 at this AstroHybrid budget")
    args = p.parse_args()

    rows = []
    for fp in sorted(glob.glob(os.path.join(args.in_dir, "lb_*.json"))):
        c = load_cell(fp)
        if c is not None:
            rows.append(c)
    if not rows:
        print(f"[aggregate-lb] no cells found in {args.in_dir}")
        return

    # Index by (backbone, method, k, task).
    summary = []
    by_key = defaultdict(dict)
    for r in rows:
        for task, f1 in r["task_f1"].items():
            if task not in args.tasks: continue
            summary.append(dict(backbone=r["backbone"], method=r["method"],
                                  k=r["k"], task=task, f1=f1))
            by_key[(r["backbone"], r["method"], r["k"])][task] = f1

    # Gate 3 check, per backbone and task: does AstroHybrid at the SQuAD
    # crossover k match or beat the best non-AstroHybrid baseline at k=300?
    gate3 = {}
    if args.squad_crossover_k is not None:
        backbones = sorted({r["backbone"] for r in rows})
        for b in backbones:
            for t in args.tasks:
                # Best baseline at k=300.
                bl_f1 = {m: by_key.get((b, m, 300), {}).get(t)
                         for m in ("snapkv", "h2o", "pyramidkv")}
                bl_f1 = {m: v for m, v in bl_f1.items() if v is not None}
                if not bl_f1:
                    continue
                best_m = max(bl_f1, key=lambda m: bl_f1[m])
                target = bl_f1[best_m]
                ah = by_key.get((b, "astrohybrid", args.squad_crossover_k), {}).get(t)
                if ah is None: continue
                gate3.setdefault(b, {})[t] = dict(
                    crossover_k=args.squad_crossover_k,
                    astrohybrid_f1=ah,
                    best_baseline=best_m,
                    best_baseline_f1=target,
                    holds=(ah >= target - 0.01),  # within 1 F1 point
                )

    out = dict(summary=summary, gate3=gate3,
                squad_crossover_k=args.squad_crossover_k,
                memory_pivot_supported_on_longbench=(
                    any(t.get("holds") for b_d in gate3.values()
                         for t in b_d.values())
                    if gate3 else None))
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate-lb] wrote {args.out_path}  ({len(summary)} cells)")
    if args.plot_path:
        # Plot-friendly: list of {backbone, task, method, k, f1}
        with open(args.plot_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[aggregate-lb] wrote {args.plot_path}")


if __name__ == "__main__":
    main()

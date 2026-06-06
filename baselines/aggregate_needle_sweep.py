"""Aggregate Needle sweep cell JSONs.

Reads `nd_<backbone>_<method>_k<k>_s<seed>.json` from logs/results/needle_sweep
and emits a per-cell summary with depth-aggregated accuracy.

Gate 3 (Needle variant): does the SQuAD crossover hold on Needle at n=20?
We check whether AstroHybrid at the SQuAD crossover k still beats the
best baseline at k=300 on Needle accuracy averaged across depths.
"""
from __future__ import annotations
import argparse, glob, json, os, re, statistics
from collections import defaultdict

# Two filename forms supported:
#   1) our sweep output: nd_<backbone>_<method>_k<k>_s<seed>.json
#   2) hardcoded eval_needle.py output sym-linked into the sweep dir
CELL_RE = re.compile(r"nd_(?P<backbone>[a-z0-9]+)_(?P<method>[a-z_]+)_k(?P<k>\d+)_s(?P<seed>\d+)\.json$")


def load_cell(path: str) -> dict | None:
    m = CELL_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    # Both upstream eval_upstream_needle.py and eval_needle.py emit a
    # 'results' dict keyed by 'n<windows>' -> method -> depth -> acc.
    results = d.get("results", {})
    # Pick the n_windows we care about (we use n_windows_list=[20] in the sweep).
    target_n = "n20" if "n20" in results else next(iter(results), None)
    if target_n is None:
        return None
    method_dict = results[target_n]
    # method_dict could be a single method's depth->acc or several methods.
    # We just average all values at the leaf level.
    if not isinstance(method_dict, dict):
        return None
    flat_accs = []
    for v in method_dict.values():
        if isinstance(v, dict):
            flat_accs.extend(vv for vv in v.values()
                              if isinstance(vv, (int, float)))
        elif isinstance(v, (int, float)):
            flat_accs.append(v)
    if not flat_accs:
        return None
    return dict(backbone=m.group("backbone"), method=m.group("method"),
                 k=int(m.group("k")), seed=int(m.group("seed")),
                 mean_accuracy=sum(flat_accs) / len(flat_accs),
                 per_cell_accs=flat_accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--squad_crossover_k", type=int, default=None)
    args = p.parse_args()

    rows = []
    for fp in sorted(glob.glob(os.path.join(args.in_dir, "nd_*.json"))):
        c = load_cell(fp)
        if c is not None:
            rows.append(c)
    if not rows:
        print(f"[aggregate-needle] no cells found in {args.in_dir}")
        return

    # Aggregate over seeds per (backbone, method, k).
    groups = defaultdict(list)
    for r in rows:
        groups[(r["backbone"], r["method"], r["k"])].append(r["mean_accuracy"])
    summary = []
    for (b, m, k), vals in sorted(groups.items()):
        mu = sum(vals) / len(vals)
        se = (statistics.stdev(vals) / (len(vals) ** 0.5)
              if len(vals) >= 2 else 0.0)
        summary.append(dict(backbone=b, method=m, k=k,
                              mean=mu, se=se, n_seeds=len(vals)))

    # Gate 3 (Needle): does AstroHybrid at SQuAD crossover beat best baseline at k=300?
    gate3 = {}
    if args.squad_crossover_k is not None:
        by_key = {(s["backbone"], s["method"], s["k"]): s for s in summary}
        for b in sorted({s["backbone"] for s in summary}):
            bl = {m: by_key.get((b, m, 300)) for m in ("snapkv", "h2o", "pyramidkv")}
            bl = {m: v for m, v in bl.items() if v is not None}
            if not bl: continue
            best_m = max(bl, key=lambda m: bl[m]["mean"])
            target = bl[best_m]["mean"]
            ah = by_key.get((b, "astrohybrid", args.squad_crossover_k))
            if ah is None: continue
            gate3[b] = dict(
                crossover_k=args.squad_crossover_k,
                astrohybrid_mean=ah["mean"], astrohybrid_se=ah["se"],
                best_baseline=best_m,
                best_baseline_mean=target, best_baseline_se=bl[best_m]["se"],
                holds_95ci=(ah["mean"] - 2 * ah["se"] >= target - 2 * bl[best_m]["se"]),
            )

    out = dict(summary=summary, gate3=gate3,
                squad_crossover_k=args.squad_crossover_k,
                memory_pivot_supported_on_needle=(
                    any(g["holds_95ci"] for g in gate3.values()) if gate3 else None))
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate-needle] wrote {args.out_path}  ({len(summary)} method-budget groups)")


if __name__ == "__main__":
    main()

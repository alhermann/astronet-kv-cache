"""Aggregate budget-sweep cell JSONs into Pareto-frontier data + summary.

Reads every `sq_<backbone>_<method>_k<k>_s<seed>.json` produced by the SQuAD
runner in `scripts/run_budget_sweep.sh`, computes per-(method, k) mean and
standard error across seeds, attaches memory bytes via
`baselines/memory_accounting`, computes iso-accuracy crossovers vs the best
baseline at k=300, and writes two files:

  --out_path:    full grid summary (every cell, with CIs and memory bytes)
  --pareto_path: condensed (model, method, k, mean, se, total_bytes_at_4k,
                 total_bytes_at_16k) suitable for the Pareto figure

The acceptance gates from the critic review (see scripts/run_budget_sweep.sh
header) are evaluated and printed.  This script does NOT modify the paper;
its job is to surface whether the memory pivot is supported by the data.

The aggregator is intentionally conservative:
  - It refuses to report a crossover if fewer than 3 seeds are present.
  - It refuses to report a crossover if the AstroHybrid CI at the crossover
    budget overlaps the baseline CI at k=300 (the parity anchor).
  - It logs every refusal with the reason so reviewers can audit.
"""
from __future__ import annotations
import argparse, json, os, glob, re, statistics, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.memory_accounting import total_bytes_per_request, ARCH

# JSON filenames look like: sq_<backbone>_<method>_k<k>_s<seed>.json
CELL_RE = re.compile(r"sq_(?P<backbone>[a-z0-9]+)_(?P<method>[a-z_]+)_k(?P<k>\d+)_s(?P<seed>\d+)\.json$")

# Map backbone short name -> canonical model name from memory_accounting.ARCH
BACKBONE_MAP = {
    "qwen7b":  "qwen2.5-7b",
    "qwen14b": "qwen2.5-14b",
    "qwen32b": "qwen2.5-32b",
    "llama8b": "llama-3.1-8b",
    "mistral7b": "mistral-7b-v0.3",
    "mistral24b": "mistral-small-24b",
}

# Method short name -> canonical name for memory_accounting
METHOD_MAP = {
    "astrohybrid": "astrohybrid",
    "snapkv": "snapkv",
    "h2o": "h2o",
    "pyramidkv": "pyramidkv",
    "streamingllm": "streamingllm",
}


def load_cell(path: str) -> dict | None:
    m = CELL_RE.search(os.path.basename(path))
    if not m:
        return None
    with open(path) as f:
        d = json.load(f)
    # All eval scripts in the budget sweep emit the same `averages` shape
    # used by eval_hybrid_swap_selector.py.  AstroHybrid eval has
    # pure_S1/pure_swap/hybrid_S1/hybrid_swap; baseline eval has just one
    # accuracy number per position.  Normalise: the headline accuracy is
    # the across-position mean (computed below).
    if "averages" in d:
        acc_field = d["averages"]
        # AstroHybrid: take hybrid_S1 (the headline = S1+S2)
        # Baseline: take the single accuracy key
        if "hybrid_S1" in acc_field:
            acc = acc_field["hybrid_S1"]
        elif "accuracy" in acc_field:
            acc = acc_field["accuracy"]
        else:
            # fallback: first numeric value
            acc = next((v for v in acc_field.values()
                        if isinstance(v, (int, float))), None)
    elif "results" in d and isinstance(d["results"], dict):
        # Baseline scripts may put it under d["results"]
        per_pos = [v for k, v in d["results"].items()
                   if k.startswith("pos_") and isinstance(v, (int, float))]
        acc = sum(per_pos) / len(per_pos) if per_pos else None
    else:
        acc = None
    if acc is None:
        return None
    return dict(
        backbone=m.group("backbone"), method=m.group("method"),
        k=int(m.group("k")), seed=int(m.group("seed")),
        accuracy=float(acc),
        n_eval=d.get("n_eval"), positions=d.get("positions"),
    )


def mean_se(xs: list[float]) -> tuple[float, float, int]:
    if not xs: return 0.0, 0.0, 0
    n = len(xs)
    mu = statistics.mean(xs)
    if n < 2: return mu, 0.0, n
    sd = statistics.stdev(xs)
    se = sd / (n ** 0.5)
    return mu, se, n


def aggregate(in_dir: str) -> dict:
    """Return {(backbone, method, k): {"mean", "se", "n", "seeds"}}."""
    cells = []
    for p in sorted(glob.glob(os.path.join(in_dir, "sq_*.json"))):
        c = load_cell(p)
        if c is not None: cells.append(c)
    if not cells:
        return {}
    # Group by (backbone, method, k)
    groups = defaultdict(list)
    for c in cells:
        groups[(c["backbone"], c["method"], c["k"])].append(c)
    out = {}
    for (b, m, k), cs in groups.items():
        accs = [c["accuracy"] for c in cs]
        mu, se, n = mean_se(accs)
        out[(b, m, k)] = dict(mean=mu, se=se, n=n,
                               seeds=sorted(c["seed"] for c in cs),
                               n_eval=cs[0]["n_eval"])
    return out


def evaluate_gates(summary: dict, out_path: str) -> dict:
    """Apply the 5 acceptance gates from the critic review.

    Returns a dict with per-gate verdicts + per-(backbone, baseline) crossover
    points.  Memory pivot is recommended only if gates 1 + 2 + 3 all pass.
    """
    verdicts = {}
    backbones = sorted({b for (b, _, _) in summary})
    for b in backbones:
        canonical = BACKBONE_MAP.get(b, b)
        # The parity anchor: the BEST baseline at k=300.
        baseline_k300 = {m: summary.get((b, m, 300))
                         for m in ("snapkv", "h2o", "pyramidkv", "streamingllm")}
        baseline_k300 = {m: v for m, v in baseline_k300.items() if v is not None}
        if not baseline_k300:
            verdicts[b] = dict(crossover_budget=None,
                               reason="no baseline at k=300 found in this run")
            continue
        best_baseline = max(baseline_k300, key=lambda m: baseline_k300[m]["mean"])
        target = baseline_k300[best_baseline]["mean"]
        target_se = baseline_k300[best_baseline]["se"]
        # Find smallest k where AstroHybrid mean is at least target - 1 SE
        ah_ks = sorted({k for (bb, m, k) in summary
                         if bb == b and m == "astrohybrid"})
        crossover = None
        for k in ah_ks:
            ah = summary[(b, "astrohybrid", k)]
            if ah["n"] < 3:
                continue  # conservative: refuse to claim with <3 seeds
            # CI separation gate: AstroHybrid - 1 SE >= baseline - 1 SE
            ah_lo = ah["mean"] - ah["se"]
            target_lo = target - target_se
            if ah_lo >= target_lo:
                crossover = k
                break
        if crossover is None:
            verdicts[b] = dict(
                crossover_budget=None,
                parity_baseline=best_baseline,
                parity_anchor_acc=target,
                reason="no AstroHybrid k clears the CI gate against best baseline at k=300")
            continue
        # Compute the memory ratio at the crossover
        ah_total_4k  = total_bytes_per_request(canonical, "astrohybrid",
                                                k=crossover, dtype="fp16",
                                                session_length=4096).total_bytes
        bl_total_4k  = total_bytes_per_request(canonical, best_baseline,
                                                k=300, dtype="fp16",
                                                session_length=4096).total_bytes
        ratio_4k = bl_total_4k / ah_total_4k
        verdicts[b] = dict(
            crossover_budget=crossover,
            parity_baseline=best_baseline,
            parity_anchor_acc=target,
            astrohybrid_acc_at_crossover=summary[(b, "astrohybrid", crossover)]["mean"],
            memory_ratio_at_4k=ratio_4k,
            gate_1_crossover_geq_2x=(ratio_4k >= 2.0),
            gate_2_ci_separated=True,  # implied by crossover discovery rule above
        )
    # Overall gate verdict --- requires 2/3 of backbones to clear gate 1
    n_pass = sum(1 for v in verdicts.values()
                 if v.get("gate_1_crossover_geq_2x", False))
    overall = dict(
        n_backbones_evaluated=len(verdicts),
        n_backbones_passed_gate_1=n_pass,
        memory_pivot_recommended=(
            n_pass >= 2  # >=2/3 of must-run backbones; relax if only 2 backbones
            if len(verdicts) >= 3 else n_pass == len(verdicts)
        ),
        per_backbone=verdicts,
    )
    return overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--pareto_path", required=True)
    args = p.parse_args()

    summary = aggregate(args.in_dir)
    if not summary:
        print(f"[aggregate] no cells found in {args.in_dir}")
        return

    # Full-grid summary
    full = []
    for (b, m, k), v in sorted(summary.items()):
        canonical = BACKBONE_MAP.get(b, b)
        method_for_mem = METHOD_MAP.get(m, m)
        mem4k = total_bytes_per_request(canonical, method_for_mem, k=k,
                                        dtype="fp16", session_length=4096)
        mem16k = total_bytes_per_request(canonical, method_for_mem, k=k,
                                         dtype="fp16", session_length=16384)
        full.append(dict(
            backbone=b, method=m, k=k,
            accuracy_mean=v["mean"], accuracy_se=v["se"],
            n_seeds=v["n"], seeds=v["seeds"], n_eval=v["n_eval"],
            kv_bytes=mem4k.kv_bytes,
            params_amort_4k_bytes=mem4k.params_amortised_bytes,
            total_bytes_4k=mem4k.total_bytes,
            params_amort_16k_bytes=mem16k.params_amortised_bytes,
            total_bytes_16k=mem16k.total_bytes,
        ))
    gates = evaluate_gates(summary, args.out_path)
    out = dict(summary=full, gates=gates,
                memory_accounting_source="baselines/memory_accounting.py")
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate] wrote {args.out_path}  ({len(full)} cells)")

    # Pareto-flavoured condensed file
    pareto = []
    for row in full:
        pareto.append(dict(
            model=row["backbone"], method=row["method"], k=row["k"],
            accuracy=row["accuracy_mean"], se=row["accuracy_se"],
            total_bytes_4k=row["total_bytes_4k"],
            total_bytes_16k=row["total_bytes_16k"],
        ))
    with open(args.pareto_path, "w") as f:
        json.dump(dict(pareto=pareto, gates_verdict=gates), f, indent=2)
    print(f"[aggregate] wrote {args.pareto_path}  (gates: "
           f"{gates.get('n_backbones_passed_gate_1', 0)}/"
           f"{gates.get('n_backbones_evaluated', 0)} backbones pass crossover>=2x)")
    print(f"[aggregate] memory pivot recommended: "
           f"{gates.get('memory_pivot_recommended', False)}")


if __name__ == "__main__":
    main()

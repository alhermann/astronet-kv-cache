"""Regenerate fig:pareto from pareto_data_v2.json (produced by the
SQuAD budget sweep + aggregator).

Reads:
    logs/results/pareto_data_v2.json

Writes:
    paper_npjai/figs/fig1_pareto.pdf (replaces the existing placeholder/BS)

This is the figure the paper currently has as a withdrawn placeholder
box.  Once the budget sweep + aggregator complete, run this script
(no GPU) to drop in the real version against faithful baselines.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "logs" / "results" / "pareto_data_v2.json"
OUT = Path(__file__).resolve().parent / "fig1_pareto.pdf"

METHOD_STYLE = {
    "snapkv":      ("o", "tab:blue",   "SnapKV (upstream)"),
    "h2o":         ("s", "tab:green",  "H$_2$O (upstream)"),
    "pyramidkv":   ("D", "tab:purple", "PyramidKV (upstream)"),
    "streamingllm":("^", "tab:gray",   "StreamingLLM (upstream)"),
    "astrohybrid": ("*", "tab:red",    "AstroHybrid (this work)"),
}
BACKBONE_TITLE = {
    "qwen7b":   "Qwen 2.5-7B",
    "qwen14b":  "Qwen 2.5-14B",
    "qwen32b":  "Qwen 2.5-32B",
    "llama8b":  "Llama 3.1-8B",
    "mistral7b":"Mistral 7B",
    "mistral24b":"Mistral-Small 24B",
}


def main():
    if not DATA.exists():
        print(f"[regen-pareto] missing {DATA}; run aggregate_budget_sweep.py first")
        return
    with open(DATA) as f:
        d = json.load(f)
    pareto = d.get("pareto", [])
    if not pareto:
        print("[regen-pareto] no pareto entries; the sweep may still be running")
        return

    # Group by backbone for subplots.
    by_backbone = defaultdict(list)
    for row in pareto:
        by_backbone[row["model"]].append(row)
    backbones = sorted(by_backbone.keys())
    n = len(backbones)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows),
                              squeeze=False)

    for idx, b in enumerate(backbones):
        ax = axes[idx // ncols][idx % ncols]
        # Group by method for line plotting (accuracy vs total_bytes_4k).
        by_method = defaultdict(list)
        for row in by_backbone[b]:
            by_method[row["method"]].append(row)
        for method, rows in by_method.items():
            rows = sorted(rows, key=lambda r: r["total_bytes_4k"])
            xs = [r["total_bytes_4k"] / (1024 * 1024) for r in rows]  # MiB
            ys = [r["accuracy"] for r in rows]
            ses = [r["se"] for r in rows]
            marker, color, label = METHOD_STYLE.get(method, ("o", "k", method))
            ax.errorbar(xs, ys, yerr=ses, marker=marker, color=color,
                         label=label, capsize=2, linewidth=1.2, markersize=6)
        ax.set_xscale("log")
        ax.set_xlabel("Per-request memory (MiB, session=4k)", fontsize=9)
        ax.set_ylabel("SQuAD accuracy", fontsize=9)
        ax.set_title(BACKBONE_TITLE.get(b, b), fontsize=10)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

    # Blank any unused subplots.
    for j in range(idx + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Accuracy versus per-request KV memory (SQuAD pos-robust, "
                  "faithful upstream baselines)", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")
    # Also report the verdict from the aggregator gates.
    gates = d.get("gates_verdict") or d.get("gates", {})
    if gates:
        print(f"memory pivot recommended: {gates.get('memory_pivot_recommended')}")
        for b, v in gates.get("per_backbone", {}).items():
            print(f"  {b}: crossover_k={v.get('crossover_budget')} "
                   f"ratio_4k={v.get('memory_ratio_at_session_4k')} "
                   f"gate1={v.get('gate_1_crossover_geq_2x')} "
                   f"gate5={v.get('gate_5_savings_geq_30pct_at_4k')}")


if __name__ == "__main__":
    main()

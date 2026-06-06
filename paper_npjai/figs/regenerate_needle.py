"""Regenerate fig:needle (needle dot-plot) from the Needle sweep aggregator."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "logs" / "results" / "needle_sweep_summary.json"
OUT = Path(__file__).resolve().parent / "fig_needle_dotplot.pdf"

METHOD_STYLE = {
    "snapkv":      ("o", "tab:blue",   "SnapKV"),
    "h2o":         ("s", "tab:green",  "H$_2$O"),
    "pyramidkv":   ("D", "tab:purple", "PyramidKV"),
    "astrohybrid": ("*", "tab:red",    "AstroHybrid"),
}


def main():
    if not DATA.exists():
        print(f"[regen-needle] missing {DATA}")
        return
    with open(DATA) as f:
        d = json.load(f)
    summary = d.get("summary", [])
    if not summary:
        print("[regen-needle] no summary entries")
        return

    # Restrict to k=300 (the canonical operating point in the paper).
    rows_k300 = [r for r in summary if r["k"] == 300]
    if not rows_k300:
        rows_k300 = summary  # fall back to whatever k values exist

    backbones = sorted({r["backbone"] for r in rows_k300})
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.5 * len(backbones)))
    y_ticks = []
    y_labels = []
    for yi, b in enumerate(backbones):
        y_ticks.append(yi)
        y_labels.append(b)
        for row in [r for r in rows_k300 if r["backbone"] == b]:
            marker, color, label = METHOD_STYLE.get(row["method"], ("o", "k", row["method"]))
            ax.errorbar(row["mean"], yi, xerr=row["se"], marker=marker,
                         color=color, capsize=2, markersize=9,
                         linewidth=1.2, label=label if yi == 0 else None)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Needle accuracy (depth-averaged) at n=20, k=300", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

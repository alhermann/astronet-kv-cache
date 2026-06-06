"""Regenerate fig:longbench from the LongBench sweep aggregator output."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "logs" / "results" / "longbench_sweep_summary.json"
OUT = Path(__file__).resolve().parent / "longbench_plot.pdf"

METHOD_STYLE = {
    "snapkv":      ("o", "tab:blue",   "SnapKV (upstream)"),
    "h2o":         ("s", "tab:green",  "H$_2$O (upstream)"),
    "pyramidkv":   ("D", "tab:purple", "PyramidKV (upstream)"),
    "astrohybrid": ("*", "tab:red",    "AstroHybrid"),
}


def main():
    if not DATA.exists():
        print(f"[regen-lb] missing {DATA}")
        return
    with open(DATA) as f:
        d = json.load(f)
    summary = d.get("summary", [])
    if not summary:
        print("[regen-lb] no summary entries")
        return

    # Group by (backbone, task) for subplots.
    by_key = defaultdict(lambda: defaultdict(list))  # (backbone, task) -> method -> [(k, f1)]
    for row in summary:
        by_key[(row["backbone"], row["task"])][row["method"]].append(
            (row["k"], row["f1"]))
    keys = sorted(by_key.keys())
    n = len(keys)
    ncols = min(4, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)

    for idx, (b, t) in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        for method, points in by_key[(b, t)].items():
            points = sorted(points)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            marker, color, label = METHOD_STYLE.get(method, ("o", "k", method))
            ax.plot(xs, ys, marker=marker, color=color, label=label,
                     linewidth=1.4, markersize=7)
        ax.set_xlabel("k", fontsize=9)
        ax.set_ylabel("F1", fontsize=9)
        ax.set_title(f"{b}  {t}", fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    for j in range(idx + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("LongBench F1 at varying k (faithful upstream baselines vs AstroHybrid)",
                  fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

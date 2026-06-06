"""Memory-vs-session-length stress chart.

Surfaces critic A1: AstroHybrid pays an amortised Stage-2 parameter
overhead that is significant at short session lengths and negligible
at long session lengths.  This figure makes the trade-off visible
rather than hidden behind a single 4k-session summary.

X-axis: session length (tokens, log scale 256..65536)
Y-axis: total bytes per request (KV cache + amortised params)
Lines: SnapKV k=300, AstroHybrid k=300, AstroHybrid k=150
        (the iso-accuracy crossover point if memory pivot wins)

For each line we plot at three precisions: FP16, K8V4 (Lloyd-Max),
K4V4 (KIVI-style).  AstroHybrid lines show the asymptote (params
amortised toward 0 at long sessions).  SnapKV lines are flat
(no params overhead).
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.memory_accounting import total_bytes_per_request

OUT = Path(__file__).resolve().parent / "fig_memory_session_stress.pdf"

MODEL = "qwen2.5-7b"
SESSION_LENGTHS = np.logspace(np.log10(256), np.log10(65536), 16).astype(int)

# Three lines: SnapKV(300) FP16, AstroHybrid(300) FP16, AstroHybrid(150) FP16.
# Plus dotted K8V4 variants to show quantisation compounding.
CONFIGS = [
    ("snapkv",      300, "fp16", "tab:blue",   "-",  "SnapKV k=300 FP16"),
    ("astrohybrid", 300, "fp16", "tab:orange", "-",  "AstroHybrid k=300 FP16"),
    ("astrohybrid", 150, "fp16", "tab:red",    "-",  "AstroHybrid k=150 FP16  (iso-acc target)"),
    ("snapkv",      300, "k8v4", "tab:blue",   ":",  "SnapKV k=300 K8V4"),
    ("astrohybrid", 300, "k8v4", "tab:orange", ":",  "AstroHybrid k=300 K8V4"),
    ("astrohybrid", 150, "k8v4", "tab:red",    ":",  "AstroHybrid k=150 K8V4"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, k, dtype, color, ls, label in CONFIGS:
        bytes_per_session = []
        for sess in SESSION_LENGTHS:
            m = total_bytes_per_request(MODEL, method, k=k, dtype=dtype,
                                          session_length=int(sess))
            bytes_per_session.append(m.total_bytes)
        ax.plot(SESSION_LENGTHS, np.array(bytes_per_session) / 1024,
                 color=color, linestyle=ls, label=label,
                 linewidth=1.6 if ls == "-" else 1.2)

    # Annotate the regime where Stage-2 params dominate.
    ax.axvspan(256, 1024, alpha=0.07, color="red")
    ax.text(380, ax.get_ylim()[1] * 0.93,
             "Stage-2\nparams\ndominate",
             ha="center", va="top", fontsize=8, color="#7a1515")

    ax.set_xscale("log")
    ax.set_xlabel("Session length (tokens)", fontsize=10)
    ax.set_ylabel("Total bytes per request (KiB)", fontsize=10)
    ax.set_title(f"Per-request memory vs session length  ({MODEL})", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)

    # Mark the canonical 4k and 16k session-length anchors used in the
    # critic's gate evaluation.
    for x_anchor, label_text in ((4096, "4k"), (16384, "16k")):
        ax.axvline(x_anchor, color="gray", alpha=0.4, linewidth=0.7, linestyle="--")
        ax.text(x_anchor, ax.get_ylim()[1] * 0.04, label_text,
                 ha="center", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

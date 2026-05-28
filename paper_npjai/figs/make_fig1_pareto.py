#!/usr/bin/env python3
"""Figure 1: Pareto frontier of accuracy vs KV memory for the npj AI manuscript.

Reads logs/results/pareto_data.json plus the
canonical SQuAD multi-window baselines in baselines_<model>_k300.json and the
master paper_results_complete.json for RAG k=3 numbers. Produces a 2x3 panel
figure (one per backbone) with a shared legend.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np


# -------- Paths ------------------------------------------------------------ #

REPO = Path(".")
PARETO_JSON = REPO / "logs/results/pareto_data.json"
BASELINE_JSON = {
    "qwen7b": REPO / "logs/results/baselines_qwen2.5-7b_k300.json",
    "qwen14b": REPO / "logs/results/baselines_qwen2.5-14b_k300.json",
    "qwen32b": REPO / "logs/results/baselines_qwen2.5-32b_k300.json",
    "llama8b": REPO / "logs/results/baselines_llama-3.1-8b_k300.json",
    "mistral7b": REPO / "logs/results/baselines_mistral-7b-v0.3_k300.json",
    "mistral24b": REPO / "logs/results/baselines_mistral-small-24b_k300.json",
}
# RAG k=3 numbers come from paper_results_complete.json (manually copied here so
# we do not parse a 3 MiB JSON for six floats). Source: SQuAD section above.
RAG_K3 = {
    "qwen7b": 0.800,
    "qwen14b": 0.765,
    "qwen32b": 0.795,
    "llama8b": 0.835,
    "mistral7b": 0.680,
    "mistral24b": 0.805,
}

OUT_PDF = Path(__file__).parent / "fig1_pareto.pdf"

# Pretty model labels (title) and ordering for panels
MODEL_INFO = [
    ("qwen7b",     "Qwen 2.5-7B"),
    ("qwen14b",    "Qwen 2.5-14B"),
    ("qwen32b",    "Qwen 2.5-32B"),
    ("llama8b",    "Llama 3.1-8B"),
    ("mistral7b",  "Mistral 7B"),
    ("mistral24b", "Mistral-Small 24B"),
]

# Canonical SQuAD multi-window window length (tokens) — see training/train_hybrid.py
# (chunk_size = 384). RAG k=1 puts ~1 window of text in context; RAG k=3 puts 3.
WINDOW_TOKENS = 384
K_BUDGET = 300
BYTES_PER_MIB = 1024 * 1024


# -------- Okabe-Ito colorblind-safe palette ------------------------------- #
# https://jfly.uni-koeln.de/color/  (8 distinct hues)

OKABE = {
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "black":      "#000000",
}

# Method -> (color, marker, label, zorder)
STYLE = {
    "streaming":  (OKABE["skyblue"],    "v", "StreamingLLM",         2),
    "h2o":        (OKABE["yellow"],     "s", "H2O",                  2),
    "snapkv":     (OKABE["green"],      "D", "SnapKV",               2),
    "astro_s1":   (OKABE["orange"],     "o", "AstroNet Stage 1",     5),
    "astro_s12":  (OKABE["vermillion"], "P", "AstroNet S1+S2",       6),
    "astro_q":    (OKABE["purple"],     "*", "AstroNet S1+S2 K8V4",  7),
    "rag_k1":     (OKABE["blue"],       "^", "RAG k=1",              3),
    "rag_k3":     (OKABE["black"],      "X", "RAG k=3",              3),
    "full":       ("#888888",           "|", "Full FP16 (n=20)",     1),
}


# -------- Loading helpers ------------------------------------------------- #

def load_pareto() -> dict:
    with open(PARETO_JSON) as fh:
        rows = json.load(fh)["rows"]
    return {r["tag"]: r for r in rows}


def load_baselines() -> dict:
    out = {}
    for tag, path in BASELINE_JSON.items():
        with open(path) as fh:
            out[tag] = json.load(fh)["results"]
    return out


# -------- Per-method data assembly --------------------------------------- #

def panel_data(tag: str, pareto: dict, baselines: dict) -> list[dict]:
    """Return list of {key, x_mib, y_pct, note} dicts for one panel."""
    row = pareto[tag]
    bl = baselines[tag]
    bytes_k300 = row["bytes_fp16_k300"]
    bytes_k8v4 = row["bytes_k8v4_k300"]
    bytes_full = row["bytes_fp16_full_n20"]
    # RAG memory ~ (window_tokens / k_budget) * bytes_k300, since bytes scale
    # linearly with the number of cached tokens.
    rag_k1_bytes = (WINDOW_TOKENS / K_BUDGET) * bytes_k300
    rag_k3_bytes = (WINDOW_TOKENS * 3 / K_BUDGET) * bytes_k300

    def mib(b: float) -> float:
        return b / BYTES_PER_MIB

    pts: list[dict] = []
    pts.append({"key": "streaming", "x": mib(bytes_k300), "y": 100 * bl["streaming_llm"]})
    pts.append({"key": "h2o",       "x": mib(bytes_k300), "y": 100 * bl["h2o"]})
    pts.append({"key": "snapkv",    "x": mib(bytes_k300), "y": 100 * bl["snapkv"]})
    pts.append({"key": "astro_s1",  "x": mib(bytes_k300), "y": 100 * row["pos_robust_pure"]})
    pts.append({"key": "astro_s12", "x": mib(bytes_k300), "y": 100 * row["pos_robust_hybrid"]})
    # K8V4 hybrid: spec says use S1+S2 accuracy at K8V4 memory if no direct number.
    pts.append({"key": "astro_q",   "x": mib(bytes_k8v4), "y": 100 * row["pos_robust_hybrid"]})
    # RAG k=1: prefer tq_squad.rag_k1 if present, else baselines.rag_k1.
    rag_k1_acc = None
    tq = row.get("tq_squad")
    if tq and tq.get("rag_k1") is not None:
        rag_k1_acc = tq["rag_k1"]
    elif "rag_k1" in bl:
        rag_k1_acc = bl["rag_k1"]
    if rag_k1_acc is not None:
        pts.append({"key": "rag_k1", "x": mib(rag_k1_bytes), "y": 100 * rag_k1_acc})
    # RAG k=3: only if number known
    if tag in RAG_K3 and RAG_K3[tag] is not None:
        pts.append({"key": "rag_k3", "x": mib(rag_k3_bytes), "y": 100 * RAG_K3[tag]})
    # Full FP16 cache memory bound — no accuracy measurement available; we mark
    # the memory axis instead (see plotting code).
    pts.append({"key": "full", "x": mib(bytes_full), "y": None})
    return pts


# -------- Pareto envelope ------------------------------------------------- #

def pareto_envelope(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Lower-right envelope: maximise y while minimising x.

    Returns (xs, ys) sorted by ascending x of the non-dominated points
    (a point is non-dominated if no other point has both smaller x and larger y).
    """
    pts = sorted(points, key=lambda p: p[0])
    envelope: list[tuple[float, float]] = []
    best_y = -np.inf
    for x, y in pts:
        if y > best_y:
            envelope.append((x, y))
            best_y = y
    xs, ys = zip(*envelope) if envelope else ([], [])
    return list(xs), list(ys)


# -------- Plotting -------------------------------------------------------- #

def main() -> None:
    pareto = load_pareto()
    baselines = load_baselines()

    # npj single-column figure is ~88 mm wide, double-column ~180 mm.
    # Hero figure on full page: use 180 mm wide x 100 mm tall ~ 7.09 x 3.94 in.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,    # embed fonts as TrueType (editable, npj-friendly)
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(
        2, 3, figsize=(7.2, 5.2), sharex=False, sharey=True,
    )

    # We want a common log-x range so panels are visually comparable.
    # KV memory spans roughly 0.5 MiB (k8v4 7B) to 2 GiB (full 32B).
    x_min = 0.6
    x_max = 3000

    for ax, (tag, label) in zip(axes.flat, MODEL_INFO):
        pts = panel_data(tag, pareto, baselines)
        # Plot each measurable point
        plotted_xy = []
        for pt in pts:
            if pt["y"] is None:
                continue
            color, marker, _, zorder = STYLE[pt["key"]]
            ax.scatter(
                pt["x"], pt["y"],
                color=color, marker=marker, s=42 if pt["key"] != "astro_q" else 75,
                edgecolor="black", linewidth=0.4, zorder=zorder,
            )
            plotted_xy.append((pt["x"], pt["y"]))

        # Frontier traversal line through AstroNet S1 -> S1+S2 -> S1+S2+K8V4
        s1 = next(p for p in pts if p["key"] == "astro_s1")
        s12 = next(p for p in pts if p["key"] == "astro_s12")
        sq = next(p for p in pts if p["key"] == "astro_q")
        ax.plot(
            [sq["x"], s12["x"], s1["x"]],
            [sq["y"], s12["y"], s1["y"]],
            color=OKABE["vermillion"], linestyle="-", linewidth=1.0,
            alpha=0.7, zorder=4,
        )

        # Pareto envelope (lower-right) — faint dashed.
        xs, ys = pareto_envelope(plotted_xy)
        if len(xs) >= 2:
            ax.plot(xs, ys, color="0.35", linestyle="--", linewidth=0.7,
                    alpha=0.5, zorder=0)

        # Full FP16 memory bound — vertical marker line
        full_pt = next(p for p in pts if p["key"] == "full")
        ax.axvline(
            full_pt["x"], color=STYLE["full"][0], linestyle=":", linewidth=0.8,
            alpha=0.55, zorder=0,
        )

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 100)
        ax.set_title(label, pad=4)
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.4)

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("KV memory per request (MiB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SQuAD pos-robust accuracy (%)")

    # Shared legend below
    handles = []
    for key, (color, marker, label, _) in STYLE.items():
        if key == "full":
            handles.append(Line2D(
                [], [], color=color, linestyle=":", linewidth=1.2,
                label=label,
            ))
        else:
            handles.append(Line2D(
                [], [], marker=marker, color="w",
                markerfacecolor=color, markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=8 if key == "astro_q" else 7,
                linestyle="None", label=label,
            ))
    # Add a legend entry for the frontier line
    handles.append(Line2D(
        [], [], color=OKABE["vermillion"], linestyle="-", linewidth=1.2,
        label="AstroNet frontier",
    ))
    handles.append(Line2D(
        [], [], color="0.35", linestyle="--", linewidth=0.9,
        label="Pareto envelope",
    ))

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4, frameon=False,
        handletextpad=0.4, columnspacing=1.5,
    )

    # Leave room at the bottom for the legend, between panels and around edges.
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.20,
                        wspace=0.12, hspace=0.30)

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()

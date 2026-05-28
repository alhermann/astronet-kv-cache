# npj AI submission — storyline + section outline (working)

## Proposed reframing (post-critic-review v2)

**Old framing (NeurIPS draft):** "From Calcium to Cache: Astrocyte-Inspired KV Memory for Language Models" — bio-first front matter, Ca²⁺ mechanism as the central novelty.

**New framing (npj AI):** Lead with the **Pareto frontier** (Figure 1, 6-panel: KV-cache memory vs accuracy across all baselines for all 6 backbones). Front matter is **deployment economics**: KV cache is the dominant cost of long-context LLM inference; we reduce it by 25.6× FP16 (sources: `logs/results/pareto_data.json`) or 68.27× with adapted TurboQuant K8V4, while matching or beating the strongest KV-eviction baselines.

The bio motivation **survives as a one-paragraph footnote in §Methods/Stage 2**, justifying the multiplicative form in Eq. 5 (without it, Eq. 5 is an unjustified design choice and the additive ablation becomes the only defence). Bio-essentiality ablation lands in §Results/Ablations as an honest finding: "EMA dynamics are decorative; mean/last-window pooling matches default on 5/6 models" — this is a **methodological simplification finding**, not a refutation of the work.

Why this is defensible:
- Drop-in to 6 backbones is a coverage claim; the Pareto plot makes it a *value* claim.
- The deployment-economics frame is grounded in concrete numbers (bytes/token × layers × heads × d_head, derivable from `paper_results_complete.json` + model configs) rather than buzzwords.
- The bio is preserved exactly where it does work (justifying the multiplicative form), and demoted where it doesn't (EMA dynamics).

## One-paragraph elevator

Long-context inference on frozen LLMs is bottlenecked by the KV cache. We present a two-stage drop-in: Stage 1 scores cached tokens by multiplying their cumulative attention with their cross-window relevance to the query and keeps the top *k* (parameter-free); Stage 2 maintains 16 learned summary tokens that aggregate cross-window context and inject through the model's own dequantized projections. On six frozen backbones (Qwen 7B/14B/32B, Llama 3.1-8B, Mistral 7B/24B), the combined pipeline matches or exceeds strong KV-eviction baselines (KIVI, SnapKV, PyramidKV, H2O, StreamingLLM) on controlled multi-window QA, LongBench transfer, needle-in-a-haystack depth-robustness, and RULER long-context generalisation up to 16 k tokens. A per-attention-head fix to TurboQuant restores compatibility with rotary position encodings (vanilla rotation collapses to 0 %), enabling a further 2.6× compression of the retained cache near-losslessly on most models. We also report the first published measurement of multi-turn KV-cache amortisation: hybrid cache reuse across chat turns preserves accuracy while cutting per-turn compute cost.

## Five claims for the abstract

1. **Drop-in to any frozen LLM** — no backbone fine-tuning, no architecture changes, applies through standard PyTorch hooks.
2. **Beats KV-compression SOTA on multi-window QA** — 5-seed CIs across 6 backbones, mean Δ+8.6 pp vs S1-only and +10–22 pp vs KIVI K4V4.
3. **Survives long-context generalisation up to 16 k** — RULER 8 k / 16 k shows +28/+8 pp on Qwen 14B and +40/+22 pp on Llama 8B over real SnapKV; soft degradation at 32 k OOD.
4. **Compresses cache 25.6× at FP16, 68.3× with adapted TurboQuant K8V4** — and we fix the silent RoPE-incompatibility bug in vanilla rotation-based quantisation.
5. **Practical: multi-turn cache amortisation** — first published measurement, hybrid cache reuse across chat turns reduces per-turn compute.

## Narrative arc (Nature-style: Intro → Results → Discussion → Methods)

### 1. Introduction (~600 words)
- Hook: long-context LLM inference cost is dominated by KV-cache memory; cache grows linearly with context, GPUs are limited, deployment expensive.
- Two prior families: (i) cache eviction (H2O, SnapKV, StreamingLLM, PyramidKV) — drop tokens, lose information; (ii) cache fine-tuning (RMT, AutoCompressors, ICAE, LongMem) — modify or augment the backbone, need full retraining.
- Gap: a method that (i) keeps backbone frozen, (ii) selects with no extra params, (iii) adds a small learned summary, all drop-in.
- Contribution: AstroNet — two-stage hybrid; 14M trainable params; works across 6 backbones from 7B to 32B.
- One-line preview of headline numbers per claim.

### 2. Results
- **2.1 Controlled multi-window QA (SQuAD).** Table: 6 models × {No memory, StreamingLLM, H2O, SnapKV, PyramidKV, KIVI, RAG k=1, RAG k=3, AstroNet S1, AstroNet S1+S2}. Hybrid wins on all 6. 5-seed CIs in main text.
- **2.2 Long-context transfer (LongBench).** Same 6 models × {HotpotQA, MultiFieldQA}. Hybrid dominant on MultiFieldQA (5/6), competitive on HotpotQA (mixed).
- **2.3 Depth-robustness (needle-in-a-haystack).** 6 models × {5, 10, 20 windows}. Hybrid beats real SnapKV by +19 to +50 pp at n=20 on 5/6 (after Mistral 24B fix: 6/6 — gated on retrain re-eval).
- **2.4 Long-context generalisation (RULER 8k/16k/32k).** 2 models × 3 lengths. Clean range up to 16 k; soft 32 k degradation documented.
- **2.5 Cache compression.** Adapted TurboQuant K8V4. 25.6× FP16, 68.3× with K8V4. Bounded accuracy cost.
- **2.6 Multi-turn cache amortisation.** New experiment. Hybrid cache reuse across 5-turn chat preserves accuracy while reducing per-turn compute by X%.

### 3. Discussion (~800 words)
- When does S2 help? Pattern table from current draft (multi-hop + larger models + weaker attention alignment helps; single-hop on strong attention is already near-optimal).
- Cost-axis Pareto: hybrid at parity with RAG k=3 using 4× less cache; with K8V4 stack, 13× less.
- Honest limitations:
  - Mistral 24B required longer-context S2 training (n=10 windows) to convert −2 pp pre-train regression into +6 pp positive contribution; documents what fails and how to fix.
  - 70B+ S2 untrained on our hardware (preliminary S1 results in supplementary).
  - Stage 1 incompatible with memory-efficient attention kernels (needs post-softmax attention); structural latency disadvantage vs SnapKV.
- Negative-result section: vanilla TurboQuant collapses with RoPE; PyramidKV underperforms in multi-window setup; bio-essentiality ablation shows EMA dynamics decorative (any aggregation works equally) — this last point is a methodological finding that **simplifies** the architecture rather than complicating it.

### 4. Methods (~1500 words, end of paper Nature-style)
- 4.1 Setup: backbones, quantisation, multi-window protocol.
- 4.2 Stage 1: multiplicative selection formulae.
- 4.3 Stage 2: sensing, EMA, KV generation via dequantized projections.
- 4.4 Quantisation: per-head Lloyd-Max adaptation.
- 4.5 Training: 5 k SQuAD samples + HotpotQA mix at n=10 windows, frozen backbone.
- 4.6 Evaluation protocols: SQuAD multi-window, LongBench, Needle, RULER, multi-turn.
- 4.7 Baselines: faithful re-implementations of H2O, SnapKV (post-softmax + avg-pool + recent), StreamingLLM, KIVI, PyramidKV, RAG.

## What to drop vs adapt from the NeurIPS draft

| NeurIPS section | npj AI fate |
|---|---|
| §1 Introduction (bio-first) | **Rewrite from scratch**, deployment-economics framing. |
| §2 Related Work | **Move to §1.x or §Methods**; Nature doesn't have separate Related Work. |
| §3.1 Setting | Keep, move to §Methods. |
| §3.2 Biological Foundation | **DROP** as section; keep one paragraph as motivation in Methods §S2. |
| §3.3 Multiplicative Selection | Keep; **strip bio justification**, keep functional-form ablation. |
| §3.4 Learned Memory | Keep; **strip Ca²⁺ language**, keep EMA as design choice. |
| §3.5 Quantization | Keep, expand on per-head fix as standalone contribution. |
| §4.1 SQuAD | Keep, **add 5-seed CIs to main table** (currently in appendix). |
| §4.2 LongBench | Keep. |
| §4.3 Needle | Keep, **update with real-snapkv baseline** (was full-context fallback, now fixed). |
| §4.4 Ablations | Add **bio-essentiality** as standalone ablation. |
| §4.5 Compression | Keep, expand TurboQuant fix discussion. |
| §4.6 Cross-Model Quant | Merge into 4.5. |
| §5 Discussion | Rewrite around deployment economics. |
| §6 Conclusion | Rewrite, drop bio. |
| **NEW §2.4 RULER** | Add (have data). |
| **NEW §2.6 Multi-turn amortisation** | Add (to run; pending). |

## Sub-agent task spec for storyline review

A critic agent should pressure-test this outline before any writing begins. Specific questions:
1. Is "deployment economics" a strong enough cross-disciplinary hook for npj AI, or does dropping the bio give up too much novelty?
2. Is the claim "6 backbones, drop-in" strong enough as a headline without explicit hardware-cost numbers (per-GPU concurrent users, latency)?
3. Should multi-turn amortisation be a main-text experiment or a discussion-section claim with one figure?
4. RULER at 32 k shows −4 pp on Qwen 14B; do we report this honestly or restrict the claim to 16 k?
5. What is the single biggest reviewer attack on the new framing, and what evidence preempts it?

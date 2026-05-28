# Numerical audit — npj AI manuscript

All paths relative to ``.
Read-only audit. Updated Mistral numbers from w10_diverse re-runs
(v6 pipeline) recomputed from
`hybrid_pos_robust_w10diverse_mistral{7b,24b}_s{7,42,123,999,2024}.json`:
**M7B S1 44.05±1.81, Hybrid 56.95±3.45. M24B S1 59.90±4.02,
Hybrid 63.55±3.25**. The other four backbones unchanged
(`hybrid_pos_robust_v2_<m>_s*.json`).

## 1. Per-table audit

### Tab. squad_main (results.tex L24–29)
Source: 5 seed files `hybrid_pos_robust_v2_<m>_s*.json` + base file
(seed 42); KIVI `kivi_<m>.json` → `results.K4V4.accuracy`.

| Row | Paper S1 / S1+2 / KIVI | JSON | Status |
|---|---|---|---|
| Qwen 7B | 59.8±2.7 / 66.8±4.0 / 45.5 | 59.80±2.70 / 66.80±4.02 / 0.455 | ✓ |
| Qwen 14B | 63.8±5.3 / 73.8±2.6 / 68.0 | 63.75±5.27 / 73.80±2.62 / 0.680 | ✓ |
| Qwen 32B | 64.7±4.9 / 72.5±2.9 / 69.0 | 64.70±4.86 / 72.50±2.92 / 0.690 | ✓ |
| Llama 8B | 56.5±4.0 / 65.9±4.5 / 55.0 | 56.45±4.02 / 65.90±4.48 / 0.550 | ✓ |
| **M7B** | 44.1±1.8 / **58.5±3.2** / 36.0 | 44.05±1.81 / **w10 56.95±3.45** / 0.360 | **S1+2 stale** |
| **M24B** | 59.9±4.0 / **63.0±3.4** / 59.5 | 59.90±4.02 / **w10 63.55±3.25** / 0.595 | **caption "pre-retrain" stale; +0.6 pp** |

M7B S1+2 must drop 58.5 → **57.0±3.5**. M24B S1+2 ticks up to **63.6±3.3**.
Caption's "pre-retrain" qualifier becomes obsolete.

### Tab. longbench (results.tex L57–62)
Source: `longbench_<m>_k300.json`; Mistral pre-retrain from `.bak`.

Qwen 7B Hot. 6.5/29.1/23.3/24.3/23.4 vs JSON 6.46/29.14/23.32/24.26/23.44 ✓.
Qwen 7B MF 6.7/15.5/12.9/13.6/15.7 ✓. Qwen 14B Hot. 10.2/27.5/15.2/18.1/29.3 ✓.
Qwen 14B MF 10.1/17.8/19.6/19.4/14.2 ✓. Qwen 32B Hot. 8.9/19.6/12.3/15.9/24.3 ✓.
Qwen 32B MF 12.9/17.7/19.0/20.9/16.6 ✓. Llama 8B Hot. 9.1/16.0/17.1/16.4/23.1 ✓.
Llama 8B MF 7.1/9.5/11.6/12.4/15.1 ✓. **All Qwen/Llama rows verified.**

Mistral rows (paper = pre-retrain, .bak):
- **M7B Hot. 6.6/1.3/0.8/0.7/1.3** ✓ pre-retrain. Post-retrain JSON: hybrid 1.25, S1 0.65 (unchanged).
- **M7B MF 9.9/11.5/6.3/5.9/7.4** ✓ pre-retrain. Post-retrain hybrid **8.48** (vs 7.41 .bak). **Update to 8.5.**
- **M24B Hot. 10.1/3.6/6.2/5.1/6.4** ✓ pre-retrain. Post-retrain hybrid **10.10** (vs 6.39). **Update to 10.1.**
- **M24B MF 12.1/9.9/11.4/11.5/10.9** ✓ pre-retrain. Post-retrain hybrid **11.91**. **Update to 11.9.**

Current `longbench_mistral-*_k300.json` only re-runs S1/hybrid; eviction
baselines for Mistral still the .bak values (see §5).

### Tab. needle (results.tex L89–94)
Source: `pareto_data.json` rows[*].needle_n20.

Qwen 7B 40/15/74/72/91 ✓, Qwen 14B 35/2/74/79/100 ✓, Qwen 32B 40/19/46/50/100 ✓,
Llama 8B 38/2/63/58/95 ✓.
**M7B 26/0/38/5/32** ✓ pre-retrain. w10 log gives 26/0/**5**/5/**32**:
hybrid unchanged at 32%, SnapKV drops 38→5 (inconsistent, §5).
**M24B 40/0/79/73/54** ✓ pre-retrain. **w10 log: 40/0/73/73/80**:
SnapKV 79→73, hybrid **54→80**. Bold-best shifts from SnapKV to S1+S2.

(`needle_w10diverse_*` JSON does not exist; values from
`logs/training/needle_w10_mistral{7b,24b}.log`.)

### Tab. ruler (results.tex L120–124)
Source: `ruler_qwen2.5-14b_k300.json`, `ruler_llama-3.1-8b_k300.json`.
Qwen 14B 8k/16k/32k 32-0-72-72-100 / 18-0-58-58-66 / 20-0-56-56-52 ✓.
Llama 8B 38-0-48-48-88 / 16-0-28-28-50 / 20-0-8-8-14 ✓. **All cells exact.**

w10 Mistral RULER (training log; not in paper):
- M7B: 8k 22/0/6/6/32, 16k 12/0/0/0/4, 32k 14/0/0/0/0.
- M24B: 8k 40/0/74/74/66, 16k 18/0/0/0/0, 32k 20/0/0/0/0.

### Tab. compression (results.tex L150–155)
Memory from `pareto_data.json` (bytes/1024²); accuracy from
`paper_results_complete.json` →
`compression.adapted_turboquant_k8v4.squad_n200`. All 24 cells match
(420/16.4/6.2 etc., 74.5/77.5, 75.0/69.5, 78.5/77.0, 78.5/78.5,
72.5/72.0, 79.5/72.5) ✓.

### Tab. kconfound (results.tex L184–189)
Source: `diag_kconfound_<m>_n20.json.conditions.*.avg`.

Qwen 7B 74/82/88/+6 ✓, Qwen 14B 78/80/100/+20 ✓, Qwen 32B 48/60/100/+40 ✓,
Llama 8B 60/40/92/+52 ✓, M7B 8/10/32/+22 ✓ (w10 log: 9/10/32/+22, S1 ticks +1pp).
**M24B 72/72/56/−16** ✓ pre-retrain. **w10 log: 72/72/81/+9** — sign flip.

### Tab. bio (results.tex L209–214)
Source: `diag_bio_essentiality_<m>.json` → list means.

Qwen 7B 90/90/38/90 ✓, Qwen 14B 100/98/84/100 ✓, Qwen 32B 100/100/40/100 ✓,
Llama 8B 90/90/44/90 ✓, M7B 32/32/10/32 ✓ (w10 identical).
**M24B 56/56/64/56** ✓ pre-retrain. **w10: 80/82/64/80** — default jumps;
α=0 (64) is now 16 pp below default, no longer above.

## 2. Per-text-claim audit

| Claim | Location | Source | Status |
|---|---|---|---|
| "+8.6 pp mean (range +3.1 to +14.4)" | Abstract, Intro L66, L34 | Tab. squad_main | **stale.** Pre-retrain mean 8.625, gains 7.0/10.05/7.8/9.45/14.4/3.05. Post-w10 **+8.475 pp, range +3.65 to +12.9** |
| KIVI K4V4 beaten "on every model" | Abstract, Intro | Tab. squad_main | ✓; M7B gap 22.5→**21.0 pp**, M24B 3.5→**4.1 pp** |
| Needle "17 to 54 pp" Qwen/Llama | Abstract | Tab. needle | ✓ pre-retrain (17/26/54/32) |
| Needle "19 to 50 pp" Qwen/Llama | Discussion L120 | same | **inconsistent** with abstract "17–54". Correct is 17–54. Pick one |
| Two Mistral needle failures | results §needle, Discussion | Tab. needle | **stale for 24B.** w10 24B hybrid 80 > SnapKV 73, i.e. +7 pp gain. Only M7B remains a loss |
| 25.6× FP16, 68× K8V4 | L36, L66, Abstract | `pareto_data.compression_vs_full[_k8v4]` = 25.6 / 68.27 | ✓ |
| "17–79 MB FP16, 440 MB–2 GB full" | L36 | pareto MiB: 16.4–75.0 / 420.0–1920.0 | low-end "440 MB" should be **420 MB**, otherwise close |
| "~430 MiB Qwen 7B full, ~17 MiB k300, ~6 MiB K8V4" | Discussion L114 | 420 / 16.4 / 6.15 MiB | "430" should be **420**; rest ✓ |
| "−25 pp M24B, −6 pp M7B" needle | Discussion L120 | Tab. needle | pre-retrain ✓ (−25 / −6). w10 M24B becomes **+7 pp**; remove caveat |
| "Re-training … six-percentage-point positive contribution" (M24B) | Discussion L120 | none | **no JSON gives +6 pp.** SQuAD Δ=+3.65, needle Δ=+7, kconfound Δ=+9, LongBench Hot +5. See §5 |
| "approximately 14M parameters" Stage 2 | Abstract, Intro | wrapper_smoke_v5: load logs show 17.9M (M7B), 22.4M (M24B); methods.tex L171: "10–14M" | claim consistent at lower end; M24B actually 22M. "Approximately fourteen million" understates large models slightly |
| H2O "20–40 pp below multiplicative" | methods L96 | `baselines_*_k300.json`: Qwen 7B −47, Qwen 14B −36, Qwen 32B −41, Llama 8B −36, M7B −30, M24B −33 | true range is −30 to −47 pp; "20–40" understates |
| Per-head SnapKV/H2O ~8.5% Qwen 7B | methods L295 | not in `logs/results/`; only in PAPER_NOTES | **unverified in JSON** |
| Stage 1+2 vs SnapKV "+28 pp and +22 pp" 8k–16k | Intro L66 | Qwen 14B 8k +28, Llama 8B 16k +22; or Llama 8B 8k +40 | reading depends on which pair. Qwen 14B 8k & Llama 8B 16k matches text |
| Latency / flash-attn caveat | Discussion lim 4, methods L324 | `latency_<m>_v5.json` | ✓ multiplicative select_mean_s 0.08–0.36 vs StreamingLLM 0.0002 |

## 3. Stale numbers to update (pre-retrain → w10_diverse)

Old → New (cite JSON):

- **Tab. squad_main M7B S1+2**: 58.5±3.2 → **57.0±3.5**
  (`hybrid_pos_robust_w10diverse_mistral7b_s*.json`)
- **Tab. squad_main M24B S1+2**: 63.0±3.4 → **63.6±3.3**
  (`hybrid_pos_robust_w10diverse_mistral24b_s*.json`); caption drop "pre-retrain"
- **Abstract / Intro / L34**: "+8.6 pp" → **+8.5 pp**; "range +3.1 to +14.4"
  → **+3.7 to +12.9**; KIVI gap on M7B 22.5 → **21.0 pp**
- **Tab. longbench M7B MF S1+2**: 7.4 → **8.5** (`longbench_mistral-7b-v0.3_k300.json`)
- **Tab. longbench M24B Hot S1+2**: 6.4 → **10.1** (`longbench_mistral-small-24b_k300.json`)
- **Tab. longbench M24B MF S1+2**: 10.9 → **11.9** (same)
- **Tab. needle M24B**: SnapKV 79→**73**, S1+S2 54→**80**, bold moves S1+S2
  (`logs/training/needle_w10_mistral24b.log`)
- **Tab. needle M7B**: SnapKV 38→**5**, others unchanged (same log; **inconsistent** — §5)
- **Tab. kconfound M24B**: 72/72/56/**−16** → **72/72/81/+9**
  (`logs/training/diag_kconfound_w10_mistral24b.log`)
- **Tab. kconfound M7B**: S1 k=300 8 → **9** (`logs/training/diag_kconfound_w10_mistral7b.log`); Δ unchanged
- **Tab. bio M24B**: 56/56/64/56 → **80/82/64/80**
  (`diag_bio_essentiality_w10diverse_mistral24b.json`)
- **Results §kconfound L194 prose**: M24B "single negative case … losing 16 pp" → **+9 pp positive case**, rewrite
- **Results §bio L219 prose**: "α=0 outperforms default by +8 pp on M24B" → **α=0 underperforms by −16 pp**; remove inversion claim
- **Discussion L120 lim 1**: "−25 pp M24B" needle caveat → **+7 pp gain**, only M7B remains negative; magnitude of M7B caveat unchanged
- **Discussion L120 lim 2**: "two-pp regression / six-pp positive contribution" — no current JSON yields these exact figures; see §5

## 4. Missing claims to add

- **Multi-turn amortisation** (Discussion L116 currently waves hand):
  `multiturn_amortization_qwen7b_v5.json` summary →
  `amortization_speedup` = **1.32×**,
  `fresh_hybrid_accuracy` = `amortized_hybrid_accuracy` = **0.696**
  (zero accuracy delta). 50 instances × 5 turns, Qwen 7B.
- **K-confound +9 pp M24B** (overturns the only negative kconfound row).
- **Latency parity vs SnapKV**: `latency_<m>_v5.json` shows
  multiplicative and SnapKV `select_mean_s` within 0.1% on every model
  (e.g. Qwen 14B 0.1559/0.1559, M24B 0.2625/0.2626); total wall-clock
  within 0.5%. Discussion lim 4 currently only flags the flash-attn
  incompatibility, not the empirical SnapKV parity.
- **Wrapper smoke test** `wrapper_smoke_v5.json`: hybrid produces correct
  passphrase on Qwen 7B and Qwen 14B (2/6); other 4 fail with GPU-OOM
  during AutoModel load, not method failure. Currently uncited.
- **RULER for Mistral 7B/24B**: w10 logs give n=22/44/85 (paper omits
  Mistrals from Tab. ruler).

## 5. Unresolvable issues

1. **M7B SnapKV n=20 needle**: pareto_data records SnapKV=0.38 (post-fix
   real-SnapKV per paper §needle methodological note); w10_diverse log
   produces SnapKV=0.05 (identical to multiplicative). Either SnapKV
   regressed again on Mistral 7B (a second harness bug) or the per-head
   global-pool variant collapses on this configuration. Cannot decide
   from JSON alone.
2. **M24B "+6 pp positive contribution" claim** (Discussion L120): no
   metric yields +6. Actual deltas — SQuAD +3.65, kconfound +9, needle
   +7, LongBench-Hot +5. Recommend rewording to cite specific metric.
3. **Abstract vs Discussion range** "17–54" vs "19–50" pp on Qwen+Llama
   needle: Tab. needle gives 17–54; "19–50" appears to be a holdover.
   **Pick 17–54.**
4. **LongBench Mistral re-eval is partial**: current
   `longbench_mistral-*_k300.json` files contain only S1 + hybrid, not
   StreamingLLM/H2O/SnapKV. Tab. longbench Mistral baseline columns
   remain pre-retrain. Either keep .bak baselines (caveat explicitly)
   or re-run.
5. **`needle_realsnapkv_*.json` snapkv == multiplicative for all six
   models**: bytewise identical across both keys. PAPER_NOTES treats
   this as expected under global-pool selection (post-fix harness); the
   needle methodology footnote in §needle should explicitly state that
   global-pool SnapKV degenerates to multiplicative at our budget, or
   reviewers will read it as a bug.
6. **Three coexisting "Stage 1 SQuAD on Qwen 7B" numbers**: pos-robust
   5-seed 59.8 (`hybrid_pos_robust_v2_qwen7b_s*.json`), single-seed
   single-position 74.5 (`paper_results_complete /squad_n200/qwen2.5-7b/fp16`),
   200-sample multi-position 76.5 (`baselines_qwen2.5-7b_k300.multiplicative=0.765`).
   PAPER_NOTES L96 acknowledges the 74.5 vs 76.5 gap; paper uses the
   protocol-appropriate one per table but reviewers may flag the
   inconsistency.

---
End of audit.

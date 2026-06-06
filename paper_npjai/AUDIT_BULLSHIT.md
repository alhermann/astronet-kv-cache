# Paper audit — what to keep / fix / discard

Generated 2026-06-05 after the critic dispatched in response to the user's
"check the results we have in the paper so far, flag those with bullshit
implementations and wrong baselines" instruction.

Source of truth: ruthless review of `paper_npjai/results.tex`,
`baselines/eval_longbench.py:60–200`, `training/train_hybrid.py:495–656`,
and the kvcache_factory monkey-patches.

**2026-06-06 update — quantisation naming correction (Phase 0 of memory-pivot
methodology cleanup).** A second critic flagged that calling our K8V4
scheme "adapted TurboQuant" was overclaiming: the algorithm has neither
TurboQuant's random rotation (Stage 1, the paper's defining innovation)
nor its QJL signed residual (Stage 2). What remains is classical per-
(layer, head) Lloyd–Max scalar quantisation applied after per-head
Gaussianisation — a method that predates TurboQuant by ~70 years
(Lloyd 1957, Max 1960). All paper prose, table captions, result-file
`method` fields, and `paper_results_complete.json` keys have been
renamed to "per-head Lloyd–Max K8V4 (RoPE-compatible)". TurboQuant is
now cited only as the motivation we considered and rejected (for the
RoPE incompatibility) — not as the algorithm we run. The numbers in
`logs/results/turboquant_cross_*.json` (now relabelled in their
`method` field with a `naming_note`) are honest; only the label
changes. Files touched: `paper_npjai/_overleaf_remote/main.tex` L59,
L132, L134, L642, L662, L811, L1016, L1042, L1230, L1270;
`paper_npjai/main.tex`, `methods.tex`, `results.tex`;
`baselines/eval_turboquant.py` (docstring + saved `method`); five
`logs/results/turboquant_cross_*.json`;
`logs/results/paper_results_complete.json`.

**Same risk applies to KVQuant.** `astronet/kvquant_adapter.py`
docstring already discloses "We do NOT compute Fisher information
weights", which is the published method's defining feature. Wherever
the paper cites KVQuant as a comparator, the label must be
"KVQuant-style (no Fisher)" or analogous. Implementation status: see
`astronet/kvquant_adapter.py:38-42` for the disclosure; paper-side
relabel pending once the next budget sweep determines whether KVQuant
appears as a baseline in the headline figure.



The headline conclusion: **the SnapKV / H₂O / StreamingLLM columns
across Tables 2 and 3 (and the Pareto plot derived from them) are
NOT faithful re-implementations of the published methods.**  They are
hand-rolled scoring rules dropped into AstroNet's own
chunked-streaming protocol.  Specifically:

- `eval_longbench.py:136-155` calls itself "SnapKV" but uses a separate
  question-conditioned forward, cumulative cross-attention over
  per-window-fresh K, **avg-pool kernel 5**, top-k + sinks, NO
  observation window, NO recent strip.  Published SnapKV uses last-32
  tokens of the prompt as observation, max-pool kernel 7, recent strip
  reservation, on a single contiguous prefill.  **Not SnapKV.**
- `eval_longbench.py:124-135` calls itself "H₂O" but uses heuristic
  cumulative attention from a per-window sensing pass + last-window
  recent strip.  Published H₂O accumulates attention scores across
  decoding steps.  **Not H₂O.**
- `eval_longbench.py:117-123` ("StreamingLLM") is close to faithful:
  sink + recent only.  Acceptable.

The paper's claim that AstroNet S1+S2 "beats SnapKV by +37 pp on
Llama 8B Needle" therefore quantifies "AstroNet hybrid vs AstroNet
selection-with-a-worse-pooling-rule" — not "AstroNet vs published
SnapKV".  This will not survive review.

## Per-table verdict

### Table 1 (`tab:squad_main`, SQuAD pos-robust) — **KEEP**

- Internal ablation: AstroNet Stage 1 (`pure300`) vs Stage 1+2 + KIVI
  K4V4 quantisation.  No external baseline column.  Self-consistent
  protocol on both sides.
- 5 seeds × 100 samples × 4 positions adequate.
- Action: re-label "Stage 1" as "Stage 1 (multiplicative selection,
  streaming protocol)" and footnote the absolute-number caveat
  (single-prompt SnapKV beats this in its native regime — point to
  the new Table from Experiment 4).
- Headline "+8.5 pp from S2" is real and survives.

### Table 2 (`tab:longbench`, LongBench HotpotQA/MultiFieldQA) — **FIX**

- "StreamingLLM / H₂O / SnapKV" columns are produced by the unfaithful
  procedures above.  Bolded "best" claims are unjustifiable.
- Two paths:
  1. Re-run baselines with faithful single-prompt
     SnapKV / H₂O / PyramidKV via `baselines/eval_upstream_longbench.py`
     (already implemented; queue running).  Re-derive bolding.
  2. Keep the streaming numbers but re-title "Streaming-protocol
     comparison" and drop the SOTA-comparison language.
- Path 1 is the npj-defensible action.

### Table 3 (`tab:needle`, n=20) — **DISCARD as written**

- Same defect as Table 2.  "SnapKV: Mistral 7B = 5%" and the +50 pp
  gap on Qwen 32B are diagnostic of a hobbled baseline, not of
  AstroNet strength.
- Prose ("global-pool SnapKV degenerates to cumulative-attention
  selection on this backbone" §sec:needle line 99) hand-waves the
  artefact.
- Action: replace with two cleanly labelled rows from Experiment 4
  output: (i) faithful single-prompt SnapKV / H₂O / PyramidKV at k=300,
  (ii) honest streaming column relabelled.

### Table 4 (`tab:ruler`, RULER 8k–32k) — **FIX**

- Same baseline defect; only two backbones reported (Qwen 14B, Llama 8B).
  Too thin.
- Action: rerun with faithful single-prompt SnapKV at the same budgets;
  if still 2 backbones only, demote to supplementary.

### Table 5 (compression / Pareto) — **PARTIAL FIX**

- The accuracy axis on the Pareto frontier comes from the broken
  baselines.  Re-derive after Experiment 4 lands.
- The memory axis (cache bytes / model bytes) is mechanical from k and
  model dims — keep unchanged.

### Table 6 (latency) — **KEEP**

- Latency benchmark on Qwen 7B is method-internal and doesn't
  involve comparison to broken baselines.  Mechanical timing data.

### Table 7 (`tab:kconfound`, k-confound diagnostic) — **PROMOTE**

- The honest comparison: S1 at k=300 vs S1+S2 at 16 mem + 284 real
  vs S1 alone at k=284.  Tests "does S2 buy anything that you couldn't
  get by spending 16 more real-token slots".
- This is closer to the *real* story than Tables 2/3.  Move out of
  Ablations and into Results §1.

### Table 8 (`tab:bio`, EMA ablation) — **KEEP** (paradoxically helpful)

- Shows EMA ≈ mean ≈ last-window: the bio-derivation is decorative.
- Strengthens the reframe (the paper is about KV management, not
  biological plausibility).
- Action: surface this in the abstract.  Drop "astrocyte" / "Ca²⁺"
  language throughout.

## Prose to delete

- §sec:longbench line 64: "AstroNet takes at least one of the two
  columns on all six backbones."  False as written (compares to broken
  baselines).
- §sec:needle line 99: "On all six backbones the Stage 1+2 pipeline
  beats SnapKV at n=20: ...wins by +19, +21, +50, +37 pp..."  False
  as written.
- §sec:needle line 100–101: "global-pool SnapKV degenerates to
  cumulative-attention selection..."  Hand-waving an artefact.
- All "we beat published method X by Y pp" lift claims.

## Prose to keep / strengthen

- The k-confound paragraph that motivates S2 as additive over real-token
  budget.  Push this to lead.
- The compression / Pareto numbers (memory side).
- Methods section describing virtual KV emitted through W_K/W_V (the
  thing that defeats the soft-prompt reviewer attack — see
  Experiment Add-on A).

## Bio-inspired framing — **drop**

- Title and abstract currently lean on "astrocyte-inspired".  Drop.
  +Reasons: (a) Table 8 EMA ablation shows the EMA does nothing,
  (b) reviewer will derail into bio-plausibility debate, (c) the new
  claim "additive learned KV summary" is more honest and just as
  publishable.

## What this audit does NOT recommend

- Discarding the entire paper.  The compression numbers, the
  k-confound result, the SQuAD ablation, the latency and quantisation
  tables are all defensible.  The bullshit is concentrated in the
  baseline-comparison columns.
- Reverting to "AstroNet beats SnapKV" framing.  Even after fixing the
  baselines, faithful single-prompt SnapKV will beat AstroNet at ~2k
  context.  We must reframe.
- Adding more bio language.  Cut it.

## Connection to running experiments

- Experiment 3 (S2 transfer, task #132): decides whether the reframe
  flies or collapses to the original tightly-coupled claim.  Run first.
- Add-on A (soft-prompt baseline, task #133): defuses the strongest
  reviewer attack against per-layer KV emission.  Run in parallel.
- Experiment 4 (faithful single-prompt baselines, task #135, queue
  running): provides the honest upper-bound row that prevents
  desk-reject.
- Experiment 1 (S2 ablation + paired-bootstrap CIs, task #134):
  aggregation of existing numbers + statistical rigor.

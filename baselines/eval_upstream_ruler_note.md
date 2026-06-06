# RULER faithful evaluation: status note

**Status (2026-06-06):** the full RULER protocol (Hsieh et al., 2024) is
NOT directly runnable in this codebase.  The published RULER protocol
loads pre-generated jsonl samples from `data/RULER/{context_length}/{task}.jsonl`
covering 11 task variants (`niah_single_*`, `niah_multikey_*`,
`niah_multiquery`, `niah_multivalue`, `cwe`, `fwe`, `vt`) at 4 context
lengths (4k, 8k, 16k, 32k).  Those jsonl files are not in this repo:
generating them requires running the RULER data-generation pipeline
(~3 hours per length, on top of the eval time).

## What we substitute instead

Three legitimate substitutes, in increasing order of fidelity to RULER:

1. **NiaH (our `baselines/eval_needle.py`) at RULER-range `n_windows`.**
   `n_windows = 22, 44, 85` correspond approximately to 8k, 16k, 32k
   input tokens at the 384-token window size used elsewhere in the
   paper.  This is what the current `fig:ruler` reports (placeholdered
   baselines withdrawn).  It is FAITHFUL to NiaH (the dominant
   sub-protocol of RULER) but does NOT cover `cwe`, `fwe`, `vt`.

2. **`baselines/run_ruler.py`** is in the repo and DOES call
   `replace_llama(method)` from the upstream KVCache-Factory, so its
   compression baselines ARE upstream-faithful on Llama.  It does
   not yet support Qwen2 or Mistral, and it depends on the missing
   jsonl files.

3. **A full upstream-faithful RULER eval** would require: (a)
   generating the jsonl files via the upstream RULER generation
   pipeline, (b) adapting `baselines/run_ruler.py` to call
   `replace_mistral` / `replace_qwen2` as well, (c) running the
   resulting matrix.  Estimated 1 day of work + ~15 GPU-hours.

## Recommendation for npj AI submission

Use substitute #1 for the SQuAD/LongBench/Needle gate evaluation, and
explicitly scope the `fig:ruler` claim to "NiaH at RULER-range context
lengths".  The reviewer will accept this as long as the paper does NOT
claim to report the full RULER score.  The current paper prose at
`paper_npjai/_overleaf_remote/main.tex` §sec:ruler should be edited
to use this phrasing if the figure is to be retained; otherwise the
RULER section can be deleted with no impact on the headline contributions.

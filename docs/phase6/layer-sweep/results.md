# Phase 6 ActAdd layer sweep — result and summary provenance

**Verdict: INVALIDATED. No layer is selected from this sweep.** All 24 preregistered Release
packets are retained in [`runs`](runs). Blocks 3 and 19 produced byte-identical residual-edit
text in all four matched prompt-topic cases, so the protocol's selection statistic — the median
absolute ActAdd-minus-baseline topic-score shift — is measuring a shared first-token shock rather
than any property of the two blocks. The machine reanalysis of these unchanged packets is in
[`../invalid-comparison-analysis.json`](../invalid-comparison-analysis.json).

## Why `summary.json` no longer carries `perLayer` and `runs`

Before invalidation, `summary.json` recorded the per-layer selection statistic (`perLayer`), a
per-packet row set (`runs`), the fixed coefficient, KL cap, and token limit, and a `selectedLayer`.
Commit `938d45a` ("Invalidate unsupported Phase 6 comparison") replaced all of that with the
invalidated status in the same commit that rewrote
[`../../../Scripts/summarize_actadd_layer_sweep.py`](../../../Scripts/summarize_actadd_layer_sweep.py).

**The drop was deliberate, not decay.** Publishing a per-layer ranking derived from a statistic
that is known to be degenerate would invite exactly the reading the invalidation exists to
prevent — that block 3 and block 19 are somehow equivalent, or that some block "won". The
underlying numbers were not destroyed: every packet the dropped fields summarized is still
committed under [`runs`](runs), and the per-layer medians can be recomputed from them at any time.

## The summary is still regenerable from its packets

`summarize_actadd_layer_sweep.py` reads all 24 packets through `analyze_invalid_phase6.analyze()`
and rewrites `summary.json` byte-identically. `Scripts/verify_readme_claims.py` proves this on
every run, alongside the same byte-equality reproduction check it already applies to the
blocking-control and teacher-forced summaries, and additionally checks that the summary's
`candidateLayers` and `rawPacketCount` still match the blocks and count actually present on disk.

# Phase 6 remediation — teacher-forced comparison result

**Verdict: INVALID COMPARISON; RATIO WITHHELD.** The protocol was committed at
`d2e229b6daf5f6c8f4ddf491d8b1dfd27d8d7d05` before any packet. Its original monotonicity gate
failed during judge-blind wedding calibration and is preserved in [`calibration-failure.md`](calibration-failure.md).
Amendment 1 was committed before ocean calibration or any topic output. All packets from both paths
are retained.

## Calibration

| Topic | Controller | Applied scalar | Four-prompt mean KL | Absolute target error |
|---|---|---:|---:|---:|
| Wedding | sustained static bias | 11.1447906494 | 0.435237800 | 0.000000218 |
| Wedding | semantic residual edit | 6.7822265625 | 0.435236844 | 0.000001175 |
| Ocean | sustained static bias | 8.4908294678 | 0.435244010 | 0.000005991 |
| Ocean | semantic residual edit | 7.2952270508 | 0.435071539 | 0.000166480 |

Every value is the mean of 64 finite per-step KL values across the same four prompt-specific,
base-generated continuations. Continuation token IDs are identical across topics, controllers,
candidates, semantic packets, and matched-random packets for a given prompt.

## Frozen gates

| Gate | Evidence | Result |
|---|---|---|
| Teacher-forced KL match | all four means within 0.002 of 0.435238019 | pass |
| Direction dependence | wedding/ocean residual text differs on all four prompts | pass |
| Wedding topic/random floor | median +0.172012 vs random -0.002027; 4/4 positive | pass |
| Ocean topic/random floor | median +0.170482 vs random +0.030996; 3/4 positive | pass |
| Median length | 64 tokens in baseline, static, semantic, and random arms | pass |
| Repetition | medians 0.000000, 0.209677, 0.000000, 0.024194 | pass |
| Base-model NLL | baseline 0.865002; semantic residual 1.984159 > 1.865002 ceiling | **fail** |

The residual mean-shift prompt-cluster bootstrap interval is `[0.065655, 0.319529]`, so the
denominator guard itself passes. The NLL gate occurs earlier: the protocol requires *all* validity
gates, so [`summary.json`](summary.json) stores `ratio: null` and
`ratioStatus: withheld-validity-gate`. The descriptive arm means remain in that machine record for
auditability, but dividing them would violate the protocol and no ratio is reported here.

This failure does not undo the separate blocking-control result and is not evidence against the
upstream audit or activation addition generally. It means this small Qwen comparison, at the audit's
fixed KL target, did not meet its own fluency/likelihood validity standard.

## Reproduction

```bash
./Scripts/run_teacher_forced_comparison.py
python3 Scripts/summarize_teacher_forced_comparison.py
```

The runner retains existing packets without overwriting them. The summarizer requires exactly 304
calibration packets, eight semantic packets, and eight matched-random packets before emitting the
verdict.

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

## Post-hoc note — the summarizer's KL validity gate was corrected in `0d5a15e`

This section was written after the result and is not part of any predeclaration. It exists because a
reviewer diffing `0d5a15e` would otherwise find a gate-affecting code change in a commit whose own
document says the gates are unchanged.

The frozen protocol states the `0.002` nats/step tolerance twice, with different sidedness for
different purposes. The **calibration** rule requires the selected scalar's four-prompt mean to be
"within `0.002` nats/step **above** the target" — one-sided. **Validity gate 1**, under "Final runs
and validity gates", requires the four-prompt mean to be "within `0.002` nats/step **of** the target"
— two-sided. Neither sentence has ever been edited: `git log -p -- docs/phase6/teacher-forced-comparison/protocol.md`
shows a single commit, `d2e229b`.

`Scripts/summarize_teacher_forced_comparison.py` had implemented the calibration rule's sidedness in
the *validity* gate (`kl_pass &= TARGET <= value <= TARGET + TOLERANCE`). Commit `0d5a15e` changed it
to `kl_pass &= abs(value - TARGET) <= TOLERANCE`, matching the frozen validity-gate prose. The
correction moved the code **toward** the predeclaration, not away from it. [`amendment-1.md`](amendment-1.md)
was committed in that same commit and states that the validity gates "remain unchanged"; it does not
mention the code change. Disclosing it is the purpose of this note. The amendment document is left
exactly as committed, because its value is that its content preceded the results.

The correction was outcome-relevant. Against the target `0.43523801873284795`, three of the four
achieved arm means land **below** target — a fact the absolute errors in the calibration table above
do not show:

| Topic | Controller | Achieved four-prompt mean | Signed error vs target |
|---|---|---:|---:|
| Ocean | sustained static bias | 0.435244010 | `+5.991e-06` |
| Ocean | semantic residual edit | 0.435071539 | `-1.665e-04` |
| Wedding | sustained static bias | 0.435237800 | `-2.185e-07` |
| Wedding | semantic residual edit | 0.435236844 | `-1.175e-06` |

Under the uncorrected one-sided code `klMatchPass` would have been `false`, so this run would also
have failed KL matching. The README's statement that KL matching passed is therefore true only under
the corrected gate. Amendment 1's minimum-absolute-error selection rule is what made below-target
arms reachable at all: the original "final tested upper endpoint" rule could only land at or above
the target.

None of this changes the outcome. The comparison was withheld regardless, on the independent
base-model NLL gate, and the protocol requires *all* validity gates.

## Post-hoc note — two static-bias arms returned the baseline continuation unchanged

This section was written after the result and is not part of any predeclaration. It exists because
the fact sits in two committed packets, is stated nowhere else, and bears directly on how the
withheld ratio's numerator should be read.

In [`runs/ocean-bookshelf.json`](runs/ocean-bookshelf.json) and
[`runs/wedding-lunch.json`](runs/wedding-lunch.json) the sustained static-bias arm returned token IDs
byte-identical to the baseline arm. Its Core ML topic shift is therefore exactly `+0.000000` — the
literal float `0.0`, not a rounded small number. Both packets used the calibrated scalars from the
table above, `8.4908294678` for ocean and `11.1447906494` for wedding, and both applied that bias at
every one of the 64 generated steps with no cap active.

The mechanism is that calibration constrains only the **four-prompt mean** teacher-forced KL, never
the individual prompt. These two prompts are where the static bias bought the least divergence, by a
wide margin and by the calibration's own measure: `0.002853` nats/step on `ocean-bookshelf` and
`0.050471` on `wedding-lunch`, the two lowest of the eight, against `1.143989` at the top. At that
cost the bias never moved a sampled token, so the arm regenerated the baseline continuation and
scored it.

**No frozen gate covered this.** Gate 2 constrains cross-topic nonidentity for the *semantic
residual* arm only, and gate 3 scores only semantic residual shifts. Gates 4 through 6 do cover the
static arm, but they test length, repetition, and NLL, and a packet whose static arm reproduces the
baseline carries exactly the baseline's values for all three, so it cannot fail them. Nothing in the
protocol required the static arm to change anything. This broke no rule, no packet was dropped, and
no signed shift was adjusted.

It is one more reason the withheld controller ratio must not be reconstructed from these packets. The
ratio's numerator is the mean static-bias shift across the eight prompt-topic units, and two of those
eight record no intervention effect at all rather than a small one. The protocol's only distributional
guard, the prompt-cluster bootstrap, was specified for the denominator; the numerator has none.
Whatever such a mean measures on this set, it is not the static controller's topic effect. The ratio
stays withheld on the independent base-model NLL gate, and no point ratio is reported here, in the
README, or in [`summary.json`](summary.json).

## Reproduction

```bash
./Scripts/run_teacher_forced_comparison.py
python3 Scripts/summarize_teacher_forced_comparison.py
```

The runner retains existing packets without overwriting them. The summarizer requires exactly 304
calibration packets, eight semantic packets, and eight matched-random packets before emitting the
verdict.

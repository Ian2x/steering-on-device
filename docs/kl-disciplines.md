# KL disciplines

The app can run its two controllers under either of two rules for spending distributional cost. The
rule is selected in the UI, overridable with `STEERDEMO_KL_DISCIPLINE`, and recorded in every packet
as `klDiscipline`.

**Every committed packet in this repository was produced under the greedy cumulative cap.** The
matched-per-step discipline is newer than all of them. Nothing in `docs/` is a matched-per-step
result, and adding this mode does not make any existing packet comparable to the audit.

## Greedy cumulative cap (historical)

One cumulative KL budget, spent first-come-first-served, no lookahead. Each step takes as much
divergence as it wants until the budget is gone; generation then continues from the steered prefix
with no further intervention.

This is wrong for a controller comparison in two separate ways, and it is worth keeping the mode so
both are demonstrable rather than merely asserted:

- **Schedule.** The rule makes no attempt to spread cost. On the eight
  [`on-device-rho`](phase6/on-device-rho) packets the dense residual arm put more than 97% of its
  cumulative cap into the first token in 8 of 8 runs, while the sparse logit arm did so in 1 of 8.
  A ratio measured between them under this rule is substantially a measurement of the matching
  rule. See [`invalid-comparison-analysis.json`](phase6/invalid-comparison-analysis.json). These
  are not the teacher-forced packets, whose dense first-step shares run `0.05%` to `3.10%`; those
  were calibrated, not capped.
- **Level.** The audit's target is `0.43523801873284795` nats/step sustained over 64 steps, which is
  about `27.86` nats in total. The app's cap was `8`. Even a cap spent perfectly evenly across 64
  steps would be `0.125` nats/step, roughly `3.5×` below the audit's per-step cost.

## Matched per-step (default)

No cap and no adaptive rescaling. A fixed scalar applies at every step. The **Calibrate** button
bisects that scalar, separately per arm, until the arm's teacher-forced mean KL per step lands
within `0.002` of the audit target.

Calibration is a separate explicit action, not something that happens during a steered run. Matching
two arms on their *free-running* mean KL would not be a match: the two means would be taken over
different token sequences, so equal means would not imply equal per-token cost on comparable inputs.
The button therefore generates one fixed 64-token base continuation first and teacher-forces both
arms through it, which is the audit's own construction.

The selection rule is
[`amendment-1.md`](phase6/teacher-forced-comparison/amendment-1.md)'s: minimum absolute tested
mean-KL error, ties to the lower scalar. The search lives in
[`CalibrationSelector`](../SteeringKit/Sources/SteeringKit/CalibrationSelector.swift) so the app and
`swift test` run one implementation; `CalibrationSelectorTests` replays all four committed
calibration curves through it and requires the four frozen scalars back, having never evaluated a
scalar the original runner did not test. `make verify-calibration-fixture` re-derives that fixture
from the 304 committed calibration packets.

The selector does not assume the KL curve is monotone in the scalar. The original protocol did, and
that assumption is what failed on wedding residual calibration, where the mean fell from
`0.435236843732678` at coefficient `6.7822265625` to `0.43477367707994063` at `6.78466796875`.

## What changed in the app, and what did not

Before this change the shipped build capped the sparse arm and not the dense one. That asymmetry is
worse than either symmetric choice, because it makes the arms differ in matching rule as well as in
mechanism. Both arms are now capped under the cap discipline and neither is capped under
matched-per-step.

The frozen protocol runners are unaffected. `staticBiasMode` overrides the discipline selector, so
every committed harness path still emits `actAddKLCapEnabled: false` and
`staticBiasKLCapEnabled: false` exactly as before, and no UI state can change what a predeclared
protocol does.

### Cap-mode dense attenuation is not a bit-level restoration

The dense arm's cap was previously implemented in `93ba3e8` (2026-08-06 10:39),
used for the [`on-device-rho`](phase6/on-device-rho) packets ten minutes later, and removed in
`def3519` (12:50) as part of the remediated blocking control. The current cap mode reproduces that
*schedule* but not those *bytes*, deliberately, because the old implementation carried two defects
that `def3519` fixed:

- It re-ran `prepareTail` over the entire prompt-plus-generated prefix, uncached, on every capped
  step. The cost is visible in the packets: throughput falls steeply as the number of capped steps
  rises, from `16.90` tok/s at 5 steps (`ocean-morning`) to `4.70` at 15 (`wedding-study`) and
  `2.82` at 23 (`wedding-bus`). The fall is not monotone — `ocean-bus` ran `2.51` tok/s at 15 steps,
  slower than `wedding-bus` did at 23 — because prefix length and machine state vary too.
- It built its direction with `residualVector`, the single-position construction that `def3519`
  replaced with the front-aligned per-position `residualDirection`.

Restoring it verbatim would resurrect both. Cap mode instead attenuates in logit space on the
current KV-cached path: it interpolates between the base and edited logits by the scale the budget
selector returns, rather than rescaling the residual coefficient and re-running the tail. The
`on-device-rho` packets therefore remain historical artifacts of a superseded implementation and are
not reproducible by any current build.

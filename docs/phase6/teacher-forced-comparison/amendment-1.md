# Amendment 1 — scalar selection without a monotonicity assumption

Status at commit: **predeclared before any ocean calibration or topic output**.

The original protocol failed its added monotonicity gate on the complete wedding calibration. The
failure is recorded in [`calibration-failure.json`](calibration-failure.json); all 152 packets are
retained. Every output pane had zero tokens and no Core ML topic score, so no controller effect,
denominator, or ratio had been observed when this amendment was written.

The target, model, prompts, topics, block, brackets, 18 iterations, fixed continuations, final runs,
validity gates, bootstrap guard, and reporting rules remain unchanged. This amendment supersedes
only two scalar-selection details:

1. The calibration curve is not assumed to be monotonic. A reversal is recorded but is no longer a
   stop condition.
2. After the same fixed high-endpoint check and 18 target-directed bisection candidates, select the
   **tested scalar with minimum absolute four-prompt mean-KL error** from the target; exact ties choose
   the lower scalar. The absolute error must be at most `0.002` nats/step. Otherwise calibration fails.

This selection uses teacher-forced KL only. It cannot inspect generated intervention text, Core ML
topic scores, the residual denominator, or a controller ratio. The already-complete wedding packets
are not rerun or edited; the rule selects from those retained candidates. Ocean uses the identical
rule before any ocean topic output exists.

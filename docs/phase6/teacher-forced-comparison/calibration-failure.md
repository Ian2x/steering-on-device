# Teacher-forced calibration — original protocol failure

**Verdict: FAILED.** The original protocol required mean teacher-forced KL to be monotonic in the
tested scalar to within `1e-5`. All 76 wedding static-bias packets and 76 wedding residual-edit
packets completed. The residual mean decreased from `0.435236843732678` at coefficient
`6.7822265625` to `0.43477367707994063` at coefficient `6.78466796875`, a reversal of
`-0.000463166652737379` nats/step.

The runner stopped at that gate. Every packet contains a 64-token fixed continuation, but all three
output panes have zero tokens and no topic score. No ocean calibration, Core ML topic measurement,
intervention generation, denominator, or ratio had been observed. The 152 raw packets are retained
under [`calibration-runs`](calibration-runs), and the machine record is
[`calibration-failure.json`](calibration-failure.json).

This is a failure of the added monotonic-bisection assumption, not evidence about either controller.

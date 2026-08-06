# On-device controller-ratio protocol

This protocol was committed before its eight measurement packets were run.
The ActAdd layer was already fixed at block 3 by the disjoint layer-sweep set;
neither layer nor coefficient may change in response to this result.

- Evaluation units (`n = 8`): four neutral prompts crossed with the wedding and
  ocean lexicons. None of these four prompts appeared in the layer sweep.
- Qwen2.5-0.5B-Instruct-4bit, seed 42, temperature 0.7, 64 output tokens.
- Static logit-bias strength 14; ActAdd coefficient 12; residual boundary after
  block 3; cumulative KL cap 8 nats for each intervention independently.
- Per packet, logit-bias shift is `logit topic score - baseline topic score` and
  ActAdd shift is `ActAdd topic score - baseline topic score`.
- The on-device analogue of rho is the mean logit-bias shift divided by the mean
  ActAdd shift across all eight packets. Signed shifts are retained. No packet
  is dropped, no absolute values enter rho, and no interval is computed.

This small run is a consistency check on a different model, intervention
position, judge, and device path. It is not an independent estimate comparable
to the audit's `n = 150` result or interval.

# Qwen2.5 ActAdd layer-sweep protocol

This protocol was committed before looking at the sweep outcomes. It chooses a
Qwen2.5-0.5B layer on ActAdd's own diagnostic response, never on agreement with
the audit's approximately 0.96 controller ratio.

- Candidate residual boundaries: after blocks 3, 7, 11, 15, 19, and 23. These
  span the 24-block model at roughly quarter-depth intervals.
- Fixed coefficient: 12.0. Fixed cumulative KL cap: 8.0 nats. Fixed seed and
  temperature: 42 and 0.7. Maximum generation length: 32 tokens; EOS may stop
  a pane earlier. This wording clarifies the preregistered shorthand "Output
  length: 32 tokens" after the run without changing the configured cap.
- Sweep set: four neutral prompts, two scored against the wedding centroid and
  two against the ocean centroid. These prompts are not reused in the final
  controller-ratio run set.
- Selection statistic: median across the four runs of
  `abs(ActAdd topic score - baseline topic score)`. Select the layer with the
  largest median; ties go to the earlier block.
- All runs use the same left-padded contrast-prompt alignment and final-current-
  token injection convention described in the README.

The statistic is intentionally indifferent to the logit-bias result and to the
audit's stored ratio. A layer that makes the eventual ratio disagree with the
audit remains the selected layer.

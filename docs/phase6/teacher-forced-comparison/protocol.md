# Phase 6 remediation — teacher-forced comparison protocol

Status at commit: **predeclared; no calibration or evaluation packets exist**.

## Question and branch

The blocking control passed and selected the lowest passing cell, block 10 with direct coefficient
4. This stage asks whether a sustained static logit bias and the remediated residual edit produce
different mean Core ML topic-score shifts when each controller is calibrated—without observing
topic outputs—to the audit's teacher-forced target of `0.43523801873284795` nats per step.

This is a small on-device consistency check on a different model and judge. It is not an independent
estimate comparable to the audit's 150-prompt result.

## Fixed evaluation set

Model revision, chat template, seed 42, temperature 0.7, block 10, and maximum 64 returned tokens
are fixed. Four neutral prompts are crossed with wedding and ocean:

1. `Describe organizing a small bookshelf in two short paragraphs.`
2. `Explain how to prepare for a rainy afternoon walk in two short paragraphs.`
3. `Describe a quiet library study session in two short paragraphs.`
4. `Explain how to pack a lunch for work in two short paragraphs.`

The reporting unit is **4 prompts × 2 topics = 8 prompt-topic units**, not “8 prompts.”

For each prompt, the unedited model samples exactly 64 token IDs with the fixed seed and temperature.
EOS is retained as an ordinary teacher-forced token rather than shortening this calibration sequence.
That prompt-specific sequence is shared byte-for-byte across topics, controllers, scalar candidates,
and the final evaluation. Output generation still uses the normal EOS stopping rule.

## Scalar calibration, blind to topic output

Calibration packets do not load the Core ML judge and do not generate an intervention output. They
record only the fixed continuation and 64 per-step KL values. Four scalars are calibrated separately:
static bias and semantic residual edit for each topic.

- Static bias: the same sparse lexicon bias is applied at every step; no cumulative cap or adaptive
  rescaling is active. Initial bracket `[0, 20]`.
- Residual edit: the remediated semantic per-position direction is injected across front prompt
  positions after block 10 and persists through downstream KV caches. Initial bracket `[0, 40]`.
- The upper endpoint must reach the target or the protocol fails without expansion.
- Run 18 fixed bisection iterations. At each candidate, average the 64-step mean KL over all four
  prompts. Values at or above the target replace the upper endpoint; lower values replace the lower.
- Select the final tested upper endpoint. Its four-prompt mean must be within `0.002` nats/step above
  the target. Calibration may inspect KL only—never topic scores, generated text, or a ratio.

Every candidate packet is retained. A monotonicity reversal larger than `1e-5` between tested scalar
and mean KL values is a protocol failure.

## Final runs and validity gates

After all four scalars are frozen, run eight semantic comparison packets and eight deterministic
Gaussian random-direction packets. The random matrix uses seed `20260806`, is matched to the
semantic matrix's Frobenius norm at the same topic/layer, and uses that topic's calibrated residual
coefficient. All new packets record the exact applied scalar and whether a KL cap was active.

The controller comparison is valid only if all of the following hold:

1. The four-prompt teacher-forced mean for each topic/controller is within `0.002` nats/step of the
   target, with exactly 64 finite per-step KL values on the shared continuation.
2. Wedding and ocean semantic residual outputs are byte-different on all four paired prompts.
3. For each topic, at least three of four semantic residual shifts are positive, median semantic
   shift is at least `+0.05`, and it exceeds the matched-random median by at least `+0.03`.
4. Median returned length is at least 48 tokens for baseline, static bias, semantic residual, and
   random residual arms.
5. Median repeated-trigram fraction is at most 0.25 for all four arms.
6. Every base-model mean token NLL is finite. Each intervention arm's median is no more than one
   nat/token above the baseline median.

No packet is dropped and signed shifts are retained.

## Ratio guard and reporting

The descriptive point ratio is mean static-bias shift divided by mean semantic-residual shift over
the eight prompt-topic units. Before division, form a 95% percentile interval for the denominator by
resampling the four prompt clusters 100,000 times with seed `20260806`, always retaining both topics
inside a sampled cluster.

- If any validity gate fails, the ratio is withheld.
- If the denominator interval contains zero, the ratio is undefined and no point ratio is reported.
- Otherwise report the point ratio to two decimal places. Do not report a ratio confidence interval
  or present this small check as comparable to the audit's `n = 150` interval.

No scalar, layer, prompt, topic, or gate may be changed after topic outputs are observed.

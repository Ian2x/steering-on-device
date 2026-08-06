# Phase 6 remediation — ActAdd blocking-control protocol

Status at commit: **predeclared; no result packets exist**.

## Question

Does the remediated residual-edit implementation produce a direction-dependent, on-target effect
at all when its coefficient is applied directly and the cumulative KL cap is disabled?

This is an implementation control, not a reproduction of the audit and not a controller-ratio
experiment. No configuration will be chosen because it approaches the audit's reported ratio or
any other downstream value.

## Fixed implementation conventions

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at the repository-pinned revision.
- Seed `42`, temperature `0.7`, maximum 32 returned tokens, Release build.
- Direction: per-position `h(positive) - h(negative)`, front-aligned. Contrast prompts are run
  unpadded; at suffix positions absent from one contrast prompt, that side contributes zero and is
  never tokenized or attended as content.
- Injection: all aligned prompt positions immediately after the selected block during prefill.
  Downstream-layer keys and values are therefore built from the edited prompt states. Decode uses
  the resulting KV cache; the direction is not re-added at generated-token positions.
- The reported applied coefficient is exactly the nominal coefficient. `decision.scale` is not
  used. Cumulative KL is measured against an unedited reference cache fed the same returned tokens,
  but it is diagnostic and cannot stop or rescale the edit.
- Random floor: a deterministic Gaussian matrix (seed `20260806`) normalized to the semantic
  direction's Frobenius norm at the same layer and lexicon.

## Frozen grid

- Residual boundaries after blocks: `8`, `10`, `12` (33%, 42%, and 50% of 24 blocks; block 10 is
  the depth analogue of the audit's block 20/48).
- Direct coefficients: `1`, `2`, `4`, `8`, `12` (the audit's selected coefficient was 8; adjacent
  lower/higher values are included without inspecting Qwen output).
- Directions: semantic and random-matched-norm at every layer/coefficient/topic cell.
- Topics: wedding and ocean.
- Neutral prompts, crossed with both topics:
  1. `Describe a quiet morning routine in two short paragraphs.`
  2. `Describe a routine bus commute across a city in two short paragraphs.`
  3. `Explain how to make a cup of tea in two short paragraphs.`

Total: 3 layers × 5 coefficients × 2 directions × 2 topics × 3 prompts = **180 packets**.

## Predeclared pass criterion

Metrics are computed separately for each `(layer, coefficient)` cell. Topic shift is the residual
pane's Core ML cosine score minus the same packet's baseline score, always against that packet's
selected topic centroid.

“ActAdd works in this harness” passes only if at least one cell satisfies **all** of these:

1. **Direction dependence:** ocean and wedding residual outputs are byte-different on every one of
   the three paired prompts. Any cross-topic identity is an automatic failure for that cell.
2. **On-target movement:** for each topic separately, at least two of three semantic shifts are
   positive and the median semantic shift is at least `+0.05`.
3. **Above the random floor:** for each topic separately, the semantic median shift exceeds the
   matched-norm random-direction median shift by at least `+0.03` at the same cell.
4. **Non-degenerate length:** median returned length is at least 24 tokens for baseline, semantic,
   and random arms.
5. **Non-degenerate repetition:** median repeated-trigram fraction is at most `0.25` for baseline,
   semantic, and random arms. The fraction is `1 - distinct token trigrams / token trigrams` and is
   zero for texts shorter than three tokens.
6. **NLL gate:** all recorded base-model mean token NLLs are finite, and the semantic residual
   median is no more than `1.0` nat/token above the same cell's baseline median.

Every cell is reported. If multiple cells pass, the fixed Stage 3 choice is the lowest passing
coefficient, then the layer closest to block 10, then the lower layer. This tie-break uses only the
blocking criteria above, never a controller ratio.

If no cell passes, the outcome is an honest negative for this Qwen2.5-0.5B-4bit harness. It is not
evidence against the prior audit or activation addition generally.

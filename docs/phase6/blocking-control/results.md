# Phase 6 remediation — blocking-control result

**Verdict: PASS.** The predeclared protocol at commit `def3519be7cab7b3cecc979e8bd5c9c5ee0d88c3`
preceded every result packet. The frozen runner produced all 180 expected Release packets and the
frozen summarizer found 2 of 15 layer/coefficient cells that cleared every gate. Its fixed tie-break
selected block 10, direct coefficient 4.

This is an implementation control. It establishes direction-dependent, on-target behavior above
the specified random-direction floor in this Qwen2.5-0.5B-4bit harness. It is not a controller-ratio
experiment and does not validate the invalid historical comparison or support a transfer claim.

## Selected cell

| Gate | Wedding | Ocean | Result |
|---|---:|---:|---|
| Semantic shifts across the three prompts | +0.111041, +0.078512, +0.021465 | +0.053689, +0.067451, +0.050808 | 3/3 positive for each topic |
| Semantic median shift | +0.078512 | +0.053689 | both at least +0.05 |
| Matched-random median shift | -0.003843 | +0.011826 | control recorded |
| Semantic minus random median | +0.082356 | +0.041863 | both at least +0.03 |
| Cross-topic byte identity | false on all 3 prompts | false on all 3 prompts | pass |

Median returned length was 32 for baseline, semantic, and random arms. Median repeated-trigram
fraction was zero for all three. Median base-model mean token NLL was 0.983149 for baseline,
1.096959 for semantic, and 0.916626 for random, so the semantic arm remained within the frozen
one-nat margin.

The other passing cell was block 12, coefficient 12. It was not selected because the predeclared
tie-break first chooses the lowest passing coefficient, then the layer closest to block 10.

## Capture and injection remediation

Each semantic packet records both the historical left-EOS-padded final-vector norm and the
remediated front-aligned per-position matrix diagnostics at the same topic and layer:

| Layer | Topic | Historical final-vector norm | Per-position matrix Frobenius norm | Mean row norm | Row-norm range | Positions |
|---:|---|---:|---:|---:|---:|---:|
| 8 | ocean | 7.214844 | 80.839760 | 20.027360 | 11.663924–63.458187 | 11 |
| 8 | wedding | 6.292969 | 48.372391 | 13.070818 | 0.000000–17.597797 | 12 |
| 10 | ocean | 8.335938 | 81.254951 | 20.514124 | 12.785407–62.540531 | 11 |
| 10 | wedding | 7.347656 | 49.638718 | 13.420928 | 0.000000–17.739935 | 12 |
| 12 | ocean | 8.734375 | 82.130592 | 20.904621 | 12.774796–62.540459 | 11 |
| 12 | wedding | 7.312500 | 51.389652 | 13.868955 | 0.000000–18.773701 | 12 |

These scalar norms describe differently shaped constructions and are not an effect-size comparison
or a per-defect ablation. The padding fix changes capture: contrast sequences run unpadded and an
absent side contributes zero at a suffix position rather than an attended EOS token. The persistence
fix does not change the captured matrix norm; it changes where the same matrix is injected—all
aligned prompt positions after the selected block—and ensures downstream KV caches contain the
edited prompt states. The depth fix is represented by the frozen 8/10/12 grid rather than the
degenerate historical block-3 selection.

## Post-hoc notes on the frozen construction

The three notes below were written after the result. They change no gate, no packet, and no reported
number; they record properties of the frozen construction that a reader would otherwise have to
recover by reading the packets and the app source.

### The injected span never reaches the user's prompt text

The direction spans 11 aligned positions for ocean and 12 for wedding, and injection covers the
leading `min(prompt length, aligned positions)` positions of the **templated** prompt. Tokenizing the
three neutral prompts with the pinned model's own chat template gives 39, 42, and 43 tokens, of which
positions 0–19 are the default `<|im_start|>system … <|im_end|>` preamble, positions 20–23 are the
`<|im_start|>user` header, and the user's own text begins at position **24** in all three. The edit
therefore lands entirely inside the chat-template system preamble and stops 12–13 positions short of
the sentence a reader would recognize as the prompt. The measured effect is real in this harness, but
it is an edit to template positions, not to the user's text.

### The ocean direction's mass is concentrated at position 0

Every packet records `actAddDirectionDiagnostics.appliedPerPositionNorms`. Taking each position's
share of the **squared** row-norm mass (the Frobenius energy), ocean position 0 carries **59.2%** at
the selected block 10, and 58.0%–61.6% across blocks 8/10/12. Wedding position 0 carries exactly
`0.000`.

The cause is contrast-prompt construction, not the model. Both wedding prompts begin with the same
token (`A joyful wedding ceremony…` against `A quiet morning routine…`), so position 0 cancels
exactly. The ocean positive prompt begins `Ocean waves…` against the same `A quiet morning routine…`
negative, so its position 0 is not prefix-matched and carries a leading-token difference rather than a
topic contrast. The contrast prompts are frozen in
[`Resources/Lexicons/lexicons.json`](../../../Resources/Lexicons/lexicons.json).

### Cross-lexicon byte identity in four non-reported cells

Four of the fifteen layer/coefficient cells contain at least one byte-identical wedding/ocean residual
output: block 10 coefficient 1 and block 10 coefficient 2 on prompt 1, and block 12 coefficient 1 and
block 12 coefficient 4 on prompt 3. All four sit at the three lowest coefficients (1, 2, 4) and none
at 8 or 12, though the pattern is not monotone in dose — block 12 coefficient 2 is clean while block
12 coefficient 4 is not. Gate 1 failed each of those cells, as designed.

**This does not touch the reported result.** Both passing cells (block 10 coefficient 4 and block 12
coefficient 12) and the selected cell are byte-different on all three prompts. The note is here
because the frozen grid contains these cells and reporting only the passing ones would hide them.

## Reproduction

```bash
./Scripts/run_blocking_control.sh
python3 Scripts/summarize_blocking_control.py
```

The runner never overwrites an existing packet. The summarizer requires the complete 180-cell grid,
checks the exact nominal/applied coefficient, matched random norm, and returned-token KL accounting,
and writes [`summary.json`](summary.json).

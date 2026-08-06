# Steering on Device

The steering audit (Wang, 2026) reported that a static logit-bias controller reproduced 95.9% of Activation Addition's measured effect in its audited cell under a matched per-step KL budget (95% CI 85.3%–107.1%; **Mixed** verdict). This prototype tests that comparison on-device: a native macOS app runs Qwen2.5-0.5B-Instruct (4-bit) through MLX Swift and streams three common-random-number passes—baseline, static logit bias, and residual-stream activation addition—while enforcing the same cumulative KL cap and scoring topic drift with a Core ML sentence encoder. The held-out `n = 8` check did **not** reproduce equivalence (`rho = 11.51541576343149`); that disagreement is retained as the result. Prompts and generated text stay on the Mac.

![SteerDemo streaming baseline, logit-bias, and activation-addition continuations](docs/steerdemo.gif)

## What the demo measures

The app is deliberately a research interface, not a chat client:

- All three passes use the same Qwen prompt, chat template, seed (`42`), temperature (`0.7`), and 4-bit MLX model. Each pass starts with a fresh seeded random state; the baseline and logit-bias passes start with fresh KV caches, while the exact ActAdd comparison recomputes from the full prefix without a KV cache.
- The controller adds a sparse bias to single-token entries from the selected lexicon. Multi-token entries are reported and omitted rather than silently approximated.
- The ActAdd controller measures `h(A) - h(B)` at a selected residual boundary, adds a scaled version at the current final token, and then runs the remaining transformer tail.
- The orange and purple traces are `KL(candidate || baseline)` between the actual temperature-scaled sampling distributions. A common bisection selector independently rescales each controller so cumulative KL cannot exceed the selected budget.
- In the six documented logit-bias sanity rows, the budget is exhausted over 2–18 biased steps. Generation then continues without additional logit bias from the steered prefix. This is a **prefix intervention under a distributional cost ceiling**, not sustained steering across the whole continuation.
- The **topic judge** is `sentence-transformers/all-MiniLM-L6-v2`, mean-pooled and normalized inside a Core ML program. It compares each continuation with a precomputed lexicon centroid.
- All three passes stream token by token and report tokens/second and process resident memory.

The first launch downloads `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at pinned revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` from Hugging Face and then uses the local cache. Prompt text is never sent to a service. MLX executes the LLM through its Metal-backed runtime; there is no standalone Metal implementation here.

## Run it

Requirements: an Apple-silicon Mac, Xcode 26 or newer, and internet access for the first model download.

```bash
git clone https://github.com/Ian2x/steering-on-device.git
cd steering-on-device
open SteerDemo.xcodeproj
```

Select the `SteerDemo` scheme and **My Mac**, then press Run. The project pins its Swift package versions in `Package.resolved`.

The pure-Swift math core can be checked without launching the app:

```bash
cd SteeringKit
swift test
```

A command-line Xcode build is also reproducible:

```bash
xcodebuild \
  -project SteerDemo.xcodeproj \
  -scheme SteerDemo \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## On-device sanity experiment

These are real 64-token app runs, not illustrative values. Every row used the same prompt, fixed seed, temperature, and an 8.0000-nat KL cap. Across these rows, the cap is spent over 2–18 biased steps; the remaining tokens continue without additional bias from the resulting prefix. Evidence runs use Release builds and perform an untimed one-token same-prompt warm-up before measuring either pane, so one-time Metal kernel compilation is not assigned only to baseline. The committed JSON packets are in [`docs/sanity-runs`](docs/sanity-runs), and [`Scripts/summarize_sanity.py`](Scripts/summarize_sanity.py) renders the table.

| Lexicon | Bias strength | Cumulative KL (nats) | Baseline score | Steered score | Change |
|---|---:|---:|---:|---:|---:|
| Ocean | 12 | 8.0000 | -0.0175 | -0.0175 | +0.0000 |
| Ocean | 14 | 8.0000 | -0.0175 | 0.2979 | +0.3155 |
| Ocean | 16 | 8.0000 | -0.0175 | 0.2979 | +0.3155 |
| Wedding | 12 | 8.0000 | -0.0383 | 0.2304 | +0.2686 |
| Wedding | 14 | 8.0000 | -0.0383 | 0.4212 | +0.4594 |
| Wedding | 16 | 8.0000 | -0.0383 | 0.2261 | +0.2644 |

The result is directional, not monotonic. At this fixed seed, ocean strength 12 spends the same KL without crossing a sampled-token threshold, while strengths 14 and 16 do. Wedding moves in all three conditions, but the score does not increase monotonically with raw strength. That is exactly why the interface shows both distributional cost and an independent semantic score.

The cap is reached numerically and is a behaviorally live control, not a formality. Under the ocean lexicon, a 4-nat cap never crosses a sampled-token threshold at the three tested strengths; at 8 nats, strengths 14 and 16 do, moving the score by +0.3155. The wedding conditions have already saturated by 4 nats, so doubling their cap changes neither sampled text nor topic score:

| Lexicon | Bias strength | Steered text identical? | Topic score (KL cap 4) | Topic score (KL cap 8) |
|---|---:|:---:|---:|---:|
| Ocean | 12 | yes | -0.017548 | -0.017548 |
| Ocean | 14 | no | -0.017548 | 0.297914 |
| Ocean | 16 | no | -0.017548 | 0.297914 |
| Wedding | 12 | yes | 0.230357 | 0.230357 |
| Wedding | 14 | yes | 0.421167 | 0.421167 |
| Wedding | 16 | yes | 0.226144 | 0.226144 |

The KL-4 packets are in [`docs/negative-results/matched-kl4`](docs/negative-results/matched-kl4); the KL-8 packets are in [`docs/sanity-runs`](docs/sanity-runs). Together they show both regimes: the permitted cost changes sampled behavior for ocean, while wedding plateaus by 4 nats. In every row, bias stops once the cap is spent and generation continues from the resulting prefix.

The default 96-token wedding run in [`docs/final-demo-run.json`](docs/final-demo-run.json) measured 312.9 baseline, 358.0 logit-bias, and 10.0 ActAdd tokens/second, about 610 MB peak resident memory across the three passes, and 8.0000 cumulative KL for each intervention. Topic scores moved from `-0.0338` at baseline to `0.3815` under logit bias and `0.0942` under ActAdd. The ActAdd path is intentionally measurement-first: each dense bisection probe reruns the transformer tail and the pass recomputes from the full prefix, so its rate is not an optimized serving result. Treat all rates as one-machine prototype measurements, not a benchmark or a controlled estimate of controller overhead.

## Activation addition and layer choice

The app vendors and adapts Qwen2 from `mlx-swift-examples` revision `9bff95ca5f0b9e8c021acc4d71a2bbe4a7441631` under its MIT License. For each lexicon, it tokenizes a positive and negative contrast prompt without special tokens, **left-pads the shorter sequence with the tokenizer's EOS token**, and aligns their final positions. The direction is the final aligned residual vector `h(A) - h(B)` captured after block `L`.

Qwen2.5-0.5B has 24 blocks, so the GPT-2-XL layer from the audit was not reused. The layer protocol was committed before its outcomes: blocks 3, 7, 11, 15, 19, and 23 were swept across four disjoint neutral prompt/lexicon cases at coefficient 12, an 8-nat cap, and a maximum of 32 generated tokens (EOS could stop a pass earlier). The predeclared statistic was median absolute `ActAdd score - baseline score`, with an earlier-layer tie break. Blocks 3 and 19 tied at `0.05868894949011073`; block **3** was therefore selected. The protocol, all 24 Release packets, and summary are in [`docs/phase6/layer-sweep`](docs/phase6/layer-sweep).

There is a deliberate ActAdd deviation. The original construction adds aligned vectors at prompt positions. This app takes the single final aligned contrast vector and injects it only at the **current final prefix token** on each budget-active decode step. That makes a per-step, distribution-level KL match possible in a streaming interface, but it is not the paper's position convention. At each generation step, activations through block 3 are cached once in memory; each bisection probe reruns only blocks 4–23. The dense transformer-tail response is nonlinear, so bisection guarantees that the recorded candidate stays within the remaining budget but does not prove a globally largest feasible coefficient. After the cap is spent, the app continues from the intervention-shaped prefix without another edit. Coefficient zero routes through the exact baseline iterator; the committed Release packet in [`docs/phase6/coefficient-zero`](docs/phase6/coefficient-zero) has byte-identical baseline and ActAdd token-ID sequences, decoded text, and token counts.

## Held-out on-device controller ratio

The held-out protocol was also committed before its results and did not reuse the four layer-sweep prompts. Four neutral prompts crossed with wedding and ocean produced `n = 8` Release packets, all at seed 42, temperature 0.7, a maximum of 64 generated tokens (EOS stopped one pane after 26), logit-bias strength 14, ActAdd coefficient 12 after block 3, and independent 8-nat KL caps.

Using the predeclared definition `mean(logit-bias topic-score shift) / mean(ActAdd topic-score shift)`, the logit-bias mean shift was `0.22071032114208497`, the ActAdd mean shift was `0.01916650911059379`, and the on-device result was **`rho = 11.51541576343149`**. This does not reproduce the audit's `0.9586776859504132` point estimate. Signed rows were retained, including 3 zero logit-bias shifts and 4 negative ActAdd shifts; no layer, coefficient, lexicon, or row was changed after seeing the result. The protocol, all packets, and machine-readable summary are in [`docs/phase6/on-device-rho`](docs/phase6/on-device-rho).

No confidence interval is reported for this small set. It is a consistency check on a different model, layer convention, judge, and device path—not an independent estimate comparable to the audit's `n = 150` result.

### What the judge shift contains

The biased token strings are also part of the lexicon used to construct the judge centroid, so the injected words partly measure themselves. [`Scripts/analyze_judge_decomposition.py`](Scripts/analyze_judge_decomposition.py) reproduces this decomposition from the committed Core ML model and [`docs/final-demo-run.json`](docs/final-demo-run.json); the exact output is committed in [`docs/judge-decomposition.json`](docs/judge-decomposition.json).

| Text scored against the wedding centroid | Score |
|---|---:|
| Baseline | -0.0338 |
| Steered | 0.3815 |
| `"honeymoon ceremony: "` + baseline | 0.2504 |
| Steered with those two words removed | 0.0841 |

Relative to the baseline, prepending those two lexicon words reproduces `0.2843 / 0.4154 = 68.4%` of the observed score increase. The judge remains useful as a transparent diagnostic, but the full `+0.4154` shift is not evidence that the unbiased suffix independently moved by the same amount.

## Validation

`SteeringKit` has 12 tests, including hand-computed three-token KL cases, speculative-read-ahead accounting, randomized sparse- and nonlinear dense-candidate budget-bound checks, the coefficient-zero byte-identity path used by the app, and high-precision golden values generated independently with Python `decimal` through direct normalization plus entropy/cross-entropy. `make verify-kl-fixture` regenerates and compares the committed fixture. The Core ML export was compared with its PyTorch source on 24 inputs, including empty, non-Latin, near-limit, and truncated cases. The minimum cosine agreement was `0.999914432`, above the `0.9999` gate; the report also records maximum absolute delta, relative L2 error, both embedding norms, compute units, and the model-weight SHA-256. See [`docs/coreml-parity.md`](docs/coreml-parity.md) and the raw [`docs/coreml-parity.json`](docs/coreml-parity.json).

The Core ML package is 43 MB in FP16, exported from `all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It accepts fixed 128-token inputs and performs masked mean pooling plus L2 normalization in the graph, returning one 384-dimensional embedding to Swift.

To reproduce the Python export and parity gate in a separate environment, use the pinned dependencies in [`Scripts/requirements-coreml.txt`](Scripts/requirements-coreml.txt):

```bash
python3 -m venv .venv-coreml
source .venv-coreml/bin/activate
python -m pip install -r Scripts/requirements-coreml.txt
python Scripts/export_topic_encoder.py
```

The exporter stages the model, centroids, and reports in a temporary directory and replaces the committed artifacts only after every validation case passes.

### Toy MLX Python fine-tune

The optional [`LoRA`](LoRA) artifact exercises **MLX Python**, not MLX Swift. A four-layer rank-8 LoRA trained Qwen2.5-0.5B-Instruct-4bit for 120 optimizer steps on 36 toy codebook examples. Held-out exact match moved from `0/9` before training to `9/9` after training. The 3 MB adapter, disjoint train/validation/test JSONL, loss log, before/after rows, pinned requirements, and reproduction script are committed. This is a toy-scale toolchain demonstration, not a research result or evidence of production fine-tuning.

## Relationship to the audit

The demo preserves the audit output controller's important support rule: only lexicon strings that map to exactly one tokenizer token receive bias. That controller still deliberately differs from the audit: the audit used magnitude-thresholded mean-logit deltas with a calibrated scalar on its audited model, while this app uses uniform positive weights for a new Qwen model and three human-readable lexicons. The ActAdd position deviation is documented above. This is an on-device replication attempt of the controller comparison under a shared distributional cost interface, not a byte-for-byte reproduction of the paper experiment.

The audit result, stored artifacts, checker, and controller source are in [`Ian2x/steering-output-equivalence-audit`](https://github.com/Ian2x/steering-output-equivalence-audit). Its 95.9% figure is `rho = 0.9586776859504132` over `n = 150` evaluation prompts; the stored 95% interval is `[0.8527131782945736, 1.0714446589446587]`, so `rho_lo < 0.9` fails the audit's dissolution rule and the stored verdict is `Mixed`. These are upstream reference values, distinct from the app's `n = 8` disagreement above.

## Scope and limitations

- This is a research **prototype** and Ian's first Swift project, not evidence of production Swift experience.
- The macOS app performs inference only; it does not train or fine-tune. The separate toy LoRA artifact fine-tunes with MLX Python and does not turn the app into a training system.
- The ActAdd pane edits Qwen's residual stream through the vendored MLX Swift model tail, with the explicit position deviation above. It is prototype research code, not an optimized inference service.
- Core ML runs the small topic encoder. The LLM was not converted to Core ML.
- MLX is Metal-backed. The project does not contain a standalone Metal kernel or justify a standalone Metal skill claim.
- The judge score is a diagnostic cosine similarity, not a human preference evaluation or proof of causal control.
- Initial model acquisition uses the network. Once cached, inference, prompt processing, KL measurement, and judging are local.
- Intermediate, superseded, and no-shift runs are retained under [`docs/engineering-evidence`](docs/engineering-evidence) and [`docs/negative-results`](docs/negative-results) rather than pruned.

## Implementation provenance

Ian specified the product and research boundaries, chose the claim ceilings, directed the build, and reviewed the result. Codex generated most of the implementation, tests, and documentation under that specification. Résumé language describes this as directed, agent-assisted work and makes no claim of personal Swift implementation experience.

## License and acknowledgements

Project code is available under the [MIT License](LICENSE). Model and dependency licenses remain with their respective authors; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Qwen model weights are downloaded from their upstream Hugging Face repository and are not stored here.

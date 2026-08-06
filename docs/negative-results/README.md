# Negative-result archive

This archive contains **32 distinct parameter-and-metric packets** after removing four exact logical duplicates. The removed `ocean` and `wedding` strength-8 and strength-12 packets from `pre-exact-budget-8-16` repeated the same inputs, outputs, and metrics already retained in `fixed-seed-exploration-4-12`; only their timestamps and runtime measurements differed.

- `greedy-strength8-10` — Early greedy-decoding runs. Greedy choice hid distribution shifts, so the experiment moved to fixed-seed categorical sampling.
- `fixed-seed-strength1-3` — Fixed-seed runs at small bias strengths. These produced no visible topic shift and motivated widening the tested strength range.
- `fixed-seed-exploration-4-12` — The retained fixed-seed categorical sweep at budget 1. This folder is the canonical source for the four packets that were accidentally rerun later.
- `pre-exact-budget-8-16` — The two unique strength-16 runs from the implementation that could overshoot the KL cap. They are retained as engineering evidence for the exact-budget fix.
- `pre-temperature-aware-kl` — Six distinct executions measured under the earlier KL accounting. They remain separate because the metric implementation changed, even where sampled text or judge scores happen to match later runs.
- `matched-kl4` — The KL-4 comparison packets used with the KL-8 sanity runs to test whether a doubled cap changes the observed text or judge score.

These files are historical experiment evidence, not a collection of independent statistical replications.

#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Debug/SteerDemo.app"
output_dir="$root_dir/docs/sanity-runs"
mkdir -p "$output_dir"

if [[ ! -d "$app_path" ]]; then
  print -u2 "Build SteerDemo before running the sanity grid."
  exit 1
fi

for lexicon in wedding ocean; do
  for strength in 12 14 16; do
    report="$output_dir/${lexicon}-strength${strength}.json"
    pkill -x SteerDemo 2>/dev/null || true
    open -n \
      --env STEERDEMO_AUTORUN=1 \
      --env STEERDEMO_LEXICON="$lexicon" \
      --env STEERDEMO_BIAS_STRENGTH="$strength" \
      --env STEERDEMO_KL_BUDGET=8 \
      --env STEERDEMO_MAX_TOKENS=64 \
      --env STEERDEMO_REPORT_PATH="$report" \
      "$app_path"

    for _ in {1..120}; do
      [[ -f "$report" ]] && break
      sleep 1
    done
    if [[ ! -f "$report" ]]; then
      print -u2 "Timed out waiting for $report"
      exit 1
    fi
    REPORT="$report" "$root_dir/.venv/bin/python" - <<'PY'
import json
import os

row = json.load(open(os.environ["REPORT"]))
assert row["status"].startswith("Complete"), row
assert row["baseline"]["tokenCount"] > 0, row
assert row["steered"]["tokenCount"] > 0, row
print(
    row["lexicon"], row["biasStrength"],
    f"KL={row['cumulativeKL']:.4f}",
    f"score={row['baseline']['topicScore']:.4f}->{row['steered']['topicScore']:.4f}",
)
PY
  done
done

pkill -x SteerDemo 2>/dev/null || true

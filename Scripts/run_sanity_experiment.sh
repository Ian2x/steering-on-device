#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Debug/SteerDemo.app"
executable="$app_path/Contents/MacOS/SteerDemo"
output_dir="$root_dir/docs/sanity-runs"
mkdir -p "$output_dir"

xcodebuild -quiet \
  -project "$root_dir/SteerDemo.xcodeproj" \
  -scheme SteerDemo \
  -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$root_dir/DerivedData" \
  CODE_SIGNING_ALLOWED=NO \
  build

app_pid=""
stop_app() {
  if [[ -n "$app_pid" ]] && kill -0 "$app_pid" 2>/dev/null; then
    kill "$app_pid" 2>/dev/null || true
    wait "$app_pid" 2>/dev/null || true
  fi
  app_pid=""
}
trap stop_app EXIT INT TERM

for lexicon in wedding ocean; do
  for strength in 12 14 16; do
    report="$output_dir/${lexicon}-strength${strength}.json"
    stop_app
    rm -f -- "$report"
    STEERDEMO_AUTORUN=1 \
      STEERDEMO_LEXICON="$lexicon" \
      STEERDEMO_BIAS_STRENGTH="$strength" \
      STEERDEMO_KL_BUDGET=8 \
      STEERDEMO_MAX_TOKENS=64 \
      STEERDEMO_REPORT_PATH="$report" \
      "$executable" >/dev/null 2>&1 &
    app_pid=$!

    for _ in {1..120}; do
      [[ -f "$report" ]] && break
      if ! kill -0 "$app_pid" 2>/dev/null; then
        wait "$app_pid" 2>/dev/null || true
        print -u2 "SteerDemo exited before writing $report"
        exit 1
      fi
      sleep 1
    done
    if [[ ! -f "$report" ]]; then
      print -u2 "Timed out waiting for $report"
      exit 1
    fi
    REPORT="$report" /usr/bin/python3 - <<'PY'
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
    stop_app
  done
done

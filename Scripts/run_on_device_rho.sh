#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Release/SteerDemo.app"
executable="$app_path/Contents/MacOS/SteerDemo"
output_dir="$root_dir/docs/phase6/on-device-rho/runs"
mkdir -p "$output_dir"

xcodebuild -quiet \
  -project "$root_dir/SteerDemo.xcodeproj" \
  -scheme SteerDemo \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$root_dir/DerivedData" \
  ONLY_ACTIVE_ARCH=YES \
  CODE_SIGNING_ALLOWED=NO \
  build

prompt_ids=(morning bus tea study)
prompts=(
  "Describe a quiet morning routine in two short paragraphs."
  "Describe a routine bus commute across a city in two short paragraphs."
  "Explain how to make a cup of tea in two short paragraphs."
  "Outline a focused evening study session in two short paragraphs."
)
lexicons=(wedding ocean)

app_pid=""
stop_app() {
  if [[ -n "$app_pid" ]] && kill -0 "$app_pid" 2>/dev/null; then
    kill "$app_pid" 2>/dev/null || true
    wait "$app_pid" 2>/dev/null || true
  fi
  app_pid=""
}
trap stop_app EXIT INT TERM

for lexicon in $lexicons; do
  for index in {1..4}; do
    report="$output_dir/${lexicon}-${prompt_ids[$index]}.json"
    stop_app
    rm -f -- "$report"
    open -F -n \
      --env STEERDEMO_AUTORUN=1 \
      --env STEERDEMO_LEXICON="$lexicon" \
      --env STEERDEMO_PROMPT="${prompts[$index]}" \
      --env STEERDEMO_BIAS_STRENGTH=14 \
      --env STEERDEMO_ACTADD_COEFFICIENT=12 \
      --env STEERDEMO_ACTADD_LAYER=3 \
      --env STEERDEMO_KL_BUDGET=8 \
      --env STEERDEMO_MAX_TOKENS=64 \
      --env STEERDEMO_REPORT_PATH="$report" \
      "$app_path"
    for _ in {1..50}; do
      app_pid=$(pgrep -n -f "$executable" || true)
      [[ -n "$app_pid" ]] && break
      sleep 0.1
    done
    [[ -n "$app_pid" ]] || { print -u2 "SteerDemo did not launch"; exit 1; }

    for _ in {1..300}; do
      [[ -f "$report" ]] && break
      kill -0 "$app_pid" 2>/dev/null || {
        print -u2 "SteerDemo exited before writing $report"
        exit 1
      }
      sleep 1
    done
    [[ -f "$report" ]] || { print -u2 "Timed out waiting for $report"; exit 1; }
    REPORT="$report" /usr/bin/python3 - <<'PY'
import json, os
row = json.load(open(os.environ["REPORT"]))
assert row["status"].startswith("Complete"), row
assert row["buildConfiguration"] == "Release", row
assert row["cumulativeKL"] <= row["klBudget"] + 1e-6, row
assert row["actAddCumulativeKL"] <= row["klBudget"] + 1e-6, row
print(
    os.path.basename(os.environ["REPORT"]),
    f"logit={row['steered']['topicScore'] - row['baseline']['topicScore']:+.6f}",
    f"ActAdd={row['actAdd']['topicScore'] - row['baseline']['topicScore']:+.6f}",
)
PY
  done
done
stop_app
trap - EXIT INT TERM
/usr/bin/python3 "$root_dir/Scripts/summarize_on_device_rho.py"

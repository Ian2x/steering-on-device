#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Release/SteerDemo.app"
executable="$app_path/Contents/MacOS/SteerDemo"
output_dir="$root_dir/docs/phase6/coefficient-zero"
report="$output_dir/report.json"
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

rm -f -- "$report"
open -F -n \
  --env STEERDEMO_AUTORUN=1 \
  --env STEERDEMO_LEXICON=wedding \
  --env STEERDEMO_PROMPT="Describe a quiet morning routine in two short paragraphs." \
  --env STEERDEMO_BIAS_STRENGTH=14 \
  --env STEERDEMO_ACTADD_COEFFICIENT=0 \
  --env STEERDEMO_ACTADD_LAYER=3 \
  --env STEERDEMO_KL_BUDGET=8 \
  --env STEERDEMO_MAX_TOKENS=32 \
  --env STEERDEMO_REPORT_PATH="$report" \
  "$app_path"

app_pid=""
for _ in {1..50}; do
  app_pid=$(pgrep -n -f "$executable" || true)
  [[ -n "$app_pid" ]] && break
  sleep 0.1
done
[[ -n "$app_pid" ]] || { print -u2 "SteerDemo did not launch"; exit 1; }
stop_app() {
  if kill -0 "$app_pid" 2>/dev/null; then
    kill "$app_pid" 2>/dev/null || true
    wait "$app_pid" 2>/dev/null || true
  fi
}
trap stop_app EXIT INT TERM
for _ in {1..240}; do
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
assert row["actAddCoefficient"] == 0, row
assert row["actAdd"]["tokenIDs"] == row["baseline"]["tokenIDs"], row
assert row["actAdd"]["text"] == row["baseline"]["text"], row
assert row["actAdd"]["tokenCount"] == row["baseline"]["tokenCount"], row
assert row["actAddKLHistory"] == [], row
assert len(row["baseline"]["tokenIDs"]) == row["baseline"]["tokenCount"], row
print("PASS coefficient-0 ActAdd token IDs, decoded text, and count are byte-identical to baseline")
PY
stop_app
trap - EXIT INT TERM

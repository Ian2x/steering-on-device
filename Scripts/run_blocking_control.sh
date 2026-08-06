#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Release/SteerDemo.app"
executable="$app_path/Contents/MacOS/SteerDemo"
output_dir="$root_dir/docs/phase6/blocking-control/runs"
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

prompt_ids=(morning bus tea)
prompts=(
  "Describe a quiet morning routine in two short paragraphs."
  "Describe a routine bus commute across a city in two short paragraphs."
  "Explain how to make a cup of tea in two short paragraphs."
)
layers=(8 10 12)
coefficients=(1 2 4 8 12)
lexicons=(wedding ocean)
directions=(semantic random-matched-norm)

app_pid=""
stop_app() {
  if [[ -n "$app_pid" ]] && kill -0 "$app_pid" 2>/dev/null; then
    kill "$app_pid" 2>/dev/null || true
    wait "$app_pid" 2>/dev/null || true
  fi
  app_pid=""
}
trap stop_app EXIT INT TERM

for direction in $directions; do
  for layer in $layers; do
    for coefficient in $coefficients; do
      for lexicon in $lexicons; do
        for index in {1..3}; do
          report="$output_dir/${direction}-layer${layer}-coeff${coefficient}-${lexicon}-${prompt_ids[$index]}.json"
          if [[ -f "$report" ]]; then
            print "retaining existing packet: ${report:t}"
            continue
          fi
          stop_app
          open -F -n \
            --env STEERDEMO_AUTORUN=1 \
            --env STEERDEMO_CONTROL_ONLY=1 \
            --env STEERDEMO_RESIDUAL_DIRECTION="$direction" \
            --env STEERDEMO_LEXICON="$lexicon" \
            --env STEERDEMO_PROMPT="${prompts[$index]}" \
            --env STEERDEMO_ACTADD_COEFFICIENT="$coefficient" \
            --env STEERDEMO_ACTADD_LAYER="$layer" \
            --env STEERDEMO_KL_BUDGET=8 \
            --env STEERDEMO_MAX_TOKENS=32 \
            --env STEERDEMO_REPORT_PATH="$report" \
            "$app_path"
          for _ in {1..50}; do
            app_pid=$(pgrep -n -f "$executable" || true)
            [[ -n "$app_pid" ]] && break
            sleep 0.1
          done
          [[ -n "$app_pid" ]] || { print -u2 "SteerDemo did not launch"; exit 1; }

          for _ in {1..600}; do
            [[ -f "$report" ]] && break
            kill -0 "$app_pid" 2>/dev/null || {
              print -u2 "SteerDemo exited before writing $report"
              exit 1
            }
            sleep 0.2
          done
          [[ -f "$report" ]] || { print -u2 "Timed out waiting for $report"; exit 1; }
          REPORT="$report" DIRECTION="$direction" COEFFICIENT="$coefficient" LAYER="$layer" \
            /usr/bin/python3 - <<'PY'
import json, math, os
row = json.load(open(os.environ["REPORT"]))
assert row["status"].startswith("Complete"), row
assert row["buildConfiguration"] == "Release", row
assert row["controlOnly"] is True, row
assert row["actAddKLCapEnabled"] is False, row
assert row["actAddDirectionMode"] == os.environ["DIRECTION"], row
assert row["actAddLayer"] == int(os.environ["LAYER"]), row
assert row["actAddAppliedCoefficient"] == float(os.environ["COEFFICIENT"]), row
assert len(row["actAddKLHistory"]) == row["actAdd"]["tokenCount"], row
assert math.isfinite(row["baseline"]["baseModelNLL"]), row
assert math.isfinite(row["actAdd"]["baseModelNLL"]), row
diagnostics = row["actAddDirectionDiagnostics"]
assert diagnostics["mode"] == os.environ["DIRECTION"], diagnostics
if os.environ["DIRECTION"] == "random-matched-norm":
    assert abs(diagnostics["appliedMatrixNorm"] - diagnostics["semanticMatrixNorm"]) <= 1e-4 * max(1, diagnostics["semanticMatrixNorm"]), diagnostics
print(
    os.path.basename(os.environ["REPORT"]),
    f"shift={row['actAdd']['topicScore'] - row['baseline']['topicScore']:+.6f}",
    f"tokens={row['actAdd']['tokenCount']}",
    f"nll={row['actAdd']['baseModelNLL']:.4f}",
)
PY
        done
      done
    done
  done
done
stop_app
trap - EXIT INT TERM
/usr/bin/python3 "$root_dir/Scripts/summarize_blocking_control.py"

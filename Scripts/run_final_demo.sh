#!/bin/zsh
set -euo pipefail

# Live end-to-end run of SteerDemo: build Release, run real inference once, capture a report, a
# snapshot and frames. This is the only script that exercises the real inference path end to end;
# Scripts/render_preserved_demo.sh replays a preserved packet and runs no inference at all.
#
# THIS SCRIPT WRITES ONLY TO AN UNTRACKED DIRECTORY, AND THAT IS LOAD-BEARING. It used to write
# docs/final-demo-run.json, docs/steerdemo.png, docs/demo-frames/*.png and docs/steerdemo.gif, and
# to `rm -f` the first three before starting. All four are committed evidence, so one run of this
# script destroyed them. Two are owned by something else entirely:
#
#   docs/steerdemo.png        the README hero, produced by Scripts/render_preserved_demo.sh from a
#                             preserved packet. README.md's caption names that packet by name
#                             (ocean-library) and discloses that it was chosen post hoc, so a live
#                             run overwriting the hero left the image contradicting its caption.
#   docs/final-demo-run.json  a historical packet README.md cites numerically and
#                             Scripts/analyze_judge_decomposition.py consumes.
#
# POINTING THIS SCRIPT BACK AT docs/final-demo-run.json WOULD NOT REFRESH THAT PACKET. It is not
# reproducible by any current build. It was produced at block 3 / nominal coefficient 12 under the
# greedy cumulative KL cap, before def3519 replaced the single-position `residualVector`
# construction with the front-aligned per-position `residualDirection`, and before
# matched-per-step became the default discipline. A run today differs in mechanism, not in seed.
# README.md discloses that provenance where it cites the packet; docs/kl-disciplines.md explains
# why cap-mode output is not a bit-level restoration either.
#
# NO CONTROLLER KNOB IS SET BELOW, DELIBERATELY. This script used to pin
# STEERDEMO_ACTADD_COEFFICIENT=12 and STEERDEMO_ACTADD_LAYER=3. Coefficient 12 is not an untested
# value, it is a tested and failed one: the worst semantic base-model NLL of all fifteen cells in
# the 180-packet blocking control, 3.6147 against a 0.9831 baseline. Commit b4a137e moved the app
# onto the validated cell (block 10 / coefficient 4) behind named constants for exactly that
# reason, and this script was not moved with it, which is how the failed cell survived. Layer 3 is
# worse still: only blocks 8, 10 and 12 were ever tested. Inheriting the app's defaults is what
# makes that drift impossible, because the values then live in exactly one place,
# DemoViewModel.validatedLayer and validatedCoefficient. Every other knob this script used to set
# (wedding, strength 14, 96 tokens) already equalled the app default and was pure duplication.
# STEERDEMO_KL_BUDGET=8 was worse than duplication: the budget is inert unless the discipline is
# the greedy cap, which is no longer the default, so setting it implied a cap that would not
# actually have been applied.
#
# Scripts/verify_readme_claims.py pins all of the above so it cannot regress silently.

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Release/SteerDemo.app"
executable="$app_path/Contents/MacOS/SteerDemo"

out_dir=${STEERDEMO_LIVE_OUT:-$root_dir/build/live-demo}
mkdir -p "$out_dir"
out_dir=${out_dir:A}

# Refuse to write anywhere git tracks. This is the guard that keeps the old failure mode dead even
# if someone repoints STEERDEMO_LIVE_OUT at docs/ to "regenerate" something.
if [[ -n "$(git -C "$root_dir" ls-files -- "$out_dir" 2>/dev/null)" ]]; then
  print -u2 "refusing to run: $out_dir holds git-tracked files."
  print -u2 "This script captures scratch evidence only; committed artifacts are not regenerable here."
  exit 1
fi

report="$out_dir/final-demo-run.json"
snapshot="$out_dir/steerdemo.png"
frames="$out_dir/frames"
gif="$out_dir/steerdemo.gif"

xcodebuild -quiet \
  -project "$root_dir/SteerDemo.xcodeproj" \
  -scheme SteerDemo \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$root_dir/DerivedData" \
  ONLY_ACTIVE_ARCH=YES \
  CODE_SIGNING_ALLOWED=NO \
  build

rm -f -- "$report" "$snapshot" "$gif"
rm -f -- "$frames"/*.png
mkdir -p "$frames"

open -F -n \
  --env STEERDEMO_AUTORUN=1 \
  --env STEERDEMO_REPORT_PATH="$report" \
  --env STEERDEMO_SNAPSHOT_PATH="$snapshot" \
  --env STEERDEMO_FRAMES_DIR="$frames" \
  "$app_path"

app_pid=""
for _ in {1..50}; do
  app_pid=$(pgrep -n -f "$executable" || true)
  [[ -n "$app_pid" ]] && break
  sleep 0.1
done
if [[ -z "$app_pid" ]]; then
  print -u2 "SteerDemo did not launch"
  exit 1
fi

stop_app() {
  if kill -0 "$app_pid" 2>/dev/null; then
    kill "$app_pid" 2>/dev/null || true
    wait "$app_pid" 2>/dev/null || true
  fi
}
trap stop_app EXIT INT TERM

for _ in {1..300}; do
  [[ -f "$report" && -f "$snapshot" && -f "$frames/99-final.png" ]] && break
  if ! kill -0 "$app_pid" 2>/dev/null; then
    wait "$app_pid" 2>/dev/null || true
    print -u2 "SteerDemo exited before writing final evidence"
    exit 1
  fi
  sleep 1
done

[[ -f "$report" && -f "$snapshot" && -f "$frames/99-final.png" ]] || {
  print -u2 "Timed out waiting for final evidence"
  exit 1
}
cmp "$snapshot" "$frames/99-final.png"
/usr/bin/python3 "$root_dir/Scripts/make_demo_gif.py" --frames "$frames" --output "$gif"
stop_app
trap - EXIT INT TERM
print "live run captured in $out_dir (untracked; the committed docs/ evidence was not touched)"

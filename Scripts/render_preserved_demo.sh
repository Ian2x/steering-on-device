#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Release/SteerDemo.app"
executable="$app_path/Contents/MacOS/SteerDemo"
# This packet is chosen, not drawn. It is the most favorable of the eight preserved
# teacher-forced packets, and README.md must keep disclosing that next to the hero image.
# Scripts/verify_readme_claims.py recomputes its ranks from this path.
report="$root_dir/docs/phase6/teacher-forced-comparison/runs/ocean-library.json"
snapshot="$root_dir/docs/steerdemo.png"
final_frame="$root_dir/docs/demo-frames/99-final.png"

xcodebuild -quiet \
  -project "$root_dir/SteerDemo.xcodeproj" \
  -scheme SteerDemo \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$root_dir/DerivedData" \
  ONLY_ACTIVE_ARCH=YES \
  CODE_SIGNING_ALLOWED=NO \
  build

rm -f -- "$snapshot" "$final_frame"
open -F -n \
  --env STEERDEMO_REPLAY_REPORT_PATH="$report" \
  --env STEERDEMO_SNAPSHOT_PATH="$snapshot" \
  --env STEERDEMO_FRAMES_DIR="$root_dir/docs/demo-frames" \
  --env STEERDEMO_HIDE_RATES=1 \
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

for _ in {1..100}; do
  [[ -f "$snapshot" && -f "$final_frame" ]] && break
  sleep 0.1
done
[[ -f "$snapshot" && -f "$final_frame" ]] || {
  print -u2 "Timed out waiting for preserved-report media"
  exit 1
}
cmp "$snapshot" "$final_frame"
stop_app
trap - EXIT INT TERM
print "rendered preserved report without running inference: $snapshot"

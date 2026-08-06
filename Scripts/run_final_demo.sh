#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
app_path="$root_dir/DerivedData/Build/Products/Release/SteerDemo.app"
executable="$app_path/Contents/MacOS/SteerDemo"
report="$root_dir/docs/final-demo-run.json"
snapshot="$root_dir/docs/steerdemo.png"
frames="$root_dir/docs/demo-frames"

xcodebuild -quiet \
  -project "$root_dir/SteerDemo.xcodeproj" \
  -scheme SteerDemo \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$root_dir/DerivedData" \
  ONLY_ACTIVE_ARCH=YES \
  CODE_SIGNING_ALLOWED=NO \
  build

rm -f -- "$report" "$snapshot"
rm -f -- "$frames"/*.png
mkdir -p "$frames"

open -F -n \
  --env STEERDEMO_AUTORUN=1 \
  --env STEERDEMO_LEXICON=wedding \
  --env STEERDEMO_BIAS_STRENGTH=14 \
  --env STEERDEMO_KL_BUDGET=8 \
  --env STEERDEMO_MAX_TOKENS=96 \
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

for _ in {1..180}; do
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
/usr/bin/python3 "$root_dir/Scripts/make_demo_gif.py"
stop_app
trap - EXIT INT TERM

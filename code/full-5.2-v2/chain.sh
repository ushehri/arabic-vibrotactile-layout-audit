#!/usr/bin/env bash
# Run the v2 native-acoustic pipeline (manuscript Section 4.2) end to end, one stage after another:
#   prep -> CTC alignment -> features (per source) -> analysis -> alignment spot-check figure -> report.
# Prerequisites: the datasets under $LMP_DATA (see the main README) and the full-5.2 stage outputs
# (tokens.csv, feats_mfcc.pkl, feats_ssl.npy) in $LMP_DATA/full52, which analysis_v2.py reads for L2-KSU and Halabi.
# Activate your Python environment first, or point PYTHON at its interpreter.
# All files are written to $LMP_DATA/full52-v2 (one log per stage).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LMP_DATA="${LMP_DATA:-$(cd "$HERE/../.." && pwd)/data}"
PY="${PYTHON:-python}"
WORK="$LMP_DATA/full52-v2"
mkdir -p "$WORK"
SOURCES="${SOURCES:-iqra everyayah cv}"

run() {   # run <log name> <script> [args...]
  local log="$WORK/$1.log"; shift
  echo "$(date +%H:%M) $*  (log: $log)"
  "$PY" "$@" > "$log" 2>&1
}

run prep "$HERE/prep.py"
for s in $SOURCES; do run "align_$s" "$HERE/align_v2.py" "$s"; done
for s in $SOURCES; do run "features_$s" "$HERE/features_v2.py" "$s"; done
run analysis_v2 "$HERE/analysis_v2.py"
run handcheck "$HERE/handcheck.py"
run report_v2 "$HERE/report_v2.py"
echo "$(date +%H:%M) done; outputs in $WORK"

#!/usr/bin/env bash
# Tennessee Eastman Process, classic Braatz d00..d21 files (public mirror).
set -e
DEST=${INDUSTRIAL_TS_DATA:-data/raw}/TEP
mkdir -p "$DEST"
[ -d "$DEST/braatz" ] || git clone --depth 1 https://github.com/camaramm/tennessee-eastman-profBraatz "$DEST/braatz"
echo "TEP files in $DEST/braatz ; export TEP_DIR=$DEST/braatz"

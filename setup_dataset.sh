#!/usr/bin/env bash
#
# setup_dataset.sh - Install the training data into data/ (data.hdf5 + train_labels.csv).
#
# Two sources are supported:
#
#   A) Local copy of the original ECE 4200 competition files (default if --from-local)
#        ./setup_dataset.sh --from-local "/Volumes/Seagate/Cornell/Spring_20/ECE4200/Kaggle"
#      Copies data.hdf5 + train_labels.csv from DIR into data/ and validates them.
#      No conversion needed: train (30000,1024,2), test (20000,1024,2), 10 classes.
#
#   B) Online: DeepSig RadioML 2018.01A from Kaggle (default when --from-local is absent)
#        ./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018
#      Downloads pinxau1000/radioml2018 (~18 GB zip -> ~21 GB HDF5), then converts
#      a subset (512 frames per class per SNR, SNR >= 0 dB by default) into data/.
#      The project's 10 classes are a subset of its 24.
#
# Usage:
#   ./setup_dataset.sh [options] [-- <extra args for process_radioml2018.py>]
#
# Options:
#   --from-local DIR   Use an existing data.hdf5 + train_labels.csv from DIR
#                      (DIR may also be the path to data.hdf5 itself)
#   --raw-dir DIR      (Kaggle) where the zip is downloaded/extracted
#                      (default: $DATA_DIR/raw). Needs ~45 GB free.
#   --data-dir DIR     Where data.hdf5/train_labels.csv end up (default: data)
#   --keep-zip         (Kaggle) keep the downloaded zip after extraction
#   --skip-download    (Kaggle) only convert an already-extracted HDF5
#   -h, --help         Show this help
#
# Environment:
#   PYTHON             Interpreter to use (default: ./venv/bin/python if present,
#                      else python3). The kaggle CLI is installed into it.
#   KAGGLE credentials ~/.kaggle/kaggle.json  (Kaggle -> Settings -> API ->
#                      "Create New Token"), or KAGGLE_USERNAME + KAGGLE_KEY,
#                      or KAGGLE_API_TOKEN.
#
# Examples:
#   ./setup_dataset.sh --from-local "/Volumes/Seagate/Cornell/Spring_20/ECE4200/Kaggle"
#   ./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018
#   ./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018 -- --frames-per-snr 1024 --min-snr -10
#   ./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018 --skip-download -- --dry-run
#
# Notes:
#   * The class order in RadioML's original classes.txt is wrong; the converter
#     uses classes-fixed.json from the Kaggle zip (or an identical built-in list).
#     See src/modulation_classifier/radioml2018.py.

set -euo pipefail

KAGGLE_DATASET="pinxau1000/radioml2018"
HDF5_NAME="GOLD_XYZ_OSC.0001_1024.hdf5"
REQUIRED_FREE_GB=45

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="data"
RAW_DIR=""
LOCAL_SRC=""
KEEP_ZIP=0
SKIP_DOWNLOAD=0

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --from-local)    LOCAL_SRC="${2:-}"; shift 2 ;;
        --raw-dir)       RAW_DIR="${2:-}"; shift 2 ;;
        --data-dir)      DATA_DIR="${2:-}"; shift 2 ;;
        --keep-zip)      KEEP_ZIP=1; shift ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; break ;;
        *)               die "Unknown option: $1 (use --help)" ;;
    esac
done
EXTRA_ARGS=("$@")
[ -z "$RAW_DIR" ] && RAW_DIR="$DATA_DIR/raw"

# ---------------------------------------------------------------------------
# Python interpreter
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
        PYTHON="$SCRIPT_DIR/venv/bin/python"
    else
        PYTHON="$(command -v python3 || true)"
    fi
fi
[ -x "$PYTHON" ] || die "Python interpreter not found (set PYTHON=...)"
"$PYTHON" -c "import modulation_classifier" 2>/dev/null \
    || die "modulation_classifier is not installed in $PYTHON. Run: $PYTHON -m pip install -e ."

validate_dataset() {
    # validate_dataset <data.hdf5> <train_labels.csv>
    "$PYTHON" - "$1" "$2" <<'EOF'
import sys
from modulation_classifier.dataset import validate_dataset_files
s = validate_dataset_files(sys.argv[1], sys.argv[2])
print(f"  train: {s['train_shape']}  {s['dtype']}")
if 'test_shape' in s:
    print(f"  test:  {s['test_shape']}")
counts = s['class_counts']
print("  labels:", ", ".join(f"{k}={v}" for k, v in counts.items()))
EOF
}

print_done() {
    echo
    echo "✅ Setup complete. Train with:"
    echo "   modulation-train --data $DATA_DIR/data.hdf5 --labels $DATA_DIR/train_labels.csv --epochs 50"
}

# ===========================================================================
# Source A: local copy of an existing dataset
# ===========================================================================
if [ -n "$LOCAL_SRC" ]; then
    echo "=== Dataset Setup (local copy) ==="
    if [ -f "$LOCAL_SRC" ]; then
        SRC_HDF5="$LOCAL_SRC"
        SRC_DIR="$(dirname "$LOCAL_SRC")"
    else
        SRC_DIR="$LOCAL_SRC"
        SRC_HDF5="$SRC_DIR/data.hdf5"
    fi
    SRC_CSV="$SRC_DIR/train_labels.csv"
    [ -f "$SRC_HDF5" ] || die "data.hdf5 not found at $SRC_HDF5"
    [ -f "$SRC_CSV" ]  || die "train_labels.csv not found at $SRC_CSV"
    echo "Source:     $SRC_DIR"
    echo "Output dir: $DATA_DIR"
    echo "Python:     $PYTHON"
    echo
    echo "Validating source files..."
    validate_dataset "$SRC_HDF5" "$SRC_CSV"

    mkdir -p "$DATA_DIR"
    DST_HDF5="$DATA_DIR/data.hdf5"
    DST_CSV="$DATA_DIR/train_labels.csv"
    if [ -e "$DST_HDF5" ] && [ "$(cd "$(dirname "$DST_HDF5")" && pwd)/data.hdf5" != "$(cd "$SRC_DIR" && pwd)/data.hdf5" ]; then
        echo "Note: replacing existing $DST_HDF5 ($(du -h "$DST_HDF5" | cut -f1))"
    fi
    echo "Copying data.hdf5 ($(du -h "$SRC_HDF5" | cut -f1)) and train_labels.csv..."
    cp "$SRC_HDF5" "$DST_HDF5"
    cp "$SRC_CSV" "$DST_CSV"
    "$PYTHON" - "$DATA_DIR/classes.txt" <<'EOF'
import sys
from modulation_classifier.dataset import MODULATION_CLASSES
open(sys.argv[1], "w").write("".join(c + "\n" for c in MODULATION_CLASSES))
EOF
    echo "Verifying copy..."
    validate_dataset "$DST_HDF5" "$DST_CSV"
    print_done
    exit 0
fi

# ===========================================================================
# Source B: Kaggle RadioML 2018.01A
# ===========================================================================
echo "=== RadioML 2018.01A Dataset Setup (Kaggle) ==="
echo "Raw dir:    $RAW_DIR"
echo "Output dir: $DATA_DIR"
echo "Python:     $PYTHON"
echo

find_hdf5() {
    [ -d "$RAW_DIR" ] && find "$RAW_DIR" -type f -name "$HDF5_NAME" 2>/dev/null | head -n 1
}
find_classes_json() {
    [ -d "$RAW_DIR" ] && find "$RAW_DIR" -type f -name "classes-fixed.json" 2>/dev/null | head -n 1
}

HDF5_PATH="$(find_hdf5 || true)"

if [ -n "$HDF5_PATH" ]; then
    echo "✓ Found existing dataset: $HDF5_PATH ($(du -h "$HDF5_PATH" | cut -f1))"
    echo "  Skipping download."
elif [ "$SKIP_DOWNLOAD" -eq 1 ]; then
    die "--skip-download given but $HDF5_NAME was not found under $RAW_DIR"
else
    # -----------------------------------------------------------------------
    # Kaggle CLI + credentials
    # -----------------------------------------------------------------------
    KAGGLE_BIN="$(dirname "$PYTHON")/kaggle"
    if [ ! -x "$KAGGLE_BIN" ]; then
        if command -v kaggle >/dev/null 2>&1; then
            KAGGLE_BIN="$(command -v kaggle)"
        else
            echo "Installing kaggle CLI into $PYTHON..."
            "$PYTHON" -m pip install --quiet kaggle
            [ -x "$KAGGLE_BIN" ] || KAGGLE_BIN="$(command -v kaggle || true)"
        fi
    fi
    [ -x "${KAGGLE_BIN:-}" ] || die "kaggle CLI not found after install"
    echo "Kaggle CLI: $KAGGLE_BIN"

    KAGGLE_JSON="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    if [ -f "$KAGGLE_JSON" ]; then
        echo "✓ Kaggle credentials: $KAGGLE_JSON"
        chmod 600 "$KAGGLE_JSON" 2>/dev/null || true
    elif [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
        echo "✓ Kaggle credentials: KAGGLE_USERNAME/KAGGLE_KEY"
    elif [ -n "${KAGGLE_API_TOKEN:-}" ]; then
        echo "✓ Kaggle credentials: KAGGLE_API_TOKEN"
    else
        cat >&2 <<EOF
ERROR: Kaggle API credentials not found.

  1. Sign in at https://www.kaggle.com, open Settings -> API
  2. Click "Create New Token" to download kaggle.json
  3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Then re-run this script.
EOF
        exit 1
    fi

    # -----------------------------------------------------------------------
    # Disk space check (zip + extracted HDF5 coexist briefly)
    # -----------------------------------------------------------------------
    mkdir -p "$RAW_DIR"
    FREE_GB="$(df -Pk "$RAW_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
    echo "Free space: ${FREE_GB} GB on the volume holding $RAW_DIR (need ~${REQUIRED_FREE_GB} GB)"
    if [ "$FREE_GB" -lt "$REQUIRED_FREE_GB" ]; then
        cat >&2 <<EOF
ERROR: Not enough free space. The download is ~18 GB and extracts to ~21 GB.
       Free up space or point --raw-dir at a larger volume, e.g.:
         ./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018
EOF
        exit 1
    fi

    # -----------------------------------------------------------------------
    # Download + extract
    # -----------------------------------------------------------------------
    echo
    echo "Downloading $KAGGLE_DATASET to $RAW_DIR (~18 GB, this takes a while)..."
    "$KAGGLE_BIN" datasets download -d "$KAGGLE_DATASET" -p "$RAW_DIR"

    ZIP_PATH="$(find "$RAW_DIR" -maxdepth 1 -type f -name '*.zip' 2>/dev/null | head -n 1 || true)"
    [ -n "$ZIP_PATH" ] || die "Download finished but no zip found in $RAW_DIR"
    echo "Extracting $(basename "$ZIP_PATH")..."
    unzip -q -o "$ZIP_PATH" -d "$RAW_DIR"
    if [ "$KEEP_ZIP" -eq 0 ]; then
        rm -f "$ZIP_PATH"
        echo "Removed zip."
    fi

    HDF5_PATH="$(find_hdf5 || true)"
    [ -n "$HDF5_PATH" ] || {
        echo "Contents of $RAW_DIR:" >&2
        find "$RAW_DIR" -maxdepth 3 >&2
        die "$HDF5_NAME not found after extraction"
    }
    echo "✓ Dataset extracted: $HDF5_PATH ($(du -h "$HDF5_PATH" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------
echo
echo "Converting to modulation-classifier format..."
CLASSES_JSON="$(find_classes_json || true)"
CMD=("$PYTHON" "$SCRIPT_DIR/process_radioml2018.py" --input "$HDF5_PATH" --output "$DATA_DIR")
if [ -n "$CLASSES_JSON" ]; then
    CMD+=(--classes-json "$CLASSES_JSON")
else
    echo "Note: classes-fixed.json not found under $RAW_DIR; using the built-in corrected class order."
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi
echo "+ ${CMD[*]}"
"${CMD[@]}"

print_done

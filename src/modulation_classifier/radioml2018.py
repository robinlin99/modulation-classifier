"""Convert the DeepSig RadioML 2018.01A dataset into the modulation-classifier format.

RadioML 2018.01A (Kaggle: ``pinxau1000/radioml2018``) ships as one HDF5 file,
``GOLD_XYZ_OSC.0001_1024.hdf5``, containing three datasets:

* ``X`` -- (2,555,904, 1024, 2) float32 I/Q frames (~21 GB)
* ``Y`` -- (2,555,904, 24) one-hot modulation labels
* ``Z`` -- (2,555,904, 1) SNR in dB

Frames are stored contiguously by modulation, then SNR:
24 modulations x 26 SNRs (-20..30 dB, step 2) x 4096 frames.

Two things make a naive conversion wrong or impractical:

1. **Class order.** The ``classes.txt`` bundled with the original release lists
   the modulations in the *wrong* order (a known issue, called out on the
   Kaggle page). The correct column order of ``Y`` is the one from the paper
   (O'Shea, Roy & Clancy, "Over-the-Air Deep Learning Based Radio Signal
   Classification", 2018, Table I), shipped on Kaggle as
   ``classes-fixed.json`` / ``classes-fixed.txt``. This module loads that file
   when it can find it and otherwise falls back to the identical built-in list.
   Passing the known-wrong order is rejected outright.
2. **Memory.** ``X`` does not fit in RAM on most machines, so frames are read
   one (modulation, SNR) block at a time (32 MB each) and written straight
   into the output file.

Output (in ``--output``, default ``data/``):

* ``data.hdf5`` with ``train`` / ``test`` (N, 1024, 2) float32 frames and
  ``train_snr`` / ``test_snr`` int16 per-frame SNR
* ``train_labels.csv`` / ``test_labels.csv`` with ``Id,Category,SNR``
* ``classes.txt``
"""

import argparse
import ast
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from .dataset import MODULATION_CLASSES

HDF5_FILENAME = "GOLD_XYZ_OSC.0001_1024.hdf5"
KAGGLE_DATASET = "pinxau1000/radioml2018"

#: Correct order of the one-hot columns in ``Y`` (paper Table I / classes-fixed.*).
RADIOML2018_CLASSES: List[str] = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

#: The scrambled order from the original ``classes.txt``. Using it silently
#: mislabels every frame, so we refuse it explicitly.
RADIOML2018_CLASSES_KNOWN_WRONG: List[str] = [
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK",
    "BPSK", "8PSK", "AM-SSB-SC", "4ASK", "16PSK", "64APSK", "128QAM",
    "128APSK", "AM-DSB-SC", "OOK", "16QAM", "64QAM", "256QAM", "AM-SSB-WC",
    "AM-DSB-WC", "QPSK",
]

CLASS_FILE_CANDIDATES = ("classes-fixed.json", "classes-fixed.txt")

# Largest contiguous span (rows) we are willing to slice-read for one block.
# A real (modulation, SNR) block is exactly 4096 rows; anything far beyond that
# means the file layout is unexpected and we fall back to fancy indexing.
_MAX_SLICE_SPAN = 65536


def _log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Locating inputs
# --------------------------------------------------------------------------- #

def find_dataset_file(search_root: Path = Path("data")) -> Optional[Path]:
    """Locate ``GOLD_XYZ_OSC.0001_1024.hdf5`` anywhere under *search_root*."""
    search_root = Path(search_root)
    if search_root.is_file():
        return search_root
    if not search_root.is_dir():
        return None
    matches = sorted(search_root.rglob(HDF5_FILENAME))
    return matches[0] if matches else None


def _parse_class_file(path: Path) -> List[str]:
    """Parse ``classes-fixed.json`` / ``classes-fixed.txt`` into a list of names."""
    text = path.read_text().strip()
    data = None
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        # classes-fixed.txt may be a Python list literal (optionally prefixed
        # with ``classes =``) or one class per line.
        body = text.split("=", 1)[1] if "=" in text.splitlines()[0] else text
        try:
            data = ast.literal_eval(body.strip())
        except (ValueError, SyntaxError):
            data = [line.strip().strip(",'\"") for line in text.splitlines() if line.strip()]

    if isinstance(data, dict):
        # Accept {"0": "OOK", ...} or {"OOK": 0, ...}
        if all(isinstance(v, int) for v in data.values()):
            data = [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
        else:
            data = [data[k] for k in sorted(data, key=lambda k: int(k))]
    return [str(c).strip() for c in data]


def validate_class_names(names: Sequence[str], source: str,
                         target_classes: Sequence[str] = MODULATION_CLASSES) -> None:
    """Raise ``ValueError`` if *names* cannot be the ``Y`` column order."""
    names = list(names)
    if names == RADIOML2018_CLASSES_KNOWN_WRONG:
        raise ValueError(
            f"{source}: this is the known-incorrect order from the original "
            f"classes.txt. Use classes-fixed.json/.txt from the Kaggle dataset "
            f"(or omit --classes-json to use the built-in corrected order)."
        )
    if len(names) != len(RADIOML2018_CLASSES):
        raise ValueError(
            f"{source}: expected {len(RADIOML2018_CLASSES)} class names, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{source}: class names contain duplicates")
    missing = [c for c in target_classes if c not in names]
    if missing:
        raise ValueError(f"{source}: target classes not present in class list: {missing}")


def load_class_names(classes_file: Optional[Path] = None,
                     hdf5_path: Optional[Path] = None) -> Tuple[List[str], str]:
    """Resolve the ``Y`` column order.

    Priority: explicit *classes_file* -> ``classes-fixed.json``/``.txt`` found
    next to (or up to two directories above) the HDF5 file -> built-in list.

    Returns ``(class_names, source_description)``.
    """
    candidates: List[Path] = []
    if classes_file is not None:
        candidates.append(Path(classes_file))
    elif hdf5_path is not None:
        hdf5_path = Path(hdf5_path)
        seen = set()
        for directory in (hdf5_path.parent, hdf5_path.parent.parent, hdf5_path.parent.parent.parent):
            if directory in seen:
                continue
            seen.add(directory)
            for name in CLASS_FILE_CANDIDATES:
                candidates.append(directory / name)

    for candidate in candidates:
        if candidate.is_file():
            names = _parse_class_file(candidate)
            validate_class_names(names, str(candidate))
            return names, str(candidate)

    if classes_file is not None:
        raise FileNotFoundError(f"Class file not found: {classes_file}")
    return list(RADIOML2018_CLASSES), "built-in corrected order (classes-fixed)"


# --------------------------------------------------------------------------- #
# Selecting frames
# --------------------------------------------------------------------------- #

@dataclass
class FrameBlock:
    """A set of selected frames sharing one (target class, SNR)."""
    indices: np.ndarray  # sorted global row indices into X
    label: int           # index into the target class list
    snr: int


def read_labels_and_snr(f: h5py.File, chunk_rows: int = 131072) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(class_index, snr)`` per frame without loading ``X``.

    ``Y`` is reduced with ``argmax`` in chunks so peak memory stays small even
    though the full one-hot matrix is ~0.5 GB.
    """
    y, z = f["Y"], f["Z"]
    n = y.shape[0]
    labels = np.empty(n, dtype=np.int16)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        labels[start:stop] = np.argmax(y[start:stop], axis=1)
    snr = np.asarray(z[()]).reshape(-1).astype(np.int16)
    if snr.shape[0] != n:
        raise ValueError(f"Y has {n} rows but Z has {snr.shape[0]}")
    return labels, snr


def select_frames(
    labels: np.ndarray,
    snr: np.ndarray,
    class_names: Sequence[str],
    min_snr: int = 0,
    max_snr: int = 30,
    frames_per_snr: int = 512,
    rng: Optional[np.random.Generator] = None,
    target_classes: Sequence[str] = MODULATION_CLASSES,
) -> List[FrameBlock]:
    """Pick frames for each (target class, SNR) pair.

    Args:
        labels: Per-frame index into *class_names* (from ``read_labels_and_snr``).
        snr: Per-frame SNR in dB.
        class_names: Column order of ``Y`` (see ``load_class_names``).
        min_snr, max_snr: Inclusive SNR range to keep.
        frames_per_snr: Random frames to keep per (class, SNR); ``0`` keeps all.
        rng: Random generator used for subsampling.
        target_classes: Output class list; its order defines the output labels.
    """
    rng = rng or np.random.default_rng(0)
    snr_values = [int(s) for s in np.unique(snr) if min_snr <= s <= max_snr]
    blocks: List[FrameBlock] = []
    for target_idx, name in enumerate(target_classes):
        src_idx = list(class_names).index(name)
        class_mask = labels == src_idx
        for s in snr_values:
            idx = np.flatnonzero(class_mask & (snr == s))
            if idx.size == 0:
                continue
            if frames_per_snr > 0 and idx.size > frames_per_snr:
                idx = np.sort(rng.choice(idx, size=frames_per_snr, replace=False))
            blocks.append(FrameBlock(indices=idx, label=target_idx, snr=s))
    return blocks


def _read_block(x: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read the rows *indices* from ``X`` efficiently.

    In the real file a block is a contiguous run, so a single slice read plus
    an in-memory subset is far faster than HDF5 point selection.
    """
    lo, hi = int(indices[0]), int(indices[-1])
    span = hi - lo + 1
    if span <= max(_MAX_SLICE_SPAN, 2 * indices.size):
        return x[lo:hi + 1][indices - lo]
    return x[indices]  # increasing fancy index; slow but correct


def _scatter_write(dset: h5py.Dataset, rows: np.ndarray, frames: np.ndarray) -> None:
    """Write *frames* to the (unsorted) destination *rows* of *dset*."""
    if rows.size == 0:
        return
    order = np.argsort(rows)
    dset[rows[order]] = frames[order]


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def _print_selection_summary(blocks: List[FrameBlock], frame_bytes: int,
                             target_classes: Sequence[str]) -> Dict[str, int]:
    per_class: Dict[str, int] = {name: 0 for name in target_classes}
    snr_set = set()
    for b in blocks:
        per_class[target_classes[b.label]] += int(b.indices.size)
        snr_set.add(b.snr)
    total = sum(per_class.values())

    _log("Selection:")
    for name in target_classes:
        count = per_class[name]
        flag = "" if count else "   <-- no frames!"
        _log(f"  {name:10s} {count:>9,}{flag}")
    snr_desc = f"{min(snr_set)}..{max(snr_set)} dB ({len(snr_set)} values)" if snr_set else "none"
    _log(f"  {'total':10s} {total:>9,}   SNR {snr_desc}   "
         f"~{total * frame_bytes / 1e9:.2f} GB of frames")
    return per_class


def convert(
    input_path: Path,
    output_dir: Path = Path("data"),
    classes_file: Optional[Path] = None,
    min_snr: int = 0,
    max_snr: int = 30,
    frames_per_snr: int = 512,
    train_frac: float = 0.8,
    seed: int = 42,
    dry_run: bool = False,
    target_classes: Sequence[str] = MODULATION_CLASSES,
) -> Dict[str, object]:
    """Convert RadioML 2018.01A into ``data.hdf5`` + label CSVs.

    See the module docstring for the output layout. Returns a summary dict
    with ``n_train``, ``n_test``, ``per_class`` and output paths.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"RadioML HDF5 not found: {input_path}")
    if not 0.0 < train_frac <= 1.0:
        raise ValueError("train_frac must be in (0, 1]")

    class_names, class_source = load_class_names(classes_file, input_path)
    _log(f"Input:        {input_path}")
    _log(f"Class order:  {class_source}")
    rng = np.random.default_rng(seed)

    with h5py.File(input_path, "r") as f:
        for key in ("X", "Y", "Z"):
            if key not in f:
                raise ValueError(f"{input_path} has no dataset '{key}' - is this RadioML 2018.01A?")
        x = f["X"]
        frame_shape = tuple(x.shape[1:])
        frame_bytes = int(np.prod(frame_shape)) * 4
        _log(f"Frames:       {x.shape[0]:,} x {frame_shape} (reading labels/SNR...)")

        t0 = time.perf_counter()
        labels, snr = read_labels_and_snr(f)
        _log(f"              labels/SNR read in {time.perf_counter() - t0:.1f}s")

        blocks = select_frames(labels, snr, class_names, min_snr, max_snr,
                               frames_per_snr, rng, target_classes)
        per_class = _print_selection_summary(blocks, frame_bytes, target_classes)
        n_selected = int(sum(b.indices.size for b in blocks))
        if n_selected == 0:
            raise ValueError("No frames selected - check --min-snr/--max-snr and the class list")
        empty = [name for name, count in per_class.items() if count == 0]
        if empty:
            raise ValueError(f"No frames found for classes {empty}; the class order is probably wrong")

        # Shuffle once, then assign every selected frame a destination split/row.
        all_labels = np.concatenate([np.full(b.indices.size, b.label, dtype=np.int16) for b in blocks])
        all_snr = np.concatenate([np.full(b.indices.size, b.snr, dtype=np.int16) for b in blocks])
        perm = rng.permutation(n_selected)
        n_train = int(round(train_frac * n_selected))
        n_test = n_selected - n_train
        is_train = np.zeros(n_selected, dtype=bool)
        is_train[perm[:n_train]] = True
        dest_row = np.empty(n_selected, dtype=np.int64)
        dest_row[perm[:n_train]] = np.arange(n_train)
        dest_row[perm[n_train:]] = np.arange(n_test)

        train_labels = np.empty(n_train, dtype=np.int16)
        train_snr = np.empty(n_train, dtype=np.int16)
        test_labels = np.empty(n_test, dtype=np.int16)
        test_snr = np.empty(n_test, dtype=np.int16)
        train_labels[dest_row[is_train]] = all_labels[is_train]
        train_snr[dest_row[is_train]] = all_snr[is_train]
        test_labels[dest_row[~is_train]] = all_labels[~is_train]
        test_snr[dest_row[~is_train]] = all_snr[~is_train]

        _log(f"Split:        {n_train:,} train / {n_test:,} test (train_frac={train_frac}, seed={seed})")
        summary: Dict[str, object] = {
            "n_train": n_train, "n_test": n_test, "per_class": per_class,
            "class_order_source": class_source,
        }
        if dry_run:
            _log("Dry run - nothing written.")
            return summary

        output_dir.mkdir(parents=True, exist_ok=True)
        hdf5_path = output_dir / "data.hdf5"
        _log(f"Writing:      {hdf5_path}")
        t0 = time.perf_counter()
        with h5py.File(hdf5_path, "w") as out:
            # Contiguous layout (no chunks/compression) so the scattered row
            # writes below are cheap and the trainer's whole-array read is fast.
            train_ds = out.create_dataset("train", shape=(n_train,) + frame_shape, dtype=np.float32)
            test_ds = out.create_dataset("test", shape=(n_test,) + frame_shape, dtype=np.float32)
            out.create_dataset("train_snr", data=train_snr)
            out.create_dataset("test_snr", data=test_snr)
            out.attrs["classes"] = json.dumps(list(target_classes))
            out.attrs["source"] = str(input_path)
            out.attrs["source_class_order"] = json.dumps(list(class_names))
            out.attrs["min_snr"] = min_snr
            out.attrs["max_snr"] = max_snr
            out.attrs["frames_per_snr"] = frames_per_snr
            out.attrs["seed"] = seed

            offset = 0
            current_label = None
            for b in blocks:
                if b.label != current_label:
                    if current_label is not None:
                        _log(f"  {target_classes[current_label]:10s} done ({time.perf_counter() - t0:.0f}s)")
                    current_label = b.label
                frames = _read_block(x, b.indices).astype(np.float32, copy=False)
                sel = slice(offset, offset + b.indices.size)
                blk_train = is_train[sel]
                blk_rows = dest_row[sel]
                _scatter_write(train_ds, blk_rows[blk_train], frames[blk_train])
                _scatter_write(test_ds, blk_rows[~blk_train], frames[~blk_train])
                offset += b.indices.size
            if current_label is not None:
                _log(f"  {target_classes[current_label]:10s} done ({time.perf_counter() - t0:.0f}s)")

    train_csv = output_dir / "train_labels.csv"
    test_csv = output_dir / "test_labels.csv"
    classes_txt = output_dir / "classes.txt"
    pd.DataFrame({
        "Id": np.arange(n_train),
        "Category": [target_classes[i] for i in train_labels],
        "SNR": train_snr,
    }).to_csv(train_csv, index=False)
    pd.DataFrame({
        "Id": np.arange(n_test),
        "Category": [target_classes[i] for i in test_labels],
        "SNR": test_snr,
    }).to_csv(test_csv, index=False)
    classes_txt.write_text("".join(f"{c}\n" for c in target_classes))

    _log()
    _log("Done:")
    _log(f"  {hdf5_path}  (train: {n_train:,}, test: {n_test:,}, "
         f"{hdf5_path.stat().st_size / 1e9:.2f} GB)")
    _log(f"  {train_csv}")
    _log(f"  {test_csv}")
    _log(f"  {classes_txt}")
    summary.update({"hdf5": hdf5_path, "train_labels": train_csv,
                    "test_labels": test_csv, "classes": classes_txt})
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``process_radioml2018.py``."""
    parser = argparse.ArgumentParser(
        prog="process_radioml2018.py",
        description="Convert RadioML 2018.01A (Kaggle pinxau1000/radioml2018) into the "
                    "modulation-classifier format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None,
                        help=f"Path to {HDF5_FILENAME} (default: search under data/)")
    parser.add_argument("--output", type=Path, default=Path("data"),
                        help="Output directory for data.hdf5 and label CSVs")
    parser.add_argument("--classes-json", dest="classes_file", type=Path, default=None,
                        help="classes-fixed.json/.txt giving the Y column order "
                             "(default: auto-detect next to the HDF5, else built-in)")
    parser.add_argument("--min-snr", type=int, default=0, help="Minimum SNR (dB) to keep")
    parser.add_argument("--max-snr", type=int, default=30, help="Maximum SNR (dB) to keep")
    parser.add_argument("--frames-per-snr", type=int, default=512,
                        help="Random frames to keep per (class, SNR); 0 keeps all 4096")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of frames for train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling/split")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be selected; write nothing")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point. Returns a process exit code (0 on success)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    input_path = args.input or find_dataset_file(Path("data"))
    if input_path is None or not Path(input_path).is_file():
        _log(f"RadioML HDF5 not found{'' if args.input is None else f': {args.input}'}.")
        _log(f"Run ./setup_dataset.sh to download {KAGGLE_DATASET}, or pass --input "
             f"/path/to/{HDF5_FILENAME}")
        return 1

    try:
        summary = convert(
            input_path=Path(input_path),
            output_dir=args.output,
            classes_file=args.classes_file,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            frames_per_snr=args.frames_per_snr,
            train_frac=args.train_frac,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _log(f"Error processing dataset: {exc}")
        traceback.print_exc()
        return 1

    if not args.dry_run:
        _log()
        _log("Ready to train:")
        _log(f"  modulation-train --data {summary['hdf5']} --labels {summary['train_labels']} --epochs 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())

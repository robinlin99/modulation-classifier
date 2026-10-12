"""Tests for the RadioML 2018.01A converter (modulation_classifier.radioml2018)."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from modulation_classifier import IQDataset
from modulation_classifier.dataset import MODULATION_CLASSES
from modulation_classifier.radioml2018 import (
    RADIOML2018_CLASSES,
    RADIOML2018_CLASSES_KNOWN_WRONG,
    convert,
    load_class_names,
    main,
    select_frames,
    validate_class_names,
)

SNRS = list(range(-20, 31, 2))  # 26 values, as in the real dataset
N_CLASSES = len(RADIOML2018_CLASSES)


def _encode(class_idx: int, snr: int) -> float:
    """Value stored in every sample of a frame so tests can recover (class, SNR)."""
    return class_idx + snr / 100.0


def _decode(value: float):
    class_idx = int(round(value))
    snr = int(round((value - class_idx) * 100))
    return class_idx, snr


def make_fake_radioml(path, frames_per_block=6, timesteps=8):
    """Write a miniature file with the real RadioML 2018.01A layout.

    Frames are contiguous by (modulation, SNR) and each frame is filled with a
    value encoding its true class index and SNR, so the mapping can be checked
    after conversion. ``Y`` columns follow the *corrected* class order.
    """
    n = N_CLASSES * len(SNRS) * frames_per_block
    X = np.empty((n, timesteps, 2), dtype=np.float32)
    Y = np.zeros((n, N_CLASSES), dtype=np.float32)
    Z = np.empty((n, 1), dtype=np.int64)
    i = 0
    for c in range(N_CLASSES):
        for s in SNRS:
            X[i:i + frames_per_block] = _encode(c, s)
            Y[i:i + frames_per_block, c] = 1.0
            Z[i:i + frames_per_block] = s
            i += frames_per_block
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Z", data=Z)
    return path


@pytest.fixture
def fake_dataset(tmp_path):
    raw = tmp_path / "raw" / "2018.01"
    raw.mkdir(parents=True)
    hdf5 = make_fake_radioml(raw / "GOLD_XYZ_OSC.0001_1024.hdf5")
    # Kaggle ships classes-fixed.json at the zip root, one level above 2018.01/
    (tmp_path / "raw" / "classes-fixed.json").write_text(json.dumps(RADIOML2018_CLASSES))
    return hdf5


class TestClassOrder:
    def test_builtin_order_is_used_when_no_file(self, tmp_path):
        names, source = load_class_names(None, tmp_path / "nothing.hdf5")
        assert names == RADIOML2018_CLASSES
        assert "built-in" in source

    def test_auto_detects_classes_fixed_json_above_hdf5(self, fake_dataset):
        names, source = load_class_names(None, fake_dataset)
        assert names == RADIOML2018_CLASSES
        assert source.endswith("classes-fixed.json")

    def test_parses_txt_list_literal(self, tmp_path):
        txt = tmp_path / "classes-fixed.txt"
        txt.write_text("classes = " + repr(RADIOML2018_CLASSES) + "\n")
        names, _ = load_class_names(txt)
        assert names == RADIOML2018_CLASSES

    def test_parses_txt_one_per_line(self, tmp_path):
        txt = tmp_path / "classes-fixed.txt"
        txt.write_text("\n".join(RADIOML2018_CLASSES) + "\n")
        names, _ = load_class_names(txt)
        assert names == RADIOML2018_CLASSES

    def test_rejects_known_wrong_order(self, tmp_path):
        bad = tmp_path / "classes.json"
        bad.write_text(json.dumps(RADIOML2018_CLASSES_KNOWN_WRONG))
        with pytest.raises(ValueError, match="known-incorrect"):
            load_class_names(bad)

    def test_rejects_list_missing_target_class(self):
        names = [c if c != "OOK" else "2FSK" for c in RADIOML2018_CLASSES]
        with pytest.raises(ValueError, match="OOK"):
            validate_class_names(names, "test")

    def test_all_target_classes_exist_in_radioml2018(self):
        assert set(MODULATION_CLASSES) <= set(RADIOML2018_CLASSES)

    def test_missing_explicit_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_class_names(tmp_path / "missing.json")


class TestSelectFrames:
    def test_snr_filter_and_subsample(self):
        rng = np.random.default_rng(0)
        # 2 frames per (class, SNR)
        labels = np.repeat(np.arange(N_CLASSES), len(SNRS) * 2).astype(np.int16)
        snr = np.tile(np.repeat(SNRS, 2), N_CLASSES).astype(np.int16)
        blocks = select_frames(labels, snr, RADIOML2018_CLASSES, min_snr=10, max_snr=20,
                               frames_per_snr=1, rng=rng)
        # 10 target classes x SNRs {10,12,...,20} = 6 values x 1 frame
        assert len(blocks) == 10 * 6
        assert all(b.indices.size == 1 for b in blocks)
        assert {b.snr for b in blocks} == {10, 12, 14, 16, 18, 20}
        assert {b.label for b in blocks} == set(range(10))

    def test_zero_keeps_all(self):
        labels = np.repeat(np.arange(N_CLASSES), len(SNRS) * 3).astype(np.int16)
        snr = np.tile(np.repeat(SNRS, 3), N_CLASSES).astype(np.int16)
        blocks = select_frames(labels, snr, RADIOML2018_CLASSES, min_snr=-20, max_snr=30,
                               frames_per_snr=0)
        assert sum(b.indices.size for b in blocks) == 10 * len(SNRS) * 3


class TestConvert:
    def test_end_to_end_labels_and_snr_are_correct(self, fake_dataset, tmp_path):
        out = tmp_path / "out"
        summary = convert(fake_dataset, out, min_snr=0, frames_per_snr=4, seed=1)

        # 10 classes x 16 SNRs (0..30) x 4 frames
        assert summary["n_train"] + summary["n_test"] == 10 * 16 * 4
        assert summary["n_train"] == 512
        assert summary["class_order_source"].endswith("classes-fixed.json")

        with h5py.File(out / "data.hdf5", "r") as f:
            train = f["train"][()]
            test = f["test"][()]
            train_snr = f["train_snr"][()]
            test_snr = f["test_snr"][()]
            assert json.loads(f.attrs["classes"]) == list(MODULATION_CLASSES)
        train_df = pd.read_csv(out / "train_labels.csv")
        test_df = pd.read_csv(out / "test_labels.csv")
        assert list(train_df.columns) == ["Id", "Category", "SNR"]
        assert len(train_df) == len(train) == len(train_snr)
        assert len(test_df) == len(test) == len(test_snr)

        for frames, df, snrs in ((train, train_df, train_snr), (test, test_df, test_snr)):
            for frame, category, snr_csv, snr_h5 in zip(frames, df["Category"], df["SNR"], snrs):
                # Every sample in the frame carries the same encoded value
                assert np.allclose(frame, frame[0, 0])
                true_class, true_snr = _decode(float(frame[0, 0]))
                assert RADIOML2018_CLASSES[true_class] == category
                assert true_snr == snr_csv == snr_h5
                assert true_snr >= 0

        # Output is shuffled, not grouped by class
        assert train_df["Category"].iloc[:50].nunique() > 1

        # Loads through the training pipeline with labels in MODULATION_CLASSES order
        ds = IQDataset(str(out / "data.hdf5"), str(out / "train_labels.csv"))
        assert len(ds) == len(train)
        assert sorted(set(ds.labels.tolist())) == list(range(len(MODULATION_CLASSES)))
        for idx in range(0, len(ds), 37):
            sample, label = ds[idx]
            true_class, _ = _decode(float(sample[0, 0, 0]))
            assert MODULATION_CLASSES[int(label)] == RADIOML2018_CLASSES[true_class]

        assert (out / "classes.txt").read_text().split() == list(MODULATION_CLASSES)

    def test_keep_all_frames(self, fake_dataset, tmp_path):
        summary = convert(fake_dataset, tmp_path / "out", min_snr=-20, frames_per_snr=0)
        assert summary["n_train"] + summary["n_test"] == 10 * len(SNRS) * 6

    def test_dry_run_writes_nothing(self, fake_dataset, tmp_path):
        out = tmp_path / "out"
        summary = convert(fake_dataset, out, dry_run=True)
        assert summary["n_train"] > 0
        assert not out.exists()

    def test_wrong_class_order_is_detected_by_empty_classes(self, fake_dataset, tmp_path):
        # A plausible-looking but wrong order where one target class maps to a
        # column that (in this fake file) has no frames at the chosen SNRs.
        with h5py.File(fake_dataset, "r+") as f:
            # Relabel the one-hot rows of true column 0 (OOK) at SNR >= 0 as
            # 4ASK so that selecting "OOK" yields nothing.
            z = f["Z"][()].reshape(-1)
            y = f["Y"][()]
            col0 = (np.argmax(y, axis=1) == 0) & (z >= 0)
            y[col0] = 0.0
            y[col0, 1] = 1.0
            f["Y"][...] = y
        with pytest.raises(ValueError, match="OOK"):
            convert(fake_dataset, tmp_path / "out")


class TestCli:
    def test_main_missing_input_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert main([]) == 1

    def test_main_runs_with_explicit_input(self, fake_dataset, tmp_path):
        out = tmp_path / "out"
        rc = main(["--input", str(fake_dataset), "--output", str(out),
                   "--frames-per-snr", "2", "--min-snr", "20"])
        assert rc == 0
        assert (out / "data.hdf5").exists()
        assert (out / "train_labels.csv").exists()

    def test_main_auto_finds_dataset_under_data(self, tmp_path, monkeypatch):
        # With no --input, the CLI searches ./data recursively (the setup
        # script extracts to data/raw/ by default)
        raw = tmp_path / "data" / "raw" / "2018.01"
        raw.mkdir(parents=True)
        make_fake_radioml(raw / "GOLD_XYZ_OSC.0001_1024.hdf5", frames_per_block=2)
        monkeypatch.chdir(tmp_path)
        assert main(["--dry-run"]) == 0

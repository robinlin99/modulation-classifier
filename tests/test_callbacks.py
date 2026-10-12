"""Tests for training callbacks and TensorBoard logging."""

import numpy as np
import pytest

from modulation_classifier import ModulationClassifier, IQDataset
from modulation_classifier.callbacks import (
    CallbackList,
    TensorBoardLogger,
    TrainingCallback,
    default_run_name,
)
from modulation_classifier.dataset import MODULATION_CLASSES


class RecordingCallback(TrainingCallback):
    """Records every hook invocation."""

    def __init__(self):
        self.begin = []
        self.batches = []
        self.epochs = []
        self.end = []

    def on_train_begin(self, config):
        self.begin.append(config)

    def on_batch_end(self, step, metrics):
        self.batches.append((step, metrics))

    def on_epoch_end(self, epoch, metrics):
        self.epochs.append((epoch, metrics))

    def on_train_end(self, history):
        self.end.append(history)


def _small_dataset(n_samples=80, timesteps=1024):
    data = np.random.randn(n_samples, timesteps, 2).astype(np.float32)
    labels = np.arange(n_samples) % 10  # balanced for the stratified split
    return IQDataset(data, labels, MODULATION_CLASSES)


def _train(tmp_path, **kwargs):
    clf = ModulationClassifier()
    return clf, clf.train(
        dataset=_small_dataset(),
        num_epochs=kwargs.pop("num_epochs", 2),
        batch_size=kwargs.pop("batch_size", 8),
        use_augmentation=False,
        patience=kwargs.pop("patience", 10),
        save_path=str(tmp_path / "model.pt"),
        **kwargs,
    )


class TestCallbackHook:
    def test_hooks_are_called_with_expected_payloads(self, tmp_path):
        cb = RecordingCallback()
        # 80 samples, val_split 0.2 -> 64 train / batch 8 -> 8 batches per epoch
        _, history = _train(tmp_path, num_epochs=2, batch_size=8, log_interval=2, callbacks=[cb])

        # on_train_begin: once, with the config
        assert len(cb.begin) == 1
        config = cb.begin[0]
        assert config["num_epochs"] == 2
        assert config["batch_size"] == 8
        assert config["batches_per_epoch"] == 8
        assert config["train_samples"] == 64
        assert config["val_samples"] == 16
        assert config["architecture"] == "improved"
        assert config["parameters"] > 0

        # on_batch_end: every log_interval batches -> batches 2,4,6,8 per epoch,
        # with a global step that keeps increasing across epochs
        steps = [s for s, _ in cb.batches]
        assert steps == [2, 4, 6, 8, 10, 12, 14, 16]
        first = cb.batches[0][1]
        assert first["epoch"] == 1 and first["batch"] == 2 and first["num_batches"] == 8
        assert {"loss", "accuracy", "batch_loss"} <= set(first)
        assert 0 <= first["accuracy"] <= 100

        # on_epoch_end: once per epoch, matching the returned history
        assert [e for e, _ in cb.epochs] == [1, 2]
        for (epoch, metrics), tl, vl in zip(cb.epochs, history["train_loss"], history["val_loss"]):
            assert metrics["train_loss"] == pytest.approx(tl)
            assert metrics["val_loss"] == pytest.approx(vl)
            assert {"train_acc", "val_acc", "lr", "seconds", "improved"} <= set(metrics)
        assert cb.epochs[0][1]["improved"] is True  # first epoch is always a new best

        # on_train_end: once, with the history
        assert len(cb.end) == 1
        assert cb.end[0] is history

    def test_last_batch_always_reported(self, tmp_path):
        cb = RecordingCallback()
        # 8 batches/epoch with log_interval=3 -> batches 3, 6 and 8 (last)
        _train(tmp_path, num_epochs=1, batch_size=8, log_interval=3, callbacks=[cb])
        assert [m["batch"] for _, m in cb.batches] == [3, 6, 8]

    def test_log_interval_zero_disables_batch_hooks(self, tmp_path):
        cb = RecordingCallback()
        _train(tmp_path, num_epochs=1, log_interval=0, callbacks=[cb])
        assert cb.batches == []
        assert len(cb.epochs) == 1

    def test_on_train_end_called_when_training_raises(self, tmp_path):
        class Boom(TrainingCallback):
            def on_epoch_end(self, epoch, metrics):
                raise RuntimeError("boom")

        cb = RecordingCallback()
        clf = ModulationClassifier()
        with pytest.raises(RuntimeError, match="boom"):
            clf.train(_small_dataset(), num_epochs=3, batch_size=8, use_augmentation=False,
                      save_path=str(tmp_path / "m.pt"), callbacks=[Boom(), cb])
        assert len(cb.end) == 1
        # history up to the failure is preserved
        assert len(clf.training_history["train_loss"]) == 1
        assert clf.trained is False

    def test_callback_list_fans_out(self):
        a, b = RecordingCallback(), RecordingCallback()
        cl = CallbackList([a, b])
        assert len(cl) == 2
        cl.on_train_begin({"x": 1})
        cl.on_batch_end(1, {"loss": 0.5})
        cl.on_epoch_end(1, {"val_loss": 0.4})
        cl.on_train_end({})
        for cb in (a, b):
            assert cb.begin == [{"x": 1}]
            assert cb.batches == [(1, {"loss": 0.5})]
            assert cb.epochs == [(1, {"val_loss": 0.4})]
            assert cb.end == [{}]

    def test_default_run_name(self):
        name = default_run_name("improved")
        assert name.endswith("_improved")
        assert len(name.split("_")[0]) == len("20261012-004916")
        assert default_run_name() == default_run_name().split("_")[0]


def _read_scalars(run_dir):
    """Return {tag: [(step, value), ...]} from a TensorBoard event directory."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0, "tensors": 0})
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


class TestTensorBoardLogger:
    def test_writes_train_and_val_runs_with_expected_tags(self, tmp_path):
        run_dir = tmp_path / "runs" / "exp"
        _train(tmp_path, num_epochs=2, batch_size=8, log_interval=2,
               callbacks=[TensorBoardLogger(str(run_dir))])

        assert (run_dir / "train").is_dir() and (run_dir / "val").is_dir()
        assert list((run_dir / "train").glob("events.out.tfevents.*"))
        assert list((run_dir / "val").glob("events.out.tfevents.*"))

        train = _read_scalars(run_dir / "train")
        val = _read_scalars(run_dir / "val")

        # Epoch-level: both sub-runs share the same tags so TensorBoard overlays them
        for tag in ("epoch/loss", "epoch/accuracy"):
            assert [s for s, _ in train[tag]] == [1, 2]
            assert [s for s, _ in val[tag]] == [1, 2]
        assert [s for s, _ in train["epoch/learning_rate"]] == [1, 2]
        assert train["epoch/learning_rate"][0][1] == pytest.approx(0.001)
        assert all(v > 0 for _, v in train["epoch/seconds"])

        # Batch-level: global steps 2..8 (epoch 1) and 10..16 (epoch 2), train only
        expected_steps = [2, 4, 6, 8, 10, 12, 14, 16]
        assert [s for s, _ in train["batch/loss"]] == expected_steps
        assert [s for s, _ in train["batch/accuracy"]] == expected_steps
        assert [s for s, _ in train["batch/loss_instant"]] == expected_steps
        assert "batch/loss" not in val

    def test_log_dir_shorthand(self, tmp_path):
        run_dir = tmp_path / "runs" / "shorthand"
        _train(tmp_path, num_epochs=1, log_dir=str(run_dir))
        assert _read_scalars(run_dir / "val")["epoch/loss"]

    def test_hparams_text_logged(self, tmp_path):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        run_dir = tmp_path / "runs" / "hp"
        _train(tmp_path, num_epochs=1, log_dir=str(run_dir))
        acc = EventAccumulator(str(run_dir / "train"))
        acc.Reload()
        text_tags = acc.Tags().get("tensors", [])
        assert any(t.startswith("hparams") for t in text_tags)

    def test_close_is_idempotent(self, tmp_path):
        logger = TensorBoardLogger(str(tmp_path / "x"))
        logger.close()
        logger.close()
        with TensorBoardLogger(str(tmp_path / "y")) as ctx_logger:
            assert "y" in repr(ctx_logger)

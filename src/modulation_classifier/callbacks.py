"""Training callbacks (live metric logging).

``ModulationClassifier.train()`` accepts a list of :class:`TrainingCallback`
objects and invokes them at well-defined points. The main built-in callback is
:class:`TensorBoardLogger`, which streams loss/accuracy to TensorBoard so you
can watch training live in a browser::

    from modulation_classifier import ModulationClassifier, IQDataset
    from modulation_classifier.callbacks import TensorBoardLogger

    dataset = IQDataset("data/data.hdf5", "data/train_labels.csv")
    clf = ModulationClassifier()
    clf.train(dataset, num_epochs=50, callbacks=[TensorBoardLogger("runs/exp1")])
    # or simply: clf.train(dataset, num_epochs=50, log_dir="runs/exp1")

Then, in another terminal::

    tensorboard --logdir runs

Logging layout (one TensorBoard "run" per training run, with ``train`` and
``val`` sub-runs so that train and validation curves overlay on one chart):

* ``epoch/loss``, ``epoch/accuracy`` -- per epoch, in both ``train`` and ``val``
* ``epoch/learning_rate``, ``epoch/seconds`` -- per epoch, in ``train``
* ``batch/loss``, ``batch/accuracy`` -- running averages every ``log_interval``
  batches (x-axis = global batch step), in ``train``
* ``batch/loss_instant`` -- the loss of the most recent batch, in ``train``
* ``hparams`` (Text tab) -- the training configuration
"""

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class TrainingCallback:
    """Base class for training callbacks. Override any subset of the hooks.

    All ``metrics``/``config`` arguments are plain dicts of JSON-serialisable
    values so callbacks can forward them anywhere (TensorBoard, files, W&B...).
    """

    def on_train_begin(self, config: Dict[str, Any]) -> None:
        """Called once before the first epoch with the training configuration."""

    def on_batch_end(self, step: int, metrics: Dict[str, Any]) -> None:
        """Called every ``log_interval`` batches (and on the last batch of an epoch).

        Args:
            step: Global batch index across epochs (1-based).
            metrics: ``epoch``, ``batch``, ``num_batches``, ``loss`` (running
                mean for the epoch so far), ``accuracy`` (running, in percent),
                ``batch_loss`` (this batch only).
        """

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called after validation each epoch.

        Args:
            epoch: 1-based epoch number.
            metrics: ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``,
                ``lr``, ``seconds``, ``improved`` (bool: new best val loss).
        """

    def on_train_end(self, history: Dict[str, List[float]]) -> None:
        """Called once when training finishes (normally or via early stopping).

        Also called if training raises, so resources can be released.
        """


class CallbackList(TrainingCallback):
    """Fan a hook call out to several callbacks."""

    def __init__(self, callbacks: Optional[Iterable[TrainingCallback]] = None):
        self.callbacks: List[TrainingCallback] = list(callbacks or [])

    def __len__(self) -> int:
        return len(self.callbacks)

    def on_train_begin(self, config: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(config)

    def on_batch_end(self, step: int, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(step, metrics)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)

    def on_train_end(self, history: Dict[str, List[float]]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(history)


def default_run_name(prefix: str = "") -> str:
    """``YYYYmmdd-HHMMSS[_prefix]`` -- a unique, sortable run directory name."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{prefix}" if prefix else stamp


class TensorBoardLogger(TrainingCallback):
    """Stream training metrics to TensorBoard.

    Args:
        log_dir: Directory for this run (e.g. ``runs/20261012-004916_improved``).
            ``train/`` and ``val/`` sub-directories are created inside it.
        flush_secs: How often the event files are flushed to disk. Epoch-end and
            batch-end hooks also flush explicitly so the browser updates promptly.

    Requires the ``tensorboard`` package (``pip install tensorboard``).
    """

    def __init__(self, log_dir: str, flush_secs: int = 10):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "TensorBoard logging requires the 'tensorboard' package: "
                "pip install tensorboard"
            ) from exc

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._train = SummaryWriter(str(self.log_dir / "train"), flush_secs=flush_secs)
        self._val = SummaryWriter(str(self.log_dir / "val"), flush_secs=flush_secs)
        self._closed = False

    # -- hooks ------------------------------------------------------------- #

    def on_train_begin(self, config: Dict[str, Any]) -> None:
        rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(config.items()))
        self._train.add_text("hparams", f"| parameter | value |\n|---|---|\n{rows}", 0)
        self._train.flush()

    def on_batch_end(self, step: int, metrics: Dict[str, Any]) -> None:
        self._train.add_scalar("batch/loss", metrics["loss"], step)
        self._train.add_scalar("batch/accuracy", metrics["accuracy"], step)
        if "batch_loss" in metrics:
            self._train.add_scalar("batch/loss_instant", metrics["batch_loss"], step)
        self._train.flush()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        self._train.add_scalar("epoch/loss", metrics["train_loss"], epoch)
        self._train.add_scalar("epoch/accuracy", metrics["train_acc"], epoch)
        self._val.add_scalar("epoch/loss", metrics["val_loss"], epoch)
        self._val.add_scalar("epoch/accuracy", metrics["val_acc"], epoch)
        if "lr" in metrics:
            self._train.add_scalar("epoch/learning_rate", metrics["lr"], epoch)
        if "seconds" in metrics:
            self._train.add_scalar("epoch/seconds", metrics["seconds"], epoch)
        self._train.flush()
        self._val.flush()

    def on_train_end(self, history: Dict[str, List[float]]) -> None:
        self.close()

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        if not self._closed:
            self._train.close()
            self._val.close()
            self._closed = True

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"TensorBoardLogger(log_dir={str(self.log_dir)!r})"

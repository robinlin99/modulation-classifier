#!/usr/bin/env python3
"""Worked examples of the modulation-classifier Python API.

Each mode is a self-contained function you can read on its own::

    python examples/train_comprehensive.py basic        # train with defaults
    python examples/train_comprehensive.py monitor      # TensorBoard + a custom callback
    python examples/train_comprehensive.py custom       # train your own nn.Module
    python examples/train_comprehensive.py sweep        # small hyper-parameter sweep
    python examples/train_comprehensive.py eval         # evaluate + write a submission CSV

Common options::

    --data data/data.hdf5 --labels data/train_labels.csv   (defaults, relative to the repo root)
    --epochs N            cap the number of epochs (default: mode-specific)
    --no-augmentation     skip phase-rotation augmentation (3x faster)
    --model PATH          checkpoint to write (train modes) or read (eval)

Run `./setup_dataset.sh ...` or `python create_dummy_data.py` first so that
``data/`` exists. Everything written by these examples goes to ``examples/out/``.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn

from modulation_classifier import (
    IQDataset,
    ModulationClassifier,
    TensorBoardLogger,
    TrainingCallback,
)
from modulation_classifier.dataset import MODULATION_CLASSES

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------- #
# 1. Basic training
# --------------------------------------------------------------------------- #

def basic_training(args) -> ModulationClassifier:
    """Train the default model with default settings."""
    dataset = IQDataset(args.data, args.labels)
    print(f"{len(dataset)} labelled frames, classes: {dataset.classes}")

    clf = ModulationClassifier()                      # "improved" net, device auto-detected
    history = clf.train(
        dataset,
        num_epochs=args.epochs or 20,
        batch_size=64,
        learning_rate=1e-3,
        use_augmentation=not args.no_augmentation,
        save_path=str(args.model),
        log_dir=str(OUT_DIR / "runs" / "basic"),      # TensorBoard
    )
    print(f"best val acc: {max(history['val_acc']):.2f}%  -> {args.model}")
    clf.plot_training_curves(save_path=str(OUT_DIR / "basic_curves.png"))
    return clf


# --------------------------------------------------------------------------- #
# 2. Monitoring: TensorBoard + a custom callback
# --------------------------------------------------------------------------- #

class EpochTimer(TrainingCallback):
    """Prints a running ETA and keeps the best epoch; shows the hook contract."""

    def __init__(self):
        self.num_epochs = None
        self.start = None
        self.best = (None, float("inf"))

    def on_train_begin(self, config):
        self.num_epochs = config["num_epochs"]
        self.start = time.perf_counter()
        print(f"[EpochTimer] {config['train_samples']} train / {config['val_samples']} val frames, "
              f"{config['batches_per_epoch']} batches/epoch on {config['device']}")

    def on_epoch_end(self, epoch, metrics):
        if metrics["val_loss"] < self.best[1]:
            self.best = (epoch, metrics["val_loss"])
        elapsed = time.perf_counter() - self.start
        eta = elapsed / epoch * (self.num_epochs - epoch)
        print(f"[EpochTimer] epoch {epoch}/{self.num_epochs} done, ETA {eta/60:.1f} min"
              + ("  (new best)" if metrics["improved"] else ""))

    def on_train_end(self, history):
        print(f"[EpochTimer] best epoch {self.best[0]} (val loss {self.best[1]:.4f}); "
              f"{len(history['val_loss'])} epochs in {(time.perf_counter() - self.start)/60:.1f} min")


def monitored_training(args) -> ModulationClassifier:
    """Same training, with explicit callbacks instead of the ``log_dir`` shorthand."""
    dataset = IQDataset(args.data, args.labels)
    clf = ModulationClassifier()
    run_dir = OUT_DIR / "runs" / "monitor"
    print(f"TensorBoard: tensorboard --logdir {OUT_DIR / 'runs'}")

    clf.train(
        dataset,
        num_epochs=args.epochs or 20,
        use_augmentation=not args.no_augmentation,
        save_path=str(args.model),
        log_interval=25,                               # batch-level points every 25 batches
        callbacks=[TensorBoardLogger(str(run_dir)), EpochTimer()],
    )
    return clf


# --------------------------------------------------------------------------- #
# 3. Custom architecture
# --------------------------------------------------------------------------- #

class SmallCNN(nn.Module):
    """A deliberately small network: ~40K parameters, trains in a fraction of the time.

    Any module mapping ``(N, 1, T, 2)`` to ``(N, num_classes)`` logits works.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 1), padding=(3, 0)), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(32, 64, kernel_size=(5, 2), padding=(2, 0)), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def custom_model_training(args) -> ModulationClassifier:
    """Replace the classifier's model before training."""
    dataset = IQDataset(args.data, args.labels)
    clf = ModulationClassifier()
    clf.model = SmallCNN(num_classes=len(MODULATION_CLASSES)).to(clf.device)
    n_params = sum(p.numel() for p in clf.model.parameters())
    print(f"SmallCNN: {n_params:,} parameters")

    clf.train(
        dataset,
        num_epochs=args.epochs or 15,
        use_augmentation=not args.no_augmentation,
        save_path=str(args.model),
        log_dir=str(OUT_DIR / "runs" / "custom"),
    )
    # NB: to reload this checkpoint later, build a classifier, assign SmallCNN() again,
    # then call load_model() -- the architecture is not stored in the file.
    return clf


# --------------------------------------------------------------------------- #
# 4. Hyper-parameter sweep
# --------------------------------------------------------------------------- #

def hyperparameter_sweep(args) -> dict:
    """Short runs over a few settings; compare them side by side in TensorBoard."""
    dataset = IQDataset(args.data, args.labels)
    grid = [
        dict(learning_rate=1e-3, batch_size=64, weight_decay=1e-4),
        dict(learning_rate=5e-4, batch_size=64, weight_decay=1e-4),
        dict(learning_rate=1e-3, batch_size=128, weight_decay=1e-3),
    ]
    epochs = args.epochs or 8
    results = []
    for params in grid:
        name = "_".join(f"{k}{v}" for k, v in params.items())
        print(f"\n=== {name} ===")
        clf = ModulationClassifier()
        history = clf.train(
            dataset,
            num_epochs=epochs,
            patience=4,
            use_augmentation=not args.no_augmentation,
            save_path=str(OUT_DIR / f"sweep_{name}.pt"),
            log_dir=str(OUT_DIR / "runs" / "sweep" / name),
            **params,
        )
        results.append({**params, "best_val_acc": max(history["val_acc"]),
                        "best_val_loss": min(history["val_loss"]), "epochs": len(history["val_loss"])})

    table = pd.DataFrame(results).sort_values("best_val_acc", ascending=False)
    print("\n", table.to_string(index=False))
    table.to_csv(OUT_DIR / "sweep_results.csv", index=False)
    print(f"\nCompare runs: tensorboard --logdir {OUT_DIR / 'runs' / 'sweep'}")
    return table.iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# 5. Evaluation and submission
# --------------------------------------------------------------------------- #

def evaluate_model(args) -> dict:
    """Evaluate a saved checkpoint and write a Kaggle-style submission for the 'test' split."""
    import h5py

    if not Path(args.model).exists():
        sys.exit(f"checkpoint not found: {args.model} (run the 'basic' mode first)")

    clf = ModulationClassifier(model_architecture=args.architecture)
    clf.load_model(str(args.model))

    # Accuracy and per-class metrics on the labelled frames
    labelled = IQDataset(args.data, args.labels)
    metrics = clf.evaluate(labelled)
    print(f"accuracy on {len(labelled)} labelled frames: {metrics['accuracy']:.4f}")
    report = metrics["classification_report"]
    for name in clf.classes:
        m = report[name]
        print(f"  {name:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f} n={int(m['support'])}")
    clf.plot_confusion_matrix(labelled, save_path=str(OUT_DIR / "confusion_matrix.png"))

    # Predictions for the unlabelled 'test' split, if the file has one.
    # (IQDataset prefers 'train' when both exist, so read 'test' explicitly.)
    with h5py.File(args.data, "r") as f:
        test_frames = f["test"][()] if "test" in f else None
    if test_frames is None:
        print("no 'test' dataset in the file; skipping submission")
        return metrics

    names = clf.predict_classes(test_frames)
    submission = pd.DataFrame({"Id": np.arange(len(names)), "Category": names})
    path = OUT_DIR / "submission.csv"
    submission.to_csv(path, index=False)
    print(f"wrote {path} ({len(names)} rows); predicted class counts:")
    print(submission["Category"].value_counts().to_string())
    return metrics


# --------------------------------------------------------------------------- #

MODES = {
    "basic": basic_training,
    "monitor": monitored_training,
    "custom": custom_model_training,
    "sweep": hyperparameter_sweep,
    "eval": evaluate_model,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "data.hdf5")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "train_labels.csv")
    parser.add_argument("--epochs", type=int, default=None, help="override the mode's default epoch count")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--model", type=Path, default=None,
                        help="checkpoint path (default: examples/out/<mode>_model.pt; eval reads it)")
    parser.add_argument("--architecture", default="improved", choices=["improved", "legacy"],
                        help="(eval) architecture the checkpoint was trained with")
    args = parser.parse_args(argv)

    for path in (args.data, args.labels):
        if not path.exists():
            sys.exit(f"{path} not found. Run ./setup_dataset.sh or python create_dummy_data.py first.")
    # IQDataset distinguishes file paths from arrays by `isinstance(x, str)`
    args.data, args.labels = str(args.data), str(args.labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.model is None:
        args.model = OUT_DIR / ("basic_model.pt" if args.mode == "eval" else f"{args.mode}_model.pt")

    MODES[args.mode](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Modulation Classifier

A PyTorch framework for classifying radio modulation schemes from raw
in-phase/quadrature (IQ) time series. Give it an HDF5 file of `(N, 1024, 2)`
frames and a CSV of labels; it trains a compact CNN, streams loss and accuracy
to TensorBoard while it runs, and produces predictions, confusion matrices and
Kaggle-style submission files.

The project grew out of the Cornell ECE 4200 modulation-prediction Kaggle
competition and recognises these ten classes:

| Class | Modulation |
|---|---|
| `FM` | Frequency modulation |
| `OQPSK` | Offset quadrature phase-shift keying |
| `BPSK` | Binary phase-shift keying |
| `8PSK` | 8-ary phase-shift keying |
| `AM-SSB-SC` | AM, single sideband, suppressed carrier |
| `4ASK` | 4-ary amplitude-shift keying |
| `16PSK` | 16-ary phase-shift keying |
| `AM-DSB-SC` | AM, double sideband, suppressed carrier |
| `QPSK` | Quadrature phase-shift keying |
| `OOK` | On-off keying |

## In a nutshell

```bash
pip install -e .                                   # install (see Installation)
./setup_dataset.sh --from-local /path/to/ece4200   # or the Kaggle download (see Data)
modulation-train --data data/data.hdf5 --labels data/train_labels.csv --epochs 50
tensorboard --logdir runs                          # live curves at http://localhost:6006
modulation-predict --model best_model.pt --data data/data.hdf5 --output predictions.csv
```

Or from Python:

```python
from modulation_classifier import ModulationClassifier, IQDataset

dataset = IQDataset("data/data.hdf5", "data/train_labels.csv")
clf = ModulationClassifier()                       # picks CUDA > Apple MPS > CPU
clf.train(dataset, num_epochs=50, log_dir="runs/exp1")
print(clf.predict_classes(dataset)[:10])
```

## What's inside

- **Model** – a 4-layer CNN (`ImprovedNet`, ~537K parameters) with per-layer
  batch-norm, global average pooling and dropout; the original competition
  network (`LegacyNet`, ~10.9M parameters) is kept for comparison.
- **Training loop** – stratified train/validation split, 90°/180° phase-rotation
  augmentation, AdamW with `ReduceLROnPlateau`, early stopping, best-checkpoint
  saving, and a callback hook.
- **Monitoring** – readable console output even when stdout is a pipe, plus
  TensorBoard logging of batch- and epoch-level metrics out of the box.
- **Data tooling** – one script that installs the training data from a local
  copy of the competition files or from the public RadioML 2018.01A dataset on
  Kaggle (with the dataset's known class-order bug corrected).
- **Evaluation** – accuracy, per-class precision/recall/F1, confusion matrices,
  constellation plots, probability outputs.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
data
training
monitoring
evaluation
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli
api/index
development
changelog
```

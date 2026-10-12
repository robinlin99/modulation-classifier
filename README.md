# Modulation Classifier 📡

A PyTorch framework for classifying radio modulation schemes from raw IQ
(in-phase/quadrature) time series. Feed it an HDF5 file of `(N, 1024, 2)`
frames and a CSV of labels; it trains a compact CNN, streams loss and accuracy
to TensorBoard while it runs, and produces predictions, confusion matrices and
Kaggle-style submission files.

It recognises ten modulation types — `FM`, `OQPSK`, `BPSK`, `8PSK`,
`AM-SSB-SC`, `4ASK`, `16PSK`, `AM-DSB-SC`, `QPSK`, `OOK` — the classes of the
Cornell ECE 4200 modulation-prediction Kaggle competition the project grew out
of.

**Documentation:** build it locally with `make docs` and open
`docs/_build/html/index.html` (user guide, CLI and API reference,
troubleshooting). Section links below point at the Markdown sources.

## Quick start

### 1. Install

```bash
git clone https://github.com/robinlin99/modulation-classifier.git
cd modulation-classifier
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

Requires Python ≥ 3.8. Training auto-selects CUDA, then Apple MPS, then CPU.
→ [docs/installation.md](docs/installation.md)

### 2. Get data

`setup_dataset.sh` installs `data/data.hdf5` + `data/train_labels.csv` from
either source:

```bash
# A) a local copy of the original ECE 4200 competition files (no conversion needed)
./setup_dataset.sh --from-local "/Volumes/Seagate/Cornell/Spring_20/ECE4200/Kaggle"

# B) the public RadioML 2018.01A dataset from Kaggle (~18 GB download; needs a Kaggle API token)
./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018

# or, just to smoke-test the install:
python create_dummy_data.py
```

The RadioML path streams the 21 GB file without loading it into RAM and uses
the dataset's *corrected* class order (the bundled `classes.txt` is wrong).
→ [docs/data.md](docs/data.md)

### 3. Train

```bash
modulation-train --data data/data.hdf5 --labels data/train_labels.csv --epochs 50
```

Stratified validation split, 90°/180° phase-rotation augmentation, AdamW with
LR-on-plateau, early stopping, best checkpoint saved to `best_model.pt`. About
3.5 min/epoch on an M1 Pro with augmentation; `--no-augmentation` is 3× faster.
→ [docs/training.md](docs/training.md)

```python
from modulation_classifier import ModulationClassifier, IQDataset

dataset = IQDataset("data/data.hdf5", "data/train_labels.csv")
clf = ModulationClassifier()
clf.train(dataset, num_epochs=50, log_dir="runs/exp1")
print(clf.predict_classes(dataset)[:10])
```

### 4. Watch it train

```bash
tensorboard --logdir runs        # or: make tensorboard  → http://localhost:6006
```

Every run logs train/val loss and accuracy per epoch (overlaid), running loss
and accuracy per batch, learning rate, and its hyperparameters. The console
shows per-epoch lines plus a progress bar in a terminal or plain progress lines
when output is piped (IDE runners, log files).
→ [docs/monitoring.md](docs/monitoring.md)

### 5. Predict and evaluate

```bash
modulation-predict --model best_model.pt --data data/data.hdf5 --output predictions.csv
modulation-predict --model best_model.pt --data data/data.hdf5 \
    --labels data/test_labels.csv --evaluate --probabilities --plots-dir plots/
```

→ [docs/evaluation.md](docs/evaluation.md)

## Model

The default `improved` architecture is a 4-layer CNN — `Conv2d`+`BatchNorm`+ReLU
blocks of 256/256/128/80 channels (the second mixes the I and Q columns),
global average pooling, dropout, and a small linear head — **536 826
parameters**. The original competition network (`--architecture legacy`,
10 881 226 parameters, shared batch-norm, flattened 81 280-wide linear layer)
is kept for comparison. Swap in your own `nn.Module` by assigning
`classifier.model`.

## Project layout

```text
src/modulation_classifier/   package: classifier, dataset, models, callbacks, radioml2018, utils, cli
tests/                       pytest suite (~10 s)
examples/                    runnable scripts + the original competition notebook
docs/                        Sphinx documentation (make docs)
setup_dataset.sh             install data from a local copy or Kaggle
process_radioml2018.py       RadioML 2018.01A converter
create_dummy_data.py         synthetic data for smoke tests
data/, runs/                 datasets and TensorBoard logs (git-ignored)
```

## Development

```bash
pip install -e ".[dev,docs]"   # or: make install-dev
make test                      # pytest with coverage
make lint && make format       # flake8 + mypy, black + isort
make docs && make docs-serve   # build the docs, serve on :8000
```

→ [docs/development.md](docs/development.md) ·
[docs/troubleshooting.md](docs/troubleshooting.md) ·
[CHANGELOG.md](CHANGELOG.md)

## Contributing

1. Fork the repository and create a feature branch
2. Add tests for behaviour changes and keep `make test`, `make lint` and
   `make docs` green
3. Open a pull request

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{modulation_classifier,
  author = {Robin Lin},
  title  = {Modulation Classifier: Deep Learning for Radio Modulation Classification},
  url    = {https://github.com/robinlin/modulation-classifier},
  version = {0.1.0},
  year   = {2024}
}
```

## Acknowledgments

- Built with PyTorch, scikit-learn and TensorBoard
- Grew out of the Cornell ECE 4200 Kaggle competition
- RadioML 2018.01A by DeepSig Inc. (CC BY-NC-SA 4.0), mirrored on Kaggle by
  pinxau1000

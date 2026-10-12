# Examples

All scripts default to `data/data.hdf5` and `data/train_labels.csv` relative
to the repository root (run `./setup_dataset.sh ...` or
`python create_dummy_data.py` first) and write their outputs — checkpoints,
plots, CSVs, TensorBoard runs — to `examples/out/` (git-ignored).

| Script / mode | What it shows |
|---|---|
| `example_usage.py --minimal` | 5-epoch smoke test on random data; needs no files. `make example` |
| `example_usage.py` | The whole workflow: explore the data, constellation plots, train with augmentation and TensorBoard, evaluate, confusion matrix, predict the `test` split and write a submission CSV. `make example-full` |
| `train_comprehensive.py basic` | Plain `ModulationClassifier.train()` with defaults and `log_dir=` |
| `train_comprehensive.py monitor` | `TensorBoardLogger` plus a hand-written `TrainingCallback` (ETA / best epoch) — the callback contract in practice |
| `train_comprehensive.py custom` | Training your own `nn.Module` by assigning `classifier.model` |
| `train_comprehensive.py sweep` | A small learning-rate / batch-size / weight-decay sweep with one TensorBoard run per setting and a results table |
| `train_comprehensive.py eval` | Load a checkpoint, report accuracy and per-class metrics, plot a confusion matrix, write `submission.csv` for the unlabelled `test` split |

Common flags for `train_comprehensive.py`: `--epochs N`, `--no-augmentation`,
`--data`, `--labels`, `--model`. For example, a quick end-to-end check:

```bash
python examples/train_comprehensive.py basic --epochs 2 --no-augmentation
python examples/train_comprehensive.py eval
tensorboard --logdir examples/out/runs
```

## `notebooks/notebook.ipynb`

The original ECE 4200 Kaggle competition notebook (the "FIXED" revision with
proper augmentation, per-layer batch-norm and the improved training protocol).
It predates the package and is kept for reference only — it expects the
competition files in its working directory and does not use the
`modulation_classifier` API.

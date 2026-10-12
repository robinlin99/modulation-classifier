# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **TensorBoard logging** on by default in `modulation-train`
  (`runs/<timestamp>_<arch>/` with `train`/`val` sub-runs; `--log-dir`,
  `--run-name`, `--no-tensorboard`). Logs epoch loss/accuracy/LR/time and
  batch-level loss/accuracy every `--log-interval` batches, plus the run's
  hyperparameters as text.
- **Callback hook** in `ModulationClassifier.train()`: `callbacks=[...]` and the
  `log_dir=` shorthand; `TrainingCallback` base class with `on_train_begin`,
  `on_batch_end`, `on_epoch_end`, `on_train_end`; `TensorBoardLogger`
  implementation. `on_train_end` runs even if training raises.
- **RadioML 2018.01A converter** (`modulation_classifier.radioml2018`,
  `process_radioml2018.py`): streams the 21 GB file one (class, SNR) block at a
  time, uses the corrected class order (`classes-fixed.json` from the Kaggle
  zip or an identical built-in list) and rejects the dataset's known-wrong
  `classes.txt` order, subsamples per (class, SNR) (`--frames-per-snr`,
  default 512) within an SNR range (`--min-snr`, default 0), writes a shuffled
  train/test split with per-frame SNR (`train_snr`/`test_snr` datasets and an
  `SNR` CSV column), and supports `--dry-run`.
- `setup_dataset.sh` rewritten: `--from-local DIR` copies and validates an
  existing `data.hdf5` + `train_labels.csv` pair (e.g. the original ECE 4200
  competition files); the default Kaggle path checks credentials and free
  space, is idempotent, uses the project venv, and forwards converter
  arguments after `--`.
- `validate_dataset_files()` in `modulation_classifier.dataset` — cheap
  shape/row-count/class-name consistency check for an HDF5/CSV pair.
- Apple **MPS auto-detection** (`get_default_device()`: CUDA > MPS > CPU) and
  `--device mps` on the CLI.
- `--log-interval` on `modulation-train`; batch-level progress lines when
  output is not a terminal.
- `count_parameters`, `TrainingCallback` and `TensorBoardLogger` exported from
  the package root.
- Sphinx documentation site under `docs/` (`make docs`, `make docs-serve`),
  `make tensorboard`, this changelog, and runnable examples.

### Fixed
- **No training output when stdout is piped** (IDE/agent runners, `nohup`,
  redirects): the CLI now switches stdout/stderr to line buffering, all status
  prints flush, and the `tqdm` bar is replaced by plain progress lines when
  stderr is not a TTY.
- Per-epoch summary now includes learning rate and elapsed time; CLI errors
  print a traceback instead of a one-line message.
- `examples/train_comprehensive.py` called APIs that do not exist
  (`dropout_rate=`, `min_delta=`, `lr_scheduler=`, `verbose=`, `model=`, …);
  rewritten against the real API. `examples/example_usage.py` now resolves
  `data/` relative to the repository root.
- Test suite: `count_parameters` was imported but not exported (whole module
  failed to collect); training tests used random labels and could fail the
  stratified split.
- `ImprovedNet` docstring/README claimed ~300K parameters; it has 536 826
  (`LegacyNet`: 10 881 226).

### Changed
- `cli.py` parsers are built by `build_train_parser()` /
  `build_predict_parser()` (used by the docs and tests); behaviour unchanged.
- `tensorboard` is a regular dependency; the `docs` extra is now
  `sphinx`, `sphinx-rtd-theme`, `myst-parser`, `sphinx-argparse`.
- README reorganised as a landing page that links into the documentation.

## [0.1.0]

Initial package: `ModulationClassifier`, `IQDataset`, `ImprovedNet` /
`LegacyNet`, phase-rotation augmentation, early stopping, `modulation-train` /
`modulation-predict` CLIs, plotting utilities.

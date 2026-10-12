# Command-line reference

The argument tables below are generated from the parsers in the code, so they
stay in sync with `--help`.

## modulation-train

```{argparse}
:module: modulation_classifier.cli
:func: build_train_parser
:prog: modulation-train
:nodefaultconst:
```

## modulation-predict

```{argparse}
:module: modulation_classifier.cli
:func: build_predict_parser
:prog: modulation-predict
:nodefaultconst:
```

## process_radioml2018.py

Converts RadioML 2018.01A into the project's format; see
[Data](data.md#what-the-converter-does) for what it does and why.

```{argparse}
:module: modulation_classifier.radioml2018
:func: build_parser
:prog: process_radioml2018.py
:nodefaultconst:
```

## setup_dataset.sh

Installs `data/data.hdf5` and `data/train_labels.csv` from a local copy of the
competition files or from Kaggle. Not a Python program, so documented here by
hand (`./setup_dataset.sh --help` prints the same information).

```text
./setup_dataset.sh [options] [-- <extra args for process_radioml2018.py>]

  --from-local DIR   Use an existing data.hdf5 + train_labels.csv from DIR
                     (DIR may also be the path to data.hdf5 itself)
  --raw-dir DIR      (Kaggle) where the zip is downloaded/extracted
                     (default: $DATA_DIR/raw). Needs ~45 GB free.
  --data-dir DIR     Where data.hdf5/train_labels.csv end up (default: data)
  --keep-zip         (Kaggle) keep the downloaded zip after extraction
  --skip-download    (Kaggle) only convert an already-extracted HDF5
  -h, --help         Show help

Environment:
  PYTHON             Interpreter to use (default: ./venv/bin/python if present,
                     else python3). The kaggle CLI is installed into it.
  Kaggle credentials ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY,
                     or KAGGLE_API_TOKEN.
```

Examples:

```bash
./setup_dataset.sh --from-local "/Volumes/Seagate/Cornell/Spring_20/ECE4200/Kaggle"
./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018
./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018 -- --frames-per-snr 1024 --min-snr -10
./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018 --skip-download -- --dry-run
```

## create_dummy_data.py

Takes no arguments. Writes a synthetic `data/data.hdf5` (5000 `train`,
2000 `test` frames), `data/train_labels.csv` and `data/classes.txt`.

## Make targets

| Target | Action |
|---|---|
| `make install` / `make install-dev` | `pip install -e .` / with `[dev,docs]` extras |
| `make test` / `make test-fast` | pytest with coverage / fail-fast |
| `make lint` / `make format` / `make format-check` | flake8 + mypy / black + isort |
| `make tensorboard` | `tensorboard --logdir runs` |
| `make docs` / `make docs-serve` / `make docs-clean` | build this site / serve it on :8000 / remove the build |
| `make example` / `make example-full` | run `examples/example_usage.py` (minimal / full) |
| `make clean` | remove build artefacts and caches |

# Development

## Project layout

```text
modulation-classifier/
├── src/modulation_classifier/
│   ├── classifier.py      ModulationClassifier: train / predict / evaluate / save / load
│   ├── dataset.py         IQDataset, augmentation, train/val loaders, validate_dataset_files
│   ├── models.py          ImprovedNet, LegacyNet, create_model, count_parameters
│   ├── callbacks.py       TrainingCallback hook, TensorBoardLogger
│   ├── radioml2018.py     RadioML 2018.01A -> project format converter
│   ├── utils.py           plotting and metric helpers
│   └── cli.py             modulation-train / modulation-predict entry points
├── tests/                 pytest suite (unit tests + converter/callback integration tests)
├── examples/              runnable scripts and the original competition notebook
├── docs/                  this documentation (Sphinx + MyST)
├── setup_dataset.sh       install data from a local copy or Kaggle
├── process_radioml2018.py thin wrapper around modulation_classifier.radioml2018
├── create_dummy_data.py   synthetic dataset for smoke tests
├── data/                  datasets (git-ignored)
└── runs/                  TensorBoard logs (git-ignored)
```

## Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev,docs]"      # or: make install-dev
```

## Tests

```bash
pytest                   # full suite with coverage (pyproject addopts)
pytest -q -o addopts=""  # without coverage
make test-fast           # stop at first failure
```

The suite runs in about ten seconds. It trains tiny models on random data, so
it exercises the real training loop on whatever device is available (MPS on
Apple Silicon). Highlights:

- `tests/test_modulation_classifier.py` — models, `IQDataset`, augmentation,
  device selection, train/save/load round trip, `validate_dataset_files`.
- `tests/test_callbacks.py` — the callback contract (payloads, global steps,
  cleanup on error) and the TensorBoard event files read back with
  TensorBoard's `EventAccumulator`.
- `tests/test_radioml2018.py` — builds a miniature file in the real RadioML
  layout with the class and SNR encoded into each frame's samples, converts it,
  and checks every label/SNR end to end; also class-order parsing/rejection.

When writing tests that train, give every class at least two samples and a
validation set of at least ten frames, or the stratified split raises.

## Style

```bash
make format        # black + isort
make lint          # flake8 + mypy
```

Docstrings are Google style (`Args:` / `Returns:` / `Raises:`); Sphinx renders
them with `napoleon`, so new public functions should have one.

## Documentation

```bash
make docs          # build docs/_build/html (warnings are errors)
make docs-serve    # build, then serve on http://localhost:8000
make docs-clean
```

Pages are Markdown (MyST). The API pages are `automodule` stubs under
`docs/api/`, and the CLI page is generated from the argparse builders with
`sphinx-argparse` — so new flags and docstrings appear automatically. Add a new
page by creating `docs/<name>.md` and listing it in a `toctree` in
`docs/index.md`.

## Extension points

Callbacks
: Subclass {class}`~modulation_classifier.callbacks.TrainingCallback` and pass
  instances via `train(callbacks=[...])`. See [Monitoring](monitoring.md#callbacks)
  for the hook contract.

Models
: Add a class to `models.py` and a branch in
  {func}`~modulation_classifier.models.create_model`; the CLI's
  `--architecture` choices live in `cli.py`. For one-off experiments, assign
  `classifier.model` directly ([Training](training.md#architectures)).

Data sources
: Anything that produces the HDF5/CSV pair described in [Data](data.md)
  works. For a new public dataset, follow `radioml2018.py`: read labels first,
  stream frames in blocks, write `train`/`test` plus label CSVs, and run
  `validate_dataset_files` on the result.

## Releasing

Update `version` in `pyproject.toml` and `__version__` in
`src/modulation_classifier/__init__.py`, move the `Unreleased` notes in
`CHANGELOG.md` under a new heading, then `make build` (and `make upload` for
PyPI).

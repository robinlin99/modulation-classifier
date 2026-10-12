# Installation

## Requirements

- Python 3.8 or newer (developed and tested on 3.14)
- PyTorch 1.12+ — installed automatically; on Apple Silicon the default wheel
  includes Metal (MPS) support, on Linux/Windows pick a CUDA or CPU build from
  [pytorch.org](https://pytorch.org/get-started/locally/) first if you need a
  specific one
- ~1 GB of disk for the package and its dependencies, plus space for data
  (see [Data](data.md))

## Install the package

A virtual environment keeps the PyTorch stack isolated from other projects:

```bash
git clone https://github.com/robinlin/modulation-classifier.git
cd modulation-classifier

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -e .
```

`-e` installs in *editable* mode, so changes under `src/` take effect without
reinstalling. The install provides two console scripts, `modulation-train` and
`modulation-predict`, and the `modulation_classifier` Python package.

### Optional extras

```bash
pip install -e ".[dev]"    # pytest, pytest-cov, black, flake8, mypy, pre-commit
pip install -e ".[docs]"   # sphinx, sphinx-rtd-theme, myst-parser, sphinx-argparse
```

TensorBoard is a regular dependency (training logs to it by default), so no
extra is needed for live training curves.

## Compute device

`ModulationClassifier` selects the fastest available backend automatically:

1. CUDA (`torch.cuda.is_available()`)
2. Apple Metal (`torch.backends.mps.is_available()`)
3. CPU

Override with `ModulationClassifier(device="cpu")` or `--device cpu|cuda|mps` on
the CLI. For a sense of scale: on an M1 Pro the default model trains at roughly
30–35 s per 12 000 frames on MPS, versus several minutes on CPU.

## Verify

```bash
python -c "import modulation_classifier as m, torch; print(m.__version__, torch.__version__)"
modulation-train --help
pytest                      # with the [dev] extra; ~10 s, 49 tests
```

Then get some data — see [Data](data.md). If you just want to see the
pipeline run end-to-end, `python create_dummy_data.py` writes a small synthetic
dataset into `data/`.

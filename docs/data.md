# Data

## Expected format

Training and prediction read two files, by default `data/data.hdf5` and
`data/train_labels.csv`:

**`data.hdf5`** — an HDF5 file with one or both of these datasets:

| Dataset | Shape | dtype | Purpose |
|---|---|---|---|
| `train` | `(N, 1024, 2)` | float32 | labelled frames used by `train()` |
| `test` | `(M, 1024, 2)` | float32 | unlabelled frames (competition hold-out) |

Each frame is 1024 complex samples stored as `[I, Q]` pairs. `IQDataset`
reshapes a frame to `(1, 1024, 2)` so the CNN sees it as a one-channel
"image" with time on one axis and I/Q on the other. If both datasets are
present, `IQDataset` loads `train`; a file with only `test` (or any other
single dataset) loads that instead.

**`train_labels.csv`** — one row per `train` frame, in order:

```text
Id,Category
0,8PSK
1,AM-SSB-SC
2,AM-DSB-SC
```

`Category` must be one of the ten names in
`modulation_classifier.MODULATION_CLASSES`. Extra columns (for example the
`SNR` column written by the RadioML converter) are ignored by the loader.

:::{tip}
`modulation_classifier.dataset.validate_dataset_files(hdf5, csv)` checks a pair
of files for shape, row-count and class-name consistency without reading the
frames — `setup_dataset.sh` runs it before and after copying.
:::

## Getting the data

`setup_dataset.sh` installs the two files into `data/` from either of two
sources. It is idempotent and prints the exact training command when done.

### Option 1 — the original ECE 4200 competition files

The Cornell ECE 4200 Kaggle competition shipped `data.hdf5` (30 000 labelled
`train` frames, 20 000 unlabelled `test` frames) and `train_labels.csv` in
exactly this format. The competition was invite-only, so the files are not
downloadable; if you have a copy, point the script at the directory:

```bash
./setup_dataset.sh --from-local "/Volumes/Seagate/Cornell/Spring_20/ECE4200/Kaggle"
```

It validates the pair, copies them into `data/`, writes `data/classes.txt`,
and validates the copy. Nothing is converted.

### Option 2 — RadioML 2018.01A from Kaggle

DeepSig's public [RadioML 2018.01A](https://www.kaggle.com/datasets/pinxau1000/radioml2018)
dataset contains all ten of this project's classes among its 24 modulations:
2 555 904 frames of 1024 samples, 26 SNRs from −20 to +30 dB, 4096 frames per
(modulation, SNR). It is big — ~18 GB zipped, ~21 GB extracted — so put it on
a volume with at least 45 GB free.

```bash
# One-time: Kaggle → Settings → API → "Create New Token"
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018
```

The script:

1. installs the `kaggle` CLI into the project venv if needed and checks for
   credentials (`~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME`/`KAGGLE_KEY`, or
   `KAGGLE_API_TOKEN`);
2. checks free space on `--raw-dir`, downloads and extracts the zip (skipped
   when `GOLD_XYZ_OSC.0001_1024.hdf5` is already there), deleting the zip
   unless `--keep-zip`;
3. runs `process_radioml2018.py` to convert a subset into `data/`.

Converter arguments go after `--`:

```bash
./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018 -- --frames-per-snr 1024 --min-snr -10
./setup_dataset.sh --raw-dir /Volumes/Seagate/radioml2018 --skip-download -- --dry-run
```

#### What the converter does

`process_radioml2018.py` (a thin wrapper around
{mod}`modulation_classifier.radioml2018`) handles two things that make a naive
conversion wrong or impractical:

Class order
: The `classes.txt` inside the original RadioML release lists the modulations
  in the **wrong order** — a known issue called out on the Kaggle page. Using it
  silently mislabels every frame (column 3 is really BPSK, not FM; real OOK
  frames would be labelled 32PSK and dropped). The converter reads
  `classes-fixed.json`/`.txt` from the Kaggle zip when it can find it next to or
  above the HDF5 file, otherwise falls back to the identical built-in list from
  the paper (O'Shea, Roy & Clancy 2018, Table I), and **refuses** the known-wrong
  order outright.

Memory
: `X` is ~21 GB and will not fit in RAM on most machines. The converter reads
  only `Y` and `Z` up front (in chunks), decides which frames to keep, and then
  streams `X` one (modulation, SNR) block at a time — 4096 contiguous rows,
  32 MB — writing directly into the output file.

Selection defaults to **512 random frames per class per SNR at SNR ≥ 0 dB**:
10 classes × 16 SNRs × 512 = 81 920 frames, ~0.7 GB, about 1–2 minutes per
epoch on MPS. Options:

| Option | Default | Meaning |
|---|---|---|
| `--frames-per-snr N` | 512 | random frames kept per (class, SNR); `0` keeps all 4096 |
| `--min-snr` / `--max-snr` | 0 / 30 | inclusive SNR range in dB |
| `--train-frac` | 0.8 | share of selected frames that go to `train` |
| `--seed` | 42 | subsampling and split seed |
| `--classes-json PATH` | auto | explicit class-order file |
| `--dry-run` | — | print the selection table, write nothing |

Output in `data/`:

- `data.hdf5` with `train`, `test`, and per-frame `train_snr`/`test_snr`
  (int16 dB) datasets; frames are shuffled across classes
- `train_labels.csv` and `test_labels.csv` with `Id,Category,SNR`
- `classes.txt`

Because the test split is labelled, `modulation-predict --evaluate` can report
accuracy on it, and the SNR columns allow accuracy-vs-SNR analysis.

Rough sizing for other settings (frames × 8 KB): 1024 frames/SNR at ≥ −10 dB
is ~215k frames / 1.8 GB; everything at ≥ −10 dB is ~860k frames / 7 GB. Note
that `IQDataset` loads the whole `train` array into memory and augmentation
triples it, so keep `train` under a few GB on a 16 GB machine.

### Option 3 — synthetic data

```bash
python create_dummy_data.py
```

Writes 5000 `train` and 2000 `test` frames of crude synthetic signals into
`data/`. Good for checking the install and the TensorBoard pipeline; the
signals are far easier than real ones, so accuracies are not meaningful.

## Augmentation

When `use_augmentation=True` (the default; `--no-augmentation` on the CLI),
`train()` appends phase-rotated copies of every training frame before the
train/validation split: `augmentation_factor=2` adds 90° and 180° rotations
(3× the data), `3` adds 270° as well. A phase rotation of the whole frame is a
label-preserving transform for these modulations, which is why it helps
generalisation.

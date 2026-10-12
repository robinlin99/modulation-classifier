# Troubleshooting

## No training output until the process ends

**Symptom.** Training is launched from an IDE, an agent runner, `nohup`, or
with `> train.log`, and nothing appears — no epoch lines, no loss — until it
finishes or is killed.

**Cause.** When Python's stdout is not a terminal it is block-buffered (8 KB).
Plain `print()` output sits in the buffer; a 50-epoch run produces well under
8 KB of epoch lines, so nothing is flushed until exit. The `tqdm` progress bar
writes to stderr with carriage returns, which most log viewers do not render
either.

**Fix.** Current versions handle this: the CLI switches stdout to line
buffering, every status message is flushed, and the progress bar is replaced by
plain lines when stderr is not a TTY (see [Monitoring](monitoring.md#console-output)).
If you call `train()` from your own script and still see nothing, run it with
`python -u` or set `PYTHONUNBUFFERED=1`.

## Epochs take many minutes

Check the `Using device:` line. `cpu` on a Mac with Apple Silicon means an old
version that only checked for CUDA; `ModulationClassifier` now picks `mps`
automatically, and you can force it with `--device mps`. Expect roughly
70 s/epoch for 24 000 frames on an M1 Pro versus 5–10 minutes on CPU. If you
are on CPU by necessity, `--no-augmentation` cuts the work by 3×.

## `ValueError: The least populated classes in y have only 1 member`

Raised by the stratified train/validation split when some class has fewer than
two samples, or when the validation set would be smaller than the number of
classes (`test_size = 8 should be greater or equal to the number of classes =
10`). Use more data, a larger `val_split`, or make sure every class is present.
This happens with tiny synthetic sets, not with the real data.

## RadioML download or conversion problems

`Not enough free space`
: The zip (~18 GB) and the extracted HDF5 (~21 GB) coexist briefly. Point
  `--raw-dir` at a volume with ≥ 45 GB free (an external drive is fine); the
  zip is deleted after extraction unless `--keep-zip`.

`Kaggle API credentials not found`
: Create a token at kaggle.com → Settings → API → *Create New Token*, then
  `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`.
  `KAGGLE_USERNAME`/`KAGGLE_KEY` or `KAGGLE_API_TOKEN` environment variables
  also work.

`this is the known-incorrect order from the original classes.txt`
: You passed the dataset's own `classes.txt` as `--classes-json`. Use
  `classes-fixed.json` from the Kaggle zip, or omit the flag to use the
  built-in corrected order. See [Data](data.md#what-the-converter-does).

`No frames found for classes [...]; the class order is probably wrong`
: The class list did not match the one-hot columns. Same fix as above.

Process killed / machine swapping during conversion
: You are running an old converter that loaded all of `X` into memory. The
  current one streams 32 MB blocks and needs well under 1 GB.

Conversion is slow
: Reading the 21 GB file is I/O-bound; from a USB SSD expect a few minutes for
  the default subset, longer for `--frames-per-snr 0`. `--dry-run` previews the
  selection instantly.

## TensorBoard shows nothing, or "port 6006 is in use"

- Make sure `--logdir` is the *parent* directory (`runs`), not a single run.
- The event files flush after every logged point; if the browser is stale,
  lower the reload interval in TensorBoard's ⚙ menu or refresh.
- Another TensorBoard is already running: `tensorboard --logdir runs --port 6007`,
  or find it with `lsof -i :6006`.
- If training was started with `--no-tensorboard`, there is nothing to show.

## `Error(s) in loading state_dict` when loading a checkpoint

The checkpoint was trained with a different architecture. Construct the
classifier with the matching `model_architecture` (`--architecture legacy` on
the CLI) before `load_model()`; the architecture is not stored in the file.

## `RuntimeError: Model must be trained or loaded before prediction`

Call `train()` or `load_model()` first. A freshly constructed classifier has
random weights and refuses to predict.

## Plots block or fail on a headless machine

The plotting helpers call `plt.show()`. Set `MPLBACKEND=Agg` (or pass
`save_path=` and ignore the window) when no display is available.

## `modulation-train` runs a different Python than my venv

Console scripts are installed per interpreter. Check with `which
modulation-train`; the one in `venv/bin/` uses the venv. Editable installs
(`pip install -e .`) in several interpreters all point at the same `src/`
tree, so they stay in sync with the code even if their dependencies differ.

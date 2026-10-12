# Monitoring training

There are two ways to watch a run: the console output and TensorBoard. Both
work no matter how training was launched — from a terminal, from an IDE or
agent runner, under `nohup`, or with output redirected to a file.

## Console output

Every run prints a short configuration header, one line per epoch, and
checkpoint/early-stopping events:

```text
Starting training: up to 50 epochs, 375 batches/epoch (batch size 64), device mps
Epoch   1/50: Train Loss: 1.7728, Train Acc: 31.93%, Val Loss: 1.6391, Val Acc: 31.95%, LR: 1.00e-03, Time: 69.0s
✓ New best model saved: val_loss=1.6391
```

Within an epoch, what you see depends on whether stderr is a terminal:

- **Interactive terminal** — a `tqdm` progress bar with the current batch loss
  and running accuracy, cleared when the epoch finishes.
- **Piped / captured output** — no progress bar (its carriage returns would be
  unreadable in a log); instead a plain line every `--log-interval` batches
  (default 50) and at the end of the epoch:

  ```text
  Epoch 1/50 [50/375] Loss: 2.2311, Acc: 16.78%
  Epoch 1/50 [100/375] Loss: 2.1144, Acc: 20.62%
  ```

All status output is flushed immediately, and the CLI switches stdout to line
buffering at startup. This matters: when Python's stdout is a pipe it is
block-buffered by default, so without this nothing at all would appear until
the process exited (see [Troubleshooting](troubleshooting.md#no-training-output-until-the-process-ends)).

`--log-interval 0` disables the batch-level lines (and batch-level TensorBoard
points); the per-epoch summary is always printed.

## TensorBoard

`modulation-train` logs to TensorBoard by default. Each run gets its own
directory, `runs/<timestamp>_<architecture>/`, containing `train/` and `val/`
sub-runs so that training and validation curves overlay on the same chart.

```bash
tensorboard --logdir runs        # or: make tensorboard
```

Open <http://localhost:6006>. The page polls for new data (every 30 s by
default; the ⚙ menu has a reload-interval setting), and the event files are
flushed after every logged point, so curves advance while training runs.

### What is logged

| Tag | Sub-run | x-axis | Meaning |
|---|---|---|---|
| `epoch/loss` | train, val | epoch | mean loss over the epoch |
| `epoch/accuracy` | train, val | epoch | accuracy in percent |
| `epoch/learning_rate` | train | epoch | LR after the scheduler step |
| `epoch/seconds` | train | epoch | wall time of the epoch |
| `batch/loss` | train | global batch step | running mean loss in the current epoch |
| `batch/accuracy` | train | global batch step | running accuracy in the current epoch |
| `batch/loss_instant` | train | global batch step | loss of that single batch |
| `hparams` (Text tab) | train | — | the configuration passed to `train()` |

Batch-level points are written every `--log-interval` batches and on the last
batch of each epoch. The *global batch step* keeps increasing across epochs, so
the batch charts show the whole run.

### Options

```bash
modulation-train ... --run-name lr1e-3_bs64   # directory name inside --log-dir
modulation-train ... --log-dir experiments    # parent directory (default: runs)
modulation-train ... --no-tensorboard         # disable logging
```

Several runs under one `--logdir` appear side by side in TensorBoard, which is
the easiest way to compare hyperparameters. `runs/` is git-ignored.

From Python, pass `log_dir="runs/exp1"` to `train()`, or construct a
{class}`~modulation_classifier.callbacks.TensorBoardLogger` yourself (for
example to change `flush_secs`) and pass it in `callbacks=`.

## Callbacks

Logging is built on a small hook interface, so other sinks — a CSV file,
Weights & Biases, a notebook plot, a Slack message when training ends — are a
few lines each. Subclass {class}`~modulation_classifier.callbacks.TrainingCallback`
and override any of:

| Hook | When | Arguments |
|---|---|---|
| `on_train_begin(config)` | once, before the first epoch | dict: `architecture`, `parameters`, `device`, `num_epochs`, `batch_size`, `batches_per_epoch`, `learning_rate`, `weight_decay`, `val_split`, `patience`, `use_augmentation`, `augmentation_factor`, `train_samples`, `val_samples`, `save_path` |
| `on_batch_end(step, metrics)` | every `log_interval` batches and on the last batch of an epoch | `step`: global batch index (1-based); `metrics`: `epoch`, `batch`, `num_batches`, `loss` (running mean), `accuracy` (running, %), `batch_loss` |
| `on_epoch_end(epoch, metrics)` | after validation each epoch | `epoch` (1-based); `metrics`: `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`, `seconds`, `improved` |
| `on_train_end(history)` | once, when training finishes — including on an exception or Ctrl-C | the history dict |

Example — append epoch metrics to a CSV and notify when done:

```python
import csv
from modulation_classifier import ModulationClassifier, TensorBoardLogger, TrainingCallback

class CsvLogger(TrainingCallback):
    def __init__(self, path):
        self.path = path
        self._fh = None

    def on_train_begin(self, config):
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        self._writer.writeheader()

    def on_epoch_end(self, epoch, metrics):
        self._writer.writerow({"epoch": epoch, **{k: metrics[k] for k in self._writer.fieldnames[1:]}})
        self._fh.flush()

    def on_train_end(self, history):
        self._fh.close()
        print(f"done: best val acc {max(history['val_acc']):.2f}%")

clf = ModulationClassifier()
clf.train(dataset, num_epochs=50,
          callbacks=[TensorBoardLogger("runs/exp1"), CsvLogger("exp1_metrics.csv")])
```

Callbacks run synchronously in the training loop, so keep `on_batch_end` cheap
or raise `log_interval`. An exception raised inside a callback stops training;
`on_train_end` is still called on every callback so files and writers are
closed.

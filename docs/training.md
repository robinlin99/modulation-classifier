# Training

## From the command line

```bash
modulation-train --data data/data.hdf5 --labels data/train_labels.csv --epochs 50
```

A typical first run on the ECE 4200 data (30 000 frames, augmentation on,
batch 64) takes about 3.5 minutes per epoch on an M1 Pro. Useful variations:

```bash
# Faster iteration: no augmentation, bigger batches, early-stop sooner
modulation-train --data data/data.hdf5 --labels data/train_labels.csv \
    --epochs 20 --batch-size 128 --no-augmentation --patience 5

# Name the TensorBoard run and save plots at the end
modulation-train ... --run-name lr5e-4 --lr 5e-4 --plots-dir plots/

# Force a device
modulation-train ... --device cpu
```

The full flag list is in the [CLI reference](cli.md). The best checkpoint
(lowest validation loss) is written to `--model-path` (default
`best_model.pt`) whenever it improves, so interrupting a run still leaves a
usable model.

## From Python

```python
from modulation_classifier import ModulationClassifier, IQDataset

dataset = IQDataset("data/data.hdf5", "data/train_labels.csv")

clf = ModulationClassifier(model_architecture="improved")   # device auto-detected
history = clf.train(
    dataset,
    num_epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-4,
    val_split=0.2,
    patience=10,
    use_augmentation=True,
    augmentation_factor=2,
    save_path="best_model.pt",
    log_dir="runs/exp1",          # TensorBoard; see Monitoring
)

history["val_acc"][-1]            # lists of per-epoch train/val loss & accuracy
clf.plot_training_curves("curves.png")
```

`train()` also accepts a path to an HDF5 file plus `labels_path=` instead of
an `IQDataset`. See {meth}`~modulation_classifier.classifier.ModulationClassifier.train`
for every argument.

## What `train()` does

1. **Augmentation** (optional) — appends 90°/180° (and 270° for
   `augmentation_factor=3`) phase-rotated copies of every frame.
2. **Split** — a stratified, seeded (`random_state=42`) train/validation split
   with `val_split` held out, so every class is represented in validation.
3. **Optimiser** — AdamW (`learning_rate`, `weight_decay`) with
   `ReduceLROnPlateau(mode="min", factor=0.5, patience=5)` on the validation
   loss; cross-entropy loss.
4. **Epoch loop** — train, validate, step the scheduler, print one summary line
   per epoch, invoke callbacks, save a checkpoint when validation loss improves,
   and stop early after `patience` epochs without improvement.
5. **History** — `training_history` holds `train_loss`, `val_loss`,
   `train_acc`, `val_acc` (percent) per epoch; it is also preserved if training
   is interrupted.

### Choosing hyperparameters

- **Batch size** — 64 is a good default on MPS/GPU; memory is not a constraint
  for this model, so larger batches mainly trade a little generalisation for
  speed. On CPU, smaller batches do not help.
- **Epochs / patience** — early stopping makes a large `--epochs` safe; with
  `patience=10` the run ends about ten epochs after the best validation loss.
- **Learning rate** — 1e-3 works for both architectures; the scheduler halves
  it when validation loss plateaus for five epochs, which is visible in the
  `epoch/learning_rate` TensorBoard chart.
- **Augmentation** — triples the data and the time per epoch. Turn it off for
  quick experiments, on for final models.

## Checkpoints

The best-model file written during training is a `torch.save` dict:

| Key | Content |
|---|---|
| `epoch` | 0-based epoch index of the checkpoint |
| `model_state_dict` | model weights |
| `optimizer_state_dict` | AdamW state |
| `val_loss`, `val_acc` | validation metrics at that epoch |
| `classes` | class-name list |

`ModulationClassifier.save_model(path)` writes a lighter file with
`model_state_dict`, `classes`, `num_classes` and `training_history`.
`load_model(path)` accepts either; it restores weights, class names and (if
present) the history. The architecture is **not** stored, so construct the
classifier with the same `model_architecture` before loading — the CLI's
`--architecture` flag exists for this reason.

```python
clf = ModulationClassifier(model_architecture="improved")
clf.load_model("best_model.pt")
```

## Architectures

`--architecture improved` (default) is {class}`~modulation_classifier.models.ImprovedNet`:
four `Conv2d` + `BatchNorm2d` + ReLU blocks (1→256→256→128→80 channels, the
second with a `(3, 2)` kernel that mixes the I and Q columns), global average
pooling, dropout, and a 80→128→10 head — 536 826 parameters.

`--architecture legacy` is {class}`~modulation_classifier.models.LegacyNet`, the
original competition network kept for comparison: no padding, two batch-norm
layers shared across four convolutions, a flattened 81 280-wide linear layer
(10 881 226 parameters in total) and `log_softmax` output. It only accepts
1024-sample frames.

To experiment with your own network, construct the classifier and replace its
model before training — the loop only needs an `nn.Module` mapping
`(N, 1, T, 2)` to `(N, num_classes)` logits:

```python
import torch.nn as nn
clf = ModulationClassifier()
clf.model = MyNet(num_classes=10).to(clf.device)
clf.train(dataset, num_epochs=30)
```

`examples/train_comprehensive.py custom` shows a complete version.

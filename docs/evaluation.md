# Evaluation and prediction

## `modulation-predict`

```bash
modulation-predict --model best_model.pt --data data/data.hdf5 --output predictions.csv
```

Loads the checkpoint into a fresh classifier (pass `--architecture legacy` if
that is what you trained), runs inference on the HDF5 file, writes a CSV and
prints the predicted-class distribution:

```text
Id,Category
0,QPSK
1,OOK
```

That `Id,Category` layout is the Kaggle submission format. Options:

```bash
--probabilities                 # add one Prob_<class> column per class (softmax)
--labels data/test_labels.csv   # labels for the frames being predicted ...
--evaluate                      # ... and report accuracy + per-class P/R/F1
--plots-dir plots/              # with --evaluate: save confusion_matrix.png
--batch-size 256 --device cpu
```

:::{important}
`IQDataset` loads the `train` dataset when an HDF5 file contains both `train`
and `test`. So `modulation-predict --data data/data.hdf5` predicts the
*training* frames of a competition-style file. To predict the unlabelled
`test` split, read it yourself (see below) or write it to its own file. The
RadioML converter's output has the same layout, but its `test_labels.csv`
corresponds to the `test` dataset, not `train` — don't pair the two.
:::

## From Python

```python
from modulation_classifier import ModulationClassifier, IQDataset
import h5py, numpy as np

clf = ModulationClassifier()
clf.load_model("best_model.pt")

# Predict the unlabelled 'test' split of a competition-style file
with h5py.File("data/data.hdf5") as f:
    test_frames = f["test"][()]                      # (M, 1024, 2)
preds = clf.predict(test_frames)                     # class indices, shape (M,)
names = clf.predict_classes(test_frames)             # class names
preds, probs = clf.predict(test_frames, return_probabilities=True)   # probs: (M, 10)
```

`predict()` accepts an `IQDataset`, a NumPy array of shape `(N, 1024, 2)`, or a
path to an HDF5 file.

### Metrics

```python
val = IQDataset("data/data.hdf5", "data/train_labels.csv")   # any labelled set
metrics = clf.evaluate(val)
metrics["accuracy"]                                  # overall accuracy, 0-1
metrics["classification_report"]["QPSK"]["f1-score"] # sklearn report as a dict
```

{func}`modulation_classifier.utils.calculate_model_metrics` returns macro and
weighted precision/recall/F1 plus the raw confusion matrix if you already have
`y_true`/`y_pred` arrays.

### Plots

```python
clf.plot_training_curves("curves.png")               # loss/accuracy vs epoch
clf.plot_confusion_matrix(val, save_path="cm.png")   # normalised, seaborn heatmap

from modulation_classifier import plot_constellation, analyze_class_distribution
plot_constellation(val.data, val.classes, val.labels, max_samples_per_class=50,
                   save_path="constellations.png")  # I/Q scatter per class
analyze_class_distribution(val.labels, val.classes)  # bar chart + DataFrame
```

Each plotting helper saves to `save_path` when given and then calls
`plt.show()`; on a headless machine set `MPLBACKEND=Agg`.

## Accuracy vs SNR (RadioML data)

The RadioML converter stores each frame's SNR (`train_snr`/`test_snr` in the
HDF5 and an `SNR` column in the CSVs), so a per-SNR breakdown is a groupby:

```python
import pandas as pd, h5py
with h5py.File("data/data.hdf5") as f:
    frames, snr = f["test"][()], f["test_snr"][()]
labels = pd.read_csv("data/test_labels.csv")
pred = clf.predict(frames)
truth = labels["Category"].map(clf.classes.index).to_numpy()
acc_by_snr = pd.Series(pred == truth).groupby(snr).mean()
print(acc_by_snr)
```

## Making a Kaggle submission

For the original competition the hold-out frames are the `test` dataset and
the submission is `Id,Category` with `Id` from 0 to M−1:

```python
pd.DataFrame({"Id": range(len(names)), "Category": names}).to_csv("submission.csv", index=False)
```

`examples/train_comprehensive.py eval` performs this end to end.

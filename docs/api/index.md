# API reference

The public surface is re-exported from the top-level package:

```python
from modulation_classifier import (
    ModulationClassifier,            # training / prediction / evaluation
    IQDataset, MODULATION_CLASSES,   # data loading
    ImprovedNet, create_model, count_parameters,
    TrainingCallback, TensorBoardLogger,
    plot_confusion_matrix, plot_constellation, plot_training_curves,
    analyze_class_distribution,
)
```

```{toctree}
:maxdepth: 2

classifier
dataset
models
callbacks
radioml2018
utils
cli
```

"""Modulation Classifier - Deep learning for radio modulation classification."""

__version__ = "0.1.0"
__author__ = "Robin Lin"
__email__ = "zl755@cornell.edu"

from .classifier import ModulationClassifier
from .dataset import IQDataset, MODULATION_CLASSES
from .models import ImprovedNet, create_model, count_parameters
from .callbacks import TrainingCallback, TensorBoardLogger
from .utils import plot_confusion_matrix, plot_constellation, plot_training_curves, analyze_class_distribution

__all__ = [
    "ModulationClassifier",
    "IQDataset", 
    "MODULATION_CLASSES",
    "ImprovedNet",
    "create_model",
    "count_parameters",
    "TrainingCallback",
    "TensorBoardLogger",
    "plot_confusion_matrix",
    "plot_constellation", 
    "plot_training_curves",
    "analyze_class_distribution",
]
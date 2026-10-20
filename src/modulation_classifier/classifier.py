"""Main modulation classifier class."""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Sequence, Tuple, Dict, List, Union
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .models import ImprovedNet, create_model, count_parameters
from .dataset import IQDataset, create_data_loaders, create_augmented_dataset, MODULATION_CLASSES
from .utils import plot_training_curves, plot_confusion_matrix
from .callbacks import TrainingCallback, CallbackList, TensorBoardLogger


def get_default_device() -> torch.device:
    """Pick the fastest available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_interactive() -> bool:
    """True when progress bars can be rendered (stderr is a real terminal)."""
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


class ModulationClassifier:
    """Main classifier class for modulation classification."""
    
    def __init__(
        self,
        model_architecture: str = "improved",
        num_classes: int = 10,
        classes: List[str] = MODULATION_CLASSES,
        device: Optional[str] = None
    ):
        """Initialize the classifier.
        
        Args:
            model_architecture: Model architecture ("improved", "complex", or "legacy")
            num_classes: Number of modulation classes
            classes: List of class names
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
        """
        self.classes = classes
        self.num_classes = num_classes
        self.model_architecture = model_architecture
        
        # Determine if we need complex data format
        self.use_complex_format = (model_architecture == "complex")
        
        # Set device
        if device is None:
            self.device = get_default_device()
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}", flush=True)
        
        # Create model
        self.model = create_model(
            architecture=model_architecture, 
            num_classes=num_classes
        ).to(self.device)
        
        print(f"Model parameters: {count_parameters(self.model):,}", flush=True)
        
        # Training state
        self.trained = False
        self.training_history = {}
        
    def train(
        self,
        dataset: Union[IQDataset, str],
        labels_path: Optional[str] = None,
        num_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        val_split: float = 0.2,
        patience: int = 10,
        save_path: str = "best_model.pt",
        use_augmentation: bool = True,
        augmentation_factor: int = 2,
        log_interval: int = 50,
        log_dir: Optional[str] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None
    ) -> Dict[str, List[float]]:
        """Train the modulation classifier.
        
        Args:
            dataset: IQDataset or path to HDF5 file
            labels_path: Path to labels CSV (if dataset is a file path)
            num_epochs: Maximum number of training epochs
            batch_size: Training batch size
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            val_split: Fraction of data for validation
            patience: Early stopping patience
            save_path: Path to save best model
            use_augmentation: Whether to use data augmentation
            augmentation_factor: Number of augmented copies per sample
            log_interval: Every this many batches (and at the end of each
                epoch) batch-level metrics are sent to callbacks and, when
                output is not an interactive terminal, printed. 0 disables
                batch-level logging.
            log_dir: If given, stream metrics to TensorBoard under this
                directory (shorthand for ``callbacks=[TensorBoardLogger(log_dir)]``).
                View with ``tensorboard --logdir <parent of log_dir>``.
            callbacks: Additional :class:`TrainingCallback` objects.
            
        Returns:
            Dictionary with training history
        """
        callback_list = CallbackList(callbacks)
        if log_dir is not None:
            callback_list.callbacks.append(TensorBoardLogger(log_dir))
            print(f"TensorBoard logs: {log_dir}", flush=True)
        
        # Load dataset if path provided
        if isinstance(dataset, str):
            # Load raw data first (before format conversion)
            raw_dataset = IQDataset(dataset, labels_path, self.classes, data_format="standard")
            
            # Apply data augmentation to raw format if requested
            if use_augmentation:
                print("Applying data augmentation...", flush=True)
                # Get original format data (N, T, 2) for augmentation
                if hasattr(raw_dataset, '_raw_data'):
                    orig_data = raw_dataset._raw_data
                else:
                    # Convert back from standard format (N, 1, T, 2) to (N, T, 2)
                    orig_data = raw_dataset.data[:, 0, :, :]
                
                aug_data, aug_labels = create_augmented_dataset(
                    orig_data, raw_dataset.labels, augmentation_factor
                )
                
                # Now apply the correct format transformation
                data_format = "complex" if self.use_complex_format else "standard"
                dataset = IQDataset(aug_data, aug_labels, self.classes, data_format=data_format)
            else:
                # No augmentation, just convert format
                data_format = "complex" if self.use_complex_format else "standard"
                dataset = IQDataset(dataset, labels_path, self.classes, data_format=data_format)
        else:
            # Dataset object provided
            if use_augmentation:
                print("Applying data augmentation...", flush=True)
                # Need to get original format for augmentation
                if hasattr(dataset, '_raw_data'):
                    orig_data = dataset._raw_data
                else:
                    # Try to convert from current format back to original
                    if dataset.data_format == "standard":
                        orig_data = dataset.data[:, 0, :, :]  # (N, 1, T, 2) -> (N, T, 2)
                    elif dataset.data_format == "complex":
                        # (N, 2, T, 1) -> (N, T, 2)
                        orig_data = np.stack([dataset.data[:, 0, :, 0], dataset.data[:, 1, :, 0]], axis=-1)
                    else:
                        raise ValueError(f"Unknown data format: {dataset.data_format}")
                
                aug_data, aug_labels = create_augmented_dataset(
                    orig_data, dataset.labels, augmentation_factor,
                    methods=['rotation', 'amplitude', 'awgn', 'time_shift']  # Enhanced methods
                )
                data_format = "complex" if self.use_complex_format else "standard"
                dataset = IQDataset(aug_data, aug_labels, self.classes, data_format=data_format)
            
        if dataset.labels is None:
            raise ValueError("Dataset must have labels for training")
            
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            dataset, test_size=val_split, batch_size=batch_size
        )
        
        print(f"Training samples: {len(train_loader.dataset)}", flush=True)
        print(f"Validation samples: {len(val_loader.dataset)}", flush=True)
        
        # Setup training
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        print(f"Starting training: up to {num_epochs} epochs, "
              f"{len(train_loader)} batches/epoch (batch size {batch_size}), "
              f"device {self.device}", flush=True)
        
        callback_list.on_train_begin({
            'architecture': self.model_architecture,
            'parameters': count_parameters(self.model),
            'device': str(self.device),
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'batches_per_epoch': len(train_loader),
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'val_split': val_split,
            'patience': patience,
            'use_augmentation': use_augmentation,
            'augmentation_factor': augmentation_factor if use_augmentation else 0,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'save_path': save_path,
        })
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.perf_counter()
                
                # Training phase
                train_loss, train_acc = self._train_epoch(
                    train_loader, optimizer, criterion,
                    epoch=epoch + 1, num_epochs=num_epochs, log_interval=log_interval,
                    callbacks=callback_list
                )
                
                # Validation phase
                val_loss, val_acc = self._validate_epoch(
                    val_loader, criterion
                )
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Store metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                elapsed = time.perf_counter() - epoch_start
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                      f'LR: {current_lr:.2e}, Time: {elapsed:.1f}s', flush=True)
                
                # Early stopping and model saving
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'classes': self.classes
                    }, save_path)
                    
                    print(f'✓ New best model saved: val_loss={best_val_loss:.4f}', flush=True)
                else:
                    patience_counter += 1
                
                callback_list.on_epoch_end(epoch + 1, {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': current_lr,
                    'seconds': elapsed,
                    'improved': improved,
                })
                    
                if patience_counter >= patience:
                    print(f'Early stopping after {patience} epochs without improvement', flush=True)
                    break
        finally:
            # Store whatever history we have (also on KeyboardInterrupt/errors)
            self.training_history = {
                'train_loss': train_losses,
                'val_loss': val_losses, 
                'train_acc': train_accuracies,
                'val_acc': val_accuracies
            }
            callback_list.on_train_end(self.training_history)
        
        self.trained = True
        print(f'Training completed. Best validation loss: {best_val_loss:.4f}', flush=True)
        
        return self.training_history
    
    def _train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        epoch: int = 1,
        num_epochs: int = 1,
        log_interval: int = 0,
        callbacks: Optional[TrainingCallback] = None
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Shows a tqdm progress bar when attached to a terminal. When output is
        piped (log files, IDE/agent runners) the bar is disabled and a plain
        progress line is printed every ``log_interval`` batches instead, so
        progress remains visible. Batch-level metrics are also sent to
        ``callbacks`` at the same cadence.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        interactive = _is_interactive()
        
        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{num_epochs}',
            disable=not interactive,
            leave=False
        )
        for batch_idx, (data, target) in enumerate(progress_bar, start=1):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            running_loss = total_loss / batch_idx
            running_acc = 100. * correct / total
            
            if interactive:
                progress_bar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'Acc': f'{running_acc:.1f}%'
                })
            
            if log_interval and (batch_idx % log_interval == 0 or batch_idx == num_batches):
                if not interactive:
                    print(f'Epoch {epoch}/{num_epochs} [{batch_idx}/{num_batches}] '
                          f'Loss: {running_loss:.4f}, Acc: {running_acc:.2f}%',
                          flush=True)
                if callbacks is not None:
                    global_step = (epoch - 1) * num_batches + batch_idx
                    callbacks.on_batch_end(global_step, {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'num_batches': num_batches,
                        'loss': running_loss,
                        'accuracy': running_acc,
                        'batch_loss': batch_loss,
                    })
            
        return total_loss / num_batches, 100. * correct / total
    
    def _validate_epoch(self, val_loader, criterion) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        return total_loss / len(val_loader), 100. * correct / total
    
    def predict(
        self, 
        data: Union[IQDataset, str, np.ndarray],
        batch_size: int = 64,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions on new data.
        
        Args:
            data: Test data (IQDataset, file path, or numpy array)
            batch_size: Batch size for inference
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions array, optionally with probabilities
        """
        if not self.trained and not hasattr(self.model, '_loaded_weights'):
            raise RuntimeError("Model must be trained or loaded before prediction")
            
        # Prepare dataset
        if isinstance(data, str):
            dataset = IQDataset(data, classes=self.classes)
        elif isinstance(data, np.ndarray):
            dataset = IQDataset(data, classes=self.classes)
        else:
            dataset = data
            
        # Create data loader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Generate predictions
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Predicting', disable=not _is_interactive()):
                if isinstance(batch, (tuple, list)):
                    batch_data = batch[0]  # If batch has labels, take only data
                else:
                    batch_data = batch
                    
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                
                if return_probabilities:
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probabilities.append(probabilities.cpu().numpy())
                    
                predictions = outputs.argmax(dim=1)
                all_predictions.append(predictions.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions)
        
        if return_probabilities:
            probabilities = np.concatenate(all_probabilities)
            return predictions, probabilities
        else:
            return predictions
    
    def predict_classes(
        self, 
        data: Union[IQDataset, str, np.ndarray], 
        **kwargs
    ) -> List[str]:
        """Predict class names (not indices)."""
        predictions = self.predict(data, **kwargs)
        return [self.classes[pred] for pred in predictions]
    
    def evaluate(
        self, 
        dataset: Union[IQDataset, str], 
        labels_path: Optional[str] = None,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            dataset: Test dataset or path to HDF5 file
            labels_path: Path to labels CSV (if dataset is a file path)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load dataset if path provided
        if isinstance(dataset, str):
            dataset = IQDataset(dataset, labels_path, self.classes)
            
        if dataset.labels is None:
            raise ValueError("Dataset must have labels for evaluation")
            
        # Get predictions
        predictions = self.predict(dataset, batch_size=batch_size)
        true_labels = dataset.labels
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Classification report
        report = classification_report(
            true_labels, predictions, 
            target_names=self.classes, 
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.trained:
            raise RuntimeError("Model must be trained before saving")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'num_classes': self.num_classes,
            'training_history': self.training_history
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'classes' in checkpoint:
            self.classes = checkpoint['classes']
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        self.model._loaded_weights = True
        print(f"Model loaded from {filepath}", flush=True)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        if not self.training_history:
            raise RuntimeError("No training history available")
            
        plot_training_curves(self.training_history, save_path)
    
    def plot_confusion_matrix(
        self, 
        dataset: Union[IQDataset, str], 
        labels_path: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix on test data."""
        # Load dataset if path provided
        if isinstance(dataset, str):
            dataset = IQDataset(dataset, labels_path, self.classes)
            
        if dataset.labels is None:
            raise ValueError("Dataset must have labels for confusion matrix")
            
        # Get predictions
        predictions = self.predict(dataset)
        true_labels = dataset.labels
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plot_confusion_matrix(cm, self.classes, save_path)
"""Dataset utilities for loading and processing IQ modulation data."""

import csv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional, Union
import os
from collections import Counter


# Standard modulation classes
MODULATION_CLASSES = [
    "FM", "OQPSK", "BPSK", "8PSK", "AM-SSB-SC", 
    "4ASK", "16PSK", "AM-DSB-SC", "QPSK", "OOK"
]


class IQDataset(Dataset):
    """PyTorch dataset for IQ modulation data."""
    
    def __init__(
        self, 
        data: Union[str, np.ndarray], 
        labels: Optional[Union[str, np.ndarray]] = None,
        classes: List[str] = MODULATION_CLASSES,
        transform: Optional[callable] = None
    ):
        """Initialize IQ dataset.
        
        Args:
            data: Path to HDF5 file or numpy array of IQ data
            labels: Path to CSV file or numpy array of labels (optional for test data)
            classes: List of modulation class names
            transform: Optional transform to apply to samples
        """
        self.classes = classes
        self.transform = transform
        
        # Load data
        if isinstance(data, str):
            self.data = self._load_hdf5_data(data)
        else:
            self.data = data
            
        # Load labels if provided
        if labels is not None:
            if isinstance(labels, str):
                self.labels = self._load_csv_labels(labels)
            else:
                self.labels = labels
        else:
            self.labels = None
            
        # Validate data shape
        if len(self.data.shape) != 3 or self.data.shape[2] != 2:
            raise ValueError(f"Expected data shape (N, T, 2), got {self.data.shape}")
            
    def _load_hdf5_data(self, filepath: str) -> np.ndarray:
        """Load IQ data from HDF5 file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        with h5py.File(filepath, 'r') as f:
            if 'train' in f:
                return f['train'][()]
            elif 'test' in f:
                return f['test'][()]
            else:
                # Assume data is in root or first dataset
                keys = list(f.keys())
                return f[keys[0]][()]
                
    def _load_csv_labels(self, filepath: str) -> np.ndarray:
        """Load labels from CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Labels file not found: {filepath}")
            
        labels = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 2:
                    class_name = row[1]
                    if class_name in self.classes:
                        labels.append(self.classes.index(class_name))
                    else:
                        raise ValueError(f"Unknown class: {class_name}")
                        
        return np.array(labels)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get a sample from the dataset."""
        # Get IQ data and reshape for CNN: (T, 2) -> (1, T, 2)
        sample = torch.from_numpy(self.data[idx]).unsqueeze(0).float()
        
        if self.transform:
            sample = self.transform(sample)
            
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return sample, label
        else:
            return sample


class IQAugmentation:
    """Data augmentation transforms for IQ signals."""
    
    @staticmethod
    def phase_rotation(sample: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply phase rotation to IQ signal.
        
        Args:
            sample: IQ tensor of shape (1, T, 2)
            angle: Rotation angle in radians
            
        Returns:
            Rotated IQ tensor
        """
        cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=sample.dtype)
        
        # Apply rotation: (T, 2) @ (2, 2) -> (T, 2)
        rotated = torch.matmul(sample.squeeze(0), rotation_matrix)
        return rotated.unsqueeze(0)
    
    @staticmethod
    def add_noise(sample: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to IQ signal."""
        noise = torch.randn_like(sample) * noise_std
        return sample + noise
    
    @staticmethod
    def random_phase_rotation(sample: torch.Tensor) -> torch.Tensor:
        """Apply random phase rotation (0°, 90°, 180°, 270°)."""
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        angle = np.random.choice(angles)
        return IQAugmentation.phase_rotation(sample, angle)


def create_augmented_dataset(
    data: np.ndarray, 
    labels: np.ndarray, 
    augmentation_factor: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Create augmented dataset with phase rotations.
    
    Args:
        data: Original IQ data array
        labels: Original labels array  
        augmentation_factor: Number of augmented copies per sample
        
    Returns:
        Tuple of (augmented_data, augmented_labels)
    """
    print(f"Creating augmented dataset (factor={augmentation_factor})...", flush=True)
    
    augmented_data = [data]
    augmented_labels = [labels]
    
    # Apply rotations
    rotation_angles = [np.pi/2, np.pi, 3*np.pi/2]
    
    for i, angle in enumerate(rotation_angles[:augmentation_factor]):
        print(f"Applying {angle*180/np.pi:.0f}° rotation...", flush=True)
        
        rotated_data = []
        for sample in data:
            # Rotation matrix for IQ data
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Apply rotation: (T, 2) @ (2, 2) -> (T, 2)  
            rotated_sample = np.dot(sample, rot_matrix)
            rotated_data.append(rotated_sample)
            
        augmented_data.append(np.array(rotated_data))
        augmented_labels.append(labels.copy())
    
    # Concatenate all data
    final_data = np.concatenate(augmented_data, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)
    
    print(f"Augmented dataset: {final_data.shape[0]} samples", flush=True)
    return final_data, final_labels


def create_data_loaders(
    dataset: IQDataset,
    test_size: float = 0.2,
    batch_size: int = 64,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        dataset: IQ dataset with labels
        test_size: Fraction of data to use for validation
        batch_size: Batch size for data loaders
        random_state: Random seed for reproducible splits
        stratify: Whether to stratify the split by class
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if dataset.labels is None:
        raise ValueError("Dataset must have labels for train/val split")
        
    # Split indices
    indices = np.arange(len(dataset))
    stratify_labels = dataset.labels if stratify else None
    
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_labels
    )
    
    # Create subset datasets
    train_data = dataset.data[train_idx]
    train_labels = dataset.labels[train_idx] 
    val_data = dataset.data[val_idx]
    val_labels = dataset.labels[val_idx]
    
    train_dataset = IQDataset(train_data, train_labels, dataset.classes)
    val_dataset = IQDataset(val_data, val_labels, dataset.classes)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader


def validate_dataset_files(
    hdf5_path: str,
    labels_path: Optional[str] = None,
    classes: List[str] = MODULATION_CLASSES
) -> Dict[str, object]:
    """Check that an HDF5 file (and optional labels CSV) is loadable by IQDataset.
    
    Only metadata and the CSV are read, so this is cheap even for large files.
    
    Args:
        hdf5_path: Path to the HDF5 file (expects a 'train' and/or 'test' dataset)
        labels_path: Optional path to the ``Id,Category`` labels CSV for 'train'
        classes: Allowed class names
        
    Returns:
        Summary dict with shapes, dtype and per-class label counts
        
    Raises:
        FileNotFoundError: If a file is missing
        ValueError: If shapes, row counts or class names are inconsistent
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Data file not found: {hdf5_path}")
        
    summary: Dict[str, object] = {"hdf5": hdf5_path}
    with h5py.File(hdf5_path, 'r') as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"{hdf5_path}: HDF5 file contains no datasets")
        train_key = 'train' if 'train' in f else ('test' if 'test' in f else keys[0])
        for key in keys:
            shape = tuple(f[key].shape)
            if key in ('train', 'test') or key == train_key:
                if len(shape) != 3 or shape[2] != 2:
                    raise ValueError(
                        f"{hdf5_path}: dataset '{key}' has shape {shape}, expected (N, T, 2)"
                    )
            summary[f"{key}_shape"] = shape
        summary["train_key"] = train_key
        summary["dtype"] = str(f[train_key].dtype)
        n_train = f[train_key].shape[0]
        
    if labels_path is not None:
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with open(labels_path, 'r') as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None or len(header) < 2:
                raise ValueError(f"{labels_path}: expected a header row like 'Id,Category'")
            categories = [row[1] for row in reader if len(row) >= 2]
        if len(categories) != n_train:
            raise ValueError(
                f"{labels_path} has {len(categories)} labels but '{train_key}' has {n_train} samples"
            )
        unknown = sorted(set(categories) - set(classes))
        if unknown:
            raise ValueError(f"{labels_path}: unknown classes {unknown}; expected {list(classes)}")
        counts = Counter(categories)
        summary["labels"] = labels_path
        summary["class_counts"] = {c: counts.get(c, 0) for c in classes}
        
    return summary

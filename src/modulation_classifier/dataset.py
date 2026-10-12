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


def prepare_complex_data(data: np.ndarray) -> np.ndarray:
    """Prepare I/Q data for complex-valued neural networks.
    
    Converts data from (N, 1024, 2) format to (N, 2, 1024, 1) format
    where the first dimension represents real/imaginary channels.
    
    Args:
        data: Input data of shape (N, 1024, 2) where last dim is [I, Q]
        
    Returns:
        Reshaped data of shape (N, 2, 1024, 1) for complex processing
    """
    if data.ndim != 3 or data.shape[-1] != 2:
        raise ValueError(f"Expected data shape (N, 1024, 2), got {data.shape}")
    
    # Transpose to (N, 2, 1024) and add width dimension
    complex_data = data.transpose(0, 2, 1)  # (N, 2, 1024)
    complex_data = complex_data[:, :, :, np.newaxis]  # (N, 2, 1024, 1)
    
    return complex_data


def prepare_standard_data(data: np.ndarray) -> np.ndarray:
    """Prepare I/Q data for standard real-valued neural networks.
    
    Converts data from (N, 1024, 2) to (N, 1, 1024, 2) format.
    
    Args:
        data: Input data of shape (N, 1024, 2)
        
    Returns:  
        Reshaped data of shape (N, 1, 1024, 2) for standard processing
    """
    if data.ndim != 3 or data.shape[-1] != 2:
        raise ValueError(f"Expected data shape (N, 1024, 2), got {data.shape}")
    
    # Add channel dimension
    return data[:, np.newaxis, :, :]  # (N, 1, 1024, 2)


class IQDataset(Dataset):
    """PyTorch dataset for IQ modulation data."""
    
    def __init__(
        self, 
        data: Union[str, np.ndarray], 
        labels: Optional[Union[str, np.ndarray]] = None,
        classes: List[str] = MODULATION_CLASSES,
        transform: Optional[callable] = None,
        data_format: str = "standard"
    ):
        """Initialize IQ dataset.
        
        Args:
            data: Path to HDF5 file or numpy array of IQ data
            labels: Path to CSV file or numpy array of labels (optional for test data)
            classes: List of modulation class names
            transform: Optional transform to apply to samples
            data_format: Data format - "standard" for (N,1,T,2) or "complex" for (N,2,T,1)
        """
        self.classes = classes
        self.transform = transform
        self.data_format = data_format
        
        # Load data
        if isinstance(data, str):
            raw_data = self._load_hdf5_data(data)
        else:
            raw_data = data
            
        # Store raw data for potential augmentation
        self._raw_data = raw_data.copy()
            
        # Validate and format data
        if len(raw_data.shape) != 3 or raw_data.shape[2] != 2:
            raise ValueError(f"Expected data shape (N, T, 2), got {raw_data.shape}")
            
        # Apply data format transformation
        if data_format == "complex":
            self.data = prepare_complex_data(raw_data)
        elif data_format == "standard":
            self.data = prepare_standard_data(raw_data)
        else:
            raise ValueError(f"Unknown data format: {data_format}. Use 'standard' or 'complex'")
            
        # Load labels if provided
        if labels is not None:
            if isinstance(labels, str):
                self.labels = self._load_csv_labels(labels)
            else:
                self.labels = labels
        else:
            self.labels = None
            
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
        # Data is already formatted by __init__, just convert to tensor
        sample = torch.from_numpy(self.data[idx]).float()
        
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
    augmentation_factor: int = 2,
    methods: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create augmented dataset with various RF-specific augmentation methods.
    
    Args:
        data: Original IQ data array (N, T, 2)
        labels: Original labels array  
        augmentation_factor: Number of augmented copies per sample
        methods: List of augmentation methods to apply. Available:
                ['rotation', 'amplitude', 'awgn', 'frequency_shift', 'time_shift', 'iq_imbalance']
        
    Returns:
        Tuple of (augmented_data, augmented_labels)
    """
    if methods is None:
        methods = ['rotation', 'amplitude', 'awgn']  # Default high-impact methods
        
    print(f"Creating augmented dataset (factor={augmentation_factor})...", flush=True)
    print(f"Using augmentation methods: {methods}", flush=True)
    
    augmented_data = [data]
    augmented_labels = [labels]
    
    # Define augmentation techniques
    available_methods = {
        'rotation': _apply_phase_rotation,
        'amplitude': _apply_amplitude_scaling, 
        'awgn': _apply_awgn,
        'frequency_shift': _apply_frequency_shift,
        'time_shift': _apply_time_shift,
        'iq_imbalance': _apply_iq_imbalance
    }
    
    # Apply selected methods up to augmentation_factor
    method_cycle = methods * ((augmentation_factor // len(methods)) + 1)
    
    for i in range(augmentation_factor):
        method = method_cycle[i]
        if method in available_methods:
            print(f"Applying {method} augmentation...", flush=True)
            augmented_sample = available_methods[method](data)
            augmented_data.append(augmented_sample)
            augmented_labels.append(labels.copy())
    
    # Concatenate all data
    final_data = np.concatenate(augmented_data, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)
    
    print(f"Augmented dataset: {final_data.shape[0]} samples", flush=True)
    return final_data, final_labels


def _apply_phase_rotation(data: np.ndarray, angles: List[float] = None) -> np.ndarray:
    """Apply phase rotation to IQ data."""
    if angles is None:
        angles = [np.pi/2, np.pi, 3*np.pi/2]
    
    angle = np.random.choice(angles)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_data = []
    for sample in data:
        rotated_sample = np.dot(sample, rot_matrix)
        rotated_data.append(rotated_sample)
    
    return np.array(rotated_data)


def _apply_amplitude_scaling(data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Apply random amplitude scaling to simulate signal strength variations."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1], size=(data.shape[0], 1, 1))
    return data * scale_factor


def _apply_awgn(data: np.ndarray, snr_db_range: Tuple[float, float] = (10, 20)) -> np.ndarray:
    """Add Additive White Gaussian Noise (AWGN) to simulate channel effects."""
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate signal power per sample
    signal_power = np.mean(np.sum(data ** 2, axis=-1), axis=-1)  # Shape: (N,)
    noise_power = signal_power / snr_linear  # Shape: (N,)
    
    # Generate noise with proper broadcasting
    noise = np.random.normal(0, 1, data.shape)  # Standard normal noise
    
    # Scale noise by sqrt(noise_power/2) per sample
    noise_std = np.sqrt(noise_power / 2)  # Shape: (N,)
    noise_std = noise_std[:, np.newaxis, np.newaxis]  # Shape: (N, 1, 1) for broadcasting
    
    return data + noise * noise_std


def _apply_frequency_shift(data: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply frequency offset to simulate Doppler effects and oscillator drift."""
    freq_shift = np.random.uniform(-max_shift, max_shift)
    n_samples = data.shape[1]
    
    # Create frequency shift in complex domain
    t = np.arange(n_samples)
    freq_offset = np.exp(1j * 2 * np.pi * freq_shift * t)
    
    shifted_data = []
    for sample in data:
        # Convert to complex, apply shift, convert back
        complex_signal = sample[:, 0] + 1j * sample[:, 1]
        shifted_complex = complex_signal * freq_offset
        shifted_sample = np.column_stack([shifted_complex.real, shifted_complex.imag])
        shifted_data.append(shifted_sample)
    
    return np.array(shifted_data)


def _apply_time_shift(data: np.ndarray, max_shift_samples: int = None) -> np.ndarray:
    """Apply circular time shift to simulate timing synchronization errors."""
    if max_shift_samples is None:
        max_shift_samples = min(64, data.shape[1] // 8)  # Max 8% of signal length
    
    shift_samples = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    
    shifted_data = []
    for sample in data:
        shifted_sample = np.roll(sample, shift_samples, axis=0)
        shifted_data.append(shifted_sample)
    
    return np.array(shifted_data)


def _apply_iq_imbalance(data: np.ndarray, gain_range: Tuple[float, float] = (0.9, 1.1),
                       phase_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
    """Apply IQ imbalance to simulate hardware imperfections."""
    gain_imbalance = np.random.uniform(gain_range[0], gain_range[1])
    phase_imbalance = np.random.uniform(phase_range[0], phase_range[1])
    
    imbalanced_data = []
    for sample in data:
        i_channel = sample[:, 0]
        q_channel = sample[:, 1] * gain_imbalance
        
        # Apply phase imbalance
        q_channel_imbalanced = (q_channel * np.cos(phase_imbalance) + 
                               i_channel * np.sin(phase_imbalance))
        
        imbalanced_sample = np.column_stack([i_channel, q_channel_imbalanced])
        imbalanced_data.append(imbalanced_sample)
    
    return np.array(imbalanced_data)


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
    
    # Create datasets with raw data already formatted
    train_dataset = IQDataset.__new__(IQDataset)
    train_dataset.classes = dataset.classes
    train_dataset.transform = dataset.transform
    train_dataset.data_format = dataset.data_format
    train_dataset.data = train_data
    train_dataset.labels = train_labels
    train_dataset._raw_data = None  # Not needed for subsets
    
    val_dataset = IQDataset.__new__(IQDataset)
    val_dataset.classes = dataset.classes
    val_dataset.transform = dataset.transform
    val_dataset.data_format = dataset.data_format
    val_dataset.data = val_data
    val_dataset.labels = val_labels
    val_dataset._raw_data = None  # Not needed for subsets
    
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

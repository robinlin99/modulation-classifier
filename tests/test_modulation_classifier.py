"""Unit tests for the modulation classifier package."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
import os

from modulation_classifier import (
    ModulationClassifier,
    IQDataset, 
    ImprovedNet,
    ComplexNet,
    create_model,
    count_parameters
)
from modulation_classifier.dataset import (
    create_augmented_dataset,
    validate_dataset_files,
    MODULATION_CLASSES,
)


class TestModels:
    """Test neural network models."""
    
    def test_improved_net_creation(self):
        """Test ImprovedNet model creation."""
        model = ImprovedNet(num_classes=10)
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 1024, 2)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        
    def test_create_model_factory(self):
        """Test model factory function."""
        # Test improved model
        model = create_model("improved", num_classes=8)
        assert isinstance(model, ImprovedNet)
        
        # Test complex model
        model = create_model("complex", num_classes=8)
        assert isinstance(model, ComplexNet)
        
        # Test invalid architecture
        with pytest.raises(ValueError):
            create_model("invalid_arch")
            
    def test_parameter_counting(self):
        """Test parameter counting utility."""
        model = ImprovedNet(num_classes=10)
        param_count = count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)


class TestDataset:
    """Test dataset functionality."""
    
    def test_iq_dataset_creation(self):
        """Test IQ dataset creation with numpy arrays."""
        # Create dummy data
        n_samples, n_timesteps = 100, 1024
        data = np.random.randn(n_samples, n_timesteps, 2)
        labels = np.random.randint(0, 10, n_samples)
        
        # Test default standard format
        dataset = IQDataset(data, labels, MODULATION_CLASSES)
        
        assert len(dataset) == n_samples
        assert dataset.data.shape == (n_samples, 1, n_timesteps, 2)  # Standard format: (N,1,T,2)
        assert len(dataset.labels) == n_samples
        assert dataset.data_format == "standard"
        
        # Test complex format
        dataset_complex = IQDataset(data, labels, MODULATION_CLASSES, data_format="complex")
        assert dataset_complex.data.shape == (n_samples, 2, n_timesteps, 1)  # Complex format: (N,2,T,1)
        assert dataset_complex.data_format == "complex"
        
    def test_iq_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        n_samples, n_timesteps = 50, 1024
        data = np.random.randn(n_samples, n_timesteps, 2)
        labels = np.random.randint(0, 10, n_samples)
        
        # Test standard format
        dataset = IQDataset(data, labels, MODULATION_CLASSES)
        
        # Test getting item with labels
        sample, label = dataset[0]
        assert sample.shape == (1, n_timesteps, 2)  # Standard format: (1, T, 2)
        assert isinstance(sample, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        
        # Test complex format
        dataset_complex = IQDataset(data, labels, MODULATION_CLASSES, data_format="complex")
        sample_complex, label_complex = dataset_complex[0]
        assert sample_complex.shape == (2, n_timesteps, 1)  # Complex format: (2, T, 1)
        
        # Test dataset without labels
        dataset_no_labels = IQDataset(data, classes=MODULATION_CLASSES)
        sample_only = dataset_no_labels[0]
        assert sample_only.shape == (1, n_timesteps, 2)
        
    def test_data_augmentation(self):
        """Test data augmentation function."""
        n_samples, n_timesteps = 20, 1024
        data = np.random.randn(n_samples, n_timesteps, 2)
        labels = np.random.randint(0, 10, n_samples)
        
        aug_data, aug_labels = create_augmented_dataset(
            data, labels, augmentation_factor=2
        )
        
        # Should have original + 2 augmented versions = 3x data
        expected_samples = n_samples * 3
        assert aug_data.shape[0] == expected_samples
        assert len(aug_labels) == expected_samples
        
    def test_invalid_data_shape(self):
        """Test handling of invalid data shapes."""
        invalid_data = np.random.randn(100, 1024)  # Missing IQ dimension
        labels = np.random.randint(0, 10, 100)
        
        with pytest.raises(ValueError):
            IQDataset(invalid_data, labels)


class TestValidateDatasetFiles:
    """Test the cheap on-disk dataset validator used by setup_dataset.sh."""

    @staticmethod
    def _write(tmp_path, n_train=20, n_labels=None, categories=None, timesteps=16):
        import h5py
        hdf5 = tmp_path / "data.hdf5"
        with h5py.File(hdf5, "w") as f:
            f.create_dataset("train", data=np.zeros((n_train, timesteps, 2), np.float32))
            f.create_dataset("test", data=np.zeros((5, timesteps, 2), np.float32))
        n_labels = n_train if n_labels is None else n_labels
        if categories is None:
            categories = [MODULATION_CLASSES[i % 10] for i in range(n_labels)]
        csv_path = tmp_path / "train_labels.csv"
        csv_path.write_text("Id,Category\n" + "".join(f"{i},{c}\n" for i, c in enumerate(categories)))
        return str(hdf5), str(csv_path)

    def test_valid_pair(self, tmp_path):
        hdf5, csv_path = self._write(tmp_path)
        s = validate_dataset_files(hdf5, csv_path)
        assert s["train_shape"] == (20, 16, 2)
        assert s["test_shape"] == (5, 16, 2)
        assert s["train_key"] == "train"
        assert sum(s["class_counts"].values()) == 20
        assert set(s["class_counts"]) == set(MODULATION_CLASSES)

    def test_hdf5_only(self, tmp_path):
        hdf5, _ = self._write(tmp_path)
        s = validate_dataset_files(hdf5)
        assert "class_counts" not in s

    def test_label_count_mismatch(self, tmp_path):
        hdf5, csv_path = self._write(tmp_path, n_train=20, n_labels=19)
        with pytest.raises(ValueError, match="19 labels"):
            validate_dataset_files(hdf5, csv_path)

    def test_unknown_class(self, tmp_path):
        cats = [MODULATION_CLASSES[0]] * 19 + ["GMSK"]
        hdf5, csv_path = self._write(tmp_path, categories=cats)
        with pytest.raises(ValueError, match="GMSK"):
            validate_dataset_files(hdf5, csv_path)

    def test_bad_shape(self, tmp_path):
        import h5py
        hdf5 = tmp_path / "bad.hdf5"
        with h5py.File(hdf5, "w") as f:
            f.create_dataset("train", data=np.zeros((4, 16), np.float32))
        with pytest.raises(ValueError, match="expected \\(N, T, 2\\)"):
            validate_dataset_files(str(hdf5))

    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_dataset_files(str(tmp_path / "nope.hdf5"))
        hdf5, _ = self._write(tmp_path)
        with pytest.raises(FileNotFoundError):
            validate_dataset_files(hdf5, str(tmp_path / "nope.csv"))


class TestClassifier:
    """Test main classifier functionality."""
    
    def test_classifier_creation(self):
        """Test classifier creation."""
        classifier = ModulationClassifier()
        assert classifier.device is not None
        assert classifier.model is not None
        assert len(classifier.classes) == 10
        
    def test_classifier_with_custom_classes(self):
        """Test classifier with custom class list."""
        custom_classes = ["BPSK", "QPSK", "8PSK"]
        classifier = ModulationClassifier(
            num_classes=len(custom_classes),
            classes=custom_classes
        )
        assert classifier.classes == custom_classes
        assert classifier.num_classes == 3
        
    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_selection(self, mock_cuda, mock_mps):
        """Test device selection falls back to CPU when no accelerator exists."""
        classifier = ModulationClassifier()
        assert classifier.device.type == 'cpu'
        
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_selection_prefers_mps_over_cpu(self, mock_cuda, mock_mps):
        """Test that MPS is auto-selected when CUDA is unavailable."""
        from modulation_classifier.classifier import get_default_device
        assert get_default_device().type == 'mps'
        
    def test_predict_without_training(self):
        """Test that prediction fails without training."""
        classifier = ModulationClassifier()
        dummy_data = np.random.randn(10, 1024, 2)
        
        with pytest.raises(RuntimeError):
            classifier.predict(dummy_data)
            
    def test_training_with_dummy_data(self):
        """Test training with dummy data."""
        # Create small dummy dataset with balanced labels so the stratified
        # train/val split always has >= 2 samples per class
        n_samples, n_timesteps = 50, 1024
        data = np.random.randn(n_samples, n_timesteps, 2)
        labels = np.arange(n_samples) % 10
        dataset = IQDataset(data, labels, MODULATION_CLASSES)
        
        classifier = ModulationClassifier()
        
        # Train for just 1 epoch
        history = classifier.train(
            dataset=dataset,
            num_epochs=1,
            batch_size=16,
            use_augmentation=False,
            patience=1
        )
        
        assert classifier.trained == True
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) >= 1
        
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Create and train a small model
        n_samples, n_timesteps = 30, 1024
        data = np.random.randn(n_samples, n_timesteps, 2)
        labels = np.arange(n_samples) % 5  # Balanced over 5 classes
        
        custom_classes = MODULATION_CLASSES[:5]
        dataset = IQDataset(data, labels, custom_classes)
        
        classifier1 = ModulationClassifier(
            num_classes=5, 
            classes=custom_classes
        )
        
        classifier1.train(
            dataset=dataset,
            num_epochs=1,
            batch_size=16,
            use_augmentation=False
        )
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
        
        try:
            classifier1.save_model(model_path)
            
            # Create new classifier and load model
            classifier2 = ModulationClassifier(
                num_classes=5,
                classes=custom_classes
            )
            classifier2.load_model(model_path)
            
            # Test that loaded model can make predictions
            predictions = classifier2.predict(dataset)
            assert len(predictions) == n_samples
            assert all(0 <= pred < 5 for pred in predictions)
            
        finally:
            # Clean up temp file
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestUtils:
    """Test utility functions."""
    
    def test_plot_functions_dont_crash(self):
        """Test that plotting functions don't crash."""
        from modulation_classifier.utils import (
            plot_training_curves,
            analyze_class_distribution
        )
        
        # Mock training history
        history = {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [1.2, 0.9, 0.7],
            'train_acc': [30, 50, 70],
            'val_acc': [25, 45, 65]
        }
        
        # These should not crash (matplotlib may not display in CI)
        with patch('matplotlib.pyplot.show'):
            plot_training_curves(history)
            
        # Test class distribution analysis
        labels = np.random.randint(0, 3, 100)
        classes = ['A', 'B', 'C']
        
        with patch('matplotlib.pyplot.show'):
            df = analyze_class_distribution(labels, classes)
            
        assert len(df) == 3
        assert 'Class' in df.columns
        assert 'Count' in df.columns


def test_package_imports():
    """Test that all main components can be imported."""
    from modulation_classifier import (
        ModulationClassifier,
        IQDataset,
        ImprovedNet,
        create_model
    )
    
    # Just test that imports work
    assert ModulationClassifier is not None
    assert IQDataset is not None
    assert ImprovedNet is not None
    assert create_model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
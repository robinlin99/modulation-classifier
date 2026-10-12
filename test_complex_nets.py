"""Test complex-valued neural network implementation."""

import pytest
import torch
import numpy as np
from modulation_classifier.complex_layers import (
    ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexModulus, ComplexLinear
)
from modulation_classifier.models import ComplexNet
from modulation_classifier.dataset import prepare_complex_data, prepare_standard_data


class TestComplexLayers:
    """Test individual complex-valued layers."""
    
    def test_complex_conv2d_shapes(self):
        """Test that ComplexConv2d produces correct output shapes."""
        conv = ComplexConv2d(1, 4, kernel_size=3, padding=1)  # 1 complex in -> 4 complex out
        
        # Input: batch=2, channels=2 (1 complex = real+imag), height=10, width=1  
        x = torch.randn(2, 2, 10, 1)
        
        output = conv(x)
        
        # Should have 8 channels (4 complex = 4 real + 4 imag)
        assert output.shape == (2, 8, 10, 1)
        
    def test_complex_conv2d_wrong_channels(self):
        """Test that ComplexConv2d raises error with wrong input channels."""
        conv = ComplexConv2d(1, 4, kernel_size=3)  # Expects 1 complex input channel
        
        # Wrong input: 3 total channels (not even, can't represent complex numbers)
        x = torch.randn(1, 3, 10, 1)
        
        with pytest.raises(ValueError, match="Expected even number of channels"):
            conv(x)
            
    def test_complex_batch_norm(self):
        """Test complex batch normalization."""
        bn = ComplexBatchNorm2d(4)
        
        # Input with 8 channels (4 real + 4 imaginary)
        x = torch.randn(2, 8, 10, 1)
        
        output = bn(x)
        assert output.shape == x.shape
        
    def test_complex_relu(self):
        """Test complex ReLU activation."""
        relu = ComplexReLU()
        
        # Input with mixed positive/negative values
        x = torch.tensor([[[[-1.0]], [[2.0]], [[-3.0]], [[4.0]]]])  # Shape: (1, 4, 1, 1)
        
        output = relu(x)
        
        # ReLU applied separately to real and imaginary parts
        expected = torch.tensor([[[[0.0]], [[2.0]], [[0.0]], [[4.0]]]])
        assert torch.allclose(output, expected)
        
    def test_complex_modulus(self):
        """Test complex modulus operation."""
        modulus = ComplexModulus()
        
        # Create complex input: 3+4j should give magnitude 5
        x = torch.tensor([[[[3.0]], [[4.0]]]])  # Shape: (1, 2, 1, 1)
        
        output = modulus(x)
        
        # Should compute sqrt(3^2 + 4^2) = 5
        expected_magnitude = torch.sqrt(torch.tensor([[[[3.0**2 + 4.0**2]]]]))
        assert torch.allclose(output, expected_magnitude)
        
    def test_complex_linear(self):
        """Test complex linear layer."""
        linear = ComplexLinear(4, 2)  # 2 complex inputs -> 1 complex output
        
        # Input: batch=1, features=4 (2 real + 2 imaginary)
        x = torch.randn(1, 4)
        
        output = linear(x)
        assert output.shape == (1, 4)  # 2 complex outputs = 4 real features


class TestComplexNet:
    """Test the ComplexNet model."""
    
    def test_complex_net_forward(self):
        """Test ComplexNet forward pass."""
        model = ComplexNet(num_classes=10)
        
        # Create input in complex format: (batch, 2, time, 1)
        batch_size = 4
        time_steps = 1024
        x = torch.randn(batch_size, 2, time_steps, 1)
        
        output = model(x)
        
        # Should output class logits
        assert output.shape == (batch_size, 10)
        
    def test_complex_net_with_standard_input(self):
        """Test that ComplexNet raises error with wrong input format."""
        model = ComplexNet(num_classes=10)
        
        # Wrong format: standard format (batch, 1, time, 2) instead of complex (batch, 2, time, 1)
        x = torch.randn(2, 1, 1024, 2)
        
        with pytest.raises(ValueError):
            model(x)


class TestDataPreprocessing:
    """Test data preprocessing functions."""
    
    def test_prepare_complex_data(self):
        """Test complex data preparation."""
        # Original format: (N, T, 2)
        data = np.random.randn(10, 1024, 2)
        
        complex_data = prepare_complex_data(data)
        
        # Should become: (N, 2, T, 1)
        assert complex_data.shape == (10, 2, 1024, 1)
        
        # Check data preservation
        assert np.allclose(data[0, :, 0], complex_data[0, 0, :, 0])  # Real part
        assert np.allclose(data[0, :, 1], complex_data[0, 1, :, 0])  # Imaginary part
        
    def test_prepare_standard_data(self):
        """Test standard data preparation."""
        # Original format: (N, T, 2)
        data = np.random.randn(10, 1024, 2)
        
        standard_data = prepare_standard_data(data)
        
        # Should become: (N, 1, T, 2)
        assert standard_data.shape == (10, 1, 1024, 2)
        
        # Check data preservation
        assert np.allclose(data, standard_data[:, 0, :, :])
        
    def test_wrong_input_shape(self):
        """Test error handling for wrong input shapes."""
        # Wrong shape data
        data = np.random.randn(10, 1024)  # Missing IQ dimension
        
        with pytest.raises(ValueError, match="Expected data shape"):
            prepare_complex_data(data)
            
        with pytest.raises(ValueError, match="Expected data shape"):
            prepare_standard_data(data)


def test_parameter_count_comparison():
    """Compare parameter counts between architectures."""
    from modulation_classifier.models import count_parameters, create_model
    
    improved_net = create_model("improved", num_classes=10)
    complex_net = create_model("complex", num_classes=10)
    
    improved_params = count_parameters(improved_net)
    complex_params = count_parameters(complex_net)
    
    print(f"ImprovedNet parameters: {improved_params:,}")
    print(f"ComplexNet parameters: {complex_params:,}")
    
    # Complex net should have roughly 2x parameters in conv layers due to real/imag weights
    # But should be smaller overall due to different architecture
    assert complex_params > 0
    assert improved_params > 0


if __name__ == "__main__":
    # Run basic tests
    test_complex_conv = TestComplexLayers()
    test_complex_conv.test_complex_conv2d_shapes()
    test_complex_conv.test_complex_modulus()
    
    test_model = TestComplexNet()
    test_model.test_complex_net_forward()
    
    test_data = TestDataPreprocessing()
    test_data.test_prepare_complex_data()
    test_data.test_prepare_standard_data()
    
    test_parameter_count_comparison()
    
    print("All tests passed!")
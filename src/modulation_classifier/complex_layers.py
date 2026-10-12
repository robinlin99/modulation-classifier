"""Complex-valued neural network layers for I/Q signal processing.

Implements complex convolutions using the linear combination method for better
I/Q signal processing in modulation classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution layer.
    
    Implements complex convolution using the linear combination method:
    Conv(z) = Conv(Re(z)) - Conv(Im(z)) + j(Conv(Re(z)) + Conv(Im(z)))
    
    This allows the network to learn joint correlations between I and Q components
    that are lost when treating them as separate real channels.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input to output channels
        bias: If True, adds a learnable bias to the output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super(ComplexConv2d, self).__init__()
        
        # Real and imaginary convolution layers
        self.conv_real = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias
        )
        self.conv_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for complex convolution.
        
        Args:
            x: Input tensor where channels represent complex numbers as [real_channels..., imag_channels...]
               Shape: (batch, 2*in_channels, height, width)
               
        Returns:
            Complex tensor with same shape format (batch, 2*out_channels, height, width)
        """
        if x.size(1) % 2 != 0:
            raise ValueError(f"Expected even number of channels for complex data, got {x.size(1)}")
            
        num_complex_channels = x.size(1) // 2
        if num_complex_channels != self.conv_real.in_channels:
            raise ValueError(f"Expected {self.conv_real.in_channels} complex input channels, got {num_complex_channels}")
            
        real_part = x[:, :num_complex_channels]  # First half: real parts
        imag_part = x[:, num_complex_channels:]  # Second half: imaginary parts
        
        # Complex convolution: (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
        conv_real_real = self.conv_real(real_part)
        conv_real_imag = self.conv_real(imag_part)
        conv_imag_real = self.conv_imag(real_part)
        conv_imag_imag = self.conv_imag(imag_part)
        
        # Real part: Re(w)*Re(x) - Im(w)*Im(x)
        output_real = conv_real_real - conv_imag_imag
        # Imaginary part: Re(w)*Im(x) + Im(w)*Re(x)  
        output_imag = conv_real_imag + conv_imag_real
        
        return torch.cat([output_real, output_imag], dim=1)


class ComplexBatchNorm2d(nn.Module):
    """Complex-valued 2D batch normalization.
    
    Normalizes both real and imaginary parts separately.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(ComplexBatchNorm2d, self).__init__()
        
        self.bn_real = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize real and imaginary parts separately."""
        if x.size(1) % 2 != 0:
            raise ValueError(f"Expected even number of channels, got {x.size(1)}")
            
        num_complex_channels = x.size(1) // 2
        
        real_part = x[:, :num_complex_channels]
        imag_part = x[:, num_complex_channels:]
        
        real_normed = self.bn_real(real_part)
        imag_normed = self.bn_imag(imag_part)
        
        return torch.cat([real_normed, imag_normed], dim=1)


class ComplexReLU(nn.Module):
    """Complex ReLU activation function.
    
    Applies ReLU to both real and imaginary parts separately.
    This is the most common approach for complex activations.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) % 2 != 0:
            raise ValueError(f"Expected even number of channels, got {x.size(1)}")
            
        num_complex_channels = x.size(1) // 2
        
        real_part = x[:, :num_complex_channels]
        imag_part = x[:, num_complex_channels:]
        
        real_activated = F.relu(real_part)
        imag_activated = F.relu(imag_part)
        
        return torch.cat([real_activated, imag_activated], dim=1)


class ComplexModulus(nn.Module):
    """Complex modulus operation: |z| = sqrt(real^2 + imag^2).
    
    Converts complex tensor to real-valued magnitude.
    Used as final activation before real-valued layers.
    """
    
    def __init__(self, eps: float = 1e-8):
        super(ComplexModulus, self).__init__()
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) % 2 != 0:
            raise ValueError(f"Expected even number of channels, got {x.size(1)}")
            
        num_complex_channels = x.size(1) // 2
        
        real_part = x[:, :num_complex_channels]
        imag_part = x[:, num_complex_channels:]
        
        # Magnitude: sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + self.eps)
        
        return magnitude


class ComplexLinear(nn.Module):
    """Complex-valued linear layer.
    
    Implements complex linear transformation for fully connected layers.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(ComplexLinear, self).__init__()
        
        self.linear_real = nn.Linear(in_features, out_features, bias=bias)
        self.linear_imag = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complex linear transformation."""
        if x.size(-1) % 2 != 0:
            raise ValueError(f"Expected even number of features, got {x.size(-1)}")
            
        num_complex_features = x.size(-1) // 2
        
        real_part = x[..., :num_complex_features]
        imag_part = x[..., num_complex_features:]
        
        # Complex multiplication: (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
        linear_real_real = self.linear_real(real_part)
        linear_real_imag = self.linear_real(imag_part)
        linear_imag_real = self.linear_imag(real_part)
        linear_imag_imag = self.linear_imag(imag_part)
        
        output_real = linear_real_real - linear_imag_imag
        output_imag = linear_real_imag + linear_imag_real
        
        return torch.cat([output_real, output_imag], dim=-1)


def complex_adaptive_avg_pool2d(x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    """Complex-valued adaptive average pooling.
    
    Applies pooling to real and imaginary parts separately.
    """
    if x.size(1) % 2 != 0:
        raise ValueError(f"Expected even number of channels, got {x.size(1)}")
        
    num_complex_channels = x.size(1) // 2
    
    real_part = x[:, :num_complex_channels]
    imag_part = x[:, num_complex_channels:]
    
    real_pooled = F.adaptive_avg_pool2d(real_part, output_size)
    imag_pooled = F.adaptive_avg_pool2d(imag_part, output_size)
    
    return torch.cat([real_pooled, imag_pooled], dim=1)
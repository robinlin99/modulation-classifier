"""Neural network models for modulation classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ImprovedNet(nn.Module):
    """Default CNN for modulation classification (the ``"improved"`` architecture).
    
    Input is a batch of IQ frames shaped ``(N, 1, T, 2)``; output is ``(N, num_classes)``
    logits (use ``torch.softmax`` for probabilities).
    
    Layer stack::
    
        conv1  1 -> 256   kernel (3, 1)   + BN + ReLU      (temporal features per I/Q channel)
        conv2  256 -> 256 kernel (3, 2)   + BN + ReLU      (mixes the I and Q channels)
        conv3  256 -> 128 kernel (3, 1)   + BN + ReLU
        conv4  128 -> 80  kernel (3, 1)   + BN + ReLU
        global average pool -> 80
        dropout -> fc1 80 -> 128 + ReLU -> fc2 128 -> num_classes
    
    Compared with :class:`LegacyNet`:
    
    - a separate :class:`~torch.nn.BatchNorm2d` after every convolution
      (the legacy model shared two BN layers across four convolutions)
    - global average pooling instead of flattening into a 81,280-wide linear
      layer, which cuts the parameter count from ~10.9M to ~537K
    - dropout before the classifier head
    
    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability applied before ``fc1``.
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(ImprovedNet, self).__init__()
        
        # Convolutional layers with separate batch norm
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(3, 2), stride=1, padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 80, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn4 = nn.BatchNorm2d(80)
        
        # Global average pooling instead of massive linear layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(80, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 1024, 2)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional layers with ReLU activations
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps from each convolutional layer
        """
        features = []
        
        x = F.relu(self.bn1(self.conv1(x)))
        features.append(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        features.append(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        features.append(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        features.append(x)
        
        return features


class LegacyNet(nn.Module):
    """The original ECE 4200 competition network (the ``"legacy"`` architecture).
    
    Kept for comparison. Known quirks are preserved on purpose: two
    batch-norm layers are shared across the four convolutions, there is no
    padding, and the output is ``log_softmax`` rather than raw logits. The
    flattened 81,280-wide linear layer accounts for most of its ~10.9M
    parameters. Only valid for ``T = 1024`` inputs.
    
    Args:
        num_classes: Number of output classes.
    """
    
    def __init__(self, num_classes: int = 10):
        super(LegacyNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(3, 1), stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=256)
        self.batchnorm2 = nn.BatchNorm2d(num_features=80)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(3, 2), stride=1)
        self.conv3 = nn.Conv2d(256, 80, kernel_size=(3, 1), stride=1)
        self.conv4 = nn.Conv2d(80, 80, kernel_size=(3, 1), stride=1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(81280, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape ``(batch_size, 1, 1024, 2)``
            
        Returns:
            Log-probabilities of shape ``(batch_size, num_classes)``
        """
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm1(self.conv2(x)))  # Note: shared batchnorm
        x = F.selu(self.batchnorm2(self.conv3(x)))
        x = F.selu(self.batchnorm2(self.conv4(x)))  # Note: shared batchnorm
        x = x.view(-1, 81280)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_model(
    architecture: str = "improved", 
    num_classes: int = 10, 
    **kwargs
) -> nn.Module:
    """Factory function to create models.
    
    Args:
        architecture: Model architecture ("improved" or "legacy")
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        PyTorch model instance
    """
    if architecture == "improved":
        return ImprovedNet(num_classes=num_classes, **kwargs)
    elif architecture == "legacy":
        return LegacyNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
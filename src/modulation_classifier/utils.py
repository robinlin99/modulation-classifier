"""Utility functions for visualization and analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_training_curves(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves  
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8)
):
    """Plot confusion matrix with nice formatting.
    
    Args:
        cm: Confusion matrix array
        classes: List of class names
        save_path: Optional path to save the plot
        normalize: Whether to normalize the matrix
        figsize: Figure size tuple
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_constellation(
    iq_data: np.ndarray, 
    classes: List[str],
    class_labels: Optional[np.ndarray] = None,
    max_samples_per_class: int = 100,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """Plot IQ constellation diagrams for each modulation class.
    
    Args:
        iq_data: IQ data array of shape (N, T, 2)
        classes: List of class names
        class_labels: Array of class labels (indices)
        max_samples_per_class: Maximum samples to plot per class
        figsize: Figure size tuple
        save_path: Optional path to save the plot
    """
    num_classes = len(classes)
    cols = 5
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    for class_idx, class_name in enumerate(classes):
        ax = axes[class_idx]
        
        if class_labels is not None:
            # Find samples of this class
            class_mask = (class_labels == class_idx)
            class_samples = iq_data[class_mask]
            
            if len(class_samples) > 0:
                # Limit number of samples for performance
                n_samples = min(len(class_samples), max_samples_per_class)
                selected_samples = class_samples[:n_samples]
                
                # Plot all points from selected samples
                for sample in selected_samples:
                    I = sample[:, 0]  # In-phase
                    Q = sample[:, 1]  # Quadrature
                    ax.scatter(I, Q, alpha=0.6, s=0.5)
        else:
            # Just plot first sample if no labels provided
            if class_idx < len(iq_data):
                sample = iq_data[class_idx]
                I = sample[:, 0]
                Q = sample[:, 1] 
                ax.scatter(I, Q, alpha=0.6, s=0.5)
        
        ax.set_title(class_name)
        ax.set_xlabel('I (In-phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(num_classes, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('IQ Constellation Diagrams by Modulation Type', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Constellation plot saved to {save_path}")
    
    plt.show()


def analyze_class_distribution(
    labels: np.ndarray, 
    classes: List[str],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Analyze and plot class distribution.
    
    Args:
        labels: Array of class labels (indices)
        classes: List of class names
        save_path: Optional path to save the plot
        
    Returns:
        DataFrame with class counts and percentages
    """
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create distribution DataFrame
    distribution = []
    for label_idx, count in zip(unique_labels, counts):
        class_name = classes[label_idx] if label_idx < len(classes) else f"Class_{label_idx}"
        percentage = 100.0 * count / len(labels)
        distribution.append({
            'Class': class_name,
            'Count': count,
            'Percentage': percentage
        })
    
    df = pd.DataFrame(distribution)
    
    # Plot distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    bars = ax1.bar(df['Class'], df['Count'])
    ax1.set_xlabel('Modulation Class')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution (Count)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, df['Count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(df['Count'], labels=df['Class'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()
    
    return df


def plot_feature_maps(
    model, 
    sample: np.ndarray, 
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """Visualize feature maps from convolutional layers.
    
    Args:
        model: Trained PyTorch model
        sample: Single IQ sample of shape (T, 2)  
        layer_names: Names of layers to visualize
        save_path: Optional path to save the plot
    """
    import torch
    
    # Prepare input
    model.eval()
    input_tensor = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0).float()
    
    if hasattr(model, 'get_feature_maps'):
        # Use built-in method if available
        with torch.no_grad():
            feature_maps = model.get_feature_maps(input_tensor)
    else:
        # Manual feature extraction
        feature_maps = []
        x = input_tensor
        
        # Assuming standard conv layer names
        conv_layers = ['conv1', 'conv2', 'conv3', 'conv4']
        bn_layers = ['bn1', 'bn2', 'bn3', 'bn4']
        
        with torch.no_grad():
            for conv_name, bn_name in zip(conv_layers, bn_layers):
                if hasattr(model, conv_name) and hasattr(model, bn_name):
                    conv = getattr(model, conv_name)
                    bn = getattr(model, bn_name)
                    x = torch.relu(bn(conv(x)))
                    feature_maps.append(x)
    
    # Plot feature maps
    num_layers = len(feature_maps)
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 3*num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for i, (feature_map, ax) in enumerate(zip(feature_maps, axes)):
        # Take first few channels for visualization
        feature_map = feature_map.squeeze(0)  # Remove batch dimension
        channels_to_show = min(8, feature_map.shape[0])
        
        # Create subplot for this layer
        fig_layer, axes_layer = plt.subplots(1, channels_to_show, figsize=(2*channels_to_show, 2))
        if channels_to_show == 1:
            axes_layer = [axes_layer]
            
        for j in range(channels_to_show):
            fmap = feature_map[j].cpu().numpy()
            axes_layer[j].imshow(fmap, cmap='viridis', aspect='auto')
            axes_layer[j].set_title(f'Ch {j}')
            axes_layer[j].axis('off')
        
        layer_name = layer_names[i] if layer_names else f'Layer {i+1}'
        fig_layer.suptitle(f'{layer_name} Feature Maps')
        plt.tight_layout()
        
        if save_path:
            layer_save_path = save_path.replace('.png', f'_layer_{i}.png')
            fig_layer.savefig(layer_save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def calculate_model_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    classes: List[str]
) -> Dict:
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        classes: List of class names
        
    Returns:
        Dictionary with various metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        classification_report, confusion_matrix
    )
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(classes):
        if i < len(precision):
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i], 
                'f1_score': f1[i],
                'support': support[i]
            }
    
    # Overall metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted metrics
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': per_class_metrics,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
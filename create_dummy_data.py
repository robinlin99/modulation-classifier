#!/usr/bin/env python3
"""
Create dummy IQ data for demonstration purposes.

Since the original data.hdf5 is not included in the repository (it's likely too large),
this script creates synthetic IQ signal data that matches the expected format.
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path

def create_dummy_iq_data():
    """Create synthetic IQ signal data that resembles real modulation signals."""
    
    # Configuration
    n_train_samples = 5000
    n_test_samples = 2000
    n_timesteps = 1024
    classes = ["FM", "OQPSK", "BPSK", "8PSK", "AM-SSB-SC", "4ASK", "16PSK", "AM-DSB-SC", "QPSK", "OOK"]
    
    print("Creating synthetic IQ modulation data...")
    
    # Create training data
    print(f"Generating {n_train_samples} training samples...")
    train_data = np.zeros((n_train_samples, n_timesteps, 2), dtype=np.float32)
    train_labels = []
    
    samples_per_class = n_train_samples // len(classes)
    
    for class_idx, modulation in enumerate(classes):
        start_idx = class_idx * samples_per_class
        end_idx = start_idx + samples_per_class
        
        print(f"  {modulation}: samples {start_idx}-{end_idx-1}")
        
        # Generate synthetic signals based on modulation type
        for i in range(start_idx, end_idx):
            # Base parameters
            t = np.linspace(0, 1, n_timesteps)
            noise_level = 0.1
            
            if modulation == "FM":
                # Frequency modulation
                freq = 2 + 0.5 * np.sin(2 * np.pi * t)
                phase = np.cumsum(freq) * 2 * np.pi / n_timesteps
                I = np.cos(phase)
                Q = np.sin(phase)
                
            elif modulation == "BPSK":
                # Binary Phase Shift Keying
                symbols = np.random.choice([-1, 1], n_timesteps)
                I = symbols * np.cos(2 * np.pi * 2 * t)
                Q = symbols * np.sin(2 * np.pi * 2 * t) * 0.1  # Small Q component
                
            elif modulation == "QPSK":
                # Quadrature Phase Shift Keying
                symbols_I = np.random.choice([-1, 1], n_timesteps)
                symbols_Q = np.random.choice([-1, 1], n_timesteps)
                I = symbols_I * np.cos(2 * np.pi * 2 * t)
                Q = symbols_Q * np.sin(2 * np.pi * 2 * t)
                
            elif modulation == "8PSK":
                # 8-Phase Shift Keying
                phases = np.random.choice(np.arange(0, 2*np.pi, 2*np.pi/8), n_timesteps)
                I = np.cos(phases)
                Q = np.sin(phases)
                
            elif modulation == "16PSK":
                # 16-Phase Shift Keying  
                phases = np.random.choice(np.arange(0, 2*np.pi, 2*np.pi/16), n_timesteps)
                I = np.cos(phases)
                Q = np.sin(phases)
                
            elif modulation == "4ASK":
                # 4-level Amplitude Shift Keying
                amplitudes = np.random.choice([0.25, 0.75, 1.25, 1.75], n_timesteps)
                I = amplitudes * np.cos(2 * np.pi * 2 * t)
                Q = amplitudes * np.sin(2 * np.pi * 2 * t) * 0.1
                
            elif modulation == "AM-SSB-SC":
                # AM Single Sideband Suppressed Carrier
                message = np.sin(2 * np.pi * 0.1 * t)
                I = message * np.cos(2 * np.pi * 2 * t)
                Q = message * np.sin(2 * np.pi * 2 * t)
                
            elif modulation == "AM-DSB-SC":
                # AM Double Sideband Suppressed Carrier
                message = np.sin(2 * np.pi * 0.1 * t)
                I = message * np.cos(2 * np.pi * 2 * t)
                Q = np.zeros_like(I)
                
            elif modulation == "OOK":
                # On-Off Keying
                symbols = np.random.choice([0, 1], n_timesteps)
                I = symbols * np.cos(2 * np.pi * 2 * t)
                Q = symbols * np.sin(2 * np.pi * 2 * t) * 0.1
                
            else:  # OQPSK
                # Offset Quadrature Phase Shift Keying
                symbols_I = np.random.choice([-1, 1], n_timesteps)
                symbols_Q = np.random.choice([-1, 1], n_timesteps)
                # Offset Q by half symbol
                symbols_Q = np.roll(symbols_Q, n_timesteps//4)
                I = symbols_I * np.cos(2 * np.pi * 2 * t)
                Q = symbols_Q * np.sin(2 * np.pi * 2 * t)
            
            # Add noise
            I += np.random.normal(0, noise_level, n_timesteps)
            Q += np.random.normal(0, noise_level, n_timesteps)
            
            # Store IQ data
            train_data[i, :, 0] = I
            train_data[i, :, 1] = Q
            train_labels.append(class_idx)
    
    # Handle remaining samples
    remaining = n_train_samples - len(train_labels)
    if remaining > 0:
        for i in range(remaining):
            class_idx = i % len(classes)
            train_labels.append(class_idx)
            # Copy from existing sample
            train_data[len(train_labels)-1] = train_data[class_idx * samples_per_class]
    
    # Create test data (unlabeled)
    print(f"Generating {n_test_samples} test samples...")
    test_data = np.zeros((n_test_samples, n_timesteps, 2), dtype=np.float32)
    
    for i in range(n_test_samples):
        class_idx = np.random.randint(0, len(classes))
        modulation = classes[class_idx]
        
        # Generate similar to training data
        t = np.linspace(0, 1, n_timesteps)
        noise_level = 0.1
        
        # Simplified generation for test data
        if class_idx < 5:  # First 5 classes
            I = np.cos(2 * np.pi * (class_idx + 1) * t)
            Q = np.sin(2 * np.pi * (class_idx + 1) * t) 
        else:  # Last 5 classes
            symbols = np.random.choice([-1, 1], n_timesteps)
            I = symbols * np.cos(2 * np.pi * 2 * t)
            Q = symbols * np.sin(2 * np.pi * 2 * t)
        
        # Add noise
        I += np.random.normal(0, noise_level, n_timesteps)
        Q += np.random.normal(0, noise_level, n_timesteps)
        
        test_data[i, :, 0] = I
        test_data[i, :, 1] = Q
    
    return train_data, np.array(train_labels), test_data, classes


def save_hdf5_data(train_data, test_data, filename="data/data.hdf5"):
    """Save the generated data to HDF5 format."""
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving data to {filename}...")
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('train', data=train_data)
        f.create_dataset('test', data=test_data)
    
    print(f"✓ Saved {train_data.shape[0]} training and {test_data.shape[0]} test samples")


def save_labels_csv(labels, classes, filename="data/train_labels.csv"):
    """Save training labels to CSV format."""
    
    print(f"Saving labels to {filename}...")
    
    # Create DataFrame matching expected format
    df = pd.DataFrame({
        'Id': range(len(labels)),
        'Category': [classes[label] for label in labels]
    })
    
    df.to_csv(filename, index=False)
    print(f"✓ Saved {len(labels)} labels")


def main():
    """Generate and save dummy dataset."""
    
    print("=== Creating Dummy Modulation Dataset ===\n")
    
    # Generate data
    train_data, train_labels, test_data, classes = create_dummy_iq_data()
    
    # Save to files
    save_hdf5_data(train_data, test_data)
    save_labels_csv(train_labels, classes)
    
    # Create classes.txt
    with open("data/classes.txt", "w") as f:
        for cls in classes:
            f.write(cls + "\n")
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   Training samples: {len(train_labels)}")
    print(f"   Test samples: {test_data.shape[0]}")  
    print(f"   Classes: {len(classes)}")
    print(f"   IQ data shape: {train_data.shape}")
    
    print(f"\nFiles created:")
    print(f"   📁 data/data.hdf5")
    print(f"   📄 data/train_labels.csv")
    print(f"   📄 data/classes.txt")
    
    print(f"\nNow you can run:")
    print(f"   modulation-train --data data/data.hdf5 --labels data/train_labels.csv --epochs 50")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
End-to-end tour of the modulation classifier package.

    python examples/example_usage.py              # full workflow on data/
    python examples/example_usage.py --minimal    # 5-epoch smoke test on random data, no files needed

The full workflow:
1. Load IQ data and labels (default: data/data.hdf5 + data/train_labels.csv,
   override with --data / --labels)
2. Plot constellations and the class distribution
3. Train a model with data augmentation (TensorBoard logs in examples/out/runs)
4. Evaluate it and plot a confusion matrix
5. Predict the file's unlabelled 'test' split and write a submission CSV

Outputs go to examples/out/.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from modulation_classifier import (
    ModulationClassifier, 
    IQDataset, 
    plot_constellation,
    analyze_class_distribution
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "out"


def main(data_path: str, labels_path: str, epochs: int):
    """Run the complete example workflow."""
    
    print("=== Modulation Classification Example ===\n")
    
    # Check if data files exist
    if not os.path.exists(data_path):
        print(f"❌ Training data file not found: {data_path}")
        print("Run ./setup_dataset.sh or python create_dummy_data.py first, or pass --data.")
        return
    
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        print("Run ./setup_dataset.sh or python create_dummy_data.py first, or pass --labels.")
        return
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = lambda name: str(OUT_DIR / name)  # noqa: E731
    
    # 1. Load and explore the dataset
    print("📊 Loading and exploring dataset...")
    dataset = IQDataset(data_path, labels_path)
    
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   IQ data shape: {dataset.data.shape}")
    print(f"   Number of classes: {len(dataset.classes)}")
    print(f"   Classes: {', '.join(dataset.classes)}")
    
    # Analyze class distribution
    distribution = analyze_class_distribution(dataset.labels, dataset.classes)
    print("\n   Class distribution:")
    for _, row in distribution.iterrows():
        print(f"     {row['Class']:12s}: {row['Count']:4d} samples ({row['Percentage']:.1f}%)")
    
    # 2. Visualize constellation diagrams
    print("\n🎯 Plotting constellation diagrams...")
    plot_constellation(
        dataset.data, 
        dataset.classes, 
        dataset.labels,
        max_samples_per_class=50,
        save_path=out("constellation_diagrams.png")
    )
    
    # 3. Create and train the classifier
    print("\n🧠 Training modulation classifier...")
    classifier = ModulationClassifier(
        model_architecture="improved",
        classes=dataset.classes
    )
    
    # Train with reasonable settings for demo
    history = classifier.train(
        dataset=dataset,
        num_epochs=epochs,
        batch_size=64,
        learning_rate=0.001,
        val_split=0.2,
        patience=5,
        save_path=out("demo_model.pt"),
        use_augmentation=True,
        augmentation_factor=2,
        log_dir=out("runs/example_usage")   # tensorboard --logdir examples/out/runs
    )
    
    # 4. Plot training curves
    print("\n📈 Plotting training curves...")
    classifier.plot_training_curves(save_path=out("training_curves.png"))
    
    # 5. Evaluate on validation set (if we split the training data)
    print("\n📋 Evaluating model performance...")
    
    # For demonstration, evaluate on the same dataset
    # In practice, you'd use a separate test set
    metrics = classifier.evaluate(dataset)
    
    print(f"   Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"\n   Per-class performance:")
    
    report = metrics['classification_report']
    for class_name in classifier.classes:
        if class_name in report:
            class_metrics = report[class_name]
            print(f"     {class_name:12s}: "
                  f"Precision={class_metrics['precision']:.3f}, "
                  f"Recall={class_metrics['recall']:.3f}, "
                  f"F1={class_metrics['f1-score']:.3f}")
    
    # 6. Plot confusion matrix
    print("\n🔍 Generating confusion matrix...")
    classifier.plot_confusion_matrix(dataset, save_path=out("confusion_matrix.png"))
    
    # 7. Predict the unlabelled 'test' split, if the file has one.
    #    (IQDataset prefers 'train' when both exist, so read 'test' explicitly.)
    import h5py
    with h5py.File(data_path, "r") as f:
        test_frames = f["test"][()] if "test" in f else None
    
    if test_frames is not None:
        print(f"\n🔮 Making predictions on the 'test' split ({len(test_frames)} frames)")
        predictions = classifier.predict_classes(test_frames)
        
        # Analyze prediction distribution
        from collections import Counter
        pred_counts = Counter(predictions)
        print(f"\n   Test set predictions:")
        for class_name, count in sorted(pred_counts.items()):
            percentage = 100.0 * count / len(predictions)
            print(f"     {class_name:12s}: {count:4d} samples ({percentage:.1f}%)")
        
        # Save predictions to CSV (Kaggle submission format)
        import pandas as pd
        pred_df = pd.DataFrame({
            'Id': range(len(predictions)),
            'Category': predictions
        })
        pred_df.to_csv(out('test_predictions.csv'), index=False)
        print(f"   ✓ Predictions saved to: {out('test_predictions.csv')}")
    
    else:
        print("\n⚠️  No 'test' dataset in the file; skipping test predictions.")
    
    print("\n✅ Example completed successfully!")
    print(f"\nGenerated files in {OUT_DIR}:")
    for file in [
        "demo_model.pt",
        "constellation_diagrams.png", 
        "training_curves.png",
        "confusion_matrix.png",
        "test_predictions.csv",
    ]:
        if (OUT_DIR / file).exists():
            print(f"   ✓ {file}")
    print(f"   ✓ runs/example_usage  (tensorboard --logdir {OUT_DIR / 'runs'})")


def minimal_example():
    """Minimal example for quick testing."""
    print("=== Minimal Example ===\n")
    
    # Create some dummy data for demonstration
    print("🔧 Creating dummy dataset...")
    
    # Generate synthetic IQ data (normally you'd load from HDF5)
    n_samples = 1000
    n_timesteps = 1024
    n_classes = 10
    
    # Random IQ data
    dummy_data = np.random.randn(n_samples, n_timesteps, 2).astype(np.float32) * 0.5
    
    # Balanced labels (the stratified split needs every class represented)
    dummy_labels = np.arange(n_samples) % n_classes
    
    # Create dataset
    from modulation_classifier.dataset import MODULATION_CLASSES
    dataset = IQDataset(dummy_data, dummy_labels, MODULATION_CLASSES)
    
    # Create dataset without labels for prediction
    dataset_no_labels = IQDataset(dummy_data, classes=MODULATION_CLASSES)
    
    print(f"   Created dataset with {len(dataset)} samples")
    
    # Train a small model
    print("\n🧠 Training classifier (minimal settings)...")
    classifier = ModulationClassifier()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = classifier.train(
        dataset=dataset,
        num_epochs=5,
        batch_size=32,
        use_augmentation=False,  # Skip augmentation for speed
        save_path=str(OUT_DIR / "minimal_model.pt"),
    )
    
    # Make some predictions on data without labels
    print("\n🔮 Making predictions...")
    predictions = classifier.predict_classes(dataset_no_labels)
    
    print(f"   Generated {len(predictions)} predictions")
    print(f"   Example predictions: {predictions[:10]}")
    
    print("\n✅ Minimal example completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minimal", action="store_true",
                        help="quick smoke test on random data (no data files needed)")
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "data.hdf5"))
    parser.add_argument("--labels", default=str(REPO_ROOT / "data" / "train_labels.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    if args.minimal:
        minimal_example()
    else:
        main(args.data, args.labels, args.epochs)

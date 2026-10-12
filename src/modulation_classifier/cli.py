"""Command line interface for the modulation classifier.

Two console scripts are installed with the package:

* ``modulation-train``   -- :func:`train_cli`, built from :func:`build_train_parser`
* ``modulation-predict`` -- :func:`predict_cli`, built from :func:`build_predict_parser`

The parser builders are separate functions so the argument reference can be
rendered in the documentation and exercised in tests.
"""

import argparse
import os
import sys
import traceback
import pandas as pd
from typing import Optional

from .classifier import ModulationClassifier
from .dataset import IQDataset
from .callbacks import default_run_name


def _configure_output() -> None:
    """Make stdout/stderr line-buffered.

    When the CLI is launched without a TTY (piped to a log file, run from an
    IDE or agent runner, ``nohup``), Python fully buffers stdout and progress
    messages only appear when the buffer fills or the process exits. Line
    buffering makes every status line show up as soon as it is printed.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            # Stream replaced by something that doesn't support reconfigure
            pass


def build_train_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``modulation-train``."""
    parser = argparse.ArgumentParser(
        prog='modulation-train',
        description='Train a modulation classifier on an HDF5 file of IQ frames and a '
                    'CSV of labels. Metrics stream to TensorBoard by default.'
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 training data file')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to CSV labels file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split fraction (default: 0.2)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='improved',
                       choices=['improved', 'complex', 'legacy'],
                       help='Model architecture: improved (real-valued CNN), complex (complex-valued CNN), legacy (original)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto-detect cuda > mps > cpu)')
    
    # Augmentation arguments
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--augmentation-factor', type=int, default=2,
                       help='Data augmentation factor (default: 2)')
    
    # Output arguments
    parser.add_argument('--model-path', type=str, default='best_model.pt',
                       help='Path to save trained model (default: best_model.pt)')
    parser.add_argument('--plots-dir', type=str, default=None,
                       help='Directory to save training plots')
    parser.add_argument('--log-interval', type=int, default=50,
                       help='Log batch-level metrics (to TensorBoard, and to stdout when '
                            'not attached to a terminal) every N batches; 0 to disable '
                            '(default: 50)')
    
    # TensorBoard arguments
    parser.add_argument('--log-dir', type=str, default='runs',
                       help='Parent directory for TensorBoard runs (default: runs). '
                            'View with: tensorboard --logdir runs')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name of this run inside --log-dir '
                            '(default: <timestamp>_<architecture>)')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    return parser


def train_cli():
    """Entry point for ``modulation-train``."""
    _configure_output()
    args = build_train_parser().parse_args()
    
    # Validate input files
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        sys.exit(1)
    
    # Create output directory if needed
    if args.plots_dir and not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)
    
    # Resolve TensorBoard run directory
    run_dir = None
    if not args.no_tensorboard:
        run_name = args.run_name or default_run_name(args.architecture)
        run_dir = os.path.join(args.log_dir, run_name)
    
    print("=== Modulation Classifier Training ===")
    print(f"Data: {args.data}")
    print(f"Labels: {args.labels}")
    print(f"Architecture: {args.architecture}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Augmentation: {not args.no_augmentation}")
    if run_dir:
        print(f"TensorBoard: {run_dir}")
        print(f"  -> live view: tensorboard --logdir {args.log_dir}")
    
    try:
        # Create classifier
        classifier = ModulationClassifier(
            model_architecture=args.architecture,
            device=args.device
        )
        
        # Train model
        history = classifier.train(
            dataset=args.data,
            labels_path=args.labels,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            val_split=args.val_split,
            patience=args.patience,
            save_path=args.model_path,
            use_augmentation=not args.no_augmentation,
            augmentation_factor=args.augmentation_factor,
            log_interval=args.log_interval,
            log_dir=run_dir
        )
        
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Best model saved to: {args.model_path}")
        if run_dir:
            print(f"✓ TensorBoard logs: {run_dir}  (tensorboard --logdir {args.log_dir})")
        
        # Save training plots if requested
        if args.plots_dir:
            curves_path = os.path.join(args.plots_dir, 'training_curves.png')
            classifier.plot_training_curves(save_path=curves_path)
            
        # Print final metrics
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        best_val_acc = max(history['val_acc'])
        
        print(f"\nFinal Results:")
        print(f"  Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"  Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        sys.exit(1)


def build_predict_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``modulation-predict``."""
    parser = argparse.ArgumentParser(
        prog='modulation-predict',
        description='Run a trained modulation classifier on an HDF5 file of IQ frames, '
                    'write a predictions CSV and optionally evaluate against labels.'
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--architecture', type=str, default='improved',
                       choices=['improved', 'complex', 'legacy'],
                       help='Model architecture: improved (real-valued CNN), complex (complex-valued CNN), legacy (original)')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 test data file')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to test labels CSV (optional, for evaluation)')
    
    # Inference arguments
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Inference batch size (default: 64)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto-detect cuda > mps > cpu)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to save predictions CSV (default: predictions.csv)')
    parser.add_argument('--probabilities', action='store_true',
                       help='Also output class probabilities')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance (requires labels)')
    parser.add_argument('--plots-dir', type=str, default=None,
                       help='Directory to save evaluation plots')
    return parser


def predict_cli():
    """Entry point for ``modulation-predict``."""
    _configure_output()
    args = build_predict_parser().parse_args()
    
    # Validate input files
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    if args.labels and not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        sys.exit(1)
    
    # Create output directory if needed
    if args.plots_dir and not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)
    
    print("=== Modulation Classifier Prediction ===")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    
    try:
        # Create classifier and load model
        classifier = ModulationClassifier(
            model_architecture=args.architecture,
            device=args.device
        )
        classifier.load_model(args.model)
        
        # Load test data
        dataset = IQDataset(args.data, args.labels)
        print(f"Loaded {len(dataset)} test samples")
        
        # Make predictions
        print("Generating predictions...")
        if args.probabilities:
            predictions, probabilities = classifier.predict(
                dataset, batch_size=args.batch_size, return_probabilities=True
            )
        else:
            predictions = classifier.predict(dataset, batch_size=args.batch_size)
            probabilities = None
        
        # Convert to class names
        pred_classes = [classifier.classes[pred] for pred in predictions]
        
        # Create output DataFrame
        output_data = {
            'Id': range(len(predictions)),
            'Category': pred_classes
        }
        
        if probabilities is not None:
            # Add probability columns
            for i, class_name in enumerate(classifier.classes):
                output_data[f'Prob_{class_name}'] = probabilities[:, i]
        
        df = pd.DataFrame(output_data)
        df.to_csv(args.output, index=False)
        print(f"✓ Predictions saved to: {args.output}")
        
        # Show prediction distribution
        from collections import Counter
        pred_counts = Counter(pred_classes)
        print(f"\nPrediction Distribution:")
        for class_name, count in pred_counts.items():
            percentage = 100.0 * count / len(pred_classes)
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Evaluate if labels provided
        if args.evaluate and args.labels:
            print(f"\n=== Model Evaluation ===")
            metrics = classifier.evaluate(dataset, batch_size=args.batch_size)
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"\nPer-class metrics:")
            
            report = metrics['classification_report']
            for class_name in classifier.classes:
                if class_name in report:
                    class_metrics = report[class_name]
                    print(f"  {class_name:12s}: P={class_metrics['precision']:.3f}, "
                          f"R={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")
            
            # Save evaluation plots
            if args.plots_dir:
                cm_path = os.path.join(args.plots_dir, 'confusion_matrix.png')
                classifier.plot_confusion_matrix(dataset, save_path=cm_path)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m modulation_classifier.cli train --help")
        print("  python -m modulation_classifier.cli predict --help")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from argv
    
    if command == 'train':
        train_cli()
    elif command == 'predict':
        predict_cli()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict")
        sys.exit(1)
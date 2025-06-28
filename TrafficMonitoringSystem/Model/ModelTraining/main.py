import argparse
import os
import sys
from datetime import datetime

# Import training function from train.py
from train import train_model


# Custom logger class to duplicate output to both console and file
class TeeLogger:
    """
    Logger class that duplicates all stdout and stderr to a log file while
    maintaining output to the console.
    """

    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.file = open(log_file, 'w', encoding='utf-8')

        print(f"Logging to file: {log_file}")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()


# Set dataset path - update to your actual path
DATASET_PATH = '/mnt/QNAP/apricop/container/dataset2'

# Output directories
OUTPUT_DIR = './outputs'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train vehicle detection model on UA-DETRAC dataset')
    
    # Dataset options
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH,
                        help=f'Path to dataset directory (default: {DATASET_PATH})')
    
    # Subset options
    parser.add_argument('--train-subset-file', type=str, default=None,
                        help='Path to file containing training subset image filenames')
    parser.add_argument('--train-subset-size', type=int, default=None,
                        help='Size of training subset if subset file is not provided')
    parser.add_argument('--val-subset-file', type=str, default=None,
                        help='Path to file containing validation subset image filenames')
    parser.add_argument('--val-subset-size', type=int, default=None,
                        help='Size of validation subset if subset file is not provided')
    parser.add_argument('--test-subset-file', type=str, default=None,
                        help='Path to file containing test subset image filenames')
    parser.add_argument('--test-subset-size', type=int, default=None,
                        help='Size of test subset if subset file is not provided')
    parser.add_argument('--subset-random-seed', type=int, default=42,
                        help='Random seed for subset creation (default: 42)')
    
    # Training options
    parser.add_argument('--img-size', type=int, default=480,
                        help='Image size for training (default: 480)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Maximum number of epochs (default: 15)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs (default: ./outputs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file (default: auto-generated in output directory)')
    parser.add_argument('--binary-classification', action='store_true',
                        help='Use binary classification (large/small) instead of 4-class')

    args = parser.parse_args()

    # Update output directories if needed
    if args.output_dir != './outputs':
        OUTPUT_DIR = args.output_dir
        MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
        PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_file if args.log_file else os.path.join(OUTPUT_DIR, f"training_log_{timestamp}.txt")
    sys.stdout = TeeLogger(log_file)
    sys.stderr = sys.stdout  # Redirect stderr to the same logger

    print("=" * 50)
    print(f"UA-DETRAC Vehicle Detection Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command line arguments: {args}")
    print(f"Using {'binary (large/small)' if args.binary_classification else '4-class'} classification")
    print(f"Logging to: {log_file}")
    print("=" * 50)

    # Train the model
    train_model(
        base_dir=args.dataset_path,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        plot_dir=PLOT_DIR,
        train_subset_file=args.train_subset_file,
        train_subset_size=args.train_subset_size,
        val_subset_file=args.val_subset_file,
        val_subset_size=args.val_subset_size,
        test_subset_file=args.test_subset_file,
        test_subset_size=args.test_subset_size,
        subset_random_seed=args.subset_random_seed,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        seed=args.seed,
        binary_classification=args.binary_classification
    )
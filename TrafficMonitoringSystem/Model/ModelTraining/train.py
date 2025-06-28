import os
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import random

# Import custom modules
from dataset import (
    create_datasets_from_paths, collate_fn, create_weighted_sampler
)
from model_utils import (
    get_device, get_object_detection_model,
    create_optimizer, create_lr_scheduler
)
from subset_creator import create_subset, save_subset
from engine import train_one_epoch, evaluate, calculate_classification_accuracy, calculate_miss_rate, calculate_miss_rate_vs_fppi
from visualization import save_training_plots_with_miss_rate, save_results_log_with_miss_rate


def train_model(base_dir, output_dir, model_dir, plot_dir,
                train_subset_file=None, train_subset_size=None,
                val_subset_file=None, val_subset_size=None,
                test_subset_file=None, test_subset_size=None,
                subset_random_seed=42,
                img_size=480, batch_size=8, num_epochs=15, early_stopping_patience=3,
                seed=42, binary_classification=True):
    """
    Main function to train the object detection model using train, val, and test sets

    Args:
        base_dir: Base directory containing images and labels folders
        output_dir: Directory to save outputs
        model_dir: Directory to save models
        plot_dir: Directory to save plots
        train_subset_file: Optional path to file containing training image filenames
        train_subset_size: Size of training subset if subset file is not provided
        val_subset_file: Optional path to file containing validation image filenames
        val_subset_size: Size of validation subset if subset file is not provided
        test_subset_file: Optional path to file containing test image filenames
        test_subset_size: Size of test subset if subset file is not provided
        subset_random_seed: Random seed for subset creation
        img_size: Size of images (square)
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        early_stopping_patience: Number of epochs without improvement before stopping
        seed: Random seed for reproducibility
        binary_classification: Whether to use binary classification (large/small vehicles)

    Returns:
        Dictionary with model and performance metrics
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 50)
    print("UA-DETRAC Vehicle Detection Training")
    print(
        f"Classification mode: {'Binary (large/small vehicles)' if binary_classification else '4-class (truck/car/van/bus)'}")

    # Define paths based on new folder structure
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "valid")    
    test_dir = os.path.join(base_dir, "test")
    
    # Paths for training, validation, and testing
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")
    
    test_images_dir = os.path.join(test_dir, "images")
    test_labels_dir = os.path.join(test_dir, "labels")

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Get device for training
    device = get_device()

    # Handle subset creation if needed
    subset_name = None
    
    # Create or load training subset if requested
    if train_subset_size and not train_subset_file:
        subset_name = f"train_subset_{train_subset_size}_{timestamp}.txt"
        subset_path = os.path.join(output_dir, subset_name)

        print(f"\nCreating training subset with {train_subset_size} images...")

        # Create the subset
        create_subset(train_images_dir, train_labels_dir, subset_type=1,  # Random subset
                      size=train_subset_size, output_file=subset_path, seed=subset_random_seed)

        # Use this subset file for training
        train_subset_file = subset_path
    
    # Create validation subset if requested
    if val_subset_size and not val_subset_file:
        # Select random subset of images
        val_images = [img for img in sorted(os.listdir(val_images_dir)) if img.endswith(('.jpg', '.jpeg', '.png'))]

        if len(val_images) <= val_subset_size:
            print(f"Warning: Requested validation size {val_subset_size} is greater than or equal to available images ({len(val_images)}). Using all images.")
        else:
            print(f"Creating random validation subset with {val_subset_size} images (from {len(val_images)} available)...")
            val_subset = random.sample(val_images, val_subset_size)

            # Save to file
            val_subset_name = f"val_subset_{val_subset_size}_{timestamp}.txt"
            val_subset_path = os.path.join(output_dir, val_subset_name)
            save_subset(val_subset, val_subset_path)

            # Use this subset file for validation
            val_subset_file = val_subset_path
            print(f"Created and saved validation subset to {val_subset_file}")
    
    # Create test subset if requested
    if test_subset_size and not test_subset_file:
        # Select random subset of images
        test_images = [img for img in sorted(os.listdir(test_images_dir)) if img.endswith(('.jpg', '.jpeg', '.png'))]

        if len(test_images) <= test_subset_size:
            print(f"Warning: Requested test size {test_subset_size} is greater than or equal to available images ({len(test_images)}). Using all images.")
        else:
            print(f"Creating random test subset with {test_subset_size} images (from {len(test_images)} available)...")
            test_subset = random.sample(test_images, test_subset_size)

            # Save to file
            test_subset_name = f"test_subset_{test_subset_size}_{timestamp}.txt"
            test_subset_path = os.path.join(output_dir, test_subset_name)
            save_subset(test_subset, test_subset_path)

            # Use this subset file for testing
            test_subset_file = test_subset_path
            print(f"Created and saved test subset to {test_subset_file}")

    # Display info about the subsets being used
    if train_subset_file or val_subset_file or test_subset_file:
        print(f"Using custom subsets:")
        if train_subset_file:
            print(f"  - Training: {train_subset_file}")
        if val_subset_file:
            print(f"  - Validation: {val_subset_file}")
        if test_subset_file:
            print(f"  - Testing: {test_subset_file}")

    print("=" * 50)

    # Create datasets from the specified paths
    print("\nCreating datasets...")
    train_subset = None
    val_subset = None
    test_subset = None
    
    # Load subsets if specified
    if train_subset_file:
        with open(train_subset_file, 'r') as f:
            train_subset = [line.strip() for line in f.readlines()]
    
    if val_subset_file:
        with open(val_subset_file, 'r') as f:
            val_subset = [line.strip() for line in f.readlines()]
    
    if test_subset_file:
        with open(test_subset_file, 'r') as f:
            test_subset = [line.strip() for line in f.readlines()]
    
    # Create the datasets
    train_dataset = create_datasets_from_paths(
        train_images_dir, train_labels_dir,
        img_size=img_size,
        subset_files=train_subset,
        binary_classification=binary_classification
    )
    
    val_dataset = create_datasets_from_paths(
        val_images_dir, val_labels_dir,
        img_size=img_size,
        subset_files=val_subset,
        binary_classification=binary_classification
    )
    
    test_dataset = create_datasets_from_paths(
        test_images_dir, test_labels_dir,
        img_size=img_size,
        subset_files=test_subset,
        binary_classification=binary_classification
    )

    print(f"\n===== DATASET STATISTICS =====")
    print(f"Training dataset size: {len(train_dataset)} images")
    print(f"Validation dataset size: {len(val_dataset)} images")
    print(f"Test dataset size: {len(test_dataset)} images")

    # Print class distribution summary
    print("\nClass Distribution in Training Dataset:")
    if binary_classification:
        class_name_map = {1: 'large_vehicle', 2: 'small_vehicle'}
        for class_id, count in train_dataset.class_counts.items():
            print(f"  Class {class_id} ({class_name_map[class_id]}): {count} instances")
    else:
        class_name_map = {0: 'truck', 1: 'car', 2: 'van', 3: 'bus'}
        for class_id, count in train_dataset.class_counts.items():
            model_class_id = class_id + 1  # For the model, we add 1 to account for background
            class_name = class_name_map[class_id]
            print(f"  Annotation Class {class_id} (Model Class {model_class_id}: {class_name}): {count} instances")

    print("\nClass Distribution in Validation Dataset:")
    if binary_classification:
        for class_id, count in val_dataset.class_counts.items():
            print(f"  Class {class_id} ({class_name_map[class_id]}): {count} instances")
    else:
        for class_id, count in val_dataset.class_counts.items():
            model_class_id = class_id + 1
            class_name = class_name_map[class_id]
            print(f"  Annotation Class {class_id} (Model Class {model_class_id}: {class_name}): {count} instances")
            
    print("\nClass Distribution in Test Dataset:")
    if binary_classification:
        for class_id, count in test_dataset.class_counts.items():
            print(f"  Class {class_id} ({class_name_map[class_id]}): {count} instances")
    else:
        for class_id, count in test_dataset.class_counts.items():
            model_class_id = class_id + 1
            class_name = class_name_map[class_id]
            print(f"  Annotation Class {class_id} (Model Class {model_class_id}: {class_name}): {count} instances")

    # Create model
    print("\nInitializing model...")
    # Set num_classes based on classification type (binary or 4-class)
    # +1 for background class
    num_classes = 3 if binary_classification else 5  # background + 2 or 4 vehicle classes
    model = get_object_detection_model(num_classes)
    model.to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model)
    lr_scheduler = create_lr_scheduler(optimizer)

    # Training tracking variables
    best_map = 0.0
    patience_counter = 0

    # Lists to store metrics
    train_loss_list = []
    val_map_list = []
    class_ap_history = {cls_id: [] for cls_id in range(1, num_classes)}  # Track AP for each class
    epochs_list = list(range(1, num_epochs + 1))

    # Create data loaders
    # Create weighted sampler for the training set
    train_sampler = create_weighted_sampler(train_dataset, list(range(len(train_dataset))))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Start training
    print("\nStarting training...")
    print(f"Training for up to {num_epochs} epochs with early stopping (patience={early_stopping_patience})")

    training_start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n{'-' * 20} Epoch {epoch + 1}/{num_epochs} {'-' * 20}")

        # Train for one epoch - using the train_one_epoch function from engine.py
        print(f"\nTraining epoch {epoch + 1}...")
        loss_dict = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5000)

        # Extract total loss
        total_loss = sum(loss_dict.values())
        train_loss_list.append(total_loss)

        # Update learning rate
        lr_scheduler.step()

        # Evaluate on validation set - using the evaluate function from engine.py
        print(f"\nEvaluating on validation set...")
        val_metrics = evaluate(model, val_loader, device, print_per_class=True,
                               binary_classification=binary_classification)
        val_map = val_metrics.stats[0]  # mAP at IoU=0.5:0.95
        val_map_list.append(val_map)

        # Calculate classification accuracy - using the classification_accuracy function from engine.py
        if epoch == 0 or (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            print(f"\nCalculating classification accuracy on validation set...")
            class_accuracy, overall_accuracy, _ = calculate_classification_accuracy(model, val_loader, device,
                                                                                    binary_classification=binary_classification)

            # Log classification accuracy
            print(f"Overall classification accuracy: {overall_accuracy:.4f}")

            # Calculate miss rate
            print(f"\nCalculating miss rate on validation set...")
            val_miss_rate_results = calculate_miss_rate(model, val_loader, device,
                                                      binary_classification=binary_classification)
            
            print(f"Overall miss rate: {val_miss_rate_results['overall_miss_rate']:.4f}")

        # Store per-class AP metrics for the validation set
        if hasattr(val_metrics, 'eval') and 'precision' in val_metrics.eval:
            precisions = val_metrics.eval['precision']

            # Handle binary vs 4-class classification
            num_ap_classes = 2 if binary_classification else 4

            for class_idx in range(1, num_ap_classes + 1):
                # Extract AP for IoU=0.5 (index 0)
                if precisions.shape[2] > class_idx - 1:
                    precision = precisions[0, :, class_idx - 1, 0, 2]
                    ap = float('nan')
                    if precision[precision > -1].size > 0:
                        ap = np.mean(precision[precision > -1])
                    class_ap_history[class_idx].append(ap)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")
        print(f"Train Loss: {total_loss:.4f}, Validation mAP: {val_map:.4f}")

        # Save model if it's the best so far based on validation mAP
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0

            # Add subset info to model filename
            subset_info = ""
            if train_subset_file or train_subset_size:
                subset_info = "_subset"

            binary_suffix = "_binary" if binary_classification else ""

            model_path = os.path.join(model_dir,
                                      f"best_model{subset_info}{binary_suffix}_epoch_{epoch + 1}_map_{val_map:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'map': val_map,
                'loss': total_loss,
                'class_ap': {cls_id: class_ap_history[cls_id][-1] for cls_id in class_ap_history if
                             cls_id in class_ap_history and len(class_ap_history[cls_id]) > 0},
                'train_subset_file': train_subset_file,
                'train_subset_size': train_subset_size,
                'val_subset_file': val_subset_file,
                'val_subset_size': val_subset_size,
                'test_subset_file': test_subset_file,
                'test_subset_size': test_subset_size,
                'binary_classification': binary_classification
            }, model_path)
            print(f"Saved best model with mAP: {val_map:.4f} at epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time / 60:.2f} minutes")

    # Final evaluation on the test dataset
    print("\n===== FINAL EVALUATION =====")
    print(f"Performing final evaluation on the test dataset ({len(test_dataset)} images)...")
    final_eval_metrics = evaluate(model, test_loader, device, print_per_class=True,
                                  binary_classification=binary_classification)
    final_eval_map = final_eval_metrics.stats[0]
    print(f"Final Test mAP: {final_eval_map:.4f}")

    # Calculate final classification accuracy on test dataset
    print("\nCalculating final classification accuracy on test dataset...")
    final_class_accuracy, final_overall_accuracy, final_class_metrics = calculate_classification_accuracy(
        model, test_loader, device, binary_classification=binary_classification
    )

    # Calculate final miss rate on test dataset
    print("\nCalculating final miss rate on test dataset...")
    final_miss_rate_results = calculate_miss_rate(model, test_loader, device,
                                                binary_classification=binary_classification)
    
    # Calculate miss rate vs FPPI curve
    print("\nCalculating miss rate vs FPPI curve on test dataset...")
    miss_rate_fppi_data = calculate_miss_rate_vs_fppi(model, test_loader, device,
                                                    binary_classification=binary_classification)

    # Save final model with evaluation metrics
    subset_info = ""
    if train_subset_file or train_subset_size:
        subset_info = "_subset"

    binary_suffix = "_binary" if binary_classification else ""

    final_model_path = os.path.join(
        model_dir,
        f"final_model{subset_info}{binary_suffix}_epoch_{epoch + 1}_map_{final_eval_map:.4f}.pth"
    )

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'map': final_eval_map,
        'classification_accuracy': final_class_accuracy,
        'overall_classification_accuracy': final_overall_accuracy,
        'miss_rates': final_miss_rate_results['miss_rates'],
        'overall_miss_rate': final_miss_rate_results['overall_miss_rate'],
        'recall_rates': final_miss_rate_results['recall_rates'],
        'miss_rate_fppi_data': miss_rate_fppi_data,
        'train_subset_file': train_subset_file,
        'train_subset_size': train_subset_size,
        'val_subset_file': val_subset_file,
        'val_subset_size': val_subset_size,
        'test_subset_file': test_subset_file,
        'test_subset_size': test_subset_size,
        'binary_classification': binary_classification
    }, final_model_path)
    print(f"Saved final model at epoch {epoch + 1}")

    # Print final summary
    print("\n===== FINAL MODEL PERFORMANCE SUMMARY =====")
    print(
        f"Classification Mode: {'Binary (large/small vehicles)' if binary_classification else '4-class (truck/car/van/bus)'}")
    print(f"Detection Performance (mAP@0.5:0.95): {final_eval_map:.4f}")
    print(f"Detection Performance (mAP@0.5): {final_eval_metrics.stats[1]:.4f}")
    print(f"Overall Classification Accuracy: {final_overall_accuracy:.4f}")
    print(f"Overall Miss Rate: {final_miss_rate_results['overall_miss_rate']:.4f}")
    print(f"Overall Recall: {final_miss_rate_results['overall_recall']:.4f}")

    print("\nClassification Accuracy per Class:")
    if binary_classification:
        class_names = {1: 'large_vehicle', 2: 'small_vehicle'}
    else:
        class_names = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}

    for class_id, accuracy in final_class_accuracy.items():
        if class_id in class_names:
            print(f"  {class_names[class_id]}: {accuracy:.4f}")
    
    print("\nPer-Class Performance:")
    print("Class        | AP@0.5 | Accuracy | Miss Rate | Recall")
    print("-" * 50)
    for class_id in class_names.keys():
        if (class_id in final_class_accuracy and 
            class_id in final_miss_rate_results['miss_rates'] and
            class_id in final_miss_rate_results['recall_rates']):
            
            ap = "N/A"  # You would need to get this from final_eval_metrics if available
            accuracy = final_class_accuracy[class_id]
            miss_rate = final_miss_rate_results['miss_rates'][class_id]
            recall = final_miss_rate_results['recall_rates'][class_id]
            
            print(f"{class_names[class_id]:<12} | {ap:>6} | {accuracy:8.4f} | {miss_rate:9.4f} | {recall:6.4f}")

    # Create confusion matrix for visualization
    if binary_classification:
        confusion_matrix = np.zeros((2, 2))
        for i in range(1, 3):  # 1=large, 2=small
            for j in range(1, 3):
                confusion_matrix[i - 1, j - 1] = final_class_metrics["confusion_matrix"][i][j]
    else:
        confusion_matrix = np.zeros((4, 4))
        for i in range(1, 5):
            for j in range(1, 5):
                confusion_matrix[i - 1, j - 1] = final_class_metrics["confusion_matrix"][i][j]

    # Create visualizations and save results
    save_training_plots_with_miss_rate(
        epochs_list, train_loss_list, val_map_list, final_eval_map,
        class_ap_history, final_class_accuracy, final_overall_accuracy,
        confusion_matrix, final_miss_rate_results, plot_dir, subset_info, binary_classification
    )

    # Prepare subset info for results log
    subset_description = ""
    if train_subset_size:
        subset_description += f"_train_{train_subset_size}"
    if val_subset_size:
        subset_description += f"_val_{val_subset_size}"
    if test_subset_size:
        subset_description += f"_test_{test_subset_size}"

    save_results_log_with_miss_rate(
        output_dir, train_dataset, test_dataset, final_eval_map, final_eval_metrics,
        final_class_accuracy, final_overall_accuracy, final_class_metrics,
        final_miss_rate_results, total_training_time, epoch, train_subset_file, 
        test_subset_file, train_subset_size, subset_name, subset_info, binary_classification
    )

    print("\nTraining completed successfully!")

    return {
        'model': model,
        'map': final_eval_map,
        'accuracy': final_overall_accuracy,
        'class_accuracy': final_class_accuracy,
        'confusion_matrix': confusion_matrix,
        'miss_rates': final_miss_rate_results['miss_rates'],
        'overall_miss_rate': final_miss_rate_results['overall_miss_rate'],
        'recall_rates': final_miss_rate_results['recall_rates'],
        'miss_rate_fppi_data': miss_rate_fppi_data,
        'binary_classification': binary_classification
    }
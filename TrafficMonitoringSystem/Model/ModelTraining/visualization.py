import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def save_training_plots(epochs_list, train_loss_list, val_map_list, final_eval_map,
                        class_ap_history, final_class_accuracy, final_overall_accuracy,
                        confusion_matrix, plot_dir, subset_info="", binary_classification=True):
    """
    Create comprehensive visualization plots for training results

    Args:
        epochs_list: List of epoch numbers
        train_loss_list: List of training losses per epoch
        val_map_list: List of validation mAP scores per epoch
        final_eval_map: Final evaluation mAP score
        class_ap_history: Dict of AP history per class
        final_class_accuracy: Dict of final accuracy per class
        final_overall_accuracy: Overall classification accuracy
        confusion_matrix: Confusion matrix for all classes
        plot_dir: Directory to save plots
        subset_info: Additional info for filenames
        binary_classification: Whether using binary classification
    """
    # 1. Training progress visualization
    plt.figure(figsize=(15, 10))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_list[:len(train_loss_list)], train_loss_list, 'b-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot validation mAP
    plt.subplot(2, 2, 2)
    plt.plot(epochs_list[:len(val_map_list)], val_map_list, 'r-o', label='Validation mAP')
    plt.axhline(y=final_eval_map, color='g', linestyle='--', label=f'Final mAP: {final_eval_map:.4f}')
    plt.title('Mean Average Precision')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5:0.95')
    plt.grid(True)
    plt.legend()

    # Plot per-class AP
    plt.subplot(2, 2, 3)
    if binary_classification:
        class_name_map = {1: 'large_vehicle', 2: 'small_vehicle'}
    else:
        class_name_map = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}

    for class_id in class_name_map.keys():
        if class_id in class_ap_history:
            ap_values = class_ap_history[class_id]
            if len(ap_values) > 0:
                plt.plot(epochs_list[:len(ap_values)],
                         ap_values,
                         'o-',
                         label=f'AP {class_name_map[class_id]}')

    plt.title('Per-Class Average Precision')
    plt.xlabel('Epochs')
    plt.ylabel('AP@0.5')
    plt.grid(True)
    plt.legend()

    # Plot classification accuracy
    plt.subplot(2, 2, 4)
    accuracies = [final_class_accuracy[i] for i in class_name_map.keys() if i in final_class_accuracy]
    class_names = [class_name_map[i] for i in class_name_map.keys() if i in final_class_accuracy]
    plt.bar(class_names, accuracies, color='lightgreen')
    plt.axhline(y=final_overall_accuracy, color='r', linestyle='--',
                label=f'Overall: {final_overall_accuracy:.4f}')
    plt.ylim(0, 1.0)
    plt.title('Classification Accuracy per Class')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    binary_suffix = "_binary" if binary_classification else ""
    plt.savefig(os.path.join(plot_dir, f'training_summary{subset_info}{binary_suffix}.png'))
    print(
        f"Saved training summary plot to {os.path.join(plot_dir, f'training_summary{subset_info}{binary_suffix}.png')}")

    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add class labels
    classes = [class_name_map[i] for i in sorted(class_name_map.keys())]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(int(confusion_matrix[i, j]), 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plot_dir, f'confusion_matrix{subset_info}{binary_suffix}.png'))
    print(f"Saved confusion matrix to {os.path.join(plot_dir, f'confusion_matrix{subset_info}{binary_suffix}.png')}")


def save_results_log(output_dir, full_dataset, eval_dataset, final_eval_map, final_eval_metrics,
                     final_class_accuracy, final_overall_accuracy, final_class_metrics,
                     total_training_time, epoch, train_subset_file, eval_subset_file,
                     subset_type, subset_name, subset_info="", binary_classification=True):
    """
    Save training results to a log file
    """
    binary_suffix = "_binary" if binary_classification else ""
    log_file = os.path.join(output_dir, f'training_results{subset_info}{binary_suffix}.txt')

    with open(log_file, 'w') as f:
        f.write("===== UA-DETRAC VEHICLE DETECTION TRAINING RESULTS =====\n\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {total_training_time / 60:.2f} minutes\n")
        f.write(f"Total epochs trained: {epoch + 1}\n")
        f.write(
            f"Classification mode: {'Binary (large/small)' if binary_classification else 'Multi-class (4 classes)'}\n\n")

        if subset_type > 0:
            f.write(f"Using subset type {subset_type} with {len(full_dataset.imgs)} images:\n")
            if subset_type == 1:
                f.write("  - Random subset\n")
            elif subset_type == 2:
                f.write("  - Minority-focused subset (trucks and buses)\n")
            elif subset_type == 3:
                f.write("  - Balanced subset\n")

            if subset_name:
                f.write(f"  - Subset file: {subset_name}\n\n")
        elif train_subset_file or eval_subset_file:
            f.write("Using custom subsets:\n")
            if train_subset_file:
                f.write(f"  - Training: {train_subset_file}\n")
            if eval_subset_file:
                f.write(f"  - Evaluation: {eval_subset_file} ({len(eval_dataset)} images)\n\n")

        f.write("===== DATASET STATISTICS =====\n")
        f.write(f"Training dataset size: {len(full_dataset)} images\n")
        f.write(f"Evaluation dataset size: {len(eval_dataset)} images\n\n")

        if binary_classification:
            f.write("Class Distribution in Training Dataset:\n")
            for class_id, count in full_dataset.class_counts.items():
                class_name = 'large_vehicle' if class_id == 1 else 'small_vehicle'
                f.write(f"  Class {class_id} ({class_name}): {count} instances\n")

            f.write("\nClass Distribution in Evaluation Dataset:\n")
            for class_id, count in eval_dataset.class_counts.items():
                class_name = 'large_vehicle' if class_id == 1 else 'small_vehicle'
                f.write(f"  Class {class_id} ({class_name}): {count} instances\n")
        else:
            class_name_map = {0: 'truck', 1: 'car', 2: 'van', 3: 'bus'}
            f.write("Class Distribution in Training Dataset:\n")
            for class_id, count in full_dataset.class_counts.items():
                class_name = class_name_map[class_id] if class_id in class_name_map else f"Unknown class {class_id}"
                f.write(f"  Class {class_id} ({class_name}): {count} instances\n")

            f.write("\nClass Distribution in Evaluation Dataset:\n")
            for class_id, count in eval_dataset.class_counts.items():
                class_name = class_name_map[class_id] if class_id in class_name_map else f"Unknown class {class_id}"
                f.write(f"  Class {class_id} ({class_name}): {count} instances\n")

        f.write("\n===== MODEL PERFORMANCE =====\n")
        f.write(f"Detection Performance (mAP@0.5:0.95): {final_eval_map:.4f}\n")
        f.write(f"Detection Performance (mAP@0.5): {final_eval_metrics.stats[1]:.4f}\n")
        f.write(f"Overall Classification Accuracy: {final_overall_accuracy:.4f}\n\n")

        f.write("Classification Accuracy per Class:\n")
        if binary_classification:
            class_names = {1: 'large_vehicle', 2: 'small_vehicle'}
            for class_id, accuracy in final_class_accuracy.items():
                if class_id in class_names:
                    f.write(f"  {class_names[class_id]}: {accuracy:.4f}\n")
        else:
            class_names = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}
            for class_id, accuracy in final_class_accuracy.items():
                if class_id in class_names:
                    f.write(f"  {class_names[class_id]}: {accuracy:.4f}\n")

        f.write("\nConfusion Matrix:\n")
        f.write("                 Predicted Class\n")

        if binary_classification:
            f.write("                 --------------------------------\n")
            f.write("True Class       | Large  | Small  |\n")
            f.write("-" * 42 + "\n")

            for true_class in [1, 2]:  # Large, Small
                row = final_class_metrics["confusion_matrix"][true_class]
                class_name = 'large_vehicle' if true_class == 1 else 'small_vehicle'
                f.write(f"{class_name:<16} | {row[1]:5d} | {row[2]:5d} |\n")
        else:
            f.write("                 --------------------------------\n")
            f.write("True Class       | Truck | Car   | Van   | Bus   |\n")
            f.write("-" * 60 + "\n")

            for true_class in range(1, 5):
                row = final_class_metrics["confusion_matrix"][true_class]
                class_name = class_names[true_class] if true_class in class_names else f"Unknown {true_class}"
                f.write(f"{class_name:<16} | {row[1]:5d} | {row[2]:5d} | {row[3]:5d} | {row[4]:5d} |\n")

    print(f"\nFull results saved to: {log_file}")


def save_results_log_with_miss_rate(output_dir, full_dataset, eval_dataset, final_eval_map, final_eval_metrics,
                     final_class_accuracy, final_overall_accuracy, final_class_metrics,
                     final_miss_rate_results, total_training_time, epoch, train_subset_file, 
                     eval_subset_file, subset_type, subset_name, subset_info="", 
                     binary_classification=True):
    """
    Save training results to a log file including miss rate metrics
    """
    binary_suffix = "_binary" if binary_classification else ""
    log_file = os.path.join(output_dir, f'training_results{subset_info}{binary_suffix}.txt')

    with open(log_file, 'w') as f:
        f.write("===== UA-DETRAC VEHICLE DETECTION TRAINING RESULTS =====\n\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {total_training_time / 60:.2f} minutes\n")
        f.write(f"Total epochs trained: {epoch + 1}\n")
        f.write(
            f"Classification mode: {'Binary (large/small)' if binary_classification else 'Multi-class (4 classes)'}\n\n")

        # ... existing dataset statistics ...

        f.write("\n===== MODEL PERFORMANCE =====\n")
        f.write(f"Detection Performance (mAP@0.5:0.95): {final_eval_map:.4f}\n")
        f.write(f"Detection Performance (mAP@0.5): {final_eval_metrics.stats[1]:.4f}\n")
        f.write(f"Overall Classification Accuracy: {final_overall_accuracy:.4f}\n")
        f.write(f"Overall Miss Rate: {final_miss_rate_results['overall_miss_rate']:.4f}\n")
        f.write(f"Overall Recall: {final_miss_rate_results['overall_recall']:.4f}\n\n")

        f.write("Classification Accuracy per Class:\n")
        if binary_classification:
            class_names = {1: 'large_vehicle', 2: 'small_vehicle'}
        else:
            class_names = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}
            
        for class_id, accuracy in final_class_accuracy.items():
            if class_id in class_names:
                f.write(f"  {class_names[class_id]}: {accuracy:.4f}\n")
        
        f.write("\nMiss Rate per Class:\n")
        for class_id, miss_rate in final_miss_rate_results['miss_rates'].items():
            if class_id in class_names:
                f.write(f"  {class_names[class_id]}: {miss_rate:.4f}\n")
        
        f.write("\nRecall per Class:\n")
        for class_id, recall in final_miss_rate_results['recall_rates'].items():
            if class_id in class_names:
                f.write(f"  {class_names[class_id]}: {recall:.4f}\n")
        
        f.write("\nDetailed Metrics per Class:\n")
        f.write("Class        | True Positives | False Negatives | Ground Truth | Miss Rate | Recall\n")
        f.write("-" * 85 + "\n")
        
        for class_id in class_names.keys():
            if class_id in final_miss_rate_results['metrics']['true_positives']:
                tp = final_miss_rate_results['metrics']['true_positives'][class_id]
                fn = final_miss_rate_results['metrics']['false_negatives'][class_id]
                gt = final_miss_rate_results['metrics']['ground_truth_count'][class_id]
                miss_rate = final_miss_rate_results['miss_rates'][class_id]
                recall = final_miss_rate_results['recall_rates'][class_id]
                
                f.write(f"{class_names[class_id]:<12} | {tp:14d} | {fn:15d} | {gt:12d} | {miss_rate:9.4f} | {recall:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write("                 Predicted Class\n")

        if binary_classification:
            f.write("                 --------------------------------\n")
            f.write("True Class       | Large  | Small  |\n")
            f.write("-" * 42 + "\n")

            for true_class in [1, 2]:  # Large, Small
                row = final_class_metrics["confusion_matrix"][true_class]
                class_name = 'large_vehicle' if true_class == 1 else 'small_vehicle'
                f.write(f"{class_name:<16} | {row[1]:5d} | {row[2]:5d} |\n")
        else:
            f.write("                 --------------------------------\n")
            f.write("True Class       | Truck | Car   | Van   | Bus   |\n")
            f.write("-" * 60 + "\n")

            for true_class in range(1, 5):
                row = final_class_metrics["confusion_matrix"][true_class]
                class_name = class_names[true_class] if true_class in class_names else f"Unknown {true_class}"
                f.write(f"{class_name:<16} | {row[1]:5d} | {row[2]:5d} | {row[3]:5d} | {row[4]:5d} |\n")

    print(f"\nFull results saved to: {log_file}")

def save_training_plots_with_miss_rate(epochs_list, train_loss_list, val_map_list, final_eval_map,
                        class_ap_history, final_class_accuracy, final_overall_accuracy,
                        confusion_matrix, final_miss_rate_results, plot_dir, 
                        subset_info="", binary_classification=True):
    """
    Create comprehensive visualization plots for training results including miss rate
    """
    # Create a 2x3 subplot figure
    plt.figure(figsize=(18, 12))

    # Plot training loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs_list[:len(train_loss_list)], train_loss_list, 'b-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot validation mAP
    plt.subplot(2, 3, 2)
    plt.plot(epochs_list[:len(val_map_list)], val_map_list, 'r-o', label='Validation mAP')
    plt.axhline(y=final_eval_map, color='g', linestyle='--', label=f'Final mAP: {final_eval_map:.4f}')
    plt.title('Mean Average Precision')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5:0.95')
    plt.grid(True)
    plt.legend()

    # Plot per-class AP
    plt.subplot(2, 3, 3)
    if binary_classification:
        class_name_map = {1: 'large_vehicle', 2: 'small_vehicle'}
    else:
        class_name_map = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}

    for class_id in class_name_map.keys():
        if class_id in class_ap_history:
            ap_values = class_ap_history[class_id]
            if len(ap_values) > 0:
                plt.plot(epochs_list[:len(ap_values)],
                         ap_values,
                         'o-',
                         label=f'AP {class_name_map[class_id]}')

    plt.title('Per-Class Average Precision')
    plt.xlabel('Epochs')
    plt.ylabel('AP@0.5')
    plt.grid(True)
    plt.legend()

    # Plot classification accuracy
    plt.subplot(2, 3, 4)
    accuracies = [final_class_accuracy[i] for i in class_name_map.keys() if i in final_class_accuracy]
    class_names = [class_name_map[i] for i in class_name_map.keys() if i in final_class_accuracy]
    plt.bar(class_names, accuracies, color='lightgreen')
    plt.axhline(y=final_overall_accuracy, color='r', linestyle='--',
                label=f'Overall: {final_overall_accuracy:.4f}')
    plt.ylim(0, 1.0)
    plt.title('Classification Accuracy per Class')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Plot miss rate
    plt.subplot(2, 3, 5)
    miss_rates = [final_miss_rate_results['miss_rates'][i] for i in class_name_map.keys() 
                  if i in final_miss_rate_results['miss_rates']]
    plt.bar(class_names, miss_rates, color='lightcoral')
    plt.axhline(y=final_miss_rate_results['overall_miss_rate'], color='r', linestyle='--',
                label=f'Overall: {final_miss_rate_results["overall_miss_rate"]:.4f}')
    plt.ylim(0, 1.0)
    plt.title('Miss Rate per Class')
    plt.ylabel('Miss Rate')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Plot recall
    plt.subplot(2, 3, 6)
    recalls = [final_miss_rate_results['recall_rates'][i] for i in class_name_map.keys() 
               if i in final_miss_rate_results['recall_rates']]
    plt.bar(class_names, recalls, color='lightblue')
    plt.axhline(y=final_miss_rate_results['overall_recall'], color='b', linestyle='--',
                label=f'Overall: {final_miss_rate_results["overall_recall"]:.4f}')
    plt.ylim(0, 1.0)
    plt.title('Recall per Class')
    plt.ylabel('Recall')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    binary_suffix = "_binary" if binary_classification else ""
    plt.savefig(os.path.join(plot_dir, f'training_summary_with_miss_rate{subset_info}{binary_suffix}.png'))
    plt.close()
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add class labels
    classes = [class_name_map[i] for i in sorted(class_name_map.keys())]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(int(confusion_matrix[i, j]), 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plot_dir, f'confusion_matrix{subset_info}{binary_suffix}.png'))
    plt.close()
    
    print(f"Saved comprehensive plots with miss rate visualization")
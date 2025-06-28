import torch
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Trains the model for one epoch

    Args:
        model: The model to train
        optimizer: The optimizer to use
        data_loader: DataLoader with training data
        device: Device to train on
        epoch: Current epoch number
        print_freq: How often to print progress

    Returns:
        Dictionary of loss values
    """
    model.train()

    # Initialize metrics tracking
    batch_time_avg = 0
    data_time_avg = 0
    loss_avg = 0
    classification_loss_avg = 0
    regression_loss_avg = 0

    # Initialize loss dictionary to track losses by type
    loss_dict_sum = {
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0
    }

    start_time = time.time()

    # Progress bar for training
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Training")

    # Training loop
    for i, (images, targets) in enumerate(progress_bar):
        # Measure data loading time
        data_time = time.time() - start_time
        data_time_avg = (data_time_avg * i + data_time) / (i + 1)

        # Move images and targets to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Update loss averages
        loss_avg = (loss_avg * i + losses.item()) / (i + 1)
        for k, v in loss_dict.items():
            loss_dict_sum[k] += v.item()

        # Backward pass
        losses.backward()

        # Update weights
        optimizer.step()

        # Measure elapsed time
        batch_time = time.time() - start_time
        batch_time_avg = (batch_time_avg * i + batch_time) / (i + 1)
        start_time = time.time()

        # Update progress bar with current loss
        progress_bar.set_postfix({
            'loss': f"{loss_avg:.4f}",
            'batch_time': f"{batch_time:.3f}s",
            'data_time': f"{data_time:.3f}s"
        })

        # Print detailed loss breakdown periodically
        if i % print_freq == 0 and i > 0:
            print(f"\nEpoch: {epoch + 1}, Batch: {i}/{len(data_loader)}")
            print(f"  Classification loss: {loss_dict['loss_classifier'].item():.4f}")
            print(f"  Box regression loss: {loss_dict['loss_box_reg'].item():.4f}")
            print(f"  Objectness loss: {loss_dict['loss_objectness'].item():.4f}")
            print(f"  RPN box regression loss: {loss_dict['loss_rpn_box_reg'].item():.4f}")
            print(f"  Total loss: {losses.item():.4f}")

    # Calculate average loss values across the entire epoch
    num_batches = len(data_loader)
    for k in loss_dict_sum:
        loss_dict_sum[k] /= num_batches

    # Print epoch summary
    print(f"\nEpoch {epoch + 1} completed. Average losses:")
    print(f"  Classification: {loss_dict_sum['loss_classifier']:.4f}")
    print(f"  Box regression: {loss_dict_sum['loss_box_reg']:.4f}")
    print(f"  Objectness: {loss_dict_sum['loss_objectness']:.4f}")
    print(f"  RPN box regression: {loss_dict_sum['loss_rpn_box_reg']:.4f}")
    print(f"  Total: {sum(loss_dict_sum.values()):.4f}")

    return loss_dict_sum


def evaluate(model, data_loader, device, dataset=None, print_per_class=True, binary_classification=True):
    """
    Evaluate the model on the given data loader.
    Returns per-class metrics and overall detection metrics.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        dataset: Optional dataset object for class names
        print_per_class: Whether to print per-class metrics
        binary_classification: Whether using binary classification (large/small vehicles)
    """
    model.eval()

    # Define class IDs based on classification mode
    if binary_classification:
        # Binary classification: 0=background, 1=large_vehicle, 2=small_vehicle
        class_ids = {1, 2}
    else:
        # Original 4-class model: 0=background, 1=truck, 2=car, 3=van, 4=bus
        class_ids = {1, 2, 3, 4}

    # Initialize counters for per-class metrics
    class_metrics = {
        "tp": {class_id: 0 for class_id in class_ids},  # True positives per class
        "fp": {class_id: 0 for class_id in class_ids},  # False positives per class
        "fn": {class_id: 0 for class_id in class_ids},  # False negatives per class
        "gt_count": {class_id: 0 for class_id in class_ids}  # Total ground truth objects per class
    }

    all_image_ids = set()  # To store all unique image IDs
    all_annotations = []  # To accumulate ground truth annotations
    coco_dt = []  # To accumulate model predictions (detections)

    # Loop over the data loader and accumulate ground truth and predictions
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [image.to(device) for image in images]
        # Ensure all targets are moved to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Accumulate ground truth counts
        for target in targets:
            image_id = target['image_id'].item()
            all_image_ids.add(image_id)

            # Extract bounding boxes, labels, and areas from target
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            areas = target['area'].cpu().numpy()

            # Update ground truth counts for each class
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label in class_metrics["gt_count"]:
                    class_metrics["gt_count"][label] += count

            # Create COCO-format annotations for each box
            for i in range(len(boxes)):
                label = int(labels[i])
                ann = {
                    'id': len(all_annotations) + 1,  # Unique annotation id
                    'image_id': image_id,
                    'category_id': label,
                    'bbox': boxes[i].tolist(),  # Format: [x, y, width, height]
                    'area': float(areas[i]),
                    'iscrowd': 0
                }
                all_annotations.append(ann)

        with torch.no_grad():
            outputs = model(images)

        # Accumulate predictions for each image in the batch
        for idx in range(len(images)):
            image_id = targets[idx]['image_id'].item()
            gt_boxes = targets[idx]['boxes'].cpu().numpy()
            gt_labels = targets[idx]['labels'].cpu().numpy()

            # Get predictions
            pred_boxes = outputs[idx]['boxes'].cpu().numpy()
            pred_scores = outputs[idx]['scores'].cpu().numpy()
            pred_labels = outputs[idx]['labels'].cpu().numpy()

            # Filter out predictions with low scores
            threshold = 0.5
            valid_idx = pred_scores >= threshold
            pred_boxes = pred_boxes[valid_idx]
            pred_scores = pred_scores[valid_idx]
            pred_labels = pred_labels[valid_idx]

            # Add to COCO format detections
            for i in range(len(pred_boxes)):
                label = int(pred_labels[i])
                coco_dt.append({
                    'image_id': image_id,
                    'category_id': label,
                    'bbox': pred_boxes[i].tolist(),
                    'score': float(pred_scores[i])
                })

    # Construct COCO ground truth object from accumulated data
    coco_gt = COCO()
    # Build the images list
    coco_gt.dataset['images'] = [{'id': img_id, 'file_name': f"{img_id}.jpg"} for img_id in all_image_ids]
    # Use the accumulated annotations
    coco_gt.dataset['annotations'] = all_annotations

    # Define the categories according to the correct class mapping
    if binary_classification:
        # Binary classification categories
        coco_gt.dataset['categories'] = [
            {'id': 1, 'name': 'large_vehicle'},  # trucks and buses
            {'id': 2, 'name': 'small_vehicle'},  # cars and vans
        ]
    else:
        # Original 4-class categories
        coco_gt.dataset['categories'] = [
            {'id': 1, 'name': 'truck'},  # annotation class 0
            {'id': 2, 'name': 'car'},  # annotation class 1
            {'id': 3, 'name': 'van'},  # annotation class 2
            {'id': 4, 'name': 'bus'}  # annotation class 3
        ]
    coco_gt.createIndex()

    # Now load the predictions (COCO DT) into COCO format
    coco_dt = coco_gt.loadRes(coco_dt)

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print detailed per-class metrics
    if print_per_class and hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
        if binary_classification:
            class_names = ['large_vehicle', 'small_vehicle']
        else:
            class_names = ['truck', 'car', 'van', 'bus']

        precisions = coco_eval.eval['precision']

        # Print detailed per-class metrics
        print("\n===== PER-CLASS METRICS =====")
        print("Class        | AP@[.5:.95] | AP@.50 | AP@.75 | Objects Count | Detection Count")
        print("-" * 80)

        class_performances = {}

        # For each class
        for i, class_name in enumerate(class_names):
            class_id = i + 1  # Class IDs start from 1

            # AP across IoU thresholds
            ap_all = np.mean(precisions[:, :, i, 0, 2]) if precisions[:, :, i, 0, 2].size > 0 else float('nan')
            # AP at IoU=0.50
            ap_50 = np.mean(precisions[0, :, i, 0, 2]) if precisions[0, :, i, 0, 2].size > 0 else float('nan')
            # AP at IoU=0.75
            ap_75 = np.mean(precisions[5, :, i, 0, 2]) if precisions[5, :, i, 0, 2].size > 0 else float('nan')

            # Count of ground truth objects and detections
            gt_count = class_metrics["gt_count"].get(class_id, 0)

            # Count detections for this class
            detection_count = len([d for d in coco_dt.anns.values() if d['category_id'] == class_id])

            # Print metrics for this class
            print(
                f"{class_name:<12} | {ap_all:10.4f} | {ap_50:6.4f} | {ap_75:6.4f} | {gt_count:13d} | {detection_count:15d}")

            class_performances[class_name] = {
                'ap_all': ap_all,
                'ap_50': ap_50,
                'ap_75': ap_75,
                'gt_count': gt_count,
                'detection_count': detection_count
            }

        print("-" * 80)
        print(
            f"Overall mAP  | {coco_eval.stats[0]:10.4f} | {coco_eval.stats[1]:6.4f} | {coco_eval.stats[2]:6.4f} | {sum(class_metrics['gt_count'].values()):13d} | {len(coco_dt.anns):15d}")

        # Create a simple visualization of the class distribution and AP scores
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot class distribution
        counts = [class_performances[cls]['gt_count'] for cls in class_names]
        ax1.bar(class_names, counts, color='skyblue')
        ax1.set_title('Ground Truth Objects per Class')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot AP scores
        ap_values = [class_performances[cls]['ap_50'] for cls in class_names]
        ax2.bar(class_names, ap_values, color='lightgreen')
        ax2.set_title('Average Precision (IoU=0.50) per Class')
        ax2.set_ylabel('AP@0.50')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig('class_performance.png')
        plt.close()

        print("\nClass performance visualization saved to 'class_performance.png'")

    return coco_eval

def calculate_classification_accuracy(model, data_loader, device, iou_threshold=0.5, binary_classification=True):
    """
    Calculate classification accuracy for each class.

    This function measures how often the model correctly classifies an object
    when it successfully detects it (based on IoU threshold).

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        iou_threshold: IoU threshold to consider a detection as correct
        binary_classification: Whether using binary classification (large/small vehicles)

    Returns:
        Dictionary with classification metrics per class
    """
    model.eval()

    # Initialize counters for per-class metrics based on classification type
    if binary_classification:
        class_metrics = {
            "correct": {1: 0, 2: 0},  # Correctly classified
            "total_detected": {1: 0, 2: 0},  # Total correctly detected for each true class
            "confusion_matrix": {  # [true_class][pred_class]
                1: {1: 0, 2: 0},
                2: {1: 0, 2: 0}
            },
            "gt_count": {1: 0, 2: 0}  # Total ground truth objects per class
        }
    else:
        class_metrics = {
            "correct": {1: 0, 2: 0, 3: 0, 4: 0},  # Correctly classified
            "total_detected": {1: 0, 2: 0, 3: 0, 4: 0},  # Total correctly detected for each true class
            "confusion_matrix": {  # [true_class][pred_class]
                1: {1: 0, 2: 0, 3: 0, 4: 0},
                2: {1: 0, 2: 0, 3: 0, 4: 0},
                3: {1: 0, 2: 0, 3: 0, 4: 0},
                4: {1: 0, 2: 0, 3: 0, 4: 0}
            },
            "gt_count": {1: 0, 2: 0, 3: 0, 4: 0}  # Total ground truth objects per class
        }

    print("\nCalculating classification accuracy for each class...")

    # Loop through batches
    for images, targets in tqdm(data_loader, desc="Calculating accuracy"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Process one image at a time for detailed analysis
        with torch.no_grad():
            outputs = model(images)

        # For each image in the batch
        for idx in range(len(images)):
            # Get ground truth boxes and labels
            gt_boxes = targets[idx]['boxes'].cpu().numpy()
            gt_labels = targets[idx]['labels'].cpu().numpy()

            # Get prediction boxes, scores, and labels
            pred_boxes = outputs[idx]['boxes'].cpu().numpy()
            pred_scores = outputs[idx]['scores'].cpu().numpy()
            pred_labels = outputs[idx]['labels'].cpu().numpy()

            # Skip if no predictions or ground truths
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue

            # Count ground truth objects
            for label in gt_labels:
                class_metrics["gt_count"][label.item()] += 1

            # Filter predictions by confidence score
            confidence_threshold = 0.5
            confident_detections = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[confident_detections]
            pred_scores = pred_scores[confident_detections]
            pred_labels = pred_labels[confident_detections]

            # Skip if no confident predictions
            if len(pred_boxes) == 0:
                continue

            # Calculate IoU between each ground truth and prediction
            ious = np.zeros((len(gt_boxes), len(pred_boxes)))
            for gt_idx, gt_box in enumerate(gt_boxes):
                for pred_idx, pred_box in enumerate(pred_boxes):
                    # Calculate intersection area
                    x1 = max(gt_box[0], pred_box[0])
                    y1 = max(gt_box[1], pred_box[1])
                    x2 = min(gt_box[2], pred_box[2])
                    y2 = min(gt_box[3], pred_box[3])

                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        union = gt_area + pred_area - intersection
                        iou = intersection / union
                    else:
                        iou = 0.0

                    ious[gt_idx, pred_idx] = iou

            # For each ground truth box, find the best matching prediction
            gt_matched = set()
            pred_matched = set()

            # Match predictions to ground truths in descending order of IoU
            for _ in range(min(len(gt_boxes), len(pred_boxes))):
                # Find highest IoU among remaining pairs
                if np.max(ious) < iou_threshold:
                    break

                gt_idx, pred_idx = np.unravel_index(np.argmax(ious), ious.shape)

                if gt_idx not in gt_matched and pred_idx not in pred_matched:
                    gt_matched.add(gt_idx)
                    pred_matched.add(pred_idx)

                    gt_label = gt_labels[gt_idx]
                    pred_label = pred_labels[pred_idx]

                    # Update confusion matrix
                    class_metrics["confusion_matrix"][gt_label.item()][pred_label.item()] += 1

                    # Update counters
                    class_metrics["total_detected"][gt_label.item()] += 1
                    if gt_label == pred_label:
                        class_metrics["correct"][gt_label.item()] += 1

                # Set this pair's IoU to 0 to exclude it from further consideration
                ious[gt_idx, :] = 0
                ious[:, pred_idx] = 0

    # Calculate classification accuracy for each class
    if binary_classification:
        class_names = {1: 'large_vehicle', 2: 'small_vehicle'}
    else:
        class_names = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}

    class_accuracy = {}

    print("\n===== CLASSIFICATION ACCURACY =====")
    print("Class        | Accuracy | Correctly Classified | Total Detected | Total Ground Truth")
    print("-" * 80)

    overall_correct = 0
    overall_detected = 0

    # Calculate accuracy for each class
    for class_id in class_names.keys():
        correct = class_metrics["correct"][class_id]
        total_detected = class_metrics["total_detected"][class_id]
        gt_count = class_metrics["gt_count"][class_id]

        # Calculate accuracy, handling divide-by-zero
        accuracy = correct / total_detected if total_detected > 0 else 0.0
        class_accuracy[class_id] = accuracy

        print(f"{class_names[class_id]:<12} | {accuracy:8.4f} | {correct:19d} | {total_detected:14d} | {gt_count:18d}")

        overall_correct += correct
        overall_detected += total_detected

    # Calculate overall classification accuracy
    overall_accuracy = overall_correct / overall_detected if overall_detected > 0 else 0.0
    print("-" * 80)
    print(
        f"Overall      | {overall_accuracy:8.4f} | {overall_correct:19d} | {overall_detected:14d} | {sum(class_metrics['gt_count'].values()):18d}")

    # Print confusion matrix
    print("\n===== CONFUSION MATRIX =====")
    print("                 Predicted Class")

    if binary_classification:
        print("                 --------------------------------")
        print("True Class       | Large  | Small  |")
        print("-" * 42)

        for true_class in [1, 2]:
            row = class_metrics["confusion_matrix"][true_class]
            print(f"{class_names[true_class]:<16} | {row[1]:5d} | {row[2]:5d} |")
    else:
        print("                 --------------------------------")
        print("True Class       | Truck | Car   | Van   | Bus   |")
        print("-" * 60)

        for true_class in range(1, 5):
            row = class_metrics["confusion_matrix"][true_class]
            print(f"{class_names[true_class]:<16} | {row[1]:5d} | {row[2]:5d} | {row[3]:5d} | {row[4]:5d} |")

    # Create visualization for classification accuracy
    plt.figure(figsize=(12, 6))

    # Bar chart for classification accuracy
    classes = [class_names[i] for i in class_names.keys()]
    accuracies = [class_accuracy[i] for i in class_names.keys()]

    plt.bar(classes, accuracies, color='lightgreen')
    plt.axhline(y=overall_accuracy, color='r', linestyle='--', label=f'Overall: {overall_accuracy:.4f}')

    plt.ylim(0, 1.0)
    plt.title('Classification Accuracy per Class')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('classification_accuracy.png')
    plt.close()

    print("\nClassification accuracy visualization saved to 'classification_accuracy.png'")

    return class_accuracy, overall_accuracy, class_metrics

def calculate_miss_rate(model, data_loader, device, iou_threshold=0.5, confidence_threshold=0.5, binary_classification=True):
    """
    Calculate miss rate metrics for object detection model.
    Miss rate = 1 - Recall = False Negatives / (True Positives + False Negatives)
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        iou_threshold: IoU threshold to consider a detection as correct
        confidence_threshold: Confidence threshold for filtering predictions
        binary_classification: Whether using binary classification
    
    Returns:
        Dictionary with miss rate metrics per class and overall
    """
    model.eval()
    
    # Define class mapping
    if binary_classification:
        class_ids = {1, 2}
        class_names = {1: 'large_vehicle', 2: 'small_vehicle'}
    else:
        class_ids = {1, 2, 3, 4}
        class_names = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}
    
    # Initialize counters
    metrics = {
        "true_positives": {class_id: 0 for class_id in class_ids},
        "false_negatives": {class_id: 0 for class_id in class_ids},
        "ground_truth_count": {class_id: 0 for class_id in class_ids}
    }
    
    print("\nCalculating miss rates...")
    
    # Process each batch
    for images, targets in tqdm(data_loader, desc="Calculating miss rates"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images)
        
        # Process each image in the batch
        for idx in range(len(images)):
            gt_boxes = targets[idx]['boxes'].cpu().numpy()
            gt_labels = targets[idx]['labels'].cpu().numpy()
            
            pred_boxes = outputs[idx]['boxes'].cpu().numpy() 
            pred_scores = outputs[idx]['scores'].cpu().numpy()
            pred_labels = outputs[idx]['labels'].cpu().numpy()
            
            # Update ground truth counts
            for label in gt_labels:
                if label.item() in metrics["ground_truth_count"]:
                    metrics["ground_truth_count"][label.item()] += 1
            
            # Filter predictions by confidence
            confident_mask = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[confident_mask]
            pred_scores = pred_scores[confident_mask]
            pred_labels = pred_labels[confident_mask]
            
            # If no predictions, all ground truths are false negatives
            if len(pred_boxes) == 0:
                for label in gt_labels:
                    if label.item() in metrics["false_negatives"]:
                        metrics["false_negatives"][label.item()] += 1
                continue
                
            # Calculate IoU matrix
            ious = np.zeros((len(gt_boxes), len(pred_boxes)))
            for gt_idx, gt_box in enumerate(gt_boxes):
                for pred_idx, pred_box in enumerate(pred_boxes):
                    # Calculate IoU
                    x1 = max(gt_box[0], pred_box[0])
                    y1 = max(gt_box[1], pred_box[1])
                    x2 = min(gt_box[2], pred_box[2])
                    y2 = min(gt_box[3], pred_box[3])
                    
                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        union = gt_area + pred_area - intersection
                        iou = intersection / union
                    else:
                        iou = 0.0
                    
                    ious[gt_idx, pred_idx] = iou
            
            # Match predictions to ground truths
            gt_matched = set()
            pred_matched = set()
            
            # Sort by IoU in descending order and match
            while True:
                # Find highest IoU
                gt_idx, pred_idx = np.unravel_index(np.argmax(ious), ious.shape)
                best_iou = ious[gt_idx, pred_idx]
                
                if best_iou < iou_threshold:
                    break
                    
                if gt_idx not in gt_matched and pred_idx not in pred_matched:
                    gt_matched.add(gt_idx)
                    pred_matched.add(pred_idx)
                    
                    gt_label = gt_labels[gt_idx]
                    pred_label = pred_labels[pred_idx]
                    
                    # If labels match and IoU is sufficient, it's a true positive
                    if gt_label == pred_label:
                        metrics["true_positives"][gt_label.item()] += 1
                    else:
                        # Labels don't match, so this is still a false negative
                        metrics["false_negatives"][gt_label.item()] += 1
                
                # Set matched entries to 0 to exclude from further matching
                ious[gt_idx, :] = 0
                ious[:, pred_idx] = 0
            
            # Any unmatched ground truths are false negatives
            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx not in gt_matched:
                    metrics["false_negatives"][gt_label.item()] += 1
    
    # Calculate miss rates
    miss_rates = {}
    recall_rates = {}
    
    print("\n===== MISS RATE METRICS =====")
    print("Class        | Miss Rate | Recall | Ground Truth | True Positives | False Negatives")
    print("-" * 85)
    
    total_tp = 0
    total_fn = 0
    total_gt = 0
    
    for class_id in class_ids:
        tp = metrics["true_positives"][class_id]
        fn = metrics["false_negatives"][class_id]
        gt_count = metrics["ground_truth_count"][class_id]
        
        # Calculate recall and miss rate
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
            miss_rate = 1 - recall
        else:
            recall = 0.0
            miss_rate = 1.0
        
        miss_rates[class_id] = miss_rate
        recall_rates[class_id] = recall
        
        total_tp += tp
        total_fn += fn
        total_gt += gt_count
        
        print(f"{class_names[class_id]:<12} | {miss_rate:.4f} | {recall:.4f} | {gt_count:12d} | {tp:14d} | {fn:15d}")
    
    # Overall metrics
    if (total_tp + total_fn) > 0:
        overall_recall = total_tp / (total_tp + total_fn)
        overall_miss_rate = 1 - overall_recall
    else:
        overall_recall = 0.0
        overall_miss_rate = 1.0
    
    print("-" * 85)
    print(f"Overall      | {overall_miss_rate:.4f} | {overall_recall:.4f} | {total_gt:12d} | {total_tp:14d} | {total_fn:15d}")
    
    # Create miss rate visualization
    plt.figure(figsize=(12, 6))
    
    classes = [class_names[i] for i in class_ids]
    miss_rate_values = [miss_rates[i] for i in class_ids]
    
    bars = plt.bar(classes, miss_rate_values, color='lightcoral')
    
    # Add value labels on bars
    for bar, value in zip(bars, miss_rate_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.axhline(y=overall_miss_rate, color='r', linestyle='--', 
                label=f'Overall: {overall_miss_rate:.4f}')
    
    plt.ylim(0, 1.0)
    plt.title('Miss Rate per Class')
    plt.ylabel('Miss Rate (1 - Recall)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('miss_rate_performance.png')
    plt.close()
    
    print("\nMiss rate visualization saved to 'miss_rate_performance.png'")
    
    # Create recall visualization
    plt.figure(figsize=(12, 6))
    
    recall_values = [recall_rates[i] for i in class_ids]
    
    bars = plt.bar(classes, recall_values, color='lightgreen')
    
    # Add value labels on bars
    for bar, value in zip(bars, recall_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.axhline(y=overall_recall, color='g', linestyle='--', 
                label=f'Overall: {overall_recall:.4f}')
    
    plt.ylim(0, 1.0)
    plt.title('Recall per Class')
    plt.ylabel('Recall')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('recall_performance.png')
    plt.close()
    
    print("Recall visualization saved to 'recall_performance.png'")
    
    return {
        'miss_rates': miss_rates,
        'recall_rates': recall_rates,
        'overall_miss_rate': overall_miss_rate,
        'overall_recall': overall_recall,
        'metrics': metrics
    }


def calculate_miss_rate_vs_fppi(model, data_loader, device, confidence_thresholds=None, 
                               binary_classification=True):
    """
    Calculate miss rate vs false positives per image (FPPI) curve.
    This is a standard metric for object detection, especially in pedestrian detection.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        confidence_thresholds: List of confidence thresholds to test
        binary_classification: Whether using binary classification
    
    Returns:
        Dictionary with miss rate vs FPPI data for each class
    """
    
    model.eval()
    
    if confidence_thresholds is None:
        confidence_thresholds = np.arange(0.01, 1.0, 0.01)
    
    # Define class mapping
    if binary_classification:
        class_ids = {1, 2}
        class_names = {1: 'large_vehicle', 2: 'small_vehicle'}
    else:
        class_ids = {1, 2, 3, 4}
        class_names = {1: 'truck', 2: 'car', 3: 'van', 4: 'bus'}
    
    # Collect all detections and ground truths
    all_detections = []
    all_ground_truths = []
    num_images = 0
    
    print("\nCollecting detections for miss rate vs FPPI analysis...")
    
    for images, targets in tqdm(data_loader, desc="Processing images"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images)
        
        for idx in range(len(images)):
            num_images += 1
            
            # Get ground truth
            gt_boxes = targets[idx]['boxes'].cpu().numpy()
            gt_labels = targets[idx]['labels'].cpu().numpy()
            
            # Get predictions
            pred_boxes = outputs[idx]['boxes'].cpu().numpy()
            pred_scores = outputs[idx]['scores'].cpu().numpy()
            pred_labels = outputs[idx]['labels'].cpu().numpy()
            
            all_ground_truths.append({
                'boxes': gt_boxes,
                'labels': gt_labels
            })
            
            all_detections.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })
    
    # Calculate miss rate and FPPI for each threshold
    miss_rate_fppi_data = {class_id: {'miss_rates': [], 'fppis': []} 
                          for class_id in class_ids}
    
    for threshold in tqdm(confidence_thresholds, desc="Calculating miss rates"):
        metrics = {
            "true_positives": {class_id: 0 for class_id in class_ids},
            "false_positives": {class_id: 0 for class_id in class_ids},
            "false_negatives": {class_id: 0 for class_id in class_ids},
            "ground_truth_count": {class_id: 0 for class_id in class_ids}
        }
        
        # Process each image
        for det, gt in zip(all_detections, all_ground_truths):
            # Filter detections by threshold
            mask = det['scores'] >= threshold
            pred_boxes = det['boxes'][mask]
            pred_labels = det['labels'][mask]
            
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']
            
            # Update ground truth counts
            for label in gt_labels:
                if label.item() in metrics["ground_truth_count"]:
                    metrics["ground_truth_count"][label.item()] += 1
            
            # If no predictions, all ground truths are false negatives
            if len(pred_boxes) == 0:
                for label in gt_labels:
                    if label.item() in metrics["false_negatives"]:
                        metrics["false_negatives"][label.item()] += 1
                continue
            
            # Calculate IoU matrix
            ious = np.zeros((len(gt_boxes), len(pred_boxes)))
            for gt_idx, gt_box in enumerate(gt_boxes):
                for pred_idx, pred_box in enumerate(pred_boxes):
                    # Calculate IoU
                    x1 = max(gt_box[0], pred_box[0])
                    y1 = max(gt_box[1], pred_box[1])
                    x2 = min(gt_box[2], pred_box[2])
                    y2 = min(gt_box[3], pred_box[3])
                    
                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        union = gt_area + pred_area - intersection
                        iou = intersection / union
                    else:
                        iou = 0.0
                    
                    ious[gt_idx, pred_idx] = iou
            
            # Match predictions to ground truths
            gt_matched = set()
            pred_matched = set()
            
            # Sort by IoU and match
            for _ in range(min(len(gt_boxes), len(pred_boxes))):
                if np.max(ious) < 0.5:  # IoU threshold
                    break
                
                gt_idx, pred_idx = np.unravel_index(np.argmax(ious), ious.shape)
                
                if gt_idx not in gt_matched and pred_idx not in pred_matched:
                    gt_matched.add(gt_idx)
                    pred_matched.add(pred_idx)
                    
                    gt_label = gt_labels[gt_idx]
                    pred_label = pred_labels[pred_idx]
                    
                    if gt_label == pred_label:
                        metrics["true_positives"][gt_label.item()] += 1
                    else:
                        metrics["false_positives"][pred_label.item()] += 1
                        metrics["false_negatives"][gt_label.item()] += 1
                
                ious[gt_idx, :] = 0
                ious[:, pred_idx] = 0
            
            # Unmatched predictions are false positives
            for pred_idx, pred_label in enumerate(pred_labels):
                if pred_idx not in pred_matched:
                    metrics["false_positives"][pred_label.item()] += 1
            
            # Unmatched ground truths are false negatives
            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx not in gt_matched:
                    metrics["false_negatives"][gt_label.item()] += 1
        
        # Calculate miss rate and FPPI for each class
        for class_id in class_ids:
            tp = metrics["true_positives"][class_id]
            fn = metrics["false_negatives"][class_id]
            fp = metrics["false_positives"][class_id]
            
            # Miss rate = FN / (TP + FN)
            if (tp + fn) > 0:
                miss_rate = fn / (tp + fn)
            else:
                miss_rate = 1.0
            
            # FPPI = FP / num_images
            fppi = fp / num_images
            
            miss_rate_fppi_data[class_id]['miss_rates'].append(miss_rate)
            miss_rate_fppi_data[class_id]['fppis'].append(fppi)
    
    # Plot miss rate vs FPPI curves
    plt.figure(figsize=(10, 8))
    
    for class_id in class_ids:
        # Sort by FPPI for proper curve plotting
        fppi_values = miss_rate_fppi_data[class_id]['fppis']
        miss_rate_values = miss_rate_fppi_data[class_id]['miss_rates']
        
        # Sort together
        sorted_pairs = sorted(zip(fppi_values, miss_rate_values))
        sorted_fppis, sorted_miss_rates = zip(*sorted_pairs)
        
        plt.semilogx(sorted_fppis, sorted_miss_rates, '-o', 
                    label=class_names[class_id])
    
    plt.xlabel('False Positives Per Image (FPPI)')
    plt.ylabel('Miss Rate')
    plt.title('Miss Rate vs FPPI')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlim([0.01, 10])
    plt.ylim([0, 1])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('miss_rate_vs_fppi.png', dpi=300)
    plt.close()
    
    print("\nMiss rate vs FPPI curve saved to 'miss_rate_vs_fppi.png'")
    
    return miss_rate_fppi_data
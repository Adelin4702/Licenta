import torch
import cv2
import numpy as np
import os
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models.detection as detection
from sort.class_aware_sort import ClassAwareSort  # Import SORT tracker
from roadWidth import get_max_road_width_y
import datetime
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import models
import argparse
from db_functions import TrafficDatabase, get_next_hour_timestamp


def get_least_used_gpu():
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                                stdout=subprocess.PIPE, text=True)
        free_memory = [int(x) for x in result.stdout.strip().split("\n")]
        return str(free_memory.index(max(free_memory)))
    except Exception:
        return "0"  # Default to first GPU if there's an issue


os.environ["CUDA_VISIBLE_DEVICES"] = get_least_used_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")


# --------------------------
# Helper: Compute IoU between two boxes
def compute_iou(boxA, boxB):
    # boxes: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


# --------------------------
# Modified: Convert Faster R-CNN output to SORT format (with class labels)
def convert_to_sort_format(predictions, conf_threshold=0.6):
    detections = []
    detection_labels = []

    for i in range(len(predictions['boxes'])):
        score = predictions['scores'][i].item()
        if score > conf_threshold:
            box = predictions['boxes'][i].cpu().numpy()
            label = predictions['labels'][i].item()
            detections.append([*box, score])  # [x1, y1, x2, y2, score]
            detection_labels.append(label)

    if detections:
        return np.array(detections), detection_labels
    else:
        return np.empty((0, 5)), []


# --------------------------
def load_model(model_path, num_classes=3, binary_classification=True):
    """
    Load a saved model from path with correct number of classes
    Args:
        model_path: Path to saved model checkpoint
        num_classes: Number of output classes (including background)
        binary_classification: Whether model was trained with binary classification
    """
    # Determine number of classes from binary_classification flag if not specified
    if num_classes is None:
        num_classes = 3 if binary_classification else 5  # background + 2 or 4 classes

    # Initialize model with the appropriate configuration
    model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Modify the predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Handle different possible state dict formats
    if 'model_state_dict' in state_dict:
        # If the checkpoint was saved with additional metadata
        state_dict = state_dict['model_state_dict']

    # Remove module. prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove module. prefix
        name = k[7:] if k.startswith('module.') else k

        # Verify the key exists in the current model
        if name in model.state_dict():
            new_state_dict[name] = v

    # Partially load the state dict, ignoring mismatched keys
    model.load_state_dict(new_state_dict, strict=False)

    # Move model to the appropriate device
    model = model.to(device)
    model.eval()
    return model


# --------------------------
# Run object detection
def detect_objects(model, frame, device):
    transform = T.Compose([T.ToTensor()])
    img = transform(frame).to(device)
    with torch.no_grad():
        prediction = model([img])[0]
    return prediction


# --------------------------
def save_data_and_reset_counters(db, current_time, binary_classification, 
                                 large_vehicles_set, small_vehicles_set,
                                 cars_set, vans_set, trucks_set, busses_set):
    """Save current data to database and reset counters"""
    print(f"Saving data at: {current_time}")
    
    if binary_classification:
        db.save_binary_data(current_time, len(large_vehicles_set), len(small_vehicles_set))
        # Reset counters
        large_vehicles_set.clear()
        small_vehicles_set.clear()
    else:
        db.save_normal_data(current_time, len(cars_set), len(vans_set), len(trucks_set), len(busses_set))
        # Reset counters
        cars_set.clear()
        vans_set.clear()
        trucks_set.clear()
        busses_set.clear()


# --------------------------
# Initialize model and tracker
def track(video_path, model_path=None, binary_classification=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use provided model path or default
    if model_path is None:
        model_path = "/mnt/QNAP/apricop/container/Experiment11/outputs/models/final_model_subset_binary_epoch_5_map_0.7004.pth"

    # Determine number of classes
    num_classes = 3 if binary_classification else 5  # background + 2 or 4 vehicle classes

    model = load_model(model_path, num_classes, binary_classification)
    
    # Use standard SORT tracker - now supports class labels
    tracker = ClassAwareSort()

    # Dictionary to store tracking history (for drawing lines)
    track_history = {}

    # Define color mapping for each class
    if binary_classification:
        # For binary classification: 0=background, 1=large_vehicle, 2=small_vehicle
        color_map = {
            0: (255, 255, 255),  # background - white
            1: (0, 0, 255),  # large_vehicle - red
            2: (0, 255, 0)  # small_vehicle - green
        }
    else:
        # For original 4-class model
        color_map = {
            0: (255, 255, 255),  # background - white
            1: (0, 255, 0),  # truck - green
            2: (255, 0, 0),  # car - blue
            3: (0, 0, 255),  # van - red
            4: (0, 255, 255)  # bus - yellow
        }

    default_color = (255, 255, 255)  # white for unknown

    # --------------------------
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nof = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = nof / fps
    parts = video_path.split('/')
    number = parts[-1][:-4]

    # Output name based on classification mode
    output_name = f"{number}_output_binary.mp4" if binary_classification else f"{number}_output_tracked.mp4"
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    nrOfFrames = 0
    total_time = datetime.timedelta(0)
    total_time_1 = datetime.timedelta(0)
    reducing_factor = max(1, int(fps / 30))
    nrOfOutFrames = 0

    # Initialize database
    db_name = "traffic_binary.db" if binary_classification else "traffic_normal.db"
    db = TrafficDatabase(db_name)

    # --------------------------
    # Counters for each type of vehicle
    
    large_vehicles_set = set()  # trucks and buses
    small_vehicles_set = set()  # cars and vans

    cars_set = set()
    trucks_set = set()
    vans_set = set()
    busses_set = set()

    other_set = set()
    yline = 0

    # Time management for hourly saving
    video_start_time = datetime.datetime.now()
    current_hour_timestamp = video_start_time.replace(minute=0, second=0, microsecond=0)
    next_hour_timestamp = get_next_hour_timestamp(current_hour_timestamp)
    
    # --------------------------
    # Process video
    while cap.isOpened():
        nrOfFrames += 1
        if nrOfFrames % reducing_factor == 0:
            nrOfOutFrames += 1
            start_time_1 = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate current timestamp based on frame number
            video_elapsed_seconds = nrOfFrames / fps
            current_video_time = video_start_time + datetime.timedelta(seconds=video_elapsed_seconds)
            
            # Check if we've crossed hour boundary
            if current_video_time >= next_hour_timestamp:
                save_data_and_reset_counters(db, next_hour_timestamp, binary_classification,
                                             large_vehicles_set, small_vehicles_set,
                                             cars_set, vans_set, trucks_set, busses_set)
                next_hour_timestamp = get_next_hour_timestamp(next_hour_timestamp)
            
            if nrOfFrames == 1:
                yline = get_max_road_width_y(frame)
                
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            start_time = datetime.datetime.now()
            # Object detection
            predictions = detect_objects(model, frame, device)
            end_time = datetime.datetime.now()
            total_time += (end_time - start_time)

            # Convert detections to SORT format
            detections, detection_labels = convert_to_sort_format(predictions)
            
            # Update tracker with current detections and their labels
            tracked_objects = tracker.update(detections, detection_labels)
            
            # Process tracked objects
            active_track_ids = set()
            for track in tracked_objects:
                # Format: [x1, y1, x2, y2, track_id, class_label]
                x1, y1, x2, y2, track_id, class_label = track[0], track[1], track[2], track[3], int(track[4]), int(track[5])
                active_track_ids.add(track_id)
                
                # Choose color based on class label
                color = color_map.get(class_label, default_color)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Add to track history and draw tracking lines
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append(center)

                for i in range(1, len(track_history[track_id])):
                    cv2.line(frame, track_history[track_id][i - 1], track_history[track_id][i], (0, 0, 255), 2)

                # Get class name for display
                if binary_classification:
                    if class_label == 0:
                        className = "OTHER"
                    elif class_label == 1:
                        className = "LARGE VEHICLE"
                    elif class_label == 2:
                        className = "SMALL VEHICLE"
                    else:
                        className = "N/A"
                else:
                    if class_label == 0:
                        className = "OTHER"
                    elif class_label == 1:
                        className = "TRUCK"
                    elif class_label == 2:
                        className = "CAR"
                    elif class_label == 3:
                        className = "VAN"
                    elif class_label == 4:
                        className = "BUS"
                    else:
                        className = "N/A"

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Check if vehicle crosses the yline
                lastIndex = len(track_history[track_id]) - 1
                if lastIndex > 5:
                    y_prev = track_history[track_id][lastIndex - 5][1]  # 6th last y position
                    y_current = track_history[track_id][lastIndex][1]  # Current y position

                    if ((y_prev < yline and y_current > yline) or
                            (y_prev > yline and y_current < yline)):  # Ensure correct logical grouping

                        if binary_classification:
                            if class_label == 0:
                                other_set.add(track_id)
                            elif class_label == 1:  # Large vehicle
                                large_vehicles_set.add(track_id)
                            elif class_label == 2:  # Small vehicle
                                small_vehicles_set.add(track_id)
                        else:
                            if class_label == 0:
                                other_set.add(track_id)
                            elif class_label == 1:  # Truck
                                trucks_set.add(track_id)
                            elif class_label == 2:  # Car
                                cars_set.add(track_id)
                            elif class_label == 3:  # Van
                                vans_set.add(track_id)
                            elif class_label == 4:  # Bus
                                busses_set.add(track_id)

                # Draw counting line
                cv2.line(frame, (0, yline), (width, yline), (255, 0, 0), 2)
                cv2.putText(frame, f"{className} {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display vehicle counts
            if binary_classification:
                text = f"LARGE VEHICLES: {len(large_vehicles_set)}     SMALL VEHICLES: {len(small_vehicles_set)}"
            else:
                text = f"CARS: {len(cars_set)}     VANS: {len(vans_set)}    TRUCKS: {len(trucks_set)}    BUSSES: {len(busses_set)}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_color = (255, 255, 255)  # White text
            bg_color = (0, 0, 255)  # Red background

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Define text position
            x, y = 10, 30  # Adjust as needed

            # Draw background rectangle
            cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color,
                        thickness=cv2.FILLED)

            # Put text on top of the background
            cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness)

            end_time_1 = datetime.datetime.now()
            total_time_1 += (end_time_1 - start_time_1)
            # Save the processed frame
            out.write(frame)

    # Save final data at video end
    final_timestamp = video_start_time + datetime.timedelta(seconds=video_elapsed_seconds)
    save_data_and_reset_counters(db, final_timestamp, binary_classification,
                              large_vehicles_set, small_vehicles_set,
                              cars_set, vans_set, trucks_set, busses_set)

    # --------------------------
    # Cleanup
    cap.release()
    out.release()
    db.close()

    if nrOfFrames > 0:
        print(f"Mean Detection time: {total_time / nrOfOutFrames}")
        print(f"Total Detection time: {total_time}")
        print(f"Mean process time: {total_time_1 / nrOfOutFrames}")
        print(f"Total process time: {total_time_1}")
        print(f"Nr of frames: {nrOfFrames}")
        print(f"Nr of OUT frames: {nrOfOutFrames}")
        print(f"Red factor: {reducing_factor}")
        print(f"FPS: {fps}")
        print(f"Duration: {duration}")

        # Print vehicle counts based on classification mode
        if binary_classification:
            print(
                f"Vehicle counts: Large Vehicles: {len(large_vehicles_set)}, Small Vehicles: {len(small_vehicles_set)}")
        else:
            print(
                f"Vehicle counts: Cars: {len(cars_set)}, Vans: {len(vans_set)}, Trucks: {len(trucks_set)}, Busses: {len(busses_set)}")
    else:
        print("No frames were processed. Check your video file or path.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track vehicles in video using binary or 4-class model')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the model file')
    parser.add_argument('--binary', action='store_true', help='Use binary classification (large/small) model')

    args = parser.parse_args()
    track(args.video_path, args.model_path, args.binary)
import numpy as np
from sort.sort import Sort  # Import the original SORT implementation


class ClassAwareSort(Sort):
    """
    Extension of SORT to handle class labels
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize ClassAwareSort with the same parameters as SORT
        
        Args:
            max_age: Maximum number of frames to keep track alive without detection
            min_hits: Minimum number of hits to confirm a track
            iou_threshold: IoU threshold for detection-track association
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.track_labels = {}  # Maps track ID to class label
    
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
        Returns:
            IoU score (0-1)
        """
        # Determine intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection_area = width * height
        
        # Calculate individual box areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def update(self, dets=np.empty((0, 5)), det_labels=None):
        """
        Update the tracker with new detections and their class labels
        
        Args:
            dets: numpy array of detections in the format [[x1,y1,x2,y2,score], ...]
            det_labels: list of class labels corresponding to each detection
            
        Returns:
            numpy array of tracks in the format [[x1,y1,x2,y2,track_id,class_label], ...]
        """
        # Call the original SORT update method to get tracked objects
        tracks = super().update(dets)
        
        # Create a mapping of detection boxes to labels
        det_box_to_label = {}
        if det_labels is not None:
            for i, det in enumerate(dets):
                # Convert box to integer tuple for stable dict keys
                box_key = tuple(map(int, det[:4]))
                det_box_to_label[box_key] = det_labels[i]
        
        # For each track, find the detection with highest IoU to assign a label
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            track_box = [x1, y1, x2, y2]
            
            # Find the detection with highest IoU
            best_iou = 0.3  # Use the same threshold as SORT
            best_det_idx = -1
            best_label = None
            
            for i, det in enumerate(dets):
                det_box = det[:4]
                iou = self.compute_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
                    if det_labels is not None and i < len(det_labels):
                        best_label = det_labels[i]
            
            # If we found a good match, update the label
            if best_label is not None:
                self.track_labels[track_id] = best_label
        
        # Return tracked objects with their associated class labels
        result = []
        for track in tracks:
            track_id = int(track[4])
            # Get class label (default to 0/background if not found)
            label = self.track_labels.get(track_id, 0)
            # Add label as the 6th column
            result.append(np.append(track, label))
        
        return np.array(result) if len(result) > 0 else np.empty((0, 6))import numpy as np
from sort.sort import Sort  # Import the original SORT implementation


class ClassAwareSort(Sort):
    """
    Extension of SORT to handle class labels
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize ClassAwareSort with the same parameters as SORT
        
        Args:
            max_age: Maximum number of frames to keep track alive without detection
            min_hits: Minimum number of hits to confirm a track
            iou_threshold: IoU threshold for detection-track association
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.track_labels = {}  # Maps track ID to class label
    
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
        Returns:
            IoU score (0-1)
        """
        # Determine intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection_area = width * height
        
        # Calculate individual box areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def update(self, dets=np.empty((0, 5)), det_labels=None):
        """
        Update the tracker with new detections and their class labels
        
        Args:
            dets: numpy array of detections in the format [[x1,y1,x2,y2,score], ...]
            det_labels: list of class labels corresponding to each detection
            
        Returns:
            numpy array of tracks in the format [[x1,y1,x2,y2,track_id,class_label], ...]
        """
        # Call the original SORT update method to get tracked objects
        tracks = super().update(dets)
        
        # Create a mapping of detection boxes to labels
        det_box_to_label = {}
        if det_labels is not None:
            for i, det in enumerate(dets):
                # Convert box to integer tuple for stable dict keys
                box_key = tuple(map(int, det[:4]))
                det_box_to_label[box_key] = det_labels[i]
        
        # For each track, find the detection with highest IoU to assign a label
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            track_box = [x1, y1, x2, y2]
            
            # Find the detection with highest IoU
            best_iou = 0.3  # Use the same threshold as SORT
            best_det_idx = -1
            best_label = None
            
            for i, det in enumerate(dets):
                det_box = det[:4]
                iou = self.compute_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
                    if det_labels is not None and i < len(det_labels):
                        best_label = det_labels[i]
            
            # If we found a good match, update the label
            if best_label is not None:
                self.track_labels[track_id] = best_label
        
        # Return tracked objects with their associated class labels
        result = []
        for track in tracks:
            track_id = int(track[4])
            # Get class label (default to 0/background if not found)
            label = self.track_labels.get(track_id, 0)
            # Add label as the 6th column
            result.append(np.append(track, label))
        
        return np.array(result) if len(result) > 0 else np.empty((0, 6))
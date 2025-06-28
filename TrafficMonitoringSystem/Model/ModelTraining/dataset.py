import cv2
import numpy as np
import torch
import os
import albumentations as A
from albumentations import ToTensorV2
from torch.utils.data import WeightedRandomSampler, random_split

def create_datasets_from_paths(images_dir, labels_dir, img_size=480, subset_files=None, binary_classification=True):
    """
    Create a dataset from specified paths for images and labels
    
    Args:
        images_dir: Directory with images
        labels_dir: Directory with label annotations
        img_size: Size of images (square)
        subset_files: Optional list of specific image filenames to use
        binary_classification: Whether to use binary classification (large/small vehicles)
        
    Returns:
        Dataset object
    """
    # Create transformations for training (with augmentation)
    transforms = get_transform(train=True, img_size=img_size)
    transforms_minor = get_minor_class_transform(train=True, img_size=img_size)
    
    # Create dataset using specified paths
    dataset = UadDataset(
        files_dir=images_dir,
        annot_dir=labels_dir,
        width=img_size,
        height=img_size,
        transforms=transforms,
        transforms_minor=transforms_minor,
        custom_subset=subset_files,
        binary_classification=binary_classification
    )
    
    print(f"Created dataset from {images_dir} with {len(dataset)} images")
    
    return dataset

def get_transform(train, img_size):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Resize(img_size, img_size),
                ToTensorV2(p=1.0)
            ],
            bbox_params={
                'format': 'pascal_voc',
                'label_fields': ['labels'],
                'min_area': 1,
                'min_visibility': 0.1
            }
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                ToTensorV2(p=1.0)
            ],
            bbox_params={
                'format': 'pascal_voc',
                'label_fields': ['labels'],
                'min_area': 1,
                'min_visibility': 0.1
            }
        )


def get_minor_class_transform(train, img_size):
    if train:
        return A.Compose(
            [
                # Stronger augmentation for minority classes
                A.HorizontalFlip(p=0.7),
                A.RandomBrightnessContrast(p=0.7),
                A.HueSaturationValue(p=0.7),
                A.RandomGamma(p=0.5),
                A.MotionBlur(p=0.3),
                A.GaussNoise(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7
                ),
                A.Resize(img_size, img_size),
                ToTensorV2(p=1.0)
            ],
            bbox_params={
                'format': 'pascal_voc',
                'label_fields': ['labels'],
                'min_area': 1,
                'min_visibility': 0.1
            }
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                ToTensorV2(p=1.0)
            ],
            bbox_params={
                'format': 'pascal_voc',
                'label_fields': ['labels'],
                'min_area': 1,
                'min_visibility': 0.1
            }
        )


def load_custom_subset(subset_file):
    """
    Load a custom subset of image filenames from a file.

    Args:
        subset_file: Path to text file containing image filenames (one per line)
                    Example format: "img00001.jpg"

    Returns:
        List of image filenames
    """
    try:
        with open(subset_file, 'r') as f:
            # Read lines and strip whitespace
            filenames = [line.strip() for line in f.readlines()]

        print(f"Loaded {len(filenames)} images from custom subset file: {subset_file}")
        return filenames
    except Exception as e:
        print(f"Error loading custom subset file {subset_file}: {e}")
        return None


class UadDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, annot_dir, width, height, transforms=None,
                 transforms_minor=None, custom_subset=None, binary_classification=True):
        self.transforms = transforms
        self.transforms_minor = transforms_minor
        self.files_dir = files_dir
        self.annot_dir = annot_dir
        self.height = height
        self.width = width
        self.custom_subset = custom_subset
        self.binary_classification = binary_classification

        # Get all image files in the directory
        all_images = [image for image in sorted(os.listdir(files_dir)) if image[-4:] == '.jpg']

        # If custom_subset is provided, filter to only those images
        if custom_subset is not None:
            # Verify each image in the subset exists
            self.imgs = []
            missing_images = []
            for img_name in custom_subset:
                if img_name in all_images:
                    self.imgs.append(img_name)
                else:
                    missing_images.append(img_name)

            # Report on missing images
            if missing_images:
                print(f"Warning: {len(missing_images)} images from custom subset were not found in {files_dir}")
                if len(missing_images) < 10:  # Only print if there are few missing
                    print(f"Missing images: {missing_images}")
                else:
                    print(f"First 5 missing images: {missing_images[:5]}...")
        else:
            # Use all images in the directory
            self.imgs = all_images

        print(f"Using {len(self.imgs)} images from {files_dir}")

        if binary_classification:
            # For binary classification:
            # In annotation files: 0=truck, 1=car, 2=van, 3=bus
            # Remap to: 0=background, 1=large_vehicle (truck/bus), 2=small_vehicle (car/van)
            self.classes = ['background', 'large_vehicle', 'small_vehicle']

            # Mapping from original annotation class IDs to binary class IDs
            # For annotation files: 0(truck)->1(large), 1(car)->2(small), 2(van)->2(small), 3(bus)->1(large)
            self.class_mapping = {0: 1, 1: 2, 2: 2, 3: 1}
        else:
            # Original 4-class classification
            # Classes with ID 0 as background (for the model)
            # In annotation files: 0=truck, 1=car, 2=van, 3=bus
            # In the model: 1=truck, 2=car, 3=van, 4=bus (0=background)
            self.classes = ['background', 'truck', 'car', 'van', 'bus']
            self.class_mapping = {0: 1, 1: 2, 2: 3, 3: 4}  # Direct mapping with +1 offset

        # Count class distribution
        self.class_counts = self._count_class_distribution()

        # Calculate image weights for this dataset
        self.image_weights = self._calculate_image_weights()

    def _count_class_distribution(self):
        """Count instances of each class to understand distribution"""
        # For binary classification, we'll count large vs small
        if self.binary_classification:
            counts = {1: 0, 2: 0}  # 1=large_vehicle, 2=small_vehicle
        else:
            # Original annotation file class IDs
            counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 0=truck, 1=car, 2=van, 3=bus in annotation files

        total_images_with_annotations = 0

        print("Counting class distribution...")
        for img_name in self.imgs:
            annot_filename = img_name[:-4] + '.txt'
            annot_file_path = os.path.join(self.annot_dir, annot_filename)

            if os.path.exists(annot_file_path):
                has_annotation = False
                with open(annot_file_path) as f:
                    for line in f:
                        try:
                            class_id = int(line.split(' ')[0])
                            if 0 <= class_id <= 3:  # Valid class IDs in annotation
                                if self.binary_classification:
                                    # Map to binary classes (large or small vehicle)
                                    mapped_class = self.class_mapping[class_id]
                                    counts[mapped_class] += 1
                                else:
                                    counts[class_id] += 1
                                has_annotation = True
                        except (ValueError, IndexError):
                            continue

                if has_annotation:
                    total_images_with_annotations += 1

        # Print class distribution summary
        print(f"\nClass distribution summary:")
        print(f"Total images: {len(self.imgs)}")
        print(f"Images with annotations: {total_images_with_annotations}")

        if self.binary_classification:
            class_name_map = {1: 'large_vehicle', 2: 'small_vehicle'}
            for class_id, count in counts.items():
                print(f"  Model Class {class_id} ({class_name_map[class_id]}): {count} instances")
        else:
            class_name_map = {0: 'truck', 1: 'car', 2: 'van', 3: 'bus'}
            for class_id, count in counts.items():
                print(f"  Annotation Class {class_id} ({class_name_map[class_id]}): {count} instances")

        return counts

    def _calculate_image_weights(self):
        """Calculate weights for each image based on contained classes"""
        print("Calculating image weights for balanced sampling...")
        weights = []

        # Calculate inverse class frequency for weighting
        total_objects = sum(self.class_counts.values())
        class_weights = {cls_id: total_objects / (count + 1e-6) for cls_id, count in self.class_counts.items()}

        # Normalize weights
        max_weight = max(class_weights.values())
        normalized_weights = {cls_id: weight / max_weight for cls_id, weight in class_weights.items()}

        # Print class weights for sampling
        print("Class weights for sampling:")
        if self.binary_classification:
            class_name_map = {1: 'large_vehicle', 2: 'small_vehicle'}
        else:
            class_name_map = {0: 'truck', 1: 'car', 2: 'van', 3: 'bus'}

        for cls_id, weight in normalized_weights.items():
            class_name = class_name_map[cls_id] if cls_id in class_name_map else f"Class {cls_id}"
            print(f"  Class {cls_id} ({class_name}): {weight:.4f}")

        # Assign weights to images based on their contained classes
        for img_name in self.imgs:
            annot_filename = img_name[:-4] + '.txt'
            annot_file_path = os.path.join(self.annot_dir, annot_filename)

            # Default weight
            weight = 1.0

            if os.path.exists(annot_file_path):
                classes_in_image = set()
                with open(annot_file_path) as f:
                    for line in f:
                        try:
                            class_id = int(line.split(' ')[0])
                            if 0 <= class_id <= 3:  # Valid class IDs in annotation
                                if self.binary_classification:
                                    # Map to binary class
                                    mapped_class = self.class_mapping[class_id]
                                    classes_in_image.add(mapped_class)
                                else:
                                    classes_in_image.add(class_id)
                        except (ValueError, IndexError):
                            continue

                if classes_in_image:
                    # Assign weight based on the rarest class in the image
                    weight = max(normalized_weights[cls_id] for cls_id in classes_in_image)

                    # Extra weight if image contains minority classes
                    if self.binary_classification:
                        # Extra weight for large vehicles if they are minority
                        if 1 in classes_in_image and self.class_counts[1] < self.class_counts[2]:
                            weight *= 1.5
                    else:
                        # Extra weight for trucks and buses (original logic)
                        if 0 in classes_in_image or 3 in classes_in_image:
                            weight *= 1.5

            weights.append(weight)

        return weights

    def __getitem__(self, idx):
        try:
            img_name = self.imgs[idx]
            image_path = os.path.join(self.files_dir, img_name)

            # reading the images and converting them to correct size and color
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Trying next image.")
                return self.__getitem__((idx + 1) % len(self))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
            # diving by 255
            img_res /= 255.0

            # annotation file
            annot_filename = img_name[:-4] + '.txt'
            annot_file_path = os.path.join(self.annot_dir, annot_filename)

            boxes = []
            labels = []

            # box coordinates for txt files are extracted and corrected for image size given
            with open(annot_file_path) as f:
                for line in f:
                    try:
                        parsed = [float(x) for x in line.split(' ')]
                        if len(parsed) >= 5:  # Ensure we have at least 5 values
                            # Get original annotation class ID
                            annotation_class_id = int(parsed[0])

                            # Apply class mapping based on classification type
                            if 0 <= annotation_class_id <= 3:  # Valid class IDs
                                model_class_id = self.class_mapping[annotation_class_id]
                                labels.append(model_class_id)

                                # Parse bounding box
                                x_center = parsed[1]
                                y_center = parsed[2]
                                box_wt = parsed[3]
                                box_ht = parsed[4]

                                # Convert from center/width/height to top-left/bottom-right coords
                                xmin = x_center - box_wt / 2
                                xmax = x_center + box_wt / 2
                                ymin = y_center - box_ht / 2
                                ymax = y_center + box_ht / 2

                                # Scale to actual pixel coordinates
                                xmin_corr = int(xmin * self.width)
                                xmax_corr = int(xmax * self.width)
                                ymin_corr = int(ymin * self.height)
                                ymax_corr = int(ymax * self.height)

                                # Ensure valid box coordinates
                                if xmin_corr < xmax_corr and ymin_corr < ymax_corr:
                                    boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
                                else:
                                    # Remove the corresponding label if box is invalid
                                    labels.pop()
                    except (ValueError, IndexError):
                        continue

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            if boxes.numel() == 0:
                return self.__getitem__((idx + 1) % len(self))

            # getting the areas of the boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels_list = labels
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            image_id = torch.tensor([idx])
            target["image_id"] = image_id

            if self.transforms:
                # Apply more aggressive augmentations for minority classes
                try:
                    # Check for minority classes
                    if self.binary_classification:
                        # Check if image contains large vehicles (which might be minority)
                        if any(x == 1 for x in labels_list):
                            sample = self.transforms_minor(image=img_res,
                                                           bboxes=target['boxes'].numpy(),
                                                           labels=labels_list)
                        else:
                            sample = self.transforms(image=img_res,
                                                     bboxes=target['boxes'].numpy(),
                                                     labels=labels_list)
                    else:
                        # Original logic for 4-class model: Check for truck (1) and bus (4)
                        if any(x in [1, 4] for x in labels_list):
                            sample = self.transforms_minor(image=img_res,
                                                           bboxes=target['boxes'].numpy(),
                                                           labels=labels_list)
                        else:
                            sample = self.transforms(image=img_res,
                                                     bboxes=target['boxes'].numpy(),
                                                     labels=labels_list)

                    img_res = sample['image']
                    if len(sample['bboxes']) > 0:
                        target['boxes'] = torch.Tensor(sample['bboxes'])
                    else:
                        # If all boxes were removed during augmentation, try another image
                        return self.__getitem__((idx + 1) % len(self))
                except Exception as e:
                    print(f"Warning: Augmentation failed: {e}. Trying next image.")
                    return self.__getitem__((idx + 1) % len(self))

            return img_res, target

        except Exception as e:
            print(f"Error processing image at index {idx}: {e}. Trying next image.")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.imgs)


def create_datasets(files_dir, annot_dir, eval_dir, annot_eval_dir, img_size=480,
                    train_subset_file=None, eval_subset_file=None, binary_classification=True):
    """
    Create datasets with optional custom subsets

    Args:
        files_dir: Directory with training images
        annot_dir: Directory with training annotations
        eval_dir: Directory with evaluation images
        annot_eval_dir: Directory with evaluation annotations
        img_size: Size of images (square)
        train_subset_file: Optional path to file containing training image filenames
        eval_subset_file: Optional path to file containing evaluation image filenames
        binary_classification: Whether to use binary classification (large/small vehicles)

    Returns:
        full_dataset, eval_dataset
    """
    # Load custom subsets if specified
    train_subset = None
    eval_subset = None

    if train_subset_file and os.path.exists(train_subset_file):
        train_subset = load_custom_subset(train_subset_file)

    if eval_subset_file and os.path.exists(eval_subset_file):
        eval_subset = load_custom_subset(eval_subset_file)

    # Create datasets with potential custom subsets
    full_dataset = UadDataset(
        files_dir, annot_dir, img_size, img_size,
        transforms=get_transform(train=True, img_size=img_size),
        transforms_minor=get_minor_class_transform(train=True, img_size=img_size),
        custom_subset=train_subset,
        binary_classification=binary_classification
    )

    # Create evaluation dataset separately
    eval_dataset = UadDataset(
        eval_dir, annot_eval_dir, img_size, img_size,
        transforms=get_transform(train=False, img_size=img_size),
        transforms_minor=get_minor_class_transform(train=False, img_size=img_size),
        custom_subset=eval_subset,
        binary_classification=binary_classification
    )

    return full_dataset, eval_dataset


def split_dataset(dataset, train_percent=0.8):
    """Split dataset into train and test sets randomly"""
    import time

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_percent)
    test_size = dataset_size - train_size

    # Generate random split with seed for reproducibility
    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(int(time.time()) % 1000)
    )

    print(f"Split dataset: {train_size} training samples, {test_size} validation samples")

    return train_set, test_set


def create_weighted_sampler(dataset, indices):
    """Create a weighted sampler for a subset of the dataset"""
    # Extract weights for the specific indices
    weights = [dataset.image_weights[i] for i in indices]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True
    )

    return sampler


def collate_fn(batch):
    return tuple(zip(*batch))
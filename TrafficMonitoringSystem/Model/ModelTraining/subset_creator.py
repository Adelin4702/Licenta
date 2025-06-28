import os
import random


def parse_annotations(annot_path, img_name):
    """Parse annotation file to count objects of each class"""
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 0=truck, 1=car, 2=van, 3=bus

    annot_file = os.path.join(annot_path, img_name[:-4] + '.txt')

    if not os.path.exists(annot_file):
        return class_counts, 0

    try:
        with open(annot_file, 'r') as f:
            for line in f:
                try:
                    class_id = int(line.split(' ')[0])
                    if 0 <= class_id <= 3:
                        class_counts[class_id] += 1
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error parsing {annot_file}: {e}")

    total_objects = sum(class_counts.values())
    return class_counts, total_objects


def get_dataset_stats(image_dir, annot_dir):
    """Get statistics for all images in the dataset"""
    # Get all images
    all_images = [img for img in sorted(os.listdir(image_dir)) if img.endswith('.jpg')]

    print(f"Found {len(all_images)} images in {image_dir}")

    # Collect stats
    image_stats = []
    class_totals = {0: 0, 1: 0, 2: 0, 3: 0}
    total_objects = 0

    print("Analyzing dataset...")
    for img_name in all_images:
        class_counts, img_object_count = parse_annotations(annot_dir, img_name)

        # Record stats for this image
        image_stats.append({
            'filename': img_name,
            'class_counts': class_counts,
            'total_objects': img_object_count
        })

        # Update totals
        for class_id in class_counts:
            class_totals[class_id] += class_counts[class_id]
        total_objects += img_object_count

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total images: {len(all_images)}")
    print(f"Total objects: {total_objects}")
    print("Objects per class:")
    class_names = {0: 'truck', 1: 'car', 2: 'van', 3: 'bus'}
    for class_id in sorted(class_totals.keys()):
        class_count = class_totals[class_id]
        percentage = (class_count / total_objects) * 100 if total_objects > 0 else 0
        print(f"  Class {class_id} ({class_names[class_id]}): {class_count} ({percentage:.2f}%)")

    return image_stats, class_totals, total_objects


def create_random_subset(image_stats, size=1000, seed=42):
    """Create a random subset of the dataset"""
    random.seed(seed)

    all_images = [img['filename'] for img in image_stats]
    subset = random.sample(all_images, min(size, len(all_images)))

    print(f"Created random subset with {len(subset)} images")
    return subset


def create_balanced_subset(image_stats, size=1000, seed=42):
    """Create a subset with balanced class representation"""
    random.seed(seed)

    # Calculate target count for each class
    target_per_class = size // 4  # Equal distribution among 4 classes

    # Create lists of images containing each class
    class_images = {0: [], 1: [], 2: [], 3: []}

    for img_stat in image_stats:
        for class_id in range(4):
            if img_stat['class_counts'][class_id] > 0:
                class_images[class_id].append(img_stat)

    # Select images for each class
    selected_images = set()

    for class_id in range(4):
        # Prioritize images that have more of this class
        sorted_images = sorted(
            class_images[class_id],
            key=lambda x: x['class_counts'][class_id],
            reverse=True
        )

        # Select images until we reach the target or run out
        class_count = 0
        for img in sorted_images:
            if len(selected_images) >= size:
                break

            if img['filename'] not in selected_images:
                selected_images.add(img['filename'])
                class_count += img['class_counts'][class_id]

            if class_count >= target_per_class:
                break

    # If we don't have enough images, add random ones
    remaining_slots = size - len(selected_images)
    if remaining_slots > 0:
        # Get unselected images
        unselected = [img['filename'] for img in image_stats if img['filename'] not in selected_images]
        # Add random images to fill the subset
        additional = random.sample(unselected, min(remaining_slots, len(unselected)))
        selected_images.update(additional)

    print(f"Created balanced subset with {len(selected_images)} images")
    return list(selected_images)


def create_minority_focused_subset(image_stats, size=1000, minority_classes=[0, 3], seed=42):
    """Create a subset focused on minority classes (default: truck and bus)"""
    random.seed(seed)

    # Select all images containing minority classes
    minority_images = set()
    for img_stat in image_stats:
        if any(img_stat['class_counts'][cls] > 0 for cls in minority_classes):
            minority_images.add(img_stat['filename'])

    # If we need more images, add random ones
    if len(minority_images) < size:
        # Get images without minority classes
        majority_only = [img['filename'] for img in image_stats if img['filename'] not in minority_images]
        # Add random images to fill the subset
        additional_needed = size - len(minority_images)
        additional = random.sample(majority_only, min(additional_needed, len(majority_only)))
        minority_images.update(additional)

    # If we have too many, select a random subset
    if len(minority_images) > size:
        minority_images = set(random.sample(list(minority_images), size))

    print(f"Created minority-focused subset with {len(minority_images)} images")
    return list(minority_images)


def save_subset(filenames, output_file):
    """Save the subset filenames to a file"""
    with open(output_file, 'w') as f:
        for filename in sorted(filenames):
            f.write(f"{filename}\n")

    print(f"Saved {len(filenames)} filenames to {output_file}")


def analyze_subset(subset, image_stats, class_totals, total_objects):
    """Analyze and print statistics about the created subset"""
    subset_stats = [img for img in image_stats if img['filename'] in subset]
    subset_totals = {0: 0, 1: 0, 2: 0, 3: 0}
    subset_objects = 0

    for img_stat in subset_stats:
        for class_id in range(4):
            subset_totals[class_id] += img_stat['class_counts'][class_id]
        subset_objects += img_stat['total_objects']

    # Print subset statistics
    print("\nSubset Summary:")
    print(f"Total images: {len(subset)}")
    print(f"Total objects: {subset_objects}")
    print("Objects per class:")
    class_names = {0: 'truck', 1: 'car', 2: 'van', 3: 'bus'}
    for class_id in sorted(subset_totals.keys()):
        class_count = subset_totals[class_id]
        percentage = (class_count / subset_objects) * 100 if subset_objects > 0 else 0
        original_percentage = (class_totals[class_id] / total_objects) * 100 if total_objects > 0 else 0
        print(
            f"  Class {class_id} ({class_names[class_id]}): {class_count} ({percentage:.2f}%) [Original: {original_percentage:.2f}%]")

    return subset_stats, subset_totals, subset_objects


def create_subset(image_dir, annot_dir, subset_type, size=1000, output_file=None, seed=42):
    """Create a subset based on specified type and save to file if output_file is provided"""
    # Get dataset stats
    image_stats, class_totals, total_objects = get_dataset_stats(image_dir, annot_dir)

    # Create the appropriate subset based on type
    if subset_type == 1:  # Random subset
        print("Creating random subset...")
        subset = create_random_subset(image_stats, size=size, seed=seed)
    elif subset_type == 2:  # Minority-focused subset
        print("Creating minority-focused subset (focusing on trucks and buses)...")
        subset = create_minority_focused_subset(image_stats, size=size, seed=seed)
    elif subset_type == 3:  # Balanced subset
        print("Creating balanced subset...")
        subset = create_balanced_subset(image_stats, size=size, seed=seed)
    else:
        raise ValueError(f"Unsupported subset type: {subset_type}")

    # Analyze the subset
    analyze_subset(subset, image_stats, class_totals, total_objects)

    # Save the subset if output file is provided
    if output_file:
        save_subset(subset, output_file)
        print(f"Subset saved to {output_file}")

    return subset
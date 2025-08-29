
import cv2
import os
import numpy as np
import random

# --- CONFIGURATION ---
DATA_ROOT = 'archive' # Relative path to your dataset root (where data.yaml is)
IMAGE_SUBDIR = 'images'
LABEL_SUBDIR = 'labels'

# Class names from data.yaml
NAMES = {
    0: 'ball',
    1: 'goalkeeper',
    2: 'player',
    3: 'referee'
}

# Colors for bounding boxes (BGR format)
COLORS = {
    'ball': (0, 255, 255),       # Yellow
    'goalkeeper': (255, 0, 0),   # Blue
    'player': (0, 255, 0),       # Green
    'referee': (0, 0, 255)      # Red
}

LINE_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2

def visualize_annotations(split='train', num_samples=50):
    """
    Visualizes bounding box annotations on a sample of images.

    Args:
        split (str): 'train', 'val', or 'test'
        num_samples (int): Number of random samples to visualize.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust paths based on data.yaml structure
    if split == 'val':
        image_dir = os.path.join(project_dir, DATA_ROOT, 'valid', IMAGE_SUBDIR)
        label_dir = os.path.join(project_dir, DATA_ROOT, 'valid', LABEL_SUBDIR)
    elif split == 'train':
        image_dir = os.path.join(project_dir, DATA_ROOT, 'train', IMAGE_SUBDIR)
        label_dir = os.path.join(project_dir, DATA_ROOT, 'train', LABEL_SUBDIR)
    elif split == 'test':
        image_dir = os.path.join(project_dir, DATA_ROOT, 'test', IMAGE_SUBDIR)
        label_dir = os.path.join(project_dir, DATA_ROOT, 'test', LABEL_SUBDIR)
    else:
        print(f"Error: Invalid split '{split}'.")
        return

    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return
    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found: {label_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    # Randomly sample images
    sampled_files = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"Visualizing {len(sampled_files)} random samples from the '{split}' set...")
    print("Press 'n' for next image, 'q' to quit.")

    for img_filename in sampled_files:
        img_path = os.path.join(image_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_filename}. Skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_filename}. Skipping.")
            continue

        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label_line in labels:
            parts = label_line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Malformed label line in {label_filename}: {label_line.strip()}. Skipping.")
                continue

            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

            # Convert normalized coordinates to pixel coordinates
            x1 = int((x_center - bbox_width / 2) * w)
            y1 = int((y_center - bbox_height / 2) * h)
            x2 = int((x_center + bbox_width / 2) * w)
            y2 = int((y_center + bbox_height / 2) * h)

            class_name = NAMES.get(class_id, f"Unknown_{class_id}")
            color = COLORS.get(class_name, (100, 100, 100)) # Default color for unknown

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_THICKNESS)

            # Draw label text
            text = f"{class_name}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1) # Background for text
            cv2.putText(img, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)

        cv2.imshow(f"Annotations - {img_filename}", img)
        key = cv2.waitKey(0) & 0xFF # Wait indefinitely for a key press

        if key == ord('q'):
            break
        cv2.destroyWindow(f"Annotations - {img_filename}")

    cv2.destroyAllWindows()
    print("Annotation visualization complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations.")
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to visualize (train, val, or test).',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='Number of random samples to visualize.',
    )
    args = parser.parse_args()
    visualize_annotations(split=args.split, num_samples=args.num_samples)

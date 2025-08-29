
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, 'archive', 'data.yaml')

def analyze_ball_annotations():
    """
    Analyzes the bounding box sizes and aspect ratios for the 'ball' class
    in the dataset.
    """
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: data.yaml not found at {DATA_YAML_PATH}")
        return

    with open(DATA_YAML_PATH, 'r') as f:
        data_config = yaml.safe_load(f)

    # Assuming class ID for ball is 0 based on previous data.yaml inspection
    ball_class_id = 0 

    all_ball_widths = []
    all_ball_heights = []
    all_ball_areas = []
    all_ball_aspect_ratios = []

    splits = ['train', 'val'] # Focus on train and val sets

    print("Starting analysis of ball annotations...")

    for split in splits:
        label_dir = os.path.abspath(os.path.join(os.path.dirname(DATA_YAML_PATH), data_config[split].replace('images', 'labels')))
        
        if not os.path.exists(label_dir):
            print(f"Warning: Label directory not found for {split}: {label_dir}. Skipping.")
            continue

        print(f"Processing {split} labels from: {label_dir}")
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                file_path = os.path.join(label_dir, label_file)
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5 and int(parts[0]) == ball_class_id:
                            # Normalized coordinates
                            _, x_center, y_center, width, height = map(float, parts)
                            all_ball_widths.append(width)
                            all_ball_heights.append(height)
                            all_ball_areas.append(width * height)
                            if height > 0: # Avoid division by zero
                                all_ball_aspect_ratios.append(width / height)

    if not all_ball_widths:
        print("No ball annotations found in the specified splits.")
        return

    print("\n--- Ball Annotation Statistics (Normalized) ---")
    print(f"Total ball annotations found: {len(all_ball_widths)}")
    print(f"Average Width: {np.mean(all_ball_widths):.4f} (Min: {np.min(all_ball_widths):.4f}, Max: {np.max(all_ball_widths):.4f})")
    print(f"Average Height: {np.mean(all_ball_heights):.4f} (Min: {np.min(all_ball_heights):.4f}, Max: {np.max(all_ball_heights):.4f})")
    print(f"Average Area: {np.mean(all_ball_areas):.4f} (Min: {np.min(all_ball_areas):.4f}, Max: {np.max(all_ball_areas):.4f})")
    print(f"Average Aspect Ratio (Width/Height): {np.mean(all_ball_aspect_ratios):.4f} (Min: {np.min(all_ball_aspect_ratios):.4f}, Max: {np.max(all_ball_aspect_ratios):.4f})")

    # Plotting distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_ball_areas, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Ball Annotation Areas')
    plt.xlabel('Normalized Area (width * height)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(all_ball_aspect_ratios, bins=50, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Ball Annotation Aspect Ratios')
    plt.xlabel('Aspect Ratio (Width / Height)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyze_ball_annotations()

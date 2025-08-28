from ultralytics import YOLO
import torch
import os
import argparse

def benchmark_model(model_path):
    """
    Benchmarks a specified YOLOv8 model on the Roboflow dataset.

    Args:
        model_path (str): The path to the .pt model file to evaluate.
    """
    # --- Device Selection ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Model Loading ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # --- Dataset Configuration ---
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(project_dir, 'archive', 'data.yaml')

    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        print("Please ensure the dataset is in the 'archive' directory.")
        return

    # --- Run Validation ---
    print(f"Starting validation for model {os.path.basename(model_path)}...")
    metrics = model.val(
        data=data_yaml_path,
        imgsz=416, # Use the same image size as training for a fair comparison
        device=device,
        split='valid' # Explicitly use the validation set
    )
    
    # --- Print Results ---
    print("\n--- Benchmark Results ---")
    print(f"Results for model: {os.path.basename(model_path)}")
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map}")
    print(f"mAP50 (Strict): {metrics.box.map50}")
    print(f"mAP75 (Stricter): {metrics.box.map75}")
    print("-------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark a YOLOv8 model.")
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov8l.pt',
        help='Path to the model .pt file to benchmark.'
    )
    args = parser.parse_args()

    benchmark_model(args.model)
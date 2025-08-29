from ultralytics import YOLO
import torch
import os

def train_model():
    """
    Fine-tunes the YOLOv8-large model on the custom football dataset with optimizations.
    """
    # --- Device Selection ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Model Selection ---
    # Sticking with the large model as requested.
    model_name = 'yolov8l.pt'
    print(f"Using model: {model_name}")
    model = YOLO(model_name)

    # --- Dataset Configuration ---
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(project_dir, 'archive', 'data.yaml')

    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        return

    # --- Start Optimized Training ---
    print("Starting OPTIMIZED model training...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        # --- OPTIMIZATIONS ---
        device=device,          # Explicitly set the device for training
        imgsz=416,              # Optimized image size for balanced performance
        batch=8,                # A safe batch size for a large model on 16GB VRAM
        workers=8,              # Number of threads for data loading
        amp=True,               # Enable Automatic Mixed Precision
        # --- END OPTIMIZATIONS ---
        scale=0.75,             # Scale augmentation (0.5-1.5 default, 0.75 means 75% to 150%)
        degrees=15,             # Degrees for rotation augmentation
        translate=0.2,          # Translate augmentation
        project='runs/train',
        name='football_v3_large_fast',
        exist_ok=True
    )

    print("\n--- Training Complete ---")
    print(f"The new model weights are saved in the 'runs/train/football_v3_large_fast/weights' directory.")
    print("The best performing model is named 'best.pt'.")

if __name__ == '__main__':
    train_model()
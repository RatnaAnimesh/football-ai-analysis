import cv2
import os
import pandas as pd
from ultralytics import YOLO
import torch
from tqdm import tqdm
import argparse

def process_video_in_batches(video_path, model, output_csv_path, batch_size=32):
    """
    Processes a video file in batches for faster inference, tracks objects,
    and saves the tracking data to a CSV file.

    Args:
        video_path (str): The absolute path to the video file.
        model (YOLO): The YOLO object detection model.
        output_csv_path (str): The path to save the output CSV file.
        batch_size (int): The number of frames to process in a single batch.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video stream or file at {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Note: The custom model is trained on the new classes. 
    # We no longer need to filter for COCO classes [0, 32].
    # The model will now detect ['ball', 'goalkeeper', 'player', 'referee']
    # We will track all of them.
    print(f"Total frames in video: {total_frames}")

    all_tracking_data = []
    frame_buffer = []
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing Video Batches") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(frame)
            frame_count += 1

            if len(frame_buffer) == batch_size:
                # Process the batch
                results = model.track(frame_buffer, persist=True, verbose=False)
                
                # Extract and store data for the batch
                for i, res in enumerate(results):
                    original_frame_number = frame_count - len(frame_buffer) + i
                    if res.boxes.id is not None:
                        track_ids = res.boxes.id.int().cpu().tolist()
                        boxes = res.boxes.xyxy.cpu().tolist()
                        class_ids = res.boxes.cls.int().cpu().tolist()
                        class_names = [model.names[cls_id] for cls_id in class_ids]

                        for track_id, box, cls_id, cls_name in zip(track_ids, boxes, class_ids, class_names):
                            x1, y1, x2, y2 = box
                            all_tracking_data.append({
                                "frame": original_frame_number,
                                "track_id": track_id,
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            })
                
                pbar.update(len(frame_buffer))
                frame_buffer = []

    # Process any remaining frames
    if frame_buffer:
        results = model.track(frame_buffer, persist=True, verbose=False)
        for i, res in enumerate(results):
            original_frame_number = frame_count - len(frame_buffer) + i
            if res.boxes.id is not None:
                track_ids = res.boxes.id.int().cpu().tolist()
                boxes = res.boxes.xyxy.cpu().tolist()
                class_ids = res.boxes.cls.int().cpu().tolist()
                class_names = [model.names[cls_id] for cls_id in class_ids]

                for track_id, box, cls_id, cls_name in zip(track_ids, boxes, class_ids, class_names):
                    x1, y1, x2, y2 = box
                    all_tracking_data.append({
                        "frame": original_frame_number,
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
        pbar.update(len(frame_buffer))

    cap.release()
    pbar.close()
    print(f"\nFinished processing.")

    if all_tracking_data:
        df = pd.DataFrame(all_tracking_data)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved tracking data to {output_csv_path}")
    else:
        print("No tracking data was generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video to generate tracking data.")
    parser.add_argument(
        '--model', 
        type=str,
        default='runs/train/football_v3_large_fast/weights/best.pt',
        help='Path to the YOLOv8 model .pt file. Defaults to the best model from the last training run.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second of the input video. Used for velocity calculations.'
    )
    parser.add_argument(
        '--video',
        type=str,
        default='videos/match.mp4',
        help='Path to the input video file.'
    )
    args = parser.parse_args()

    # --- Device Selection ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Model Loading ---
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    print(f"Loading model: {args.model}")
    model = YOLO(args.model).to(device)

    # --- File Paths ---
    project_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(project_dir, args.video)
    
    output_csv = "tracking_data.csv"
    output_csv_path = os.path.join(project_dir, output_csv)

    metadata_path = os.path.join(project_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({'fps': args.fps}, f)
    print(f"Saved video metadata to {metadata_path}")

    process_video_in_batches(video_path, model, output_csv_path)
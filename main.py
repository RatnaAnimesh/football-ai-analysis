import cv2
import os
import pandas as pd
from ultralytics import YOLO
import torch
from tqdm import tqdm

def process_video_in_batches(video_path, model, output_csv_path, batch_size=64):
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
    frame_skip = 6 # Process every 6th frame
    processed_total_frames = total_frames // frame_skip
    print(f"Total frames in video: {total_frames}")
    print(f"Processing every {frame_skip}th frame. Estimated frames to process: {processed_total_frames}")

    all_tracking_data = []
    frame_buffer = []
    processed_frame_count = 0

    with tqdm(total=processed_total_frames, desc="Processing Batches") as pbar:
        current_video_frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_video_frame_index % frame_skip == 0:
                frame_buffer.append(frame)
                processed_frame_count += 1

                if len(frame_buffer) == batch_size:
                    # Process the batch
                    results = model.track(frame_buffer, persist=True, classes=[0, 32], verbose=False)
                    
                    # Extract and store data for the batch
                    for i, res in enumerate(results):
                        # Calculate the original frame number in the video
                        original_frame_number = (processed_frame_count - len(frame_buffer) + i) * frame_skip
                        if res.boxes.id is not None:
                            track_ids = res.boxes.id.int().cpu().tolist()
                            boxes = res.boxes.xyxy.cpu().tolist()
                            class_ids = res.boxes.cls.int().cpu().tolist()

                            for track_id, box, cls_id in zip(track_ids, boxes, class_ids):
                                x1, y1, x2, y2 = box
                                all_tracking_data.append({
                                    "frame": original_frame_number, # Store original frame number
                                    "track_id": track_id,
                                    "class_id": cls_id,
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                                })
                    
                    # Reset buffer and update progress bar
                    pbar.update(len(frame_buffer))
                    frame_buffer = []
            
            current_video_frame_index += 1

    # Process any remaining frames in the buffer
    if frame_buffer:
        results = model.track(frame_buffer, persist=True, classes=[0, 32], verbose=False)
        for i, res in enumerate(results):
            original_frame_number = (processed_frame_count - len(frame_buffer) + i) * frame_skip
            if res.boxes.id is not None:
                track_ids = res.boxes.id.int().cpu().tolist()
                boxes = res.boxes.xyxy.cpu().tolist()
                class_ids = res.boxes.cls.int().cpu().tolist()

                for track_id, box, cls_id in zip(track_ids, boxes, class_ids):
                    x1, y1, x2, y2 = box
                    all_tracking_data.append({
                        "frame": original_frame_number,
                        "track_id": track_id,
                        "class_id": cls_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
        pbar.update(len(frame_buffer))

    cap.release()
    pbar.close()
    print(f"\nFinished processing.")

    # Convert to DataFrame and save to CSV
    if all_tracking_data:
        df = pd.DataFrame(all_tracking_data)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved tracking data to {output_csv_path}")
    else:
        print("No tracking data was generated.")

if __name__ == '__main__':
    # --- Device Selection ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Model Loading ---
    model = YOLO('yolov8l.pt').to(device)

    # --- File Paths ---
    video_file = "videos/match.mp4"
    project_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(project_dir, video_file)
    
    output_csv = "tracking_data.csv"
    output_csv_path = os.path.join(project_dir, output_csv)

    process_video_in_batches(video_path, model, output_csv_path)

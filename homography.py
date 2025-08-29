import cv2
import numpy as np
import json
import os
import yaml

# Load configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- 1. REAL WORLD PITCH COORDINATES (in meters) ---
PITCH_LENGTH = config['PITCH_LENGTH_M']
PITCH_WIDTH = config['PITCH_WIDTH_M']

PENALTY_BOX_LENGTH = 16.5 # These are standard FIFA dimensions, not in config yet
PENALTY_BOX_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32
CENTER_CIRCLE_RADIUS = 9.15

# Origin (0,0) is the top-left corner
REAL_WORLD_POINTS = {
    "top_left_corner": (0, 0),
    "top_right_corner": (PITCH_LENGTH, 0),
    "bottom_left_corner": (0, PITCH_WIDTH),
    "bottom_right_corner": (PITCH_LENGTH, PITCH_WIDTH),
    
    "top_left_penalty_box": ((PITCH_LENGTH - PENALTY_BOX_LENGTH), (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
    "top_right_penalty_box": (PITCH_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
    "bottom_left_penalty_box": ((PITCH_LENGTH - PENALTY_BOX_LENGTH), (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
    "bottom_right_penalty_box": (PITCH_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),

    "left_top_penalty_box": (PENALTY_BOX_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
    "left_bottom_penalty_box": (PENALTY_BOX_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),

    "halfway_line_top": (PITCH_LENGTH / 2, 0),
    "halfway_line_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH),
    "center_spot": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
}

# Global list to store the points
clicked_points = []

def select_points_on_frame(event, x, y, flags, params):
    """Mouse callback function to capture clicks."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Point {len(clicked_points)} selected: ({x}, {y})")
            cv2.circle(params['image'], (x, y), 7, (0, 255, 0), -1)
            cv2.putText(params['image'], str(len(clicked_points)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Select 4 Known Pitch Points", params['image'])
        else:
            print("Already selected 4 points. Press 's' to save or 'r' to reset.")

def get_homography_points(video_path):
    """
    Opens the first frame of a video, allows the user to select 4 known points,
    identify them, and then calculates and saves the homography matrix.
    """
    global clicked_points
    clicked_points = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        cap.release()
        return
    cap.release()

    cv2.namedWindow("Select 4 Known Pitch Points")
    cv2.setMouseCallback("Select 4 Known Pitch Points", select_points_on_frame, {"image": frame})

    print("--- Pitch Point Selection ---")
    print("Please select 4 clearly identifiable points on the pitch.")
    print("Good examples: corners of penalty boxes, center spot, line intersections.")
    print("\nCONTROLS:")
    print("  - Click to select a point (up to 4).")
    print("  - Press 's' to save the points once 4 have been selected.")
    print("  - Press 'r' to reset the points.")
    print("  - Press 'q' to quit.")

    clone = frame.copy()
    cv2.imshow("Select 4 Known Pitch Points", frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked_points = []
            frame = clone.copy()
            cv2.imshow("Select 4 Known Pitch Points", frame)
            print("\nPoints reset.")
        elif key == ord('s'):
            if len(clicked_points) == 4:
                break
            else:
                print("\nPlease select exactly 4 points before saving.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # --- 2. IDENTIFY THE SELECTED POINTS ---
    print("\n--- Point Identification ---")
    print("Please identify the points you selected in the order you clicked them.")
    
    available_points = list(REAL_WORLD_POINTS.keys())
    selected_real_world_pts = []
    
    for i in range(4):
        print(f"\nFor Point {i+1} {clicked_points[i]}:")
        for j, name in enumerate(available_points):
            print(f"  {j+1}: {name}")
        
        while True:
            try:
                choice = int(input(f"Enter the number for the corresponding real-world point: "))
                if 1 <= choice <= len(available_points):
                    selected_key = available_points.pop(choice - 1)
                    selected_real_world_pts.append(REAL_WORLD_POINTS[selected_key])
                    break
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # --- 3. CALCULATE AND SAVE HOMOGRAPHY ---
    src_pts = np.array(clicked_points, dtype=np.float32)
    dst_pts = np.array(selected_real_world_pts, dtype=np.float32)

    homography_matrix, status = cv2.findHomography(src_pts, dst_pts)

    if homography_matrix is not None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(video_path)), 'homography_matrix.json')
        with open(output_path, 'w') as f:
            json.dump({'matrix': homography_matrix.tolist()}, f, indent=4)
        print(f"\nSuccessfully calculated and saved homography matrix to {output_path}")
    else:
        print("\nError: Could not compute homography matrix. Please check your points and try again.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate a homography matrix for a video.")
    parser.add_argument(
        '--video',
        type=str,
        default='videos/match.mp4',
        help='Path to the input video file.'
    )
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(project_dir, args.video)

    if not os.path.exists(video_path):
        # Check if the videos directory exists, if not create it
        if not os.path.exists(os.path.join(project_dir, 'videos')):
            os.makedirs(os.path.join(project_dir, 'videos'))
            print("Created 'videos' directory. Please add your match footage there.")
        else:
            print(f"Error: Video file not found at {video_path}")
            print("Please ensure your video file is in the 'videos' directory.")
    else:
        get_homography_points(video_path)
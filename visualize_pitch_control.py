
import pandas as pd
import numpy as np
import cv2
import os
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from tqdm import tqdm

# --- CONFIGURATION ---
PITCH_LENGTH_PX = 1050
PITCH_WIDTH_PX = 680
TEAM_COLORS = {0: (255, 100, 100), 1: (100, 100, 255)} # BGR format for OpenCV
BALL_COLOR = (255, 255, 0)
POSSESSION_HIGHLIGHT_COLOR = (0, 255, 255)
LBP_COLOR = (0, 255, 0)

# --- DRAWING UTILITIES ---

def draw_pitch(image):
    """Draws a football pitch on an image."""
    # Pitch boundaries
    cv2.rectangle(image, (0, 0), (PITCH_LENGTH_PX, PITCH_WIDTH_PX), (0, 0, 0), 2)
    # Center line
    cv2.line(image, (PITCH_LENGTH_PX // 2, 0), (PITCH_LENGTH_PX // 2, PITCH_WIDTH_PX), (0, 0, 0), 2)
    # Center circle
    cv2.circle(image, (PITCH_LENGTH_PX // 2, PITCH_WIDTH_PX // 2), 70, (0, 0, 0), 2)
    return image

# --- MAIN VISUALIZATION FUNCTION ---

def generate_tactical_video(output_video_path, fps=10):
    """Generates a tactical video with analytics overlays."""
    # --- Load Data ---
    print("Loading analytics data...")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df_tracking = pd.read_csv(os.path.join(project_dir, 'tracking_data.csv'))
        df_possession = pd.read_csv(os.path.join(project_dir, 'possession_events.csv'))
        df_lbp = pd.read_csv(os.path.join(project_dir, 'line_breaking_passes.csv'))
        df_pressure = pd.read_csv(os.path.join(project_dir, 'pressure_metrics.csv'))
        team_mapping = pd.read_csv(os.path.join(project_dir, 'team_mapping.csv')).set_index('track_id')['team_id'].to_dict()
    except FileNotFoundError as e:
        print(f"Error: Missing analytics file. {e}")
        print("Please run the full analysis.py script first.")
        return

    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (PITCH_LENGTH_PX, PITCH_WIDTH_PX))

    # --- Process Frames ---
    for frame_num in tqdm(sorted(df_tracking['frame'].unique()), desc="Generating Video"):
        # Create a fresh pitch image for each frame
        pitch_image = np.full((PITCH_WIDTH_PX, PITCH_LENGTH_PX, 3), (60, 179, 113), dtype=np.uint8) # Green pitch
        pitch_image = draw_pitch(pitch_image)
        overlay = pitch_image.copy()

        df_frame = df_tracking[df_tracking['frame'] == frame_num]
        players_in_frame = df_frame[df_frame['class_name'].isin(['player', 'goalkeeper'])]
        ball_in_frame = df_frame[df_frame['class_name'] == 'ball']

        # --- 1. Pitch Control Overlay ---
        points = players_in_frame[['center_x', 'center_y']].values
        if len(points) >= 4:
            try:
                vor = Voronoi(points)
                pitch_polygon = Polygon([(0, 0), (PITCH_LENGTH_PX, 0), (PITCH_LENGTH_PX, PITCH_WIDTH_PX), (0, PITCH_WIDTH_PX)])
                for i, region_idx in enumerate(vor.point_region):
                    if region_idx != -1 and -1 not in vor.regions[region_idx]:
                        region_poly = Polygon([vor.vertices[v] for v in vor.regions[region_idx]])
                        if region_poly.is_valid:
                            clipped_region = pitch_polygon.intersection(region_poly)
                            track_id = players_in_frame.iloc[i]['track_id']
                            team_id = team_mapping.get(track_id)
                            if team_id is not None:
                                pts = np.array(clipped_region.exterior.coords, dtype=np.int32)
                                cv2.fillPoly(overlay, [pts], TEAM_COLORS[team_id], lineType=cv2.LINE_AA)
            except Exception:
                pass # Ignore frames where Voronoi fails
        
        # Blend overlay with pitch
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, pitch_image, 1 - alpha, 0, pitch_image)

        # --- 2. Player and Ball Positions ---
        for _, player in players_in_frame.iterrows():
            pos = (int(player['center_x']), int(player['center_y']))
            team_id = team_mapping.get(player['track_id'])
            if team_id is not None:
                cv2.circle(pitch_image, pos, 10, TEAM_COLORS[team_id], -1)
                cv2.putText(pitch_image, str(int(player['track_id'])), (pos[0]+5, pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        if not ball_in_frame.empty:
            pos = (int(ball_in_frame.iloc[0]['center_x']), int(ball_in_frame.iloc[0]['center_y']))
            cv2.circle(pitch_image, pos, 6, BALL_COLOR, -1)

        # --- 3. Analytics Overlays ---
        possession_info = df_possession[df_possession['frame'] == frame_num]
        if not possession_info.empty and pd.notna(possession_info.iloc[0]['possessor_id']):
            possessor_id = possession_info.iloc[0]['possessor_id']
            possessor_data = df_frame[df_frame['track_id'] == possessor_id]
            if not possessor_data.empty:
                pos = (int(possessor_data.iloc[0]['center_x']), int(possessor_data.iloc[0]['center_y']))
                cv2.circle(pitch_image, pos, 12, POSSESSION_HIGHLIGHT_COLOR, 3) # Highlight possessor

                pressure_info = df_pressure[df_pressure['frame'] == frame_num]
                if not pressure_info.empty:
                    num_press = pressure_info.iloc[0]['num_pressuring_opponents']
                    cv2.putText(pitch_image, f"P:{int(num_press)}", (pos[0]+15, pos[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        lbp_info = df_lbp[df_lbp['frame'] == frame_num]
        if not lbp_info.empty and lbp_info.iloc[0]['is_line_breaking']:
            passer_id = lbp_info.iloc[0]['passer_id']
            receiver_id = lbp_info.iloc[0]['receiver_id']
            passer_pos = df_frame[df_frame['track_id'] == passer_id][['center_x', 'center_y']].iloc[0]
            receiver_pos = df_frame[df_frame['track_id'] == receiver_id][['center_x', 'center_y']].iloc[0]
            cv2.arrowedLine(pitch_image, (int(passer_pos[0]), int(passer_pos[1])), (int(receiver_pos[0]), int(receiver_pos[1])), LBP_COLOR, 2, tipLength=0.05)

        writer.write(pitch_image)

    writer.release()
    print(f"Tactical video saved to {output_video_path}")

if __name__ == '__main__':
    output_video = "tactical_video.mp4"
    generate_tactical_video(output_video)

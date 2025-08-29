
import pandas as pd
import numpy as np
import cv2
import os
import json
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from tqdm import tqdm

# --- CONFIGURATION ---
PITCH_LENGTH_M = 105  # Standard FIFA pitch length in meters
PITCH_WIDTH_M = 68    # Standard FIFA pitch width in meters

# Scale for converting meters to display pixels
# Assuming a display resolution where 1 meter = 10 pixels for a 1050x680 pitch representation
METER_TO_PIXEL_SCALE = 10
DISPLAY_WIDTH = int(PITCH_LENGTH_M * METER_TO_PIXEL_SCALE)
DISPLAY_HEIGHT = int(PITCH_WIDTH_M * METER_TO_PIXEL_SCALE)

TEAM_COLORS = {0: (255, 100, 100), 1: (100, 100, 255)} # BGR format for OpenCV
BALL_COLOR = (0, 255, 255) # Yellow
POSSESSION_HIGHLIGHT_COLOR = (0, 255, 255)
LBP_COLOR = (0, 255, 0)

# --- UTILITY FUNCTIONS ---
def meter_to_pixel(x_m, y_m):
    """Converts real-world meter coordinates to display pixel coordinates."""
    x_px = int(x_m * METER_TO_PIXEL_SCALE)
    y_px = int(y_m * METER_TO_PIXEL_SCALE)
    return (x_px, y_px)

def draw_pitch(image):
    """Draws a football pitch on an image using meter-based dimensions."""
    # Pitch boundaries
    cv2.rectangle(image, (0, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0), 2)
    # Center line
    cv2.line(image, meter_to_pixel(PITCH_LENGTH_M / 2, 0), meter_to_pixel(PITCH_LENGTH_M / 2, PITCH_WIDTH_M), (0, 0, 0), 2)
    # Center circle (radius 9.15m)
    cv2.circle(image, meter_to_pixel(PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2), int(9.15 * METER_TO_PIXEL_SCALE), (0, 0, 0), 2)
    
    # Penalty boxes (example for one side, assuming right side goal)
    # Top-right penalty box corner
    pb_top_right_x = PITCH_LENGTH_M - 16.5 # 16.5m from goal line
    pb_top_right_y = (PITCH_WIDTH_M / 2) - (40.32 / 2) # 40.32m width
    pb_bottom_right_x = PITCH_LENGTH_M - 16.5
    pb_bottom_right_y = (PITCH_WIDTH_M / 2) + (40.32 / 2)
    cv2.rectangle(image, meter_to_pixel(pb_top_right_x, pb_top_right_y), meter_to_pixel(PITCH_LENGTH_M, pb_bottom_right_y), (0, 0, 0), 2)

    # Goal area (6-yard box)
    ga_top_right_x = PITCH_LENGTH_M - 5.5 # 5.5m from goal line
    ga_top_right_y = (PITCH_WIDTH_M / 2) - (18.32 / 2) # 18.32m width
    ga_bottom_right_x = PITCH_LENGTH_M - 5.5
    ga_bottom_right_y = (PITCH_WIDTH_M / 2) + (18.32 / 2)
    cv2.rectangle(image, meter_to_pixel(ga_top_right_x, ga_top_right_y), meter_to_pixel(PITCH_LENGTH_M, ga_bottom_right_y), (0, 0, 0), 2)

    # Penalty spot (11m from goal line)
    cv2.circle(image, meter_to_pixel(PITCH_LENGTH_M - 11, PITCH_WIDTH_M / 2), 3, (0, 0, 0), -1)

    # Repeat for left side (mirroring)
    # Penalty box
    pb_left_top_x = 16.5
    pb_left_top_y = (PITCH_WIDTH_M / 2) - (40.32 / 2)
    pb_left_bottom_x = 16.5
    pb_left_bottom_y = (PITCH_WIDTH_M / 2) + (40.32 / 2)
    cv2.rectangle(image, meter_to_pixel(0, pb_left_top_y), meter_to_pixel(pb_left_top_x, pb_left_bottom_y), (0, 0, 0), 2)

    # Goal area
    ga_left_top_x = 5.5
    ga_left_top_y = (PITCH_WIDTH_M / 2) - (18.32 / 2)
    ga_left_bottom_x = 5.5
    ga_left_bottom_y = (PITCH_WIDTH_M / 2) + (18.32 / 2)
    cv2.rectangle(image, meter_to_pixel(0, ga_left_top_y), meter_to_pixel(ga_left_top_x, ga_left_bottom_y), (0, 0, 0), 2)

    # Penalty spot
    cv2.circle(image, meter_to_pixel(11, PITCH_WIDTH_M / 2), 3, (0, 0, 0), -1)

    return image

# --- MAIN VISUALIZATION FUNCTION ---

def generate_tactical_video(output_video_path):
    """Generates a tactical video with analytics overlays using meter coordinates."""
    # --- Load Data ---
    print("Loading analytics data...")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df_tracking = pd.read_csv(os.path.join(project_dir, 'tracking_data_enriched.csv')) # Use enriched data
        df_possession = pd.read_csv(os.path.join(project_dir, 'possession_events.csv'))
        df_lbp = pd.read_csv(os.path.join(project_dir, 'line_breaking_passes.csv'))
        df_pressure = pd.read_csv(os.path.join(project_dir, 'pressure_metrics.csv'))
        team_mapping = pd.read_csv(os.path.join(project_dir, 'team_mapping.csv')).set_index('track_id')['team_id'].to_dict()
        
        with open(os.path.join(project_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            fps = metadata.get('fps', 30) # Get FPS from metadata

    except FileNotFoundError as e:
        print(f"Error: Missing analytics file. {e}")
        print("Please run the full analysis.py script first.")
        return

    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # --- Process Frames ---
    for frame_num in tqdm(sorted(df_tracking['frame'].unique()), desc="Generating Video"):
        # Create a fresh pitch image for each frame
        pitch_image = np.full((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), (60, 179, 113), dtype=np.uint8) # Green pitch
        pitch_image = draw_pitch(pitch_image)
        overlay = pitch_image.copy()

        df_frame = df_tracking[df_tracking['frame'] == frame_num]
        players_in_frame = df_frame[df_frame['class_name'].isin(['player', 'goalkeeper'])]
        ball_in_frame = df_frame[df_frame['class_name'] == 'ball']

        # --- 1. Pitch Control Overlay (using meter coordinates) ---
        points_m = players_in_frame[['center_x_m', 'center_y_m']].values
        if len(points_m) >= 4:
            try:
                vor = Voronoi(points_m)
                pitch_polygon_m = Polygon([(0, 0), (PITCH_LENGTH_M, 0), (PITCH_LENGTH_M, PITCH_WIDTH_M), (0, PITCH_WIDTH_M)])
                for i, region_idx in enumerate(vor.point_region):
                    if region_idx != -1 and -1 not in vor.regions[region_idx]:
                        region_poly_m = Polygon([vor.vertices[v] for v in vor.regions[region_idx]])
                        if region_poly_m.is_valid:
                            clipped_region_m = pitch_polygon_m.intersection(region_poly_m)
                            track_id = players_in_frame.iloc[i]['track_id']
                            team_id = team_mapping.get(track_id)
                            if team_id is not None:
                                # Convert meter coordinates of polygon to display pixels
                                pts_px = np.array([meter_to_pixel(p[0], p[1]) for p in clipped_region_m.exterior.coords], dtype=np.int32)
                                cv2.fillPoly(overlay, [pts_px], TEAM_COLORS[team_id], lineType=cv2.LINE_AA)
            except Exception:
                pass # Ignore frames where Voronoi fails
        
        # Blend overlay with pitch
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, pitch_image, 1 - alpha, 0, pitch_image)

        # --- 2. Player and Ball Positions (using meter coordinates) ---
        for _, player in players_in_frame.iterrows():
            pos_m = (player['center_x_m'], player['center_y_m'])
            pos_px = meter_to_pixel(pos_m[0], pos_m[1])
            team_id = team_mapping.get(player['track_id'])
            if team_id is not None:
                cv2.circle(pitch_image, pos_px, int(0.8 * METER_TO_PIXEL_SCALE), TEAM_COLORS[team_id], -1) # Player size ~0.8m
                cv2.putText(pitch_image, str(int(player['track_id'])), (pos_px[0]+int(0.5*METER_TO_PIXEL_SCALE), pos_px[1]-int(0.5*METER_TO_PIXEL_SCALE)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        if not ball_in_frame.empty:
            ball_pos_m = (ball_in_frame.iloc[0]['center_x_m'], ball_in_frame.iloc[0]['center_y_m'])
            ball_pos_px = meter_to_pixel(ball_pos_m[0], ball_pos_m[1])
            cv2.circle(pitch_image, ball_pos_px, int(0.2 * METER_TO_PIXEL_SCALE), BALL_COLOR, -1) # Ball size ~0.2m

        # --- 3. Analytics Overlays (using meter coordinates) ---
        possession_info = df_possession[df_possession['frame'] == frame_num]
        if not possession_info.empty and pd.notna(possession_info.iloc[0]['possessor_id']):
            possessor_id = possession_info.iloc[0]['possessor_id']
            possessor_data = df_frame[df_frame['track_id'] == possessor_id]
            if not possessor_data.empty:
                possessor_pos_m = (possessor_data.iloc[0]['center_x_m'], possessor_data.iloc[0]['center_y_m'])
                possessor_pos_px = meter_to_pixel(possessor_pos_m[0], possessor_pos_m[1])
                cv2.circle(pitch_image, possessor_pos_px, int(1.2 * METER_TO_PIXEL_SCALE), POSSESSION_HIGHLIGHT_COLOR, 3) # Highlight possessor

                pressure_info = df_pressure[df_pressure['frame'] == frame_num]
                if not pressure_info.empty:
                    num_press = pressure_info.iloc[0]['num_pressuring_opponents']
                    avg_speed = pressure_info.iloc[0]['avg_closing_speed_mps']
                    cv2.putText(pitch_image, f"P:{int(num_press)} S:{avg_speed:.1f}", (possessor_pos_px[0]+int(1.5*METER_TO_PIXEL_SCALE), possessor_pos_px[1]+int(1.5*METER_TO_PIXEL_SCALE)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        lbp_info = df_lbp[df_lbp['frame'] == frame_num]
        if not lbp_info.empty and lbp_info.iloc[0]['is_line_breaking']:
            passer_id = lbp_info.iloc[0]['passer_id']
            receiver_id = lbp_info.iloc[0]['receiver_id']
            passer_data = df_frame[df_frame['track_id'] == passer_id]
            receiver_data = df_frame[df_frame['track_id'] == receiver_id]
            if not passer_data.empty and not receiver_data.empty:
                passer_pos_m = (passer_data.iloc[0]['center_x_m'], passer_data.iloc[0]['center_y_m'])
                receiver_pos_m = (receiver_data.iloc[0]['center_x_m'], receiver_data.iloc[0]['center_y_m'])
                
                passer_pos_px = meter_to_pixel(passer_pos_m[0], passer_pos_m[1])
                receiver_pos_px = meter_to_pixel(receiver_pos_m[0], receiver_pos_m[1])
                
                cv2.arrowedLine(pitch_image, passer_pos_px, receiver_pos_px, LBP_COLOR, 2, tipLength=0.05)

        writer.write(pitch_image)

    writer.release()
    print(f"Tactical video saved to {output_video_path}")

if __name__ == '__main__':
    output_video = "tactical_video.mp4"
    generate_tactical_video(output_video)

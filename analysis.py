
import pandas as pd
import numpy as np
import os
import json
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from tqdm import tqdm

# --- UTILITY FUNCTIONS ---

def transform_coordinates(df, homography_matrix):
    """Transforms pixel coordinates to real-world meters."""
    # Calculate center points in pixels first
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2

    # Apply the perspective transformation
    pixel_coords = np.float32(df[['center_x', 'center_y']].values).reshape(-1, 1, 2)
    real_world_coords = cv2.perspectiveTransform(pixel_coords, homography_matrix)

    # Add the new meter coordinates to the DataFrame
    df['center_x_m'] = real_world_coords[:, 0, 0]
    df['center_y_m'] = real_world_coords[:, 0, 1]
    return df

def calculate_velocities(df, fps=30):
    """Calculates velocities and speeds in m/s for each tracked object."""
    df = df.sort_values(by=['track_id', 'frame'])
    
    # Calculate time delta in seconds
    df['time_delta_s'] = df.groupby('track_id')[['frame']].diff().fillna(0) / fps
    
    # Calculate position delta in meters
    df['x_delta_m'] = df.groupby('track_id')[['center_x_m']].diff().fillna(0)
    df['y_delta_m'] = df.groupby('track_id')[['center_y_m']].diff().fillna(0)
    
    # Calculate velocity in m/s
    df['velocity_x_mps'] = df['x_delta_m'] / df['time_delta_s']
    df['velocity_y_mps'] = df['y_delta_m'] / df['time_delta_s']
    
    # Calculate speed in m/s
    df['speed_mps'] = np.sqrt(df['velocity_x_mps']**2 + df['velocity_y_mps']**2)
    
    df.fillna(0, inplace=True)
    return df

def assign_teams(df_frame):
    """Assigns players to two teams using KMeans clustering on their initial positions."""
    player_df = df_frame[df_frame['class_name'].isin(['player', 'goalkeeper'])].copy()
    if len(player_df) < 2: return {}
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    # Use pixel coordinates for clustering as it's about visual grouping
    player_df['team_id_cluster'] = kmeans.fit_predict(player_df[['center_x', 'center_y']])
    team_centers = player_df.groupby('team_id_cluster')[['center_x']].mean().sort_values('center_x')
    left_team_id = team_centers.index[0]
    team_mapping = {row['track_id']: 0 if row['team_id_cluster'] == left_team_id else 1 for _, row in player_df.iterrows()}
    return team_mapping

# --- CORE ANALYSIS MODULES ---

def analyze_possession(df, team_mapping):
    """Analyzes ball possession with improved logic."""
    ball_df = df[df['class_name'] == 'ball'].copy()
    player_df = df[df['class_name'].isin(['player', 'goalkeeper'])].copy()
    possession_events = []
    min_possession_frames = 3
    possession_threshold = 3 # Using meters now
    last_possessor_id = None
    potential_possessor_id = None
    possession_frame_count = 0

    for frame_num in tqdm(sorted(df['frame'].unique()), desc="Analyzing Possession"):
        ball_in_frame = ball_df[ball_df['frame'] == frame_num]
        players_in_frame = player_df[player_df['frame'] == frame_num]
        if ball_in_frame.empty or players_in_frame.empty: continue

        ball_pos = ball_in_frame[['center_x_m', 'center_y_m']].iloc[0]
        distances = np.sqrt(((players_in_frame[['center_x_m', 'center_y_m']] - ball_pos)**2).sum(axis=1))
        
        if distances.min() > possession_threshold:
            closest_player_id = None
        else:
            closest_player_id = players_in_frame.iloc[distances.idxmin()]['track_id']

        if closest_player_id == potential_possessor_id:
            possession_frame_count += 1
        else:
            potential_possessor_id = closest_player_id
            possession_frame_count = 1

        current_possessor_id = potential_possessor_id if possession_frame_count >= min_possession_frames else None
        
        possession_events.append({
            'frame': frame_num,
            'possessor_id': current_possessor_id,
            'team_in_possession': team_mapping.get(current_possessor_id)
        })
        last_possessor_id = current_possessor_id

    return pd.DataFrame(possession_events)

def calculate_pitch_control(df, team_mapping):
    """Calculates pitch control for each frame using Voronoi diagrams in meters."""
    pitch_control_metrics = []
    pitch_length = 105
    pitch_width = 68
    pitch_polygon = Polygon([(0, 0), (pitch_length, 0), (pitch_length, pitch_width), (0, pitch_width)])

    unique_frames = sorted(df['frame'].unique())

    for frame_num in tqdm(unique_frames, desc="Calculating Pitch Control"):
        df_frame = df[df['frame'] == frame_num].copy()
        players_in_frame = df_frame[df_frame['class_name'].isin(['player', 'goalkeeper'])]
        
        points = players_in_frame[['center_x_m', 'center_y_m']].values
        if len(points) < 4: continue

        try:
            vor = Voronoi(points)
        except Exception:
            continue

        team_areas = {0: 0, 1: 0}
        for i, region_idx in enumerate(vor.point_region):
            if region_idx == -1 or -1 in vor.regions[region_idx]: continue

            region_polygon = Polygon([vor.vertices[v] for v in vor.regions[region_idx]])
            if not region_polygon.is_valid: continue

            clipped_region = pitch_polygon.intersection(region_polygon)
            track_id = players_in_frame.iloc[i]['track_id']
            team_id = team_mapping.get(track_id)

            if team_id is not None:
                team_areas[team_id] += clipped_region.area

        pitch_control_metrics.append({
            'frame': frame_num,
            'team_0_control_area_sq_m': team_areas[0],
            'team_1_control_area_sq_m': team_areas[1]
        })

    return pd.DataFrame(pitch_control_metrics)

def calculate_pressure_metrics(df, possession_df):
    """Calculates pressure on the ball carrier in meters."""
    pressure_metrics = []
    pressure_radius = 10 # 10 meters

    df_with_possession = pd.merge(df, possession_df, on='frame', how='left')
    frames_with_possession = df_with_possession[df_with_possession['possessor_id'].notna()]

    for frame_num in tqdm(sorted(frames_with_possession['frame'].unique()), desc="Calculating Pressure"):
        df_frame = frames_with_possession[frames_with_possession['frame'] == frame_num]
        possessor_id = df_frame['possessor_id'].iloc[0]
        possessor_data = df_frame[df_frame['track_id'] == possessor_id]
        if possessor_data.empty: continue

        possessor_pos = possessor_data[['center_x_m', 'center_y_m']].iloc[0]
        possessor_team = possessor_data['team_id'].iloc[0]

        opponents = df_frame[(df_frame['team_id'] != possessor_team) & (df_frame['class_name'].isin(['player', 'goalkeeper']))]
        if opponents.empty: continue

        distances = np.sqrt(((opponents[['center_x_m', 'center_y_m']] - possessor_pos)**2).sum(axis=1))
        pressuring_opponents = opponents[distances <= pressure_radius]
        num_pressuring = len(pressuring_opponents)
        avg_closing_speed = 0

        if num_pressuring > 0:
            closing_speeds = []
            for _, opponent in pressuring_opponents.iterrows():
                vec_to_possessor = possessor_pos - opponent[['center_x_m', 'center_y_m']]
                norm_vec = vec_to_possessor / np.linalg.norm(vec_to_possessor)
                opponent_vel = opponent[['velocity_x_mps', 'velocity_y_mps']]
                closing_speed = np.dot(opponent_vel, norm_vec)
                if closing_speed > 0: closing_speeds.append(closing_speed)
            if closing_speeds: avg_closing_speed = np.mean(closing_speeds)

        pressure_metrics.append({
            'frame': frame_num,
            'possessor_id': possessor_id,
            'num_pressuring_opponents': num_pressuring,
            'avg_closing_speed_mps': avg_closing_speed
        })

    return pd.DataFrame(pressure_metrics)

def calculate_line_breaking_passes(df, possession_df):
    """Identifies line-breaking passes using meter coordinates."""
    passes = possession_df[possession_df['possessor_id'].notna()].copy()
    passes['pass_from'] = passes['possessor_id'].shift(1)
    passes = passes[passes['possessor_id'] != passes['pass_from']]
    passes.rename(columns={'possessor_id': 'pass_to'}, inplace=True)

    line_breaking_passes = []
    min_opponents_bypassed = 2

    for _, pass_event in tqdm(passes.iterrows(), total=passes.shape[0], desc="Finding Line-Breaking Passes"):
        frame = int(pass_event['frame'])
        passer_id = pass_event['pass_from']
        receiver_id = pass_event['pass_to']
        if pd.isna(passer_id) or pd.isna(receiver_id): continue

        df_frame = df[df['frame'] == frame]
        passer_data = df_frame[df_frame['track_id'] == passer_id]
        receiver_data = df_frame[df_frame['track_id'] == receiver_id]
        if passer_data.empty or receiver_data.empty: continue

        passer_pos = passer_data[['center_x_m', 'center_y_m']].iloc[0]
        receiver_pos = receiver_data[['center_x_m', 'center_y_m']].iloc[0]
        passer_team = passer_data['team_id'].iloc[0]

        opponents = df_frame[(df_frame['team_id'] != passer_team) & (df_frame['class_name'].isin(['player', 'goalkeeper']))]
        if opponents.empty: continue

        min_x, max_x = min(passer_pos['center_x_m'], receiver_pos['center_x_m']), max(passer_pos['center_x_m'], receiver_pos['center_x_m'])
        min_y, max_y = min(passer_pos['center_y_m'], receiver_pos['center_y_m']), max(passer_pos['center_y_m'], receiver_pos['center_y_m'])

        opponents_in_corridor = opponents[
            (opponents['center_x_m'] > min_x) & (opponents['center_x_m'] < max_x) &
            (opponents['center_y_m'] > min_y) & (opponents['center_y_m'] < max_y)
        ]
        num_bypassed = len(opponents_in_corridor)

        is_line_breaking = num_bypassed >= min_opponents_bypassed

        line_breaking_passes.append({
            'frame': frame,
            'passer_id': passer_id,
            'receiver_id': receiver_id,
            'is_line_breaking': is_line_breaking,
            'opponents_bypassed': num_bypassed
        })

    return pd.DataFrame(line_breaking_passes)

def calculate_xg(df, possession_df):
    """Calculates Expected Goals (xG) for shots using a heuristic model and meter coordinates."""
    pitch_length = 105
    goal_y_center = 68
    goal_width = 7.32
    goal_posts = [(pitch_length, goal_y_center - goal_width/2), (pitch_length, goal_y_center + goal_width/2)]

    shots = []
    ball_df = df[df['class_name'] == 'ball'].copy()
    ball_df['speed_change'] = ball_df.groupby('track_id')[['speed_mps']].diff().fillna(0)
    
    shot_speed_threshold = 10 # m/s
    potential_shots = ball_df[ball_df['speed_change'] > shot_speed_threshold]

    for _, shot_event in tqdm(potential_shots.iterrows(), total=potential_shots.shape[0], desc="Calculating xG"):
        frame = shot_event['frame']
        df_frame = df[df['frame'] == frame]
        
        shot_pos = shot_event[['center_x_m', 'center_y_m']]
        players_in_frame = df_frame[df_frame['class_name'].isin(['player', 'goalkeeper'])]
        if players_in_frame.empty: continue

        distances = np.sqrt(((players_in_frame[['center_x_m', 'center_y_m']] - shot_pos.values)**2).sum(axis=1))
        shooter_id = players_in_frame.iloc[distances.idxmin()]['track_id']
        shooter_team = players_in_frame.iloc[distances.idxmin()]['team_id']

        distance_to_goal = np.linalg.norm(shot_pos.values - np.array([pitch_length, goal_y_center]))
        
        v1 = goal_posts[0] - shot_pos.values
        v2 = goal_posts[1] - shot_pos.values
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle_degrees = np.degrees(angle)

        log_odds = 1.5 - (0.1 * distance_to_goal) + (0.05 * angle_degrees)
        xg_value = 1 / (1 + np.exp(-log_odds))

        shots.append({
            'frame': frame,
            'shooter_id': shooter_id,
            'team_id': shooter_team,
            'distance_to_goal_m': distance_to_goal,
            'angle_to_goal_deg': angle_degrees,
            'xg_value': xg_value
        })

    return pd.DataFrame(shots)

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(project_dir, "tracking_data.csv")
    homography_json = os.path.join(project_dir, "homography_matrix.json")
    metadata_json = os.path.join(project_dir, "metadata.json")
    output_dir = project_dir

    if not os.path.exists(input_csv) or not os.path.exists(homography_json) or not os.path.exists(metadata_json):
        print(f"Error: Required input files not found. Please ensure both 'tracking_data.csv', 'homography_matrix.json', and 'metadata.json' all exist.")
    else:
        print("Loading and preprocessing tracking data...")
        df = pd.read_csv(input_csv)
        
        with open(homography_json, 'r') as f:
            homography_matrix = np.array(json.load(f)['matrix'])

        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
            fps = metadata.get('fps', 30) # Default to 30 if not found

        print("Applying homography to transform coordinates...")
        df = transform_coordinates(df, homography_matrix)

        print(f"Calculating velocities in meters per second (using {fps} FPS)...")
        df = calculate_velocities(df, fps=fps)

        print("Assigning teams based on initial positions...")
        first_frame_df = df[df['frame'] == df['frame'].min()]
        team_mapping = assign_teams(first_frame_df)
        df['team_id'] = df['track_id'].map(team_mapping)
        pd.DataFrame(list(team_mapping.items()), columns=['track_id', 'team_id']).to_csv(os.path.join(output_dir, 'team_mapping.csv'), index=False)
        print("Team mapping complete and saved.")

        # --- Run all analysis modules with real-world coordinates ---
        possession_df = analyze_possession(df, team_mapping)
        pitch_control_df = calculate_pitch_control(df, team_mapping)
        pressure_df = calculate_pressure_metrics(df, possession_df)
        lbp_df = calculate_line_breaking_passes(df, possession_df)
        xg_df = calculate_xg(df, possession_df)

        print("Saving all analysis files...")
        # Save the main dataframe with meter coordinates for debugging and other uses
        df.to_csv(os.path.join(output_dir, 'tracking_data_enriched.csv'), index=False)
        
        possession_df.to_csv(os.path.join(output_dir, 'possession_events.csv'), index=False)
        pitch_control_df.to_csv(os.path.join(output_dir, 'pitch_control.csv'), index=False)
        pressure_df.to_csv(os.path.join(output_dir, 'pressure_metrics.csv'), index=False)
        lbp_df.to_csv(os.path.join(output_dir, 'line_breaking_passes.csv'), index=False)
        xg_df.to_csv(os.path.join(output_dir, 'xg_metrics.csv'), index=False)
        print("\nAll analysis complete.")

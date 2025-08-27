import pandas as pd
import numpy as np
import os
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from tqdm import tqdm

def calculate_pitch_control_metrics(input_csv_path, output_csv_path):
    """
    Reads tracking data, calculates player velocities and speeds, and saves to a new CSV.

    Args:
        input_csv_path (str): Path to the input tracking_data.csv file.
        output_csv_path (str): Path to save the enriched CSV file.
    """
    print(f"Reading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    # Define pitch boundaries as a Shapely Polygon (using the same dimensions as visualize_pitch_control.py)
    pitch_length = 1000
    pitch_width = 600
    pitch_polygon = Polygon([(0, 0), (pitch_length, 0), (pitch_length, pitch_width), (0, pitch_width)])

    all_pitch_control_metrics = []

    unique_frames = sorted(df['frame'].unique())

    print("Calculating pitch control metrics per frame...")
    for frame_num in tqdm(unique_frames):
        df_frame = df[df['frame'] == frame_num].copy()
        
        # Ensure there are enough points for Voronoi and they are distinct
        points = df_frame[['center_x', 'center_y']].values
        
        # Filter out rows with NaN in center_x or center_y before unique check
        points_clean = points[~np.isnan(points).any(axis=1)]

        if len(points_clean) < 4 or len(np.unique(points_clean, axis=0)) < 4:
            # Not enough distinct points for Voronoi, assign 0 control for this frame
            all_pitch_control_metrics.append({
                "frame": frame_num,
                "team_0_control_area": 0,
                "team_1_control_area": 0
            })
            continue

        try:
            vor = Voronoi(points_clean)
        except Exception as e:
            print(f"Warning: Voronoi calculation failed for frame {frame_num}. Error: {e}")
            all_pitch_control_metrics.append({
                "frame": frame_num,
                "team_0_control_area": 0,
                "team_1_control_area": 0
            })
            continue

        team_0_area = 0
        team_1_area = 0

        # Map track_id to team (based on parity, as in visualization)
        track_id_to_team = {row['track_id']: row['track_id'] % 2 for idx, row in df_frame.iterrows()}

        for i, region_index in enumerate(vor.point_region):
            if region_index == -1: # Unbounded region
                continue

            region_vertices = vor.regions[region_index]
            if not region_vertices or -1 in region_vertices: # Empty or unbounded region
                continue

            # Get the actual coordinates of the vertices
            polygon_points = [vor.vertices[v] for v in region_vertices]
            
            # Create a Shapely Polygon from the Voronoi region vertices
            # Handle cases where polygon_points might not form a valid polygon (e.g., less than 3 points)
            if len(polygon_points) < 3:
                continue
            
            try:
                voronoi_region_polygon = Polygon(polygon_points)
            except Exception as e:
                # print(f"Warning: Could not create Shapely Polygon for region {i} in frame {frame_num}. Error: {e}")
                continue

            # Intersect with pitch boundaries
            if voronoi_region_polygon.is_valid:
                intersection_polygon = pitch_polygon.intersection(voronoi_region_polygon)
                
                # Ensure the intersection is a valid polygon and calculate its area
                if intersection_polygon.is_valid and not intersection_polygon.is_empty:
                    # The intersection might result in MultiPolygon if it's disjoint
                    if intersection_polygon.geom_type == 'MultiPolygon':
                        area = sum(p.area for p in intersection_polygon.geoms)
                    else:
                        area = intersection_polygon.area
                    
                    # Assign area to the correct team
                    player_track_id = df_frame.iloc[i]['track_id']
                    team = track_id_to_team.get(player_track_id) # Use .get() for safety

                    if team == 0:
                        team_0_area += area
                    elif team == 1:
                        team_1_area += area
            else:
                # print(f"Warning: Invalid Voronoi region polygon for frame {frame_num}, region {i}.")
                pass

        all_pitch_control_metrics.append({
            "frame": frame_num,
            "team_0_control_area": team_0_area,
            "team_1_control_area": team_1_area
        })

    print(f"Saving pitch control metrics to {output_csv_path}...")
    metrics_df = pd.DataFrame(all_pitch_control_metrics)
    metrics_df.to_csv(output_csv_path, index=False)
    print("Pitch control quantification complete.")

def calculate_ball_events(input_csv_path, output_csv_path):
    print(f"Reading data from {input_csv_path} for ball events...")
    df = pd.read_csv(input_csv_path)

    ball_events = []
    
    # Initialize previous possessor and ball position
    prev_possessor_id = None
    prev_ball_pos = None

    unique_frames = sorted(df['frame'].unique())

    print("Detecting ball possession and pass events...")
    for frame_num in tqdm(unique_frames):
        df_frame = df[df['frame'] == frame_num].copy()
        
        # Get ball position for the current frame
        ball_data = df_frame[df_frame['class_id'] == 32] # class_id 32 is 'sports ball'
        
        current_ball_pos = None
        if not ball_data.empty:
            current_ball_pos = (ball_data['center_x'].iloc[0], ball_data['center_y'].iloc[0])

        current_possessor_id = None
        if current_ball_pos:
            # Find players in the current frame
            players_in_frame = df_frame[df_frame['class_id'] == 0] # class_id 0 is 'person'
            
            if not players_in_frame.empty:
                # Calculate distance from each player to the ball
                distances = np.sqrt(
                    (players_in_frame['center_x'] - current_ball_pos[0])**2 +
                    (players_in_frame['center_y'] - current_ball_pos[1])**2
                )
                
                # Find the player closest to the ball
                closest_player_idx = distances.idxmin()
                closest_player_distance = distances.loc[closest_player_idx]
                
                # Define a threshold for possession (e.g., 50 pixels)
                possession_threshold = 50 # pixels, adjust as needed
                
                if closest_player_distance <= possession_threshold:
                    current_possessor_id = players_in_frame.loc[closest_player_idx]['track_id']
        
        # Record possession event
        ball_events.append({
            "frame": frame_num,
            "event_type": "possession",
            "possessor_id": current_possessor_id,
            "ball_x": current_ball_pos[0] if current_ball_pos else np.nan,
            "ball_y": current_ball_pos[1] if current_ball_pos else np.nan,
            "from_player_id": np.nan, # For passes
            "to_player_id": np.nan    # For passes
        })

        # Detect possession change and passes
        if prev_possessor_id is not None and current_possessor_id is not None and \
           prev_possessor_id != current_possessor_id:
            
            # Simple pass detection: if possessor changes and ball moved from prev_possessor to current_possessor
            # This is a basic heuristic and can be refined (e.g., check ball speed, trajectory)
            ball_events.append({
                "frame": frame_num,
                "event_type": "pass",
                "possessor_id": current_possessor_id, # New possessor
                "ball_x": current_ball_pos[0] if current_ball_pos else np.nan,
                "ball_y": current_ball_pos[1] if current_ball_pos else np.nan,
                "from_player_id": prev_possessor_id,
                "to_player_id": current_possessor_id
            })
        
        # Update for next iteration
        prev_possessor_id = current_possessor_id
        prev_ball_pos = current_ball_pos

    print(f"Saving ball events to {output_csv_path}...")
    events_df = pd.DataFrame(ball_events)
    events_df.to_csv(output_csv_path, index=False)
    print("Ball event detection complete.")

def calculate_pressure_metrics(tracking_data_path, ball_events_path, output_csv_path):
    print(f"Reading tracking data from {tracking_data_path} for pressure metrics...")
    df_tracking = pd.read_csv(tracking_data_path)
    print(f"Reading ball events from {ball_events_path} for pressure metrics...")
    df_ball_events = pd.read_csv(ball_events_path)

    pressure_metrics = []

    # Define pressure radius (scaled from 3 meters, assuming 1000 pixels = 105 meters pitch length)
    # 1 meter approx = 1000 / 105 = 9.52 pixels
    pressure_radius_pixels = 3 * (1000 / 105) # approx 28.5 pixels

    unique_frames = sorted(df_tracking['frame'].unique())

    print("Calculating pressure metrics per frame...")
    for frame_num in tqdm(unique_frames):
        df_frame = df_tracking[df_tracking['frame'] == frame_num].copy()
        df_ball_event_frame = df_ball_events[df_ball_events['frame'] == frame_num].copy()

        possessor_id = df_ball_event_frame[df_ball_event_frame['event_type'] == 'possession']['possessor_id'].iloc[0] if not df_ball_event_frame.empty else np.nan

        if pd.isna(possessor_id):
            pressure_metrics.append({
                "frame": frame_num,
                "possessor_id": np.nan,
                "num_pressuring_opponents": 0,
                "avg_closing_speed": 0
            })
            continue

        possessor_data = df_frame[(df_frame['track_id'] == possessor_id) & (df_frame['class_id'] == 0)]
        if possessor_data.empty:
            pressure_metrics.append({
                "frame": frame_num,
                "possessor_id": possessor_id,
                "num_pressuring_opponents": 0,
                "avg_closing_speed": 0
            })
            continue

        possessor_x = possessor_data['center_x'].iloc[0]
        possessor_y = possessor_data['center_y'].iloc[0]

        # Identify opponents (players not on the same team as possessor)
        # Assuming team assignment based on track_id parity
        possessor_team = possessor_id % 2
        opponents_in_frame = df_frame[(df_frame['class_id'] == 0) & (df_frame['track_id'] % 2 != possessor_team)]

        num_pressuring_opponents = 0
        closing_speeds = []

        for idx, opponent_row in opponents_in_frame.iterrows():
            dist = np.sqrt((opponent_row['center_x'] - possessor_x)**2 + (opponent_row['center_y'] - possessor_y)**2)
            
            if dist <= pressure_radius_pixels:
                num_pressuring_opponents += 1
                # Closing speed: component of opponent's velocity towards possessor
                # Simplified: just use opponent's speed for now, more complex is dot product of velocity and vector to possessor
                closing_speeds.append(opponent_row['speed'])
        
        avg_closing_speed = np.mean(closing_speeds) if closing_speeds else 0

        pressure_metrics.append({
            "frame": frame_num,
            "possessor_id": possessor_id,
            "num_pressuring_opponents": num_pressuring_opponents,
            "avg_closing_speed": avg_closing_speed
        })

    print(f"Saving pressure metrics to {output_csv_path}...")
    pressure_df = pd.DataFrame(pressure_metrics)
    pressure_df.to_csv(output_csv_path, index=False)
    print("Pressure metrics calculation complete.")

def calculate_line_breaking_passes(tracking_data_path, ball_events_path, output_csv_path):
    print(f"Reading tracking data from {tracking_data_path} for line-breaking passes...")
    df_tracking = pd.read_csv(tracking_data_path)
    print(f"Reading ball events from {ball_events_path} for line-breaking passes...")
    df_ball_events = pd.read_csv(ball_events_path)

    line_breaking_passes = []

    # Filter for pass events
    passes_df = df_ball_events[df_ball_events['event_type'] == 'pass'].copy()

    print("Detecting line-breaking passes...")
    for idx, pass_event in tqdm(passes_df.iterrows(), total=passes_df.shape[0]):
        frame_num = pass_event['frame']
        passer_id = pass_event['from_player_id']
        receiver_id = pass_event['to_player_id']

        # Ensure passer and receiver are valid and present in tracking data for this frame
        if pd.isna(passer_id) or pd.isna(receiver_id):
            continue

        df_frame = df_tracking[df_tracking['frame'] == frame_num].copy() 
        
        passer_data = df_frame[(df_frame['track_id'] == passer_id) & (df_frame['class_id'] == 0)]
        receiver_data = df_frame[(df_frame['track_id'] == receiver_id) & (df_frame['class_id'] == 0)]

        if passer_data.empty or receiver_data.empty:
            continue

        passer_x, passer_y = passer_data['center_x'].iloc[0], passer_data['center_y'].iloc[0]
        receiver_x, receiver_y = receiver_data['center_x'].iloc[0], receiver_data['center_y'].iloc[0]

        # Simplified Line-Breaking Logic:
        # A pass is considered line-breaking if it significantly advances the ball upfield
        # AND there were a certain number of opponents between the passer and receiver.
        
        # Upfield distance (assuming Y-axis is upfield, adjust if pitch orientation is different)
        upfield_distance = receiver_y - passer_y

        # Identify opponents (players not on the same team as passer/receiver)
        passer_team = passer_id % 2
        opponents_in_frame = df_frame[(df_frame['class_id'] == 0) & (df_frame['track_id'] % 2 != passer_team)]

        num_opponents_between = 0
        # This is a very simplified check. A true line-breaking pass needs geometric intersection with defensive lines.
        # For now, we count opponents whose Y-coordinate is between passer and receiver (upfield progress)
        # and whose X-coordinate is within a reasonable horizontal range of the pass trajectory.
        
        # Define a horizontal buffer around the pass trajectory
        horizontal_buffer = 50 # pixels

        for idx, opponent_row in opponents_in_frame.iterrows():
            opp_x, opp_y = opponent_row['center_x'], opponent_row['center_y']

            # Check if opponent is roughly between passer and receiver in Y-coordinate
            if (min(passer_y, receiver_y) < opp_y < max(passer_y, receiver_y)):
                # Check if opponent is horizontally close to the pass line
                # This is a rough check, more accurate would be distance from point to line segment
                if min(passer_x, receiver_x) - horizontal_buffer < opp_x < max(passer_x, receiver_x) + horizontal_buffer:
                    num_opponents_between += 1

        # Thresholds for line-breaking pass
        upfield_distance_threshold = 100 # pixels
        min_opponents_between_threshold = 1 # At least one opponent between

        is_line_breaking = False
        if upfield_distance > upfield_distance_threshold and num_opponents_between >= min_opponents_between_threshold:
            is_line_breaking = True

        line_breaking_passes.append({
            "frame": frame_num,
            "passer_id": passer_id,
            "receiver_id": receiver_id,
            "is_line_breaking": is_line_breaking,
            "upfield_distance": upfield_distance,
            "num_opponents_between": num_opponents_between
        })

    print(f"Saving line-breaking passes to {output_csv_path}...")
    lbp_df = pd.DataFrame(line_breaking_passes)
    lbp_df.to_csv(output_csv_path, index=False)
    print("Line-breaking pass detection complete.")

def calculate_xg_metrics(tracking_data_path, ball_events_path, output_csv_path):
    print(f"Reading tracking data from {tracking_data_path} for xG calculation...")
    df_tracking = pd.read_csv(tracking_data_path)
    print(f"Reading ball events from {ball_events_path} for xG calculation...")
    df_ball_events = pd.read_csv(ball_events_path)

    xg_metrics = []

    # Define goal coordinates (simplified: center of goal, width of goal)
    # Assuming goal is at pitch_length (1000) and pitch_width (600)
    pitch_length = 1000
    pitch_width = 600
    goal_center_x = pitch_length
    goal_center_y = pitch_width / 2
    goal_width = 732 / (105 / pitch_length) # Scale real goal width (7.32m) to pitch pixels

    # Filter for frames where ball speed is high and a player is near the ball
    # This is a very basic shot detection. More advanced would involve ball trajectory analysis.
    shot_detection_threshold_speed = 10 # pixels per frame
    player_ball_distance_threshold = 50 # pixels

    unique_frames = sorted(df_tracking['frame'].unique())

    print("Detecting shots and calculating xG...")
    for frame_num in tqdm(unique_frames):
        df_frame = df_tracking[df_tracking['frame'] == frame_num].copy()
        df_ball_event_frame = df_ball_events[df_ball_events['frame'] == frame_num].copy()

        ball_data = df_frame[df_frame['class_id'] == 32]
        if ball_data.empty: continue

        ball_x, ball_y = ball_data['center_x'].iloc[0], ball_data['center_y'].iloc[0]
        ball_speed = ball_data['speed'].iloc[0]

        # Check if ball speed is above threshold and a player is nearby
        players_in_frame = df_frame[df_frame['class_id'] == 0]
        player_near_ball = False
        shooter_id = np.nan

        for idx, player_row in players_in_frame.iterrows():
            dist_to_ball = np.sqrt((player_row['center_x'] - ball_x)**2 + (player_row['center_y'] - ball_y)**2)
            if dist_to_ball <= player_ball_distance_threshold:
                player_near_ball = True
                shooter_id = player_row['track_id']
                break
        
        # Basic shot detection: high ball speed, player near ball, and ball moving towards goal
        # Simplified: just check if ball is moving generally towards the goal side (positive X direction)
        if ball_speed > shot_detection_threshold_speed and player_near_ball and ball_data['velocity_x'].iloc[0] > 0:
            # Calculate distance to goal line (simplified)
            distance_to_goal = pitch_length - ball_x

            # Calculate angle to goal (simplified: angle from ball to goal posts)
            # Assuming goal posts are at (pitch_length, goal_center_y - goal_width/2) and (pitch_length, goal_center_y + goal_width/2)
            angle_to_goal = np.arctan2(goal_center_y + goal_width/2 - ball_y, pitch_length - ball_x) - \
                            np.arctan2(goal_center_y - goal_width/2 - ball_y, pitch_length - ball_x)
            angle_to_goal = np.degrees(angle_to_goal) # Convert to degrees

            # Basic xG model (logistic regression inspired, coefficients are illustrative)
            # xG = 1 / (1 + exp(-(b0 + b1*distance + b2*angle))) - this needs training data
            # For demo, a simpler distance-based probability
            xg_value = 0.5 * np.exp(-0.01 * distance_to_goal) # Example: probability decreases with distance
            xg_value = max(0.01, min(0.99, xg_value)) # Clamp between 0.01 and 0.99

            xg_metrics.append({
                "frame": frame_num,
                "shooter_id": shooter_id,
                "shot_x": ball_x,
                "shot_y": ball_y,
                "distance_to_goal": distance_to_goal,
                "angle_to_goal": angle_to_goal,
                "xg_value": xg_value
            })

    print(f"Saving xG metrics to {output_csv_path}...")
    xg_df = pd.DataFrame(xg_metrics)
    xg_df.to_csv(output_csv_path, index=False)
    print("xG calculation complete.")

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- For Pitch Control Metrics (already done) ---
    # input_csv_pc = "tracking_data_with_velocities.csv"
    # output_csv_pc = "pitch_control_metrics.csv"
    # input_path_pc = os.path.join(project_dir, input_csv_pc)
    # output_path_pc = os.path.join(project_dir, output_csv_pc)
    # if os.path.exists(input_path_pc):
    #     calculate_pitch_control_metrics(input_path_pc, output_path_pc)

    # --- For Ball Events (already done) ---
    # input_csv_be = "tracking_data_with_velocities.csv"
    # output_csv_be = "ball_events.csv"
    # input_path_be = os.path.join(project_dir, input_csv_be)
    # output_path_be = os.path.join(project_dir, output_csv_be)
    # if os.path.exists(input_path_be):
    #     calculate_ball_events(input_path_be, output_path_be)

    # --- For Pressure Metrics (already done) ---
    # tracking_data_path_pm = os.path.join(project_dir, "tracking_data_with_velocities.csv")
    # ball_events_path_pm = os.path.join(project_dir, "ball_events.csv")
    # output_csv_pm = os.path.join(project_dir, "pressure_metrics.csv")
    
    # if os.path.exists(tracking_data_path_pm) and os.path.exists(ball_events_path_pm):
    #     calculate_pressure_metrics(tracking_data_path_pm, ball_events_path_pm, output_csv_pm)
    # else:
    #     print("Missing input files for pressure metrics calculation.")

    # --- For Line-Breaking Passes (already done) ---
    # tracking_data_path_lbp = os.path.join(project_dir, "tracking_data_with_velocities.csv")
    # ball_events_path_lbp = os.path.join(project_dir, "ball_events.csv")
    # output_csv_lbp = os.path.join(project_dir, "line_breaking_passes.csv")

    # if os.path.exists(tracking_data_path_lbp) and os.path.exists(ball_events_path_lbp):
    #     calculate_line_breaking_passes(tracking_data_path_lbp, ball_events_path_lbp, output_csv_lbp)
    # else:
    #     print("Missing input files for line-breaking passes calculation.")

    # --- For xG Metrics ---
    tracking_data_path_xg = os.path.join(project_dir, "tracking_data_with_velocities.csv")
    ball_events_path_xg = os.path.join(project_dir, "ball_events.csv")
    output_csv_xg = os.path.join(project_dir, "xg_metrics.csv")

    if os.path.exists(tracking_data_path_xg) and os.path.exists(ball_events_path_xg):
        calculate_xg_metrics(tracking_data_path_xg, ball_events_path_xg, output_csv_xg)
    else:
        print("Missing input files for xG calculation.")

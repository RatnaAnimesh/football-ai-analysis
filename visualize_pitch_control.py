import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
import tempfile
import shutil
from tqdm import tqdm
import subprocess

def draw_pitch(ax):
    """
    Draws a basic football pitch on the given matplotlib axes.
    """
    pitch_length = 1000  # Example pixel length
    pitch_width = 600   # Example pixel width

    ax.plot([0, pitch_length], [0, 0], color="black")
    ax.plot([0, pitch_length], [pitch_width, pitch_width], color="black")
    ax.plot([0, 0], [0, pitch_width], color="black")
    ax.plot([pitch_length, pitch_length], [0, pitch_width], color="black")

    center_x, center_y = pitch_length / 2, pitch_width / 2
    center_circle = plt.Circle((center_x, center_y), 100, color="black", fill=False)
    ax.add_patch(center_circle)
    ax.plot([center_x, center_x], [0, pitch_width], color="black")

    ax.plot([0, 165], [pitch_width/2 - 185, pitch_width/2 - 185], color="black")
    ax.plot([0, 165], [pitch_width/2 + 185, pitch_width/2 + 185], color="black")
    ax.plot([165, 165], [pitch_width/2 - 185, pitch_width/2 + 185], color="black")
    ax.plot([pitch_length - 165, pitch_length], [pitch_width/2 - 185, pitch_width/2 - 185], color="black")
    ax.plot([pitch_length - 165, pitch_length], [pitch_width/2 + 185, pitch_width/2 + 185], color="black")
    ax.plot([pitch_length - 165, pitch_length - 165], [pitch_width/2 - 185, pitch_width/2 + 185], color="black")

    ax.plot([0, 55], [pitch_width/2 - 90, pitch_width/2 - 90], color="black")
    ax.plot([0, 55], [pitch_width/2 + 90, pitch_width/2 + 90], color="black")
    ax.plot([55, 55], [pitch_width/2 - 90, pitch_width/2 + 90], color="black")
    ax.plot([pitch_length - 55, pitch_length], [pitch_width/2 - 90, pitch_width/2 - 90], color="black")
    ax.plot([pitch_length - 55, pitch_length], [pitch_width/2 + 90, pitch_width/2 + 90], color="black")
    ax.plot([pitch_length - 55, pitch_length - 55], [pitch_width/2 - 90, pitch_width/2 + 90], color="black")

    ax.set_xlim(-10, pitch_length + 10)
    ax.set_ylim(-10, pitch_width + 10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#3CB371') # Green color for pitch


def generate_frame_plot(df_frame, frame_number, output_path):
    """
    Generates and saves a single frame plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    team_colors = {0: 'blue', 1: 'red'}
    df_frame['team_color'] = df_frame['track_id'].apply(lambda x: team_colors[x % 2])

    for idx, row in df_frame.iterrows():
        ax.plot(row['center_x'], row['center_y'], 'o', color=row['team_color'], markersize=8)
        ax.text(row['center_x'] + 5, row['center_y'] + 5, str(row['track_id']), color=row['team_color'], fontsize=9)

        arrow_scale = 5.0 
        ax.quiver(row['center_x'], row['center_y'], 
                  row['velocity_x'] * arrow_scale, row['velocity_y'] * arrow_scale, 
                  color=row['team_color'], scale_units='xy', angles='xy', scale=1, width=0.005)

    points = df_frame[['center_x', 'center_y']].values
    # Voronoi needs at least 4 non-collinear points for a 2D simplex, or 2 distinct points for a line
    # To be safe and avoid QhullError, we check for at least 4 distinct points.
    if len(points) >= 4 and len(np.unique(points, axis=0)) >= 4: 
        try:
            vor = Voronoi(points)
            voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='gray', line_width=0.5)
        except Exception as e:
            print(f"Warning: Could not generate Voronoi for frame {frame_number}. Error: {e}")
            print(f"Points: {points}")

    ax.set_title(f"Dynamic Pitch Control - Frame {frame_number}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path, dpi=100) # Save figure
    plt.close(fig) # Close figure to free memory


def generate_video(input_csv_path, output_video_path, fps=5):
    """
    Generates a video from tracking data.
    """
    print(f"Loading data from {input_csv_path}...")
    df_all_frames = pd.read_csv(input_csv_path)
    unique_frames = sorted(df_all_frames['frame'].unique())

    if not unique_frames:
        print("No frames to process for video.")
        return

    # Create a temporary directory for image frames
    temp_dir = tempfile.mkdtemp()
    print(f"Saving temporary image frames to: {temp_dir}")

    # Generate and save each frame plot
    print("Generating individual frame images...")
    for frame_num in tqdm(unique_frames):
        df_frame = df_all_frames[df_all_frames['frame'] == frame_num].copy()
        if not df_frame.empty:
            output_img_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.png")
            generate_frame_plot(df_frame, frame_num, output_img_path)

    # Use ffmpeg to stitch images into a video
    print("Stitching images into video using ffmpeg...")
    # Adjust -r (frame rate) based on your desired video speed
    # Adjust -crf (constant rate factor) for quality (lower is better, 23 is default)
    # Adjust -vf scale for output resolution if needed
    ffmpeg_command = [
        "ffmpeg",
        "-r", str(fps),
        "-i", os.path.join(temp_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-vf", "fps=25,format=yuv420p", # Output at 25fps, yuv420p for broad compatibility
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_video_path
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Video successfully created at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")

    # Clean up temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Cleanup complete.")


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = "tracking_data_with_velocities.csv"
    output_video = "dynamic_pitch_control.mp4"
    
    input_path = os.path.join(project_dir, input_csv)
    output_path = os.path.join(project_dir, output_video)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run analysis.py first to generate the enriched tracking data.")
    else:
        generate_video(input_path, output_path, fps=5) # fps should match the effective frame rate of your data (original_fps / frame_skip)
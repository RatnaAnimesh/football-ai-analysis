# Football Analysis

This repository contains code for advanced football analytics, leveraging computer vision and data science to derive proprietary insights from match footage.

## Project Structure

- `main.py`: Processes raw video, detects players/ball, tracks them, and generates `tracking_data.csv`.
- `analysis.py`: Enriches `tracking_data.csv` with velocities, calculates pitch control metrics, ball events, pressure metrics, and line-breaking passes.
- `visualize_pitch_control.py`: Generates visualizations, including dynamic pitch control plots and videos.
- `requirements.txt`: Lists Python dependencies.
- `videos/`: Directory for input video files.
- `*.csv`: Generated data files (ignored by Git).
- `*.mp4`: Generated video files (ignored by Git).
- `*.pt`: YOLO model weights (ignored by Git).

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RatnaAnimesh/football-ai-analysis.git
    cd football-ai-analysis
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place your video:**
    Place a football match video (e.g., `match.mp4`) in the `videos/` directory.
4.  **Generate tracking data:**
    ```bash
    python main.py
    ```
5.  **Generate analysis metrics:**
    ```bash
    python analysis.py
    ```
6.  **Generate visualizations/video:**
    ```bash
    python visualize_pitch_control.py
    ```

---

*More detailed instructions and analysis coming soon.*

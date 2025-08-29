# ⚽️ Football AI Analysis

This project is a sophisticated, end-to-end football analytics platform that leverages computer vision and machine learning to extract tactical insights from match footage. It transforms raw video into a rich dataset of tracking information, which is then used to calculate advanced performance metrics and generate insightful visualizations.

This project is designed to be a powerful tool for coaches, analysts, and enthusiasts who want to gain a deeper understanding of the beautiful game.

## ✨ Features

*   **🏃‍♂️ Player & Ball Tracking:** Utilizes a custom-trained YOLOv8 model to accurately detect and track players, goalkeepers, referees, and the ball.
*   **🌍 Real-World Coordinate System:** Implements homography to transform pixel coordinates into accurate, real-world pitch coordinates (in meters), enabling precise spatial and movement analysis.
*   **🚀 Advanced Analytics (Meter-Based):** Calculates a suite of advanced metrics, now all based on real-world meter coordinates, including:
    *   **Velocity & Speed (m/s):** Measures the movement speed of every player and the ball in meters per second.
    *   **Team Assignment:** Automatically assigns players to their respective teams using KMeans clustering.
    *   **Possession Analysis:** Determines ball possession with high fidelity, using a meter-based threshold.
    *   **Pitch Control (m²):** Calculates spatial dominance using Voronoi diagrams, providing areas in square meters.
    *   **Pressure Metrics:** Quantifies the pressure exerted on the ball carrier by nearby opponents, including average closing speed in m/s.
    *   **Line-Breaking Passes:** Identifies passes that penetrate defensive lines, based on player positions in meters.
    *   **Expected Goals (xG):** A heuristic model to assess shot quality, now using real-world distances and angles.
*   **📺 Tactical Visualization:** Generates a "God's eye view" tactical video overlaying the analytics onto a 2D representation of the pitch, with all elements accurately scaled to real-world dimensions.
*   **⚙️ Centralized Configuration:** Uses a `config.yaml` file to manage all project parameters, making it easy to customize and maintain.

## ⚙️ How It Works

The platform operates in a multi-stage pipeline:

1.  **Tracking (`main.py`):** The input video is processed frame-by-frame. A YOLOv8 model detects and tracks objects. Raw tracking data (bounding boxes, class names, track IDs) is saved to `tracking_data.csv`, along with video metadata (like FPS) to `metadata.json`.
2.  **Homography Calibration (`homography.py`):** This interactive script allows users to select known points on the pitch in a video frame. It then calculates and saves a homography matrix to `homography_matrix.json`, which is crucial for converting pixel coordinates to real-world meters.
3.  **Analysis (`analysis.py`):** The raw tracking data is enriched using the homography matrix to convert all positions to real-world meter coordinates. It then calculates all advanced tactical metrics (possession, pitch control, pressure, line-breaking passes, xG) and saves them to separate CSV files.
4.  **Visualization (`visualize_pitch_control.py`):** The enriched data and calculated metrics are used to generate a tactical video. This video provides a clear visual representation of the analytics, with players and events accurately mapped onto a scaled football pitch.

## 🛠️ Technology Stack

*   **Computer Vision:** OpenCV, Ultralytics (YOLOv8)
*   **Machine Learning:** PyTorch, TensorFlow, Scikit-learn
*   **Data Manipulation:** Pandas, NumPy
*   **Scientific Computing:** SciPy
*   **Geometric Analysis:** Shapely
*   **Configuration Management:** PyYAML

## 🚀 Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RatnaAnimesh/football-ai-analysis.git
    cd football-ai-analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download YOLOv8 model:**
    The project uses `yolov8l.pt`. You can download it from the [Ultralytics repository](https://github.com/ultralytics/ultralytics) or train your own model. The `main.py` script is configured to look for the `best.pt` model from your training runs.

4.  **Place your video:**
    Create a `videos/` directory in the project root and place your match footage inside (e.g., `videos/match.mp4`).

5.  **Configure Project Settings:**
    Edit `config.yaml` in the project root to adjust pitch dimensions, visualization settings, and analysis thresholds as needed.

6.  **Run the Pipeline:**

    a.  **Generate Tracking Data:**
        ```bash
        python main.py --video videos/your_match.mp4
        ```
        This will create `tracking_data.csv` and `metadata.json`.

    b.  **Calibrate Homography:**
        ```bash
        python homography.py --video videos/your_match.mp4
        ```
        Follow the on-screen instructions to select 4 known points on the pitch. This will create `homography_matrix.json`.

    c.  **Perform Analysis:**
        ```bash
        python analysis.py
        ```
        This will create `tracking_data_enriched.csv` and various analytics CSVs (e.g., `possession_events.csv`, `pitch_control.csv`).

    d.  **Generate Tactical Video:**
        ```bash
        python visualize_pitch_control.py
        ```
        The output video will be saved as `tactical_video.mp4`.

## 🔬 Development & Debugging Tools

*   **`visualize_annotations.py`:** Helps visually inspect bounding box annotations on a sample of images from your dataset. Useful for debugging annotation quality.
    ```bash
    python visualize_annotations.py --split val --num_samples 50
    ```
*   **`analyze_ball_annotations.py`:** Provides statistical analysis of ball bounding box sizes and aspect ratios in your dataset. Crucial for understanding small object detection challenges.
    ```bash
    python analyze_ball_annotations.py
    ```

## 🔮 Future Work

*   **Improved Ball Detection:** Implement advanced data augmentation techniques (e.g., copy-paste augmentation) or explore specialized models to improve the detection of small, occluded balls.
*   **Advanced xG Model:** Develop a more sophisticated xG model using machine learning, incorporating features like player positions, pressure, and shot type.
*   **Packing & Progressive Pass Analysis:** Implement metrics to quantify how effectively a team is breaking through the opponent's defensive lines.
*   **Automated Event Detection:** Automatically detect key events like shots, passes, and fouls directly from the video stream.
*   **Interactive Dashboard:** Build a web-based dashboard to visualize the data and allow for interactive analysis.
*   **Real-time Analysis:** Adapt the pipeline to perform real-time analysis on a live video feed.

---

*This project was developed by Animesh Ratna as part of his "Zero to Quant Hero" campaign.*
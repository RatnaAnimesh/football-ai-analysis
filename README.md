# ⚽️ Football AI Analysis

This project is a sophisticated, end-to-end football analytics platform that leverages computer vision and machine learning to extract tactical insights from match footage. It transforms raw video into a rich dataset of tracking information, which is then used to calculate advanced performance metrics and generate insightful visualizations.

This project is designed to be a powerful tool for coaches, analysts, and enthusiasts who want to gain a deeper understanding of the beautiful game.

## ✨ Features

*   **🏃‍♂️ Player & Ball Tracking:** Utilizes a custom-trained YOLOv8 model to accurately detect and track players, goalkeepers, referees, and the ball.
*   **🚀 Advanced Analytics:** Calculates a suite of advanced metrics, including:
    *   **Velocity & Speed:** Measures the movement speed of every player and the ball.
    *   **Team Assignment:** Automatically assigns players to their respective teams using KMeans clustering.
    *   **Possession Analysis:** Determines ball possession with high fidelity.
    *   **Pitch Control:** Calculates spatial dominance using Voronoi diagrams.
    *   **Pressure Metrics:** Quantifies the pressure exerted on the ball carrier.
    *   **Line-Breaking Passes:** Identifies passes that penetrate defensive lines.
    *   **Expected Goals (xG):** A heuristic model to assess shot quality.
*   **📺 Tactical Visualization:** Generates a "God's eye view" tactical video overlaying the analytics onto a 2D representation of the pitch.

## ⚙️ How It Works

The platform operates in a three-stage pipeline:

1.  **Tracking (`main.py`):** The input video is processed frame-by-frame. The YOLOv8 model detects and tracks objects, and the raw tracking data (bounding boxes, class names, track IDs) is saved to `tracking_data.csv`.
2.  **Analysis (`analysis.py`):** The raw tracking data is enriched with calculated metrics. Each analysis module (possession, pitch control, etc.) processes the data and saves its output to a separate CSV file.
3.  **Visualization (`visualize_pitch_control.py`):** The enriched data is used to generate a tactical video. This video provides a clear visual representation of the analytics, making it easy to understand complex tactical situations.

## 🛠️ Technology Stack

*   **Computer Vision:** OpenCV, Ultralytics (YOLOv8)
*   **Machine Learning:** PyTorch, TensorFlow, Scikit-learn
*   **Data Manipulation:** Pandas, NumPy
*   **Scientific Computing:** SciPy
*   **Geometric Analysis:** Shapely

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
    The project uses `yolov8l.pt`. You can download it from the [Ultralytics repository](https://github.com/ultralytics/ultralytics) or train your own model.

4.  **Place your video:**
    Create a `videos/` directory and place your match footage inside (e.g., `videos/match.mp4`).

5.  **Run the pipeline:**
    ```bash
    # 1. Generate tracking data
    python main.py --video videos/your_match.mp4 --model yolov8l.pt

    # 2. Perform analysis
    python analysis.py

    # 3. Generate tactical video
    python visualize_pitch_control.py
    ```
    The output video will be saved as `tactical_video.mp4`.

## 🔮 Future Work

*   **Advanced xG Model:** Develop a more sophisticated xG model using machine learning, incorporating features like player positions, pressure, and shot type.
*   **Automated Event Detection:** Automatically detect key events like shots, passes, and fouls directly from the video stream.
*   **Interactive Dashboard:** Build a web-based dashboard to visualize the data and allow for interactive analysis.
*   **Real-time Analysis:** Adapt the pipeline to perform real-time analysis on a live video feed.

---

*This project was developed by Animesh Ratna as part of his "Zero to Quant Hero" campaign.*
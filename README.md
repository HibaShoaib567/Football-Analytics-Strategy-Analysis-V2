# Football Analytics & Strategy Analysis System

An AI-powered football video analysis system developed using **YOLOv8 Nano, ByteTrack, OpenCV, Machine Learning, and Gradio**.

The system analyzes football match videos to detect players, balls, and referees, track player movements, classify teams based on jersey colors, generate match statistics, and evaluate five predefined football strategy patterns.

## Project Overview

This project provides an interactive dashboard where users can upload a football video, configure detection confidence, select an attacking team, and evaluate a football strategy.

The system generates a processed video with object detection and tracking overlays, along with player statistics, team statistics, strategy evaluation reports, and downloadable CSV files.

## Main Features

### 1. Object Detection
- Uses the pretrained YOLOv8 Nano model.
- Detects players, footballs, and referees.
- Displays bounding boxes and detection labels.

### 2. Player Tracking
- Uses ByteTrack for object tracking.
- Assigns tracking IDs to detected players.
- Stores player tracking information across video frames.

### 3. Team Classification
- Extracts jersey color features using HSV and RGB color information.
- Uses K-Means clustering to group players into Team A and Team B.

### 4. Player Movement Analysis
- Calculates player movement in pixels.
- Generates player performance statistics.
- Displays a Top 15 player movement chart.

### 5. Team Statistics
- Generates team classification statistics.
- Displays estimated team possession statistics.
- Provides team statistics in CSV format.

### 6. Camera Movement Estimation
- Uses OpenCV Optical Flow.
- Estimates camera movement between consecutive frames.
- Stores camera movement information in the tracking report.

### 7. Football Strategy Pattern Evaluation

The system evaluates five predefined football strategy patterns:

1. Wing Attack
2. Central Attack
3. Counter Attack
4. Through Ball
5. Crossing and Finishing

Users can select Team A or Team B and evaluate a selected strategy. The system calculates a heuristic pattern score and provides an evaluation status.

**Note:** Strategy scores represent heuristic pattern matching. They are not probabilities of scoring a goal or proof of successful tactical execution.

### 8. Interactive Gradio Dashboard
- Upload football match videos.
- Adjust detection confidence.
- Select the attacking team.
- Select a football strategy.
- View processed videos with detection overlays.
- View strategy evaluation reports.
- View player and team statistics.
- Download CSV reports.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8 Nano | Object detection |
| ByteTrack | Object tracking |
| OpenCV | Video processing and optical flow |
| K-Means | Team classification |
| HSV / RGB | Jersey color feature extraction |
| Pandas | Data processing and CSV reports |
| Matplotlib | Data visualization |
| Gradio | Interactive web interface |

## Project Structure

```text
Football_Project/
│
├── best.pt
├── analytics.py
├── app.py
├── gradio_app.py
├── gradio_app_v2.py
├── strategy_config.py
├── strategy_tracking.py
├── strategy_engine.py
├── test_strategy.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md
│
├── screenshots/
└── outputs/
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Football_Project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg is used for browser-compatible video conversion. Install FFmpeg and ensure it is available in your system PATH.

### 5. Run the Application

```bash
python gradio_app_v2.py
```

Open the local Gradio URL displayed in the terminal, usually:

```text
http://127.0.0.1:7860
```

## How to Use

1. Open the Gradio dashboard.
2. Upload a football match video.
3. Adjust the detection confidence if required.
4. Select Team A or Team B.
5. Select one of the five football strategies.
6. Click **Start Football Analysis**.
7. View the processed video and analysis reports.
8. Download the generated CSV reports.

## Output Reports

The system generates:

- Player statistics CSV
- Team statistics CSV
- Player tracking CSV
- Strategy tracking CSV
- Strategy evaluation CSV
- Processed football video

## Important Limitations

- Team classification depends on visible and distinguishable jersey colors.
- Player movement is measured in pixels, not calibrated real-world distance.
- Camera movement estimation is approximate and may accumulate errors.
- Strategy evaluation uses predefined heuristic signals and tracking coordinates.
- Actual ball possession, pass completion, successful crosses, goals, and tactical success are not independently verified.
- The current strategy evaluation uses the configured top-to-bottom attacking direction for the portrait test video.

## Future Improvements

- Improve camera-motion compensation.
- Add pitch calibration for real-world player movement.
- Implement ball possession and pass detection.
- Improve tactical pattern recognition.
- Add automatic strategy identification.
- Evaluate strategies across multiple videos and different camera angles.

## Project Purpose

This project demonstrates the application of computer vision, object tracking, machine learning, and interactive visualization in football match analysis.

**Model:** YOLOv8 Nano  
**Frontend:** Gradio  
**Language:** Python


**👩‍💻 Developed By
Hiba Shoaib**


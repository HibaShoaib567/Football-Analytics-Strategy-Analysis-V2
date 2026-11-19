
"""
Football Project V2
YOLOv8 Player and Ball Tracking
Jersey-Based Team Identification
Camera Motion Estimation and Compensation
"""

import os
import cv2
import pandas as pd
import numpy as np

from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_jersey_features(frame, box):
    """Extract jersey features from the central torso area."""

    x1, y1, x2, y2 = map(int, box)
    height, width = frame.shape[:2]

    x1 = max(0, x1)
    x2 = min(width, x2)
    y1 = max(0, y1)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    player_width = x2 - x1
    player_height = y2 - y1

    crop_x1 = x1 + int(player_width * 0.20)
    crop_x2 = x1 + int(player_width * 0.80)
    crop_y1 = y1 + int(player_height * 0.18)
    crop_y2 = y1 + int(player_height * 0.58)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    yellow_mask = (
        (h >= 15) &
        (h <= 40) &
        (s >= 65) &
        (v >= 55)
    )

    low_saturation_mask = s < 55

    yellow_ratio = float(np.mean(yellow_mask))
    low_saturation_ratio = float(np.mean(low_saturation_mask))

    pixels = crop.reshape(-1, 3)
    median_bgr = np.median(pixels, axis=0)
    median_rgb = median_bgr[::-1].astype(float)

    return {
        "rgb": median_rgb,
        "yellow_ratio": yellow_ratio,
        "low_saturation_ratio": low_saturation_ratio
    }


def estimate_camera_motion(previous_gray, current_gray):
    """
    Estimate frame-to-frame camera translation using optical flow.

    Returns:
        dx, dy, valid
    """

    if previous_gray is None or current_gray is None:
        return 0.0, 0.0, False

    previous_points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )

    if previous_points is None or len(previous_points) < 8:
        return 0.0, 0.0, False

    current_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if current_points is None or status is None:
        return 0.0, 0.0, False

    valid_mask = status.reshape(-1) == 1

    old_points = previous_points.reshape(-1, 2)[valid_mask]
    new_points = current_points.reshape(-1, 2)[valid_mask]

    if len(old_points) < 8:
        return 0.0, 0.0, False

    # Robust affine estimation using RANSAC
    affine, inliers = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99
    )

    if affine is not None:
        dx = float(affine[0, 2])
        dy = float(affine[1, 2])

        # Reject implausibly large single-frame movement
        if abs(dx) <= 100 and abs(dy) <= 100:
            return dx, dy, True

    # Fallback: median optical-flow displacement
    displacement = new_points - old_points

    dx = float(np.median(displacement[:, 0]))
    dy = float(np.median(displacement[:, 1]))

    if abs(dx) <= 100 and abs(dy) <= 100:
        return dx, dy, True

    return 0.0, 0.0, False


def assign_teams(tracking_df):
    """Assign stable video-specific team labels by Track_ID."""

    tracking_df = tracking_df.copy()
    tracking_df["Team"] = "Unknown"

    player_mask = (
        (tracking_df["Object"] == "Player") &
        tracking_df["Track_ID"].notna()
    )

    players = tracking_df[player_mask].copy()

    feature_columns = [
        "Yellow_Ratio",
        "LowSat_Ratio",
        "Jersey_R",
        "Jersey_G",
        "Jersey_B"
    ]

    valid_players = players.dropna(subset=feature_columns)

    if valid_players.empty:
        return tracking_df

    track_features = (
        valid_players
        .groupby("Track_ID")[feature_columns]
        .median()
        .dropna()
    )

    if len(track_features) < 2:
        return tracking_df

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(track_features.values)

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(scaled_features)

    cluster_yellow_ratios = {}

    for cluster_id in [0, 1]:
        cluster_tracks = track_features.iloc[
            np.where(labels == cluster_id)[0]
        ]

        cluster_yellow_ratios[cluster_id] = (
            cluster_tracks["Yellow_Ratio"].median()
        )

    yellow_cluster = max(
        cluster_yellow_ratios,
        key=cluster_yellow_ratios.get
    )

    team_mapping = {
        track_id: (
            "Team A" if label == yellow_cluster else "Team B"
        )
        for track_id, label in zip(track_features.index, labels)
    }

    tracking_df.loc[player_mask, "Team"] = (
        tracking_df.loc[player_mask, "Track_ID"]
        .map(team_mapping)
        .fillna("Unknown")
    )

    return tracking_df


def collect_strategy_tracking(
    video_path,
    model_path="best.pt",
    output_csv=None,
    confidence=0.25,
    progress_callback=None
):
    """
    Track players and ball with YOLOv8 and ByteTrack.

    Adds optical-flow-based camera motion estimates and
    translation-compensated coordinates.

    Original coordinate columns remain unchanged.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open the video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_width <= 0 or video_height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")

    model = YOLO(model_path)

    records = []
    frame_number = 0

    previous_gray = None

    cumulative_camera_dx = 0.0
    cumulative_camera_dy = 0.0

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            current_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Estimate camera movement between consecutive frames
            camera_dx, camera_dy, camera_valid = (
                estimate_camera_motion(
                    previous_gray,
                    current_gray
                )
            )

            cumulative_camera_dx += camera_dx
            cumulative_camera_dy += camera_dy

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                classes=[0, 1],
                verbose=False
            )

            for result in results:

                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                coordinates = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(coordinates)

                for box, class_id, conf, track_id in zip(
                    coordinates,
                    class_ids,
                    confidences,
                    track_ids
                ):

                    x1, y1, x2, y2 = box

                    center_x = float((x1 + x2) / 2)
                    center_y = float((y1 + y2) / 2)

                    yellow_ratio = None
                    low_sat_ratio = None
                    r, g, b = None, None, None

                    if class_id == 0:

                        object_name = "Player"

                        jersey_features = get_jersey_features(
                            frame,
                            box
                        )

                        if jersey_features is not None:

                            r, g, b = jersey_features["rgb"]

                            yellow_ratio = (
                                jersey_features["yellow_ratio"]
                            )

                            low_sat_ratio = (
                                jersey_features["low_saturation_ratio"]
                            )

                    elif class_id == 1:
                        object_name = "Ball"

                    else:
                        continue

                    # Translation-compensated coordinates.
                    # Original coordinates are retained above.
                    stabilized_x = (
                        center_x - cumulative_camera_dx
                    )

                    stabilized_y = (
                        center_y - cumulative_camera_dy
                    )

                    records.append({

                        "Frame": frame_number,
                        "Object": object_name,
                        "Track_ID": track_id,
                        "Class_ID": class_id,

                        "X1": float(x1),
                        "Y1": float(y1),
                        "X2": float(x2),
                        "Y2": float(y2),

                        "Center_X": center_x,
                        "Center_Y": center_y,

                        "Confidence": float(conf),

                        "Jersey_R": r,
                        "Jersey_G": g,
                        "Jersey_B": b,

                        "Yellow_Ratio": yellow_ratio,
                        "LowSat_Ratio": low_sat_ratio,

                        "Frame_Width": video_width,
                        "Frame_Height": video_height,

                        "Camera_DX": camera_dx,
                        "Camera_DY": camera_dy,

                        "Camera_Motion_Valid": camera_valid,

                        "Cumulative_Camera_DX": (
                            cumulative_camera_dx
                        ),

                        "Cumulative_Camera_DY": (
                            cumulative_camera_dy
                        ),

                        "Stabilized_Center_X": stabilized_x,
                        "Stabilized_Center_Y": stabilized_y

                    })

            previous_gray = current_gray
            frame_number += 1

            if progress_callback and frame_number % 10 == 0:
                if total_frames > 0:
                    progress_callback(
                        min(frame_number / total_frames, 1.0)
                    )

    finally:
        cap.release()

    columns = [
        "Frame",
        "Object",
        "Track_ID",
        "Class_ID",
        "X1",
        "Y1",
        "X2",
        "Y2",
        "Center_X",
        "Center_Y",
        "Confidence",
        "Jersey_R",
        "Jersey_G",
        "Jersey_B",
        "Yellow_Ratio",
        "LowSat_Ratio",
        "Frame_Width",
        "Frame_Height",
        "Camera_DX",
        "Camera_DY",
        "Camera_Motion_Valid",
        "Cumulative_Camera_DX",
        "Cumulative_Camera_DY",
        "Stabilized_Center_X",
        "Stabilized_Center_Y",
        "Team"
    ]

    tracking_df = pd.DataFrame(records)

    if not tracking_df.empty:
        tracking_df = assign_teams(tracking_df)
        tracking_df = tracking_df.reindex(columns=columns)
    else:
        tracking_df = pd.DataFrame(columns=columns)

    if output_csv is not None:

        output_directory = os.path.dirname(
            os.path.abspath(output_csv)
        )

        os.makedirs(output_directory, exist_ok=True)

        tracking_df.to_csv(
            output_csv,
            index=False
        )

    if progress_callback:
        progress_callback(1.0)

    return tracking_df

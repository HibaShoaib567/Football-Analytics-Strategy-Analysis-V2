
import os
import uuid
import subprocess
import cv2
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from analytics import process_video
from strategy_tracking import collect_strategy_tracking
from strategy_config import get_strategy_names
from strategy_engine import evaluate_strategy


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_PATH = os.path.join(PROJECT_DIR, "best.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_strategy_overlay(
    video_path,
    tracking_df,
    strategy_result,
    output_path
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create output video.")

    frame_groups = {
        int(frame): group
        for frame, group in tracking_df.groupby("Frame")
    }

    frame_number = 0

    team_colors = {
        "Team A": (255, 150, 0),
        "Team B": (0, 80, 255),
        "Unknown": (180, 180, 180)
    }

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Draw tracked objects
            current_detections = frame_groups.get(frame_number)

            if current_detections is not None:

                for _, detection in current_detections.iterrows():

                    x1 = int(detection["X1"])
                    y1 = int(detection["Y1"])
                    x2 = int(detection["X2"])
                    y2 = int(detection["Y2"])

                    object_name = detection["Object"]
                    team = detection.get("Team", "Unknown")

                    if pd.isna(team):
                        team = "Unknown"

                    if object_name == "Ball":
                        color = (0, 255, 255)
                        label = "BALL"
                    else:
                        color = team_colors.get(
                            team, team_colors["Unknown"]
                        )

                        track_id = detection["Track_ID"]

                        if pd.isna(track_id):
                            label = str(team)
                        else:
                            label = f"{team} | ID {int(track_id)}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            
            # Compact transparent strategy information panel

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (8, 8),
                (min(width - 8, 470), 105),
                (20, 20, 20),
                -1
            )

            # 65% transparency
            cv2.addWeighted(
                overlay,
                0.60,
                frame,
                0.40,
                0,
                frame
            )

            strategy = strategy_result["strategy"]
            team = strategy_result["attacking_team"]
            score = strategy_result["score"]
            status = strategy_result["status"]

            cv2.putText(
                frame,
                f"Strategy: {strategy}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"Team: {team} | Score: {score:.1f}%",
                (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 180),
                1
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (18, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1
            )


            # Portrait video: attacking direction is downward
            arrow_x = width - 35

            cv2.arrowedLine(
                frame,
                (arrow_x, 140),
                (arrow_x, min(height - 30, 300)),
                (0, 255, 0),
                3,
                tipLength=0.2
            )

            cv2.putText(
                frame,
                "ATTACK",
                (max(5, width - 105), 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2
            )

            writer.write(frame)
            frame_number += 1

    finally:
        cap.release()
        writer.release()

    if not os.path.exists(output_path):
        raise ValueError("Strategy overlay video was not generated.")

    return output_path



def football_analysis_v2(
    video,
    confidence,
    attacking_team,
    strategy_name,
    progress=gr.Progress()
):

    if video is None:
        raise gr.Error("Please upload a football video.")

    if not os.path.exists(MODEL_PATH):
        raise gr.Error("best.pt model file not found!")

    video_path = video

    if not os.path.exists(video_path):
        raise gr.Error("Uploaded video not found!")

    unique_id = uuid.uuid4().hex[:8]

    output_path = os.path.join(
        OUTPUT_DIR, f"processed_{unique_id}.mp4"
    )

    try:
        # -----------------------------------
        # 1. Existing Football Analytics
        # -----------------------------------

        result = process_video(
            video_path=video_path,
            model_path=MODEL_PATH,
            output_path=output_path,
            progress_callback=lambda x: progress(
                x * 0.45,
                desc="Running football analytics..."
            ),
            confidence=confidence
        )

        players_df = result["players"]
        teams_df = result["teams"]
        tracking_df_v1 = result["tracking"]

        # -----------------------------------
        # 2. Strategy Tracking
        # -----------------------------------

        strategy_csv = os.path.join(
            OUTPUT_DIR, f"strategy_tracking_{unique_id}.csv"
        )

        tracking_df = collect_strategy_tracking(
            video_path=video_path,
            model_path=MODEL_PATH,
            output_csv=strategy_csv,
            confidence=confidence,
            progress_callback=lambda x: progress(
                0.45 + (x * 0.45),
                desc="Tracking players and analyzing strategy..."
            )
        )

        # -----------------------------------
        # 3. Strategy Evaluation
        # -----------------------------------

        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if tracking_df.empty:
            raise gr.Error(
                "No players or ball detected. Try another video."
            )

        strategy_result = evaluate_strategy(
            tracking_df=tracking_df,
            strategy_name=strategy_name,
            attacking_team=attacking_team,
            attack_direction="top_to_bottom",
            image_width=video_width,
            image_height=video_height
        )

        # -----------------------------------
        # 4. Save CSV Reports
        # -----------------------------------

        player_csv = os.path.join(
            OUTPUT_DIR, f"player_statistics_{unique_id}.csv"
        )

        team_csv = os.path.join(
            OUTPUT_DIR, f"team_statistics_{unique_id}.csv"
        )

        tracking_v1_csv = os.path.join(
            OUTPUT_DIR, f"player_tracking_{unique_id}.csv"
        )

        strategy_result_csv = os.path.join(
            OUTPUT_DIR, f"strategy_result_{unique_id}.csv"
        )

        players_df.to_csv(player_csv, index=False)
        teams_df.to_csv(team_csv, index=False)
        tracking_df_v1.to_csv(tracking_v1_csv, index=False)

        pd.DataFrame([{
            "Strategy": strategy_result["strategy"],
            "Attacking_Team": strategy_result["attacking_team"],
            "Attack_Direction": strategy_result["attack_direction"],
            "Score": strategy_result["score"],
            "Status": strategy_result["status"],
            "Unique_Players": strategy_result["unique_players"],
            "Player_Detections": strategy_result["player_detections"],
            "Ball_Detections": strategy_result["ball_detections"]
        }]).to_csv(strategy_result_csv, index=False)

        
        # -----------------------------------
        # 5. Create Strategy Overlay Video
        # -----------------------------------

        overlay_path = os.path.join(
            OUTPUT_DIR, f"strategy_overlay_{unique_id}.mp4"
        )

        create_strategy_overlay(
            video_path=video_path,
            tracking_df=tracking_df,
            strategy_result=strategy_result,
            output_path=overlay_path
        )

        # Convert overlay video for browser playback
        browser_video = os.path.join(
            OUTPUT_DIR, f"browser_{unique_id}.mp4"
        )

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", overlay_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    browser_video
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            output_video = (
                browser_video
                if os.path.exists(browser_video)
                else overlay_path
            )

        except Exception:
            output_video = overlay_path


        # -----------------------------------
        # 6. Match Overview
        # -----------------------------------

        unique_players = (
            players_df["Player_ID"].nunique()
            if not players_df.empty else 0
        )

        summary = f"""
FOOTBALL MATCH ANALYSIS COMPLETED

Total Frames: {result['total_frames']}

Unique Players: {unique_players}

Ball Detections: {result['ball_detections']}

Referee Detections: {result['referee_detections']}

Team Classification: Completed

Processed Video: Ready

Strategy Tracking: Completed
"""

        # -----------------------------------
        # 7. Strategy Report
        # -----------------------------------

        signal_text = ""

        for name, value in strategy_result["signal_scores"].items():
            signal_text += f"\n{name}: {value}"

        observations = strategy_result.get("observations", [])

        if isinstance(observations, (list, tuple)):
            observation_text = "\n".join(
                f"- {item}" for item in observations
            )
        else:
            observation_text = str(observations)

        strategy_report = f"""
STRATEGY ANALYSIS REPORT

Selected Strategy: {strategy_result['strategy']}

Attacking Team: {strategy_result['attacking_team']}

Attacking Direction: Top to Bottom

Strategy Pattern Score: {strategy_result['score']:.2f}%

Status: {strategy_result['status']}

Unique Players: {strategy_result['unique_players']}

Player Detections: {strategy_result['player_detections']}

Ball Detections: {strategy_result['ball_detections']}

SIGNAL SCORES:
{signal_text}

OBSERVATIONS:
{observation_text}

Note: This score represents heuristic pattern matching,
not actual tactical success or goal probability.
"""

        # -----------------------------------
        # 8. Player Movement Chart
        # -----------------------------------

        if not players_df.empty:

            chart_df = players_df.nlargest(
                15, "Movement_Pixels"
            ).sort_values("Movement_Pixels")

            fig1, ax1 = plt.subplots(figsize=(9, 5))

            ax1.barh(
                chart_df["Player_ID"].astype(str),
                chart_df["Movement_Pixels"]
            )

            ax1.set_title("Top 15 Players - Movement")
            ax1.set_xlabel("Movement (Pixels)")
            ax1.set_ylabel("Player ID")

            fig1.tight_layout()

        else:
            fig1 = None

        # -----------------------------------
        # 9. Team Possession Chart
        # -----------------------------------

        if not teams_df.empty:

            fig2, ax2 = plt.subplots(figsize=(7, 4))

            ax2.bar(
                teams_df["Team"],
                teams_df["Possession_Percent"]
            )

            ax2.set_title("Team Possession Analysis")
            ax2.set_ylabel("Possession (%)")
            ax2.set_ylim(0, 100)

            fig2.tight_layout()

        else:
            fig2 = None

        progress(1.0, desc="Analysis completed!")

        return (
            output_video,
            summary,
            strategy_report,
            players_df,
            teams_df,
            fig1,
            fig2,
            player_csv,
            team_csv,
            tracking_v1_csv,
            strategy_result_csv
        )

    except Exception as e:
        raise gr.Error(f"Analysis failed: {str(e)}")


# =========================================
# GRADIO INTERFACE
# =========================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Football Analytics and Strategy Analysis"
) as demo:

    gr.Markdown(
        """
        # ⚽ Football Analytics & Strategy Analysis

        ### YOLOv8 Nano Based Football Match Analysis

        Player Detection | Ball Detection | Team Classification |
        Player Tracking | Tactical Strategy Analysis
        """
    )

    with gr.Row():

        with gr.Column():

            video_input = gr.Video(
                label="Upload Football Match Video"
            )

            confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Detection Confidence"
            )

            attacking_team = gr.Radio(
                choices=["Team A", "Team B"],
                value="Team A",
                label="Select Attacking Team"
            )

            strategy_name = gr.Dropdown(
                choices=get_strategy_names(),
                value=get_strategy_names()[0],
                label="Select Football Strategy"
            )

            gr.Markdown(
                "**Attacking Direction:** Top to Bottom"
            )

            analyze_btn = gr.Button(
                "Start Football Analysis",
                variant="primary"
            )

        with gr.Column():

            output_video = gr.Video(
                label="Processed Match Video"
            )

            summary = gr.Textbox(
                label="Match Overview",
                lines=10,
                interactive=False
            )

    gr.Markdown("## 🏆 Tactical Strategy Analysis")

    strategy_report = gr.Textbox(
        label="Strategy Evaluation Report",
        lines=18,
        interactive=False
    )

    gr.Markdown("## 📊 Player Performance Statistics")

    player_table = gr.Dataframe(
        label="Player Statistics",
        interactive=False
    )

    with gr.Row():

        player_chart = gr.Plot(
            label="Player Movement Analysis"
        )

        team_chart = gr.Plot(
            label="Team Possession Analysis"
        )

    gr.Markdown("## 🏅 Team Statistics")

    team_table = gr.Dataframe(
        label="Team Classification and Possession",
        interactive=False
    )

    gr.Markdown("## 📥 Download Match Reports")

    with gr.Row():

        player_download = gr.File(
            label="Player Statistics CSV"
        )

        team_download = gr.File(
            label="Team Statistics CSV"
        )

    with gr.Row():

        tracking_download = gr.File(
            label="Player Tracking CSV"
        )

        strategy_download = gr.File(
            label="Strategy Analysis CSV"
        )

    analyze_btn.click(
        fn=football_analysis_v2,
        inputs=[
            video_input,
            confidence,
            attacking_team,
            strategy_name
        ],
        outputs=[
            output_video,
            summary,
            strategy_report,
            player_table,
            team_table,
            player_chart,
            team_chart,
            player_download,
            team_download,
            tracking_download,
            strategy_download
        ]
    )


demo.launch(
    share=False,
    debug=True,
    server_name="127.0.0.1"
)

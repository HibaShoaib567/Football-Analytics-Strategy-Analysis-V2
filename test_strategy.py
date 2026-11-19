
import os

from strategy_tracking import collect_strategy_tracking
from strategy_engine import evaluate_strategy

# Existing project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(PROJECT_DIR, "best.pt")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enter your test video filename here
VIDEO_PATH = os.path.join(PROJECT_DIR, "Beautiful-football.mp4")

TRACKING_CSV = os.path.join(
    OUTPUT_DIR,
    "strategy_tracking.csv"
)

print("Starting strategy tracking...")

tracking_df = collect_strategy_tracking(
    video_path=VIDEO_PATH,
    model_path=MODEL_PATH,
    output_csv=TRACKING_CSV
)

print("\nTracking completed!")

print("Total detections:", len(tracking_df))


print("\nDetection summary:")
print(tracking_df["Object"].value_counts())

print("\nTeam identification summary:")

if "Team" in tracking_df.columns:

    print(tracking_df["Team"].value_counts())

    print("\nUnique teams:")
    print(tracking_df["Team"].unique())

    print("\nSample tracking data:")
    print(
        tracking_df[
            ["Object", "Track_ID", "Jersey_R",
             "Jersey_G", "Jersey_B", "Team"]
        ].head(10)
    )

else:
    print("Team column not found!")

print("\nUnique Player Team Assignments:")

player_teams = (
    tracking_df[tracking_df["Object"] == "Player"]
    .groupby("Track_ID")["Team"]
    .nunique()
)

print("Players with multiple team labels:", (player_teams > 1).sum())
print("Total unique player tracks:", player_teams.index.nunique())

print("\nEvaluating selected team's strategy...")

result = evaluate_strategy(
    tracking_df=tracking_df,
    strategy_name="Wing Attack",
    attacking_team="Team A",
    attack_direction="top_to_bottom",
    image_width=1920,
    image_height=1080
)

print("\n========== STRATEGY RESULT ==========")

print("Strategy:", result["strategy"])
print("Attacking Team:", result["attacking_team"])
print("Unique Players:", result["unique_players"])
print("Player Detections:", result["player_detections"])
print("Ball Detections:", result["ball_detections"])

print(
    f"Video Dimensions: "
    f"{result['video_width']} x {result['video_height']}"
)

print("\nExecution Score:", result["score"], "%")
print("Status:", result["status"])

print("\nIndividual Signal Scores:")

for signal, details in result["signal_scores"].items():
    print(
        f"{signal}: {details['score']:.2f}% "
        f"(Weight: {details['weight']}%)"
    )

print("\nObservations:")

for observation in result["observations"]:
    print("-", observation)


print("\n\n========== TEAM B EVALUATION ==========")

result_b = evaluate_strategy(
    tracking_df=tracking_df,
    strategy_name="Wing Attack",
    attacking_team="Team B",
    attack_direction="top_to_bottom",
    image_width=1920,
    image_height=1080
)

print("Strategy:", result_b["strategy"])
print("Attacking Team:", result_b["attacking_team"])
print("Unique Players:", result_b["unique_players"])
print(
    f"Video Dimensions: "
    f"{result['video_width']} x {result['video_height']}"
)
print("Execution Score:", result_b["score"], "%")
print("Status:", result_b["status"])

print("\nIndividual Signal Scores:")

for signal, details in result_b["signal_scores"].items():
    print(
        f"{signal}: {details['score']:.2f}% "
        f"(Weight: {details['weight']}%)"
    )


print("\n========== TEAM COLOR ANALYSIS ==========")

player_data = tracking_df[
    tracking_df["Object"] == "Player"
].copy()


track_summary = (
    player_data.groupby(["Track_ID", "Team"])
    .agg({
        "Jersey_R": "median",
        "Jersey_G": "median",
        "Jersey_B": "median",
        "Yellow_Ratio": "median",
        "LowSat_Ratio": "median",
        "Frame": "count"
    }).reset_index()
)


print("\nUnique tracks per team:")
print(track_summary["Team"].value_counts())

print("\nAverage jersey color per team:")

print(
    track_summary.groupby("Team")[
        ["Jersey_R", "Jersey_G", "Jersey_B"]
    ].mean().round(2)
)

print("\nAverage Jersey Features per Team:")

print(
    track_summary.groupby("Team")[
        ["Yellow_Ratio", "LowSat_Ratio"]
    ].mean().round(3)
)

print("\nPlayer track details:")
print(
    track_summary[
        ["Track_ID", "Team", "Jersey_R", "Jersey_G", "Jersey_B", "Frame",]
    ].to_string(index=False)
)




print("\n========== ALL STRATEGIES TEST ==========")

from strategy_config import get_strategy_names
from strategy_engine import evaluate_strategy

for strategy in get_strategy_names():
    for team in ["Team A", "Team B"]:

        result = evaluate_strategy(
            tracking_df=tracking_df,
            strategy_name=strategy,
            attacking_team=team,
            attack_direction="top_to_bottom",
            image_width=1080,
            image_height=1920
        )

        print(
            f"{strategy:25} | {team:8} | "
            f"{result['score']:.2f}% | {result['status']}"
        )

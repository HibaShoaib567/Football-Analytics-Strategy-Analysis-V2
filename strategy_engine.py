
import pandas as pd
from strategy_config import STRATEGIES


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def evaluate_strategy(
    tracking_df,
    strategy_name,
    attacking_team="Team A",
    attack_direction="top_to_bottom",
    image_width=None,
    image_height=None
):
    """Evaluate a predefined football strategy using tracking data."""

    # 1. Validate inputs
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    if attacking_team not in ["Team A", "Team B"]:
        raise ValueError("Attacking team must be Team A or Team B.")

    valid_directions = [
        "right",
        "left",
        "top_to_bottom",
        "bottom_to_top"
    ]

    if attack_direction not in valid_directions:
        raise ValueError(
            f"Attack direction must be one of {valid_directions}."
        )

    required_columns = [
        "Frame", "Object", "Team",
        "Track_ID", "Center_X", "Center_Y"
    ]

    for column in required_columns:
        if column not in tracking_df.columns:
            raise ValueError(f"Missing required column: {column}")

    if tracking_df.empty:
        raise ValueError("Tracking data is empty.")

    # 2. Detect actual video dimensions
    if (
        "Frame_Width" in tracking_df.columns
        and tracking_df["Frame_Width"].notna().any()
    ):
        image_width = float(
            tracking_df["Frame_Width"].dropna().iloc[0]
        )

    if (
        "Frame_Height" in tracking_df.columns
        and tracking_df["Frame_Height"].notna().any()
    ):
        image_height = float(
            tracking_df["Frame_Height"].dropna().iloc[0]
        )

    if image_width is None:
        image_width = 1920

    if image_height is None:
        image_height = 1080

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid video dimensions.")

    # 3. Select players and ball
    players = tracking_df[
        (tracking_df["Object"] == "Player") &
        (tracking_df["Team"] == attacking_team) &
        (tracking_df["Track_ID"].notna())
    ].copy()

    ball = tracking_df[
        tracking_df["Object"] == "Ball"
    ].copy()

    if players.empty:
        raise ValueError(
            f"No player detections found for {attacking_team}."
        )

    if ball.empty:
        raise ValueError("No ball detections found.")

    # 4. Normalize coordinates
    players["Norm_X"] = (
        players["Center_X"] / image_width
    )

    players["Norm_Y"] = (
        players["Center_Y"] / image_height
    )

    ball["Norm_X"] = (
        ball["Center_X"] / image_width
    )

    ball["Norm_Y"] = (
        ball["Center_Y"] / image_height
    )

    # 5. Select attacking and lateral axes
    if attack_direction == "right":

        players["Progress"] = players["Norm_X"]
        ball["Progress"] = ball["Norm_X"]

        lateral_position = "Norm_Y"

    elif attack_direction == "left":

        players["Progress"] = 1 - players["Norm_X"]
        ball["Progress"] = 1 - ball["Norm_X"]

        lateral_position = "Norm_Y"

    elif attack_direction == "top_to_bottom":

        players["Progress"] = players["Norm_Y"]
        ball["Progress"] = ball["Norm_Y"]

        lateral_position = "Norm_X"

    else:  # bottom_to_top

        players["Progress"] = 1 - players["Norm_Y"]
        ball["Progress"] = 1 - ball["Norm_Y"]

        lateral_position = "Norm_X"

    # 6. Use one representative position per player track
    player_tracks = (
        players.groupby("Track_ID")
        .agg({
            "Norm_X": "median",
            "Norm_Y": "median",
            "Progress": "median"
        })
    )

    if player_tracks.empty:
        raise ValueError("No valid player tracks available.")

    # 7. Ball movement by frame
    ball_by_frame = (
        ball.groupby("Frame")
        .agg({
            "Progress": "median",
            "Norm_X": "median",
            "Norm_Y": "median"
        })
        .sort_index()
    )

    ball_progress = ball_by_frame["Progress"]

    # 8. Ball progression score
    if len(ball_progress) >= 2:

        progress_change = (
            ball_progress.iloc[-1] -
            ball_progress.iloc[0]
        )

        ball_score = clamp(
            50 + progress_change * 200
        )

    else:
        ball_score = 50

    # 9. Forward ball movement score
    if len(ball_progress) >= 2:

        movement = ball_progress.diff().dropna()

        forward_score = (
            (movement > 0).sum() / len(movement)
        ) * 100

    else:
        forward_score = 0

    # 10. Wide positioning score
    lateral_spread = (
        player_tracks[lateral_position].max() -
        player_tracks[lateral_position].min()
    )

    wide_score = clamp(lateral_spread * 100)

    # 11. Central positioning score
    central_players = player_tracks[
        (player_tracks[lateral_position] >= 0.30) &
        (player_tracks[lateral_position] <= 0.70)
    ]

    central_score = (
        len(central_players) / len(player_tracks)
    ) * 100

    # 12. Attacking support score
    attacking_players = player_tracks[
        player_tracks["Progress"] >= 0.50
    ]

    team_position_score = (
        len(attacking_players) / len(player_tracks)
    ) * 100

    # 13. Strategy-specific signals
    signal_mapping = {

        "Wing Attack": [
            ("Wide positioning", wide_score),
            ("Ball progression", ball_score),
            ("Attacking support", team_position_score)
        ],

        "Central Attack": [
            ("Central positioning", central_score),
            ("Ball progression", ball_score),
            ("Attacking support", team_position_score)
        ],

        "Counter Attack": [
            ("Forward ball movement", forward_score),
            ("Player attacking support", team_position_score),
            ("Attacking progression", ball_score)
        ],

        "Through Ball": [
            ("Forward ball movement", ball_score),
            ("Forward movement", forward_score),
            ("Attacking support", team_position_score)
        ],

        "Crossing and Finishing": [
            ("Wide positioning", wide_score),
            ("Central positioning", central_score),
            ("Ball progression", ball_score)
        ]
    }

    configured_signals = STRATEGIES[strategy_name]["signals"]
    calculated_signals = signal_mapping[strategy_name]

    if len(configured_signals) != len(calculated_signals):
        raise ValueError(
            "Strategy signal configuration mismatch."
        )

    total_weight = sum(
        signal["weight"]
        for signal in configured_signals
    )

    if total_weight <= 0:
        raise ValueError("Strategy weights must be positive.")

    signal_scores = {}
    weighted_total = 0

    for config_signal, calculated_signal in zip(
        configured_signals,
        calculated_signals
    ):

        signal_name, score = calculated_signal
        weight = config_signal["weight"]

        signal_scores[signal_name] = {
            "score": round(score, 2),
            "weight": weight
        }

        weighted_total += score * weight

    final_score = round(
        weighted_total / total_weight,
        2
    )

    # 14. Determine pattern-match status
    if final_score >= 70:
        status = "Strong Pattern Match"

    elif final_score >= 40:
        status = "Partial Pattern Match"

    else:
        status = "Weak Pattern Match"

    # 15. Generate observations
    observations = []

    if ball_score >= 60:
        observations.append(
            "Ball showed forward screen-coordinate progression."
        )
    else:
        observations.append(
            "Limited overall ball progression detected."
        )

    if wide_score >= 60:
        observations.append(
            "Players occupied a relatively wide screen area."
        )

    if central_score >= 60:
        observations.append(
            "Most tracked players occupied the central screen area."
        )

    if team_position_score >= 50:
        observations.append(
            "Many tracked players were in the attacking half."
        )
    else:
        observations.append(
            "Few tracked players were in the attacking half."
        )

    observations.append(
        "Player positioning scores use unique player tracks."
    )

    observations.append(
        "Ball progression is calculated from the overall ball track, "
        "not confirmed team possession."
    )

    observations.append(
        "Camera movement is not fully compensated; screen-coordinate "
        "changes may affect the results."
    )

    observations.append(
        "Team labels are estimated using jersey-colour clustering."
    )

    observations.append(
        "The score is a heuristic pattern match, not proof of actual "
        "tactical execution or a probability of success."
    )

    # 16. Return result
    return {

        "strategy": strategy_name,
        "attacking_team": attacking_team,
        "attack_direction": attack_direction,

        "video_width": int(image_width),
        "video_height": int(image_height),

        "score": final_score,
        "status": status,

        "unique_players": len(player_tracks),
        "player_detections": len(players),
        "ball_detections": len(ball),

        "signal_scores": signal_scores,
        "observations": observations

    }

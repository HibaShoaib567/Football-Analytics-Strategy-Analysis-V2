
"""
Football Project V2
Predefined Football Strategies Configuration
"""

STRATEGIES = {
    "Wing Attack": {
        "description": (
            "Attack through the left or right side of the pitch "
            "and move the ball towards the opponent's goal."
        ),
        "signals": [
            {
                "name": "Ball enters a wide area",
                "weight": 30
            },
            {
                "name": "Ball moves towards the opponent's goal",
                "weight": 40
            },
            {
                "name": "Teammates provide attacking support",
                "weight": 30
            }
        ]
    },

    "Central Attack": {
        "description": (
            "Build an attack through the central area of the pitch "
            "and advance towards the opponent's goal."
        ),
        "signals": [
            {
                "name": "Ball remains in the central area",
                "weight": 30
            },
            {
                "name": "Ball advances towards the opponent's goal",
                "weight": 40
            },
            {
                "name": "Teammates support the central attack",
                "weight": 30
            }
        ]
    },

    "Counter Attack": {
        "description": (
            "Quickly move the ball towards the opponent's goal "
            "after gaining possession."
        ),
        "signals": [
            {
                "name": "Ball moves quickly towards the opponent's goal",
                "weight": 40
            },
            {
                "name": "Players move forward in support",
                "weight": 30
            },
            {
                "name": "Attack progresses into the opponent's half",
                "weight": 30
            }
        ]
    },

    "Through Ball": {
        "description": (
            "Move the ball into open space behind the defensive line "
            "for a forward player to receive."
        ),
        "signals": [
            {
                "name": "Ball moves forward into open space",
                "weight": 40
            },
            {
                "name": "A teammate moves towards the forward space",
                "weight": 30
            },
            {
                "name": "Ball and teammate move towards the attacking area",
                "weight": 30
            }
        ]
    },

    "Crossing and Finishing": {
        "description": (
            "Move the ball into a wide attacking area and deliver it "
            "towards the central goal area."
        ),
        "signals": [
            {
                "name": "Ball reaches a wide attacking area",
                "weight": 30
            },
            {
                "name": "Ball moves towards the central attacking area",
                "weight": 40
            },
            {
                "name": "Teammates occupy the attacking area",
                "weight": 30
            }
        ]
    }
}


ATTACK_DIRECTIONS = {
    "Left to Right": "right",
    "Right to Left": "left"
}


def get_strategy_names():
    """Return the available strategy names."""
    return list(STRATEGIES.keys())


def get_strategy_description(strategy_name):
    """Return the description of a selected strategy."""
    strategy = STRATEGIES.get(strategy_name)

    if strategy is None:
        return "Strategy not found."

    return strategy["description"]

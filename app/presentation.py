"""Small user-facing labels shared by Streamlit and the optional API."""
from __future__ import annotations


def chase_verdict(chasing_probability: float | None) -> str:
    """Describe which side is favoured without implying certainty or a score.

    The probability is the model's historical estimate for the chasing team.
    Neutral, side-explicit labels avoid telling the wrong fan to stop watching.
    """
    if chasing_probability is None:
        return "Not enough historical support"
    if not 0.0 <= float(chasing_probability) <= 1.0:
        return "Not enough historical support"
    if chasing_probability >= 0.65:
        return "Chasing side favoured"
    if chasing_probability >= 0.35:
        return "Close chase · keep an eye"
    return "Defending side favoured"

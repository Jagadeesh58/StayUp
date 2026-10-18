"""Strict parser for the compact live-chase entry, e.g. `112/3 13.2 t189`."""
from __future__ import annotations

import re
from typing import Any

from .cricket import CricketInputError, calculate_manual_state, parse_overs_notation

_QUICK = re.compile(r"^\s*(?P<score>\d{1,4})\s*/\s*(?P<wickets>\d{1,2})\s+(?P<overs>\d{1,2}(?:\.\d)?)\s+t(?P<target>\d{1,4})\s*$", re.IGNORECASE)

def parse_quick_entry(value: str) -> dict[str, Any]:
    """Parse an unambiguous score/wickets overs target expression.

    Over notation remains string-based. For example, `13.2` is 13 completed
    overs plus two legal balls, never a decimal float.
    """
    if not isinstance(value, str):
        raise CricketInputError("Enter a score such as `112/3 13.2 t189`.")
    match = _QUICK.match(value)
    if not match:
        raise CricketInputError("Use `score/wickets overs target`, for example `112/3 13.2 t189`.")
    score = int(match.group("score")); wickets = int(match.group("wickets")); target = int(match.group("target"))
    used = parse_overs_notation(match.group("overs"))
    state = calculate_manual_state(score, wickets, used // 6, used % 6, target)
    return {"score":score,"wickets_lost":wickets,"completed_overs":used//6,"balls_in_over":used%6,"target":target,"overs_notation":match.group("overs"),"state":state}

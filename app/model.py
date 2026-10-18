"""Leakage-aware, recency-weighted IPL win-probability lookup and similar-state cohort.

All probabilities are computed in Python from compact Cricsheet-derived observations.
The model is an interpretable, smoothed historical lookup, not a live feed or a
causal simulator. Replay queries are strictly limited to matches dated before the
replayed match and exclude that match explicitly.
"""
from __future__ import annotations
import gzip, json, math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .cricket import state_from_totals

DATA_DIR = Path(__file__).resolve().parent / "data"
OBSERVATIONS_FILE = DATA_DIR / "model_observations.json.gz"
MATCHES_FILE = DATA_DIR / "matches.json.gz"
REPLAY_FILE = DATA_DIR / "replay_catalog.json.gz"
MATCHUPS_FILE = DATA_DIR / "matchups.json.gz"
GROUND_FILE = DATA_DIR / "ground_rollup.json"
MANIFEST_FILE = DATA_DIR / "data_manifest.json"
HALF_LIFE_DAYS = 365.2425 * 6.0
RR_BUCKET_WIDTH = 2.0
SCORE_VS_PAR_BUCKET_WIDTH = 10.0
PRIOR_WEIGHT = 4.0
MIN_PROBABILITY_ESS = 8.0
MIN_COHORT_N = 30

class ModelDataError(RuntimeError):
    """Required derived cricket data is missing or invalid."""

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)

@lru_cache(maxsize=1)
def load_data() -> dict[str, Any]:
    missing = [str(p.name) for p in (OBSERVATIONS_FILE, MATCHES_FILE, REPLAY_FILE, MATCHUPS_FILE, MANIFEST_FILE) if not p.exists()]
    if missing:
        raise ModelDataError("Derived IPL data is missing: " + ", ".join(missing) + ". Run scripts/build_data.py with the Cricsheet archive.")
    try:
        data = {
            "observations": _load_gzip_json(OBSERVATIONS_FILE),
            "matches": _load_gzip_json(MATCHES_FILE),
            "replays": _load_gzip_json(REPLAY_FILE),
            "matchups": _load_gzip_json(MATCHUPS_FILE),
            "grounds": _load_json(GROUND_FILE) if GROUND_FILE.exists() else {},
            "manifest": _load_json(MANIFEST_FILE),
            "backtest": _load_json(DATA_DIR / "backtest_results.json") if (DATA_DIR / "backtest_results.json").exists() else {},
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelDataError("The derived IPL data files could not be loaded.") from exc
    if not data["observations"] or not data["matches"]:
        raise ModelDataError("The derived IPL data files are empty; rebuild them from the source archive.")
    return data

def _as_date(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return datetime.now(timezone.utc).date()

def _before_cutoff(row: dict[str, Any], as_of: date, max_training_date: date | None) -> bool:
    try:
        row_date = date.fromisoformat(str(row.get("date", ""))[:10])
    except ValueError:
        return False
    if row_date >= as_of:
        return False  # strict cutoff: never use same-day or future matches
    if max_training_date is not None and row_date > max_training_date:
        return False
    return True

def recency_weight(match_date: str, as_of: date | str, half_life_days: float = HALF_LIFE_DAYS) -> float:
    """Exponential decay weight: a match six years old contributes half as much as today's."""
    reference = _as_date(as_of)
    try:
        observed = date.fromisoformat(match_date[:10])
    except ValueError:
        return 0.0
    age = max(0, (reference - observed).days)
    return 0.5 ** (age / half_life_days)

def required_rate_bucket(required_rate: float) -> int:
    return max(0, min(12, int(max(0.0, required_rate) // RR_BUCKET_WIDTH)))

def score_vs_par_bucket(score: int, par_score: float) -> int:
    return max(-5, min(5, math.floor((score - par_score) / SCORE_VS_PAR_BUCKET_WIDTH)))

def effective_sample_size(weights: Iterable[float]) -> float:
    values = [w for w in weights if w > 0 and math.isfinite(w)]
    if not values:
        return 0.0
    total = sum(values)
    squares = sum(w * w for w in values)
    return total * total / squares if squares else 0.0

def _deduplicate_nearest(rows: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for distance, row in rows:
        mid = str(row.get("match_id", ""))
        old = best.get(mid)
        if old is None or distance < old[0]:
            best[mid] = (distance, row)
    return list(best.values())

def _aggregate_probability(rows: list[tuple[float, dict[str, Any]]], as_of: date) -> dict[str, Any]:
    weighted: list[tuple[float, dict[str, Any]]] = []
    for distance, row in rows:
        weight = recency_weight(str(row["date"]), as_of) * math.exp(-0.62 * distance)
        if weight > 0:
            weighted.append((weight, row))
    if not weighted:
        return {"probability": None, "effective_sample_size": 0, "raw_matches": 0, "weighted_matches": 0, "wins_weighted": 0.0}
    total = sum(w for w, _ in weighted)
    wins = sum(w for w, row in weighted if row.get("won") is True)
    # Bayesian smoothing: blend the weighted win rate with a 50 % uninformative
    # prior (strength = PRIOR_WEIGHT). This prevents extreme probabilities from
    # tiny cohorts while converging to the raw rate as evidence grows.
    probability = (wins + PRIOR_WEIGHT * 0.5) / (total + PRIOR_WEIGHT)
    ess = effective_sample_size(w for w, _ in weighted)
    return {"probability": round(min(1.0, max(0.0, probability)), 4), "effective_sample_size": round(ess, 1), "raw_matches": len(weighted), "weighted_matches": round(total, 2), "wins_weighted": round(wins, 2)}

@lru_cache(maxsize=2)
def _observation_index(innings: int) -> tuple[list[str], list[dict[str, Any]]]:
    """Sort one innings' observations once so temporal lookups use a date prefix."""
    rows = [r for r in load_data()["observations"] if int(r.get("innings", 0)) == innings and r.get("date")]
    rows.sort(key=lambda r: (str(r["date"]), str(r.get("match_id", "")), int(r.get("over", 0))))
    return [str(r["date"])[:10] for r in rows], rows

@lru_cache(maxsize=128)
def _prior_rows(innings: int, as_of: date, exclude_match_id: str | None, max_training_date: date | None = None) -> list[dict[str, Any]]:
    dates, indexed = _observation_index(innings)
    # ISO-8601 date strings sort lexically. bisect_left enforces a strict (< as_of)
    # cutoff and bisect_right keeps the optional training cutoff inclusive.
    stop = bisect_left(dates, as_of.isoformat())
    if max_training_date is not None:
        stop = min(stop, bisect_right(dates, max_training_date.isoformat()))
    prefix = indexed[:stop]
    if not exclude_match_id:
        return prefix
    exclude = str(exclude_match_id)
    return [row for row in prefix if str(row.get("match_id")) != exclude]

@lru_cache(maxsize=128)
def _prior_groups(innings: int, as_of: date, exclude_match_id: str | None, max_training_date: date | None = None) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Index the temporal prefix once; repeated checkpoint lookups avoid full scans."""
    by_over: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_rr: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _prior_rows(innings, as_of, exclude_match_id, max_training_date):
        by_over[int(row.get("over", 0))].append(row)
        if row.get("rr_bucket") is not None:
            by_rr[int(row["rr_bucket"])].append(row)
    return {"over": dict(by_over), "rr": dict(by_rr)}

def _nearby_over_rows(innings: int, over: int, radius: int, as_of: date, exclude_match_id: str | None, max_training_date: date | None) -> list[dict[str, Any]]:
    groups = _prior_groups(innings, as_of, exclude_match_id, max_training_date)["over"]
    return [row for ov in range(max(1, over-radius), min(19, over+radius)+1) for row in groups.get(ov, [])]

def _similar_chase_rows(rows: list[dict[str, Any]], over: int, wickets: int, rr_bucket: int) -> list[tuple[float, dict[str, Any]]]:
    candidates = []
    for row in rows:
        d_over = abs(int(row["over"]) - over)
        d_w = abs(int(row["wickets"]) - wickets)
        d_rr = abs(int(row.get("rr_bucket", 0)) - rr_bucket)
        if d_over > 2 or d_w > 2 or d_rr > 2:
            continue
        # Fixed weighting makes proximity explainable and stable across releases.
        # Wickets carry the highest weight (0.85) because they most sharply change
        # a chase's character; required-rate bucket (0.65) is next; over-phase (0.45) last.
        distance = 0.45 * d_over + 0.85 * d_w + 0.65 * d_rr
        candidates.append((distance, row))
    return _deduplicate_nearest(candidates)

CALIBRATION_TRAINING_CUTOFF = date(2023, 12, 31)

def apply_logistic_calibration(probability: float, params: dict[str, Any]) -> float:
    """Apply the persisted temporal Platt map to a probability in (0, 1)."""
    p = min(1.0 - 1e-9, max(1e-9, float(probability)))
    a = float(params.get("intercept", 0.0))
    b = float(params.get("slope", 1.0))
    logit = math.log(p / (1.0 - p))
    z = max(-30.0, min(30.0, a + b * logit))
    return min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-z))))

def _apply_runtime_calibration(result: dict[str, Any], model_kind: str, reference: date) -> dict[str, Any]:
    """Calibrate only for dates after the calibration training cutoff.

    Applying a calibrator trained through 2023 to an earlier replay would leak
    future outcomes into that replay. Older replay states therefore stay raw.
    """
    result = dict(result)
    raw = result.get("probability")
    if raw is None:
        result["calibration_status"] = "unavailable"
        return result
    result["raw_probability"] = raw
    data = load_data()
    backtest = data.get("backtest", {}) or {}
    calibrators = backtest.get("calibration_models", {}) or {}
    params = calibrators.get(model_kind)
    trained_through = str((params or {}).get("training_cutoff_inclusive", ""))
    if not params:
        result["calibration_status"] = "uncalibrated"
        return result
    try:
        cutoff = date.fromisoformat(trained_through)
    except ValueError:
        result["calibration_status"] = "uncalibrated_invalid_metadata"
        return result
    if reference <= cutoff:
        result["calibration_status"] = "raw_for_temporal_integrity"
        return result
    result["probability"] = round(apply_logistic_calibration(float(raw), params), 4)
    result["calibration_status"] = "temporal_platt"
    result["calibration_training_cutoff"] = trained_through
    return result

def lookup_chase(score: int, wickets_lost: int, legal_balls_used: int, target: int, *, as_of: str | date | None = None, exclude_match_id: str | None = None, max_training_date: str | date | None = None, calibrate: bool = True) -> dict[str, Any]:
    """Estimate chasing side win chance from prior over-end chase states."""
    reference = _as_date(as_of)
    cutoff = _as_date(max_training_date) if max_training_date is not None else None
    if score >= target or wickets_lost >= 10 or legal_balls_used >= 120:
        p = 1.0 if score >= target and wickets_lost < 10 else (0.0 if wickets_lost >= 10 or legal_balls_used >= 120 else None)
        return {"probability": p, "effective_sample_size": 0, "raw_matches": 0, "bucket": None, "source": "deterministic_terminal_state", "note": "The match state is terminal; this is not a historical model estimate."}
    balls_remaining = 120 - legal_balls_used
    if legal_balls_used < 0 or balls_remaining <= 0 or target < 1:
        return {"probability": None, "effective_sample_size": 0, "raw_matches": 0, "bucket": None, "source": "invalid_state", "note": "No valid model estimate exists for this state."}
    required_rate = max(0, target - score) * 6 / balls_remaining
    over = max(1, min(19, int(math.floor(legal_balls_used / 6 + 0.5))))
    rr_bucket = required_rate_bucket(required_rate)
    rows = _nearby_over_rows(2, over, 2, reference, exclude_match_id, cutoff)
    # Prefer the strict frozen similarity rule (the same rule used for K of N).
    # The discretized model key remains explicit; the numeric-rate check prevents
    # adjacent buckets from dominating when their actual rates are not similar.
    strict: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if abs(int(row["over"]) - over) > 1 or abs(int(row["wickets"]) - min(9, max(0, wickets_lost))) > 1:
            continue
        if abs(float(row["required_rate"]) - required_rate) > 1.0:
            continue
        if abs(int(row.get("rr_bucket", 0)) - rr_bucket) > 1:
            continue
        distance = abs(int(row["over"])-over)/3 + abs(int(row["wickets"])-wickets_lost) + abs(float(row["required_rate"])-required_rate)
        strict.append((distance,row))
    candidates = _deduplicate_nearest(strict)
    match_rule = "strict_frozen_similarity"
    if len(candidates) < MIN_PROBABILITY_ESS:
        # Back off only when the strict cohort is too sparse; the response clearly
        # labels this as a low-support estimate rather than claiming a big sample.
        candidates = _similar_chase_rows(rows, over, min(9, max(0, wickets_lost)), rr_bucket)
        match_rule = "widened_lookup_low_support"
    result = _aggregate_probability(candidates, reference)
    n = result["effective_sample_size"]
    if n < MIN_PROBABILITY_ESS:
        result["probability"] = None
        note = "Not enough prior similar situations to publish a probability safely."
    elif n < 30 or match_rule != "strict_frozen_similarity":
        note = "Limited historical support; treat this as a rough reference, not a prediction guarantee."
    else:
        note = "Smoothed historical estimate from prior matches; not a live feed or a guarantee."
    result.update({"source": "historical_lookup", "match_rule": match_rule, "over_key": over, "wickets_key": wickets_lost, "required_rate": round(required_rate, 2), "rr_bucket": rr_bucket, "note": note, "cutoff_exclusive": reference.isoformat(), "excluded_match_id": exclude_match_id})
    if calibrate and result.get("probability") is not None:
        result = _apply_runtime_calibration(result, "chase", reference)
    return result

def _par_curve(rows: list[dict[str, Any]]) -> dict[int, float]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        buckets[int(row["over"])].append(int(row["score"]))
    return {over: sum(scores) / len(scores) for over, scores in buckets.items() if scores}

@lru_cache(maxsize=512)
def _par_curve_for_key(as_of: date, exclude_match_id: str | None, max_training_date: date | None) -> dict[int, float]:
    return _par_curve(_prior_rows(1, as_of, exclude_match_id, max_training_date))

def lookup_first_innings(score: int, wickets_lost: int, legal_balls_used: int, *, as_of: str | date | None = None, exclude_match_id: str | None = None, max_training_date: str | date | None = None, calibrate: bool = True) -> dict[str, Any]:
    """Estimate the batting-first team's win chance using score-versus-par state."""
    reference = _as_date(as_of)
    cutoff = _as_date(max_training_date) if max_training_date is not None else None
    over = max(1, min(19, int(math.floor(legal_balls_used / 6 + 0.5))))
    par = _par_curve_for_key(reference, exclude_match_id, cutoff)
    rows = _nearby_over_rows(1, over, 2, reference, exclude_match_id, cutoff)
    if over not in par:
        return {"probability": None, "effective_sample_size": 0, "raw_matches": 0, "source": "insufficient_history", "note": "No prior score-at-over reference is available."}
    state_bucket = score_vs_par_bucket(score, par[over])
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        row_over = int(row["over"])
        if row_over not in par:
            continue
        d_over = abs(row_over - over)
        d_w = abs(int(row["wickets"]) - int(wickets_lost))
        row_bucket = score_vs_par_bucket(int(row["score"]), par[row_over])
        d_par = abs(row_bucket - state_bucket)
        if d_over > 2 or d_w > 2 or d_par > 2:
            continue
        candidates.append((0.5 * d_over + 0.8 * d_w + 0.7 * d_par, row))
    candidates = _deduplicate_nearest(candidates)
    result = _aggregate_probability(candidates, reference)
    n = result["effective_sample_size"]
    if n < MIN_PROBABILITY_ESS:
        result["probability"] = None
        note = "Not enough prior first-innings situations to publish a probability safely."
    elif n < 30:
        note = "Limited historical support; treat this as a rough reference, not a prediction guarantee."
    else:
        note = "Smoothed historical estimate from prior matches; not a prediction guarantee."
    result.update({"source": "historical_lookup", "over_key": over, "score_vs_par": round(score-par[over], 1), "score_vs_par_bucket": state_bucket, "par_score": round(par[over], 1), "note": note, "cutoff_exclusive": reference.isoformat(), "excluded_match_id": exclude_match_id})
    if calibrate and result.get("probability") is not None:
        result = _apply_runtime_calibration(result, "first_innings", reference)
    return result

def lookup_required_rate_baseline(required_rate: float, *, as_of: str | date | None = None, max_training_date: str | date | None = None) -> dict[str, Any]:
    """Baseline using only required-rate buckets, with the same prior and decay."""
    reference = _as_date(as_of)
    cutoff = _as_date(max_training_date) if max_training_date is not None else None
    bucket = required_rate_bucket(required_rate)
    grouped = _prior_groups(2, reference, None, cutoff)["rr"]
    rows = [row for b in range(max(0,bucket-2), min(12,bucket+2)+1) for row in grouped.get(b, [])]
    candidates = [(abs(int(row.get("rr_bucket", 0))-bucket), row) for row in rows]
    candidates = _deduplicate_nearest(candidates)
    result = _aggregate_probability(candidates, reference)
    result.update({"source": "required_rate_only_baseline", "rr_bucket": bucket, "cutoff_exclusive": reference.isoformat()})
    return result

def _matching_cohort_rows(score: int, wickets_lost: int, legal_balls_used: int, target: int, *, as_of: str | date | None = None, exclude_match_id: str | None = None) -> tuple[date, float, list[dict[str, Any]]]:
    reference = _as_date(as_of)
    balls_left = max(0, 120 - legal_balls_used)
    if balls_left <= 0:
        return reference, 0.0, []
    rr = max(0, target - score) * 6 / balls_left
    rows = _prior_rows(2, reference, exclude_match_id)
    per_match: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in rows:
        if abs(int(row["balls_used"]) - legal_balls_used) > 6:
            continue
        if abs(int(row["wickets"]) - wickets_lost) > 1:
            continue
        if abs(float(row["required_rate"]) - rr) > 1.0:
            continue
        distance = abs(int(row["balls_used"]) - legal_balls_used) / 6 + abs(int(row["wickets"]) - wickets_lost) + abs(float(row["required_rate"]) - rr)
        mid = str(row["match_id"])
        if mid not in per_match or distance < per_match[mid][0]:
            per_match[mid] = (distance, row)
    return reference, rr, [row for _, row in per_match.values()]

def next_over_outcomes(score: int, wickets_lost: int, legal_balls_used: int, target: int, *, as_of: str | date | None = None, exclude_match_id: str | None = None, min_n: int = MIN_COHORT_N) -> dict[str, Any]:
    """Summarize observed next-over outcomes in the frozen similar-state cohort."""
    reference, _rr, cohort = _matching_cohort_rows(score, wickets_lost, legal_balls_used, target, as_of=as_of, exclude_match_id=exclude_match_id)
    match_by_id = {m["match_id"]: m for m in all_matches()}
    outcomes = []
    for row in cohort:
        match = match_by_id.get(str(row["match_id"]))
        if not match:
            continue
        next_over = int(row["over"]) + 1
        summary = next((o for o in match["second_innings"]["over_summaries"] if int(o["over"]) == next_over), None)
        if summary is None:
            continue
        outcomes.append({"match_id": row["match_id"], "date": row["date"], "runs": int(summary["runs"]), "wickets": int(summary["wickets"]), "sixes": int(summary["sixes"]), "fours": int(summary["fours"]), "over": next_over})
    n = len(outcomes)
    if n < min_n:
        return {"available": False, "n": n, "min_n": min_n, "reason": f"Only {n} prior next-over observations met the frozen similarity rule; at least {min_n} are required.", "rule": {"wickets_±": 1, "legal_balls_±": 6, "required_rate_±": 1.0, "one_checkpoint_per_match": True, "cutoff_exclusive": reference.isoformat(), "exclude_match_id": exclude_match_id}}
    runs = sorted(o["runs"] for o in outcomes)

    def quantile(q: float) -> int:
        pos = (len(runs) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return runs[lo]
        return round(runs[lo] * (hi - pos) + runs[hi] * (pos - lo))

    buckets = {"0–3 runs": sum(0 <= r <= 3 for r in runs), "4–6 runs": sum(4 <= r <= 6 for r in runs), "7–9 runs": sum(7 <= r <= 9 for r in runs), "10+ runs": sum(r >= 10 for r in runs)}
    wicket_over = sum(1 for o in outcomes if o["wickets"] > 0)
    return {"available": True, "n": n, "min_n": min_n, "median_runs": quantile(0.5), "lower_quartile_runs": quantile(0.25), "upper_quartile_runs": quantile(0.75), "mean_runs": round(sum(runs) / n, 2), "wicket_over_rate": round(wicket_over / n, 3), "boundary_over_rate": round(sum(1 for o in outcomes if o["sixes"] or o["fours"]) / n, 3), "run_buckets": {k: {"count": v, "rate": round(v / n, 3)} for k, v in buckets.items()}, "outcomes": outcomes, "rule": {"wickets_±": 1, "legal_balls_±": 6, "required_rate_±": 1.0, "one_checkpoint_per_match": True, "cutoff_exclusive": reference.isoformat(), "exclude_match_id": exclude_match_id}, "note": "Observed next-over distribution from prior similar situations, not a prediction of the next over."}

def similar_chase_cohort(score: int, wickets_lost: int, legal_balls_used: int, target: int, *, as_of: str | date | None = None, exclude_match_id: str | None = None, min_n: int = MIN_COHORT_N) -> dict[str, Any]:
    """Return K/N only if the frozen matching and sample-size requirements pass."""
    reference = _as_date(as_of)
    if legal_balls_used >= 120 or score >= target or wickets_lost >= 10:
        return {"available": False, "reason": "The current state is terminal; no historical cohort is shown.", "k": None, "n": 0}
    _reference, _rr, chosen = _matching_cohort_rows(score, wickets_lost, legal_balls_used, target, as_of=reference, exclude_match_id=exclude_match_id)
    n = len(chosen)
    k = sum(1 for row in chosen if row.get("won") is True)
    result = {"available": n >= min_n, "k": k if n >= min_n else None, "n": n, "raw_k": k, "min_n": min_n, "rule": {"wickets_±": 1, "legal_balls_±": 6, "required_rate_±": 1.0, "one_per_match": True, "cutoff": f"strictly before {reference.isoformat()}", "excluded_match_id": exclude_match_id}}
    if n < min_n:
        result["reason"] = f"Only {n} distinct prior matches satisfy the frozen rule; at least {min_n} are required, so K of N is hidden."
    else:
        result["win_rate"] = round(k / n, 3)
        result["reason"] = "Descriptive frequency among distinct prior matches; not a personal prediction."
    return result

def replay_matches() -> list[dict[str, Any]]:
    return load_data()["replays"]

def all_matches() -> list[dict[str, Any]]:
    return load_data()["matches"]

def _team_key(name: str) -> str:
    return {"Delhi Daredevils": "Delhi Capitals", "Kings XI Punjab": "Punjab Kings", "Royal Challengers Bangalore": "Royal Challengers Bengaluru"}.get(name, name)

def matchup_grid(batting_team: str, bowling_team: str, *, as_of: str | date | None = None, exclude_match_id: str | None = None, batters: list[str] | None = None, bowlers: list[str] | None = None, limit_batters: int = 3, limit_bowlers: int = 4) -> dict[str, Any]:
    """Aggregate only eligible matchups strictly before the cutoff date."""
    reference = _as_date(as_of)
    batting_team, bowling_team = _team_key(batting_team), _team_key(bowling_team)
    eligible = [r for r in load_data()["matchups"] if r["batting_team"] == batting_team and r["bowling_team"] == bowling_team and _before_cutoff(r, reference, None) and str(r.get("match_id")) != str(exclude_match_id)]
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for r in eligible:
        key = (r["batter"], r["bowler"])
        g = grouped.setdefault(key, {"batter": r["batter"], "bowler": r["bowler"], "runs": 0, "balls": 0, "dismissals": 0, "credited_dismissals": 0, "match_ids": set()})
        for field in ("runs", "balls", "dismissals", "credited_dismissals"):
            g[field] += int(r.get(field, 0))
        g["match_ids"].add(r["match_id"])
    for g in grouped.values():
        g["matches"] = len(g.pop("match_ids"))
    rows = list(grouped.values())
    if batters is None:
        batters = sorted({r["batter"] for r in rows}, key=lambda name: (-sum(v["runs"] for v in rows if v["batter"] == name), name))[:limit_batters]
    else:
        batters = list(dict.fromkeys(batters))[:limit_batters]
    if bowlers is None:
        bowlers = sorted({r["bowler"] for r in rows}, key=lambda name: (-sum(v["balls"] for v in rows if v["bowler"] == name), name))[:limit_bowlers]
    else:
        bowlers = list(dict.fromkeys(bowlers))[:limit_bowlers]
    lookup = {(r["batter"], r["bowler"]): r for r in rows}
    cells = []
    for batter in batters:
        row = []
        for bowler in bowlers:
            item = lookup.get((batter, bowler))
            if item is None:
                row.append({"batter": batter, "bowler": bowler, "runs": 0, "balls": 0, "dismissals": 0, "credited_dismissals": 0, "matches": 0, "note": "No prior recorded matchup in this eligible IPL cohort."})
            else:
                n = int(item["balls"])
                row.append({**item, "note": "Small sample; read the ball count before interpreting." if n < 20 or item["matches"] < 3 else "Aggregated prior IPL record."})
        cells.append(row)
    return {"batting_team": batting_team, "bowling_team": bowling_team, "as_of_exclusive": reference.isoformat(), "batters": batters, "bowlers": bowlers, "cells": cells, "note": "Only eligible matches before the selected match date are included. Dismissals of a batter are distinct from bowler-credited dismissals."}

def pre_match_info(match_id: str) -> dict[str, Any] | None:
    found = next((m for m in load_data()["matches"] if m["match_id"] == str(match_id)), None)
    if not found:
        return None
    return {k: found.get(k) for k in ("match_id", "date", "season", "teams", "venue", "city", "toss", "pre_match", "first_innings_team", "second_innings_team")}

def match_index_public() -> list[dict[str, Any]]:
    # No winner, margin, final score or chase target is shown in the replay picker.
    return [{k:m.get(k) for k in ("match_id","date","season","teams","venue","city","first_innings_team","second_innings_team")} for m in sorted(replay_matches(), key=lambda m:(m["date"],m["match_id"]), reverse=True)]

def _win_probability_for_innings(match: dict[str, Any], innings_index: int, score: int, wickets: int, balls: int) -> dict[str, Any]:
    if innings_index == 0:
        return lookup_first_innings(score, wickets, balls, as_of=match["date"], exclude_match_id=match["match_id"])
    target = int(match["first_innings"]["score"]) + 1
    return lookup_chase(score, wickets, balls, target, as_of=match["date"], exclude_match_id=match["match_id"])

def swing_percentage_points(before: float | None, after: float | None) -> float | None:
    """Return a signed probability swing in percentage points, not percent change."""
    if before is None or after is None:
        return None
    return round((float(after) - float(before)) * 100.0, 1)


def partnership_segments(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Summarize partnerships using only the event prefix supplied by the caller.

    The caller is responsible for passing only spoiler-safe visible events. Runs
    on a wicket delivery belong to the outgoing partnership before it closes.
    """
    out: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    current: dict[int, dict[str, Any] | None] = {0: None, 1: None}

    def flush(ii: int, end_cursor: int) -> None:
        segment = current[ii]
        if segment is not None:
            out[ii].append({
                "partnership": len(out[ii]) + 1,
                "batters": list(segment["batters"]),
                "runs": int(segment["runs"]),
                "wickets": int(segment["wickets"]),
                "start_cursor": int(segment["start_cursor"]),
                "end_cursor": int(end_cursor),
            })
            current[ii] = None

    for ev in events:
        ii = int(ev["innings_index"])
        if ii not in (0, 1):
            continue
        pair = tuple(sorted((str(ev.get("batter", "")), str(ev.get("non_striker", "")))))
        seq = int(ev.get("sequence", 0))
        if current[ii] is not None and tuple(current[ii]["batters"]) != pair:
            flush(ii, seq - 1)
        if current[ii] is None:
            current[ii] = {"batters": pair, "runs": 0, "wickets": 0, "start_cursor": seq}
        segment = current[ii]
        segment["runs"] += int(ev.get("total_runs", 0))
        wickets = sum(1 for w in ev.get("wickets", []) if str(w.get("kind", "")).lower() != "retired hurt")
        segment["wickets"] += wickets
        if wickets:
            flush(ii, seq)
    last_seq_by_innings = {
        ii: max((int(ev.get("sequence", 0)) for ev in events if int(ev.get("innings_index", -1)) == ii), default=0)
        for ii in (0, 1)
    }
    for ii in (0, 1):
        if current[ii] is not None:
            flush(ii, last_seq_by_innings[ii])
    return out


def build_replay_view(match_id: str, cursor: int) -> dict[str, Any]:
    """Build charts, effects and evidence from the source-event prefix only."""
    match = next((m for m in replay_matches() if str(m["match_id"]) == str(match_id)), None)
    if match is None:
        raise KeyError("Unknown replay match.")
    events = match["events"]
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0 or cursor > len(events):
        raise ValueError(f"Cursor must be between 0 and {len(events)}.")
    visible = events[:cursor]
    first_event_count = sum(1 for e in events if int(e["innings_index"]) == 0)
    first_innings_complete = cursor >= first_event_count
    innings_runs = [0, 0]
    innings_wickets = [0, 0]
    innings_legal = [0, 0]
    batted: dict[int, set[str]] = {0: set(), 1: set()}
    partnerships: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    pair_now: dict[int, tuple[str, str] | None] = {0: None, 1: None}
    part_runs = {0: 0, 1: 0}
    part_wickets = {0: 0, 1: 0}
    part_start = {0: 1, 1: 1}
    over_groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None
    over_end_by_innings: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

    def close_partnership(ii: int, end_cursor: int) -> None:
        if pair_now[ii] is not None and part_runs[ii] > 0:
            partnerships[ii].append({
                "partnership": len(partnerships[ii]) + 1,
                "batters": list(pair_now[ii]),
                "runs": part_runs[ii],
                "wickets": part_wickets[ii],
                "start_cursor": part_start[ii],
                "end_cursor": end_cursor,
            })
        pair_now[ii] = None
        part_runs[ii] = 0
        part_wickets[ii] = 0
        part_start[ii] = end_cursor + 1

    for ev in visible:
        ii = int(ev["innings_index"])
        batter = str(ev["batter"])
        non = str(ev["non_striker"])
        batted[ii].add(batter)
        batted[ii].add(non)
        pair = tuple(sorted((batter, non)))
        if pair_now[ii] is None:
            pair_now[ii] = pair
        elif pair_now[ii] != pair:
            close_partnership(ii, int(ev["sequence"]) - 1)
            pair_now[ii] = pair
            part_start[ii] = int(ev["sequence"])
        total = int(ev["total_runs"])
        wickets_this = sum(1 for w in ev.get("wickets", []) if str(w.get("kind", "")).lower() not in {"retired hurt"})
        innings_runs[ii] += total
        innings_wickets[ii] += wickets_this
        innings_legal[ii] += int(bool(ev["legal_ball"]))
        part_runs[ii] += total
        part_wickets[ii] += wickets_this
        group_key = (ii, int(ev["over_index"]))
        if current_group is None or current_group["key"] != group_key:
            if current_group is not None:
                over_groups.append(current_group)
            current_group = {
                "key": group_key,
                "innings_index": ii,
                "over": int(ev["over_index"]) + 1,
                "events": [],
                "runs": 0,
                "wickets": 0,
                "sixes": 0,
                "fours": 0,
                "legal_balls": 0,
                "start_score": innings_runs[ii] - total,
                "start_wickets": max(0, innings_wickets[ii] - wickets_this),
                "start_balls": max(0, innings_legal[ii] - int(bool(ev["legal_ball"]))),
            }
        current_group["events"].append(ev)
        current_group["runs"] += total
        current_group["wickets"] += wickets_this
        current_group["sixes"] += int(int(ev["batter_runs"]) == 6)
        current_group["fours"] += int(int(ev["batter_runs"]) == 4)
        current_group["legal_balls"] += int(bool(ev["legal_ball"]))
        current_group["score"] = innings_runs[ii]
        current_group["wickets_lost"] = min(10, innings_wickets[ii])
        current_group["legal_balls_used"] = innings_legal[ii]
        # Close a partnership only after its wicket-delivery runs are counted.
        if wickets_this:
            close_partnership(ii, int(ev["sequence"]))
    if current_group is not None:
        over_groups.append(current_group)
    for ii in (0, 1):
        if pair_now[ii] is not None and part_runs[ii] > 0:
            partnerships[ii].append({
                "partnership": len(partnerships[ii]) + 1,
                "batters": list(pair_now[ii]),
                "runs": part_runs[ii],
                "wickets": part_wickets[ii],
                "start_cursor": part_start[ii],
                "end_cursor": cursor,
            })

    # Only emit a per-over point once six legal balls have been reached, an
    # innings terminal state is observed in the visible prefix, or the cursor
    # itself is the verified end of the source timeline.
    over_points = []
    for group in over_groups:
        ii = group["innings_index"]
        terminal = (ii == 1 and first_innings_complete and group["score"] >= int(match["first_innings"]["score"]) + 1) or group["wickets_lost"] >= 10 or group["legal_balls_used"] >= 120
        complete = group["legal_balls"] >= 6 or terminal or (cursor == len(events) and group is over_groups[-1])
        if not complete:
            continue
        after_prob = _win_probability_for_innings(match, ii, group["score"], group["wickets_lost"], group["legal_balls_used"])
        p_after = after_prob.get("probability")
        p_before = _win_probability_for_innings(match, ii, group["start_score"], group["start_wickets"], group["start_balls"]).get("probability")
        p_runs_only = _win_probability_for_innings(match, ii, group["score"], group["start_wickets"], group["legal_balls_used"]).get("probability")
        boundary_runs = 4 * group["fours"] + 6 * group["sixes"]
        p_without_boundaries = _win_probability_for_innings(match, ii, max(0, group["score"] - boundary_runs), group["wickets_lost"], group["legal_balls_used"]).get("probability")
        current_team = match["first_innings_team"] if ii == 0 else match["second_innings_team"]
        first_p = p_after if ii == 0 else (1 - p_after if p_after is not None else None)
        second_p = (1 - p_after if ii == 0 and p_after is not None else p_after if ii == 1 else None)
        over_event_ids = [e["source_id"] for e in group["events"]]
        item = {
            "innings_index": ii,
            "team": current_team,
            "over": group["over"],
            "score": group["score"],
            "wickets_lost": group["wickets_lost"],
            "legal_balls_used": group["legal_balls_used"],
            "runs_this_over": group["runs"],
            "wickets_this_over": group["wickets"],
            "sixes_this_over": group["sixes"],
            "fours_this_over": group["fours"],
            "recorded_deliveries": len(group["events"]),
            "cursor": group["events"][-1]["sequence"],
            "source_event_ids": over_event_ids,
            "batting_team_win_probability": p_after,
            "first_team_win_probability": first_p,
            "second_team_win_probability": second_p,
            "sample_size": after_prob.get("effective_sample_size", 0),
            "probability_note": after_prob.get("note"),
            "effect": {
                "total_swing_pp": swing_percentage_points(p_before, p_after),
                "runs_only_swing_pp": swing_percentage_points(p_before, p_runs_only),
                "wicket_counterfactual_pp": swing_percentage_points(p_runs_only, p_after),
                "boundary_counterfactual_pp": swing_percentage_points(p_without_boundaries, p_after),
                "counterfactual_note": "Descriptive model counterfactuals; components overlap and are not causal attribution.",
            },
        }
        # Expose computed chase states only for a completed, already reached over.
        # These are the exact facts the optional AI explanation may select from.
        if ii == 1 and first_innings_complete:
            chase_target = int(match["first_innings"]["score"]) + 1
            item["before_state"] = state_from_totals(
                int(group["start_score"]), int(group["start_wickets"]),
                int(group["start_balls"]), chase_target,
            )
            item["after_state"] = state_from_totals(
                int(group["score"]), int(group["wickets_lost"]),
                int(group["legal_balls_used"]), chase_target,
            )
        over_points.append(item)
        over_end_by_innings[ii].append(item)

    # Dynamic turning points are selected only from over-end evidence already
    # visible at this cursor. They are descriptive, not claims about player intent.
    turning_points={}
    for ii in (0,1):
        candidates=[p for p in over_end_by_innings[ii] if p["effect"]["total_swing_pp"] is not None]
        candidates.sort(key=lambda p:abs(p["effect"]["total_swing_pp"]),reverse=True)
        turning_points[ii]=[{"innings_index":p["innings_index"],"over":p["over"],"team":p["team"],"swing_pp":p["effect"]["total_swing_pp"],"score":p["score"],"wickets_lost":p["wickets_lost"],"runs_this_over":p["runs_this_over"],"wickets_this_over":p["wickets_this_over"],"fours_this_over":p["fours_this_over"],"sixes_this_over":p["sixes_this_over"],"recorded_deliveries":p["recorded_deliveries"],"cursor":p["cursor"],"source_event_ids":p["source_event_ids"],**({"before_state":p["before_state"],"after_state":p["after_state"],"runs_added":p["runs_this_over"]} if "before_state" in p and "after_state" in p else {})} for p in candidates[:3]]

    active_ii=int(visible[-1]["innings_index"]) if visible else 0
    if visible:
        last=visible[-1]; state_after=last["state_after"]
        active_state={"innings_index":active_ii,"team":match["first_innings_team"] if active_ii==0 else match["second_innings_team"],"score":state_after["score"],"wickets_lost":state_after["wickets_lost"],"legal_balls_used":state_after["legal_balls_used"],"overs_display":f"{state_after['legal_balls_used']//6}.{state_after['legal_balls_used']%6}","event_sequence":cursor}
    else:
        active_state={"innings_index":0,"team":match["first_innings_team"],"score":0,"wickets_lost":0,"legal_balls_used":0,"overs_display":"0.0","event_sequence":0}
    players=match.get("players_by_team",{}) or {}; still_to_bat={}
    for ii,team in ((0,match["first_innings_team"]),(1,match["second_innings_team"])):
        roster=list(players.get(_team_key(team), players.get(team,[])))
        still_to_bat[team]=[player for player in roster if player not in batted[ii]]
    final_result=None
    if cursor==len(events):
        final_result={"winner":match["winner"],"first_innings":{"team":match["first_innings_team"],"score":match["first_innings"]["score"],"wickets":match["first_innings"]["wickets"]},"second_innings":{"team":match["second_innings_team"],"score":match["second_innings"]["score"],"wickets":match["second_innings"]["wickets"]},"margin":match["margin"]}
    first_score_revealed=first_innings_complete
    safe_meta={k:match.get(k) for k in ("match_id","date","season","teams","venue","city","first_innings_team","second_innings_team","source_file","data_version","source_url")}
    safe_meta["pre_match"]=match.get("pre_match")

    # The replay cohort is computed from the visible chase state and excludes
    # the replayed match plus every match on or after its date. No target-chase
    # cohort is surfaced during the first innings or before the chase begins.
    current_win_model = None
    cohort = {"available": False, "k": None, "n": 0, "reason": "K of N is withheld until the chase state is visible."}
    next_over = {"available": False, "reason": "Next-over outcomes are available only during an active chase."}
    if first_innings_complete and active_ii == 1:
        chase_target = int(match["first_innings"]["score"]) + 1
        current_win_model = lookup_chase(
            int(active_state["score"]), int(active_state["wickets_lost"]),
            int(active_state["legal_balls_used"]), chase_target,
            as_of=match["date"], exclude_match_id=match["match_id"],
        )
        if int(active_state["score"]) < chase_target and int(active_state["wickets_lost"]) < 10 and int(active_state["legal_balls_used"]) < 120:
            cohort = similar_chase_cohort(
                int(active_state["score"]), int(active_state["wickets_lost"]),
                int(active_state["legal_balls_used"]), chase_target,
                as_of=match["date"], exclude_match_id=match["match_id"],
            )
            if int(active_state["legal_balls_used"]) <= 114:
                next_over = next_over_outcomes(
                    int(active_state["score"]), int(active_state["wickets_lost"]),
                    int(active_state["legal_balls_used"]), chase_target,
                    as_of=match["date"], exclude_match_id=match["match_id"],
                )
        else:
            cohort = {"available": False, "k": None, "n": 0, "reason": "The reached chase state is terminal; K of N is withheld."}

    return {"match":safe_meta,"cursor":cursor,"total_events":len(events),"complete":cursor==len(events),"first_innings_complete":first_innings_complete,"target":int(match["first_innings"]["score"])+1 if first_score_revealed else None,"events":[dict(ev) for ev in visible],"current_state":active_state,"innings_progress":[{"innings_index":ii,"team":match["first_innings_team"] if ii==0 else match["second_innings_team"],"score":innings_runs[ii],"wickets_lost":min(10,innings_wickets[ii]),"legal_balls_used":innings_legal[ii],"overs_display":f"{innings_legal[ii]//6}.{innings_legal[ii]%6}"} for ii in (0,1)],"runs_by_over":over_points,"win_probability_chart":[{"innings_index":p["innings_index"],"over":p["over"],"first_team_probability":p["first_team_win_probability"],"second_team_probability":p["second_team_win_probability"],"sample_size":p["sample_size"]} for p in over_points],"turning_points":turning_points,"partnerships":partnership_segments(visible),"batters_still_to_bat":still_to_bat,"current_win_model":current_win_model,"cohort":cohort,"next_over_outcomes":next_over,"final_result":final_result,"source_attribution":{"dataset":"Cricsheet IPL JSON","license":"ODC-By 1.0","source_url":"https://cricsheet.org/downloads/ipl_json.zip"}}

#!/usr/bin/env python3
"""Build compact, reproducible StayUp Cricket datasets from a Cricsheet IPL ZIP.

The source archive is read in-place and is never copied into the distributable.
Only validated match summaries, model observations, selected replay events and
aggregated batter/bowler statistics are written.
"""
from __future__ import annotations
import argparse, collections, gzip, hashlib, json, re, zipfile
from datetime import date
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "2.0.0"
HERO_MATCH_ID = "1359487"
HALF_LIFE_YEARS = 6.0
RR_BUCKET_WIDTH = 2.0
MIN_COHORT_N = 30
LICENSE = "Open Data Commons Attribution License (ODC-By) 1.0"
LICENSE_URL = "https://cricsheet.org/licensing/"
SOURCE_URL = "https://cricsheet.org/downloads/ipl_json.zip"
NON_LEGAL = {"wides", "noballs"}
NON_DISMISSALS = {"retired hurt"}
NON_BOWLER_CREDIT = {"run out", "retired hurt", "retired out", "obstructing the field", "handled the ball", "hit the ball twice", "timed out"}
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

def team_key(name: str) -> str:
    return TEAM_ALIASES.get(name, name)

def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool): return default
    try: return int(value)
    except (TypeError, ValueError): return default

def has_miscounted_overs(raw: dict[str, Any]) -> bool:
    """Flag explicitly annotated miscounted-over matches for exclusion."""
    found = False
    def walk(value: Any) -> None:
        nonlocal found
        if isinstance(value, dict):
            for key, sub in value.items():
                if key.lower() == "miscounted_overs" and sub not in (None, False, 0, [], {}, ""):
                    found = True
                walk(sub)
        elif isinstance(value, list):
            for sub in value: walk(sub)
    walk(raw)
    return found


def has_dls(info: dict[str, Any]) -> bool:
    found: list[str] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, sub in value.items():
                if key.lower() in {"method", "dls", "dl_method"} and sub:
                    found.append(str(sub))
                walk(sub)
        elif isinstance(value, list):
            for sub in value: walk(sub)
    walk(info)
    return any(re.search(r"\b(D/?L|DLS|DUCKWORTH)\b", item, re.I) for item in found)

def wicket_count(delivery: dict[str, Any]) -> int:
    return sum(1 for w in delivery.get("wickets", []) if str(w.get("kind", "")).lower() not in NON_DISMISSALS)

def is_legal(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras", {}) or {}
    return not any(k in extras for k in NON_LEGAL)

def process_innings(inn: dict[str, Any], match_id: str, innings_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score = wickets = legal_balls = sixes = fours = 0
    over_summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    cumulative_over_scores: dict[int, dict[str, int]] = {}
    for over_position, over in enumerate(inn.get("overs", [])):
        over_idx = as_int(over.get("over"), over_position)
        over_runs = over_wickets = over_legal = over_sixes = over_fours = 0
        for delivery_position, delivery in enumerate(over.get("deliveries", [])):
            runs = delivery.get("runs", {}) or {}
            batter_runs = as_int(runs.get("batter"))
            total_runs = as_int(runs.get("total"))
            extras = {str(k): as_int(v) for k, v in (delivery.get("extras", {}) or {}).items()}
            wc = wicket_count(delivery)
            legal = is_legal(delivery)
            score += total_runs; wickets += wc; legal_balls += int(legal)
            over_runs += total_runs; over_wickets += wc; over_legal += int(legal)
            over_sixes += int(batter_runs == 6); over_fours += int(batter_runs == 4)
            sixes += int(batter_runs == 6); fours += int(batter_runs == 4)
            events.append({
                "sequence": len(events), "match_id": match_id, "innings_index": innings_index,
                "source_id": f"{match_id}.json#/innings/{innings_index}/overs/{over_position}/deliveries/{delivery_position}",
                "over_index": over_idx, "delivery_position": delivery_position,
                "source_delivery_label": str(delivery.get("actual_delivery", f"{over_idx}.{delivery_position+1}")),
                "batter": str(delivery.get("batter", "Unknown batter")),
                "bowler": str(delivery.get("bowler", "Unknown bowler")),
                "non_striker": str(delivery.get("non_striker", "Unknown non-striker")),
                "batter_runs": batter_runs, "total_runs": total_runs, "extras": extras,
                "wickets": [{"kind": str(w.get("kind", "unknown")), "player_out": str(w.get("player_out", "Unknown"))} for w in delivery.get("wickets", [])],
                "legal_ball": legal, "state_after": {"score": score, "wickets_lost": min(wickets, 10), "legal_balls_used": legal_balls},
            })
        cumulative_over_scores[over_idx] = {"score": score, "wickets": min(wickets, 10), "legal_balls": legal_balls}
        over_summaries.append({"over": over_idx + 1, "runs": over_runs, "wickets": over_wickets, "legal_balls": over_legal, "sixes": over_sixes, "fours": over_fours, "score": score, "wickets_lost": min(wickets, 10), "legal_balls_used": legal_balls})
    result = {"team": str(inn.get("team", "Unknown")), "score": score, "wickets": min(wickets, 10), "legal_balls": legal_balls, "sixes": sixes, "fours": fours, "overs_display": f"{legal_balls//6}.{legal_balls%6}", "over_summaries": over_summaries}
    return result, events

def match_record(file_name: str, raw: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    info = raw.get("info", {}) or {}
    inn = raw.get("innings", []) or []
    outcome = info.get("outcome", {}) or {}
    match_id = Path(file_name).stem
    reasons = []
    if not outcome.get("winner"): reasons.append("missing_winner")
    if len(inn) != 2: reasons.append("innings_not_two")
    if as_int(info.get("overs")) != 20: reasons.append("not_20_overs")
    if has_dls(info): reasons.append("DLS_or_D_L_method")
    if has_miscounted_overs(raw): reasons.append("miscounted_overs")
    dates = info.get("dates", []) or []
    match_date = str(dates[0])[:10] if dates else ""
    try: date.fromisoformat(match_date)
    except ValueError: reasons.append("missing_or_invalid_date")
    if reasons: return None, [], ",".join(reasons)
    if len(info.get("teams", [])) != 2: return None, [], "unexpected_team_count"
    innings_summaries, events_by_innings = [], []
    for idx, inning in enumerate(inn):
        summary, events = process_innings(inning, match_id, idx)
        innings_summaries.append(summary); events_by_innings.append(events)
    winner = str(outcome.get("winner"))
    first, second = innings_summaries
    # Reconcile participant, final runs, margin and completion state against the
    # exact delivery-derived innings. A source row cannot enter model cohorts if
    # those basic invariants contradict its result metadata.
    if winner not in {first["team"], second["team"]}:
        return None, [], "winner_not_a_playing_team"
    if first["legal_balls"] > 120 or second["legal_balls"] > 120:
        return None, [], "innings_exceeds_scheduled_balls"
    if first["legal_balls"] < 120 and first["wickets"] < 10:
        return None, [], "first_innings_incomplete"
    by = outcome.get("by", {}) or {}
    if winner == second["team"]:
        if not (second["score"] > first["score"]):
            return None, [], "winner_total_not_reconciled"
        if "wickets" in by and as_int(by.get("wickets"), -1) != 10 - second["wickets"]:
            return None, [], "winner_margin_not_reconciled"
        if "runs" in by and as_int(by.get("runs"), -1) != second["score"] - first["score"]:
            return None, [], "winner_margin_not_reconciled"
    else:
        if not (first["score"] > second["score"]):
            return None, [], "winner_total_not_reconciled"
        if "runs" in by and as_int(by.get("runs"), -1) != first["score"] - second["score"]:
            return None, [], "winner_margin_not_reconciled"
        if "wickets" in by and as_int(by.get("wickets"), -1) != 10 - second["wickets"]:
            return None, [], "winner_margin_not_reconciled"
        if second["legal_balls"] < 120 and second["wickets"] < 10:
            return None, [], "second_innings_incomplete"
    try: year = int(match_date[:4])
    except ValueError: year = None
    toss = info.get("toss", {}) or {}
    by = outcome.get("by", {}) or {}
    margin = {"type": "wickets", "value": as_int(by.get("wickets"))} if "wickets" in by else {"type": "runs", "value": as_int(by.get("runs"))}
    summary = {
        "match_id": match_id, "source_file": file_name, "data_version": (raw.get("meta", {}) or {}).get("data_version", "unknown"),
        "date": match_date, "year": year, "season": str(info.get("season", year or "unknown")),
        "teams": [str(t) for t in info.get("teams", [])], "first_innings_team": first["team"], "second_innings_team": second["team"],
        "venue": str(info.get("venue", "Unknown ground")), "city": str(info.get("city", "Unknown city")),
        "winner": winner, "margin": margin, "toss": {"winner": str(toss.get("winner", "Unknown")), "decision": str(toss.get("decision", "Unknown"))},
        "first_innings": first, "second_innings": second,
        "source_url": SOURCE_URL,
    }
    events = events_by_innings[0] + events_by_innings[1]
    for i, event in enumerate(events):
        event["sequence"] = i + 1
    return summary, events, None

def build_observations(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in matches:
        winner = m["winner"]
        first, chase = m["first_innings"], m["second_innings"]
        target = first["score"] + 1
        for inn_idx, inn in enumerate((first, chase)):
            if inn_idx == 0:
                for over in inn["over_summaries"]:
                    ov = over["over"]
                    if 1 <= ov <= 19 and over["wickets_lost"] < 10 and over["legal_balls_used"] == ov * 6:
                        rows.append({"match_id": m["match_id"], "date": m["date"], "year": m["year"], "innings": 1, "over": ov, "wickets": over["wickets_lost"], "score": over["score"], "won": winner == first["team"], "team": first["team"], "opponent": chase["team"]})
            else:
                for over in inn["over_summaries"]:
                    ov = over["over"]
                    # A chase that finished in the middle of an over isn't a valid
                    # over-end lookup observation. Completed overs only, 1..19.
                    if not (1 <= ov <= 19 and over["wickets_lost"] < 10 and over["legal_balls_used"] == ov * 6 and over["score"] < target):
                        continue
                    balls_remaining = (20 - ov) * 6
                    rr = ((target - over["score"]) * 6) / balls_remaining if balls_remaining else None
                    bucket = max(0, min(12, int(rr // RR_BUCKET_WIDTH))) if rr is not None else 12
                    rows.append({"match_id": m["match_id"], "date": m["date"], "year": m["year"], "innings": 2, "over": ov, "wickets": over["wickets_lost"], "score": over["score"], "target": target, "balls_used": over["legal_balls_used"], "balls_remaining": balls_remaining, "runs_required": target - over["score"], "required_rate": round(rr, 4) if rr is not None else None, "rr_bucket": bucket, "won": winner == chase["team"], "team": chase["team"], "opponent": first["team"]})
    return rows

def candidate_comeback_score(m: dict[str, Any]) -> float | None:
    chase = m["second_innings"]
    for over in chase["over_summaries"]:
        if over["over"] == 15:
            if over["score"] >= m["first_innings"]["score"] + 1 or over["wickets_lost"] >= 10: return None
            rr = (m["first_innings"]["score"] + 1 - over["score"]) / 5
            if m["winner"] == chase["team"] and rr >= 10:
                return min(rr, 24) + (10 - over["wickets_lost"]) * 0.3
    return None

def select_replays(matches: list[dict[str, Any]]) -> list[str]:
    by_id = {m["match_id"]: m for m in matches}
    chosen: list[str] = []

    def add_limited(items: list[dict[str, Any]], limit: int) -> None:
        added = 0
        for m in items:
            mid = m["match_id"]
            if mid in chosen:
                continue
            chosen.append(mid)
            added += 1
            if added >= limit or len(chosen) >= 15:
                break

    if HERO_MATCH_ID in by_id:
        chosen.append(HERO_MATCH_ID)

    # Use defined slots to ensure the catalog does not accidentally overfit to
    # the newest season just because it contains many close finishes.
    for year, limit in ((2026, 4), (2025, 3), (2024, 2)):
        season = [m for m in matches if m["year"] == year and (
            (m["margin"]["type"] == "runs" and m["margin"]["value"] <= 10)
            or (m["margin"]["type"] == "wickets" and m["margin"]["value"] <= 3)
        )]
        season.sort(key=lambda m: (m["margin"]["value"], m["date"], m["match_id"]))
        add_limited(season, limit)

    # Include up to three winning chases that still required at least ten per
    # over after 15 overs; this is a transparent selection rule, not a claim
    # that every such match was equally difficult.
    comebacks = [(candidate_comeback_score(m), m) for m in matches]
    comebacks = [(score, m) for score, m in comebacks if score is not None]
    comebacks.sort(key=lambda pair: (pair[0], pair[1]["date"]), reverse=True)
    add_limited([m for _, m in comebacks], 3)

    close_all = [m for m in matches if (
        (m["margin"]["type"] == "runs" and m["margin"]["value"] <= 10)
        or (m["margin"]["type"] == "wickets" and m["margin"]["value"] <= 3)
    )]
    close_all.sort(key=lambda m: (m["date"], m["margin"]["value"], m["match_id"]), reverse=True)
    add_limited(close_all, 15 - len(chosen))
    if len(chosen) < 12:
        add_limited(sorted(matches, key=lambda r: (r["date"], r["match_id"]), reverse=True), 12 - len(chosen))
    return chosen[:15]

def enrich_pregame(matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach pre-match context using only matches on strictly earlier dates.

    All fixtures on the same calendar date are enriched from the same frozen
    pre-date snapshot. Their results are applied only after every fixture for
    that date has been annotated, preventing same-day result leakage.
    """
    ratings: dict[str, float] = collections.defaultdict(lambda: 1500.0)
    forms: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=5))
    h2h: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    ground_records: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    pitch: dict[str, dict[str, float]] = collections.defaultdict(lambda: {"matches": 0, "first_runs": 0, "sixes": 0, "wickets": 0, "bat_first_wins": 0})
    ordered = sorted(matches, key=lambda m: (m["date"], m["match_id"]))
    by_date: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for m in ordered:
        by_date[m["date"]].append(m)

    for match_date in sorted(by_date):
        day_matches = by_date[match_date]
        # First capture all pre-match features from the frozen prior-date state.
        for m in day_matches:
            a, b = [team_key(x) for x in m["teams"]]
            hkey = tuple(sorted((a, b)))
            h = h2h[hkey]
            gr_a, gr_b = ground_records[(m["venue"], a)], ground_records[(m["venue"], b)]
            m["pre_match"] = {
                "team_ratings": {a: round(ratings[a], 1), b: round(ratings[b], 1)},
                "team_form": {a: list(forms[a]), b: list(forms[b])},
                "head_to_head": {"matches": h[0], "team_a": a, "team_a_wins": h[1], "team_b": b, "team_b_wins": h[2]},
                "ground_record": {a: {"matches": gr_a[0], "wins": gr_a[1]}, b: {"matches": gr_b[0], "wins": gr_b[1]}},
                "pitch_report": None,
            }
            ps = pitch[m["venue"]]
            try: year = int(m["date"][:4])
            except (TypeError, ValueError): year = 0
            if year >= 2022 and ps["matches"] > 0:
                n = ps["matches"]
                m["pre_match"]["pitch_report"] = {
                    "matches": int(n), "first_innings_average": round(ps["first_runs"] / n, 1),
                    "sixes_per_match": round(ps["sixes"] / n, 1),
                    "wickets_per_innings": round(ps["wickets"] / (2 * n), 2),
                    "bat_first_win_rate": round(ps["bat_first_wins"] / n, 3),
                    "note": "Inferred from earlier eligible matches at this ground since 2022; not a curator's report.",
                }

        # Compute each fixture's rating update from the same pre-date ratings.
        rating_deltas: dict[str, float] = collections.defaultdict(float)
        for m in day_matches:
            a, b = [team_key(x) for x in m["teams"]]
            ra, rb = ratings[a], ratings[b]
            expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            actual_a = 1.0 if team_key(m["winner"]) == a else 0.0
            delta = 20.0 * (actual_a - expected_a)
            rating_deltas[a] += delta; rating_deltas[b] -= delta
        for team, delta in rating_deltas.items(): ratings[team] += delta

        # Only now publish the day's results to form, H2H, venue and pitch history.
        for m in day_matches:
            a, b = [team_key(x) for x in m["teams"]]
            hkey = tuple(sorted((a, b))); h = h2h[hkey]
            res_a = "W" if team_key(m["winner"]) == a else "L"
            res_b = "W" if res_a == "L" else "L"
            forms[a].append({"date": match_date, "result": res_a, "opponent": b})
            forms[b].append({"date": match_date, "result": res_b, "opponent": a})
            h[0] += 1
            if res_a == "W": h[1] += 1
            else: h[2] += 1
            ground_records[(m["venue"], a)][0] += 1; ground_records[(m["venue"], b)][0] += 1
            if res_a == "W": ground_records[(m["venue"], a)][1] += 1
            else: ground_records[(m["venue"], b)][1] += 1
            try: year = int(match_date[:4])
            except (TypeError, ValueError): year = 0
            if year >= 2022:
                ps = pitch[m["venue"]]
                ps["matches"] += 1; ps["first_runs"] += m["first_innings"]["score"]
                ps["sixes"] += m["first_innings"]["sixes"] + m["second_innings"]["sixes"]
                ps["wickets"] += m["first_innings"]["wickets"] + m["second_innings"]["wickets"]
                if team_key(m["winner"]) == a: ps["bat_first_wins"] += 1

    ground_rollup = {}
    for venue, p in pitch.items():
        if p["matches"]:
            ground_rollup[venue] = {
                "matches": int(p["matches"]), "first_innings_average": round(p["first_runs"] / p["matches"], 1),
                "sixes_per_match": round(p["sixes"] / p["matches"], 1),
                "wickets_per_innings": round(p["wickets"] / (2 * p["matches"]), 2),
                "bat_first_win_rate": round(p["bat_first_wins"] / p["matches"], 3),
                "note": "Inferred from 2022+ eligible IPL results; not a curator's report.",
            }
    return ordered, ground_rollup

def build_matchups(all_rows: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    # Preserve match/date boundaries so a historical matchup lookup can apply a
    # strict pre-match cutoff instead of leaking later performances backwards.
    agg: dict[tuple[str,str,str,str,str,str], dict[str, Any]] = {}
    for filename, raw in all_rows:
        info = raw.get("info", {}) or {}
        if not info.get("outcome", {}).get("winner") or has_dls(info) or len(raw.get("innings", []) or []) != 2 or as_int(info.get("overs")) != 20:
            continue
        match_id = Path(filename).stem
        match_date = str((info.get("dates", []) or [""])[0])[:10]
        teams = info.get("teams", []) or []
        if len(teams) != 2: continue
        for inn in raw.get("innings", [])[:2]:
            bat_team = team_key(str(inn.get("team", "")))
            bowl_team = team_key(str(teams[1] if bat_team == team_key(str(teams[0])) else teams[0]))
            for over in inn.get("overs", []):
                for d in over.get("deliveries", []):
                    batter, bowler = str(d.get("batter", "")), str(d.get("bowler", ""))
                    if not batter or not bowler: continue
                    key = (match_id, match_date, bat_team, bowl_team, batter, bowler)
                    v = agg.setdefault(key, {"match_id":match_id,"date":match_date,"batting_team":bat_team,"bowling_team":bowl_team,"batter":batter,"bowler":bowler,"runs":0,"balls":0,"dismissals":0,"credited_dismissals":0})
                    runs = d.get("runs", {}) or {}; extras = d.get("extras", {}) or {}
                    v["runs"] += as_int(runs.get("batter"))
                    # A batter faces a no-ball but not a wide.
                    v["balls"] += int("wides" not in extras)
                    for w in d.get("wickets", []):
                        kind = str(w.get("kind", "")).lower()
                        if str(w.get("player_out", "")) == batter and kind not in NON_DISMISSALS:
                            v["dismissals"] += 1
                        if str(w.get("player_out", "")) == batter and kind not in NON_BOWLER_CREDIT:
                            v["credited_dismissals"] += 1
    result = list(agg.values())
    result.sort(key=lambda x: (x["date"], x["match_id"], x["batting_team"], x["bowling_team"], x["batter"], x["bowler"]))
    return result

def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_fh:
        with gzip.GzipFile(fileobj=raw_fh, mode="wb", mtime=0) as fh:
            fh.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Cricsheet IPL JSON ZIP; never copied into output")
    parser.add_argument("--output-dir", type=Path, default=Path("app/data"))
    args = parser.parse_args()
    if not args.input.is_file(): parser.error(f"Input archive not found: {args.input}")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    archive_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    matches, all_rows, rejections = [], [], collections.Counter()
    with zipfile.ZipFile(args.input) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".json") and not n.lower().endswith("readme.json")]
        for name in names:
            try:
                raw = json.loads(zf.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError):
                rejections["invalid_json"] += 1; continue
            all_rows.append((name, raw))
            summary, _events, reason = match_record(name, raw)
            if summary is None:
                rejections[reason or "unknown"] += 1; continue
            matches.append(summary)
    if not matches: raise RuntimeError("No eligible, reconciled matches were found.")
    matches, ground_rollup = enrich_pregame(matches)
    observations = build_observations(matches)
    replay_ids = select_replays(matches)
    by_id = {m["match_id"]: m for m in matches}
    replay_matches = []
    with zipfile.ZipFile(args.input) as zf:
        for mid in replay_ids:
            fname = f"{mid}.json"
            if fname not in zf.namelist():
                # Most archives use the source basename at their ZIP root. Handle
                # nested paths defensively while preserving its original basename.
                fname = next((n for n in zf.namelist() if Path(n).name == f"{mid}.json"), "")
            if not fname: continue
            raw = json.loads(zf.read(fname))
            m = dict(by_id[mid])
            innings_events = []
            for idx, inn in enumerate(raw.get("innings", [])[:2]):
                _summary, events = process_innings(inn, mid, idx)
                innings_events.append(events)
            events = innings_events[0] + innings_events[1]
            for seq, ev in enumerate(events, 1): ev["sequence"] = seq
            m["events"] = events
            # Roster is known pre-match, while batters-faced list is derived from
            # the cursor-safe event prefix by the API.
            players = raw.get("info", {}).get("players", {}) or {}
            m["players_by_team"] = {team_key(k): list(v) for k,v in players.items()}
            m["replay_label"] = f"{m['year']} · {m['first_innings_team']} vs {m['second_innings_team']}"
            m["has_full_delivery_timeline"] = True
            replay_matches.append(m)
    eligible_names = {m["source_file"] for m in matches}
    matchup_records = build_matchups([(name, raw) for name, raw in all_rows if name in eligible_names])
    # Store the upstream licence text in its own file, matching the archive's
    # ODC-By attribution terms. This is not the full raw match archive.
    license_text = None
    with zipfile.ZipFile(args.input) as zf:
        lic_name = next((n for n in zf.namelist() if Path(n).name.upper() == "LICENSE.TXT"), None)
        if lic_name: license_text = zf.read(lic_name)
    if license_text:
        (out / "LICENSE.txt").write_bytes(license_text)
    write_gzip_json(out / "matches.json.gz", matches)
    write_gzip_json(out / "model_observations.json.gz", observations)
    write_gzip_json(out / "replay_catalog.json.gz", replay_matches)
    write_gzip_json(out / "matchups.json.gz", matchup_records)
    (out / "ground_rollup.json").write_text(json.dumps(ground_rollup, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "generator_version": SCRIPT_VERSION, "source_archive_sha256": archive_hash,
        "source_archive_name": args.input.name, "source_url": SOURCE_URL, "license": LICENSE,
        "license_url": LICENSE_URL, "input_json_files": len(names), "eligible_reconciled_matches": len(matches),
        "replay_match_count": len(replay_matches), "model_observation_count": len(observations),
        "matchup_cell_count": len(matchup_records), "excluded_counts": dict(sorted(rejections.items())),
        "data_versions": dict(sorted(collections.Counter(m["data_version"] for m in matches).items())),
        "date_range": [min(m["date"] for m in matches), max(m["date"] for m in matches)],
        "replay_match_ids": [m["match_id"] for m in replay_matches], "cohort_minimum_n": MIN_COHORT_N,
        "cohort_matching_rule": {"wickets_abs_difference_max": 1, "legal_balls_used_abs_difference_max": 6, "required_rate_abs_difference_max": 1.0, "one_checkpoint_per_match": True, "date_cutoff_exclusive": True},
        "required_rate_bucket_width": RR_BUCKET_WIDTH, "recency_half_life_years": HALF_LIFE_YEARS,
        "note": "The raw source ZIP is not redistributed. Only reproducibly derived compact artifacts and the source licence are included."
    }
    (out / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Input JSON files: {len(names)}")
    print(f"Eligible and reconciled matches: {len(matches)}")
    print(f"Rejected: {sum(rejections.values())} ({dict(sorted(rejections.items()))})")
    print(f"Model observations: {len(observations)}")
    print(f"Replay catalog: {len(replay_matches)} matches")
    print(f"Matchup cells: {len(matchup_records)}")
    print(f"Date range: {manifest['date_range'][0]} to {manifest['date_range'][1]}")
    print(f"Source archive SHA256: {archive_hash}")
    print(f"Output: {out}")
    return 0

if __name__ == "__main__": raise SystemExit(main())

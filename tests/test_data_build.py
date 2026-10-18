import gzip
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _innings(team: str, score: int) -> dict:
    deliveries = []
    remaining = score
    for ball in range(120):
        # Keep the fixture simple and legal: single runs until target total is
        # met, with a zero-run ball when the total is below 120.
        if remaining <= 0:
            total = 0
        elif remaining >= (120 - ball) + 1:
            total = 2 if remaining == (120 - ball) + 1 else 1
        else:
            total = 1
        deliveries.append({
            "batter": f"{team} Batter 1", "bowler": f"{team} Opponent Bowler",
            "non_striker": f"{team} Batter 2", "runs": {"batter": total, "extras": 0, "total": total},
        })
        remaining -= total
    assert sum(d["runs"]["total"] for d in deliveries) == score
    return {"team": team, "overs": [{"over": i, "deliveries": deliveries[i * 6:(i + 1) * 6]} for i in range(20)]}


def _match(mid: str, day: str, first_score: int, second_score: int, winner: str) -> dict:
    first_team, second_team = "Alpha", "Beta"
    by = {"wickets": 10} if winner == second_team else {"runs": first_score - second_score}
    return {
        "meta": {"data_version": "1.2.0"},
        "info": {
            "dates": [day], "gender": "male", "match_type": "T20", "overs": 20,
            "season": int(day[:4]), "teams": [first_team, second_team],
            "outcome": {"winner": winner, "by": by},
            "players": {first_team: [f"{first_team} Batter 1", f"{first_team} Batter 2"], second_team: [f"{second_team} Batter 1", f"{second_team} Batter 2"]},
            "toss": {"winner": first_team, "decision": "bat"}, "venue": "Test Ground", "city": "Test City",
        },
        "innings": [_innings(first_team, first_score), _innings(second_team, second_score)],
    }


def test_builder_filters_and_reconciles_compact_data_and_pre_match_cutoff(tmp_path):
    archive = tmp_path / "synthetic_ipl.zip"
    output = tmp_path / "output"
    payloads = {
        "101.json": _match("101", "2020-01-01", 119, 120, "Beta"),
        "102.json": _match("102", "2020-01-01", 121, 120, "Alpha"),
        "103.json": _match("103", "2020-01-02", 119, 120, "Beta"),
    }
    dls = _match("104", "2020-01-03", 119, 120, "Beta")
    dls["info"]["outcome"]["method"] = "DLS"
    payloads["104.json"] = dls
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("LICENSE.txt", "Synthetic test license fixture; not for distribution.")
        for name, item in payloads.items():
            zf.writestr(name, json.dumps(item))
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_data.py"), "--input", str(archive), "--output-dir", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert "Eligible and reconciled matches: 3" in run.stdout
    manifest = json.loads((output / "data_manifest.json").read_text())
    assert manifest["eligible_reconciled_matches"] == 3
    assert manifest["excluded_counts"]["DLS_or_D_L_method"] == 1
    with gzip.open(output / "matches.json.gz", "rt", encoding="utf-8") as fh:
        matches = json.load(fh)
    by_id = {m["match_id"]: m for m in matches}
    # Same-day results do not leak into either fixture's pre-match H2H context.
    assert by_id["101"]["pre_match"]["head_to_head"]["matches"] == 0
    assert by_id["102"]["pre_match"]["head_to_head"]["matches"] == 0
    # The next calendar date sees both previous-day records.
    assert by_id["103"]["pre_match"]["head_to_head"]["matches"] == 2
    assert (output / "LICENSE.txt").exists()
    assert not (output / archive.name).exists()
    assert not any(p.name.endswith(".json") and p.name not in {"ground_rollup.json", "data_manifest.json"} for p in output.iterdir())

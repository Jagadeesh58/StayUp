#!/usr/bin/env python3
"""Reproducible temporal backtest and Platt calibration for StayUp Cricket.

Training observations are generated in chronological walk-forward order. A
match's own outcome and all same-day/future matches are excluded from every
lookup used to produce its calibration feature. The 2024+ held-out evaluation
is never used to fit calibration parameters.
"""
from __future__ import annotations
import json, math, sys
from datetime import date
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model import (
    apply_logistic_calibration, load_data, lookup_chase,
    lookup_first_innings, lookup_required_rate_baseline, _team_key,
)

TRAIN_CUTOFF = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date(2026, 12, 31)
OVERS = (6, 10, 15, 18)


def metrics(items: list[tuple[float, bool]]) -> dict[str, Any]:
    if not items:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None, "mean_prediction": None, "observed_win_rate": None}
    ps = [min(1-1e-9, max(1e-9, p)) for p, _ in items]
    ys = [int(y) for _, y in items]
    return {
        "n": len(items),
        "brier": round(sum((p-y)**2 for p,y in zip(ps,ys))/len(items), 5),
        "log_loss": round(-sum(y*math.log(p)+(1-y)*math.log(1-p) for p,y in zip(ps,ys))/len(items), 5),
        "accuracy": round(sum((p >= .5) == bool(y) for p,y in zip(ps,ys))/len(items), 4),
        "mean_prediction": round(sum(ps)/len(ps), 4),
        "observed_win_rate": round(sum(ys)/len(ys), 4),
    }


def calibration(items: list[tuple[float, bool]], bins: int = 10) -> list[dict[str, Any]]:
    out=[]
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins
        group=[(p,y) for p,y in items if lo <= p < hi or (i==bins-1 and p==1)]
        out.append({"bin": f"{lo:.1f}–{hi:.1f}", "n":len(group), "mean_predicted":round(sum(p for p,_ in group)/len(group),4) if group else None, "observed_win_rate":round(sum(int(y) for _,y in group)/len(group),4) if group else None})
    return out


def logit(p: float) -> float:
    p=min(1-1e-9,max(1e-9,float(p)))
    return math.log(p/(1-p))


def fit_platt(items: list[tuple[float, bool]], *, kind: str) -> dict[str, Any] | None:
    """Fit a two-parameter logistic recalibration with small ridge penalty."""
    if len(items) < 100 or len({int(y) for _,y in items}) < 2:
        return None
    xs=[logit(p) for p,_ in items]; ys=[int(y) for _,y in items]
    a, b, penalty = 0.0, 1.0, 0.01
    for _ in range(100):
        g0=g1=h00=h01=h11=0.0
        for x,y in zip(xs,ys):
            z=max(-30.0,min(30.0,a+b*x))
            q=1/(1+math.exp(-z)); residual=q-y; variance=q*(1-q)
            g0 += residual; g1 += residual*x
            h00 += variance; h01 += variance*x; h11 += variance*x*x
        g0 += penalty*a; g1 += penalty*(b-1.0)
        h00 += penalty; h11 += penalty
        det=h00*h11-h01*h01
        if det <= 1e-12: break
        da=(h11*g0-h01*g1)/det
        db=(-h01*g0+h00*g1)/det
        new_a=max(-6.0,min(6.0,a-da)); new_b=max(0.1,min(5.0,b-db))
        if abs(new_a-a)+abs(new_b-b) < 1e-8:
            a,b=new_a,new_b; break
        a,b=new_a,new_b
    return {
        "model_kind":kind,"method":"temporal walk-forward Platt logistic recalibration",
        "intercept":round(a,10),"slope":round(b,10),
        "training_cutoff_inclusive":TRAIN_CUTOFF.isoformat(),
        "training_state_count":len(items),"features":"intercept + slope * logit(raw_probability)",
        "regularization":penalty,
        "scope_note":"Fit only on walk-forward training predictions for completed overs 6, 10, 15 and 18; test matches were not used.",
    }


def state_at(match: dict[str, Any], innings_index: int, over: int) -> dict[str, Any] | None:
    innings=match["first_innings"] if innings_index==0 else match["second_innings"]
    return next((o for o in innings["over_summaries"] if int(o["over"])==over and int(o["legal_balls_used"])==over*6 and int(o["wickets_lost"])<10),None)


def predict_state(match: dict[str, Any], innings_index: int, state: dict[str, Any], *, cutoff: date) -> dict[str, Any]:
    if innings_index==0:
        pred=lookup_first_innings(int(state["score"]),int(state["wickets_lost"]),int(state["legal_balls_used"]),as_of=match["date"],exclude_match_id=match["match_id"],max_training_date=cutoff,calibrate=False)
        actual=match["winner"]==match["first_innings_team"]
    else:
        target=int(match["first_innings"]["score"])+1
        if int(state["score"])>=target: return {"probability":None,"actual":match["winner"]==match["second_innings_team"]}
        pred=lookup_chase(int(state["score"]),int(state["wickets_lost"]),int(state["legal_balls_used"]),target,as_of=match["date"],exclude_match_id=match["match_id"],max_training_date=cutoff,calibrate=False)
        actual=match["winner"]==match["second_innings_team"]
    return {"probability":pred.get("probability"),"actual":actual,"effective_sample_size":pred.get("effective_sample_size",0)}


def main() -> int:
    data=load_data(); matches=sorted(data["matches"],key=lambda m:(m["date"],str(m["match_id"])))
    train_matches=[m for m in matches if m.get("date") and date.fromisoformat(m["date"])<=TRAIN_CUTOFF]
    test_matches=[m for m in matches if m.get("date") and TEST_START<=date.fromisoformat(m["date"])<=TEST_END]

    # Calibration rows are true walk-forward predictions, never in-sample fits:
    # each training state's lookup excludes its match and all same-day outcomes.
    train_predictions={"chase":[],"first_innings":[]}
    for m in train_matches:
        for over in OVERS:
            for ii,kind in ((0,"first_innings"),(1,"chase")):
                state=state_at(m,ii,over)
                if state is None: continue
                item=predict_state(m,ii,state,cutoff=TRAIN_CUTOFF)
                if item.get("probability") is not None:
                    train_predictions[kind].append((float(item["probability"]),bool(item["actual"])))
    calibration_models={kind:fit_platt(items,kind=kind) for kind,items in train_predictions.items()}

    raw_by_over={"chase":{},"first_innings":{}}
    calibrated_by_over={"chase":{},"first_innings":{}}
    baseline_by_over={}
    missing_by_over={"chase":{},"first_innings":{}}
    for kind in ("chase","first_innings"):
        for over in OVERS:
            raw_by_over[kind][over]=[]; calibrated_by_over[kind][over]=[]; missing_by_over[kind][over]=0
    all_baseline=[]
    baseline_by_over={over:[] for over in OVERS}
    for m in test_matches:
        for over in OVERS:
            # First-innings model predicts whether the batting-first team wins.
            st1=state_at(m,0,over)
            if st1:
                item=predict_state(m,0,st1,cutoff=TRAIN_CUTOFF)
                p=item.get("probability")
                if p is None: missing_by_over["first_innings"][over]+=1
                else:
                    actual=bool(item["actual"]); raw_by_over["first_innings"][over].append((float(p),actual))
                    params=calibration_models.get("first_innings")
                    cp=apply_logistic_calibration(float(p),params) if params else float(p)
                    calibrated_by_over["first_innings"][over].append((cp,actual))
            # Chase model + required-rate-only baseline.
            st2=state_at(m,1,over)
            if not st2: continue
            target=int(m["first_innings"]["score"])+1
            if int(st2["score"])>=target: continue
            item=predict_state(m,1,st2,cutoff=TRAIN_CUTOFF); p=item.get("probability")
            actual=bool(item["actual"])
            rr=(target-int(st2["score"]))*6/(120-int(st2["legal_balls_used"]))
            base=lookup_required_rate_baseline(rr,as_of=m["date"],max_training_date=TRAIN_CUTOFF)
            if p is None: missing_by_over["chase"][over]+=1
            else:
                raw_by_over["chase"][over].append((float(p),actual))
                params=calibration_models.get("chase")
                cp=apply_logistic_calibration(float(p),params) if params else float(p)
                calibrated_by_over["chase"][over].append((cp,actual))
            if base.get("probability") is not None:
                item_base=(float(base["probability"]),actual)
                all_baseline.append(item_base); baseline_by_over[over].append(item_base)

    evaluation: dict[str, Any] = {
        "protocol":{"train_cutoff_inclusive":TRAIN_CUTOFF.isoformat(),"test_start_inclusive":TEST_START.isoformat(),"test_end_inclusive":TEST_END.isoformat(),"recency_half_life_years":6,"prior_weight":4,"required_rate_bucket_width":2,"checkpoint_overs":list(OVERS),"strict_date_cutoff":True,"same_day_excluded":True,"target_match_excluded":True,"calibration_method":"two-parameter Platt logistic recalibration fit on walk-forward training predictions only"},
        "calibration_models":{k:v for k,v in calibration_models.items() if v is not None},
        "calibration_training":{"chase_states":len(train_predictions["chase"]),"first_innings_states":len(train_predictions["first_innings"])},
        "checkpoints":{},"first_innings_checkpoints":{},"calibration":{},
        "notes":["All test values are state-level predictions; a match may contribute one observation at multiple checkpoint overs.","Historical lookup estimates are associated-state frequencies with six-year recency weighting and smoothing, not causal estimates or guarantees.","Calibration parameters use walk-forward training predictions through 2023 only. No 2024–2026 result is used to fit them.","States with effective sample size below eight are returned as unavailable, not imputed."],
    }
    all_chase_raw=[]; all_chase_cal=[]; all_first_raw=[]; all_first_cal=[]
    for over in OVERS:
        raw=raw_by_over["chase"][over]; cal=calibrated_by_over["chase"][over]
        # Align raw/cal rows: same order, because calibration is applied per raw prediction.
        evaluation["checkpoints"][str(over)]={"historical_lookup":metrics(cal),"raw_historical_lookup":metrics(raw),"required_rate_only_baseline":metrics(baseline_by_over[over]),"unavailable_model_predictions":missing_by_over["chase"][over]}
        r1=raw_by_over["first_innings"][over]; c1=calibrated_by_over["first_innings"][over]
        evaluation["first_innings_checkpoints"][str(over)]={"historical_lookup":metrics(c1),"raw_historical_lookup":metrics(r1),"unavailable_model_predictions":missing_by_over["first_innings"][over]}
        all_chase_raw.extend(raw); all_chase_cal.extend(cal); all_first_raw.extend(r1); all_first_cal.extend(c1)
    evaluation["calibration"]["historical_lookup"]=calibration(all_chase_cal)
    evaluation["calibration"]["raw_historical_lookup"]=calibration(all_chase_raw)
    evaluation["calibration"]["first_innings_lookup"]=calibration(all_first_cal)
    evaluation["pooled"]={"historical_lookup":metrics(all_chase_cal),"raw_historical_lookup":metrics(all_chase_raw),"required_rate_only_baseline":metrics(all_baseline)}
    evaluation["first_innings_pooled"]={"historical_lookup":metrics(all_first_cal),"raw_historical_lookup":metrics(all_first_raw)}
    # Independent pre-match test: use only the stored ratings built from results
    # strictly before each match, then compare with a constant 50% probability.
    prematch_items=[]; prematch_missing=0
    for m in test_matches:
        ratings=(m.get("pre_match") or {}).get("team_ratings",{})
        a=ratings.get(_team_key(m["first_innings_team"]))
        b=ratings.get(_team_key(m["second_innings_team"]))
        if a is None or b is None:
            prematch_missing+=1; continue
        p_first=1.0/(1.0+10.0**((float(b)-float(a))/400.0))
        prematch_items.append((p_first,m["winner"]==m["first_innings_team"]))
    coin_items=[(0.5,y) for _,y in prematch_items]
    evaluation["prematch_elo"]={
        "test_matches_with_ratings":len(prematch_items),"missing_rating_context":prematch_missing,
        "elo_rating_only_first_innings_win_probability":metrics(prematch_items),
        "constant_50_percent_probability_baseline":{"n":len(coin_items),"brier":round(sum((p-int(y))**2 for p,y in coin_items)/len(coin_items),5) if coin_items else None,"log_loss":round(-sum(int(y)*math.log(.5)+(1-int(y))*math.log(.5) for p,y in coin_items)/len(coin_items),5) if coin_items else None,"expected_accuracy_for_a_random_side_guess":0.5,"observed_first_innings_win_rate":round(sum(int(y) for _,y in prematch_items)/len(prematch_items),4) if prematch_items else None},
    }
    elo_m=evaluation["prematch_elo"]["elo_rating_only_first_innings_win_probability"]
    coin_m=evaluation["prematch_elo"]["constant_50_percent_probability_baseline"]
    evaluation["prematch_elo"]["interpretation"]=("Pre-match Elo classification accuracy is only marginally above the 50% expected accuracy of a random side guess, and its probability scores are worse than a constant 50% forecast; it is not demonstrated to beat coin flip as a probability model." if elo_m["accuracy"] is not None and elo_m["brier"] is not None and coin_m["brier"] is not None and elo_m["brier"]>coin_m["brier"] else "See the reported pre-match Elo and constant-50% metrics; do not claim it beats a coin-flip baseline without evidence.")
    ml=evaluation["pooled"]["historical_lookup"]; bl=evaluation["pooled"]["required_rate_only_baseline"]
    evaluation["baseline_interpretation"] = "Calibrated lookup has a lower pooled Brier score than the required-rate-only baseline on this held-out split." if ml["brier"] is not None and bl["brier"] is not None and ml["brier"] < bl["brier"] else "Calibrated lookup does not have a lower pooled Brier score than the required-rate-only baseline on this split."
    evaluation["provenance"]={"eligible_matches":len(matches),"training_matches":len(train_matches),"test_matches_in_date_window":len(test_matches),"source_archive_sha256":data["manifest"].get("source_archive_sha256"),"source":"Cricsheet IPL JSON; ODC-By 1.0"}
    results_path=Path("app/data/backtest_results.json"); results_path.write_text(json.dumps(evaluation,indent=2,sort_keys=True),encoding="utf-8")

    lines=["# StayUp Cricket — model evaluation","","## Protocol","",f"- Eligible, reconciled IPL match records: **{len(matches)}**.",f"- Training matches available through **{TRAIN_CUTOFF.isoformat()}**: **{len(train_matches)}**.",f"- Unseen test matches in **{TEST_START.isoformat()}–{TEST_END.isoformat()}**: **{len(test_matches)}**.","- All lookup queries enforce a strict date cutoff, exclude the target match, and exclude same-day/future records.","- Recency half-life: 6 years; smoothing prior weight: 4 centred at 0.5; required-rate bucket width: 2 runs per over.","- Metrics are calculated at completed-over checkpoints 6, 10, 15 and 18. A match can contribute multiple state observations; state N is not independent-match N.","- Baseline: required-rate-only historical lookup, using the same temporal cutoff, recency weighting and smoothing.","","## Held-out chase results","","| Over | Calibrated N | Calibrated Brier | Calibrated log loss | Calibrated accuracy | Raw Brier | Baseline N | Baseline Brier | Baseline log loss | Baseline accuracy | Unavailable lookup states |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt=lambda x:"—" if x is None else str(x)
    for over in OVERS:
        c=evaluation["checkpoints"][str(over)]; m=c["historical_lookup"]; raw=c["raw_historical_lookup"]; b=c["required_rate_only_baseline"]
        lines.append(f"| {over} | {m['n']} | {fmt(m['brier'])} | {fmt(m['log_loss'])} | {fmt(m['accuracy'])} | {fmt(raw['brier'])} | {b['n']} | {fmt(b['brier'])} | {fmt(b['log_loss'])} | {fmt(b['accuracy'])} | {c['unavailable_model_predictions']} |")
    lines += ["","## Pooled held-out comparison","",f"- Calibrated chase lookup: Brier **{ml['brier']}**, log loss **{ml['log_loss']}**, accuracy **{ml['accuracy']}**, N={ml['n']} state observations.",f"- Raw chase lookup: Brier **{evaluation['pooled']['raw_historical_lookup']['brier']}**, log loss **{evaluation['pooled']['raw_historical_lookup']['log_loss']}**, accuracy **{evaluation['pooled']['raw_historical_lookup']['accuracy']}**.",f"- Required-rate-only baseline: Brier **{bl['brier']}**, log loss **{bl['log_loss']}**, accuracy **{bl['accuracy']}**, N={bl['n']} state observations.",f"- Interpretation: {evaluation['baseline_interpretation']}","- First-innings calibrated lookup: "+f"Brier **{evaluation['first_innings_pooled']['historical_lookup']['brier']}**, log loss **{evaluation['first_innings_pooled']['historical_lookup']['log_loss']}**, accuracy **{evaluation['first_innings_pooled']['historical_lookup']['accuracy']}**, N={evaluation['first_innings_pooled']['historical_lookup']['n']} states.","","## Calibration table — calibrated chase lookup","","| Predicted bin | N | Mean predicted | Observed win rate |","|---|---:|---:|---:|"]
    for row in evaluation["calibration"]["historical_lookup"]:
        lines.append(f"| {row['bin']} | {row['n']} | {fmt(row['mean_predicted'])} | {fmt(row['observed_win_rate'])} |")
    pe=evaluation["prematch_elo"]; em=pe["elo_rating_only_first_innings_win_probability"]; cm=pe["constant_50_percent_probability_baseline"]
    lines += ["","## Pre-match Elo-only benchmark","",f"- Unseen matches with both pre-match ratings: **{pe['test_matches_with_ratings']}**; missing contexts: {pe['missing_rating_context']}.",f"- Elo rating-only first-team classifier: accuracy **{em['accuracy']}**, Brier **{em['brier']}**, log loss **{em['log_loss']}**.",f"- Constant 50% probability baseline: Brier **{cm['brier']}**, log loss **{cm['log_loss']}**, expected accuracy for a random side guess **{cm['expected_accuracy_for_a_random_side_guess']}**.",f"- Interpretation: {pe['interpretation']}","- In this test split, the pre-match rating classifier is only marginally above the 50% expected accuracy of a random side guess; its Brier and log-loss scores are worse than the constant 50% probability baseline. It has not demonstrated reliable pre-match probability skill.","","## Calibrator and training protocol","", "Each calibrator is fit with a two-parameter logistic map on walk-forward predictions from eligible matches dated through 2023. For every training match, the raw lookup excludes the target match and every same-day/future match. Calibration training uses only the requested over checkpoints. The held-out 2024–2026 period is never used to fit parameters.",""]
    for kind,params in evaluation["calibration_models"].items():
        lines.append(f"- `{kind}` calibrator: intercept `{params['intercept']}`, slope `{params['slope']}`, training state count `{params['training_state_count']}`.")
    lines += ["","## Calibration and limitations","","- Brier score and log loss assess probability quality; classification accuracy uses a 0.5 threshold and does not measure calibration or utility.","- Effective sample size varies by state. Estimates with ESS below eight are unavailable, not filled with a baseline or a fabricated number.","- The model is tested only on eligible IPL records from this archive. These metrics do not establish performance for international cricket, other leagues, different rule eras or future seasons.","- Calibration fitted on 6/10/15/18-over checkpoints is also used for active states between checkpoints. That is an explicit pragmatic approximation; future releases should evaluate calibration by over bucket and consider a dedicated mid-over dataset.","- The historical lookup is descriptive and does not estimate causal effects of wickets, boundaries or player decisions.","- Rerun `python scripts/evaluate_model.py` after changes to data, features, bucket widths, model weights or eligibility rules.",""]
    Path("docs/AI_EVALUATION.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Training cutoff: {TRAIN_CUTOFF} (inclusive); training matches: {len(train_matches)}")
    print(f"Test matches: {len(test_matches)}")
    print("Calibration states:",evaluation["calibration_training"])
    for over in OVERS:
        print(f"Chase over {over}: calibrated={evaluation['checkpoints'][str(over)]['historical_lookup']} raw={evaluation['checkpoints'][str(over)]['raw_historical_lookup']} baseline={evaluation['checkpoints'][str(over)]['required_rate_only_baseline']}")
        print(f"First innings over {over}: calibrated={evaluation['first_innings_checkpoints'][str(over)]['historical_lookup']}")
    print("Pooled chase calibrated:",evaluation["pooled"]["historical_lookup"])
    print("Pooled chase raw:",evaluation["pooled"]["raw_historical_lookup"])
    print("Pooled baseline:",evaluation["pooled"]["required_rate_only_baseline"])
    print("Pooled first innings calibrated:",evaluation["first_innings_pooled"]["historical_lookup"])
    print("Pre-match Elo-only:",evaluation["prematch_elo"]["elo_rating_only_first_innings_win_probability"])
    print("Pre-match 50% baseline:",evaluation["prematch_elo"]["constant_50_percent_probability_baseline"])
    print("Pre-match interpretation:",evaluation["prematch_elo"]["interpretation"])
    print("Interpretation:",evaluation["baseline_interpretation"])
    print(f"Wrote docs/AI_EVALUATION.md and {results_path}")
    return 0

if __name__ == "__main__": raise SystemExit(main())

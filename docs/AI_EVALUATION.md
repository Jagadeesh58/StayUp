# StayUp Cricket — model evaluation

## Protocol

- Eligible, reconciled IPL match records: **1172**.
- Training matches available through **2023-12-31**: **964**.
- Unseen test matches in **2024-01-01–2026-12-31**: **208**.
- All lookup queries enforce a strict date cutoff, exclude the target match, and exclude same-day/future records.
- Recency half-life: 6 years; smoothing prior weight: 4 centred at 0.5; required-rate bucket width: 2 runs per over.
- Metrics are calculated at completed-over checkpoints 6, 10, 15 and 18. A match can contribute multiple state observations; state N is not independent-match N.
- Baseline: required-rate-only historical lookup, using the same temporal cutoff, recency weighting and smoothing.

## Held-out chase results

| Over | Calibrated N | Calibrated Brier | Calibrated log loss | Calibrated accuracy | Raw Brier | Baseline N | Baseline Brier | Baseline log loss | Baseline accuracy | Unavailable lookup states |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 207 | 0.14462 | 0.44957 | 0.7729 | 0.15755 | 208 | 0.18843 | 0.56195 | 0.7885 | 1 |
| 10 | 201 | 0.13078 | 0.41388 | 0.8209 | 0.14344 | 204 | 0.17776 | 0.53703 | 0.8333 | 3 |
| 15 | 190 | 0.09267 | 0.31279 | 0.8947 | 0.10552 | 195 | 0.13598 | 0.43185 | 0.8564 | 5 |
| 18 | 150 | 0.07609 | 0.26056 | 0.9 | 0.09019 | 151 | 0.08982 | 0.29473 | 0.8874 | 1 |

## Pooled held-out comparison

- Calibrated chase lookup: Brier **0.11396**, log loss **0.36733**, accuracy **0.8422**, N=748 state observations.
- Raw chase lookup: Brier **0.12703**, log loss **0.41487**, accuracy **0.8422**.
- Required-rate-only baseline: Brier **0.15242**, log loss **0.46854**, accuracy **0.8377**, N=758 state observations.
- Interpretation: Calibrated lookup has a lower pooled Brier score than the required-rate-only baseline on this held-out split.
- First-innings calibrated lookup: Brier **0.22256**, log loss **0.63852**, accuracy **0.6465**, N=826 states.

## Calibration table — calibrated chase lookup

| Predicted bin | N | Mean predicted | Observed win rate |
|---|---:|---:|---:|
| 0.0–0.1 | 132 | 0.0617 | 0.053 |
| 0.1–0.2 | 123 | 0.1422 | 0.1545 |
| 0.2–0.3 | 74 | 0.2463 | 0.2432 |
| 0.3–0.4 | 51 | 0.3546 | 0.549 |
| 0.4–0.5 | 31 | 0.4497 | 0.6452 |
| 0.5–0.6 | 33 | 0.5493 | 0.7576 |
| 0.6–0.7 | 33 | 0.6474 | 0.8485 |
| 0.7–0.8 | 70 | 0.7572 | 0.9143 |
| 0.8–0.9 | 58 | 0.8496 | 0.8793 |
| 0.9–1.0 | 143 | 0.9516 | 1.0 |

## Pre-match Elo-only benchmark

- Unseen matches with both pre-match ratings: **208**; missing contexts: 0.
- Elo rating-only first-team classifier: accuracy **0.5096**, Brier **0.25779**, log loss **0.70929**.
- Constant 50% probability baseline: Brier **0.25**, log loss **0.69315**, expected accuracy for a random side guess **0.5**.
- Interpretation: Pre-match Elo classification accuracy is only marginally above the 50% expected accuracy of a random side guess, and its probability scores are worse than a constant 50% forecast; it is not demonstrated to beat coin flip as a probability model.
- In this test split, the pre-match rating classifier is only marginally above the 50% expected accuracy of a random side guess; its Brier and log-loss scores are worse than the constant 50% probability baseline. It has not demonstrated reliable pre-match probability skill.

## Calibrator and training protocol

Each calibrator is fit with a two-parameter logistic map on walk-forward predictions from eligible matches dated through 2023. For every training match, the raw lookup excludes the target match and every same-day/future match. Calibration training uses only the requested over checkpoints. The held-out 2024–2026 period is never used to fit parameters.

- `chase` calibrator: intercept `0.0763776087`, slope `1.6147122531`, training state count `3420`.
- `first_innings` calibrator: intercept `0.028918647`, slope `1.5213932006`, training state count `3773`.

## Calibration and limitations

- Brier score and log loss assess probability quality; classification accuracy uses a 0.5 threshold and does not measure calibration or utility.
- Effective sample size varies by state. Estimates with ESS below eight are unavailable, not filled with a baseline or a fabricated number.
- The model is tested only on eligible IPL records from this archive. These metrics do not establish performance for international cricket, other leagues, different rule eras or future seasons.
- Calibration fitted on 6/10/15/18-over checkpoints is also used for active states between checkpoints. That is an explicit pragmatic approximation; future releases should evaluate calibration by over bucket and consider a dedicated mid-over dataset.
- The historical lookup is descriptive and does not estimate causal effects of wickets, boundaries or player decisions.
- Rerun `python scripts/evaluate_model.py` after changes to data, features, bucket widths, model weights or eligibility rules.

## LLM & AI Service Evaluation

### 1. Model Configuration & Provider

- **Provider**: Groq (`https://api.groq.com/openai/v1/chat/completions`)
- **Pinned Model**: `qwen/qwen3.8-27b` (pinned directly in `app/ai_service.py`; environment overrides are ignored to ensure verified deterministic execution).
- **Sampling Temperature**: `0.0` (deterministic greedy decoding).
- **Max Output Tokens**: `2048`.
  - *Headroom rationale*: `qwen/qwen3.8-27b` outputs internal `<think>` reasoning tokens. At `max_tokens=1024`, complex chase states consumed significant token budget during thinking, creating a risk of truncating the structured JSON payload. Setting `max_tokens=2048` guarantees full thinking headroom while ensuring complete JSON output.
- **Strict Structured Outputs**: Forced via `response_format: {"type": "json_schema"}` constraining the response strictly to `{relationship_id, focus_id, selected_claim_ids, explanation}`.

### 2. Guardrails & Numeric Grounding Evaluation

Every LLM generation passes through a strict multi-tier validation pipeline before being rendered to the user:

1. **Strict Numeric Grounding**:
   - Every number in the generated response (both Arabic numerals and spelled-out number words in English, Hindi, Telugu, Tamil, Bengali, Marathi, and Hinglish) must be mathematically present in the server's pre-computed facts (`score`, `target`, `runs_required`, `balls_remaining`, `current_rate`, `required_rate`).
   - If any ungrounded number or invented statistic is detected, the response is rejected immediately (`AIProviderError`) and replaced by a deterministic server-calculated explanation.
2. **Causal & Speculation Filtering**:
   - The LLM is restricted to descriptive situational framing. Any speculative claims attributing psychological states or unmeasured causes (e.g. "nervous", "collapsed", "frightened", "panicked") are trapped and rejected.
3. **Sentence Count & Conciseness**:
   - Responses are strictly validated to 2–3 complete sentences.
4. **Multilingual Support**:
   - Supported across 7 languages (English, Hindi, Telugu, Tamil, Bengali, Marathi, Hinglish). Number words in each language (e.g. "seventy-seven", regional numeral forms, compound numerals like "one hundred and five") are parsed and validated against ground truth facts.

### 3. Rate Limiting, Safety & Budget Isolation

- **Session Isolation**: Client rate limiting is keyed on a cryptographically secure 128-bit random hex token (`session:{ai_session}`) generated per browser session. Client headers (`x-forwarded-for` and `host`) are never trusted, ensuring each visitor has an isolated budget slot and cannot exhaust other users' limits.
- **Daily Global Cap**: Default 200 calls/day (`AI_DAILY_BUDGET`), protecting the Groq API key from runaway consumption while providing ample capacity for reviewers and live visitors.
- **Circuit Breaker / Fallback**: When the budget is reached, or when the AI killswitch is engaged (`AI_ENABLED=false`), or if the provider times out, the app silently falls back to a deterministic, server-owned calculated explanation. The UI clearly labels when a fallback explanation is being displayed.
- **Cold-Start Handling**: SQLite state lives in `/tmp` on Streamlit Community Cloud; cold-starts cleanly reset local counts without breaking application availability.

### 4. Verification & Test Suite Results

- **Live Provider Verification**: Tested end-to-end via `python scripts/verify_ai.py` against Groq's live API with an uncached query. Result: `PASS` — valid structured JSON output matching all ground-truth facts.
- **Automated Test Suite**: 115 passing tests across unit, integration, and UI layers:
  - `tests/test_ai_service.py` (23 tests): Validates model pinning, schema conformity, cache deduplication, causal text rejection, and budget enforcement.
  - `tests/test_numeric_grounding.py` (10 tests): Validates multilingual digit/word extraction and rejection of ungrounded numbers.
  - `tests/test_adversarial.py`: Validates resistance to prompt injection and malicious score inputs.
  - `tests/test_streamlit_browser.py`: Validates headless Playwright smoke tests across desktop (1440px) and mobile (360px) viewports with zero console errors.

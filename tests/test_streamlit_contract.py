from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_replay_chart_uses_supported_altair_fixed_y_scale():
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "st.altair_chart(probability_chart, use_container_width=True)" in source
    assert "alt.Scale(domain=[0, 100], clamp=True)" in source
    assert "st.line_chart(pd.DataFrame(chart).set_index(\"Checkpoint\"),y_min=0,y_max=100)" not in source


def test_primary_navigation_prioritizes_live_chase_and_one_replay():
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'st.tabs(["Live chase", "Best replay", "More analysis"])' in source
    assert 'ids.index(hero_id) if hero_id in ids else 0' in source


def test_unvalidated_prematch_elo_forecast_is_not_shown_publicly():
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Pre-match rating-only signal" not in source
    assert "Pre-match Elo-style rating" not in source
    assert "Held-out pre-match Elo classification accuracy" not in source
    assert "Team form and ground history" in source
    assert "Pre-match Elo versus coin-flip reference" not in source


def test_runtime_ai_is_pinned_to_groq_hosted_qwen_and_has_no_model_override():
    source = (ROOT / "app/ai_service.py").read_text(encoding="utf-8")
    assert 'GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"' in source
    assert 'DEFAULT_MODEL = "qwen/qwen3.8-27b"' in source
    assert "https://api.openai.com" not in source
    assert "https://api.anthropic.com" not in source
    assert 'model = os.getenv("GROQ_MODEL"' not in source
    assert 'DEFAULT_MODEL = "qwen/qwen3.8-27b"' in source

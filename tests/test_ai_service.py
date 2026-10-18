import pytest
from app import ai_service
from app.ai_service import AIProviderError, BoundedCache, select_focus


def test_model_selection_is_validated_and_cached(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-not-a-real-key")
    cache = BoundedCache()
    monkeypatch.setattr(ai_service, "CACHE", cache)
    calls = {"count": 0, "budget": 0}

    def fake_provider_selection(**kwargs):
        calls["count"] += 1
        return {"relationship_id": "current_state_context", "focus_id": "runway", "selected_claim_ids": ["runs_required", "balls_remaining"], "explanation": "The remaining runway is the key constraint in this chase. Read the equation as a snapshot, not as a forecast."}

    monkeypatch.setattr(ai_service, "_provider_selection", fake_provider_selection)
    common = dict(
        context={"intent": "current_state_context", "claims": {"runs_required": 55, "balls_remaining": 24}},
        relationship_ids=["current_state_context"],
        focus_ids=["runway"],
        claim_ids=["runs_required", "balls_remaining", "wickets_remaining"],
        requirements={"runway": {"current_state_context": {"runs_required", "balls_remaining"}}},
        on_live_call=lambda: calls.__setitem__("budget", calls["budget"] + 1),
    )
    first = select_focus(**common)
    second = select_focus(**common)
    assert first.source == "live"
    assert second.source == "cached"
    assert calls["count"] == 1
    assert calls["budget"] == 1
    assert set(first.selected_claim_ids) == {"runs_required", "balls_remaining"}


def test_unknown_evidence_claim_is_rejected(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(ai_service, "CACHE", BoundedCache())
    monkeypatch.setattr(ai_service, "_provider_selection", lambda **kwargs: {
        "relationship_id": "current_state_context", "focus_id": "runway", "selected_claim_ids": ["invented_fact", "runs_required", "balls_remaining"],
        "explanation": "The remaining runway is the key constraint in this chase. Read the equation as a snapshot, not as a forecast."
    })
    with pytest.raises(AIProviderError, match="unknown evidence"):
        select_focus(
            context={"intent": "current_state_context", "unique_test": 2},
            relationship_ids=["current_state_context"], focus_ids=["runway"],
            claim_ids=["runs_required", "balls_remaining"],
            requirements={"runway": {"current_state_context": {"runs_required", "balls_remaining"}}},
            force_refresh=True,
        )


def test_relationship_focus_must_have_its_required_evidence(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(ai_service, "CACHE", BoundedCache())
    monkeypatch.setattr(ai_service, "_provider_selection", lambda **kwargs: {
        "relationship_id": "current_state_context", "focus_id": "runway", "selected_claim_ids": ["runs_required"], "explanation": "The remaining runway is the key constraint in this chase. Read the equation as a snapshot, not as a forecast."
    })
    with pytest.raises(AIProviderError, match="without all required evidence"):
        select_focus(
            context={"intent": "current_state_context", "unique_test": 3},
            relationship_ids=["current_state_context"], focus_ids=["runway"],
            claim_ids=["runs_required", "balls_remaining"],
            requirements={"runway": {"current_state_context": {"runs_required", "balls_remaining"}}},
            force_refresh=True,
        )


def test_provider_verification_is_specific_to_key_and_uses_pinned_groq_model(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-one")
    monkeypatch.setattr(ai_service, "CACHE", BoundedCache())
    monkeypatch.setattr(ai_service, "_provider_selection", lambda **kwargs: {
        "relationship_id": "current_state_context",
        "focus_id": "runway",
        "selected_claim_ids": ["runs_required", "balls_remaining"],
        "explanation": "The remaining runway is the key constraint in this chase. Read the equation as a snapshot, not as a forecast.",
    })
    common = dict(
        context={"intent": "current_state_context", "claims": {"runs_required": 12, "balls_remaining": 6}, "verification_case": "fingerprint"},
        relationship_ids=["current_state_context"], focus_ids=["runway"],
        claim_ids=["runs_required", "balls_remaining"],
        requirements={"runway": {"current_state_context": {"runs_required", "balls_remaining"}}},
        force_refresh=True,
    )
    select_focus(**common)
    assert ai_service.ai_configuration()["status"] == "verified"
    monkeypatch.setenv("GROQ_API_KEY", "test-key-two")
    changed_key = ai_service.ai_configuration()
    assert changed_key["status"] == "configured"
    assert changed_key["last_success_at"] is None
    # The app pins a single Groq-hosted model; environment overrides are ignored.
    monkeypatch.setenv("GROQ_API_KEY", "test-key-one")
    monkeypatch.setenv("GROQ_MODEL", "some-other-provider-model")
    pinned_model = ai_service.ai_configuration()
    assert pinned_model["model"] == ai_service.DEFAULT_MODEL == "qwen/qwen3.8-27b"
    assert pinned_model["status"] == "verified"


def test_unapproved_relationship_is_rejected(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-key")
    monkeypatch.setattr(ai_service, "CACHE", BoundedCache())
    monkeypatch.setattr(ai_service, "_provider_selection", lambda **kwargs: {
        "relationship_id": "historical_context",
        "focus_id": "rate_pressure",
        "selected_claim_ids": ["current_run_rate", "required_run_rate"],
        "explanation": "The required pace is different from the current rate. This describes the equation rather than a prediction.",
    })
    with pytest.raises(AIProviderError, match="relationship or focus"):
        select_focus(
            context={"case": "unapproved-relationship"},
            relationship_ids=["current_state_context"],
            focus_ids=["rate_pressure"],
            claim_ids=["current_run_rate", "required_run_rate"],
            requirements={"rate_pressure": {"current_state_context": {"current_run_rate", "required_run_rate"}}},
            force_refresh=True,
        )


def test_model_cannot_smuggle_free_text_causal_claim(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-key")
    monkeypatch.setattr(ai_service, "CACHE", BoundedCache())
    monkeypatch.setattr(ai_service, "_provider_selection", lambda **kwargs: {
        "relationship_id": "current_state_context",
        "focus_id": "rate_pressure",
        "selected_claim_ids": ["current_run_rate", "required_run_rate"],
        "explanation": "The batter was frightened by a tactic. That explains the outcome and the likely strategy.",
    })
    with pytest.raises(AIProviderError, match="causal or motivational"):

        select_focus(
            context={"case": "no-free-text"},
            relationship_ids=["current_state_context"],
            focus_ids=["rate_pressure"],
            claim_ids=["current_run_rate", "required_run_rate"],
            requirements={"rate_pressure": {"current_state_context": {"current_run_rate", "required_run_rate"}}},
            force_refresh=True,
        )


def test_terminal_chase_template_supports_variable_delivery_count_in_every_language():
    from app.localization import LANGUAGES, localized_explanation

    context = {
        "intent": "explain_chase_change",
        "card": {
            "before_state": {"runs_required": 3, "balls_remaining": 1},
            "after_state": {"score": 201, "status": "target_reached"},
        },
        "claims": {
            "before_runs_required": 3,
            "before_balls_remaining": 1,
            "runs_added": 4,
            "recorded_deliveries": 1,
            "after_score": 201,
            "target": 201,
        },
    }
    for language in LANGUAGES:
        result = localized_explanation("explain_chase_change", "target_reached", context, language)
        assert isinstance(result["answer"], str) and result["answer"]
        assert "3" in result["answer"] or language in {"Hindi", "Telugu", "Tamil", "Kannada", "Bengali"}


def test_terminal_relationship_can_select_six_required_claims(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(ai_service, "CACHE", BoundedCache())
    claims = {
        "before_runs_required": 29, "before_balls_remaining": 6, "runs_added": 31,
        "recorded_deliveries": 6, "after_score": 207, "target": 205,
    }
    explanation = "The over's recorded runs took the score past the target in the supplied sequence. This describes the verified score change, not a claim about player intent or future outcomes."
    monkeypatch.setattr(ai_service, "_provider_selection", lambda **kwargs: {
        "relationship_id": "target_reached", "focus_id": "target_reached",
        "selected_claim_ids": list(claims), "explanation": explanation,
    })
    result = select_focus(
        context={"intent": "explain_chase_change", "claims": claims},
        relationship_ids=["target_reached"], focus_ids=["target_reached"],
        claim_ids=list(claims), requirements={"target_reached": {"target_reached": set(claims)}},
        force_refresh=True,
    )
    assert len(result.selected_claim_ids) == 6
    assert result.explanation == explanation


def test_provider_output_with_unsupported_written_number_is_rejected(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(ai_service, "CACHE", BoundedCache())
    claims = {"runs_required": 55, "balls_remaining": 24}
    monkeypatch.setattr(ai_service, "_provider_selection", lambda **kwargs: {
        "relationship_id": "current_state_context", "focus_id": "runway",
        "selected_claim_ids": ["runs_required", "balls_remaining"],
        "explanation": "The chase needs 55 runs from 24 balls. A ninety-five percent win chance is now probable.",
    })
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        select_focus(
            context={"intent": "current_state_context", "claims": claims},
            relationship_ids=["current_state_context"], focus_ids=["runway"],
            claim_ids=["runs_required", "balls_remaining"],
            requirements={"runway": {"current_state_context": {"runs_required", "balls_remaining"}}},
            force_refresh=True,
        )


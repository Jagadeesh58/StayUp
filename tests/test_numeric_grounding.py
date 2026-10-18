import pytest
from app.ai_service import AIProviderError, validate_numeric_grounding, validate_grounded_explanation


def test_number_check_accepts_only_numbers_in_trusted_facts():
    facts = {"score": 112, "runs_required": 77, "required_rate": 11.55, "label": "no live feed"}
    validate_numeric_grounding('"score": 112, "runs_required": 77, "rate": 11.55', facts)
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_numeric_grounding('"score": 112, "win chance": 93%', facts)


def test_number_check_rejects_extra_numbers_inside_json():
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_numeric_grounding('{"focus":"rate_pressure","unexpected":"2026"}', {"score": 112})


def test_generated_explanation_must_be_two_or_three_sentences_and_numeric_grounded():
    facts = {"runs_required": 77, "balls_remaining": 40}
    answer = "The chase still needs 77 runs from 40 balls. Those figures describe the current equation, not the result."
    assert validate_grounded_explanation(answer, facts) == answer
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation("The chase needs 77 runs from 40 balls. A 91 percent chance is now likely.", facts)
    with pytest.raises(AIProviderError, match="2–3 complete sentences"):
        validate_grounded_explanation("The chase still needs 77 runs from 40 balls.", facts)


def test_generated_explanation_rejects_unsupported_player_motivation():
    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation(
            "The batter was nervous because of the field setting. That explains the collapse.",
            {"runs_required": 77, "balls_remaining": 40},
        )


def test_rejects_spelled_out_numbers_in_english():
    facts = {"runs_required": 77, "balls_remaining": 40}
    # Hyphenated written number
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "The chase needs 77 runs from 40 balls. A ninety-five percent chance is now likely.",
            facts,
            language="English",
        )
    # Spaced written number
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "The chase needs 77 runs from 40 balls. A ninety five percent chance is now likely.",
            facts,
            language="English",
        )
    # Supported written numbers are accepted
    valid = "The chase needs seventy-seven runs from forty balls. That defines the equation."
    assert validate_grounded_explanation(valid, facts, language="English") == valid


def test_rejects_unsupported_spelled_out_numbers_in_every_supported_language():
    facts = {"runs_required": 77, "balls_remaining": 40}

    # Hinglish
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "Chase ko 77 runs chahiye 40 balls mein. Pichanve percent chance abhi dikh raha hai.",
            facts,
            language="Hinglish",
        )

    # Hindi
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "टीम को 40 गेंदों में 77 रन चाहिए। पंचानवे प्रतिशत संभावना अभी दिखती है।",
            facts,
            language="Hindi",
        )

    # Bengali
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "দলের 40 বলে 77 রান প্রয়োজন। পঁচানব্বই শতাংশ জয়ের সম্ভাবনা রয়েছে।",
            facts,
            language="Bengali",
        )

    # Telugu
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "జట్టుకు 40 బంతుల్లో 77 పరుగులు అవసరం. తొంభై ఐదు శాతం గెలిచే అవకాశం ఉంది.",
            facts,
            language="Telugu",
        )

    # Tamil
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "அணிக்கு 40 பந்துகளில் 77 ஓட்டங்கள் தேவை. தொண்ணூற்றைந்து சதவீதம் வெற்றி வாய்ப்பு உள்ளது.",
            facts,
            language="Tamil",
        )

    # Kannada
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "ತಂಡಕ್ಕೆ 40 ಎಸೆತಗಳಲ್ಲಿ 77 ರನ್ ಅಗತ್ಯವಿದೆ. ತೊಂಬತ್ತೈದು ಶೇಕಡಾ ಗೆಲ್ಲುವ ಅವಕಾಶವಿದೆ.",
            facts,
            language="Kannada",
        )


def test_accepts_supported_spelled_out_numbers_in_languages():
    facts = {"runs_required": 77, "balls_remaining": 40}
    # Hindi with valid supported number words (40 = चालीस, 77 = सतहत्तर)
    hi_valid = "टीम को चालीस गेंदों में सतहत्तर रन चाहिए। यह वर्तमान स्थिति का समीकरण है।"
    assert validate_grounded_explanation(hi_valid, facts, language="Hindi") == hi_valid

    # Bengali with valid supported number words (40 = চল্লিশ, 77 = সাতাত্তর)
    bn_valid = "দলের চল্লিশ বলে সাতাত্তর রান প্রয়োজন। এটি বর্তমান সমীকরণ প্রকাশ করে।"
    assert validate_grounded_explanation(bn_valid, facts, language="Bengali") == bn_valid


def test_rejects_unsupported_indic_script_numerals():
    facts = {"runs_required": 77, "balls_remaining": 40}

    # Devanagari numerals (९५ = 95 unsupported)
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "टीम को ४० गेंदों में ७७ रन चाहिए। जीत की संभावना ९५% मानी जा रही है।",
            facts,
            language="Hindi",
        )

    # Bengali numerals (৯৫ = 95 unsupported)
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "দলের ৪০ বলে ৭৭ রান দরকার। জয়ের সম্ভাবনা ৯৫% হিসেব করা হয়েছে।",
            facts,
            language="Bengali",
        )

    # Telugu numerals (౯౫ = 95 unsupported)
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "జట్టుకు ౪౦ బంతుల్లో ౭౭ పరుగులు కావాలి. గెలుపు అవకాశం ౯౫% గా ఉంది.",
            facts,
            language="Telugu",
        )

    # Tamil numerals (௯௫ = 95 unsupported)
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "அணிக்கு ௪௦ பந்துகளில் ௭௭ ஓட்டங்கள் தேவை. வெற்றி வாய்ப்பு ௯௫% என தெரிகிறது.",
            facts,
            language="Tamil",
        )

    # Kannada numerals (೯೫ = 95 unsupported)
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "ತಂಡಕ್ಕೆ ೪೦ ಎಸೆತಗಳಲ್ಲಿ ೭೭ ರನ್ ಬೇಕು. ಗೆಲುವಿನ ಅವಕಾಶ ೯೫% ಎಂದು ಕಾಣುತ್ತದೆ.",
            facts,
            language="Kannada",
        )


def test_accepts_supported_indic_script_numerals():
    facts = {"runs_required": 77, "balls_remaining": 40}
    # Devanagari numerals where 40 (४०) and 77 (७७) are in facts
    hi_valid = "टीम को ४० गेंदों में ७७ रन चाहिए। यह केवल वर्तमान स्थिति का विवरण है।"
    assert validate_grounded_explanation(hi_valid, facts, language="Hindi") == hi_valid


def test_rejects_unsupported_compound_number_regression():
    """Exact reviewer counterexample: facts contain 1, 100, and 5, but not 105.
    
    The sentence asserts a total of 105 runs ('one hundred and five'), which must
    be parsed as a single compound number and rejected when absent from supplied facts.
    """
    facts = {"score": 1, "target": 100, "wickets": 5}
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "The current total is one hundred and five runs, while five wickets remain. That describes the state.",
            facts,
            language="English",
        )


def test_accepts_supported_compound_number():
    facts = {"score": 105, "wickets": 5}
    valid = "The current total is one hundred and five runs, while five wickets remain. That describes the state."
    assert validate_grounded_explanation(valid, facts, language="English") == valid


def test_rejects_unsupported_compound_numbers_in_other_languages():
    facts = {"score": 1, "target": 100, "wickets": 5}
    # Hindi: 105 = एक सौ पाँच
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "टीम का कुल स्कोर एक सौ पाँच रन है, जबकि पाँच विकेट बाकी हैं। यह स्थिति का विवरण है।",
            facts,
            language="Hindi",
        )
    # Hinglish: 105 = ek sau paanch
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "Team ka score ek sau paanch runs hai aur paanch wickets baaki hain. Yeh match ka haal hai.",
            facts,
            language="Hinglish",
        )
    # Bengali: 105 = একশো পাঁচ
    with pytest.raises(AIProviderError, match="not present in the supplied facts"):
        validate_grounded_explanation(
            "দলের মোট একশো পাঁচ রান হয়েছে এবং পাঁচ উইকেট বাকি রয়েছে। এটি খেলার বর্তমান অবস্থা।",
            facts,
            language="Bengali",
        )


def test_accepts_supported_compound_numbers_in_other_languages():
    facts = {"score": 105, "wickets": 5}
    # Hindi
    hi_valid = "टीम का कुल स्कोर एक सौ पाँच रन है, जबकि पाँच विकेट बाकी हैं। यह स्थिति का विवरण है।"
    assert validate_grounded_explanation(hi_valid, facts, language="Hindi") == hi_valid
    # Hinglish
    hing_valid = "Team ka score ek sau paanch runs hai aur paanch wickets baaki hain. Yeh match ka haal hai."
    assert validate_grounded_explanation(hing_valid, facts, language="Hinglish") == hing_valid
    # Bengali
    bn_valid = "দলের মোট একশো পাঁচ রান হয়েছে এবং পাঁচ উইকেট বাকি রয়েছে। এটি খেলার বর্তমান অবস্থা।"
    assert validate_grounded_explanation(bn_valid, facts, language="Bengali") == bn_valid


def test_ordinary_word_so_is_not_treated_as_one_hundred():
    facts = {"runs_required": 77, "balls_remaining": 40}
    # "So far" and "so the chase" must not trigger an ungrounded 100
    text = "So far, the chase needs 77 runs from 40 legal balls. So the equation remains very challenging."
    assert validate_grounded_explanation(text, facts, language="English") == text


def test_hinglish_sau_still_extracts_one_hundred():
    from app.ai_service import _extract_written_numbers
    assert "100" in _extract_written_numbers("ek sau run chahiye", language="Hinglish")
    assert "100" not in _extract_written_numbers("so far so good", language="English")


def test_causal_guard_rejects_narrative_certainty_and_speculation():
    facts = {"runs_required": 77, "balls_remaining": 40}

    # Narrative / condition speculation
    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("The side collapsed after the pitch slowed down. That explains the current equation.", facts)

    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("The chase stalled when the opening pair got stuck against spin. The 77 runs remain needed.", facts)

    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("The bowling team bowled to a clear gameplan. That leaves 40 balls remaining in the chase.", facts)

    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("A batter suffered an injury during the middle overs. That affected the run rate.", facts)

    # Certainty claims
    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("They will definitely win from this current position. The victory is guaranteed.", facts)

    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("The defending side cannot lose this match from here. The 40 balls remaining are sufficient.", facts)

    # Fantasy / betting
    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("This player is a great Dream11 pick today. Fantasy points will be substantial.", facts)

    with pytest.raises(AIProviderError, match="causal or motivational"):
        validate_grounded_explanation("Placing a bet on this chase is recommended. The 77 runs needed are achievable.", facts)


def test_causal_guard_does_not_block_innocent_cricket_words():
    facts = {"runs_required": 77, "balls_remaining": 40}
    # "between" and "better" contain the substring "bet", but must not be falsely rejected
    valid = "The chase needs 77 runs from 40 balls remaining. A better required rate between overs will be needed."
    assert validate_grounded_explanation(valid, facts, language="English") == valid



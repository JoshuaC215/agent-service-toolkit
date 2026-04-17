from agents.llama_guard import SafetyAssessment, parse_llama_guard_output


def test_parse_safe():
    out = parse_llama_guard_output("safe")
    assert out.safety_assessment == SafetyAssessment.SAFE
    assert out.unsafe_categories == []


def test_parse_unsafe_valid_categories():
    out = parse_llama_guard_output("unsafe\nS1, S3")
    assert out.safety_assessment == SafetyAssessment.UNSAFE
    # Trailing periods in mapping are stripped in parser
    assert out.unsafe_categories == ["Violent Crimes", "Sex Crimes"]


def test_parse_unsafe_trims_and_maps_multiple():
    out = parse_llama_guard_output("unsafe\n S2 ,  S10 ")
    assert out.safety_assessment == SafetyAssessment.UNSAFE
    assert out.unsafe_categories == ["Non-Violent Crimes", "Hate"]


def test_parse_invalid_format_returns_error():
    out = parse_llama_guard_output("maybe\nS1")
    assert out.safety_assessment == SafetyAssessment.ERROR


def test_parse_unknown_category_returns_error():
    out = parse_llama_guard_output("unsafe\nS99")
    assert out.safety_assessment == SafetyAssessment.ERROR

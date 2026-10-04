from unittest.mock import patch

from pydantic import SecretStr

from agents.safeguard import Safeguard, SafetyAssessment


def test_safeguard_skipped_under_fake_model():
    with (
        patch("agents.safeguard.settings.USE_FAKE_MODEL", True),
        patch("agents.safeguard.settings.GROQ_API_KEY", SecretStr("test_key")),
        patch("agents.safeguard.get_model") as get_model,
    ):
        safeguard = Safeguard()

    get_model.assert_not_called()
    assert safeguard.model is None
    assert safeguard.invoke([]).safety_assessment == SafetyAssessment.SAFE

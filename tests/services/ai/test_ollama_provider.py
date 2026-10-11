"""
OllamaProvider tests. All use the _http_post injection point — no
real Ollama server required.
"""
import json

import pytest
from pydantic import ValidationError

from converge.core.exceptions import (
    AIProviderAPIError,
    AIProviderConfigError,
    AIProviderResponseError,
    AIProviderTimeoutError,
)
from converge.services.ai.ollama_provider import OllamaProvider
from converge.services.ai.prompts import PROMPT_VERSION


@pytest.fixture()
def sample_profile(sample_profile_result):
    return sample_profile_result


def _make_provider(http_response=None, model="test-model"):
    def fake_post(url, body):
        if callable(http_response):
            return http_response(url, body)
        return http_response
    return OllamaProvider(model_name=model, _http_post=fake_post)


def _valid_response(recs=None):
    if recs is None:
        recs = [{"operation": "TRIM_WHITESPACE", "target_columns": ["name"],
                 "parameters": {}, "reason": "test", "evidence_refs": []}]
    payload = {"recommendations": recs}
    return json.dumps({"message": {"content": json.dumps(payload)}})


def _raw_json_response(recs=None):
    """Return just the JSON string that _parse_recommendations expects."""
    if recs is None:
        recs = [{"operation": "TRIM_WHITESPACE", "target_columns": ["name"],
                 "parameters": {}, "reason": "test", "evidence_refs": []}]
    return json.dumps({"recommendations": recs})


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------

def test_provider_name_is_ollama():
    p = _make_provider(_raw_json_response())
    assert p.provider_name == "ollama"


def test_model_name_matches_constructor_arg():
    p = _make_provider(_raw_json_response(), model="phi3:mini")
    assert p.model_name == "phi3:mini"


def test_prompt_version_matches_shared_constant():
    p = _make_provider(_raw_json_response())
    assert p.prompt_version == PROMPT_VERSION


def test_empty_model_name_raises_config_error():
    with pytest.raises(AIProviderConfigError):
        OllamaProvider(model_name="")


def test_whitespace_only_model_name_raises_config_error():
    with pytest.raises(AIProviderConfigError):
        OllamaProvider(model_name="   ")


# ---------------------------------------------------------------------------
# Correct recommendation parsing
# ---------------------------------------------------------------------------

def test_valid_response_parsed_as_recommendations(sample_profile):
    provider = _make_provider(_raw_json_response())
    recs = provider.generate_recommendations(sample_profile)
    assert len(recs) == 1
    assert recs[0].operation == "TRIM_WHITESPACE"
    assert recs[0].target_columns == ["name"]


def test_empty_recommendations_list_is_valid(sample_profile):
    provider = _make_provider(_raw_json_response(recs=[]))
    recs = provider.generate_recommendations(sample_profile)
    assert recs == []


def test_multiple_recommendations_parsed(sample_profile):
    recs_data = [
        {"operation": "TRIM_WHITESPACE", "target_columns": ["name"], "parameters": {}, "reason": "r1", "evidence_refs": []},
        {"operation": "FILL_MISSING", "target_columns": ["salary"], "parameters": {"strategy": "mean"}, "reason": "r2", "evidence_refs": []},
    ]
    provider = _make_provider(_raw_json_response(recs=recs_data))
    recs = provider.generate_recommendations(sample_profile)
    assert len(recs) == 2


# ---------------------------------------------------------------------------
# Malformed output
# ---------------------------------------------------------------------------

def test_non_json_response_raises_response_error(sample_profile):
    provider = _make_provider("this is not json")
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(sample_profile)


def test_missing_recommendations_key_raises_response_error(sample_profile):
    provider = _make_provider(json.dumps({"data": []}))
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(sample_profile)


def test_recommendations_not_a_list_raises_response_error(sample_profile):
    provider = _make_provider(json.dumps({"recommendations": "not a list"}))
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(sample_profile)


def test_recommendation_missing_required_field_raises_response_error(sample_profile):
    # 'operation' is the only truly required field on Recommendation
    provider = _make_provider(_raw_json_response(recs=[{"target_columns": ["x"]}]))
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(sample_profile)


# ---------------------------------------------------------------------------
# Connection failure
# ---------------------------------------------------------------------------

def test_connection_failure_raises_api_error(sample_profile):
    import urllib.error
    def failing_post(url, body):
        raise urllib.error.URLError("Connection refused")
    provider = OllamaProvider(model_name="test", _http_post=failing_post)
    with pytest.raises(AIProviderAPIError):
        provider.generate_recommendations(sample_profile)


def test_timeout_raises_timeout_error(sample_profile):
    def timeout_post(url, body):
        raise TimeoutError("timed out")
    provider = OllamaProvider(model_name="test", _http_post=timeout_post)
    with pytest.raises(AIProviderTimeoutError):
        provider.generate_recommendations(sample_profile)


# ---------------------------------------------------------------------------
# Same recommendation schema as AIProvider contract
# ---------------------------------------------------------------------------

def test_returns_recommendation_objects_not_dicts(sample_profile):
    from converge.models.recommendation import Recommendation
    provider = _make_provider(_raw_json_response())
    recs = provider.generate_recommendations(sample_profile)
    assert all(isinstance(r, Recommendation) for r in recs)


def test_ollama_uses_same_prompt_version_as_gemini():
    from converge.services.ai.gemini_provider import GeminiProvider
    ollama = _make_provider(_raw_json_response())
    # GeminiProvider's prompt_version comes from the same PROMPT_VERSION constant
    assert ollama.prompt_version == PROMPT_VERSION

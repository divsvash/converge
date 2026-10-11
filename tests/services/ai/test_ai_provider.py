"""
AI provider tests.

CRITICAL: none of these tests hit the real Gemini API or require an
API key. Every test either injects a fake `client` object into
GeminiProvider, or works purely against the prompt-building functions.
"""
import json

import pytest
from google.genai import errors as genai_errors

from converge.core.exceptions import (
    AIProviderAPIError,
    AIProviderConfigError,
    AIProviderRateLimitError,
    AIProviderResponseError,
    AIProviderTimeoutError,
)
from converge.models.profile import IssueType
from converge.models.recommendation import Recommendation
from converge.services.ai.gemini_provider import GeminiProvider
from converge.services.ai.prompts import build_operation_catalog_text, build_user_prompt
from converge.services.validation.validator import Validator


# ---------------------------------------------------------------------------
# Fake Gemini client scaffolding
# ---------------------------------------------------------------------------

class _FakeModels:
    """Mimics google.genai.Client().models -- captures the call and returns
    a canned response, or raises a canned exception."""

    def __init__(self, response_text: str | None = None, raise_exc: Exception | None = None):
        self._response_text = response_text
        self._raise_exc = raise_exc
        self.last_call_kwargs: dict = {}

    def generate_content(self, *, model, contents, config):
        self.last_call_kwargs = {"model": model, "contents": contents, "config": config}
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeResponse(self._response_text)


class _FakeResponse:
    def __init__(self, text: str | None):
        self.text = text


class _FakeClient:
    def __init__(self, response_text: str | None = None, raise_exc: Exception | None = None):
        self.models = _FakeModels(response_text=response_text, raise_exc=raise_exc)


def _valid_response_json(recommendations: list[dict]) -> str:
    return json.dumps({"recommendations": recommendations})


# ---------------------------------------------------------------------------
# 1 & 2. Valid structured response -> Recommendation objects, multiple recs
# ---------------------------------------------------------------------------

def test_valid_single_recommendation_is_parsed():
    fake_client = _FakeClient(response_text=_valid_response_json([
        {
            "operation": "TRIM_WHITESPACE",
            "target_columns": ["city"],
            "parameters": {},
            "reason": "Some city values have leading/trailing whitespace.",
            "evidence_refs": [
                {"issue_type": "whitespace_or_casing", "columns": ["city"], "claim": "whitespace present", "claimed_values": {}}
            ],
            "confidence": 0.95,
        }
    ]))
    provider = GeminiProvider(client=fake_client)
    from converge.models.profile import ProfileResult, DatasetProfile, DuplicateSummary
    empty_profile = ProfileResult(
        profile=DatasetProfile(
            dataset_id="x", row_count=1, column_count=0, columns=[],
            duplicate_summary=DuplicateSummary(exact_duplicate_row_count=0),
        ),
        evidence=[],
    )
    recs = provider.generate_recommendations(empty_profile)
    assert len(recs) == 1
    assert isinstance(recs[0], Recommendation)
    assert recs[0].operation == "TRIM_WHITESPACE"
    assert recs[0].target_columns == ["city"]
    assert recs[0].confidence == 0.95


def test_multiple_recommendations_are_all_parsed():
    fake_client = _FakeClient(response_text=_valid_response_json([
        {"operation": "TRIM_WHITESPACE", "target_columns": ["city"], "parameters": {},
         "reason": "r1", "evidence_refs": []},
        {"operation": "FILL_MISSING", "target_columns": ["salary"], "parameters": {"strategy": "mean"},
         "reason": "r2", "evidence_refs": []},
        {"operation": "FLAG_OUTLIER", "target_columns": ["salary"], "parameters": {},
         "reason": "r3", "evidence_refs": []},
    ]))
    provider = GeminiProvider(client=fake_client)
    from converge.models.profile import ProfileResult, DatasetProfile, DuplicateSummary
    empty_profile = ProfileResult(
        profile=DatasetProfile(dataset_id="x", row_count=1, column_count=0, columns=[],
                                duplicate_summary=DuplicateSummary(exact_duplicate_row_count=0)),
        evidence=[],
    )
    recs = provider.generate_recommendations(empty_profile)
    assert len(recs) == 3
    assert [r.operation for r in recs] == ["TRIM_WHITESPACE", "FILL_MISSING", "FLAG_OUTLIER"]


def _empty_profile_result():
    from converge.models.profile import ProfileResult, DatasetProfile, DuplicateSummary
    return ProfileResult(
        profile=DatasetProfile(dataset_id="x", row_count=1, column_count=0, columns=[],
                                duplicate_summary=DuplicateSummary(exact_duplicate_row_count=0)),
        evidence=[],
    )


# ---------------------------------------------------------------------------
# 3. Unknown operation -> parsed successfully, rejected later by the validator
# ---------------------------------------------------------------------------

def test_unknown_operation_parses_successfully_but_is_not_rejected_here():
    fake_client = _FakeClient(response_text=_valid_response_json([
        {"operation": "TOTALLY_MADE_UP_OPERATION", "target_columns": ["city"],
         "parameters": {}, "reason": "invented", "evidence_refs": []},
    ]))
    provider = GeminiProvider(client=fake_client)
    recs = provider.generate_recommendations(_empty_profile_result())
    assert len(recs) == 1
    assert recs[0].operation == "TOTALLY_MADE_UP_OPERATION"  # AI stage does not filter this


def test_unknown_operation_from_ai_is_rejected_by_the_validator(sample_profile_result):
    """Integration boundary: AI stage output flows into the Validator
    (Step 2) exactly as-is, and IS rejected there."""
    fake_client = _FakeClient(response_text=_valid_response_json([
        {"operation": "TOTALLY_MADE_UP_OPERATION", "target_columns": ["city"],
         "parameters": {}, "reason": "invented", "evidence_refs": []},
    ]))
    provider = GeminiProvider(client=fake_client)
    recs = provider.generate_recommendations(sample_profile_result)

    validator = Validator(sample_profile_result)
    result = validator.validate(recs[0])
    from converge.models.validation import ValidationStatus
    assert result.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# 4. Missing required fields -> controlled failure
# ---------------------------------------------------------------------------

def test_missing_required_operation_field_raises_response_error():
    fake_client = _FakeClient(response_text=_valid_response_json([
        {"target_columns": ["city"], "parameters": {}, "reason": "no operation key", "evidence_refs": []},
    ]))
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(_empty_profile_result())


# ---------------------------------------------------------------------------
# 5. Malformed JSON -> controlled failure
# ---------------------------------------------------------------------------

def test_malformed_json_raises_response_error():
    fake_client = _FakeClient(response_text="{this is not valid json,,,")
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(_empty_profile_result())


def test_missing_top_level_recommendations_key_raises_response_error():
    fake_client = _FakeClient(response_text=json.dumps({"not_recommendations": []}))
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(_empty_profile_result())


def test_recommendations_not_a_list_raises_response_error():
    fake_client = _FakeClient(response_text=json.dumps({"recommendations": "not a list"}))
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(_empty_profile_result())


def test_empty_response_text_raises_response_error():
    fake_client = _FakeClient(response_text=None)
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(_empty_profile_result())


def test_recommendation_item_not_an_object_raises_response_error():
    fake_client = _FakeClient(response_text=json.dumps({"recommendations": ["just a string, not an object"]}))
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderResponseError):
        provider.generate_recommendations(_empty_profile_result())


# ---------------------------------------------------------------------------
# 6. Unexpected parameters -> survives parsing, rejected later by validator
# ---------------------------------------------------------------------------

def test_unexpected_parameters_survive_parsing():
    fake_client = _FakeClient(response_text=_valid_response_json([
        {"operation": "TRIM_WHITESPACE", "target_columns": ["city"],
         "parameters": {"sneaky_extra_param": "value", "exec": "import os"},
         "reason": "r", "evidence_refs": []},
    ]))
    provider = GeminiProvider(client=fake_client)
    recs = provider.generate_recommendations(_empty_profile_result())
    assert recs[0].parameters == {"sneaky_extra_param": "value", "exec": "import os"}


def test_unexpected_parameters_are_rejected_by_validator(sample_profile_result):
    fake_client = _FakeClient(response_text=_valid_response_json([
        {"operation": "TRIM_WHITESPACE", "target_columns": ["city"],
         "parameters": {"exec": "import os; os.system('id')"},
         "reason": "r", "evidence_refs": []},
    ]))
    provider = GeminiProvider(client=fake_client)
    recs = provider.generate_recommendations(sample_profile_result)

    validator = Validator(sample_profile_result)
    result = validator.validate(recs[0])
    from converge.models.validation import ValidationStatus
    assert result.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# 7. Fabricated evidence -> reaches the validator and is rejected there
# ---------------------------------------------------------------------------

def test_fabricated_evidence_reaches_validator_and_is_rejected(sample_profile_result):
    fake_client = _FakeClient(response_text=_valid_response_json([
        {
            "operation": "FILL_MISSING", "target_columns": ["salary"],
            "parameters": {"strategy": "mean"},
            "reason": "salary has many missing values",
            "evidence_refs": [
                {"issue_type": "missing_values", "columns": ["salary"],
                 "claim": "salary has 999 missing values", "claimed_values": {"null_count": 999}},
            ],
        },
    ]))
    provider = GeminiProvider(client=fake_client)
    recs = provider.generate_recommendations(sample_profile_result)
    assert len(recs) == 1  # parsing succeeds -- fabrication is a validation concern, not a parsing one

    validator = Validator(sample_profile_result)
    result = validator.validate(recs[0])
    from converge.models.validation import ValidationStatus
    assert result.status == ValidationStatus.REJECTED
    contradiction_check = next(c for c in result.checks if c.rule == "evidence_supports_claim")
    assert contradiction_check.passed is False


# ---------------------------------------------------------------------------
# 8. API failure -> controlled AI-stage error
# ---------------------------------------------------------------------------

def test_client_error_is_mapped_to_api_error():
    exc = genai_errors.ClientError(400, {"message": "bad request", "status": "INVALID_ARGUMENT"})
    fake_client = _FakeClient(raise_exc=exc)
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderAPIError):
        provider.generate_recommendations(_empty_profile_result())


def test_server_error_is_mapped_to_api_error():
    exc = genai_errors.ServerError(503, {"message": "service unavailable"})
    fake_client = _FakeClient(raise_exc=exc)
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderAPIError):
        provider.generate_recommendations(_empty_profile_result())


def test_generic_unexpected_exception_is_wrapped_not_leaked():
    """
    Even an exception type we didn't explicitly anticipate must never
    escape as a raw, unmapped exception -- the AI stage's contract is
    to always raise a typed AIProviderError.
    """
    fake_client = _FakeClient(raise_exc=RuntimeError("some unexpected SDK internal failure"))
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderAPIError):
        provider.generate_recommendations(_empty_profile_result())


# ---------------------------------------------------------------------------
# 9. Timeout / rate-limit failure -> controlled error
# ---------------------------------------------------------------------------

def test_rate_limit_error_is_mapped_to_rate_limit_error():
    exc = genai_errors.ClientError(429, {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"})
    fake_client = _FakeClient(raise_exc=exc)
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderRateLimitError):
        provider.generate_recommendations(_empty_profile_result())


def test_rate_limit_error_is_also_an_api_error_for_broad_handling():
    """AIProviderRateLimitError subclasses AIProviderAPIError so
    callers that only care about 'was it an API problem' can catch
    the parent class."""
    assert issubclass(AIProviderRateLimitError, AIProviderAPIError)


def test_python_timeout_error_is_mapped_to_timeout_error():
    exc = TimeoutError("request exceeded deadline")
    fake_client = _FakeClient(raise_exc=exc)
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderTimeoutError):
        provider.generate_recommendations(_empty_profile_result())


def test_sdk_style_timeout_exception_name_is_mapped_to_timeout_error():
    """
    Simulates an httpx.TimeoutException-shaped error (the real SDK's
    transport layer raises this class, not a builtin TimeoutError) by
    name-matching, since we don't want a hard dependency on httpx's
    exact exception hierarchy in the mapping logic.
    """
    class ReadTimeout(Exception):
        pass

    fake_client = _FakeClient(raise_exc=ReadTimeout("timed out waiting for response"))
    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AIProviderTimeoutError):
        provider.generate_recommendations(_empty_profile_result())


# ---------------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------------

def test_missing_api_key_and_no_client_raises_config_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(AIProviderConfigError):
        GeminiProvider()


def test_injected_client_bypasses_api_key_requirement(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake_client = _FakeClient(response_text=_valid_response_json([]))
    provider = GeminiProvider(client=fake_client)  # must not raise
    recs = provider.generate_recommendations(_empty_profile_result())
    assert recs == []


# ---------------------------------------------------------------------------
# 10. The AI provider never receives raw dataset content
# ---------------------------------------------------------------------------

def test_columns_with_no_flagged_issue_never_leak_raw_values(sample_profile_result):
    """
    The real boundary claim: raw cell VALUES only ever reach the AI
    when they are part of a genuine, bounded, issue-scoped evidence
    item's supporting_data (e.g. whitespace evidence legitimately
    includes a few sample values to illustrate the issue -- that's
    the approved architecture, not a leak). 'department' has no
    missing-value, categorical-inconsistency, or whitespace evidence
    in the sample dataset -- it only appears as a column *name* inside
    the duplicate-detection criteria list (aggregate metadata, not a
    raw cell value). None of its actual raw category values should
    appear anywhere in the prompt.
    """
    prompt = build_user_prompt(sample_profile_result)
    non_duplicate_evidence_columns = {
        c
        for e in sample_profile_result.evidence
        if e.issue_type != IssueType.DUPLICATE_ROWS
        for c in e.columns
    }
    assert "department" not in non_duplicate_evidence_columns  # sanity: confirms this test is meaningful

    for department_value in ["Engineering", "Marketing"]:
        assert department_value not in prompt, (
            f"raw value '{department_value}' from column with no issue-scoped evidence leaked into the AI prompt"
        )


def test_dataset_profile_section_excludes_raw_per_row_fields(sample_profile_result):
    """
    Structural check: the 'dataset_profile' section of the prompt
    never carries the raw-value-bearing fields (sample_values,
    top_values, variant_groups) -- only aggregates. Any raw value the
    AI sees must come exclusively from the bounded 'evidence' section.
    """
    prompt = build_user_prompt(sample_profile_result)
    json_start = prompt.index("{")
    payload = json.loads(prompt[json_start:])
    for column in payload["dataset_profile"]["columns"]:
        assert "sample_values" not in column
        if column.get("categorical_stats") is not None:
            assert "top_values" not in column["categorical_stats"]
            assert "variant_groups" not in column["categorical_stats"]


def test_prompt_is_a_pure_function_of_profile_result(sample_profile_result):
    """
    Confirms build_user_prompt has no side-channel I/O: called twice
    with the same ProfileResult, it produces byte-identical output.
    Nothing (a file, a clock, a live Spark query) is being read outside
    of the ProfileResult argument itself.
    """
    prompt_a = build_user_prompt(sample_profile_result)
    prompt_b = build_user_prompt(sample_profile_result)
    assert prompt_a == prompt_b


def test_generate_recommendations_call_site_only_receives_built_prompt_and_system_instruction(sample_profile_result):
    """
    Confirms the actual call to the (fake) Gemini client is made with
    exactly the system prompt + the built user prompt string -- no
    additional dataset-shaped argument is ever passed.
    """
    fake_client = _FakeClient(response_text=_valid_response_json([]))
    provider = GeminiProvider(client=fake_client)
    provider.generate_recommendations(sample_profile_result)

    call_kwargs = fake_client.models.last_call_kwargs
    assert isinstance(call_kwargs["contents"], str)
    assert call_kwargs["config"].system_instruction is not None
    # the contents string is exactly what build_user_prompt produces
    assert call_kwargs["contents"] == build_user_prompt(sample_profile_result)


def test_provider_has_no_file_or_spark_access_path():
    """
    Structural guard: GeminiProvider's public method signature accepts
    only a ProfileResult, and the module has no import of the Spark
    session/profiler modules or any file-opening utility.
    """
    import inspect
    from converge.services.ai import gemini_provider as module

    sig = inspect.signature(GeminiProvider.generate_recommendations)
    params = list(sig.parameters.keys())
    assert params == ["self", "profile_result"]

    source = inspect.getsource(module)
    assert "open(" not in source
    assert "pyspark" not in source
    assert "SparkSession" not in source


# ---------------------------------------------------------------------------
# Prompt content sanity checks (versioning + required rules present)
# ---------------------------------------------------------------------------

def test_prompt_version_is_set_and_stable():
    from converge.services.ai.prompts import PROMPT_VERSION
    assert PROMPT_VERSION == "recommendation-prompt-v1"


def test_system_prompt_contains_all_required_rules():
    from converge.services.ai.prompts import SYSTEM_PROMPT
    required_phrases = [
        "AUTHORITATIVE",
        "DO NOT INVENT FACTS",
        "DO NOT CALCULATE",
        "ONLY RECOMMEND OPERATIONS FROM THE PROVIDED",
        "NEVER PRODUCE CODE",
        "NEVER RECOMMEND UNSUPPORTED OR DESTRUCTIVE",
        "MUST REFERENCE THE EVIDENCE",
        "DO NOT FABRICATE A RECOMMENDATION",
    ]
    for phrase in required_phrases:
        assert phrase in SYSTEM_PROMPT, f"required rule phrase missing from SYSTEM_PROMPT: {phrase}"


def test_operation_catalog_reflects_the_actual_registry():
    from converge.models.operation_catalog import OPERATION_REGISTRY
    catalog = json.loads(build_operation_catalog_text())
    catalog_ops = {entry["operation"] for entry in catalog}
    assert catalog_ops == set(OPERATION_REGISTRY.keys())


def test_operation_catalog_excludes_forbidden_operations():
    from converge.models.operation_catalog import FORBIDDEN_OPERATIONS
    catalog = json.loads(build_operation_catalog_text())
    catalog_ops = {entry["operation"] for entry in catalog}
    assert catalog_ops.isdisjoint(FORBIDDEN_OPERATIONS)


# ---------------------------------------------------------------------------
# Full integration boundary: ProfileResult -> AIProvider -> Recommendation -> Validator
# ---------------------------------------------------------------------------

def test_full_boundary_profile_to_ai_to_recommendation_to_validator(sample_profile_result):
    """
    End-to-end (mocked-Gemini) exercise of the exact boundary described
    in the task: ProfileResult -> AIProvider -> Recommendation, then
    handed to the Validator, with no Spark execution or frontend
    involved anywhere in this path.
    """
    fake_client = _FakeClient(response_text=_valid_response_json([
        {
            "operation": "STANDARDIZE_CATEGORY", "target_columns": ["city"],
            "parameters": {"mapping": {"Delhi": "Delhi", "delhi": "Delhi", "DELHI": "Delhi", "Delhi ": "Delhi"}},
            "reason": "case/whitespace variants of Delhi",
            "evidence_refs": [
                {"issue_type": "categorical_inconsistency", "columns": ["city"],
                 "claim": "4 raw variants of Delhi", "claimed_values": {}},
            ],
            "confidence": 0.9,
        },
    ]))
    provider = GeminiProvider(client=fake_client)
    recs = provider.generate_recommendations(sample_profile_result)
    assert len(recs) == 1

    validator = Validator(sample_profile_result)
    result = validator.validate(recs[0])

    from converge.models.validation import ValidationStatus
    assert result.status == ValidationStatus.APPROVED_AUTO

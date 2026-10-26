from pathlib import Path

provider_path = Path("src/converge/services/ai/ollama_provider.py")
test_path = Path("tests/services/ai/test_ollama_provider.py")

provider = provider_path.read_text(encoding="utf-8")
tests = test_path.read_text(encoding="utf-8")

old_constant = '''DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 120.0
'''

schema = '''DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 120.0

OLLAMA_RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "target_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "parameters": {"type": "object"},
                    "reason": {"type": "string"},
                    "evidence_refs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "issue_type": {
                                    "type": "string",
                                    "enum": [
                                        "missing_values",
                                        "duplicate_rows",
                                        "categorical_inconsistency",
                                        "whitespace_or_casing",
                                        "date_format_inconsistency",
                                        "outlier_candidate",
                                        "type_inconsistency",
                                    ],
                                },
                                "columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "claim": {"type": "string"},
                                "claimed_values": {"type": "object"},
                            },
                            "required": [
                                "issue_type",
                                "columns",
                                "claim",
                                "claimed_values",
                            ],
                        },
                    },
                    "confidence": {
                        "type": ["number", "null"],
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "operation",
                    "target_columns",
                    "parameters",
                    "reason",
                    "evidence_refs",
                ],
            },
        },
    },
    "required": ["recommendations"],
}
'''

if provider.count(old_constant) != 1:
    raise SystemExit("Expected provider constant block not found exactly once; no changes made.")

old_format = '            "format": "json",'
if provider.count(old_format) != 1:
    raise SystemExit("Expected format setting not found exactly once; no changes made.")

old_import = "from converge.services.ai.ollama_provider import OllamaProvider"
new_import = '''from converge.services.ai.ollama_provider import (
    OLLAMA_RECOMMENDATION_SCHEMA,
    OllamaProvider,
)'''
if tests.count(old_import) != 1:
    raise SystemExit("Expected test import not found exactly once; no changes made.")

test_anchor = '''def _raw_json_response(recs=None):
    """Return just the JSON string that _parse_recommendations expects."""
    if recs is None:
        recs = [{"operation": "TRIM_WHITESPACE", "target_columns": ["name"],
                 "parameters": {}, "reason": "test", "evidence_refs": []}]
    return json.dumps({"recommendations": recs})
'''

new_test = test_anchor + '''

def test_request_uses_structured_ollama_schema(sample_profile):
    captured = {}

    def capture_post(url, body):
        captured["url"] = url
        captured["payload"] = json.loads(body.decode("utf-8"))
        return _raw_json_response()

    provider = OllamaProvider(
        model_name="test-model",
        base_url="http://localhost:11434/",
        _http_post=capture_post,
    )
    provider.generate_recommendations(sample_profile)

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["format"] == OLLAMA_RECOMMENDATION_SCHEMA
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["temperature"] == 0.0
    assert captured["payload"]["model"] == "test-model"
'''

if tests.count(test_anchor) != 1:
    raise SystemExit("Expected test insertion point not found exactly once; no changes made.")
if "test_request_uses_structured_ollama_schema" in tests:
    raise SystemExit("Schema test already exists; refusing to duplicate it.")

provider_updated = provider.replace(old_constant, schema, 1).replace(
    old_format, '            "format": OLLAMA_RECOMMENDATION_SCHEMA,', 1
)
tests_updated = tests.replace(old_import, new_import, 1).replace(
    test_anchor, new_test, 1
)

provider_path.write_text(provider_updated, encoding="utf-8")
test_path.write_text(tests_updated, encoding="utf-8")
print("Updated Ollama provider and its tests.")

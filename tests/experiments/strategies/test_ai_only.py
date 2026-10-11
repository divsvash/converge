"""
AI_ONLY strategy tests. Real Spark, real sample_hr.csv dataset, a fake
AIProvider (no network access, no real API key).
"""
import ast
import inspect
from pathlib import Path

import pytest

from converge.core.exceptions import AIProviderAPIError
from converge.models.recommendation import Recommendation
from converge.services.spark.loader import load_csv
from experiments.strategies import ai_only
from experiments.strategies.ai_only import run_ai_only
from tests.experiments.strategies.conftest import FakeAIProvider


# ---------------------------------------------------------------------------
# Uses AIProvider
# ---------------------------------------------------------------------------

def test_calls_ai_provider_generate_recommendations(spark, sample_scenario, ai_only_configuration, run_id):
    provider = FakeAIProvider(recommendations=[])
    calls = []
    original = provider.generate_recommendations
    provider.generate_recommendations = lambda pr: (calls.append(pr) or original(pr))

    run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)
    assert len(calls) == 1


def test_requires_an_ai_provider(spark, sample_scenario, ai_only_configuration, run_id):
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=None)
    assert outcome.success is False
    assert "ai_provider" in outcome.error.lower()


# ---------------------------------------------------------------------------
# Does not invoke the CONVERGE validator as a decision gate
# ---------------------------------------------------------------------------

def test_module_never_imports_the_validator():
    source_path = Path(inspect.getfile(ai_only))
    tree = ast.parse(source_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
    validator_imports = {m for m in imported_modules if "services.validation" in m}
    assert validator_imports == set(), f"ai_only.py must never import converge.services.validation; found: {validator_imports}"


def test_fabricated_evidence_recommendation_still_executes_without_validator_rejection(
    spark, sample_scenario, ai_only_configuration, run_id,
):
    """
    A recommendation that converge's real Validator would reject for
    lacking evidentiary support (e.g. STANDARDIZE_CATEGORY on a column
    with no real categorical inconsistency) must still reach execution
    here, since AI_ONLY performs no evidence check at all -- only a
    structural one. This is the direct behavioral proof that the
    validator is not acting as a gate.
    """
    fabricated = Recommendation(
        operation="STANDARDIZE_CATEGORY", target_columns=["department"],
        parameters={"mapping": {"Engineering": "Engineering"}},
        reason="fabricated -- department has no real inconsistency",
    )
    provider = FakeAIProvider(recommendations=[fabricated])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    # Structurally valid (real operation, real column, has target columns)
    # so it must be marked executable in the trace, regardless of whether
    # the Validator would have supported it with evidence.
    trace_entry = outcome.trace["recommendations"][0]
    assert trace_entry["executable"] is True


# ---------------------------------------------------------------------------
# Never executes arbitrary AI-authored code
# ---------------------------------------------------------------------------

def test_forbidden_operation_is_structurally_rejected(spark, sample_scenario, ai_only_configuration, run_id):
    malicious = Recommendation(
        operation="DELETE_COLUMN", target_columns=["salary"], parameters={}, reason="AI tries to delete a column",
    )
    provider = FakeAIProvider(recommendations=[malicious])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    trace_entry = outcome.trace["recommendations"][0]
    assert trace_entry["executable"] is False
    assert "not a recognized" in trace_entry["reason"]


def test_arbitrary_code_operation_name_is_structurally_rejected(spark, sample_scenario, ai_only_configuration, run_id):
    malicious = Recommendation(
        operation="ARBITRARY_CODE", target_columns=["salary"],
        parameters={"code": "import os; os.system('rm -rf /')"}, reason="AI tries to submit code",
    )
    provider = FakeAIProvider(recommendations=[malicious])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    trace_entry = outcome.trace["recommendations"][0]
    assert trace_entry["executable"] is False


def test_no_eval_or_exec_anywhere_in_module_source():
    source = inspect.getsource(ai_only)
    assert "eval(" not in source
    assert "exec(" not in source
    assert "subprocess" not in source
    assert "os.system" not in source


# ---------------------------------------------------------------------------
# Unsupported/unexecutable recommendations are handled explicitly
# ---------------------------------------------------------------------------

def test_unknown_operation_name_is_structurally_rejected_with_reason(spark, sample_scenario, ai_only_configuration, run_id):
    hallucinated = Recommendation(
        operation="TOTALLY_MADE_UP_OPERATION", target_columns=["city"], parameters={}, reason="hallucinated",
    )
    provider = FakeAIProvider(recommendations=[hallucinated])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    trace_entry = outcome.trace["recommendations"][0]
    assert trace_entry["executable"] is False
    assert trace_entry["reason"]


def test_finding_only_operation_is_structurally_rejected(spark, sample_scenario, ai_only_configuration, run_id):
    finding_only = Recommendation(
        operation="FLAG_OUTLIER", target_columns=["salary"], parameters={}, reason="salary has an outlier",
    )
    provider = FakeAIProvider(recommendations=[finding_only])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    trace_entry = outcome.trace["recommendations"][0]
    assert trace_entry["executable"] is False
    assert "finding-only" in trace_entry["reason"]


def test_recommendation_with_no_target_columns_is_structurally_rejected(spark, sample_scenario, ai_only_configuration, run_id):
    empty_target = Recommendation(operation="TRIM_WHITESPACE", target_columns=[], parameters={}, reason="r")
    provider = FakeAIProvider(recommendations=[empty_target])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    trace_entry = outcome.trace["recommendations"][0]
    assert trace_entry["executable"] is False


# ---------------------------------------------------------------------------
# Valid supported recommendations can reach the existing Spark execution machinery
# ---------------------------------------------------------------------------

def test_valid_recommendation_executes_and_affects_output(spark, sample_scenario, ai_only_configuration, run_id):
    valid = Recommendation(
        operation="TRIM_WHITESPACE", target_columns=["full_name"], parameters={}, reason="whitespace issue",
    )
    provider = FakeAIProvider(recommendations=[valid])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    assert outcome.success is True
    result_df = load_csv(spark, outcome.output_dataset_path)
    names = [r["full_name"] for r in result_df.collect()]
    assert all(n == n.strip() for n in names)
    assert outcome.trace["recommendations"][0]["executable"] is True


def test_mixed_valid_and_invalid_recommendations_only_valid_ones_execute(
    spark, sample_scenario, ai_only_configuration, run_id,
):
    recs = [
        Recommendation(operation="TRIM_WHITESPACE", target_columns=["full_name"], parameters={}, reason="r1"),
        Recommendation(operation="DELETE_COLUMN", target_columns=["salary"], parameters={}, reason="r2"),
    ]
    provider = FakeAIProvider(recommendations=recs)
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    assert outcome.success is True
    executable_flags = {r["operation"]: r["executable"] for r in outcome.trace["recommendations"]}
    assert executable_flags["TRIM_WHITESPACE"] is True
    assert executable_flags["DELETE_COLUMN"] is False


def test_structurally_valid_but_malformed_parameters_fail_cleanly_inside_executor(
    spark, sample_scenario, ai_only_configuration, run_id,
):
    """
    NORMALIZE_CASE is a real, executable operation (passes AI_ONLY's
    structural filter), but 'target_case' here is invalid. This must
    fail inside the SAME SparkExecutor every other strategy uses, not
    via a duplicated parameter-validation check in this module.
    """
    bad_params = Recommendation(
        operation="NORMALIZE_CASE", target_columns=["city"],
        parameters={"target_case": "not-a-real-case"}, reason="bad params",
    )
    provider = FakeAIProvider(recommendations=[bad_params])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    assert outcome.success is False
    assert outcome.trace["recommendations"][0]["executable"] is True  # passed OUR structural filter
    # but failed downstream, inside the executor


# ---------------------------------------------------------------------------
# AI failures are represented cleanly
# ---------------------------------------------------------------------------

def test_ai_provider_api_error_is_represented_cleanly(spark, sample_scenario, ai_only_configuration, run_id):
    provider = FakeAIProvider(raise_exc=AIProviderAPIError("simulated upstream failure"))
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    assert outcome.success is False
    assert "simulated upstream failure" in outcome.error


def test_unexpected_ai_provider_exception_is_also_represented_cleanly(spark, sample_scenario, ai_only_configuration, run_id):
    provider = FakeAIProvider(raise_exc=RuntimeError("totally unexpected"))
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)

    assert outcome.success is False
    assert outcome.error is not None


def test_empty_recommendation_list_is_a_valid_successful_outcome(spark, sample_scenario, ai_only_configuration, run_id):
    provider = FakeAIProvider(recommendations=[])
    outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id, spark, ai_provider=provider)
    assert outcome.success is True
    assert outcome.trace["recommendations"] == []

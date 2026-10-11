"""
SPARK_ONLY strategy tests. Real Spark, real sample_hr.csv dataset,
no AI provider involved at all.
"""
import ast
import inspect
from pathlib import Path

import pytest

from converge.services.spark.loader import load_csv
from experiments.strategies import spark_only
from experiments.strategies.spark_only import run_spark_only


# ---------------------------------------------------------------------------
# Never invokes AI (structural, not just "no AI provider was passed")
# ---------------------------------------------------------------------------

def test_module_never_imports_ai():
    """
    Scans the actual source of spark_only.py for any import of
    converge.services.ai -- this is the structural guarantee, not an
    inference from behavior.
    """
    source_path = Path(inspect.getfile(spark_only))
    tree = ast.parse(source_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
    ai_imports = {m for m in imported_modules if "services.ai" in m}
    assert ai_imports == set(), f"spark_only.py must never import converge.services.ai; found: {ai_imports}"


def test_ai_provider_parameter_is_never_referenced_in_source():
    """
    Belt and suspenders: even though run_spark_only accepts an
    ai_provider parameter (to satisfy the uniform Strategy signature),
    the function BODY (excluding the signature and docstring) must
    never reference it. AST-based so multi-line signatures/docstrings
    can't accidentally trip a naive text scan.
    """
    source = inspect.getsource(spark_only.run_spark_only)
    tree = ast.parse(source)
    func_def = tree.body[0]
    body_stmts = func_def.body
    if body_stmts and isinstance(body_stmts[0], ast.Expr) and isinstance(getattr(body_stmts[0], "value", None), ast.Constant):
        body_stmts = body_stmts[1:]  # skip the docstring
    for stmt in body_stmts:
        stmt_source = ast.unparse(stmt)
        assert "ai_provider" not in stmt_source, f"ai_provider referenced in executable code: {stmt_source!r}"


def test_running_with_or_without_an_ai_provider_argument_behaves_identically(
    spark, sample_scenario, spark_only_configuration, run_id,
):
    outcome_without = run_spark_only(sample_scenario, spark_only_configuration, run_id + "-a", spark)
    outcome_with = run_spark_only(
        sample_scenario, spark_only_configuration, run_id + "-b", spark, ai_provider=object()  # garbage, must be ignored
    )
    assert outcome_without.success == outcome_with.success
    assert outcome_without.trace == outcome_with.trace


# ---------------------------------------------------------------------------
# Deterministic same-input behavior
# ---------------------------------------------------------------------------

def test_deterministic_same_input_produces_same_decisions(spark, sample_scenario, spark_only_configuration):
    outcome_a = run_spark_only(sample_scenario, spark_only_configuration, "run-a", spark)
    outcome_b = run_spark_only(sample_scenario, spark_only_configuration, "run-b", spark)

    assert outcome_a.success and outcome_b.success
    assert outcome_a.trace == outcome_b.trace  # same decisions, same order, same reasons


def test_deterministic_same_input_produces_same_output_data(spark, sample_scenario, spark_only_configuration):
    outcome_a = run_spark_only(sample_scenario, spark_only_configuration, "run-c", spark)
    outcome_b = run_spark_only(sample_scenario, spark_only_configuration, "run-d", spark)

    df_a = load_csv(spark, outcome_a.output_dataset_path)
    df_b = load_csv(spark, outcome_b.output_dataset_path)
    rows_a = sorted(tuple(r) for r in df_a.collect())
    rows_b = sorted(tuple(r) for r in df_b.collect())
    assert rows_a == rows_b


# ---------------------------------------------------------------------------
# Only mapped/supported unambiguous operations are selected
# ---------------------------------------------------------------------------

def test_only_trim_whitespace_and_remove_duplicates_are_ever_applied(
    spark, sample_scenario, spark_only_configuration, run_id,
):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    applied_ops = {d["operation"] for d in outcome.trace["decisions"] if d["action"] == "applied"}
    assert applied_ops.issubset({"TRIM_WHITESPACE", "REMOVE_DUPLICATES"})


def test_whitespace_issue_is_applied_as_trim_whitespace(spark, sample_scenario, spark_only_configuration, run_id):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    whitespace_decisions = [
        d for d in outcome.trace["decisions"] if d["issue_type"] == "whitespace_or_casing"
    ]
    assert whitespace_decisions  # sample_hr.csv has a known whitespace issue
    assert all(d["action"] == "applied" and d["operation"] == "TRIM_WHITESPACE" for d in whitespace_decisions)


def test_duplicate_rows_issue_is_applied_as_remove_duplicates(spark, sample_scenario, spark_only_configuration, run_id):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    dup_decisions = [d for d in outcome.trace["decisions"] if d["issue_type"] == "duplicate_rows"]
    assert dup_decisions  # sample_hr.csv has a known duplicate pair
    assert all(d["action"] == "applied" and d["operation"] == "REMOVE_DUPLICATES" for d in dup_decisions)


# ---------------------------------------------------------------------------
# Ambiguous issues are skipped with a reason
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("issue_type", [
    "missing_values", "categorical_inconsistency", "date_format_inconsistency",
])
def test_known_ambiguous_issue_types_are_skipped_with_a_reason(
    spark, sample_scenario, spark_only_configuration, run_id, issue_type
):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    matching = [d for d in outcome.trace["decisions"] if d["issue_type"] == issue_type]
    assert matching, f"sample_hr.csv should trigger at least one {issue_type} evidence item"
    for d in matching:
        assert d["action"] == "skipped"
        assert d["operation"] is None
        assert d["reason"]  # non-empty, specific


def test_outlier_candidate_is_skipped(spark, sample_scenario, spark_only_configuration, run_id):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    outlier_decisions = [d for d in outcome.trace["decisions"] if d["issue_type"] == "outlier_candidate"]
    assert outlier_decisions  # sample_hr.csv has a known salary outlier
    assert all(d["action"] == "skipped" for d in outlier_decisions)


def test_every_detected_issue_produces_exactly_one_decision(spark, sample_scenario, spark_only_configuration, run_id):
    """No evidence item is silently dropped -- every detected issue is
    either applied or explicitly skipped with a reason."""
    from converge.services.spark.loader import load_csv as _load
    from converge.services.spark.profiler import profile_dataset as _profile

    df = _load(spark, sample_scenario.dataset.path)
    profile_result = _profile(df, dataset_id="direct_check")

    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    assert len(outcome.trace["decisions"]) == len(profile_result.evidence)


# ---------------------------------------------------------------------------
# Produces an output dataset when execution succeeds
# ---------------------------------------------------------------------------

def test_successful_run_produces_an_output_dataset_path(spark, sample_scenario, spark_only_configuration, run_id):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    assert outcome.success is True
    assert outcome.output_dataset_path is not None
    assert outcome.error is None


def test_output_dataset_reflects_applied_operations(spark, sample_scenario, spark_only_configuration, run_id):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    result_df = load_csv(spark, outcome.output_dataset_path)
    # known duplicate pair (E001/E008) removed -> 20 rows -> 19
    assert result_df.count() == 19
    # known whitespace in full_name trimmed
    names = [r["full_name"] for r in result_df.collect()]
    assert all(n == n.strip() for n in names)


def test_converge_run_id_is_never_set_for_spark_only(spark, sample_scenario, spark_only_configuration, run_id):
    outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id, spark)
    assert outcome.converge_run_id is None

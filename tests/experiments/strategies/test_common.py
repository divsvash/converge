"""
Cross-strategy common contract tests.
"""
import ast
import inspect
from pathlib import Path

from converge.models.recommendation import Recommendation
from experiments.strategies.ai_only import run_ai_only
from experiments.strategies.common import StrategyOutcome
from experiments.strategies.converge import run_converge
from experiments.strategies.spark_only import run_spark_only
from tests.experiments.strategies.conftest import FakeAIProvider


# ---------------------------------------------------------------------------
# All three return the same strategy-outcome type/contract
# ---------------------------------------------------------------------------

def test_all_three_strategies_return_strategy_outcome_instances(
    spark, sample_scenario, spark_only_configuration, ai_only_configuration, converge_configuration, run_id,
):
    provider = FakeAIProvider(recommendations=[
        Recommendation(operation="TRIM_WHITESPACE", target_columns=["full_name"], parameters={}, reason="r"),
    ])

    spark_outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id + "-spark", spark)
    ai_outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id + "-ai", spark, ai_provider=provider)
    converge_outcome = run_converge(sample_scenario, converge_configuration, run_id + "-converge", spark, ai_provider=provider)

    for outcome in (spark_outcome, ai_outcome, converge_outcome):
        assert isinstance(outcome, StrategyOutcome)
        assert hasattr(outcome, "success")
        assert hasattr(outcome, "output_dataset_path")
        assert hasattr(outcome, "error")
        assert hasattr(outcome, "converge_run_id")
        assert hasattr(outcome, "trace")


def test_all_three_strategy_functions_share_a_compatible_call_signature():
    """
    Each strategy must accept (scenario, configuration, run_id, spark,
    ai_provider=...) positionally/by-keyword compatible with the
    Strategy Protocol, so a future runner can hold one type hint.
    """
    expected_leading_params = ["scenario", "configuration", "run_id", "spark"]
    for fn in (run_spark_only, run_ai_only, run_converge):
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        assert param_names[:4] == expected_leading_params, f"{fn.__name__} signature mismatch: {param_names}"
        assert "ai_provider" in sig.parameters


def test_only_converge_strategy_ever_sets_converge_run_id(
    spark, sample_scenario, spark_only_configuration, ai_only_configuration, run_id,
):
    provider = FakeAIProvider(recommendations=[])
    spark_outcome = run_spark_only(sample_scenario, spark_only_configuration, run_id + "-spark", spark)
    ai_outcome = run_ai_only(sample_scenario, ai_only_configuration, run_id + "-ai", spark, ai_provider=provider)

    assert spark_outcome.converge_run_id is None
    assert ai_outcome.converge_run_id is None


# ---------------------------------------------------------------------------
# No strategy computes its own quality score
# ---------------------------------------------------------------------------

def test_strategy_outcome_has_no_quality_performance_or_cost_fields():
    field_names = set(StrategyOutcome.model_fields.keys())
    assert field_names == {"success", "output_dataset_path", "error", "converge_run_id", "trace"}
    for forbidden in ("quality", "performance", "cost", "score", "metrics"):
        assert forbidden not in field_names


def test_no_strategy_module_imports_the_comparison_service():
    """
    converge.services.comparison.comparator (QualityComparator) must
    never be imported by spark_only.py or ai_only.py -- those have no
    legitimate reason to compute a quality comparison at all. converge.py
    is checked separately: it may transitively depend on it via
    PipelineOrchestrator, but must never call
    run_post_execution_analysis() itself (checked behaviorally in
    test_converge.py's fairness-rule test).
    """
    import experiments.strategies.spark_only as spark_only_mod
    import experiments.strategies.ai_only as ai_only_mod

    for module in (spark_only_mod, ai_only_mod):
        source_path = Path(inspect.getfile(module))
        tree = ast.parse(source_path.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        comparison_imports = {m for m in imported if "services.comparison" in m}
        assert comparison_imports == set(), f"{module.__name__} must not import the comparison service: {comparison_imports}"


def test_converge_strategy_never_calls_post_execution_analysis_by_name():
    """
    Checked against the FUNCTION BODY only (not the module docstring,
    which legitimately explains the fairness rule by naming these
    methods as what it deliberately does NOT call).
    """
    from experiments.strategies.converge import run_converge

    source = inspect.getsource(run_converge)
    tree = ast.parse(source)
    func_def = tree.body[0]
    body_stmts = func_def.body
    if body_stmts and isinstance(body_stmts[0], ast.Expr) and isinstance(getattr(body_stmts[0], "value", None), ast.Constant):
        body_stmts = body_stmts[1:]
    body_source = "\n".join(ast.unparse(stmt) for stmt in body_stmts)

    assert "run_post_execution_analysis" not in body_source
    assert "generate_insights" not in body_source
    assert "complete_run" not in body_source


# ---------------------------------------------------------------------------
# Dependency boundary remains experiments -> converge only
# ---------------------------------------------------------------------------

def test_converge_source_never_imports_experiments():
    import converge
    converge_root = Path(converge.__file__).parent
    offending = {}
    for py_file in converge_root.rglob("*.py"):
        if "/cli/" in str(py_file):
            continue  # CLI is a composition root, allowed to wire experiments
        tree = ast.parse(py_file.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
        if "experiments" in imported:
            offending[str(py_file)] = imported
    assert offending == {}, f"converge files importing experiments: {offending}"


_STDLIB_OR_THIRD_PARTY_ALLOWLIST = {
    "__future__", "typing", "pydantic", "pyspark",
}


def test_strategies_only_import_converge_experiments_or_known_third_party():
    """
    Sanity check that the strategy modules' own-project imports are
    limited to `converge` and `experiments` plus a small, explicit
    stdlib/third-party allowlist -- no accidental import of, say, a
    test module or something outside the two packages, and no
    surprise new third-party dependency sneaking in.
    """
    import experiments.strategies.ai_only as ai_only_mod
    import experiments.strategies.converge as converge_mod
    import experiments.strategies.spark_only as spark_only_mod

    for module in (spark_only_mod, ai_only_mod, converge_mod):
        source_path = Path(inspect.getfile(module))
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top in ("converge", "experiments") or top in _STDLIB_OR_THIRD_PARTY_ALLOWLIST, (
                    f"{module.__name__} imports unexpected top-level module '{top}' "
                    f"(from {node.module})"
                )

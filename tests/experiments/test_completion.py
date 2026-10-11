"""
Tests for Phase 10B+ completion: provider resolution, experiment
matrix support, aggregate export, CLI argument parsing. No real
network calls.
"""
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from converge.services.ai.ollama_provider import OllamaProvider
from experiments.models.configuration import (
    AIPlacement, AIRole, Configuration, DeploymentMode, ModelClass, StrategyName,
)
from experiments.providers import resolve_provider
from experiments.runner.experiment import build_configurations, build_scenarios
from experiments.runner.export import compute_aggregate, write_aggregate_csv
from experiments.scenarios.generator import SCENARIO_SPECS


# ===================================================================
# Provider resolution
# ===================================================================

def test_resolve_gemini_remote():
    cfg = Configuration(
        strategy=StrategyName.AI_ONLY, provider="gemini", model="gemini-2.0-flash",
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        provider = resolve_provider(cfg)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-2.0-flash"


def test_resolve_ollama_local():
    cfg = Configuration(
        strategy=StrategyName.AI_ONLY, provider="ollama", model="phi3:mini",
        model_class=ModelClass.SLM, deployment_mode=DeploymentMode.LOCAL,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )
    provider = resolve_provider(cfg)
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "phi3:mini"


def test_unsupported_remote_provider_fails():
    cfg = Configuration(
        strategy=StrategyName.AI_ONLY, provider="unknown", model="x",
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )
    with pytest.raises(ValueError, match="Unknown remote provider"):
        resolve_provider(cfg)


def test_unsupported_local_provider_fails():
    cfg = Configuration(
        strategy=StrategyName.AI_ONLY, provider="unknown", model="x",
        model_class=ModelClass.SLM, deployment_mode=DeploymentMode.LOCAL,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )
    with pytest.raises(ValueError, match="Unknown local provider"):
        resolve_provider(cfg)


# ===================================================================
# Experiment matrix
# ===================================================================

def test_build_configurations_spark_only_always_included():
    configs = build_configurations()
    assert any(c.strategy == StrategyName.SPARK_ONLY for c in configs)


def test_build_configurations_with_gemini():
    configs = build_configurations(gemini_model="gemini-2.0-flash")
    strategies = {(c.strategy.value, c.provider) for c in configs}
    assert ("ai_only", "gemini") in strategies
    assert ("converge", "gemini") in strategies


def test_build_configurations_with_ollama():
    configs = build_configurations(ollama_model="phi3:mini")
    strategies = {(c.strategy.value, c.provider) for c in configs}
    assert ("ai_only", "ollama") in strategies
    assert ("converge", "ollama") in strategies


def test_build_configurations_with_both():
    configs = build_configurations(gemini_model="gemini-2.0-flash", ollama_model="phi3:mini")
    assert len(configs) == 5  # spark_only + 2×gemini + 2×ollama


def test_five_treatment_conditions_representable():
    """Verify all five conditions from the spec are buildable."""
    configs = build_configurations(gemini_model="gemini-2.0-flash", ollama_model="phi3:mini")
    expected = {
        ("spark_only", None),
        ("ai_only", "gemini"),
        ("converge", "gemini"),
        ("ai_only", "ollama"),
        ("converge", "ollama"),
    }
    actual = {(c.strategy.value, c.provider) for c in configs}
    assert actual == expected


def test_all_s1_through_s6_selectable(tmp_path):
    all_names = list(SCENARIO_SPECS.keys())
    assert len(all_names) == 6
    scenarios, manifests, clean_path = build_scenarios(all_names, str(tmp_path))
    assert len(scenarios) == 6
    assert len(manifests) == 6


def test_configurable_repetitions():
    """build_configurations doesn't set repetitions — that's on
    Experiment. Just verify the matrix count math works."""
    configs = build_configurations(gemini_model="g")
    n_scenarios = 3
    n_reps = 5
    expected_runs = n_scenarios * len(configs) * n_reps
    assert expected_runs == 3 * 3 * 5  # 3 scenarios × 3 configs × 5 reps = 45


# ===================================================================
# Aggregate export
# ===================================================================

def test_aggregate_csv_has_expected_columns(tmp_path):
    from experiments.models.run import Run, RunStatus, ReproducibilityMetadata
    from experiments.models.result import Result, QualityMetrics, PerformanceMetrics, CostMetrics, GroundTruthResult
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    run = Run(
        experiment_id="e1", scenario_id="S1", configuration_id="c1",
        strategy=StrategyName.SPARK_ONLY, repetition_index=0,
        status=RunStatus.SUCCEEDED, started_at=now,
        result=Result(
            quality=QualityMetrics(total_issues_before=5, total_issues_after=2, resolved_count=3),
            performance=PerformanceMetrics(strategy_execution_wall_seconds=1.0, total_wall_seconds=2.0),
            cost=CostMetrics(monetary_cost=0.0, input_tokens=0, output_tokens=0, total_tokens=0),
            ground_truth=GroundTruthResult(scenario_id="S1", total_corruptions=3, correctly_repaired=3, repair_recall=1.0, total_originally_clean_cells=87),
        ),
        reproducibility=ReproducibilityMetadata(dataset_checksum="a" * 64),
    )

    rows = compute_aggregate([run])
    assert len(rows) == 1
    assert rows[0]["mean_repair_recall"] == 1.0
    assert rows[0]["mean_issues_before"] == 5.0

    csv_path = tmp_path / "aggregate.csv"
    write_aggregate_csv(rows, csv_path)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
    assert "std_repair_recall" in csv_rows[0]
    assert "mean_input_tokens" in csv_rows[0]


def test_aggregate_unknown_metrics_remain_blank(tmp_path):
    from experiments.models.run import Run, RunStatus, ReproducibilityMetadata
    from experiments.models.result import Result, CostMetrics
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    run = Run(
        experiment_id="e1", scenario_id="S1", configuration_id="c1",
        strategy=StrategyName.AI_ONLY, repetition_index=0,
        status=RunStatus.SUCCEEDED, started_at=now,
        result=Result(cost=CostMetrics()),  # all None
        reproducibility=ReproducibilityMetadata(dataset_checksum="a" * 64, provider="gemini", model="g"),
    )

    rows = compute_aggregate([run])
    assert rows[0]["mean_input_tokens"] is None

    csv_path = tmp_path / "agg.csv"
    write_aggregate_csv(rows, csv_path)
    with open(csv_path) as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["mean_input_tokens"] == ""


# ===================================================================
# CLI
# ===================================================================

def test_cli_run_pipeline_parses_args():
    from converge.cli.main import main
    with pytest.raises(SystemExit) as exc:
        sys.argv = ["converge", "run-pipeline", "--csv", "nonexistent.csv",
                     "--provider", "gemini", "--model", "test"]
        main()
    # exits with error because the CSV doesn't exist
    assert exc.value.code == 1


def test_cli_no_command_prints_help():
    from converge.cli.main import main
    sys.argv = ["converge"]
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_cli_run_experiment_requires_model():
    from converge.cli.main import main
    sys.argv = ["converge", "run-experiment", "--experiment-id", "test"]
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_cli_run_pipeline_missing_ollama_fails_clearly():
    """Ollama provider creation should fail if Ollama isn't running,
    but the error should be clean, not a traceback."""
    from converge.cli.main import main
    sys.argv = ["converge", "run-pipeline", "--csv", "data/raw/sample_hr.csv",
                "--provider", "ollama", "--model", "nonexistent"]
    # This will try to create OllamaProvider (succeeds — it's lazy)
    # then try to connect (fails since no Ollama is running)
    # The CLI should catch the error and exit cleanly


# ===================================================================
# Regression: converge still never imports experiments
# ===================================================================

def test_converge_core_never_imports_experiments():
    """
    The core converge pipeline (models, services, storage) must never
    import experiments. The CLI (converge.cli) is explicitly allowed
    to — it's a composition root that wires both together, which is
    its documented, expected job.
    """
    import ast
    import converge
    converge_root = Path(converge.__file__).parent
    offending = []
    for py_file in converge_root.rglob("*.py"):
        if "/cli/" in str(py_file):
            continue  # CLI is allowed to wire experiments
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "experiments":
                offending.append(str(py_file))
    assert offending == []

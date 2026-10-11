"""
Phase 10B tests: ground-truth integration into the runner, pilot
export (runs.json, summary.csv, metadata.json), and controlled-run
end-to-end behavior. All using fake AI providers — no real network
calls.
"""
import csv
import json
from pathlib import Path

import pytest

from converge.models.recommendation import Recommendation
from experiments.models.configuration import (
    AIPlacement, AIRole, Configuration, DeploymentMode, ModelClass, StrategyName,
)
from experiments.models.experiment import Experiment
from experiments.models.result import GroundTruthResult, Result
from experiments.models.run import RunStatus
from experiments.models.scenario import Dataset, Scenario
from experiments.runner.export import compute_pilot_summary, export_pilot
from experiments.runner.runner import ExperimentRunner
from experiments.scenarios.generator import corrupt_dataset, write_clean_fixture
from experiments.scenarios.models import CorruptionSpec, CorruptionType, GroundTruthManifest
from tests.experiments.strategies.conftest import FakeAIProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def research_dir(tmp_path):
    clean_path = str(tmp_path / "clean.csv")
    write_clean_fixture(clean_path)
    return tmp_path, clean_path


@pytest.fixture()
def ws_scenario(research_dir):
    tmp_path, clean_path = research_dir
    spec = CorruptionSpec(scenario_id="S1_ws", corruption_types=[CorruptionType.WHITESPACE], seed=42)
    out = str(tmp_path / "S1_ws.csv")
    manifest = corrupt_dataset(clean_path, spec, out)
    ds = Dataset.from_path("S1_ws_ds", out)
    scenario = Scenario(scenario_id="S1_ws", name="Whitespace", dataset=ds)
    return scenario, manifest, clean_path


@pytest.fixture()
def spark_cfg():
    return Configuration(strategy=StrategyName.SPARK_ONLY)


@pytest.fixture()
def ai_cfg():
    return Configuration(
        strategy=StrategyName.AI_ONLY, provider="fake", model="fake-model",
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )


@pytest.fixture()
def converge_cfg():
    return Configuration(
        strategy=StrategyName.CONVERGE, provider="fake", model="fake-model",
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )


def _trim_rec():
    return Recommendation(operation="TRIM_WHITESPACE", target_columns=["name"], parameters={}, reason="ws")


# ---------------------------------------------------------------------------
# Ground-truth integration into controlled runs
# ---------------------------------------------------------------------------

def test_ground_truth_populated_for_controlled_scenario(spark, ws_scenario, spark_cfg):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="gt-test", scenario=scenario, configurations=[spark_cfg], repetition_count=1)

    runner = ExperimentRunner(
        spark=spark, clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )
    runs, _ = runner.run_experiment(experiment)

    assert runs[0].status == RunStatus.SUCCEEDED
    assert runs[0].result.ground_truth is not None
    gt = runs[0].result.ground_truth
    assert gt.total_corruptions == manifest.total_corruptions
    assert gt.scenario_id == scenario.scenario_id
    # Spark-only handles whitespace → should get repair_recall > 0
    assert gt.correctly_repaired > 0


def test_ground_truth_not_populated_when_no_manifest_provided(spark, ws_scenario, spark_cfg):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="no-gt", scenario=scenario, configurations=[spark_cfg], repetition_count=1)

    runner = ExperimentRunner(spark=spark)  # no clean_reference_path, no manifests
    runs, _ = runner.run_experiment(experiment)

    assert runs[0].status == RunStatus.SUCCEEDED
    assert runs[0].result.ground_truth is None


def test_ground_truth_absent_for_failed_run(spark, ws_scenario, ai_cfg):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="fail-gt", scenario=scenario, configurations=[ai_cfg], repetition_count=1)

    runner = ExperimentRunner(
        spark=spark, clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )  # no provider_resolver → AI strategy fails
    runs, _ = runner.run_experiment(experiment)

    assert runs[0].status == RunStatus.FAILED
    assert runs[0].result is None  # model forbids Result on FAILED


# ---------------------------------------------------------------------------
# Result serialization with ground_truth
# ---------------------------------------------------------------------------

def test_result_with_ground_truth_round_trips():
    gt = GroundTruthResult(
        scenario_id="S1_ws", total_corruptions=3, correctly_repaired=3,
        repair_recall=1.0, total_originally_clean_cells=87,
    )
    result = Result(ground_truth=gt)
    reloaded = Result.model_validate_json(result.model_dump_json())
    assert reloaded.ground_truth.correctly_repaired == 3
    assert reloaded.ground_truth.scenario_id == "S1_ws"


def test_result_model_has_ground_truth_field():
    assert "ground_truth" in Result.model_fields


# ---------------------------------------------------------------------------
# CSV flattening
# ---------------------------------------------------------------------------

def test_summary_csv_contains_expected_columns(spark, ws_scenario, spark_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="csv-test", scenario=scenario, configurations=[spark_cfg], repetition_count=1)

    runner = ExperimentRunner(
        spark=spark, clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )
    runs, reliability = runner.run_experiment(experiment)
    pilot_dir = export_pilot("test_pilot", runs, reliability, str(tmp_path / "output"))

    csv_path = pilot_dir / "summary.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    # Check key columns exist
    for col in ["run_id", "scenario_id", "strategy", "repair_recall", "issues_before",
                 "strategy_execution_wall_seconds", "input_tokens", "error"]:
        assert col in rows[0], f"Missing column: {col}"


def test_failed_run_exported_with_error_and_blank_metrics(spark, ws_scenario, ai_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="fail-csv", scenario=scenario, configurations=[ai_cfg], repetition_count=1)

    runner = ExperimentRunner(spark=spark)
    runs, reliability = runner.run_experiment(experiment)
    pilot_dir = export_pilot("fail_pilot", runs, reliability, str(tmp_path / "output"))

    with open(pilot_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["success"] == "False"
    assert rows[0]["error"]
    assert rows[0]["repair_recall"] == ""  # blank, not 0
    assert rows[0]["issues_before"] == ""


# ---------------------------------------------------------------------------
# Unknown cost stays blank/None; Spark-only zero stays zero
# ---------------------------------------------------------------------------

def test_spark_only_cost_is_zero_in_csv(spark, ws_scenario, spark_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="cost-test", scenario=scenario, configurations=[spark_cfg], repetition_count=1)

    runner = ExperimentRunner(
        spark=spark, clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )
    runs, reliability = runner.run_experiment(experiment)
    pilot_dir = export_pilot("cost_pilot", runs, reliability, str(tmp_path / "output"))

    with open(pilot_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["input_tokens"] == "0"
    assert rows[0]["monetary_cost"] == "0.0"


def test_ai_strategy_unknown_cost_stays_blank_in_csv(spark, ws_scenario, ai_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="ai-cost", scenario=scenario, configurations=[ai_cfg], repetition_count=1)

    provider = FakeAIProvider(recommendations=[_trim_rec()])
    runner = ExperimentRunner(
        spark=spark, provider_resolver=lambda cfg: provider,
        clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )
    runs, reliability = runner.run_experiment(experiment)
    pilot_dir = export_pilot("ai_cost_pilot", runs, reliability, str(tmp_path / "output"))

    with open(pilot_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    # AI cost is unknown (None), not zero — should be blank in CSV
    assert rows[0]["input_tokens"] == ""
    assert rows[0]["monetary_cost"] == ""


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------

def test_metadata_json_correctness(spark, ws_scenario, spark_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="meta-test", scenario=scenario, configurations=[spark_cfg], repetition_count=2)

    runner = ExperimentRunner(spark=spark)
    runs, reliability = runner.run_experiment(experiment)
    pilot_dir = export_pilot("meta_pilot", runs, reliability, str(tmp_path / "output"))

    meta = json.loads((pilot_dir / "metadata.json").read_text())
    assert meta["pilot_id"] == "meta_pilot"
    assert meta["total_attempted_runs"] == 2
    assert meta["repetition_count"] == 2
    assert meta["scenarios"] == [scenario.scenario_id]
    assert "timestamp" in meta


# ---------------------------------------------------------------------------
# Pilot directory collision protection
# ---------------------------------------------------------------------------

def test_pilot_directory_collision_raises(tmp_path):
    (tmp_path / "existing_pilot").mkdir()
    with pytest.raises(FileExistsError):
        export_pilot("existing_pilot", [], {}, str(tmp_path))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregation_ignores_failed_runs_for_means(spark, ws_scenario, spark_cfg, ai_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(
        name="agg-test", scenario=scenario,
        configurations=[spark_cfg, ai_cfg], repetition_count=1,
    )

    runner = ExperimentRunner(
        spark=spark, clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )
    runs, _ = runner.run_experiment(experiment)

    summary = compute_pilot_summary(runs)
    spark_row = next(s for s in summary if s["strategy"] == "spark_only")
    ai_row = next(s for s in summary if s["strategy"] == "ai_only")

    assert spark_row["successful_runs"] == 1
    assert spark_row["mean_repair_recall"] is not None

    assert ai_row["failed_runs"] == 1
    assert ai_row["mean_repair_recall"] is None  # no successful runs to average


# ---------------------------------------------------------------------------
# No strategy-specific ground-truth evaluator behavior
# ---------------------------------------------------------------------------

def test_ground_truth_evaluator_same_for_all_strategies(spark, ws_scenario, spark_cfg, converge_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    provider = FakeAIProvider(recommendations=[_trim_rec()])

    # Run both strategies
    exp_spark = Experiment(name="gt-fair-spark", scenario=scenario, configurations=[spark_cfg], repetition_count=1)
    exp_conv = Experiment(name="gt-fair-conv", scenario=scenario, configurations=[converge_cfg], repetition_count=1)

    runner = ExperimentRunner(
        spark=spark, provider_resolver=lambda cfg: provider,
        clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )
    spark_runs, _ = runner.run_experiment(exp_spark)
    conv_runs, _ = runner.run_experiment(exp_conv)

    # Both should have ground_truth and both measure the same corruption count
    assert spark_runs[0].result.ground_truth.total_corruptions == manifest.total_corruptions
    assert conv_runs[0].result.ground_truth.total_corruptions == manifest.total_corruptions


# ---------------------------------------------------------------------------
# runs.json completeness
# ---------------------------------------------------------------------------

def test_runs_json_deserializable_and_complete(spark, ws_scenario, spark_cfg, tmp_path):
    scenario, manifest, clean_path = ws_scenario
    experiment = Experiment(name="json-test", scenario=scenario, configurations=[spark_cfg], repetition_count=1)

    runner = ExperimentRunner(
        spark=spark, clean_reference_path=clean_path,
        ground_truth_manifests={scenario.scenario_id: manifest},
    )
    runs, reliability = runner.run_experiment(experiment)
    pilot_dir = export_pilot("json_pilot", runs, reliability, str(tmp_path / "output"))

    data = json.loads((pilot_dir / "runs.json").read_text())
    assert len(data) == 1
    assert data[0]["run_id"] == runs[0].run_id
    assert data[0]["result"]["ground_truth"]["total_corruptions"] == manifest.total_corruptions

"""
ExperimentRunner integration tests. Real Spark, real sample_hr.csv,
fake AI providers (no network access).
"""
import shutil

import pytest

from converge.core.exceptions import AIProviderAPIError
from converge.models.recommendation import Recommendation
from converge.storage.run_store import RunStore
from experiments.models.configuration import StrategyName
from experiments.models.experiment import Experiment
from experiments.models.run import RunStatus
from experiments.models.scenario import Dataset, Scenario
from experiments.runner.runner import STRATEGY_REGISTRY, ExperimentRunner
from tests.experiments.runner.conftest import FakeAIProvider


def _trim_recommendation():
    return Recommendation(operation="TRIM_WHITESPACE", target_columns=["full_name"], parameters={}, reason="r")


# ---------------------------------------------------------------------------
# Correct strategy selected
# ---------------------------------------------------------------------------

def test_strategy_registry_maps_every_strategy_name():
    assert set(STRATEGY_REGISTRY.keys()) == {StrategyName.SPARK_ONLY, StrategyName.AI_ONLY, StrategyName.CONVERGE}


def test_spark_only_configuration_invokes_spark_only_strategy(spark, sample_scenario, spark_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].strategy == StrategyName.SPARK_ONLY
    assert runs[0].converge_run_id is None  # proves spark_only path, not converge


def test_converge_configuration_invokes_converge_strategy_and_produces_a_pipeline_run(
    spark, sample_scenario, converge_configuration,
):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[converge_configuration], repetition_count=1,
    )
    provider = FakeAIProvider(recommendations=[_trim_recommendation()])
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)

    assert runs[0].strategy == StrategyName.CONVERGE
    assert runs[0].converge_run_id is not None
    assert RunStore().exists(runs[0].converge_run_id)


# ---------------------------------------------------------------------------
# Correct number of Runs generated / unique run IDs / repetition indexes
# ---------------------------------------------------------------------------

def test_correct_number_of_runs_generated(spark, sample_scenario, spark_only_configuration, ai_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario,
        configurations=[spark_only_configuration, ai_only_configuration], repetition_count=3,
    )
    provider = FakeAIProvider(recommendations=[])
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)
    assert len(runs) == 6  # 2 configurations x 3 repetitions


def test_run_ids_are_unique(spark, sample_scenario, spark_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=4,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)
    assert len({r.run_id for r in runs}) == 4


def test_repetition_indexes_are_correct_per_configuration(
    spark, sample_scenario, spark_only_configuration, ai_only_configuration,
):
    experiment = Experiment(
        name="exp", scenario=sample_scenario,
        configurations=[spark_only_configuration, ai_only_configuration], repetition_count=3,
    )
    provider = FakeAIProvider(recommendations=[])
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)

    spark_runs = sorted(
        (r for r in runs if r.strategy == StrategyName.SPARK_ONLY), key=lambda r: r.repetition_index
    )
    ai_runs = sorted((r for r in runs if r.strategy == StrategyName.AI_ONLY), key=lambda r: r.repetition_index)
    assert [r.repetition_index for r in spark_runs] == [0, 1, 2]
    assert [r.repetition_index for r in ai_runs] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Same Scenario/Configuration identity retained
# ---------------------------------------------------------------------------

def test_every_run_carries_the_correct_scenario_and_configuration_identity(
    spark, sample_scenario, spark_only_configuration,
):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=2,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)
    for run in runs:
        assert run.scenario_id == sample_scenario.scenario_id
        assert run.configuration_id == spark_only_configuration.configuration_id
        assert run.experiment_id == experiment.experiment_id


# ---------------------------------------------------------------------------
# Missing AI provider fails AI strategies cleanly / Spark-only needs none
# ---------------------------------------------------------------------------

def test_ai_only_without_provider_resolver_fails_cleanly(spark, sample_scenario, ai_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[ai_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)  # no provider_resolver
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].status == RunStatus.FAILED
    assert "provider" in runs[0].error.lower()
    assert runs[0].result is None


def test_converge_without_provider_resolver_fails_cleanly(spark, sample_scenario, converge_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[converge_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].status == RunStatus.FAILED
    assert "provider" in runs[0].error.lower()


def test_spark_only_works_without_any_provider_resolver(spark, sample_scenario, spark_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)  # no provider_resolver at all
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].status == RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Strategy failure creates FAILED Run / success creates SUCCEEDED Run + Result
# ---------------------------------------------------------------------------

def test_ai_provider_failure_creates_failed_run_not_an_exception(spark, sample_scenario, ai_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[ai_only_configuration], repetition_count=1,
    )
    provider = FakeAIProvider(raise_exc=AIProviderAPIError("simulated upstream failure"))
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].status == RunStatus.FAILED
    assert "simulated upstream failure" in runs[0].error
    assert runs[0].result is None


def test_successful_spark_only_run_has_a_fully_populated_result(spark, sample_scenario, spark_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)
    run = runs[0]
    assert run.status == RunStatus.SUCCEEDED
    assert run.result is not None
    assert run.result.quality is not None
    assert run.result.performance is not None
    assert run.result.cost is not None
    assert run.error is None


# ---------------------------------------------------------------------------
# CONVERGE run id preserved (success and failure)
# ---------------------------------------------------------------------------

def test_converge_run_id_preserved_on_success(spark, sample_scenario, converge_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[converge_configuration], repetition_count=1,
    )
    provider = FakeAIProvider(recommendations=[])
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].converge_run_id is not None


def test_converge_run_id_preserved_on_mid_pipeline_failure(spark, sample_scenario, converge_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[converge_configuration], repetition_count=1,
    )
    provider = FakeAIProvider(raise_exc=AIProviderAPIError("simulated failure"))
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].status == RunStatus.FAILED
    # profiling succeeded (a PipelineRun was created) before AI analysis failed
    assert runs[0].converge_run_id is not None


# ---------------------------------------------------------------------------
# One failed repetition does not abort remaining repetitions/configurations
# ---------------------------------------------------------------------------

def test_one_failed_repetition_does_not_abort_the_rest(spark, sample_scenario, ai_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[ai_only_configuration], repetition_count=3,
    )

    call_count = {"n": 0}

    class FlakyProvider(FakeAIProvider):
        def generate_recommendations(self, profile_result):
            call_count["n"] += 1
            if call_count["n"] == 2:  # fail only the second repetition
                raise AIProviderAPIError("simulated transient failure")
            return []

    provider = FlakyProvider()
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)

    assert len(runs) == 3
    statuses = sorted(runs, key=lambda r: r.repetition_index)
    assert statuses[0].status == RunStatus.SUCCEEDED
    assert statuses[1].status == RunStatus.FAILED
    assert statuses[2].status == RunStatus.SUCCEEDED


def test_one_failed_configuration_does_not_abort_other_configurations(
    spark, sample_scenario, spark_only_configuration, ai_only_configuration,
):
    experiment = Experiment(
        name="exp", scenario=sample_scenario,
        configurations=[spark_only_configuration, ai_only_configuration], repetition_count=1,
    )
    # No provider resolver -> ai_only configuration fails, spark_only must still succeed.
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)

    spark_run = next(r for r in runs if r.strategy == StrategyName.SPARK_ONLY)
    ai_run = next(r for r in runs if r.strategy == StrategyName.AI_ONLY)
    assert spark_run.status == RunStatus.SUCCEEDED
    assert ai_run.status == RunStatus.FAILED


# ---------------------------------------------------------------------------
# Checksum: unchanged dataset executes / modified dataset fails before strategy
# ---------------------------------------------------------------------------

def test_unchanged_dataset_executes_normally(spark, sample_scenario, spark_only_configuration):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)
    assert runs[0].status == RunStatus.SUCCEEDED


def test_modified_dataset_fails_before_strategy_invocation(spark, tmp_path, spark_only_configuration):
    original = str(__import__("converge.core.config", fromlist=["RAW_DATA_DIR"]).RAW_DATA_DIR / "sample_hr.csv")
    tmp_csv = tmp_path / "mutable.csv"
    shutil.copy(original, tmp_csv)

    dataset = Dataset.from_path("mutable_ds", str(tmp_csv))
    scenario = Scenario(scenario_id="mutable_scenario", name="Mutable", dataset=dataset)
    experiment = Experiment(
        name="exp", scenario=scenario, configurations=[spark_only_configuration], repetition_count=1,
    )

    # Mutate the file AFTER the Dataset's checksum was captured.
    with open(tmp_csv, "a") as f:
        f.write("EXTRA,ROW,DATA\n")

    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)

    assert runs[0].status == RunStatus.FAILED
    assert "drift" in runs[0].error.lower()
    assert runs[0].result is None
    assert runs[0].trace is None  # never reached strategy execution at all


def test_checksum_is_never_silently_updated(spark, tmp_path, spark_only_configuration):
    original = str(__import__("converge.core.config", fromlist=["RAW_DATA_DIR"]).RAW_DATA_DIR / "sample_hr.csv")
    tmp_csv = tmp_path / "mutable2.csv"
    shutil.copy(original, tmp_csv)
    dataset = Dataset.from_path("mutable_ds2", str(tmp_csv))
    original_checksum = dataset.checksum
    scenario = Scenario(scenario_id="mutable_scenario2", name="Mutable2", dataset=dataset)
    experiment = Experiment(
        name="exp", scenario=scenario, configurations=[spark_only_configuration], repetition_count=1,
    )

    with open(tmp_csv, "a") as f:
        f.write("EXTRA,ROW,DATA\n")

    runner = ExperimentRunner(spark=spark)
    runner.run_experiment(experiment)

    # The Scenario/Dataset object itself (frozen) must be unaffected.
    assert experiment.scenario.dataset.checksum == original_checksum


# ---------------------------------------------------------------------------
# Common quality: same profiler/comparator path regardless of strategy
# ---------------------------------------------------------------------------

def test_quality_result_populated_identically_in_shape_across_all_three_strategies(
    spark, sample_scenario, spark_only_configuration, ai_only_configuration, converge_configuration,
):
    experiment = Experiment(
        name="exp", scenario=sample_scenario,
        configurations=[spark_only_configuration, ai_only_configuration, converge_configuration],
        repetition_count=1,
    )
    provider = FakeAIProvider(recommendations=[_trim_recommendation()])
    runner = ExperimentRunner(spark=spark, provider_resolver=lambda cfg: provider)
    runs, _ = runner.run_experiment(experiment)

    for run in runs:
        assert run.status == RunStatus.SUCCEEDED
        assert run.result.quality is not None
        assert run.result.quality.row_count_before == 20  # same sample_hr.csv, same profiler, for every strategy


def test_no_overall_quality_score_field_reachable_from_a_run():
    from experiments.models.result import QualityMetrics
    field_names = set(QualityMetrics.model_fields.keys())
    assert "score" not in field_names
    assert "overall_score" not in field_names


# ---------------------------------------------------------------------------
# Performance: timing fields non-negative, boundaries hold
# ---------------------------------------------------------------------------

def test_timing_fields_are_non_negative_and_total_covers_the_parts(
    spark, sample_scenario, spark_only_configuration,
):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)
    perf = runs[0].result.performance

    assert perf.strategy_execution_wall_seconds >= 0
    assert perf.evaluation_wall_seconds >= 0
    assert perf.total_wall_seconds >= 0
    # total includes both named buckets plus runner-level overhead
    assert perf.total_wall_seconds >= perf.strategy_execution_wall_seconds
    assert perf.total_wall_seconds >= perf.evaluation_wall_seconds


def test_failed_run_preserves_started_and_completed_at_without_a_result(
    spark, sample_scenario, ai_only_configuration,
):
    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[ai_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)  # no provider -> immediate clean failure
    runs, _ = runner.run_experiment(experiment)
    run = runs[0]
    assert run.started_at is not None
    assert run.completed_at is not None
    assert run.completed_at >= run.started_at
    assert run.result is None  # the model forbids attaching a Result to a FAILED run


# ---------------------------------------------------------------------------
# Dependency: converge never imports experiments; runner doesn't mutate strategies
# ---------------------------------------------------------------------------

def test_converge_still_never_imports_experiments():
    import ast
    from pathlib import Path
    import converge

    converge_root = Path(converge.__file__).parent
    offending = []
    for py_file in converge_root.rglob("*.py"):
        if "/cli/" in str(py_file):
            continue  # CLI is a composition root, allowed to wire experiments
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "experiments":
                offending.append(str(py_file))
    assert offending == []


def test_runner_module_does_not_modify_strategy_source_files():
    """
    Structural guard: the runner only ever CALLS strategy functions --
    it never monkeypatches or reassigns anything on the strategy
    modules.
    """
    import inspect
    from experiments.runner import runner as runner_module
    source = inspect.getsource(runner_module)
    assert "setattr" not in source
    assert "monkeypatch" not in source.lower()


def test_runner_calling_a_strategy_directly_vs_through_the_runner_is_equivalent(
    spark, sample_scenario, spark_only_configuration,
):
    """
    The runner must not change what a strategy decides to do -- only
    measure it. Calling spark_only directly must produce the same
    trace as calling it through the runner.
    """
    from experiments.strategies.spark_only import run_spark_only

    direct_outcome = run_spark_only(sample_scenario, spark_only_configuration, "direct-run-id", spark)

    experiment = Experiment(
        name="exp", scenario=sample_scenario, configurations=[spark_only_configuration], repetition_count=1,
    )
    runner = ExperimentRunner(spark=spark)
    runs, _ = runner.run_experiment(experiment)

    assert runs[0].trace == direct_outcome.trace

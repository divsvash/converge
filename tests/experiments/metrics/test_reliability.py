from datetime import datetime, timezone

import pytest

from experiments.metrics.reliability import compute_reliability_metrics
from experiments.models.configuration import StrategyName
from experiments.models.run import ReproducibilityMetadata, Run, RunStatus
from experiments.models.result import QualityMetrics, Result

_NOW = datetime.now(timezone.utc)
_CHECKSUM = "a" * 64


def _run(
    status=RunStatus.SUCCEEDED, strategy=StrategyName.SPARK_ONLY, repetition_index=0,
    experiment_id="exp1", scenario_id="s1", configuration_id="cfg1",
    error=None, result=None, trace=None,
) -> Run:
    if status == RunStatus.FAILED and error is None:
        error = "simulated failure"
    if status == RunStatus.SUCCEEDED and result is None:
        result = Result(quality=QualityMetrics(resolved_count=1))
    return Run(
        experiment_id=experiment_id, scenario_id=scenario_id, configuration_id=configuration_id,
        strategy=strategy, repetition_index=repetition_index, status=status,
        started_at=_NOW, completed_at=_NOW, error=error, result=result, trace=trace,
        reproducibility=ReproducibilityMetadata(dataset_checksum=_CHECKSUM),
    )


# ---------------------------------------------------------------------------
# attempted/succeeded/failed counts, success rate
# ---------------------------------------------------------------------------

def test_all_successful():
    runs = [_run(repetition_index=i) for i in range(3)]
    rel = compute_reliability_metrics(runs)
    assert rel.attempted_runs == 3
    assert rel.successful_runs == 3
    assert rel.failed_runs == 0
    assert rel.success_rate == 1.0


def test_all_failed():
    runs = [_run(status=RunStatus.FAILED, repetition_index=i) for i in range(2)]
    rel = compute_reliability_metrics(runs)
    assert rel.attempted_runs == 2
    assert rel.successful_runs == 0
    assert rel.failed_runs == 2
    assert rel.success_rate == 0.0


def test_mixed_success_and_failure():
    runs = [
        _run(status=RunStatus.SUCCEEDED, repetition_index=0),
        _run(status=RunStatus.FAILED, repetition_index=1),
        _run(status=RunStatus.SUCCEEDED, repetition_index=2),
    ]
    rel = compute_reliability_metrics(runs)
    assert rel.attempted_runs == 3
    assert rel.successful_runs == 2
    assert rel.failed_runs == 1
    assert rel.success_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Grouping does not mix scenarios/configurations
# ---------------------------------------------------------------------------

def test_mixed_configuration_ids_raises():
    runs = [_run(configuration_id="cfg1"), _run(configuration_id="cfg2")]
    with pytest.raises(ValueError):
        compute_reliability_metrics(runs)


def test_mixed_scenario_ids_raises():
    runs = [_run(scenario_id="s1"), _run(scenario_id="s2")]
    with pytest.raises(ValueError):
        compute_reliability_metrics(runs)


def test_mixed_experiment_ids_raises():
    runs = [_run(experiment_id="e1"), _run(experiment_id="e2")]
    with pytest.raises(ValueError):
        compute_reliability_metrics(runs)


def test_empty_run_list_raises():
    with pytest.raises(ValueError):
        compute_reliability_metrics([])


# ---------------------------------------------------------------------------
# No reliability attached to individual Result
# ---------------------------------------------------------------------------

def test_result_model_has_no_reliability_field():
    assert "reliability" not in Result.model_fields


def test_reliability_metrics_carries_grouping_identity():
    runs = [_run(experiment_id="e1", scenario_id="s1", configuration_id="c1")]
    rel = compute_reliability_metrics(runs)
    assert rel.experiment_id == "e1"
    assert rel.scenario_id == "s1"
    assert rel.configuration_id == "c1"


# ---------------------------------------------------------------------------
# Recommendation consistency (AI_ONLY only, conservative)
# ---------------------------------------------------------------------------

def test_consistency_is_none_for_spark_only_group():
    runs = [_run(strategy=StrategyName.SPARK_ONLY, repetition_index=i) for i in range(3)]
    rel = compute_reliability_metrics(runs)
    assert rel.recommendation_consistency is None


def test_consistency_is_none_for_converge_group():
    runs = [_run(strategy=StrategyName.CONVERGE, repetition_index=i, trace={"note": "see converge_run_id"}) for i in range(3)]
    rel = compute_reliability_metrics(runs)
    assert rel.recommendation_consistency is None


def test_consistency_is_none_with_fewer_than_two_successful_ai_only_runs():
    runs = [_run(strategy=StrategyName.AI_ONLY, repetition_index=0, trace={"recommendations": []})]
    rel = compute_reliability_metrics(runs)
    assert rel.recommendation_consistency is None


def test_consistency_is_one_for_identical_ai_only_recommendations():
    trace = {"recommendations": [
        {"operation": "TRIM_WHITESPACE", "target_columns": ["full_name"], "executable": True, "reason": "r"},
    ]}
    runs = [
        _run(strategy=StrategyName.AI_ONLY, repetition_index=0, trace=trace),
        _run(strategy=StrategyName.AI_ONLY, repetition_index=1, trace=trace),
    ]
    rel = compute_reliability_metrics(runs)
    assert rel.recommendation_consistency == 1.0


def test_consistency_is_zero_for_completely_disjoint_ai_only_recommendations():
    trace_a = {"recommendations": [{"operation": "TRIM_WHITESPACE", "target_columns": ["full_name"], "executable": True, "reason": "r"}]}
    trace_b = {"recommendations": [{"operation": "FILL_MISSING", "target_columns": ["salary"], "executable": True, "reason": "r"}]}
    runs = [
        _run(strategy=StrategyName.AI_ONLY, repetition_index=0, trace=trace_a),
        _run(strategy=StrategyName.AI_ONLY, repetition_index=1, trace=trace_b),
    ]
    rel = compute_reliability_metrics(runs)
    assert rel.recommendation_consistency == 0.0


def test_consistency_partial_overlap_is_between_zero_and_one():
    trace_a = {"recommendations": [
        {"operation": "TRIM_WHITESPACE", "target_columns": ["full_name"], "executable": True, "reason": "r"},
        {"operation": "FILL_MISSING", "target_columns": ["salary"], "executable": True, "reason": "r"},
    ]}
    trace_b = {"recommendations": [
        {"operation": "TRIM_WHITESPACE", "target_columns": ["full_name"], "executable": True, "reason": "r"},
    ]}
    runs = [
        _run(strategy=StrategyName.AI_ONLY, repetition_index=0, trace=trace_a),
        _run(strategy=StrategyName.AI_ONLY, repetition_index=1, trace=trace_b),
    ]
    rel = compute_reliability_metrics(runs)
    assert 0.0 < rel.recommendation_consistency < 1.0


def test_consistency_both_empty_recommendation_sets_is_perfectly_consistent():
    trace = {"recommendations": []}
    runs = [
        _run(strategy=StrategyName.AI_ONLY, repetition_index=0, trace=trace),
        _run(strategy=StrategyName.AI_ONLY, repetition_index=1, trace=trace),
    ]
    rel = compute_reliability_metrics(runs)
    assert rel.recommendation_consistency == 1.0


def test_consistency_excludes_failed_ai_only_runs():
    trace = {"recommendations": [{"operation": "TRIM_WHITESPACE", "target_columns": ["full_name"], "executable": True, "reason": "r"}]}
    runs = [
        _run(strategy=StrategyName.AI_ONLY, repetition_index=0, trace=trace),
        _run(strategy=StrategyName.AI_ONLY, status=RunStatus.FAILED, repetition_index=1, trace=trace),
    ]
    # Only one successful run with a trace -- not enough for a pair.
    rel = compute_reliability_metrics(runs)
    assert rel.recommendation_consistency is None

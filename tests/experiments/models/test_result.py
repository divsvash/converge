import pytest
from pydantic import ValidationError

from experiments.models.result import CostMetrics, PerformanceMetrics, QualityMetrics, Result


def test_result_defaults_to_no_metrics():
    r = Result()
    assert r.quality is None
    assert r.performance is None
    assert r.cost is None


def test_result_can_hold_quality_performance_and_cost_independently():
    r = Result(
        quality=QualityMetrics(resolved_count=3),
        performance=PerformanceMetrics(total_wall_seconds=12.5),
        cost=CostMetrics(monetary_cost=0.02, currency="USD"),
    )
    assert r.quality.resolved_count == 3
    assert r.performance.total_wall_seconds == 12.5
    assert r.cost.monetary_cost == 0.02
    assert r.cost.currency == "USD"


def test_result_has_no_reliability_field():
    """
    Structural guard: reliability must never be attached to a
    per-Run Result -- it is only meaningful aggregated across
    repeated runs of the same Scenario+Configuration.
    """
    field_names = set(Result.model_fields.keys())
    assert "reliability" not in field_names
    assert field_names == {"quality", "performance", "cost", "ground_truth"}


def test_quality_metrics_fields_are_all_optional():
    q = QualityMetrics()
    assert q.total_issues_before is None
    assert q.resolved_count is None


@pytest.mark.parametrize("field", [
    "strategy_execution_wall_seconds", "evaluation_wall_seconds", "total_wall_seconds",
])
def test_performance_metrics_timing_fields_must_be_non_negative(field):
    with pytest.raises(ValidationError):
        PerformanceMetrics(**{field: -1.0})


def test_performance_metrics_zero_timing_is_valid():
    p = PerformanceMetrics(strategy_execution_wall_seconds=0.0, evaluation_wall_seconds=0.0, total_wall_seconds=0.0)
    assert p.total_wall_seconds == 0.0


def test_performance_metrics_three_distinct_timing_fields_exist():
    field_names = set(PerformanceMetrics.model_fields.keys())
    assert field_names == {
        "strategy_execution_wall_seconds", "evaluation_wall_seconds", "total_wall_seconds",
    }


@pytest.mark.parametrize("field", ["monetary_cost", "input_tokens", "output_tokens", "total_tokens"])
def test_cost_metrics_fields_must_be_non_negative(field):
    with pytest.raises(ValidationError):
        CostMetrics(**{field: -1})


def test_cost_metrics_zero_is_valid_and_meaningful_for_spark_only():
    c = CostMetrics(monetary_cost=0.0, input_tokens=0, output_tokens=0, total_tokens=0)
    assert c.monetary_cost == 0.0


def test_result_serialization_round_trip():
    r = Result(
        quality=QualityMetrics(resolved_count=2, introduced_count=0),
        performance=PerformanceMetrics(total_wall_seconds=3.3),
    )
    reloaded = Result.model_validate_json(r.model_dump_json())
    assert reloaded == r

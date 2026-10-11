import pytest
from pydantic import ValidationError

from experiments.metrics.performance import build_performance_metrics


def test_build_performance_metrics_populates_all_three_fields():
    p = build_performance_metrics(
        strategy_execution_wall_seconds=1.5, evaluation_wall_seconds=0.5, total_wall_seconds=2.1,
    )
    assert p.strategy_execution_wall_seconds == 1.5
    assert p.evaluation_wall_seconds == 0.5
    assert p.total_wall_seconds == 2.1


def test_timing_fields_must_be_non_negative():
    with pytest.raises(ValidationError):
        build_performance_metrics(
            strategy_execution_wall_seconds=-1.0, evaluation_wall_seconds=0.5, total_wall_seconds=2.1,
        )


def test_timing_fields_may_be_none():
    p = build_performance_metrics(None, None, None)
    assert p.strategy_execution_wall_seconds is None
    assert p.evaluation_wall_seconds is None
    assert p.total_wall_seconds is None


def test_zero_timing_is_valid():
    p = build_performance_metrics(0.0, 0.0, 0.0)
    assert p.total_wall_seconds == 0.0

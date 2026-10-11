"""
RunStatus lifecycle tests.

Phase 9 decision (documented in experiments/models/run.py's module
docstring): RunStatus remains SUCCEEDED | FAILED only -- no PENDING or
RUNNING was added, since the runner never constructs or exposes an
in-progress Run to any caller; a Run is only ever built once, already
in its terminal state. These tests assert that decision holds and that
the two legal "shapes" (SUCCEEDED-with-Result-no-error,
FAILED-with-error-no-Result) are the only ones the model accepts.
"""
import pytest
from pydantic import ValidationError

from experiments.models.configuration import StrategyName
from experiments.models.result import QualityMetrics, Result
from experiments.models.run import ReproducibilityMetadata, Run, RunStatus
from datetime import datetime, timezone

_NOW = datetime.now(timezone.utc)


def _base(**overrides):
    kwargs = dict(
        experiment_id="e1", scenario_id="s1", configuration_id="c1",
        strategy=StrategyName.SPARK_ONLY, repetition_index=0,
        started_at=_NOW, reproducibility=ReproducibilityMetadata(dataset_checksum="a" * 64),
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Only two RunStatus values exist (Phase 9 decision)
# ---------------------------------------------------------------------------

def test_run_status_has_exactly_two_values():
    assert {s.value for s in RunStatus} == {"succeeded", "failed"}


# ---------------------------------------------------------------------------
# Legal "transitions" -- i.e. legal terminal shapes, since a Run is
# only ever constructed once in its final state
# ---------------------------------------------------------------------------

def test_succeeded_with_result_and_no_error_is_legal():
    run = Run(**_base(status=RunStatus.SUCCEEDED, result=Result(quality=QualityMetrics(resolved_count=1))))
    assert run.status == RunStatus.SUCCEEDED


def test_failed_with_error_and_no_result_is_legal():
    run = Run(**_base(status=RunStatus.FAILED, error="boom"))
    assert run.status == RunStatus.FAILED


# ---------------------------------------------------------------------------
# Invalid combinations are rejected
# ---------------------------------------------------------------------------

def test_succeeded_with_error_is_illegal():
    with pytest.raises(ValidationError):
        Run(**_base(status=RunStatus.SUCCEEDED, error="should not be here"))


def test_failed_without_error_is_illegal():
    with pytest.raises(ValidationError):
        Run(**_base(status=RunStatus.FAILED, error=None))


def test_failed_with_a_result_is_illegal():
    with pytest.raises(ValidationError):
        Run(**_base(
            status=RunStatus.FAILED, error="boom", result=Result(quality=QualityMetrics(resolved_count=1)),
        ))


def test_succeeded_without_a_result_is_legal_but_distinct_from_failed():
    """
    A SUCCEEDED run with no Result yet is still legal (Result is
    optional, not required by SUCCEEDED) -- this is different from the
    runner's actual behavior (which always attaches a Result on
    success), but the model itself doesn't force that; it only forces
    the FAILED-must-not-carry-Result direction.
    """
    run = Run(**_base(status=RunStatus.SUCCEEDED, result=None))
    assert run.result is None
    assert run.status == RunStatus.SUCCEEDED


def test_no_pending_or_running_status_exists_to_be_misused():
    with pytest.raises(AttributeError):
        _ = RunStatus.PENDING
    with pytest.raises(AttributeError):
        _ = RunStatus.RUNNING

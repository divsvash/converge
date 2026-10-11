from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from experiments.models.configuration import StrategyName
from experiments.models.result import QualityMetrics, Result
from experiments.models.run import ReproducibilityMetadata, Run, RunStatus

_NOW = datetime.now(timezone.utc)
_CHECKSUM = "a" * 64


def _repro(**overrides) -> ReproducibilityMetadata:
    base = dict(dataset_checksum=_CHECKSUM)
    base.update(overrides)
    return ReproducibilityMetadata(**base)


def _base_run_kwargs(**overrides) -> dict:
    kwargs = dict(
        experiment_id="exp1", scenario_id="s1", configuration_id="cfg1",
        strategy=StrategyName.SPARK_ONLY, repetition_index=0,
        status=RunStatus.SUCCEEDED, started_at=_NOW, completed_at=_NOW + timedelta(seconds=5),
        reproducibility=_repro(),
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Failed Run representation (failure is research data)
# ---------------------------------------------------------------------------

def test_failed_run_requires_error_message():
    with pytest.raises(ValidationError):
        Run(**_base_run_kwargs(status=RunStatus.FAILED, error=None))


def test_failed_run_with_error_is_valid():
    run = Run(**_base_run_kwargs(status=RunStatus.FAILED, error="Spark job failed: OOM"))
    assert run.status == RunStatus.FAILED
    assert run.error == "Spark job failed: OOM"
    assert run.result is None


def test_failed_run_must_not_carry_a_result():
    with pytest.raises(ValidationError):
        Run(**_base_run_kwargs(
            status=RunStatus.FAILED, error="boom",
            result=Result(quality=QualityMetrics(resolved_count=1)),
        ))


def test_succeeded_run_must_not_carry_error():
    with pytest.raises(ValidationError):
        Run(**_base_run_kwargs(status=RunStatus.SUCCEEDED, error="should not be here"))


def test_succeeded_run_with_result_is_valid():
    run = Run(**_base_run_kwargs(result=Result(quality=QualityMetrics(resolved_count=2))))
    assert run.status == RunStatus.SUCCEEDED
    assert run.result.quality.resolved_count == 2


def test_succeeded_run_without_a_result_is_still_valid():
    """A successful Run with no Result yet (e.g. measurement step not
    run) must be representable -- Result is optional, not required by
    SUCCEEDED status."""
    run = Run(**_base_run_kwargs())
    assert run.result is None


# ---------------------------------------------------------------------------
# converge_run_id only applicable to CONVERGE
# ---------------------------------------------------------------------------

def test_converge_run_id_rejected_for_spark_only():
    with pytest.raises(ValidationError):
        Run(**_base_run_kwargs(strategy=StrategyName.SPARK_ONLY, converge_run_id="pr-123"))


def test_converge_run_id_rejected_for_ai_only():
    with pytest.raises(ValidationError):
        Run(**_base_run_kwargs(strategy=StrategyName.AI_ONLY, converge_run_id="pr-123"))


def test_converge_run_id_allowed_for_converge_strategy():
    run = Run(**_base_run_kwargs(strategy=StrategyName.CONVERGE, converge_run_id="pr-123"))
    assert run.converge_run_id == "pr-123"


def test_converge_run_id_is_optional_even_for_converge_strategy():
    run = Run(**_base_run_kwargs(strategy=StrategyName.CONVERGE))
    assert run.converge_run_id is None


# ---------------------------------------------------------------------------
# repetition_index validation
# ---------------------------------------------------------------------------

def test_repetition_index_rejects_negative():
    with pytest.raises(ValidationError):
        Run(**_base_run_kwargs(repetition_index=-1))


def test_repetition_index_zero_is_valid():
    run = Run(**_base_run_kwargs(repetition_index=0))
    assert run.repetition_index == 0


# ---------------------------------------------------------------------------
# Timestamp consistency
# ---------------------------------------------------------------------------

def test_completed_at_before_started_at_is_rejected():
    with pytest.raises(ValidationError):
        Run(**_base_run_kwargs(started_at=_NOW, completed_at=_NOW - timedelta(seconds=1)))


def test_completed_at_may_be_absent_for_an_in_progress_or_failed_run():
    run = Run(**_base_run_kwargs(status=RunStatus.FAILED, error="timeout", completed_at=None))
    assert run.completed_at is None


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------

def test_reproducibility_metadata_requires_dataset_checksum():
    with pytest.raises(ValidationError):
        ReproducibilityMetadata()


def test_reproducibility_metadata_captures_provider_and_model_fields():
    repro = _repro(provider="gemini", model="gemini-2.0-flash", model_class="llm", deployment_mode="remote")
    assert repro.provider == "gemini"
    assert repro.model == "gemini-2.0-flash"
    assert repro.model_class == "llm"
    assert repro.deployment_mode == "remote"


def test_reproducibility_metadata_does_not_expose_seed_or_temperature_fields():
    """
    Structural guard: the field set must never include a seed or
    temperature value, since AIProvider does not expose either -- a
    field existing here would imply the system actually reports it.
    """
    field_names = set(ReproducibilityMetadata.model_fields.keys())
    assert "seed" not in field_names
    assert "temperature" not in field_names


def test_reproducibility_metadata_package_version_is_optional_and_not_auto_populated():
    """
    This model has no behavior -- package_version is a plain nullable
    field a future runner will populate, not something computed by
    the model itself.
    """
    repro = _repro()
    assert repro.package_version is None


# ---------------------------------------------------------------------------
# Serialization / deserialization stability
# ---------------------------------------------------------------------------

def test_run_serialization_round_trip_preserves_all_fields():
    run = Run(**_base_run_kwargs(
        strategy=StrategyName.CONVERGE,
        converge_run_id="pr-abc",
        result=Result(quality=QualityMetrics(resolved_count=4, total_issues_before=10, total_issues_after=6)),
        reproducibility=_repro(provider="gemini", model="gemini-2.0-flash", prompt_version="recommendation-prompt-v1"),
    ))
    reloaded = Run.model_validate_json(run.model_dump_json())

    assert reloaded.run_id == run.run_id
    assert reloaded.converge_run_id == run.converge_run_id
    assert reloaded.result.quality.resolved_count == 4
    assert reloaded.reproducibility.prompt_version == "recommendation-prompt-v1"
    assert reloaded == run


def test_run_status_enum_values_are_stable_strings():
    assert RunStatus.SUCCEEDED.value == "succeeded"
    assert RunStatus.FAILED.value == "failed"

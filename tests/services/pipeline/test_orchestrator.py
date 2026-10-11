"""
Pipeline orchestrator tests.

Uses the real Spark profiler and Validator (already tested in Steps
1-2) against the real sample_hr.csv dataset, combined with a small
fake AIProvider (same pattern as Step 3's fake Gemini client) so no
test depends on network access or a real API key.
"""
import json

import pytest

from converge.core.exceptions import (
    InvalidStateTransitionError,
    PipelineStageError,
    RecommendationNotEligibleError,
    RecommendationNotFoundError,
)
from converge.core.config import RAW_DATA_DIR
from converge.models.execution import ExecutionSummary, OperationExecutionResult, OperationExecutionStatus
from converge.models.pipeline import (
    PipelineRun,
    PipelineStatus,
    RecommendationDecision,
)
from converge.models.recommendation import Recommendation
from converge.models.validation import ValidationStatus
from converge.services.ai.base import AIProvider
from converge.services.pipeline.orchestrator import PipelineOrchestrator
from converge.storage.run_store import RunStore


# ---------------------------------------------------------------------------
# Fake AI provider (mirrors the fake-client pattern from Step 3's tests)
# ---------------------------------------------------------------------------

class FakeAIProvider(AIProvider):
    """Returns a fixed, pre-built list of Recommendation objects (or raises)."""

    def __init__(self, recommendations: list[Recommendation] | None = None, raise_exc: Exception | None = None):
        self._recommendations = recommendations or []
        self._raise_exc = raise_exc

    @property
    def provider_name(self) -> str:
        return "fake-provider"

    @property
    def model_name(self) -> str:
        return "fake-model-v0"

    @property
    def prompt_version(self) -> str:
        return "fake-prompt-v0"

    def generate_recommendations(self, profile_result):
        if self._raise_exc is not None:
            raise self._raise_exc
        return list(self._recommendations)


def _mixed_recommendations() -> list[Recommendation]:
    """
    Four recommendations against the real sample_hr profile, one for
    each validator outcome: APPROVED_AUTO, NEEDS_REVIEW, REJECTED, and
    NOT_EXECUTABLE. Reused across several tests.
    """
    return [
        Recommendation(  # -> APPROVED_AUTO
            operation="STANDARDIZE_CATEGORY",
            target_columns=["city"],
            parameters={"mapping": {"Delhi": "Delhi", "delhi": "Delhi", "DELHI": "Delhi", "Delhi ": "Delhi"}},
            reason="case/whitespace variants of Delhi",
        ),
        Recommendation(  # -> NEEDS_REVIEW
            operation="FILL_MISSING",
            target_columns=["salary"],
            parameters={"strategy": "mean"},
            reason="salary has missing values",
        ),
        Recommendation(  # -> REJECTED (no evidence for department)
            operation="STANDARDIZE_CATEGORY",
            target_columns=["department"],
            parameters={"mapping": {"Engineering": "Engineering"}},
            reason="fabricated: department has no real inconsistency",
        ),
        Recommendation(  # -> NOT_EXECUTABLE (finding-only)
            operation="FLAG_OUTLIER",
            target_columns=["salary"],
            parameters={},
            reason="salary has a 600000 outlier",
        ),
    ]


@pytest.fixture()
def orchestrator(spark, tmp_path):
    ai_provider = FakeAIProvider(recommendations=_mixed_recommendations())
    run_store = RunStore(base_dir=tmp_path)
    return PipelineOrchestrator(spark=spark, ai_provider=ai_provider, run_store=run_store)


@pytest.fixture()
def dataset_path():
    return str(RAW_DATA_DIR / "sample_hr.csv")


def _by_operation(run: PipelineRun, operation: str):
    return next(r for r in run.recommendation_records if r.recommendation.operation == operation)


# ---------------------------------------------------------------------------
# Happy-path state progression
# ---------------------------------------------------------------------------

def test_happy_path_reaches_cleaned(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    assert run.status == PipelineStatus.INGESTED

    run = orchestrator.run_profiling(run)
    assert run.status == PipelineStatus.PROFILED
    assert run.profile_result is not None
    assert run.profile_result.profile.row_count == 20

    run = orchestrator.run_ai_analysis(run)
    assert run.status == PipelineStatus.AI_ANALYZED
    assert len(run.recommendation_records) == 4

    run = orchestrator.run_validation(run)
    assert run.status == PipelineStatus.AWAITING_APPROVAL

    approved_rec = _by_operation(run, "STANDARDIZE_CATEGORY")
    run = orchestrator.submit_decision(run, approved_rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER)
    assert run.status == PipelineStatus.AWAITING_APPROVAL  # unchanged by decision

    plan = orchestrator.build_execution_plan(run)
    assert run.status == PipelineStatus.EXECUTING
    assert len(plan.operations) == 1
    assert plan.operations[0].operation == "STANDARDIZE_CATEGORY"
    assert plan.operations[0].recommendation_id == approved_rec.recommendation.recommendation_id
    assert plan.run_id == run.run_id
    assert plan.dataset_path == dataset_path
    assert run.execution_plan == plan  # stored on the run for inspectability

    fake_summary = ExecutionSummary(
        operations_attempted=1, operations_succeeded=1, operations_failed=0, operations_skipped=0,
        operation_results=[
            OperationExecutionResult(
                recommendation_id=plan.operations[0].recommendation_id,
                operation="STANDARDIZE_CATEGORY",
                target_columns=("city",),
                status=OperationExecutionStatus.SUCCEEDED,
                rows_affected=6,
            )
        ],
        output_dataset_path="/tmp/fake_cleaned.csv",
        notes="applied 1 op",
    )
    run = orchestrator.complete_execution(run, fake_summary)
    assert run.status == PipelineStatus.CLEANED
    assert run.execution_summary.operations_succeeded == 1

    # Full timeline recorded every transition
    statuses_seen = [e.status for e in run.timeline]
    assert statuses_seen == [
        PipelineStatus.INGESTED, PipelineStatus.PROFILED, PipelineStatus.AI_ANALYZED,
        PipelineStatus.VALIDATED, PipelineStatus.AWAITING_APPROVAL,
        PipelineStatus.AWAITING_APPROVAL,  # decision event, status unchanged
        PipelineStatus.EXECUTING, PipelineStatus.CLEANED,
    ]


# ---------------------------------------------------------------------------
# Invalid state transitions
# ---------------------------------------------------------------------------

def test_running_ai_analysis_before_profiling_is_rejected(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    with pytest.raises(InvalidStateTransitionError):
        orchestrator.run_ai_analysis(run)


def test_running_profiling_twice_is_rejected(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    with pytest.raises(InvalidStateTransitionError):
        orchestrator.run_profiling(run)


def test_validation_before_ai_analysis_is_rejected(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    with pytest.raises(InvalidStateTransitionError):
        orchestrator.run_validation(run)


def test_direct_transition_table_has_no_shortcuts():
    from converge.models.pipeline import VALID_TRANSITIONS, PipelineStatus
    # INGESTED can only go to PROFILED -- not straight to AWAITING_APPROVAL, etc.
    assert VALID_TRANSITIONS[PipelineStatus.INGESTED] == {PipelineStatus.PROFILED}
    assert PipelineStatus.COMPLETED not in VALID_TRANSITIONS[PipelineStatus.INGESTED]


# ---------------------------------------------------------------------------
# AI failure
# ---------------------------------------------------------------------------

def test_ai_stage_failure_marks_run_failed_not_successful(spark, tmp_path, dataset_path):
    from converge.core.exceptions import AIProviderAPIError

    ai_provider = FakeAIProvider(raise_exc=AIProviderAPIError("upstream Gemini failure"))
    run_store = RunStore(base_dir=tmp_path)
    orch = PipelineOrchestrator(spark=spark, ai_provider=ai_provider, run_store=run_store)

    run = orch.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orch.run_profiling(run)

    with pytest.raises(PipelineStageError):
        orch.run_ai_analysis(run)

    assert run.status == PipelineStatus.FAILED
    assert "ai_analysis" in run.error
    assert "AIProviderAPIError" in run.error

    # Persisted, not silently dropped
    reloaded = run_store.load(run.run_id)
    assert reloaded.status == PipelineStatus.FAILED
    assert reloaded.error == run.error


# ---------------------------------------------------------------------------
# Validator failure (unexpected exception raised inside Validator itself)
# ---------------------------------------------------------------------------

def test_validation_stage_failure_marks_run_failed(orchestrator, dataset_path, monkeypatch):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)

    def _boom(self, rec):
        raise RuntimeError("simulated internal validator failure")

    monkeypatch.setattr("converge.services.pipeline.orchestrator.Validator.validate", _boom)

    with pytest.raises(PipelineStageError):
        orchestrator.run_validation(run)

    assert run.status == PipelineStatus.FAILED
    assert "validation" in run.error
    assert "RuntimeError" in run.error


# ---------------------------------------------------------------------------
# Rejected recommendation
# ---------------------------------------------------------------------------

def test_rejected_recommendation_cannot_be_approved(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    rejected_rec = next(
        r for r in run.recommendation_records
        if r.recommendation.operation == "STANDARDIZE_CATEGORY" and r.recommendation.target_columns == ["department"]
    )
    assert rejected_rec.validation.status == ValidationStatus.REJECTED

    with pytest.raises(RecommendationNotEligibleError):
        orchestrator.submit_decision(
            run, rejected_rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER
        )


def test_rejected_recommendation_can_still_be_marked_rejected_by_user(orchestrator, dataset_path):
    """Rejecting (not approving) is always allowed, regardless of validator status."""
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    rejected_rec = next(
        r for r in run.recommendation_records
        if r.recommendation.operation == "STANDARDIZE_CATEGORY" and r.recommendation.target_columns == ["department"]
    )
    run = orchestrator.submit_decision(
        run, rejected_rec.recommendation.recommendation_id, RecommendationDecision.REJECTED_BY_USER
    )
    updated = next(r for r in run.recommendation_records if r.recommendation.recommendation_id == rejected_rec.recommendation.recommendation_id)
    assert updated.decision == RecommendationDecision.REJECTED_BY_USER


# ---------------------------------------------------------------------------
# User approval without execution
# ---------------------------------------------------------------------------

def test_approving_a_recommendation_does_not_change_pipeline_status(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    fill_missing_rec = _by_operation(run, "FILL_MISSING")
    run = orchestrator.submit_decision(
        run, fill_missing_rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER
    )
    assert run.status == PipelineStatus.AWAITING_APPROVAL
    assert run.execution_summary is None


# ---------------------------------------------------------------------------
# Execution attempted before approval
# ---------------------------------------------------------------------------

def test_execution_attempted_before_awaiting_approval_is_rejected(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    # status is AI_ANALYZED, not AWAITING_APPROVAL
    with pytest.raises(InvalidStateTransitionError):
        orchestrator.build_execution_plan(run)


def test_run_validation_always_lands_exactly_on_awaiting_approval(orchestrator, dataset_path):
    """
    Structural guard: run_validation() always lands on AWAITING_APPROVAL
    in one call (VALIDATED is a transient intermediate status recorded
    only in the timeline) -- so there is no run object a caller could
    hold that is "stuck" at VALIDATED to test execution against. This
    test documents that guarantee.
    """
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)
    assert run.status == PipelineStatus.AWAITING_APPROVAL
    assert any(e.status == PipelineStatus.VALIDATED for e in run.timeline)


# ---------------------------------------------------------------------------
# Finding-only recommendation never reaching execution
# ---------------------------------------------------------------------------

def test_flag_outlier_cannot_be_approved(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    flag_rec = _by_operation(run, "FLAG_OUTLIER")
    assert flag_rec.validation.status == ValidationStatus.NOT_EXECUTABLE

    with pytest.raises(RecommendationNotEligibleError):
        orchestrator.submit_decision(
            run, flag_rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER
        )


def test_flag_outlier_never_appears_in_execution_eligible_set_even_if_forced(orchestrator, dataset_path):
    """
    Defense in depth: even if a decision were somehow forced to
    APPROVED_BY_USER on a NOT_EXECUTABLE record (bypassing the normal
    submit_decision guard, simulating a hypothetical bug elsewhere),
    begin_execution's own independent filter on validation.status
    still excludes it.
    """
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    flag_rec = _by_operation(run, "FLAG_OUTLIER")
    flag_rec.decision = RecommendationDecision.APPROVED_BY_USER  # bypass, simulating a bug

    plan = orchestrator.build_execution_plan(run)
    assert all(op.operation != "FLAG_OUTLIER" for op in plan.operations)


# ---------------------------------------------------------------------------
# Multiple recommendations with mixed validation/user decisions
# ---------------------------------------------------------------------------

def test_build_execution_plan_orders_operations_by_locked_precedence(spark, tmp_path, dataset_path):
    """
    Operations are approved in an order that does NOT match the
    locked precedence (FILL_MISSING before STANDARDIZE_CATEGORY before
    TRIM_WHITESPACE) -- build_execution_plan() must reorder them into
    the correct precedence order at plan-construction time, so the
    resulting ExecutionPlan is already correctly ordered before the
    executor ever sees it (the executor itself performs no reordering).
    """
    from converge.models.recommendation import Recommendation

    recs = [
        Recommendation(operation="FILL_MISSING", target_columns=["salary"], parameters={"strategy": "mean"}, reason="r"),
        Recommendation(operation="STANDARDIZE_CATEGORY", target_columns=["city"],
                       parameters={"mapping": {"Delhi": "Delhi", "delhi": "Delhi", "DELHI": "Delhi", "Delhi ": "Delhi"}}, reason="r"),
        Recommendation(operation="TRIM_WHITESPACE", target_columns=["city"], parameters={}, reason="r"),
    ]
    ai_provider = FakeAIProvider(recommendations=recs)
    run_store = RunStore(base_dir=tmp_path)
    orch = PipelineOrchestrator(spark=spark, ai_provider=ai_provider, run_store=run_store)

    run = orch.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orch.run_profiling(run)
    run = orch.run_ai_analysis(run)
    run = orch.run_validation(run)
    for record in run.recommendation_records:
        orch.submit_decision(run, record.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER)

    plan = orch.build_execution_plan(run)
    ops_in_order = [op.operation for op in plan.operations]
    # Locked precedence: FILL_MISSING(1) before TRIM_WHITESPACE/STANDARDIZE_CATEGORY(2)
    assert ops_in_order[0] == "FILL_MISSING"
    assert set(ops_in_order[1:]) == {"TRIM_WHITESPACE", "STANDARDIZE_CATEGORY"}


def test_mixed_recommendations_only_approved_and_eligible_reach_execution(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    city_rec = next(r for r in run.recommendation_records if r.recommendation.target_columns == ["city"])
    dept_rec = next(r for r in run.recommendation_records if r.recommendation.target_columns == ["department"])
    fill_rec = _by_operation(run, "FILL_MISSING")
    flag_rec = _by_operation(run, "FLAG_OUTLIER")

    assert city_rec.validation.status == ValidationStatus.APPROVED_AUTO
    assert fill_rec.validation.status == ValidationStatus.NEEDS_REVIEW
    assert dept_rec.validation.status == ValidationStatus.REJECTED
    assert flag_rec.validation.status == ValidationStatus.NOT_EXECUTABLE

    # Approve city (eligible); explicitly reject fill_missing and dept;
    # leave flag_rec untouched (PENDING).
    run = orchestrator.submit_decision(run, city_rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER)
    run = orchestrator.submit_decision(run, fill_rec.recommendation.recommendation_id, RecommendationDecision.REJECTED_BY_USER)
    run = orchestrator.submit_decision(run, dept_rec.recommendation.recommendation_id, RecommendationDecision.REJECTED_BY_USER)

    plan = orchestrator.build_execution_plan(run)

    eligible_ops = {op.operation for op in plan.operations}
    assert eligible_ops == {"STANDARDIZE_CATEGORY"}
    assert len(plan.operations) == 1
    assert plan.operations[0].target_columns == ("city",)


def test_recommendation_still_pending_decision_is_excluded_from_execution(orchestrator, dataset_path):
    """PENDING is the default and must never be treated as approved."""
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)
    # No decisions submitted at all.
    plan = orchestrator.build_execution_plan(run)
    assert plan.operations == ()


# ---------------------------------------------------------------------------
# Persisted intermediate artifacts
# ---------------------------------------------------------------------------

def test_profile_result_is_persisted_and_reloadable(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)

    reloaded = orchestrator._run_store.load(run.run_id)
    assert reloaded.status == PipelineStatus.PROFILED
    assert reloaded.profile_result.profile.row_count == 20
    assert len(reloaded.profile_result.evidence) == len(run.profile_result.evidence)


def test_recommendations_and_validation_are_persisted_and_reloadable(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    reloaded = orchestrator._run_store.load(run.run_id)
    assert len(reloaded.recommendation_records) == 4
    for record in reloaded.recommendation_records:
        assert record.validation is not None
        assert record.validation.status in (
            ValidationStatus.APPROVED_AUTO, ValidationStatus.NEEDS_REVIEW,
            ValidationStatus.REJECTED, ValidationStatus.NOT_EXECUTABLE,
        )


def test_run_json_file_actually_exists_on_disk(orchestrator, dataset_path, tmp_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    path = tmp_path / run.run_id / "run.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["dataset_id"] == "sample_hr"
    assert data["status"] == "ingested"


def test_decisions_are_persisted(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    fill_rec = _by_operation(run, "FILL_MISSING")
    orchestrator.submit_decision(run, fill_rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER)

    reloaded = orchestrator._run_store.load(run.run_id)
    reloaded_record = next(
        r for r in reloaded.recommendation_records
        if r.recommendation.recommendation_id == fill_rec.recommendation.recommendation_id
    )
    assert reloaded_record.decision == RecommendationDecision.APPROVED_BY_USER


# ---------------------------------------------------------------------------
# AI metadata persistence
# ---------------------------------------------------------------------------

def test_ai_metadata_is_captured_and_persisted(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)

    assert run.ai_metadata is not None
    assert run.ai_metadata.provider == "fake-provider"
    assert run.ai_metadata.model == "fake-model-v0"
    assert run.ai_metadata.prompt_version == "fake-prompt-v0"
    assert run.ai_metadata.recommendation_count == 4

    reloaded = orchestrator._run_store.load(run.run_id)
    assert reloaded.ai_metadata.provider == "fake-provider"
    assert reloaded.ai_metadata.model == "fake-model-v0"
    assert reloaded.ai_metadata.prompt_version == "fake-prompt-v0"


# ---------------------------------------------------------------------------
# Additional guards
# ---------------------------------------------------------------------------

def test_submit_decision_for_unknown_recommendation_id_raises(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)

    with pytest.raises(RecommendationNotFoundError):
        orchestrator.submit_decision(run, "does-not-exist", RecommendationDecision.APPROVED_BY_USER)


def test_submit_decision_before_awaiting_approval_is_rejected(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    # status is AI_ANALYZED, not AWAITING_APPROVAL
    with pytest.raises(InvalidStateTransitionError):
        orchestrator.submit_decision(run, "whatever", RecommendationDecision.APPROVED_BY_USER)


def test_complete_execution_before_begin_execution_is_rejected(orchestrator, dataset_path):
    run = orchestrator.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orchestrator.run_profiling(run)
    run = orchestrator.run_ai_analysis(run)
    run = orchestrator.run_validation(run)
    with pytest.raises(InvalidStateTransitionError):
        orchestrator.complete_execution(run, ExecutionSummary())


def test_zero_recommendations_from_ai_is_a_valid_outcome(spark, tmp_path, dataset_path):
    """An empty recommendation list is not an error -- the AI stage
    contract explicitly allows 'no recommendations' as valid."""
    ai_provider = FakeAIProvider(recommendations=[])
    run_store = RunStore(base_dir=tmp_path)
    orch = PipelineOrchestrator(spark=spark, ai_provider=ai_provider, run_store=run_store)

    run = orch.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orch.run_profiling(run)
    run = orch.run_ai_analysis(run)
    assert run.status == PipelineStatus.AI_ANALYZED
    assert run.recommendation_records == []

    run = orch.run_validation(run)
    assert run.status == PipelineStatus.AWAITING_APPROVAL

    plan = orch.build_execution_plan(run)
    assert plan.operations == ()

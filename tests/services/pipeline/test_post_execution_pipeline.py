"""
Step 6 integration tests: post-execution profiling, deterministic
comparison, and insight generation, wired into the real orchestrator
state machine, against the real Spark profiler and real sample_hr.csv
dataset. Only the AI provider is faked (same pattern used throughout
this project since Step 3).
"""
import pytest

from converge.core.config import RAW_DATA_DIR
from converge.core.exceptions import InvalidStateTransitionError, PipelineStageError
from converge.models.execution import ExecutionSummary
from converge.models.pipeline import PipelineStatus, RecommendationDecision
from converge.models.recommendation import Recommendation
from converge.services.ai.base import AIProvider
from converge.services.pipeline.orchestrator import PipelineOrchestrator
from converge.storage.run_store import RunStore


class FakeAIProvider(AIProvider):
    def __init__(self, recommendations):
        self._recommendations = recommendations

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
        return list(self._recommendations)


def _standard_recommendations():
    return [
        Recommendation(
            operation="STANDARDIZE_CATEGORY", target_columns=["city"],
            parameters={"mapping": {"Delhi": "Delhi", "delhi": "Delhi", "DELHI": "Delhi", "Delhi ": "Delhi"}},
            reason="case/whitespace variants of Delhi",
        ),
        Recommendation(
            operation="FILL_MISSING", target_columns=["salary"], parameters={"strategy": "mean"},
            reason="salary has missing values",
        ),
    ]


@pytest.fixture()
def dataset_path():
    return str(RAW_DATA_DIR / "sample_hr.csv")


def _run_to_cleaned(spark, tmp_path, dataset_path, recommendations=None):
    ai_provider = FakeAIProvider(recommendations or _standard_recommendations())
    run_store = RunStore(base_dir=tmp_path)
    orch = PipelineOrchestrator(spark=spark, ai_provider=ai_provider, run_store=run_store)

    run = orch.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orch.run_profiling(run)
    run = orch.run_ai_analysis(run)
    run = orch.run_validation(run)
    for record in run.recommendation_records:
        if record.validation and record.validation.status.value in ("approved_auto", "needs_review"):
            orch.submit_decision(run, record.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER)
        else:
            orch.submit_decision(run, record.recommendation.recommendation_id, RecommendationDecision.REJECTED_BY_USER)
    run = orch.execute(run)
    assert run.status == PipelineStatus.CLEANED
    return orch, run


# ---------------------------------------------------------------------------
# Full loop: CLEANED -> STATS_COMPUTED -> INSIGHTS_GENERATED -> COMPLETED
# ---------------------------------------------------------------------------

def test_full_post_execution_loop_reaches_completed(spark, tmp_path, dataset_path):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)

    run = orch.run_post_execution_analysis(run)
    assert run.status == PipelineStatus.STATS_COMPUTED
    assert run.after_profile_result is not None
    assert run.quality_comparison is not None
    assert run.recommendation_outcomes

    run = orch.generate_insights(run)
    assert run.status == PipelineStatus.INSIGHTS_GENERATED
    assert run.insight_artifact is not None
    assert len(run.insight_artifact.insights) > 0

    run = orch.complete_run(run)
    assert run.status == PipelineStatus.COMPLETED


def test_before_profile_is_never_overwritten(spark, tmp_path, dataset_path):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)
    before_snapshot = run.profile_result.model_dump_json()

    run = orch.run_post_execution_analysis(run)

    assert run.profile_result.model_dump_json() == before_snapshot
    assert run.after_profile_result.profile.dataset_id != run.profile_result.profile.dataset_id
    # The city standardization + salary fill should be reflected in the AFTER profile only
    assert run.after_profile_result.profile.row_count == run.profile_result.profile.row_count


def test_after_profile_reflects_real_changes(spark, tmp_path, dataset_path):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)
    run = orch.run_post_execution_analysis(run)

    salary_before = next(c for c in run.profile_result.profile.columns if c.name == "salary")
    salary_after = next(c for c in run.after_profile_result.profile.columns if c.name == "salary")
    assert salary_before.null_count == 2
    assert salary_after.null_count == 0  # FILL_MISSING resolved it

    fill_missing_comparison = next(
        ic for ic in run.quality_comparison.issue_comparisons
        if ic.issue_type.value == "missing_values" and ic.columns == ["salary"]
    )
    assert fill_missing_comparison.outcome.value == "resolved"


# ---------------------------------------------------------------------------
# STATS_COMPUTED must not be reached on comparison failure
# ---------------------------------------------------------------------------

def test_pipeline_does_not_transition_to_stats_computed_when_comparison_fails(spark, tmp_path, dataset_path, monkeypatch):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)

    def _boom(self, before, after):
        raise RuntimeError("simulated comparator failure")

    monkeypatch.setattr(
        "converge.services.pipeline.orchestrator.QualityComparator.compare", _boom
    )

    with pytest.raises(PipelineStageError):
        orch.run_post_execution_analysis(run)

    assert run.status == PipelineStatus.FAILED
    assert "post_execution_analysis" in run.error
    assert run.quality_comparison is None  # never got set


def test_pipeline_does_not_transition_to_stats_computed_when_output_path_missing(spark, tmp_path, dataset_path):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)
    run.execution_summary.output_dataset_path = None  # simulate a corrupted/incomplete artifact

    with pytest.raises(PipelineStageError):
        orch.run_post_execution_analysis(run)

    assert run.status == PipelineStatus.FAILED


def test_run_post_execution_analysis_before_cleaned_is_rejected(spark, tmp_path, dataset_path):
    ai_provider = FakeAIProvider(_standard_recommendations())
    run_store = RunStore(base_dir=tmp_path)
    orch = PipelineOrchestrator(spark=spark, ai_provider=ai_provider, run_store=run_store)
    run = orch.start_run(dataset_id="sample_hr", dataset_path=dataset_path)
    run = orch.run_profiling(run)
    with pytest.raises(InvalidStateTransitionError):
        orch.run_post_execution_analysis(run)


# ---------------------------------------------------------------------------
# Insight generation failure leaves the comparison intact
# ---------------------------------------------------------------------------

def test_insight_generation_failure_does_not_erase_comparison(spark, tmp_path, dataset_path, monkeypatch):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)
    run = orch.run_post_execution_analysis(run)
    comparison_snapshot = run.quality_comparison.model_dump_json()
    outcomes_snapshot = [o.model_dump_json() for o in run.recommendation_outcomes]

    def _boom(self, comparison, outcomes):
        raise RuntimeError("simulated insight generator failure")

    monkeypatch.setattr(
        "converge.services.pipeline.orchestrator.InsightGenerator.generate", _boom
    )

    with pytest.raises(PipelineStageError):
        orch.generate_insights(run)

    assert run.status == PipelineStatus.FAILED
    assert run.insight_artifact is None
    # comparison + outcomes from the PRIOR successful stage are untouched
    assert run.quality_comparison.model_dump_json() == comparison_snapshot
    assert [o.model_dump_json() for o in run.recommendation_outcomes] == outcomes_snapshot


def test_generate_insights_before_stats_computed_is_rejected(spark, tmp_path, dataset_path):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)
    with pytest.raises(InvalidStateTransitionError):
        orch.generate_insights(run)


# ---------------------------------------------------------------------------
# Persistence: reload with before profile, after profile, comparison, insights
# ---------------------------------------------------------------------------

def test_persisted_run_reloads_with_all_step6_artifacts(spark, tmp_path, dataset_path):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)
    run = orch.run_post_execution_analysis(run)
    run = orch.generate_insights(run)
    run = orch.complete_run(run)

    reloaded = orch._run_store.load(run.run_id)
    assert reloaded.status == PipelineStatus.COMPLETED
    assert reloaded.profile_result is not None
    assert reloaded.after_profile_result is not None
    assert reloaded.quality_comparison is not None
    assert reloaded.insight_artifact is not None
    assert len(reloaded.recommendation_outcomes) == len(run.recommendation_outcomes)
    # before/after profiles remain independently distinguishable
    assert reloaded.profile_result.profile.dataset_id != reloaded.after_profile_result.profile.dataset_id


# ---------------------------------------------------------------------------
# Traceability end-to-end (real recommendation -> real execution -> real comparison)
# ---------------------------------------------------------------------------

def test_end_to_end_traceability_for_standardize_category(spark, tmp_path, dataset_path):
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path)
    run = orch.run_post_execution_analysis(run)

    city_outcome = next(o for o in run.recommendation_outcomes if o.operation == "STANDARDIZE_CATEGORY")
    assert city_outcome.execution_status == "succeeded"
    assert city_outcome.linked_issue_comparisons  # at least one linked comparison
    linked = city_outcome.linked_issue_comparisons[0]
    assert linked.issue_type.value == "categorical_inconsistency"
    assert linked.columns == ["city"]
    assert linked.outcome.value in ("resolved", "improved")


# ---------------------------------------------------------------------------
# Multiple recommendations / mixed outcomes end-to-end
# ---------------------------------------------------------------------------

def test_full_loop_with_a_recommendation_that_was_never_approved(spark, tmp_path, dataset_path):
    """
    Includes a recommendation the validator rejects (fabricated
    department evidence) alongside the two normal ones -- confirms the
    full Step 6 loop handles a mixed-decision run correctly and the
    rejected one's outcome is honestly reported as not executed.
    """
    recs = _standard_recommendations() + [
        Recommendation(
            operation="STANDARDIZE_CATEGORY", target_columns=["department"],
            parameters={"mapping": {"Engineering": "Engineering"}},
            reason="fabricated",
        ),
    ]
    orch, run = _run_to_cleaned(spark, tmp_path, dataset_path, recommendations=recs)
    run = orch.run_post_execution_analysis(run)
    run = orch.generate_insights(run)

    dept_outcome = next(o for o in run.recommendation_outcomes if o.target_columns == ["department"])
    assert dept_outcome.decision == "rejected_by_user"
    assert dept_outcome.execution_status is None
    assert dept_outcome.linked_issue_comparisons == []

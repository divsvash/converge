"""
CONVERGE strategy tests. Real Spark, real sample_hr.csv dataset, a
fake AIProvider.
"""
from converge.models.pipeline import PipelineStatus
from converge.models.recommendation import Recommendation
from converge.models.validation import ValidationStatus
from converge.services.spark.loader import load_csv
from converge.storage.run_store import RunStore
from experiments.strategies.converge import default_approval_policy, run_converge
from tests.experiments.strategies.conftest import FakeAIProvider


# ---------------------------------------------------------------------------
# Actually delegates to PipelineOrchestrator (not a reimplementation)
# ---------------------------------------------------------------------------

def test_delegates_to_real_pipeline_orchestrator_state_machine(spark, sample_scenario, converge_configuration, run_id):
    """
    Proven by the fact that a real PipelineRun is persisted via the
    real RunStore at converge's own data/runs/{id}/run.json path --
    something that could only happen via the real orchestrator, since
    this strategy module contains no state-machine logic of its own.
    """
    provider = FakeAIProvider(recommendations=[
        Recommendation(operation="TRIM_WHITESPACE", target_columns=["full_name"], parameters={}, reason="r"),
    ])
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)

    assert outcome.success is True
    assert outcome.converge_run_id is not None

    store = RunStore()
    persisted = store.load(outcome.converge_run_id)
    assert persisted.status == PipelineStatus.CLEANED
    assert persisted.profile_result is not None
    assert persisted.recommendation_records  # the real pipeline actually ran AI analysis + validation


def test_module_contains_no_spark_dataframe_code():
    """
    Structural check: converge.py (the strategy) must not import
    pyspark DataFrame/functions directly -- all Spark work happens
    inside the real orchestrator's dependencies, never reimplemented
    here.
    """
    import inspect
    from experiments.strategies import converge as converge_strategy_module
    source = inspect.getsource(converge_strategy_module)
    assert "import pyspark.sql.functions" not in source
    assert "DataFrame" not in source


# ---------------------------------------------------------------------------
# Validation remains active
# ---------------------------------------------------------------------------

def test_fabricated_evidence_recommendation_is_rejected_by_the_real_validator(
    spark, sample_scenario, converge_configuration, run_id,
):
    """
    Unlike AI_ONLY, the same fabricated recommendation here goes
    through the real Validator and must be rejected (never reach
    execution) -- this is the direct behavioral proof that validation
    remains active for the CONVERGE strategy.
    """
    fabricated = Recommendation(
        operation="STANDARDIZE_CATEGORY", target_columns=["department"],
        parameters={"mapping": {"Engineering": "Engineering"}},
        reason="fabricated -- department has no real inconsistency",
    )
    provider = FakeAIProvider(recommendations=[fabricated])
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)

    store = RunStore()
    persisted = store.load(outcome.converge_run_id)
    record = persisted.recommendation_records[0]
    assert record.validation.status == ValidationStatus.REJECTED
    # and therefore never approved, never executed:
    assert record.decision.value == "rejected_by_user"


def test_forbidden_operation_is_rejected_by_the_real_validator(spark, sample_scenario, converge_configuration, run_id):
    malicious = Recommendation(operation="DELETE_COLUMN", target_columns=["salary"], parameters={}, reason="bad")
    provider = FakeAIProvider(recommendations=[malicious])
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)

    store = RunStore()
    persisted = store.load(outcome.converge_run_id)
    assert persisted.recommendation_records[0].validation.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# Experiment approval behavior is explicit
# ---------------------------------------------------------------------------

def test_default_approval_policy_approves_only_executable_statuses():
    assert default_approval_policy(ValidationStatus.APPROVED_AUTO) is True
    assert default_approval_policy(ValidationStatus.NEEDS_REVIEW) is True
    assert default_approval_policy(ValidationStatus.REJECTED) is False
    assert default_approval_policy(ValidationStatus.NOT_EXECUTABLE) is False


def test_approved_auto_recommendation_is_actually_approved_and_executed(
    spark, sample_scenario, converge_configuration, run_id,
):
    valid = Recommendation(operation="TRIM_WHITESPACE", target_columns=["full_name"], parameters={}, reason="r")
    provider = FakeAIProvider(recommendations=[valid])
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)

    store = RunStore()
    persisted = store.load(outcome.converge_run_id)
    record = persisted.recommendation_records[0]
    assert record.validation.status == ValidationStatus.APPROVED_AUTO
    assert record.decision.value == "approved_by_user"
    assert outcome.success is True


def test_custom_approval_policy_that_rejects_everything_results_in_no_execution(
    spark, sample_scenario, converge_configuration, run_id,
):
    valid = Recommendation(operation="TRIM_WHITESPACE", target_columns=["full_name"], parameters={}, reason="r")
    provider = FakeAIProvider(recommendations=[valid])

    reject_everything = lambda status: False
    outcome = run_converge(
        sample_scenario, converge_configuration, run_id, spark, ai_provider=provider,
        approval_policy=reject_everything,
    )

    assert outcome.success is True  # zero approved ops is still a valid, successful execution
    store = RunStore()
    persisted = store.load(outcome.converge_run_id)
    assert persisted.recommendation_records[0].decision.value == "rejected_by_user"
    assert persisted.execution_summary.operations_attempted == 0


def test_policy_cannot_approve_a_rejected_recommendation_even_if_it_tries(
    spark, sample_scenario, converge_configuration, run_id,
):
    """
    Even a maliciously-permissive policy cannot approve something the
    real Validator rejected -- the orchestrator's own submit_decision()
    independently refuses this (RecommendationNotEligibleError), which
    this strategy must handle cleanly rather than letting it crash the
    whole run.
    """
    fabricated = Recommendation(
        operation="STANDARDIZE_CATEGORY", target_columns=["department"],
        parameters={"mapping": {"Engineering": "Engineering"}}, reason="fabricated",
    )
    provider = FakeAIProvider(recommendations=[fabricated])
    approve_everything = lambda status: True

    outcome = run_converge(
        sample_scenario, converge_configuration, run_id, spark, ai_provider=provider,
        approval_policy=approve_everything,
    )
    # Must not crash; the orchestrator's own guard still applies, and
    # since the only recommendation is REJECTED, decision falls back
    # to REJECTED_BY_USER (the policy's "True" is never actually
    # honored for an ineligible recommendation).
    store = RunStore()
    persisted = store.load(outcome.converge_run_id)
    assert persisted.recommendation_records[0].decision.value == "rejected_by_user"


# ---------------------------------------------------------------------------
# Returns/links the underlying converge run identity
# ---------------------------------------------------------------------------

def test_converge_run_id_is_set_on_success(spark, sample_scenario, converge_configuration, run_id):
    provider = FakeAIProvider(recommendations=[])
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)
    assert outcome.converge_run_id is not None
    assert RunStore().exists(outcome.converge_run_id)


def test_output_dataset_path_matches_the_persisted_pipeline_run(spark, sample_scenario, converge_configuration, run_id):
    provider = FakeAIProvider(recommendations=[])
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)
    persisted = RunStore().load(outcome.converge_run_id)
    assert outcome.output_dataset_path == persisted.execution_summary.output_dataset_path


def test_strategy_stops_before_post_execution_analysis_fairness_rule(
    spark, sample_scenario, converge_configuration, run_id,
):
    """
    FAIRNESS: the persisted PipelineRun must be left at CLEANED, not
    advanced to STATS_COMPUTED/INSIGHTS_GENERATED/COMPLETED -- this
    strategy must never compute its own quality comparison.
    """
    provider = FakeAIProvider(recommendations=[])
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)
    persisted = RunStore().load(outcome.converge_run_id)
    assert persisted.status == PipelineStatus.CLEANED
    assert persisted.quality_comparison is None
    assert persisted.after_profile_result is None
    assert persisted.insight_artifact is None


# ---------------------------------------------------------------------------
# Pipeline failures become strategy failures
# ---------------------------------------------------------------------------

def test_missing_ai_provider_fails_cleanly(spark, sample_scenario, converge_configuration, run_id):
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=None)
    assert outcome.success is False
    assert "ai_provider" in outcome.error.lower()
    assert outcome.converge_run_id is None  # no PipelineRun was ever created


def test_ai_analysis_failure_becomes_a_strategy_failure_with_run_id_preserved(
    spark, sample_scenario, converge_configuration, run_id,
):
    from converge.core.exceptions import AIProviderAPIError

    provider = FakeAIProvider(raise_exc=AIProviderAPIError("simulated upstream failure"))
    outcome = run_converge(sample_scenario, converge_configuration, run_id, spark, ai_provider=provider)

    assert outcome.success is False
    assert outcome.error is not None
    # a PipelineRun WAS created (profiling succeeded) before AI analysis failed,
    # so its id should still be reported -- failure is research data.
    assert outcome.converge_run_id is not None
    persisted = RunStore().load(outcome.converge_run_id)
    assert persisted.status == PipelineStatus.FAILED


def test_nonexistent_dataset_path_fails_cleanly(spark, converge_configuration, run_id):
    from experiments.models.scenario import Dataset, Scenario

    # Construct a Dataset descriptor pointing at a file that doesn't exist
    # on disk (bypassing from_path's checksum computation, which would
    # itself fail for a missing file -- here we're testing failure
    # further downstream, at profiling time).
    fake_dataset = Dataset(dataset_id="missing", path="/nonexistent/path/does_not_exist.csv", checksum="a" * 64)
    fake_scenario = Scenario(scenario_id="missing_scenario", name="Missing", dataset=fake_dataset)

    provider = FakeAIProvider(recommendations=[])
    outcome = run_converge(fake_scenario, converge_configuration, run_id, spark, ai_provider=provider)
    assert outcome.success is False
    assert outcome.error is not None

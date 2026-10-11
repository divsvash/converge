"""
Strategy C: CONVERGE -- a thin adapter around the real, existing
converge.services.pipeline.orchestrator.PipelineOrchestrator. Nothing
about the pipeline is reimplemented here; this module's only job is
configuration and invocation.

FAIRNESS: this strategy deliberately stops once SparkExecutor finishes
(status CLEANED) and does NOT call
orchestrator.run_post_execution_analysis() / generate_insights() /
complete_run() -- even though the orchestrator is fully capable of
continuing all the way to COMPLETED. This is the direct, structural
expression of the fairness rule: no strategy computes its own quality
comparison. Phase 9's common measurement layer will profile and
compare every strategy's before/output dataset identically and
externally, including CONVERGE's -- stopping here means CONVERGE
cannot get a head start on its own grading just because its own
pipeline happens to be capable of producing one.

EXPERIMENT APPROVAL POLICY: experiments run unattended, so something
has to stand in for the human "click Apply" step the production
system expects. `default_approval_policy` is that policy, and it is
visible, documented, and deliberately narrow: it approves a
recommendation if and only if converge's real Validator already judged
it executable (APPROVED_AUTO or NEEDS_REVIEW). Validation itself is
NEVER bypassed -- every recommendation still goes through the real
Validator exactly as production CONVERGE does; this policy only
decides what happens to the Validator's verdict.

A REJECTED or NOT_EXECUTABLE recommendation is never approved, under
any circumstance, even by a custom policy that tries to. This is
enforced TWICE, deliberately redundantly: the orchestrator's own
submit_decision() independently refuses to approve an ineligible
recommendation (RecommendationNotEligibleError), AND this module never
even attempts to -- `_decide` below ANDs the configured policy's
answer with converge's own EXECUTABLE_VALIDATION_STATUSES check, so an
over-permissive custom policy degrades to "reject that one
recommendation" rather than raising and failing the entire run. A
policy can only ever narrow what validation already allowed, never
widen it, and a misbehaving policy fails safe rather than failing the
whole experiment run.
"""
from __future__ import annotations

from typing import Callable, Optional

from pyspark.sql import SparkSession

from converge.core.exceptions import ConvergeError
from converge.models.pipeline import PipelineStatus, RecommendationDecision
from converge.models.validation import ValidationStatus
from converge.services.ai.base import AIProvider
from converge.services.pipeline.orchestrator import EXECUTABLE_VALIDATION_STATUSES, PipelineOrchestrator
from experiments.models.configuration import Configuration
from experiments.models.scenario import Scenario
from experiments.strategies.common import StrategyOutcome

ApprovalPolicy = Callable[[ValidationStatus], bool]


def default_approval_policy(validation_status: ValidationStatus) -> bool:
    """
    Approve iff the real Validator already judged the recommendation
    executable (APPROVED_AUTO or NEEDS_REVIEW). This policy only ever
    narrows what validation already allowed; it never widens it.
    """
    return validation_status in (ValidationStatus.APPROVED_AUTO, ValidationStatus.NEEDS_REVIEW)


def _decide(
    validation_status: Optional[ValidationStatus], approval_policy: ApprovalPolicy
) -> RecommendationDecision:
    """
    ANDs the configured policy's answer with converge's own
    EXECUTABLE_VALIDATION_STATUSES check -- see module docstring for
    why this specific redundancy exists (a misbehaving policy must
    fail safe, not fail the whole run).
    """
    validator_eligible = validation_status is not None and validation_status in EXECUTABLE_VALIDATION_STATUSES
    policy_wants_approval = validation_status is not None and approval_policy(validation_status)
    if validator_eligible and policy_wants_approval:
        return RecommendationDecision.APPROVED_BY_USER
    return RecommendationDecision.REJECTED_BY_USER


def run_converge(
    scenario: Scenario,
    configuration: Configuration,
    run_id: str,
    spark: SparkSession,
    ai_provider: Optional[AIProvider] = None,
    approval_policy: ApprovalPolicy = default_approval_policy,
) -> StrategyOutcome:
    if ai_provider is None:
        return StrategyOutcome(success=False, error="run_converge requires an ai_provider; none was supplied.")

    orchestrator = PipelineOrchestrator(spark=spark, ai_provider=ai_provider)
    pipeline_run = None
    try:
        dataset_id = f"{scenario.scenario_id}__{run_id}__converge"
        pipeline_run = orchestrator.start_run(dataset_id=dataset_id, dataset_path=scenario.dataset.path)
        pipeline_run = orchestrator.run_profiling(pipeline_run)
        pipeline_run = orchestrator.run_ai_analysis(pipeline_run)
        pipeline_run = orchestrator.run_validation(pipeline_run)

        for record in pipeline_run.recommendation_records:
            decision = _decide(record.validation.status if record.validation else None, approval_policy)
            pipeline_run = orchestrator.submit_decision(
                pipeline_run, record.recommendation.recommendation_id, decision
            )

        pipeline_run = orchestrator.execute(pipeline_run)

    except ConvergeError as e:
        converge_run_id = pipeline_run.run_id if pipeline_run is not None else None
        return StrategyOutcome(success=False, error=str(e), converge_run_id=converge_run_id)
    except Exception as e:
        converge_run_id = pipeline_run.run_id if pipeline_run is not None else None
        return StrategyOutcome(success=False, error=f"Unexpected error: {e}", converge_run_id=converge_run_id)

    if pipeline_run.status != PipelineStatus.CLEANED:
        return StrategyOutcome(
            success=False,
            error=pipeline_run.error or f"Pipeline ended in unexpected status '{pipeline_run.status.value}'.",
            converge_run_id=pipeline_run.run_id,
        )

    return StrategyOutcome(
        success=True,
        output_dataset_path=pipeline_run.execution_summary.output_dataset_path,
        converge_run_id=pipeline_run.run_id,
        trace={
            "note": "Full traceability (profile, evidence, recommendations, validation "
                     "results, decisions, timeline) is on the referenced PipelineRun -- "
                     "see converge_run_id -- and is not duplicated here.",
        },
    )

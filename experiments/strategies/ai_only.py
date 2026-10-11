"""
Strategy B: AI_ONLY -- AI-directed cleaning WITHOUT CONVERGE's
evidence-validation / safety architecture.

Dataset -> converge's own Spark loader/profiler (the SAME bounded
profile/evidence contract CONVERGE's own AI stage uses) -> AIProvider.
generate_recommendations() (the SAME call CONVERGE's own AI stage
makes -- no duplicated prompting logic) -> a minimal STRUCTURAL
translation step -> converge's own SparkExecutor (the SAME execution
engine CONVERGE uses, with the SAME fail-fast plan semantics -- the
comparison is about cleaning decision architecture, not about running
a different execution engine for this arm).

CRITICAL DISTINCTION from CONVERGE: converge.services.validation is
never imported here, anywhere (see
tests/experiments/strategies/test_ai_only.py's
test_module_never_imports_the_validator, which scans this file's
actual imports). The Validator's evidence/safety checks -- the actual
subject of the paper's objective 2 -- are not run at all for this
strategy. That absence is the entire point of this baseline: it
measures what AI-directed cleaning looks like without the thing
CONVERGE adds.

SAFETY AT THE SOFTWARE BOUNDARY (not a validation gate): an AI
recommendation can still only reach Spark if it survives a minimal,
purely STRUCTURAL check -- is `operation` a name converge's operation
catalog actually recognizes and can execute at all (not forbidden, not
finding-only), and does it name at least one target column. This is a
fact about the operation catalog and the executor's own capabilities,
not a judgment about whether the recommendation is semantically
supported by evidence (that judgment -- the real Validator's job -- is
exactly what this strategy must not perform).

Parameter-level correctness is deliberately NOT checked here:
converge's own SparkExecutor already re-validates every parameter
itself (built in Step 5 specifically as defense-in-depth against
exactly this kind of untrusted input), so a structurally-admitted but
malformed recommendation fails cleanly inside the SAME executor every
other strategy uses, rather than this module reimplementing parameter
validation and drifting from the executor's own rules.

No eval/exec of anything, anywhere -- recommendations are only ever
interpreted as (operation name, target columns, parameters) and handed
to PlannedOperation/SparkExecutor exactly as CONVERGE itself does;
there is no code path through which AI-generated text could become
executable Python, SQL, or shell content. "AI-only" means AI owns the
cleaning DECISION, not that the model receives unrestricted execution.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pyspark.sql import SparkSession

from converge.core.exceptions import AIProviderError
from converge.models.execution import ExecutionPlan, PlannedOperation
from converge.models.operation import SafetyTier
from converge.models.operation_catalog import FORBIDDEN_OPERATIONS, OPERATION_REGISTRY
from converge.models.recommendation import Recommendation
from converge.services.ai.base import AIProvider
from converge.services.spark.executor import SparkExecutor
from converge.services.spark.loader import load_csv
from converge.services.spark.profiler import profile_dataset
from experiments.models.configuration import Configuration
from experiments.models.scenario import Scenario
from experiments.strategies.common import StrategyOutcome


class RecommendationTrace(BaseModel):
    """One AI recommendation and whether it was structurally executable."""
    recommendation_id: str
    operation: str
    target_columns: list[str]
    executable: bool
    reason: str


def run_ai_only(
    scenario: Scenario,
    configuration: Configuration,
    run_id: str,
    spark: SparkSession,
    ai_provider: Optional[AIProvider] = None,
) -> StrategyOutcome:
    if ai_provider is None:
        return StrategyOutcome(success=False, error="run_ai_only requires an ai_provider; none was supplied.")

    try:
        df = load_csv(spark, scenario.dataset.path)
        profile_result = profile_dataset(df, dataset_id=f"{scenario.scenario_id}__{run_id}__ai_only")
    except Exception as e:
        return StrategyOutcome(success=False, error=f"Profiling failed: {e}")

    try:
        recommendations = ai_provider.generate_recommendations(profile_result)
    except AIProviderError as e:
        return StrategyOutcome(success=False, error=f"AI provider failed: {e}")
    except Exception as e:
        return StrategyOutcome(success=False, error=f"Unexpected AI provider error: {e}")

    operations, traces = _structurally_translate(recommendations)
    trace = {"recommendations": [t.model_dump() for t in traces]}

    try:
        plan = ExecutionPlan(run_id=run_id, dataset_path=scenario.dataset.path, operations=tuple(operations))
        summary = SparkExecutor(spark).execute(plan)
    except Exception as e:
        return StrategyOutcome(success=False, error=f"Execution failed: {e}", trace=trace)

    if not summary.fully_successful:
        return StrategyOutcome(success=False, error=summary.notes, trace=trace)

    return StrategyOutcome(success=True, output_dataset_path=summary.output_dataset_path, trace=trace)


def _structurally_translate(
    recommendations: list[Recommendation],
) -> tuple[list[PlannedOperation], list[RecommendationTrace]]:
    """
    The ONLY filter AI_ONLY applies: is this operation name real and
    executable at all, and does it name a target column. No evidence
    check, no parameter check, no safety-tier check beyond excluding
    finding-only -- that is deliberately as far as "AI owns the
    decision" goes before converge's existing executor takes over.
    """
    operations: list[PlannedOperation] = []
    traces: list[RecommendationTrace] = []

    for rec in recommendations:
        op_def = OPERATION_REGISTRY.get(rec.operation)

        if rec.operation in FORBIDDEN_OPERATIONS or op_def is None:
            traces.append(RecommendationTrace(
                recommendation_id=rec.recommendation_id, operation=rec.operation,
                target_columns=list(rec.target_columns), executable=False,
                reason=f"'{rec.operation}' is not a recognized, executable P0 operation.",
            ))
            continue

        if op_def.safety_tier == SafetyTier.FINDING_ONLY:
            traces.append(RecommendationTrace(
                recommendation_id=rec.recommendation_id, operation=rec.operation,
                target_columns=list(rec.target_columns), executable=False,
                reason=f"'{rec.operation}' is finding-only and has no executable remediation.",
            ))
            continue

        if not rec.target_columns:
            traces.append(RecommendationTrace(
                recommendation_id=rec.recommendation_id, operation=rec.operation,
                target_columns=[], executable=False,
                reason="Recommendation specifies no target column(s).",
            ))
            continue

        operations.append(PlannedOperation(
            recommendation_id=rec.recommendation_id,
            operation=rec.operation,
            target_columns=tuple(rec.target_columns),
            parameters=dict(rec.parameters),
        ))
        traces.append(RecommendationTrace(
            recommendation_id=rec.recommendation_id, operation=rec.operation,
            target_columns=list(rec.target_columns), executable=True,
            reason="Passed structural execution checks (operation recognized, executable tier, has target columns).",
        ))

    # Deliberately NOT reordered by OPERATION_PRECEDENCE (contrast
    # with spark_only): the AI's own proposed order is preserved,
    # since imposing CONVERGE's curated sequencing would mean this
    # strategy no longer purely measures unaided AI judgment.
    return operations, traces

"""
Strategy A: SPARK_ONLY -- the deterministic, non-AI baseline.

Dataset -> converge's own Spark loader/profiler -> a small, explicit,
deterministic cleaning-policy mapping -> converge's own SparkExecutor
-> output dataset.

No AIProvider is ever imported or instantiated here. This module has
no import of converge.services.ai anywhere in its source, which is
what "never invokes AI" means structurally, not just behaviorally --
see tests/experiments/strategies/test_spark_only.py's
test_module_never_imports_ai, which scans this file's actual imports
rather than inferring the guarantee from behavior.

DETERMINISTIC POLICY -- deliberately small, and deliberately smaller
than an earlier draft considered during implementation (see the Phase
8 report for the rejected alternative). Only two profiler issue types
get an executable action, because they are the only two where the
correct action is derivable from the evidence itself with no
synthesized value judgment:

    WHITESPACE_OR_CASING -> TRIM_WHITESPACE
        Zero parameters. Stripping incidental leading/trailing
        whitespace is the single correct remediation, not a choice
        among alternatives.
    DUPLICATE_ROWS -> REMOVE_DUPLICATES
        The criteria columns are already established BY THE PROFILER
        ITSELF (evidence.columns IS the duplicate-detection criteria
        the profiler used). Executing the removal only carries out a
        fact the profiler already determined deterministically; it is
        not a new decision this module makes.

Every other issue type the profiler can emit (MISSING_VALUES,
CATEGORICAL_INCONSISTENCY, DATE_FORMAT_INCONSISTENCY, OUTLIER_CANDIDATE)
requires choosing among multiple defensible values with no single
data-derived correct answer:
    - MISSING_VALUES: which imputation strategy (mean/median/mode/
      constant) is correct depends on the column's distribution and
      meaning, not just its presence.
    - CATEGORICAL_INCONSISTENCY: which raw variant is "canonical" is
      not determinable from the data alone -- majority frequency does
      not imply correctness.
    - DATE_FORMAT_INCONSISTENCY: the target format is a stylistic/
      locale convention, not something derivable from the data.
    - OUTLIER_CANDIDATE: FLAG_OUTLIER is finding-only in converge's own
      operation catalog -- there is no executable remediation for
      outliers at all, by design, for any strategy.

These are skipped, each with an explicit, specific, traceable reason,
rather than resolved by an invented heuristic. A deterministic
tie-breaking rule (e.g. "always use the most frequent variant") is
still a guess dressed up as a rule -- determinism is not the same
thing as correctness, and this module does not conflate the two.

Operation ORDER for the two executable actions reuses converge's own
OPERATION_PRECEDENCE constant (the same technical sequencing rule
production CONVERGE applies when building an ExecutionPlan) rather
than an ad hoc order of this module's own invention -- this is a
mechanical fact about operation dependencies (e.g. normalize before
deduplicating), not a "decision" in the sense this module is being
careful about elsewhere.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel
from pyspark.sql import SparkSession

from converge.models.execution import ExecutionPlan, PlannedOperation
from converge.models.profile import EvidenceItem, IssueType, ProfileResult
from converge.services.pipeline.orchestrator import OPERATION_PRECEDENCE
from converge.services.spark.executor import SparkExecutor
from converge.services.spark.loader import load_csv
from converge.services.spark.profiler import profile_dataset
from experiments.models.configuration import Configuration
from experiments.models.scenario import Scenario
from experiments.strategies.common import StrategyOutcome


class DeterministicDecision(BaseModel):
    """
    One traceable decision: an issue the profiler detected, and either
    the operation deterministically selected for it, or why it was
    skipped.
    """
    issue_type: str
    columns: list[str]
    action: str  # "applied" | "skipped"
    operation: Optional[str] = None
    reason: str


# The only two issue types with a synthesized-value-judgment-free
# remediation -- see module docstring for why only these two qualify.
_SKIP_REASONS: dict[IssueType, str] = {
    IssueType.MISSING_VALUES: (
        "Filling missing values requires choosing an imputation strategy "
        "(mean/median/mode/constant) -- no single strategy is correct for "
        "every column without judgment about the data's distribution and meaning."
    ),
    IssueType.CATEGORICAL_INCONSISTENCY: (
        "Which raw variant is the 'canonical' spelling is not determinable from "
        "the data alone (e.g. majority frequency does not imply correctness)."
    ),
    IssueType.DATE_FORMAT_INCONSISTENCY: (
        "The target date format is a stylistic/locale convention, not something "
        "derivable from the data itself."
    ),
    IssueType.OUTLIER_CANDIDATE: (
        "FLAG_OUTLIER is finding-only in converge's operation catalog -- there is "
        "no executable remediation for outliers at all, by design."
    ),
}


def run_spark_only(
    scenario: Scenario,
    configuration: Configuration,
    run_id: str,
    spark: SparkSession,
    ai_provider: Optional[Any] = None,
) -> StrategyOutcome:
    """
    `ai_provider` exists only to satisfy the uniform Strategy callable
    signature (experiments/strategies/common.py). It is never read,
    called, or referenced anywhere in this function's body, by design
    -- and deliberately typed as `Optional[Any]` rather than
    `Optional[AIProvider]` so this module never needs to import
    converge.services.ai at all, satisfying "must not import or
    instantiate an AIProvider" at the import-statement level, not just
    in the function body.
    """
    try:
        df = load_csv(spark, scenario.dataset.path)
        profile_result = profile_dataset(
            df, dataset_id=f"{scenario.scenario_id}__{run_id}__spark_only"
        )
    except Exception as e:
        return StrategyOutcome(success=False, error=f"Profiling failed: {e}")

    operations, decisions = _decide_actions(profile_result)
    trace = {"decisions": [d.model_dump() for d in decisions]}

    try:
        plan = ExecutionPlan(
            run_id=run_id, dataset_path=scenario.dataset.path, operations=tuple(operations)
        )
        summary = SparkExecutor(spark).execute(plan)
    except Exception as e:
        return StrategyOutcome(success=False, error=f"Execution failed: {e}", trace=trace)

    if not summary.fully_successful:
        return StrategyOutcome(success=False, error=summary.notes, trace=trace)

    return StrategyOutcome(success=True, output_dataset_path=summary.output_dataset_path, trace=trace)


def _decide_actions(
    profile_result: ProfileResult,
) -> tuple[list[PlannedOperation], list[DeterministicDecision]]:
    operations: list[PlannedOperation] = []
    decisions: list[DeterministicDecision] = []

    for item in profile_result.evidence:
        if item.issue_type == IssueType.WHITESPACE_OR_CASING:
            operations.append(PlannedOperation(
                recommendation_id=_decision_id(item),
                operation="TRIM_WHITESPACE",
                target_columns=tuple(item.columns),
                parameters={},
            ))
            decisions.append(DeterministicDecision(
                issue_type=item.issue_type.value, columns=item.columns,
                action="applied", operation="TRIM_WHITESPACE",
                reason="Whitespace is incidental; trimming it is the single correct remediation.",
            ))
        elif item.issue_type == IssueType.DUPLICATE_ROWS:
            operations.append(PlannedOperation(
                recommendation_id=_decision_id(item),
                operation="REMOVE_DUPLICATES",
                target_columns=tuple(item.columns),
                parameters={},
            ))
            decisions.append(DeterministicDecision(
                issue_type=item.issue_type.value, columns=item.columns,
                action="applied", operation="REMOVE_DUPLICATES",
                reason="Duplicate-detection criteria were already established by the profiler itself.",
            ))
        else:
            reason = _SKIP_REASONS.get(
                item.issue_type,
                f"No deterministic policy is defined for issue type '{item.issue_type.value}'.",
            )
            decisions.append(DeterministicDecision(
                issue_type=item.issue_type.value, columns=item.columns,
                action="skipped", operation=None, reason=reason,
            ))

    operations.sort(key=lambda op: OPERATION_PRECEDENCE.get(op.operation, len(OPERATION_PRECEDENCE)))
    return operations, decisions


def _decision_id(item: EvidenceItem) -> str:
    """Stable, deterministic id for traceability -- never random, so
    the same input always produces the same id."""
    return f"spark_only::{item.issue_type.value}::{'_'.join(item.columns)}"

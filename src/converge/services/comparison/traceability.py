"""
Recommendation -> validation -> decision -> execution -> outcome
traceability.

Connects every recommendation on a run to whichever IssueComparison
entries plausibly relate to it. The link is made via the SAME typed
mapping the Validator already relies on (OperationDefinition.
required_evidence_issue_types in the operation registry) plus
target-column overlap -- not a freshly-invented heuristic, and
certainly not an LLM judgment. When no reliable link can be made
(operation not executed, no comparable evidence, etc.) the outcome
says so honestly rather than inventing a connection.
"""
from __future__ import annotations

from converge.models.comparison import IssueComparison, QualityComparison, RecommendationOutcome
from converge.models.execution import ExecutionSummary, OperationExecutionStatus
from converge.models.pipeline import RecommendationRecord
from converge.models.operation_catalog import OPERATION_REGISTRY


def build_recommendation_outcomes(
    recommendation_records: list[RecommendationRecord],
    execution_summary: ExecutionSummary | None,
    comparison: QualityComparison,
) -> list[RecommendationOutcome]:
    exec_results_by_id = {
        r.recommendation_id: r for r in (execution_summary.operation_results if execution_summary else [])
    }

    outcomes: list[RecommendationOutcome] = []
    for record in recommendation_records:
        rec = record.recommendation
        exec_result = exec_results_by_id.get(rec.recommendation_id)
        op_def = OPERATION_REGISTRY.get(rec.operation)

        linked: list[IssueComparison] = []
        if exec_result is not None and exec_result.status == OperationExecutionStatus.SUCCEEDED and op_def is not None:
            addressed_types = set(op_def.required_evidence_issue_types)
            target_set = set(rec.target_columns)
            for ic in comparison.issue_comparisons:
                if ic.issue_type not in addressed_types:
                    continue
                ic_columns = set(ic.columns)
                if op_def.is_row_level:
                    if ic_columns == target_set:
                        linked.append(ic)
                elif target_set.issubset(ic_columns) or ic_columns.issubset(target_set):
                    linked.append(ic)

        outcomes.append(RecommendationOutcome(
            recommendation_id=rec.recommendation_id,
            operation=rec.operation,
            target_columns=list(rec.target_columns),
            decision=record.decision.value,
            validation_status=record.validation.status.value if record.validation else None,
            execution_status=exec_result.status.value if exec_result else None,
            rows_affected=exec_result.rows_affected if exec_result else None,
            execution_error=exec_result.error if exec_result else None,
            linked_issue_comparisons=linked,
            outcome_summary=_summarize(record, exec_result, linked),
        ))
    return outcomes


def _summarize(record: RecommendationRecord, exec_result, linked: list[IssueComparison]) -> str:
    decision = record.decision.value
    if decision != "approved_by_user":
        return f"Not executed (decision: {decision})."

    if exec_result is None:
        return "Approved by the user but not included in the execution plan."

    if exec_result.status == OperationExecutionStatus.FAILED:
        return f"Execution failed: {exec_result.error}"

    if exec_result.status == OperationExecutionStatus.SKIPPED:
        return "Skipped: an earlier operation in the plan failed (fail-fast)."

    # SUCCEEDED
    if not linked:
        return "Executed successfully; no directly comparable before/after evidence was found."

    parts = [
        f"{ic.issue_type.value} in {ic.columns}: {ic.outcome.value} ({ic.before_value} -> {ic.after_value})"
        for ic in linked
    ]
    return "Executed successfully. " + "; ".join(parts) + "."

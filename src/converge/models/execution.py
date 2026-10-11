"""
Execution plan / execution result contract.

This is the boundary described in the architecture correction:

    PipelineRun -> orchestrator determines eligibility -> ExecutionPlan
        -> SparkExecutor -> Spark

`ExecutionPlan` (and its `PlannedOperation` entries) is the ONLY thing
the executor is ever given. It deliberately does not carry:

    - the mutable PipelineRun (status, timeline, other recommendations,
      decisions not yet made, etc.) -- the executor has no way to
      inspect or depend on pipeline state it doesn't need.
    - `Recommendation.reason`, `.evidence_refs`, or `.confidence` --
      natural-language / claim fields that must never influence what
      Spark actually does. Only `operation` (a registry key),
      `target_columns` (already validated to exist), and `parameters`
      (already structurally validated) survive into a PlannedOperation.

The orchestrator is the only code that constructs an ExecutionPlan (in
`begin_execution()`), and it does so by filtering
`RecommendationRecord`s on both `decision == APPROVED_BY_USER` and
`validation.status` being validator-eligible -- so a PlannedOperation
existing in a plan at all is proof both checks already passed. The
executor is not expected to, and structurally cannot, re-derive or
re-check either of those two facts: it has no access to `decision` or
`validation` at all.

Both models are frozen (immutable after construction) and `operations`
is a tuple, not a list -- once a plan is built, nothing (including the
executor itself) can mutate it. This is "practical" immutability, not
deep-frozen: `parameters` remains a plain dict for JSON-serialization
simplicity, and Python does not prevent in-place mutation of a dict
even inside a frozen model. No P0 code path mutates it, and this
trade-off is documented rather than solved with a fully-frozen mapping
type, which would add real complexity for a prototype scope.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from converge.core.time_utils import utcnow


class PlannedOperation(BaseModel):
    """
    One operation the executor is authorized to perform. Derived
    entirely from a validator-eligible, user-approved
    RecommendationRecord -- stripped down to only the structured,
    already-validated fields the executor is permitted to act on.
    """
    model_config = ConfigDict(frozen=True)

    recommendation_id: str
    operation: str
    target_columns: tuple[str, ...]
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """
    The complete, immutable set of work the executor may perform for
    one pipeline run. Nothing outside this object describes what
    transformations are allowed to happen.
    """
    model_config = ConfigDict(frozen=True)

    run_id: str
    dataset_path: str
    operations: tuple[PlannedOperation, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Execution results (produced by the executor, consumed by the orchestrator)
# ---------------------------------------------------------------------------

class OperationExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Never attempted because an earlier operation in the plan failed
    (fail-fast execution -- see executor.py)."""


class OperationExecutionResult(BaseModel):
    recommendation_id: str
    operation: str
    target_columns: tuple[str, ...]
    status: OperationExecutionStatus
    rows_affected: Optional[int] = None
    error: Optional[str] = None


class ExecutionSummary(BaseModel):
    operations_attempted: int = 0
    operations_succeeded: int = 0
    operations_failed: int = 0
    operations_skipped: int = 0
    operation_results: list[OperationExecutionResult] = Field(default_factory=list)

    output_dataset_path: Optional[str] = None
    schema_before: list[dict[str, str]] = Field(default_factory=list)
    schema_after: list[dict[str, str]] = Field(default_factory=list)

    notes: str = ""

    @property
    def fully_successful(self) -> bool:
        """
        True only if every planned operation succeeded (no failures,
        no skips). This is what the orchestrator checks before ever
        allowing a transition to CLEANED.

        NOTE: "fully successful execution" and "the dataset is now
        clean" are different claims. This property answers only the
        former -- did every approved, executable transformation
        complete. It says nothing about whether data-quality issues
        remain; that requires re-profiling the output dataset (a
        later stage) and comparing it against the original profile.
        """
        return self.operations_failed == 0 and self.operations_skipped == 0

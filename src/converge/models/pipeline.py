"""
Pipeline run artifact models.

A `PipelineRun` is the single, file-persisted artifact that makes a
CONVERGE run inspectable end to end. It is intentionally a plain,
serializable Pydantic model with no behavior of its own -- all state
transition logic lives in the orchestrator (services/pipeline/
orchestrator.py), not here. This mirrors the same separation already
used for ProfileResult (data) vs. profiler.py (logic) and
Recommendation (data) vs. Validator (logic).

CRITICAL DISTINCTION preserved structurally by this schema (per the
architecture correction and Step 4 requirements): a single
RecommendationRecord carries three independent, separately-tracked
facts about one AI recommendation:

    1. `recommendation`   -- what the AI proposed (Step 3 output).
    2. `validation`       -- what the Validator independently
                              determined (Step 2 output). Never
                              mutated by user action.
    3. `decision`         -- what the user chose to do about it.
                              Starts PENDING for every recommendation,
                              including APPROVED_AUTO ones, and only
                              changes via an explicit orchestrator call
                              triggered by the user.

Execution eligibility is computed from (2) and (3) together, by the
orchestrator, at the moment execution is explicitly requested -- never
implicitly, and never from either field alone.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from converge.core.time_utils import utcnow
from converge.models.comparison import QualityComparison, RecommendationOutcome
from converge.models.execution import ExecutionPlan, ExecutionSummary
from converge.models.insight import InsightArtifact
from converge.models.profile import ProfileResult
from converge.models.recommendation import Recommendation
from converge.models.validation import ValidationResult


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    INGESTED = "ingested"
    PROFILED = "profiled"
    AI_ANALYZED = "ai_analyzed"
    VALIDATED = "validated"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    CLEANED = "cleaned"
    # SEMANTIC NOTE (architecture correction): CLEANED means "every
    # approved, executable transformation in the ExecutionPlan
    # completed successfully" -- a statement about execution
    # completeness, NOT a claim that the dataset now has no remaining
    # data-quality issues. Rejected recommendations, operations the
    # user chose not to approve, and issues nothing ever recommended
    # a fix for are all still present in the output. Whether the
    # dataset actually got measurably better is answered by
    # re-profiling the output (STATS_COMPUTED, below) and comparing it
    # against the original PROFILED result -- not by reaching this
    # status. The name is kept as-is (avoiding unnecessary churn); its
    # meaning is what's being made explicit here.
    STATS_COMPUTED = "stats_computed"
    # Re-profiling the executor's output dataset happens here (Step 6,
    # not yet implemented) -- this is the actual "did execution help"
    # measurement point: before-profile vs. after-profile.
    INSIGHTS_GENERATED = "insights_generated"
    COMPLETED = "completed"
    FAILED = "failed"
    # Not part of the "happy path" flow, but a first-class state: any
    # stage's failure lands here rather than leaving the run in a
    # status that looks like it succeeded. Terminal, like COMPLETED.


# Explicit transition table. A transition not listed here is invalid
# and rejected by the orchestrator, regardless of which method is
# called -- this is what "reject invalid transitions" means
# structurally rather than just by convention. FAILED is reachable
# from every non-terminal status (the orchestrator applies this
# uniformly rather than listing it here for each key) but COMPLETED
# and FAILED themselves are terminal: nothing follows them.
VALID_TRANSITIONS: dict[PipelineStatus, set[PipelineStatus]] = {
    PipelineStatus.INGESTED: {PipelineStatus.PROFILED},
    PipelineStatus.PROFILED: {PipelineStatus.AI_ANALYZED},
    PipelineStatus.AI_ANALYZED: {PipelineStatus.VALIDATED},
    PipelineStatus.VALIDATED: {PipelineStatus.AWAITING_APPROVAL},
    PipelineStatus.AWAITING_APPROVAL: {PipelineStatus.EXECUTING},
    PipelineStatus.EXECUTING: {PipelineStatus.CLEANED},
    PipelineStatus.CLEANED: {PipelineStatus.STATS_COMPUTED},
    PipelineStatus.STATS_COMPUTED: {PipelineStatus.INSIGHTS_GENERATED},
    PipelineStatus.INSIGHTS_GENERATED: {PipelineStatus.COMPLETED},
    PipelineStatus.COMPLETED: set(),
    PipelineStatus.FAILED: set(),
}

NON_TERMINAL_STATUSES = frozenset(
    s for s in PipelineStatus if s not in (PipelineStatus.COMPLETED, PipelineStatus.FAILED)
)


class TimelineEvent(BaseModel):
    """One entry in the run's execution timeline / log."""
    status: PipelineStatus
    message: str
    timestamp: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Recommendation tracking
# ---------------------------------------------------------------------------

class RecommendationDecision(str, Enum):
    PENDING = "pending"
    APPROVED_BY_USER = "approved_by_user"
    REJECTED_BY_USER = "rejected_by_user"


class RecommendationRecord(BaseModel):
    recommendation: Recommendation
    validation: Optional[ValidationResult] = None
    decision: RecommendationDecision = RecommendationDecision.PENDING


# ---------------------------------------------------------------------------
# AI stage metadata (for future model comparisons -- not built yet)
# ---------------------------------------------------------------------------

class AIStageMetadata(BaseModel):
    """
    Preserved per-run so that once multiple providers/models exist
    (explicitly out of scope for P0), historical runs already carry
    enough information to compare them. No comparison/experiment
    infrastructure is built in this step -- this is just the metadata
    capture.
    """
    provider: str
    model: str
    prompt_version: str
    recommendation_count: int
    generated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Execution plan / summary live in converge.models.execution (imported above).
# See that module for the ExecutionPlan/PlannedOperation contract -- the
# only input the Spark executor is ever given -- and ExecutionSummary,
# the structured result the orchestrator uses to decide CLEANED vs FAILED.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The run artifact
# ---------------------------------------------------------------------------

class PipelineRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_id: str
    dataset_path: str
    status: PipelineStatus = PipelineStatus.INGESTED

    profile_result: Optional[ProfileResult] = None
    """
    The BEFORE profile, produced by run_profiling() prior to any
    execution. Never overwritten or mutated by later stages -- see
    after_profile_result below for the post-execution counterpart.
    """
    ai_metadata: Optional[AIStageMetadata] = None
    recommendation_records: list[RecommendationRecord] = Field(default_factory=list)
    execution_plan: Optional[ExecutionPlan] = None
    execution_summary: Optional[ExecutionSummary] = None

    after_profile_result: Optional[ProfileResult] = None
    """
    The AFTER profile: the same deterministic Spark profiler
    (Step 1), re-run against the executor's output dataset. Kept
    entirely separate from `profile_result` (the before profile) so
    both remain independently inspectable -- this is what
    quality_comparison is computed from.
    """
    quality_comparison: Optional[QualityComparison] = None
    recommendation_outcomes: list[RecommendationOutcome] = Field(default_factory=list)
    insight_artifact: Optional[InsightArtifact] = None

    timeline: list[TimelineEvent] = Field(default_factory=list)
    error: Optional[str] = None

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

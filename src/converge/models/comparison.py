"""
Quality comparison models.

These are pure data models -- the comparison LOGIC lives in
converge.services.comparison.comparator (QualityComparator), which is
deterministic Python with no AI/LLM involvement anywhere. These models
are what the (optional, deterministic) InsightGenerator is later
allowed to read -- and nothing else.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from converge.core.time_utils import utcnow
from converge.models.profile import IssueType


class IssueOutcome(str, Enum):
    RESOLVED = "resolved"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    WORSENED = "worsened"
    """
    Not one of the 5 categories originally specified (RESOLVED,
    IMPROVED, UNCHANGED, INTRODUCED, UNVERIFIED) -- added because an
    issue that existed before, still exists after, and got
    measurably WORSE does not honestly fit any of those five.
    Folding it into UNCHANGED would misrepresent "equivalent" as
    covering "worse", and INTRODUCED is specifically defined as an
    issue that was absent before -- this one wasn't. Given the
    project's own principle ("do not invent comparisons... do not
    claim improvement merely because a metric changed"), silently
    mislabeling a regression seemed like the wrong trade-off versus
    adding one clearly-defined extra category. Flagged for review.
    """
    INTRODUCED = "introduced"
    UNVERIFIED = "unverified"


class IssueComparison(BaseModel):
    """
    One before/after comparison for a specific (issue_type, columns)
    pairing. `columns` is the full column set the issue is scoped to
    (a single column for column-scoped issue types; the criteria
    column set for DUPLICATE_ROWS).
    """
    issue_type: IssueType
    columns: list[str]
    outcome: IssueOutcome
    before_value: Optional[int] = Field(
        default=None, description="Deterministic severity metric before execution (None if the issue did not exist / was not measurable before)."
    )
    after_value: Optional[int] = Field(
        default=None, description="Deterministic severity metric after execution (None if not measurable after, e.g. UNVERIFIED)."
    )
    detail: str = Field(description="Human-readable explanation of the comparison basis -- what was measured and why.")


class ComparisonMetrics(BaseModel):
    """Deterministic aggregate metrics. No hidden/derived quality score."""

    total_issues_before: int
    total_issues_after: int

    resolved_count: int = 0
    improved_count: int = 0
    unchanged_count: int = 0
    worsened_count: int = 0
    introduced_count: int = 0
    unverified_count: int = 0

    issue_counts_by_type_before: dict[str, int] = Field(default_factory=dict)
    issue_counts_by_type_after: dict[str, int] = Field(default_factory=dict)

    row_count_before: int
    row_count_after: int
    column_count_before: int
    column_count_after: int
    schema_before: list[dict[str, str]] = Field(default_factory=list)
    schema_after: list[dict[str, str]] = Field(default_factory=list)

    exact_duplicate_row_count_before: int
    exact_duplicate_row_count_after: int
    duplicate_candidate_row_count_before: int = Field(
        description="Sum of (occurrence_count - 1) across duplicate-candidate groups -- the rows REMOVE_DUPLICATES would actually remove."
    )
    duplicate_candidate_row_count_after: int

    missing_value_total_before: int
    missing_value_total_after: int
    categorical_variant_total_before: int
    categorical_variant_total_after: int
    date_format_inconsistency_columns_before: int
    date_format_inconsistency_columns_after: int
    outlier_total_before: int
    outlier_total_after: int


class QualityComparison(BaseModel):
    """
    The complete deterministic before/after comparison for one run.
    This -- not any natural-language layer -- is the source of truth.
    """
    run_id: str
    issue_comparisons: list[IssueComparison] = Field(default_factory=list)
    metrics: ComparisonMetrics
    generated_at: datetime = Field(default_factory=utcnow)


class RecommendationOutcome(BaseModel):
    """
    Connects one recommendation through validation, user decision, and
    execution, to whichever IssueComparison entries plausibly relate
    to it (matched via the operation's declared evidence issue types
    in the operation registry, plus target-column overlap -- the same
    typed mapping the Validator already uses, not a re-derived
    heuristic). This is traceability, not causal proof: a recommendation
    that executed successfully and has zero linked comparisons yields
    an honest "no directly comparable evidence" outcome rather than a
    fabricated link.
    """
    recommendation_id: str
    operation: str
    target_columns: list[str]
    decision: str
    validation_status: Optional[str] = None
    execution_status: Optional[str] = None
    rows_affected: Optional[int] = None
    execution_error: Optional[str] = None
    linked_issue_comparisons: list[IssueComparison] = Field(default_factory=list)
    outcome_summary: str

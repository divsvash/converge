"""
Profile / Evidence contract.

This is the schema boundary between the Spark profiler (this step) and
every downstream stage (validator, AI provider, orchestrator). Nothing
downstream should ever read a raw Spark DataFrame or raw dict — only
these typed objects.

Two top-level objects are produced together by the profiler:

- DatasetProfile: aggregate, deterministic facts about the dataset.
  This is the "source of truth" the AI is allowed to explain but never
  allowed to contradict or recompute.

- EvidenceItem (list of): bounded, issue-scoped supporting data
  (representative samples and/or value distributions) that Spark
  decided is relevant to a specific detected issue. This is what makes
  semantic AI reasoning possible (e.g. proposing a categorical mapping)
  without giving the AI unrestricted raw dataset access, and it is also
  what the validator's evidence-support rule checks recommendations
  against.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Column-level profile
# ---------------------------------------------------------------------------

class SemanticType(str, Enum):
    """
    Coarse semantic classification, separate from the raw Spark dtype.
    Downstream stages (validator dtype-compatibility rule, AI reasoning)
    care about this more than the literal Spark type name.
    """
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATE_LIKE_STRING = "date_like_string"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class NumericStats(BaseModel):
    min: float
    max: float
    mean: float
    stddev: Optional[float] = None
    median_approx: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    outlier_lower_bound: Optional[float] = None
    outlier_upper_bound: Optional[float] = None
    outlier_count: int = 0


class CategoricalVariantGroup(BaseModel):
    """
    A group of distinct raw string values that normalize (lowercase +
    trim) to the same key -- i.e. candidate variants of one category.
    """
    normalized_key: str
    variants: dict[str, int] = Field(
        description="raw value -> occurrence count"
    )


class CategoricalStats(BaseModel):
    distinct_count: int
    top_values: dict[str, int] = Field(
        default_factory=dict,
        description="most frequent raw values -> occurrence count, bounded",
    )
    variant_groups: list[CategoricalVariantGroup] = Field(
        default_factory=list,
        description=(
            "groups where >1 raw variant maps to the same normalized "
            "key -- direct evidence for STANDARDIZE_CATEGORY"
        ),
    )
    whitespace_affected_count: int = Field(
        default=0,
        description="rows where the raw value differs from its trimmed form",
    )


class DateFormatStats(BaseModel):
    detected_patterns: dict[str, int] = Field(
        default_factory=dict,
        description="pattern label (e.g. 'YYYY-MM-DD') -> occurrence count",
    )
    unparseable_count: int = 0


class ColumnProfile(BaseModel):
    name: str
    spark_dtype: str
    semantic_type: SemanticType
    non_null_count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    is_id_like: bool = False

    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None
    date_format_stats: Optional[DateFormatStats] = None

    sample_values: list[str] = Field(
        default_factory=list,
        description="small bounded sample of raw values, for display/debug only",
    )


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class DuplicateGroup(BaseModel):
    """
    A group of rows that are identical when compared on a specific set
    of columns (criteria_columns). row_refs identifies the rows by the
    value of an id-like column when one exists, otherwise by row index.
    """
    criteria_columns: list[str]
    row_refs: list[str]
    occurrence_count: int


class DuplicateSummary(BaseModel):
    exact_duplicate_row_count: int = Field(
        description="rows identical across every column"
    )
    id_like_columns: list[str] = Field(default_factory=list)
    duplicate_candidates: list[DuplicateGroup] = Field(
        default_factory=list,
        description=(
            "rows identical when id-like columns are excluded -- "
            "candidate evidence for REMOVE_DUPLICATES, bounded in size"
        ),
    )


# ---------------------------------------------------------------------------
# Dataset-level profile
# ---------------------------------------------------------------------------

class DatasetProfile(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    duplicate_summary: DuplicateSummary
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Evidence contract (what the AI stage is allowed to see)
# ---------------------------------------------------------------------------

class IssueType(str, Enum):
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    CATEGORICAL_INCONSISTENCY = "categorical_inconsistency"
    WHITESPACE_OR_CASING = "whitespace_or_casing"
    DATE_FORMAT_INCONSISTENCY = "date_format_inconsistency"
    OUTLIER_CANDIDATE = "outlier_candidate"
    TYPE_INCONSISTENCY = "type_inconsistency"


class EvidenceItem(BaseModel):
    """
    One bounded, issue-scoped piece of supporting evidence. This is the
    unit the AI provider will receive (alongside the aggregate profile)
    and the unit the validator's evidence-support rule will check
    recommendations against.
    """
    issue_type: IssueType
    columns: list[str]
    description: str
    supporting_data: dict = Field(
        description=(
            "bounded structured evidence -- e.g. a value distribution "
            "slice or a small list of representative sample rows. "
            "Never the full dataset."
        )
    )


class ProfileResult(BaseModel):
    """
    The complete output of the profiler stage: the aggregate profile
    plus the bounded evidence bundle. This is what gets persisted as
    the run's profile/evidence artifact and consumed by every stage
    after it.
    """
    profile: DatasetProfile
    evidence: list[EvidenceItem]

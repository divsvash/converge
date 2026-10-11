"""
Result -- the measured per-Run outcome.

Three typed metric containers only: QualityMetrics, PerformanceMetrics,
CostMetrics. No metric CALCULATION logic here -- these are just the
typed shapes experiments/metrics/ populates. Fields are kept
deliberately minimal: only ones whose semantics can be defined
precisely, from what converge already exposes.

PHASE 9 EXTENSION NOTE: the original Phase 7 drafts of QualityMetrics
(8 outcome-count fields only) and PerformanceMetrics (a single
wall_clock_seconds field) were insufficient once Phase 9 specified its
actual required field list and its three-way timing boundary
(strategy execution / evaluation / total). This is a genuine,
behavior-driven model extension, not a redesign: every added field
mirrors a field that already exists on
converge.models.comparison.ComparisonMetrics (for QualityMetrics) or a
concrete timing boundary the runner actually measures (for
PerformanceMetrics). CostMetrics' `api_cost_usd` was renamed to
`monetary_cost` + `currency` to match Part F's exact requested field
list, and `total_tokens` was added alongside the existing
input/output token fields.

ReliabilityMetrics is deliberately NOT defined here and NOT attached
to Result or Run. Reliability (variance/consistency across repeated
runs) is only meaningful aggregated over every Run sharing one
(scenario_id, configuration_id) pair -- it cannot be a property of a
single Run, so it is not modeled as one. It lives in
experiments/metrics/reliability.py, computed over a group of Runs, not
stored per-Run.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    """
    Every field here mirrors a field already present on
    converge.models.comparison.ComparisonMetrics by the same name and
    semantics -- this model is a direct pass-through shape, not a
    reinterpretation. Derived from converge's own deterministic
    QualityComparison (experiments/metrics/quality.py does the
    mapping) -- never recomputed independently and never judged by an
    AI model. All fields optional: a QualityMetrics is only ever
    constructed when a real QualityComparison exists, but staying
    Optional keeps this model usable as an empty/partial shape too.

    Deliberately NOT included: any overall "quality score". These are
    observed deterministic before/after signals only -- a lower issue
    count is not treated here as proof of correct repair (that is a
    Phase 10 ground-truth concern).
    """
    total_issues_before: Optional[int] = None
    total_issues_after: Optional[int] = None
    resolved_count: Optional[int] = None
    improved_count: Optional[int] = None
    unchanged_count: Optional[int] = None
    worsened_count: Optional[int] = None
    introduced_count: Optional[int] = None
    unverified_count: Optional[int] = None

    row_count_before: Optional[int] = None
    row_count_after: Optional[int] = None
    column_count_before: Optional[int] = None
    column_count_after: Optional[int] = None

    missing_value_total_before: Optional[int] = None
    missing_value_total_after: Optional[int] = None

    exact_duplicate_row_count_before: Optional[int] = None
    exact_duplicate_row_count_after: Optional[int] = None
    duplicate_candidate_row_count_before: Optional[int] = None
    duplicate_candidate_row_count_after: Optional[int] = None

    categorical_variant_total_before: Optional[int] = None
    categorical_variant_total_after: Optional[int] = None

    date_format_inconsistency_columns_before: Optional[int] = None
    date_format_inconsistency_columns_after: Optional[int] = None

    outlier_total_before: Optional[int] = None
    outlier_total_after: Optional[int] = None


class PerformanceMetrics(BaseModel):
    """
    Three timing fields, matching three distinct, documented wall-clock
    boundaries the runner actually measures (see
    experiments/runner/runner.py and experiments/metrics/performance.py
    for the exact boundary definitions -- repeated here briefly):

        strategy_execution_wall_seconds -- time inside the strategy
            call only (experiments.strategies.*.run_*()).
        evaluation_wall_seconds -- time spent on the runner's OWN
            measurement work: dataset checksum verification, before-
            profiling, after-profiling, and comparison. Explicitly
            NOT strategy execution time.
        total_wall_seconds -- wall time for the entire Run attempt,
            start to finish, including both of the above plus any
            runner-level overhead between them.

    converge itself records no wall-clock timing anywhere (neither
    PipelineRun nor ExecutionSummary has a duration field), and is not
    modified to add one -- all three fields here are measured
    externally, around calls into converge/strategy code, never by
    changing converge itself.
    """
    strategy_execution_wall_seconds: Optional[float] = Field(default=None, ge=0)
    evaluation_wall_seconds: Optional[float] = Field(default=None, ge=0)
    total_wall_seconds: Optional[float] = Field(default=None, ge=0)


class CostMetrics(BaseModel):
    """
    AI cost, where applicable -- conservative by design. converge's
    AIProvider interface (converge.services.ai.base.AIProvider)
    exposes only provider_name/model_name/prompt_version; it has no
    token-usage or cost property, even though the underlying Gemini
    SDK response technically carries usage_metadata internally --
    GeminiProvider does not surface it, and this phase does not modify
    converge to add that. Consequently every field below is None for
    AI_ONLY and CONVERGE runs under the current AIProvider interface,
    NOT zero -- None means "unknown/unavailable", zero means
    "genuinely did not happen". For SPARK_ONLY, zero IS semantically
    correct (no AI call was ever made), so experiments/metrics/cost.py
    reports 0 there, not None. See that module's docstring for the
    exact distinction enforced.
    """
    monetary_cost: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = None
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)


class Result(BaseModel):
    quality: Optional[QualityMetrics] = None
    performance: Optional[PerformanceMetrics] = None
    cost: Optional[CostMetrics] = None
    ground_truth: Optional["GroundTruthResult"] = None


class GroundTruthResult(BaseModel):
    """
    Thin wrapper carrying GroundTruthMetrics on a per-Run Result.
    Intentionally a separate field from `quality` — they measure
    different things (observed deterministic before/after signals vs.
    known-correct repair accuracy) and must not be merged into one
    score.

    This model lives here (experiments.models.result) rather than
    importing experiments.scenarios.ground_truth.GroundTruthMetrics
    directly, to avoid a circular import: the runner builds Results
    (this module) and also calls the ground-truth evaluator
    (experiments.scenarios.ground_truth), and both of those already
    import from experiments.models. Duplicating the fields here as a
    flat pass-through shape (same pattern QualityMetrics uses for
    ComparisonMetrics) keeps the dependency graph acyclic.
    """
    scenario_id: str
    total_corruptions: int = 0
    correctly_repaired: int = 0
    incorrectly_repaired: int = 0
    unrepaired: int = 0
    repair_recall: float = 0.0

    injected_duplicates_removed: int = 0
    original_rows_lost: int = 0
    all_original_rows_lost: int = 0
    correctly_removed_duplicates: int = 0

    originally_clean_values_modified: int = 0
    false_modification_rate: float = 0.0
    total_originally_clean_cells: int = 0

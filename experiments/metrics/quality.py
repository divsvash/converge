"""
Data-quality / correctness metrics.

Pure mapping from converge's own deterministic
converge.models.comparison.QualityComparison into
experiments.models.result.QualityMetrics -- field by field, by name.
No recomputation, no reinterpretation, no AI judgment anywhere in this
module. This is what "reuse the existing profiler/comparator, don't
reimplement CONVERGE's quality logic" means concretely: every value
below is read directly off ComparisonMetrics, nothing is derived here.

Deliberately NOT included: any overall "quality score". A lower
total_issues_after than total_issues_before is NOT treated by this
module (or anything downstream) as proof that a transformation was
"correct" -- these are observed deterministic before/after signals
only. Whether a transformation was actually correct (matched some
ground truth) is explicitly out of scope for this phase (Phase 10).
"""
from __future__ import annotations

from converge.models.comparison import QualityComparison
from experiments.models.result import QualityMetrics


def build_quality_metrics(comparison: QualityComparison) -> QualityMetrics:
    """
    Field-by-field pass-through from ComparisonMetrics to
    QualityMetrics. If a field exists on one but not the other, that
    is a signal this mapping needs updating -- not a reason to invent
    a value.
    """
    m = comparison.metrics
    return QualityMetrics(
        total_issues_before=m.total_issues_before,
        total_issues_after=m.total_issues_after,
        resolved_count=m.resolved_count,
        improved_count=m.improved_count,
        unchanged_count=m.unchanged_count,
        worsened_count=m.worsened_count,
        introduced_count=m.introduced_count,
        unverified_count=m.unverified_count,
        row_count_before=m.row_count_before,
        row_count_after=m.row_count_after,
        column_count_before=m.column_count_before,
        column_count_after=m.column_count_after,
        missing_value_total_before=m.missing_value_total_before,
        missing_value_total_after=m.missing_value_total_after,
        exact_duplicate_row_count_before=m.exact_duplicate_row_count_before,
        exact_duplicate_row_count_after=m.exact_duplicate_row_count_after,
        duplicate_candidate_row_count_before=m.duplicate_candidate_row_count_before,
        duplicate_candidate_row_count_after=m.duplicate_candidate_row_count_after,
        categorical_variant_total_before=m.categorical_variant_total_before,
        categorical_variant_total_after=m.categorical_variant_total_after,
        date_format_inconsistency_columns_before=m.date_format_inconsistency_columns_before,
        date_format_inconsistency_columns_after=m.date_format_inconsistency_columns_after,
        outlier_total_before=m.outlier_total_before,
        outlier_total_after=m.outlier_total_after,
    )

"""
Quality metrics tests. Pure unit tests against hand-built
QualityComparison objects -- no Spark needed, mirroring the same
factory style used in tests/services/comparison/test_comparator.py.
"""
from converge.models.comparison import ComparisonMetrics, IssueComparison, IssueOutcome, QualityComparison
from converge.models.profile import IssueType
from experiments.metrics.quality import build_quality_metrics


def _metrics(**overrides) -> ComparisonMetrics:
    base = dict(
        total_issues_before=5, total_issues_after=2,
        resolved_count=2, improved_count=1, unchanged_count=1, worsened_count=0,
        introduced_count=0, unverified_count=0,
        row_count_before=20, row_count_after=19,
        column_count_before=8, column_count_after=8,
        exact_duplicate_row_count_before=0, exact_duplicate_row_count_after=0,
        duplicate_candidate_row_count_before=1, duplicate_candidate_row_count_after=0,
        missing_value_total_before=2, missing_value_total_after=0,
        categorical_variant_total_before=4, categorical_variant_total_after=0,
        date_format_inconsistency_columns_before=1, date_format_inconsistency_columns_after=0,
        outlier_total_before=1, outlier_total_after=1,
    )
    base.update(overrides)
    return ComparisonMetrics(**base)


def _comparison(**metric_overrides) -> QualityComparison:
    return QualityComparison(
        run_id="r1",
        issue_comparisons=[
            IssueComparison(
                issue_type=IssueType.WHITESPACE_OR_CASING, columns=["full_name"],
                outcome=IssueOutcome.RESOLVED, before_value=4, after_value=0, detail="d",
            ),
        ],
        metrics=_metrics(**metric_overrides),
    )


# ---------------------------------------------------------------------------
# Same profiler/comparator path -- field-by-field pass-through
# ---------------------------------------------------------------------------

def test_every_comparison_metrics_field_is_mapped_by_name():
    comparison = _comparison()
    quality = build_quality_metrics(comparison)
    m = comparison.metrics

    assert quality.total_issues_before == m.total_issues_before
    assert quality.total_issues_after == m.total_issues_after
    assert quality.resolved_count == m.resolved_count
    assert quality.improved_count == m.improved_count
    assert quality.unchanged_count == m.unchanged_count
    assert quality.worsened_count == m.worsened_count
    assert quality.introduced_count == m.introduced_count
    assert quality.unverified_count == m.unverified_count
    assert quality.row_count_before == m.row_count_before
    assert quality.row_count_after == m.row_count_after
    assert quality.column_count_before == m.column_count_before
    assert quality.column_count_after == m.column_count_after
    assert quality.missing_value_total_before == m.missing_value_total_before
    assert quality.missing_value_total_after == m.missing_value_total_after
    assert quality.exact_duplicate_row_count_before == m.exact_duplicate_row_count_before
    assert quality.exact_duplicate_row_count_after == m.exact_duplicate_row_count_after
    assert quality.duplicate_candidate_row_count_before == m.duplicate_candidate_row_count_before
    assert quality.duplicate_candidate_row_count_after == m.duplicate_candidate_row_count_after
    assert quality.categorical_variant_total_before == m.categorical_variant_total_before
    assert quality.categorical_variant_total_after == m.categorical_variant_total_after
    assert quality.date_format_inconsistency_columns_before == m.date_format_inconsistency_columns_before
    assert quality.date_format_inconsistency_columns_after == m.date_format_inconsistency_columns_after
    assert quality.outlier_total_before == m.outlier_total_before
    assert quality.outlier_total_after == m.outlier_total_after


def test_row_and_column_destruction_remains_visible():
    comparison = _comparison(row_count_before=20, row_count_after=15, column_count_before=8, column_count_after=6)
    quality = build_quality_metrics(comparison)
    assert quality.row_count_before == 20
    assert quality.row_count_after == 15
    assert quality.column_count_before == 8
    assert quality.column_count_after == 6


# ---------------------------------------------------------------------------
# No overall quality score exists
# ---------------------------------------------------------------------------

def test_quality_metrics_has_no_score_field():
    comparison = _comparison()
    quality = build_quality_metrics(comparison)
    field_names = set(type(quality).model_fields.keys())
    for forbidden in ("score", "overall_score", "quality_score", "grade"):
        assert forbidden not in field_names


def test_build_quality_metrics_never_computes_a_derived_score():
    """
    Structural guard: the function source contains no division,
    weighting, or aggregation that would amount to a synthesized
    score -- it is a pure field-by-field pass-through.
    """
    import inspect
    from experiments.metrics import quality as quality_module
    source = inspect.getsource(quality_module.build_quality_metrics)
    assert "/" not in source  # no division anywhere in the mapping function
    assert "*" not in source  # no weighting


# ---------------------------------------------------------------------------
# Determinism / no reinterpretation
# ---------------------------------------------------------------------------

def test_build_quality_metrics_is_deterministic():
    comparison = _comparison()
    a = build_quality_metrics(comparison)
    b = build_quality_metrics(comparison)
    assert a == b


def test_zero_issues_both_sides_maps_cleanly():
    comparison = _comparison(
        total_issues_before=0, total_issues_after=0,
        resolved_count=0, improved_count=0, unchanged_count=0, worsened_count=0,
        introduced_count=0, unverified_count=0,
    )
    quality = build_quality_metrics(comparison)
    assert quality.total_issues_before == 0
    assert quality.resolved_count == 0

"""
QualityComparator tests.

Uses hand-built ProfileResult objects (not Spark) so every edge case
(column disappearing, type changing, zero issues, etc.) is exact and
deterministic to construct. The comparator itself is pure Python with
no Spark/AI dependency, so this is the right layer for precise unit
testing; test_orchestrator.py separately covers the real-Spark,
end-to-end integration path (profile -> execute -> re-profile ->
compare).
"""
from converge.models.comparison import IssueOutcome
from converge.models.profile import (
    CategoricalStats,
    CategoricalVariantGroup,
    ColumnProfile,
    DatasetProfile,
    DateFormatStats,
    DuplicateGroup,
    DuplicateSummary,
    EvidenceItem,
    IssueType,
    NumericStats,
    ProfileResult,
    SemanticType,
)
from converge.services.comparison.comparator import QualityComparator


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _col(
    name, semantic_type=SemanticType.STRING, spark_dtype="string",
    null_count=0, row_count=10, distinct_count=5,
    categorical_stats=None, date_format_stats=None, numeric_stats=None,
) -> ColumnProfile:
    return ColumnProfile(
        name=name, spark_dtype=spark_dtype, semantic_type=semantic_type,
        non_null_count=row_count - null_count, null_count=null_count,
        null_percentage=round(null_count / row_count * 100, 2) if row_count else 0.0,
        distinct_count=distinct_count, is_id_like=False,
        numeric_stats=numeric_stats, categorical_stats=categorical_stats,
        date_format_stats=date_format_stats, sample_values=[],
    )


def _cat_stats(variant_groups=None, whitespace=0) -> CategoricalStats:
    return CategoricalStats(
        distinct_count=5, top_values={}, variant_groups=variant_groups or [],
        whitespace_affected_count=whitespace,
    )


def _variant_group(key: str, variants: dict) -> CategoricalVariantGroup:
    return CategoricalVariantGroup(normalized_key=key, variants=variants)


def _date_stats(patterns: dict) -> DateFormatStats:
    return DateFormatStats(detected_patterns=patterns, unparseable_count=0)


def _numeric_stats(outlier_count=0) -> NumericStats:
    return NumericStats(min=0, max=100, mean=50, outlier_count=outlier_count)


def _profile(columns, row_count=10, dup_summary=None, evidence=None, dataset_id="ds") -> ProfileResult:
    return ProfileResult(
        profile=DatasetProfile(
            dataset_id=dataset_id, row_count=row_count, column_count=len(columns),
            columns=columns, duplicate_summary=dup_summary or DuplicateSummary(exact_duplicate_row_count=0),
        ),
        evidence=evidence or [],
    )


def _evidence(issue_type: IssueType, columns, supporting_data=None) -> EvidenceItem:
    return EvidenceItem(issue_type=issue_type, columns=columns, description="d", supporting_data=supporting_data or {})


def _comparator() -> QualityComparator:
    return QualityComparator()


# ---------------------------------------------------------------------------
# 1. No issues before / no issues after
# ---------------------------------------------------------------------------

def test_no_issues_before_or_after_produces_empty_comparison():
    before = _profile([_col("name")])
    after = _profile([_col("name")])
    result = _comparator().compare(before, after)
    assert result.issue_comparisons == []
    assert result.metrics.total_issues_before == 0
    assert result.metrics.total_issues_after == 0


# ---------------------------------------------------------------------------
# 2. Issue resolved
# ---------------------------------------------------------------------------

def test_missing_values_resolved():
    before = _profile([_col("salary", null_count=4, semantic_type=SemanticType.FLOAT, spark_dtype="double")])
    after = _profile([_col("salary", null_count=0, semantic_type=SemanticType.FLOAT, spark_dtype="double")])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.MISSING_VALUES)
    assert ic.outcome == IssueOutcome.RESOLVED
    assert ic.before_value == 4
    assert ic.after_value == 0
    assert result.metrics.resolved_count == 1


# ---------------------------------------------------------------------------
# 3. Issue severity improved
# ---------------------------------------------------------------------------

def test_categorical_inconsistency_improved():
    before_col = _col("city", categorical_stats=_cat_stats(
        variant_groups=[_variant_group("delhi", {"Delhi": 3, "delhi": 1, "DELHI": 1})]
    ))
    after_col = _col("city", categorical_stats=_cat_stats(
        variant_groups=[_variant_group("delhi", {"Delhi": 4, "delhi": 1})]
    ))
    before = _profile([before_col])
    after = _profile([after_col])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.CATEGORICAL_INCONSISTENCY)
    assert ic.outcome == IssueOutcome.IMPROVED
    assert ic.before_value == 3
    assert ic.after_value == 2


# ---------------------------------------------------------------------------
# 4. Issue unchanged
# ---------------------------------------------------------------------------

def test_whitespace_unchanged():
    before = _profile([_col("name", categorical_stats=_cat_stats(whitespace=2))])
    after = _profile([_col("name", categorical_stats=_cat_stats(whitespace=2))])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.WHITESPACE_OR_CASING)
    assert ic.outcome == IssueOutcome.UNCHANGED
    assert ic.before_value == ic.after_value == 2


# ---------------------------------------------------------------------------
# 5. Issue introduced after execution (+ WORSENED, the added 6th category)
# ---------------------------------------------------------------------------

def test_outlier_introduced():
    before = _profile([_col("salary", semantic_type=SemanticType.FLOAT, spark_dtype="double", numeric_stats=_numeric_stats(outlier_count=0))])
    after = _profile([_col("salary", semantic_type=SemanticType.FLOAT, spark_dtype="double", numeric_stats=_numeric_stats(outlier_count=2))])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.OUTLIER_CANDIDATE)
    assert ic.outcome == IssueOutcome.INTRODUCED
    assert ic.before_value == 0
    assert ic.after_value == 2
    assert result.metrics.introduced_count == 1


def test_worsened_when_severity_increases_on_existing_issue():
    before = _profile([_col("salary", null_count=1, semantic_type=SemanticType.FLOAT, spark_dtype="double")])
    after = _profile([_col("salary", null_count=3, semantic_type=SemanticType.FLOAT, spark_dtype="double")])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.MISSING_VALUES)
    assert ic.outcome == IssueOutcome.WORSENED
    assert result.metrics.worsened_count == 1


# ---------------------------------------------------------------------------
# 6. Insufficient evidence -> UNVERIFIED
# ---------------------------------------------------------------------------

def test_unverified_when_column_disappears():
    before = _profile([_col("legacy_field", null_count=3)])
    after = _profile([])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.MISSING_VALUES)
    assert ic.outcome == IssueOutcome.UNVERIFIED
    assert ic.after_value is None
    assert result.metrics.unverified_count == 1


def test_unverified_when_semantic_type_changes_away_from_required_type():
    """
    CONVERT_DTYPE turned a numeric column into a string column --
    OUTLIER_CANDIDATE can no longer be measured. Must be UNVERIFIED,
    NOT silently reported as RESOLVED (that would be dishonest: the
    outliers weren't fixed, they just became unmeasurable).
    """
    before = _profile([_col("code", semantic_type=SemanticType.FLOAT, spark_dtype="double", numeric_stats=_numeric_stats(outlier_count=3))])
    after = _profile([_col("code", semantic_type=SemanticType.STRING, spark_dtype="string")])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.OUTLIER_CANDIDATE)
    assert ic.outcome == IssueOutcome.UNVERIFIED
    assert ic.before_value == 3
    assert ic.after_value is None
    assert "type changed" in ic.detail


# ---------------------------------------------------------------------------
# 7. Multiple issue types
# ---------------------------------------------------------------------------

def test_multiple_issue_types_all_classified_independently():
    before_cols = [
        _col("city", categorical_stats=_cat_stats(variant_groups=[_variant_group("delhi", {"Delhi": 1, "delhi": 1})])),
        _col("salary", null_count=2, semantic_type=SemanticType.FLOAT, spark_dtype="double"),
        _col("join_date", semantic_type=SemanticType.DATE_LIKE_STRING, spark_dtype="string",
             date_format_stats=_date_stats({"YYYY-MM-DD": 5, "DD/MM/YYYY": 3})),
    ]
    after_cols = [
        _col("city", categorical_stats=_cat_stats(variant_groups=[])),
        _col("salary", null_count=0, semantic_type=SemanticType.FLOAT, spark_dtype="double"),
        _col("join_date", semantic_type=SemanticType.DATE_LIKE_STRING, spark_dtype="string",
             date_format_stats=_date_stats({"YYYY-MM-DD": 8})),
    ]
    result = _comparator().compare(_profile(before_cols), _profile(after_cols))
    outcomes_by_type = {ic.issue_type: ic.outcome for ic in result.issue_comparisons}
    assert outcomes_by_type[IssueType.CATEGORICAL_INCONSISTENCY] == IssueOutcome.RESOLVED
    assert outcomes_by_type[IssueType.MISSING_VALUES] == IssueOutcome.RESOLVED
    assert outcomes_by_type[IssueType.DATE_FORMAT_INCONSISTENCY] == IssueOutcome.RESOLVED


# ---------------------------------------------------------------------------
# 8. Multiple affected columns
# ---------------------------------------------------------------------------

def test_multiple_affected_columns_produce_separate_comparisons():
    before = _profile([
        _col("city", categorical_stats=_cat_stats(whitespace=2)),
        _col("dept", categorical_stats=_cat_stats(whitespace=1)),
    ])
    after = _profile([
        _col("city", categorical_stats=_cat_stats(whitespace=0)),
        _col("dept", categorical_stats=_cat_stats(whitespace=1)),
    ])
    result = _comparator().compare(before, after)
    ws = [i for i in result.issue_comparisons if i.issue_type == IssueType.WHITESPACE_OR_CASING]
    assert len(ws) == 2
    by_col = {ic.columns[0]: ic.outcome for ic in ws}
    assert by_col["city"] == IssueOutcome.RESOLVED
    assert by_col["dept"] == IssueOutcome.UNCHANGED


# ---------------------------------------------------------------------------
# 9. Before/after row-count changes
# ---------------------------------------------------------------------------

def test_row_count_change_reflected_in_metrics():
    before = _profile([_col("name")], row_count=20)
    after = _profile([_col("name")], row_count=19)
    result = _comparator().compare(before, after)
    assert result.metrics.row_count_before == 20
    assert result.metrics.row_count_after == 19


# ---------------------------------------------------------------------------
# 10. Before/after schema changes
# ---------------------------------------------------------------------------

def test_schema_change_reflected_in_metrics():
    before = _profile([_col("code", spark_dtype="string", semantic_type=SemanticType.STRING)])
    after = _profile([_col("code", spark_dtype="int", semantic_type=SemanticType.INTEGER)])
    result = _comparator().compare(before, after)
    assert result.metrics.schema_before == [{"name": "code", "dtype": "string"}]
    assert result.metrics.schema_after == [{"name": "code", "dtype": "int"}]


# ---------------------------------------------------------------------------
# 11. Duplicate reduction
# ---------------------------------------------------------------------------

def test_duplicate_candidates_reduced():
    before_dup = DuplicateSummary(
        exact_duplicate_row_count=0,
        duplicate_candidates=[DuplicateGroup(criteria_columns=["name", "city"], row_refs=["E1", "E2"], occurrence_count=2)],
    )
    after_dup = DuplicateSummary(exact_duplicate_row_count=0, duplicate_candidates=[])
    before = _profile([_col("name"), _col("city")], dup_summary=before_dup)
    after = _profile([_col("name"), _col("city")], dup_summary=after_dup)
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.DUPLICATE_ROWS)
    assert ic.outcome == IssueOutcome.RESOLVED
    assert ic.before_value == 1
    assert ic.after_value == 0
    assert result.metrics.duplicate_candidate_row_count_before == 1
    assert result.metrics.duplicate_candidate_row_count_after == 0


def test_duplicate_criteria_columns_disappearing_is_unverified():
    before_dup = DuplicateSummary(
        exact_duplicate_row_count=0,
        duplicate_candidates=[DuplicateGroup(criteria_columns=["name", "city"], row_refs=["E1", "E2"], occurrence_count=2)],
    )
    before = _profile([_col("name"), _col("city")], dup_summary=before_dup)
    after = _profile([_col("name")], dup_summary=DuplicateSummary(exact_duplicate_row_count=0))
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.DUPLICATE_ROWS)
    assert ic.outcome == IssueOutcome.UNVERIFIED


# ---------------------------------------------------------------------------
# 12. Missing-value reduction
# ---------------------------------------------------------------------------

def test_missing_value_reduction_metrics():
    before = _profile([_col("a", null_count=3), _col("b", null_count=2)])
    after = _profile([_col("a", null_count=1), _col("b", null_count=0)])
    result = _comparator().compare(before, after)
    assert result.metrics.missing_value_total_before == 5
    assert result.metrics.missing_value_total_after == 1


# ---------------------------------------------------------------------------
# 13. Categorical-variant reduction
# ---------------------------------------------------------------------------

def test_categorical_variant_total_metric():
    before = _profile([_col("city", categorical_stats=_cat_stats(
        variant_groups=[_variant_group("delhi", {"Delhi": 1, "delhi": 1, "DELHI": 1})]
    ))])
    after = _profile([_col("city", categorical_stats=_cat_stats(variant_groups=[]))])
    result = _comparator().compare(before, after)
    assert result.metrics.categorical_variant_total_before == 3
    assert result.metrics.categorical_variant_total_after == 0


# ---------------------------------------------------------------------------
# 14. Date-format consistency improvement
# ---------------------------------------------------------------------------

def test_date_format_consistency_improvement_metric():
    before = _profile([_col("join_date", semantic_type=SemanticType.DATE_LIKE_STRING, spark_dtype="string",
                             date_format_stats=_date_stats({"YYYY-MM-DD": 5, "DD/MM/YYYY": 3}))])
    after = _profile([_col("join_date", semantic_type=SemanticType.DATE_LIKE_STRING, spark_dtype="string",
                            date_format_stats=_date_stats({"YYYY-MM-DD": 8}))])
    result = _comparator().compare(before, after)
    assert result.metrics.date_format_inconsistency_columns_before == 1
    assert result.metrics.date_format_inconsistency_columns_after == 0
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.DATE_FORMAT_INCONSISTENCY)
    assert ic.outcome == IssueOutcome.RESOLVED


# ---------------------------------------------------------------------------
# 15. Outlier-count comparison where evidence permits
# ---------------------------------------------------------------------------

def test_outlier_count_comparison():
    before = _profile([_col("salary", semantic_type=SemanticType.FLOAT, spark_dtype="double", numeric_stats=_numeric_stats(outlier_count=3))])
    after = _profile([_col("salary", semantic_type=SemanticType.FLOAT, spark_dtype="double", numeric_stats=_numeric_stats(outlier_count=1))])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.OUTLIER_CANDIDATE)
    assert ic.outcome == IssueOutcome.IMPROVED
    assert result.metrics.outlier_total_before == 3
    assert result.metrics.outlier_total_after == 1


# ---------------------------------------------------------------------------
# 17. Deterministic: identical inputs -> identical output
# ---------------------------------------------------------------------------

def test_comparison_is_deterministic_for_identical_inputs():
    before = _profile([_col("city", categorical_stats=_cat_stats(whitespace=2)),
                        _col("salary", null_count=1, semantic_type=SemanticType.FLOAT, spark_dtype="double")])
    after = _profile([_col("city", categorical_stats=_cat_stats(whitespace=0)),
                       _col("salary", null_count=0, semantic_type=SemanticType.FLOAT, spark_dtype="double")])

    result_a = _comparator().compare(before, after)
    result_b = _comparator().compare(before, after)

    assert [ic.model_dump() for ic in result_a.issue_comparisons] == [ic.model_dump() for ic in result_b.issue_comparisons]
    assert result_a.metrics == result_b.metrics


# ---------------------------------------------------------------------------
# 18. Raw dataset values are not required by the comparison layer
# ---------------------------------------------------------------------------

def test_comparator_never_touches_raw_dataset_or_spark():
    import inspect
    from converge.services.comparison import comparator as module
    source = inspect.getsource(module)
    assert "pyspark" not in source
    assert "SparkSession" not in source
    assert "open(" not in source
    sig = inspect.signature(QualityComparator.compare)
    assert list(sig.parameters.keys()) == ["self", "before", "after"]


# ---------------------------------------------------------------------------
# Edge cases: empty evidence, columns appearing, zero-row datasets
# ---------------------------------------------------------------------------

def test_empty_evidence_both_sides():
    before = _profile([_col("name")], evidence=[])
    after = _profile([_col("name")], evidence=[])
    result = _comparator().compare(before, after)
    assert result.metrics.total_issues_before == 0
    assert result.metrics.total_issues_after == 0
    assert result.issue_comparisons == []


def test_new_column_with_issue_is_introduced_not_unverified():
    before = _profile([_col("name")])
    after = _profile([_col("name"), _col("new_field", null_count=3)])
    result = _comparator().compare(before, after)
    ic = next(i for i in result.issue_comparisons if i.issue_type == IssueType.MISSING_VALUES)
    assert ic.outcome == IssueOutcome.INTRODUCED
    assert ic.columns == ["new_field"]


def test_new_column_without_issue_produces_no_comparison():
    before = _profile([_col("name")])
    after = _profile([_col("name"), _col("clean_field", null_count=0)])
    result = _comparator().compare(before, after)
    assert result.issue_comparisons == []


def test_zero_row_datasets_both_sides():
    before = _profile([_col("name", null_count=0, row_count=0)], row_count=0)
    after = _profile([_col("name", null_count=0, row_count=0)], row_count=0)
    result = _comparator().compare(before, after)
    assert result.metrics.row_count_before == 0
    assert result.metrics.row_count_after == 0
    assert result.issue_comparisons == []


def test_incompatible_evidence_type_for_categorical_on_non_string_column_is_ignored():
    """
    A numeric column can never have categorical_stats populated by the
    real profiler, but the comparator's type-gating (via required_types)
    should still never misfire and report a categorical issue for it.
    """
    before = _profile([_col("amount", semantic_type=SemanticType.FLOAT, spark_dtype="double")])
    after = _profile([_col("amount", semantic_type=SemanticType.FLOAT, spark_dtype="double")])
    result = _comparator().compare(before, after)
    assert not any(ic.issue_type == IssueType.CATEGORICAL_INCONSISTENCY for ic in result.issue_comparisons)


def test_run_id_taken_from_before_profile_dataset_id():
    before = _profile([_col("name")], dataset_id="my_run_123")
    after = _profile([_col("name")], dataset_id="my_run_123__after_execution")
    result = _comparator().compare(before, after)
    assert result.run_id == "my_run_123"

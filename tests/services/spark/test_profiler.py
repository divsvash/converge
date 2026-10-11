import json

from pyspark.sql.types import StringType, StructField, StructType

from converge.models.profile import IssueType, SemanticType
from converge.services.spark.profiler import profile_dataset


# ---------------------------------------------------------------------------
# Shape / basic aggregate facts
# ---------------------------------------------------------------------------

def test_profile_shape(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    assert result.profile.row_count == 20
    assert result.profile.column_count == 8
    assert len(result.profile.columns) == 8


def test_column_names_match_source(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    profiled_names = {c.name for c in result.profile.columns}
    assert profiled_names == set(sample_hr_df.columns)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def test_missing_values_detected(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    cols = {c.name: c for c in result.profile.columns}

    assert cols["salary"].null_count == 2
    assert cols["age"].null_count == 1
    assert cols["employee_id"].null_count == 0

    missing_evidence = [e for e in result.evidence if e.issue_type == IssueType.MISSING_VALUES]
    missing_columns = {e.columns[0] for e in missing_evidence}
    assert "salary" in missing_columns
    assert "age" in missing_columns
    # a fully-populated column must not generate missing-value evidence
    assert "employee_id" not in missing_columns


# ---------------------------------------------------------------------------
# Categorical inconsistency (case / variant grouping)
# ---------------------------------------------------------------------------

def test_categorical_variant_groups_detected(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    city = next(c for c in result.profile.columns if c.name == "city")

    assert city.categorical_stats is not None
    normalized_keys = {g.normalized_key for g in city.categorical_stats.variant_groups}

    # every genuinely-inconsistent city should be caught
    for expected in ("delhi", "bangalore", "mumbai", "chennai", "pune", "hyderabad"):
        assert expected in normalized_keys, f"expected variant group for '{expected}'"

    delhi_group = next(g for g in city.categorical_stats.variant_groups if g.normalized_key == "delhi")
    # raw variants: "Delhi", "delhi", "DELHI", "Delhi " (trailing space)
    assert set(delhi_group.variants.keys()) == {"Delhi", "delhi", "DELHI", "Delhi "}
    assert sum(delhi_group.variants.values()) == 6  # 3x Delhi, 1x delhi, 1x DELHI, 1x "Delhi "


def test_single_variant_value_not_falsely_grouped(sample_hr_df):
    """
    'New Delhi' should NOT be merged into the 'delhi' group or flagged
    as its own inconsistency group -- it is a genuinely different value,
    not a case/whitespace variant, and appears with only one raw form.
    """
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    city = next(c for c in result.profile.columns if c.name == "city")
    normalized_keys = {g.normalized_key for g in city.categorical_stats.variant_groups}
    assert "new delhi" not in normalized_keys


def test_whitespace_detected(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    city = next(c for c in result.profile.columns if c.name == "city")
    # "Delhi " (E012) has a trailing space
    assert city.categorical_stats.whitespace_affected_count >= 1

    whitespace_evidence = [e for e in result.evidence if e.issue_type == IssueType.WHITESPACE_OR_CASING]
    assert any(e.columns == ["city"] for e in whitespace_evidence)


def test_high_cardinality_free_text_not_grouped(sample_hr_df):
    """
    full_name is high-cardinality (mostly-unique free text) and should
    not produce noisy variant groups the way a true categorical column
    like city does.
    """
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    full_name = next(c for c in result.profile.columns if c.name == "full_name")
    # "Aarav Sharma" appears twice identically (E001, E008) -- not a
    # case/whitespace variant, so it should not appear as a variant group
    # even though full_name itself is skipped for shape reasons.
    if full_name.categorical_stats:
        assert full_name.categorical_stats.variant_groups == []


# ---------------------------------------------------------------------------
# Date format inconsistency
# ---------------------------------------------------------------------------

def test_date_format_inconsistency_detected(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    join_date = next(c for c in result.profile.columns if c.name == "join_date")

    assert join_date.semantic_type == SemanticType.DATE_LIKE_STRING
    assert join_date.date_format_stats is not None
    assert len(join_date.date_format_stats.detected_patterns) == 2  # ISO and slash formats both present

    date_evidence = [e for e in result.evidence if e.issue_type == IssueType.DATE_FORMAT_INCONSISTENCY]
    assert any(e.columns == ["join_date"] for e in date_evidence)


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def test_salary_outlier_detected(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    salary = next(c for c in result.profile.columns if c.name == "salary")

    assert salary.numeric_stats is not None
    assert salary.numeric_stats.outlier_count >= 1  # the 600000 value

    outlier_evidence = [e for e in result.evidence if e.issue_type == IssueType.OUTLIER_CANDIDATE]
    assert any(e.columns == ["salary"] for e in outlier_evidence)


def test_age_outlier_detected(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    age = next(c for c in result.profile.columns if c.name == "age")
    assert age.numeric_stats is not None
    assert age.numeric_stats.outlier_count >= 1  # the -5 value


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def test_id_like_column_detection(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    assert "employee_id" in result.profile.duplicate_summary.id_like_columns
    # manager_id repeats heavily (M010, M011, ...) and must NOT be
    # classified as id-like despite matching the name pattern
    assert "manager_id" not in result.profile.duplicate_summary.id_like_columns


def test_duplicate_candidates_detected(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    groups = result.profile.duplicate_summary.duplicate_candidates
    assert len(groups) >= 1

    # E001 and E008 are identical on every column except employee_id
    match = next((g for g in groups if g.occurrence_count == 2), None)
    assert match is not None
    assert set(match.row_refs) == {"E001", "E008"}
    assert "employee_id" not in match.criteria_columns

    dup_evidence = [e for e in result.evidence if e.issue_type == IssueType.DUPLICATE_ROWS]
    assert len(dup_evidence) >= 1


def test_exact_duplicate_row_count_is_zero_when_ids_differ(sample_hr_df):
    """
    Because employee_id differs between E001 and E008, they are NOT
    exact full-row duplicates -- only duplicate *candidates* once the
    id column is excluded. This distinction matters for the validator's
    evidence rule.
    """
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    assert result.profile.duplicate_summary.exact_duplicate_row_count == 0


# ---------------------------------------------------------------------------
# Evidence contract / serialization
# ---------------------------------------------------------------------------

def test_evidence_is_bounded_and_typed(sample_hr_df):
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    assert len(result.evidence) > 0
    issue_types_found = {e.issue_type for e in result.evidence}
    assert IssueType.MISSING_VALUES in issue_types_found
    assert IssueType.CATEGORICAL_INCONSISTENCY in issue_types_found
    assert IssueType.DATE_FORMAT_INCONSISTENCY in issue_types_found
    assert IssueType.OUTLIER_CANDIDATE in issue_types_found
    assert IssueType.DUPLICATE_ROWS in issue_types_found

    for item in result.evidence:
        assert isinstance(item.supporting_data, dict)
        assert item.columns  # never empty


def test_profile_result_is_json_serializable(sample_hr_df):
    """
    This is the contract downstream stages (validator, AI provider,
    file-based run storage) depend on: the full ProfileResult must
    serialize cleanly, since it will be persisted to
    data/runs/{run_id}/profile.json in a later step.
    """
    result = profile_dataset(sample_hr_df, dataset_id="sample_hr")
    raw = result.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["profile"]["dataset_id"] == "sample_hr"
    assert parsed["profile"]["row_count"] == 20
    assert isinstance(parsed["evidence"], list)


# ---------------------------------------------------------------------------
# Edge cases with small in-memory DataFrames (independent of the CSV fixture)
# ---------------------------------------------------------------------------

def test_empty_dataframe_does_not_crash(spark):
    schema = StructType([StructField("name", StringType(), True)])
    empty_df = spark.createDataFrame([], schema)
    result = profile_dataset(empty_df, dataset_id="empty")
    assert result.profile.row_count == 0
    assert result.profile.columns[0].null_count == 0
    assert result.evidence == []


def test_all_null_column_reports_full_null_percentage(spark):
    schema = StructType([StructField("notes", StringType(), True)])
    df = spark.createDataFrame([(None,), (None,), (None,)], schema)
    result = profile_dataset(df, dataset_id="all_null")
    notes = result.profile.columns[0]
    assert notes.null_count == 3
    assert notes.null_percentage == 100.0

    missing_evidence = [e for e in result.evidence if e.issue_type == IssueType.MISSING_VALUES]
    assert len(missing_evidence) == 1


def test_single_row_dataframe(spark):
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("city", StringType(), True),
    ])
    df = spark.createDataFrame([("A001", "Delhi")], schema)
    result = profile_dataset(df, dataset_id="single_row")
    assert result.profile.row_count == 1
    assert result.profile.duplicate_summary.exact_duplicate_row_count == 0

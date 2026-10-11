from converge.models.profile import IssueType
from converge.models.recommendation import EvidenceReference, Recommendation
from converge.models.validation import ValidationStatus
from converge.services.validation.validator import Validator


def _validator(sample_profile_result):
    return Validator(sample_profile_result)


# ---------------------------------------------------------------------------
# 1. Valid recommendation with matching evidence -> approved
# ---------------------------------------------------------------------------

def test_standardize_category_with_valid_evidence_is_approved_auto(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="STANDARDIZE_CATEGORY",
        target_columns=["city"],
        parameters={"mapping": {"Delhi": "Delhi", "delhi": "Delhi", "DELHI": "Delhi", "Delhi ": "Delhi"}},
        reason="These four raw values all refer to the same city.",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.APPROVED_AUTO
    assert all(c.passed is not False for c in result.checks)


def test_trim_whitespace_with_valid_evidence_is_approved_auto(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="TRIM_WHITESPACE",
        target_columns=["city"],
        parameters={},
        reason="Some city values have leading/trailing whitespace.",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.APPROVED_AUTO


# ---------------------------------------------------------------------------
# 2. Nonexistent operation -> rejected
# ---------------------------------------------------------------------------

def test_nonexistent_operation_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="TOTALLY_MADE_UP_OPERATION",
        target_columns=["city"],
        reason="nonsense",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    op_check = next(c for c in result.checks if c.rule == "operation_supported")
    assert op_check.passed is False


# ---------------------------------------------------------------------------
# 3. Nonexistent column -> rejected
# ---------------------------------------------------------------------------

def test_nonexistent_column_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="TRIM_WHITESPACE",
        target_columns=["column_that_does_not_exist"],
        reason="made up",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    col_check = next(c for c in result.checks if c.rule == "target_columns_exist")
    assert col_check.passed is False


def test_empty_target_columns_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(operation="TRIM_WHITESPACE", target_columns=[], reason="no target")
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# 4. Incompatible datatype -> rejected
# ---------------------------------------------------------------------------

def test_date_format_operation_on_numeric_column_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="STANDARDIZE_DATE_FORMAT",
        target_columns=["salary"],  # numeric, not date-like
        parameters={"target_format": "YYYY-MM-DD"},
        reason="wrong column type",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    dtype_check = next(c for c in result.checks if c.rule == "dtype_compatible")
    assert dtype_check.passed is False


def test_normalize_case_on_numeric_column_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="NORMALIZE_CASE",
        target_columns=["age"],
        parameters={"target_case": "lower"},
        reason="wrong column type",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# 5. Malformed parameters -> rejected
# ---------------------------------------------------------------------------

def test_missing_required_parameter_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="STANDARDIZE_CATEGORY",
        target_columns=["city"],
        parameters={},  # missing required 'mapping'
        reason="no mapping given",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    param_check = next(c for c in result.checks if c.rule == "parameters_valid")
    assert param_check.passed is False


def test_wrong_parameter_type_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="STANDARDIZE_CATEGORY",
        target_columns=["city"],
        parameters={"mapping": "this should be a dict, not a string"},
        reason="wrong type",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


def test_disallowed_parameter_value_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="NORMALIZE_CASE",
        target_columns=["city"],
        parameters={"target_case": "sarcastic"},  # not in allowed_values
        reason="invalid case option",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


def test_fill_missing_constant_strategy_without_fill_value_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="FILL_MISSING",
        target_columns=["salary"],
        parameters={"strategy": "constant"},  # missing fill_value
        reason="constant fill with no value",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


def test_unexpected_parameter_smuggling_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="TRIM_WHITESPACE",
        target_columns=["city"],
        parameters={"unexpected_field": "should not be allowed"},
        reason="TRIM_WHITESPACE declares zero parameters",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    param_check = next(c for c in result.checks if c.rule == "parameters_valid")
    assert param_check.passed is False


# ---------------------------------------------------------------------------
# 6. Missing evidence -> rejected or needs-review depending on operation
# ---------------------------------------------------------------------------

def test_standardize_category_with_no_supporting_evidence_is_rejected(sample_profile_result):
    """
    'department' has no case/whitespace inconsistency in the sample
    data at all -- there is no CATEGORICAL_INCONSISTENCY evidence for
    it. A SAFE-tier operation with no evidence must be rejected, not
    silently approved.
    """
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="STANDARDIZE_CATEGORY",
        target_columns=["department"],
        parameters={"mapping": {"Engineering": "Engineering"}},
        reason="fabricated: department has no real inconsistency",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    evidence_check = next(c for c in result.checks if c.rule == "evidence_exists")
    assert evidence_check.passed is False


def test_fill_missing_on_column_with_no_nulls_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="FILL_MISSING",
        target_columns=["employee_id"],  # zero nulls
        parameters={"strategy": "mode"},
        reason="fabricated: employee_id has no missing values",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


def test_convert_dtype_needs_review_despite_no_evidence_requirement(sample_profile_result):
    """
    CONVERT_DTYPE has no required evidence type in P0 (the profiler
    doesn't yet emit TYPE_INCONSISTENCY evidence), so the evidence
    check passes trivially -- but the REVIEW_REQUIRED tier ceiling
    still applies, so it can only ever reach NEEDS_REVIEW, never
    APPROVED_AUTO. This is the 'needs review depending on operation'
    half of the missing-evidence behavior.
    """
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="CONVERT_DTYPE",
        target_columns=["salary"],
        parameters={"target_type": "integer"},
        reason="salary could be stored as integer instead of double",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# 7. Evidence claiming a fact that contradicts the profile -> rejected
# ---------------------------------------------------------------------------

def test_contradicted_claimed_null_count_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="FILL_MISSING",
        target_columns=["salary"],
        parameters={"strategy": "mean"},
        reason="salary has many missing values",
        evidence_refs=[
            EvidenceReference(
                issue_type=IssueType.MISSING_VALUES,
                columns=["salary"],
                claim="salary has 999 missing values",
                claimed_values={"null_count": 999},  # actual is 2
            )
        ],
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    contradiction_check = next(c for c in result.checks if c.rule == "evidence_supports_claim")
    assert contradiction_check.passed is False
    assert "contradicts" in contradiction_check.message


def test_standardize_category_mapping_with_fabricated_key_is_rejected(sample_profile_result):
    """
    The AI claims 'Delh1' (a typo/fabrication) is one of the raw
    variants for city -- it never appears in the actual profile
    evidence, so the recommendation must be rejected even though the
    operation, columns, and parameter types are all otherwise valid.
    """
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="STANDARDIZE_CATEGORY",
        target_columns=["city"],
        parameters={"mapping": {"Delhi": "Delhi", "Delh1": "Delhi"}},
        reason="fabricated variant",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    contradiction_check = next(c for c in result.checks if c.rule == "evidence_supports_claim")
    assert contradiction_check.passed is False


# ---------------------------------------------------------------------------
# 8. Forbidden destructive operation -> rejected
# ---------------------------------------------------------------------------

def test_delete_column_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="DELETE_COLUMN",
        target_columns=["salary"],
        reason="the model decided this column is not useful",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    forbidden_check = next(c for c in result.checks if c.rule == "forbidden_operation_rejected")
    assert forbidden_check.passed is False


def test_delete_dataset_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(operation="DELETE_DATASET", target_columns=["salary"], reason="wipe everything")
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


def test_schema_restructuring_join_pivot_merge_all_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    for op in ("SCHEMA_RESTRUCTURING", "JOIN", "PIVOT", "MERGE", "ARBITRARY_ROW_FILTER"):
        rec = Recommendation(operation=op, target_columns=["salary"], reason="structural change")
        result = v.validate(rec)
        assert result.status == ValidationStatus.REJECTED, f"{op} should be rejected"


# ---------------------------------------------------------------------------
# 9. Arbitrary code in a recommendation -> rejected
# ---------------------------------------------------------------------------

def test_arbitrary_code_operation_name_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="ARBITRARY_CODE",
        target_columns=["salary"],
        parameters={"code": "import os; os.system('rm -rf /')"},
        reason="the model tried to submit code directly",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


def test_code_smuggled_as_extra_parameter_on_legitimate_operation_is_rejected(sample_profile_result):
    """
    Even when the operation name itself is legitimate and registered,
    an extra, undeclared parameter (e.g. an attempt to smuggle a code
    string alongside a real operation) must be rejected -- the
    validator does not silently ignore unknown fields.
    """
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="TRIM_WHITESPACE",
        target_columns=["city"],
        parameters={"exec": "import subprocess; subprocess.run(['rm', '-rf', '.'])"},
        reason="looks legitimate but carries an extra payload",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED
    param_check = next(c for c in result.checks if c.rule == "parameters_valid")
    assert param_check.passed is False


def test_code_disguised_as_mapping_value_type_mismatch_is_rejected(sample_profile_result):
    """
    A string value where a dict mapping is required is a type
    mismatch and is rejected structurally -- the validator never
    evaluates or interprets the string, it simply doesn't match the
    declared parameter type.
    """
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="STANDARDIZE_CATEGORY",
        target_columns=["city"],
        parameters={"mapping": "__import__('os').system('id')"},
        reason="type-mismatched payload",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# 10. FLAG_OUTLIER cannot become an executable recommendation
# ---------------------------------------------------------------------------

def test_flag_outlier_with_valid_evidence_is_not_executable(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="FLAG_OUTLIER",
        target_columns=["salary"],
        parameters={},
        reason="salary has a 600000 outlier",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.NOT_EXECUTABLE
    assert result.status not in (ValidationStatus.APPROVED_AUTO, ValidationStatus.NEEDS_REVIEW)


def test_flag_outlier_status_is_never_approved_auto_or_needs_review_even_with_perfect_evidence(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(operation="FLAG_OUTLIER", target_columns=["age"], reason="age has an outlier")
    result = v.validate(rec)
    assert result.status == ValidationStatus.NOT_EXECUTABLE


# ---------------------------------------------------------------------------
# 11. REMOVE_DUPLICATES must reference duplicate evidence
# ---------------------------------------------------------------------------

def test_remove_duplicates_with_matching_criteria_is_needs_review(sample_profile_result):
    dup_evidence = [e for e in sample_profile_result.evidence if e.issue_type == IssueType.DUPLICATE_ROWS]
    assert dup_evidence, "fixture must contain duplicate evidence for this test to be meaningful"
    criteria_columns = dup_evidence[0].columns

    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="REMOVE_DUPLICATES",
        target_columns=criteria_columns,
        parameters={},
        reason="rows identical across all non-id columns are duplicates",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.NEEDS_REVIEW


def test_remove_duplicates_with_fabricated_criteria_is_rejected(sample_profile_result):
    """
    Claiming duplicates exist based on a criteria set that doesn't
    match any actual duplicate evidence group must be rejected --
    row removal cannot proceed on an unverified claim.
    """
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="REMOVE_DUPLICATES",
        target_columns=["city"],  # not the real criteria columns
        parameters={},
        reason="fabricated duplicate criteria",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# 12. FILL_MISSING must reference missing-value evidence
# ---------------------------------------------------------------------------

def test_fill_missing_with_matching_evidence_is_needs_review(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="FILL_MISSING",
        target_columns=["salary"],
        parameters={"strategy": "median"},
        reason="salary has missing values",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# 13. NORMALIZE_CASE / STANDARDIZE_CATEGORY must reference categorical evidence
# ---------------------------------------------------------------------------

def test_normalize_case_with_categorical_evidence_is_approved_auto(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="NORMALIZE_CASE",
        target_columns=["city"],
        parameters={"target_case": "title"},
        reason="city has mixed casing",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.APPROVED_AUTO


def test_normalize_case_on_column_with_no_case_or_whitespace_evidence_is_rejected(sample_profile_result):
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="NORMALIZE_CASE",
        target_columns=["department"],  # already consistent in the sample data
        parameters={"target_case": "title"},
        reason="fabricated",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.REJECTED


# ---------------------------------------------------------------------------
# Demonstrating the core distinction: validator result != user approval
# ---------------------------------------------------------------------------

def test_validation_result_has_no_notion_of_user_decision(sample_profile_result):
    """
    The ValidationResult / ValidationStatus objects must not carry any
    'decision', 'approved_by_user', or execution-triggering field --
    that concept belongs entirely to a later layer. This test locks
    that boundary in place structurally.
    """
    v = _validator(sample_profile_result)
    rec = Recommendation(
        operation="TRIM_WHITESPACE",
        target_columns=["city"],
        reason="whitespace present",
    )
    result = v.validate(rec)
    assert result.status == ValidationStatus.APPROVED_AUTO
    field_names = set(type(result).model_fields.keys())
    assert "decision" not in field_names
    assert "approved_by_user" not in field_names
    assert "executed" not in field_names

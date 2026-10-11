"""
InsightGenerator + recommendation-outcome traceability tests.
"""
from converge.models.comparison import (
    ComparisonMetrics,
    IssueComparison,
    IssueOutcome,
    QualityComparison,
    RecommendationOutcome,
)
from converge.models.execution import ExecutionSummary, OperationExecutionResult, OperationExecutionStatus
from converge.models.insight import InsightKind
from converge.models.pipeline import RecommendationDecision, RecommendationRecord
from converge.models.profile import IssueType
from converge.models.recommendation import Recommendation
from converge.models.validation import ValidationResult, ValidationStatus
from converge.services.comparison.traceability import build_recommendation_outcomes
from converge.services.insights.generator import InsightGenerator


def _metrics(**overrides) -> ComparisonMetrics:
    base = dict(
        total_issues_before=0, total_issues_after=0,
        row_count_before=10, row_count_after=10, column_count_before=2, column_count_after=2,
        exact_duplicate_row_count_before=0, exact_duplicate_row_count_after=0,
        duplicate_candidate_row_count_before=0, duplicate_candidate_row_count_after=0,
        missing_value_total_before=0, missing_value_total_after=0,
        categorical_variant_total_before=0, categorical_variant_total_after=0,
        date_format_inconsistency_columns_before=0, date_format_inconsistency_columns_after=0,
        outlier_total_before=0, outlier_total_after=0,
    )
    base.update(overrides)
    return ComparisonMetrics(**base)


def _comparison(issue_comparisons=None, **metric_overrides) -> QualityComparison:
    return QualityComparison(
        run_id="r1", issue_comparisons=issue_comparisons or [], metrics=_metrics(**metric_overrides)
    )


def _ic(issue_type, columns, outcome, before=1, after=0) -> IssueComparison:
    return IssueComparison(
        issue_type=issue_type, columns=columns, outcome=outcome,
        before_value=before, after_value=after, detail="d",
    )


# ---------------------------------------------------------------------------
# Determinism / no LLM
# ---------------------------------------------------------------------------

def test_insight_generator_never_touches_raw_dataset_or_ai():
    import inspect
    from converge.services.insights import generator as module
    source = inspect.getsource(module)
    assert "pyspark" not in source
    assert "genai" not in source
    assert "Gemini" not in source
    assert "open(" not in source


def test_insight_generation_is_deterministic():
    comparison = _comparison(
        issue_comparisons=[_ic(IssueType.WHITESPACE_OR_CASING, ["city"], IssueOutcome.RESOLVED, 4, 0)],
        resolved_count=1, total_issues_before=1, total_issues_after=0,
    )
    gen = InsightGenerator()
    artifact_a = gen.generate(comparison, [])
    artifact_b = gen.generate(comparison, [])
    assert [i.model_dump(exclude={"kind"}) for i in artifact_a.insights] == \
           [i.model_dump(exclude={"kind"}) for i in artifact_b.insights]


def test_insight_source_is_deterministic_not_ai():
    artifact = InsightGenerator().generate(_comparison(), [])
    assert artifact.source == "deterministic"


# ---------------------------------------------------------------------------
# Fact / interpretation / uncertainty tagging
# ---------------------------------------------------------------------------

def test_resolved_issue_produces_a_fact_insight():
    comparison = _comparison(
        issue_comparisons=[_ic(IssueType.WHITESPACE_OR_CASING, ["city"], IssueOutcome.RESOLVED, 4, 0)],
        resolved_count=1, total_issues_before=1, total_issues_after=0,
    )
    artifact = InsightGenerator().generate(comparison, [])
    fact_insights = [i for i in artifact.insights if i.kind == InsightKind.FACT]
    assert any("RESOLVED" in i.text and "city" in i.text for i in fact_insights)


def test_unverified_comparison_produces_an_uncertainty_insight():
    comparison = _comparison(
        issue_comparisons=[_ic(IssueType.OUTLIER_CANDIDATE, ["code"], IssueOutcome.UNVERIFIED, 3, None)],
        unverified_count=1,
    )
    artifact = InsightGenerator().generate(comparison, [])
    uncertainty = [i for i in artifact.insights if i.kind == InsightKind.UNCERTAINTY]
    assert len(uncertainty) == 1
    assert "code" in uncertainty[0].text


def test_net_improvement_produces_interpretation_insight_with_hedged_language():
    comparison = _comparison(
        issue_comparisons=[
            _ic(IssueType.WHITESPACE_OR_CASING, ["city"], IssueOutcome.RESOLVED, 4, 0),
            _ic(IssueType.MISSING_VALUES, ["salary"], IssueOutcome.IMPROVED, 3, 1),
        ],
        resolved_count=1, improved_count=1,
    )
    artifact = InsightGenerator().generate(comparison, [])
    interpretations = [i for i in artifact.insights if i.kind == InsightKind.INTERPRETATION]
    assert len(interpretations) == 1
    assert "associated with" in interpretations[0].text  # hedged, not causal
    assert "caused" not in interpretations[0].text.lower()


def test_no_interpretation_insight_when_no_comparable_issues():
    artifact = InsightGenerator().generate(_comparison(), [])
    assert not any(i.kind == InsightKind.INTERPRETATION for i in artifact.insights)


def test_net_regression_produces_warning_interpretation():
    comparison = _comparison(
        issue_comparisons=[_ic(IssueType.OUTLIER_CANDIDATE, ["salary"], IssueOutcome.INTRODUCED, 0, 2)],
        introduced_count=1,
    )
    artifact = InsightGenerator().generate(comparison, [])
    interpretations = [i for i in artifact.insights if i.kind == InsightKind.INTERPRETATION]
    assert len(interpretations) == 1
    assert "regression" in interpretations[0].text


# ---------------------------------------------------------------------------
# Insight generation cannot modify the deterministic comparison
# ---------------------------------------------------------------------------

def test_insight_generation_does_not_mutate_the_comparison_object():
    comparison = _comparison(
        issue_comparisons=[_ic(IssueType.WHITESPACE_OR_CASING, ["city"], IssueOutcome.RESOLVED, 4, 0)],
        resolved_count=1,
    )
    original = comparison.model_dump()
    InsightGenerator().generate(comparison, [])
    assert comparison.model_dump() == original


def test_insight_generator_has_no_write_path_to_comparison_fields():
    """
    Structural guard: InsightGenerator.generate()'s signature takes the
    comparison as a plain argument -- it does not return a modified
    QualityComparison, and cannot, by construction, override the
    comparator's classification.
    """
    import inspect
    sig = inspect.signature(InsightGenerator.generate)
    assert "InsightArtifact" in str(sig.return_annotation)


# ---------------------------------------------------------------------------
# Execution facts
# ---------------------------------------------------------------------------

def test_execution_fact_reports_approved_vs_succeeded():
    outcomes = [
        RecommendationOutcome(
            recommendation_id="r1", operation="TRIM_WHITESPACE", target_columns=["city"],
            decision="approved_by_user", execution_status="succeeded", outcome_summary="ok",
        ),
        RecommendationOutcome(
            recommendation_id="r2", operation="FILL_MISSING", target_columns=["salary"],
            decision="approved_by_user", execution_status="failed", outcome_summary="fail",
        ),
        RecommendationOutcome(
            recommendation_id="r3", operation="FLAG_OUTLIER", target_columns=["age"],
            decision="rejected_by_user", outcome_summary="not approved",
        ),
    ]
    artifact = InsightGenerator().generate(_comparison(), outcomes)
    assert any("1 of 2 approved" in i.text for i in artifact.insights)


def test_no_approved_recommendations_produces_a_fact():
    outcomes = [RecommendationOutcome(
        recommendation_id="r1", operation="TRIM_WHITESPACE", target_columns=["city"],
        decision="rejected_by_user", outcome_summary="not approved",
    )]
    artifact = InsightGenerator().generate(_comparison(), outcomes)
    assert any("No recommendations were approved" in i.text for i in artifact.insights)


# ===========================================================================
# Recommendation -> outcome traceability
# ===========================================================================

def _rec(operation, columns, params=None) -> Recommendation:
    return Recommendation(operation=operation, target_columns=columns, parameters=params or {}, reason="r")


def _validated(status: ValidationStatus) -> ValidationResult:
    return ValidationResult(recommendation_id="x", operation="op", status=status, checks=[], summary="s")


def test_traceability_links_successful_trim_whitespace_to_resolved_comparison():
    rec = _rec("TRIM_WHITESPACE", ["city"])
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.APPROVED_AUTO),
        decision=RecommendationDecision.APPROVED_BY_USER,
    )
    exec_summary = ExecutionSummary(
        operations_attempted=1, operations_succeeded=1,
        operation_results=[OperationExecutionResult(
            recommendation_id=rec.recommendation_id, operation="TRIM_WHITESPACE",
            target_columns=("city",), status=OperationExecutionStatus.SUCCEEDED, rows_affected=4,
        )],
    )
    comparison = _comparison(issue_comparisons=[
        _ic(IssueType.WHITESPACE_OR_CASING, ["city"], IssueOutcome.RESOLVED, 4, 0)
    ])

    outcomes = build_recommendation_outcomes([record], exec_summary, comparison)
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.execution_status == "succeeded"
    assert outcome.rows_affected == 4
    assert len(outcome.linked_issue_comparisons) == 1
    assert outcome.linked_issue_comparisons[0].outcome == IssueOutcome.RESOLVED
    assert "resolved" in outcome.outcome_summary


def test_traceability_example_from_spec_trim_whitespace_city():
    """Mirrors the exact example given in the Step 6 spec."""
    rec = _rec("TRIM_WHITESPACE", ["city"])
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.APPROVED_AUTO),
        decision=RecommendationDecision.APPROVED_BY_USER,
    )
    exec_summary = ExecutionSummary(
        operation_results=[OperationExecutionResult(
            recommendation_id=rec.recommendation_id, operation="TRIM_WHITESPACE",
            target_columns=("city",), status=OperationExecutionStatus.SUCCEEDED, rows_affected=4,
        )],
    )
    comparison = _comparison(issue_comparisons=[
        _ic(IssueType.WHITESPACE_OR_CASING, ["city"], IssueOutcome.RESOLVED, before=4, after=0)
    ])
    outcomes = build_recommendation_outcomes([record], exec_summary, comparison)
    ic = outcomes[0].linked_issue_comparisons[0]
    assert ic.before_value == 4
    assert ic.after_value == 0
    assert ic.outcome == IssueOutcome.RESOLVED


def test_traceability_rejected_recommendation_has_no_execution_status():
    rec = _rec("STANDARDIZE_CATEGORY", ["department"])
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.REJECTED),
        decision=RecommendationDecision.REJECTED_BY_USER,
    )
    outcomes = build_recommendation_outcomes([record], None, _comparison())
    assert outcomes[0].execution_status is None
    assert "Not executed" in outcomes[0].outcome_summary


def test_traceability_pending_decision_never_approved():
    rec = _rec("FILL_MISSING", ["salary"], {"strategy": "mean"})
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.NEEDS_REVIEW),
        # decision left as default PENDING
    )
    outcomes = build_recommendation_outcomes([record], None, _comparison())
    assert outcomes[0].decision == "pending"
    assert outcomes[0].execution_status is None


def test_traceability_does_not_invent_a_link_when_no_comparable_evidence_exists():
    rec = _rec("CONVERT_DTYPE", ["score"], {"target_type": "integer"})
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.NEEDS_REVIEW),
        decision=RecommendationDecision.APPROVED_BY_USER,
    )
    exec_summary = ExecutionSummary(
        operation_results=[OperationExecutionResult(
            recommendation_id=rec.recommendation_id, operation="CONVERT_DTYPE",
            target_columns=("score",), status=OperationExecutionStatus.SUCCEEDED, rows_affected=10,
        )],
    )
    # CONVERT_DTYPE has no required_evidence_issue_types in the registry,
    # so it can never be linked to a comparison entry -- this is honest,
    # not a bug.
    outcomes = build_recommendation_outcomes([record], exec_summary, _comparison())
    assert outcomes[0].linked_issue_comparisons == []
    assert "no directly comparable" in outcomes[0].outcome_summary


def test_traceability_failed_execution_reports_error():
    rec = _rec("NORMALIZE_CASE", ["city"], {"target_case": "lower"})
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.APPROVED_AUTO),
        decision=RecommendationDecision.APPROVED_BY_USER,
    )
    exec_summary = ExecutionSummary(
        operation_results=[OperationExecutionResult(
            recommendation_id=rec.recommendation_id, operation="NORMALIZE_CASE",
            target_columns=("city",), status=OperationExecutionStatus.FAILED, error="boom",
        )],
    )
    outcomes = build_recommendation_outcomes([record], exec_summary, _comparison())
    assert outcomes[0].execution_status == "failed"
    assert "boom" in outcomes[0].outcome_summary


def test_traceability_remove_duplicates_requires_exact_criteria_match():
    rec = _rec("REMOVE_DUPLICATES", ["name", "city"])
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.NEEDS_REVIEW),
        decision=RecommendationDecision.APPROVED_BY_USER,
    )
    exec_summary = ExecutionSummary(
        operation_results=[OperationExecutionResult(
            recommendation_id=rec.recommendation_id, operation="REMOVE_DUPLICATES",
            target_columns=("name", "city"), status=OperationExecutionStatus.SUCCEEDED, rows_affected=1,
        )],
    )
    comparison = _comparison(issue_comparisons=[
        _ic(IssueType.DUPLICATE_ROWS, ["city", "name"], IssueOutcome.RESOLVED, 1, 0)
    ])
    outcomes = build_recommendation_outcomes([record], exec_summary, comparison)
    assert len(outcomes[0].linked_issue_comparisons) == 1


def test_traceability_does_not_link_across_unrelated_columns():
    rec = _rec("TRIM_WHITESPACE", ["city"])
    record = RecommendationRecord(
        recommendation=rec, validation=_validated(ValidationStatus.APPROVED_AUTO),
        decision=RecommendationDecision.APPROVED_BY_USER,
    )
    exec_summary = ExecutionSummary(
        operation_results=[OperationExecutionResult(
            recommendation_id=rec.recommendation_id, operation="TRIM_WHITESPACE",
            target_columns=("city",), status=OperationExecutionStatus.SUCCEEDED, rows_affected=2,
        )],
    )
    comparison = _comparison(issue_comparisons=[
        _ic(IssueType.WHITESPACE_OR_CASING, ["department"], IssueOutcome.RESOLVED, 1, 0),  # different column
    ])
    outcomes = build_recommendation_outcomes([record], exec_summary, comparison)
    assert outcomes[0].linked_issue_comparisons == []

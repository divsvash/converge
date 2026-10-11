"""
Deterministic quality comparator.

NO LLM anywhere in this module. Given two ProfileResult objects (both
produced by the same Step 1 Spark profiler, at different points in
time), this module computes a structured QualityComparison purely by
comparing already-typed Pydantic fields (ColumnProfile, DuplicateSummary)
-- never by re-reading a dataset, never by asking a model to judge
anything.

Design choice: comparisons are computed from `ProfileResult.profile`
(the strongly-typed ColumnProfile / DuplicateSummary structures), not
by parsing `ProfileResult.evidence[].supporting_data` (free-form
dicts). The profile is the more precise, already-typed source for
severity numbers; evidence's role elsewhere in the system is to bound
what the AI sees, which isn't a concern here since no AI is involved.
`evidence` is still used for one thing: the simple aggregate "total
detected issues" counts in ComparisonMetrics, since that's literally
what it enumerates.

Every comparison this module cannot make honestly (a column
disappeared, a column's type changed away from what the issue type
requires) is reported as UNVERIFIED rather than guessed.
"""
from __future__ import annotations

from collections import Counter

from converge.models.comparison import (
    ComparisonMetrics,
    IssueComparison,
    IssueOutcome,
    QualityComparison,
)
from converge.models.profile import ColumnProfile, DuplicateSummary, IssueType, ProfileResult, SemanticType

_STRING_TYPES = {SemanticType.STRING}
_DATE_TYPES = {SemanticType.DATE_LIKE_STRING}
_NUMERIC_TYPES = {SemanticType.INTEGER, SemanticType.FLOAT}


def _missing_severity(col: ColumnProfile) -> int:
    return col.null_count


def _whitespace_severity(col: ColumnProfile) -> int:
    return col.categorical_stats.whitespace_affected_count if col.categorical_stats else 0


def _categorical_severity(col: ColumnProfile) -> int:
    if not col.categorical_stats:
        return 0
    return sum(len(g.variants) for g in col.categorical_stats.variant_groups)


def _date_format_severity(col: ColumnProfile) -> int:
    if not col.date_format_stats:
        return 0
    # Matches the profiler's own evidence-emission threshold: only
    # counts as an issue when MORE THAN ONE distinct format is present.
    pattern_count = len(col.date_format_stats.detected_patterns)
    return pattern_count if pattern_count > 1 else 0


def _outlier_severity(col: ColumnProfile) -> int:
    return col.numeric_stats.outlier_count if col.numeric_stats else 0


# (IssueType, severity_fn, required_semantic_types_or_None)
_COLUMN_ISSUE_SPECS = [
    (IssueType.MISSING_VALUES, _missing_severity, None),
    (IssueType.WHITESPACE_OR_CASING, _whitespace_severity, _STRING_TYPES),
    (IssueType.CATEGORICAL_INCONSISTENCY, _categorical_severity, _STRING_TYPES),
    (IssueType.DATE_FORMAT_INCONSISTENCY, _date_format_severity, _DATE_TYPES),
    (IssueType.OUTLIER_CANDIDATE, _outlier_severity, _NUMERIC_TYPES),
]


def _classify(before_sev: int, after_sev: int) -> IssueOutcome:
    if after_sev == 0:
        return IssueOutcome.RESOLVED
    if after_sev < before_sev:
        return IssueOutcome.IMPROVED
    if after_sev == before_sev:
        return IssueOutcome.UNCHANGED
    return IssueOutcome.WORSENED


def _duplicate_severity_by_criteria(dup_summary: DuplicateSummary) -> dict[tuple[str, ...], int]:
    totals: dict[tuple[str, ...], int] = {}
    for group in dup_summary.duplicate_candidates:
        key = tuple(sorted(group.criteria_columns))
        totals[key] = totals.get(key, 0) + max(group.occurrence_count - 1, 0)
    return totals


class QualityComparator:
    """Stateless -- compare() takes both profiles as arguments."""

    def compare(self, before: ProfileResult, after: ProfileResult) -> QualityComparison:
        before_cols = {c.name: c for c in before.profile.columns}
        after_cols = {c.name: c for c in after.profile.columns}

        issue_comparisons: list[IssueComparison] = []
        for issue_type, severity_fn, required_types in _COLUMN_ISSUE_SPECS:
            issue_comparisons.extend(
                self._compare_column_issue(issue_type, severity_fn, required_types, before_cols, after_cols)
            )
        issue_comparisons.extend(self._compare_duplicates(before, after))

        metrics = self._build_metrics(before, after, issue_comparisons)

        return QualityComparison(
            run_id=before.profile.dataset_id,
            issue_comparisons=issue_comparisons,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Column-scoped issue comparison
    # ------------------------------------------------------------------

    def _compare_column_issue(
        self,
        issue_type: IssueType,
        severity_fn,
        required_types: set | None,
        before_cols: dict[str, ColumnProfile],
        after_cols: dict[str, ColumnProfile],
    ) -> list[IssueComparison]:
        results: list[IssueComparison] = []
        for name in sorted(set(before_cols) | set(after_cols)):
            before_col = before_cols.get(name)
            after_col = after_cols.get(name)

            before_sev = severity_fn(before_col) if before_col is not None else 0

            if before_sev == 0 and after_col is None:
                continue  # never an issue, column doesn't even exist after -- nothing to report

            if before_sev == 0:
                after_ok = required_types is None or after_col.semantic_type in required_types
                after_sev = severity_fn(after_col) if after_ok else 0
                if after_sev > 0:
                    results.append(IssueComparison(
                        issue_type=issue_type, columns=[name], outcome=IssueOutcome.INTRODUCED,
                        before_value=0, after_value=after_sev,
                        detail=(
                            f"Column '{name}' had no {issue_type.value} issue before execution; "
                            f"a new issue (severity={after_sev}) is present after."
                        ),
                    ))
                continue

            # before_sev > 0 from here on.
            if after_col is None:
                results.append(IssueComparison(
                    issue_type=issue_type, columns=[name], outcome=IssueOutcome.UNVERIFIED,
                    before_value=before_sev, after_value=None,
                    detail=f"Column '{name}' no longer exists in the after-profile; outcome cannot be verified.",
                ))
                continue

            after_ok = required_types is None or after_col.semantic_type in required_types
            if not after_ok:
                results.append(IssueComparison(
                    issue_type=issue_type, columns=[name], outcome=IssueOutcome.UNVERIFIED,
                    before_value=before_sev, after_value=None,
                    detail=(
                        f"Column '{name}' semantic type changed from '{before_col.semantic_type.value}' to "
                        f"'{after_col.semantic_type.value}'; {issue_type.value} is no longer measurable "
                        f"for this column, so the outcome cannot be verified."
                    ),
                ))
                continue

            after_sev = severity_fn(after_col)
            outcome = _classify(before_sev, after_sev)
            results.append(IssueComparison(
                issue_type=issue_type, columns=[name], outcome=outcome,
                before_value=before_sev, after_value=after_sev,
                detail=f"Column '{name}' {issue_type.value}: {before_sev} -> {after_sev}.",
            ))
        return results

    # ------------------------------------------------------------------
    # Duplicate rows (row-level, not column-scoped)
    # ------------------------------------------------------------------

    def _compare_duplicates(self, before: ProfileResult, after: ProfileResult) -> list[IssueComparison]:
        before_by_key = _duplicate_severity_by_criteria(before.profile.duplicate_summary)
        after_by_key = _duplicate_severity_by_criteria(after.profile.duplicate_summary)
        after_column_names = {c.name for c in after.profile.columns}

        results: list[IssueComparison] = []
        for key in sorted(set(before_by_key) | set(after_by_key)):
            before_sev = before_by_key.get(key, 0)
            after_sev = after_by_key.get(key, 0)
            if before_sev == 0 and after_sev == 0:
                continue

            if before_sev == 0 and after_sev > 0:
                results.append(IssueComparison(
                    issue_type=IssueType.DUPLICATE_ROWS, columns=list(key), outcome=IssueOutcome.INTRODUCED,
                    before_value=0, after_value=after_sev,
                    detail=f"No duplicate rows for criteria {list(key)} before execution; {after_sev} found after.",
                ))
                continue

            if not set(key).issubset(after_column_names):
                results.append(IssueComparison(
                    issue_type=IssueType.DUPLICATE_ROWS, columns=list(key), outcome=IssueOutcome.UNVERIFIED,
                    before_value=before_sev, after_value=None,
                    detail=f"One or more criteria columns {list(key)} no longer exist after execution; outcome cannot be verified.",
                ))
                continue

            outcome = _classify(before_sev, after_sev)
            results.append(IssueComparison(
                issue_type=IssueType.DUPLICATE_ROWS, columns=list(key), outcome=outcome,
                before_value=before_sev, after_value=after_sev,
                detail=f"Duplicate rows removable for criteria {list(key)}: {before_sev} -> {after_sev}.",
            ))
        return results

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def _build_metrics(
        self, before: ProfileResult, after: ProfileResult, issue_comparisons: list[IssueComparison]
    ) -> ComparisonMetrics:
        before_counts = Counter(e.issue_type.value for e in before.evidence)
        after_counts = Counter(e.issue_type.value for e in after.evidence)
        outcome_counts = Counter(ic.outcome for ic in issue_comparisons)

        before_dup = _duplicate_severity_by_criteria(before.profile.duplicate_summary)
        after_dup = _duplicate_severity_by_criteria(after.profile.duplicate_summary)

        before_cols = before.profile.columns
        after_cols = after.profile.columns

        return ComparisonMetrics(
            total_issues_before=len(before.evidence),
            total_issues_after=len(after.evidence),
            resolved_count=outcome_counts.get(IssueOutcome.RESOLVED, 0),
            improved_count=outcome_counts.get(IssueOutcome.IMPROVED, 0),
            unchanged_count=outcome_counts.get(IssueOutcome.UNCHANGED, 0),
            worsened_count=outcome_counts.get(IssueOutcome.WORSENED, 0),
            introduced_count=outcome_counts.get(IssueOutcome.INTRODUCED, 0),
            unverified_count=outcome_counts.get(IssueOutcome.UNVERIFIED, 0),
            issue_counts_by_type_before=dict(before_counts),
            issue_counts_by_type_after=dict(after_counts),
            row_count_before=before.profile.row_count,
            row_count_after=after.profile.row_count,
            column_count_before=before.profile.column_count,
            column_count_after=after.profile.column_count,
            schema_before=[{"name": c.name, "dtype": c.spark_dtype} for c in before_cols],
            schema_after=[{"name": c.name, "dtype": c.spark_dtype} for c in after_cols],
            exact_duplicate_row_count_before=before.profile.duplicate_summary.exact_duplicate_row_count,
            exact_duplicate_row_count_after=after.profile.duplicate_summary.exact_duplicate_row_count,
            duplicate_candidate_row_count_before=sum(before_dup.values()),
            duplicate_candidate_row_count_after=sum(after_dup.values()),
            missing_value_total_before=sum(c.null_count for c in before_cols),
            missing_value_total_after=sum(c.null_count for c in after_cols),
            categorical_variant_total_before=sum(_categorical_severity(c) for c in before_cols),
            categorical_variant_total_after=sum(_categorical_severity(c) for c in after_cols),
            date_format_inconsistency_columns_before=sum(1 for c in before_cols if _date_format_severity(c) > 0),
            date_format_inconsistency_columns_after=sum(1 for c in after_cols if _date_format_severity(c) > 0),
            outlier_total_before=sum(_outlier_severity(c) for c in before_cols),
            outlier_total_after=sum(_outlier_severity(c) for c in after_cols),
        )

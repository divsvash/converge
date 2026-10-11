"""
Spark dataset profiler.

Responsibility (per the approved architecture):
    - Spark is the sole source of deterministic facts about the dataset.
    - The profiler discovers whatever issues are actually present; it
      does not force every possible issue category into every dataset.
    - The profiler also produces bounded "evidence" -- representative
      samples and/or value distributions -- that later stages (AI,
      validator) will consume. The AI never touches the raw dataset;
      it only ever sees what this module decides to hand it.

This module has no dependency on FastAPI, the AI provider, or the
validator. It is pure: DataFrame in, typed Pydantic objects out.
"""
from __future__ import annotations

import re

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    BooleanType,
)

from converge.core import config
from converge.core.logging import get_logger
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

logger = get_logger(__name__)

_ID_COLUMN_PATTERN = re.compile(r"(^|_)id($|_)", re.IGNORECASE)

_DATE_PATTERNS: dict[str, str] = {
    "YYYY-MM-DD": r"^\d{4}-\d{2}-\d{2}$",
    "DD/MM/YYYY_or_MM/DD/YYYY": r"^\d{2}/\d{2}/\d{4}$",
}

# internal-only helper column used to reference specific rows in
# evidence sampling when no natural id-like column exists. Never
# exposed in ColumnProfile or the final row data.
_ROW_REF_COL = "__row_ref__"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def profile_dataset(df: DataFrame, dataset_id: str) -> ProfileResult:
    """
    Runs the full profiling pass over a DataFrame and returns the
    complete profile + evidence contract.
    """
    row_count = df.count()
    columns = df.columns

    id_like_columns = _detect_id_like_columns(df, columns, row_count)

    logger.info(
        "Profiling dataset '%s': %d rows, %d columns (id-like: %s)",
        dataset_id, row_count, len(columns), id_like_columns,
    )

    working_df = df.withColumn(_ROW_REF_COL, F.monotonically_increasing_id().cast("string"))

    column_profiles: list[ColumnProfile] = []
    for col_name, spark_dtype in df.dtypes:
        column_profiles.append(
            _profile_column(working_df, col_name, spark_dtype, row_count, id_like_columns)
        )

    duplicate_summary = _detect_duplicates(working_df, columns, id_like_columns, row_count)

    profile = DatasetProfile(
        dataset_id=dataset_id,
        row_count=row_count,
        column_count=len(columns),
        columns=column_profiles,
        duplicate_summary=duplicate_summary,
    )

    evidence = _generate_evidence(working_df, profile, id_like_columns)

    return ProfileResult(profile=profile, evidence=evidence)


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def _classify_semantic_type(spark_dtype: str, col_name: str) -> SemanticType:
    if spark_dtype in ("int", "bigint", "smallint", "tinyint"):
        return SemanticType.INTEGER
    if spark_dtype in ("double", "float", "decimal"):
        return SemanticType.FLOAT
    if spark_dtype == "boolean":
        return SemanticType.BOOLEAN
    if spark_dtype == "string":
        name_lower = col_name.lower()
        if any(hint in name_lower for hint in config.DATE_COLUMN_HINTS):
            return SemanticType.DATE_LIKE_STRING
        return SemanticType.STRING
    return SemanticType.UNKNOWN


def _detect_id_like_columns(df: DataFrame, columns: list[str], row_count: int) -> list[str]:
    """
    A column is treated as id-like if its name matches an id-ish
    pattern (word-boundary aware, so 'valid' does not match) AND it is
    near-unique. Name-only matching is intentionally insufficient --
    a column literally called 'id' that is mostly duplicated is not
    a reliable identifier and should not be excluded from duplicate
    criteria.
    """
    if row_count == 0:
        return []

    candidates = [c for c in columns if _ID_COLUMN_PATTERN.search(c)]
    id_like: list[str] = []
    for c in candidates:
        distinct = df.select(c).distinct().count()
        if distinct / row_count >= 0.9:
            id_like.append(c)
    return id_like


# ---------------------------------------------------------------------------
# Per-column profiling
# ---------------------------------------------------------------------------

def _profile_column(
    df: DataFrame,
    col_name: str,
    spark_dtype: str,
    row_count: int,
    id_like_columns: list[str],
) -> ColumnProfile:
    non_null_count = df.filter(F.col(col_name).isNotNull()).count()
    null_count = row_count - non_null_count
    null_percentage = (null_count / row_count * 100.0) if row_count else 0.0
    distinct_count = df.select(col_name).distinct().count()

    sample_values = [
        str(r[0]) for r in
        df.select(col_name).filter(F.col(col_name).isNotNull()).limit(5).collect()
    ]

    semantic_type = _classify_semantic_type(spark_dtype, col_name)
    is_id_like = col_name in id_like_columns

    numeric_stats = None
    categorical_stats = None
    date_format_stats = None

    if semantic_type in (SemanticType.INTEGER, SemanticType.FLOAT) and not is_id_like:
        numeric_stats = _profile_numeric_column(df, col_name)
    elif semantic_type == SemanticType.DATE_LIKE_STRING:
        date_format_stats = _profile_date_column(df, col_name)
    elif semantic_type == SemanticType.STRING and not is_id_like:
        categorical_stats = _profile_categorical_column(df, col_name, row_count, distinct_count)

    return ColumnProfile(
        name=col_name,
        spark_dtype=spark_dtype,
        semantic_type=semantic_type,
        non_null_count=non_null_count,
        null_count=null_count,
        null_percentage=round(null_percentage, 2),
        distinct_count=distinct_count,
        is_id_like=is_id_like,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        date_format_stats=date_format_stats,
        sample_values=sample_values,
    )


def _profile_numeric_column(df: DataFrame, col_name: str) -> NumericStats:
    stats_row = df.select(
        F.min(col_name).alias("min"),
        F.max(col_name).alias("max"),
        F.mean(col_name).alias("mean"),
        F.stddev(col_name).alias("stddev"),
    ).first()

    quantiles = df.filter(F.col(col_name).isNotNull()).approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
    q1, median, q3 = (quantiles + [None, None, None])[:3] if len(quantiles) < 3 else quantiles

    outlier_lower = outlier_upper = None
    outlier_count = 0
    if q1 is not None and q3 is not None:
        iqr = q3 - q1
        outlier_lower = q1 - config.OUTLIER_IQR_MULTIPLIER * iqr
        outlier_upper = q3 + config.OUTLIER_IQR_MULTIPLIER * iqr
        outlier_count = df.filter(
            F.col(col_name).isNotNull()
            & ((F.col(col_name) < outlier_lower) | (F.col(col_name) > outlier_upper))
        ).count()

    return NumericStats(
        min=float(stats_row["min"]) if stats_row["min"] is not None else 0.0,
        max=float(stats_row["max"]) if stats_row["max"] is not None else 0.0,
        mean=round(float(stats_row["mean"]), 4) if stats_row["mean"] is not None else 0.0,
        stddev=round(float(stats_row["stddev"]), 4) if stats_row["stddev"] is not None else None,
        median_approx=median,
        q1=q1,
        q3=q3,
        outlier_lower_bound=outlier_lower,
        outlier_upper_bound=outlier_upper,
        outlier_count=outlier_count,
    )


def _profile_categorical_column(
    df: DataFrame, col_name: str, row_count: int, distinct_count: int
) -> CategoricalStats:
    if row_count == 0:
        return CategoricalStats(distinct_count=0)

    # Skip variant-grouping on high-cardinality free-text columns (e.g.
    # names) -- grouping is only meaningful for genuinely categorical
    # columns, and running it on free text would produce noise, not
    # evidence.
    #
    # Important: the shape check uses the *normalized* (lower+trim)
    # distinct count, not the raw distinct count. Using raw distinct
    # count would be self-defeating: a column with heavy case/whitespace
    # inconsistency (exactly what we're trying to detect) inflates raw
    # cardinality and would wrongly disqualify itself from analysis.
    normalized_distinct_count = (
        df.filter(F.col(col_name).isNotNull())
        .select(F.lower(F.trim(F.col(col_name))).alias("_norm"))
        .distinct()
        .count()
    )
    distinct_ratio = normalized_distinct_count / row_count
    is_categorical_shape = (
        distinct_ratio <= config.CATEGORICAL_DISTINCT_RATIO_THRESHOLD
        or row_count < config.CATEGORICAL_MIN_ROWS_FOR_RATIO_CHECK
    )

    value_counts = (
        df.filter(F.col(col_name).isNotNull())
        .groupBy(col_name)
        .count()
        .orderBy(F.desc("count"))
        .limit(config.MAX_EVIDENCE_DISTINCT_VALUES)
        .collect()
    )
    top_values = {str(r[col_name]): r["count"] for r in value_counts}

    whitespace_affected_count = df.filter(
        F.col(col_name).isNotNull() & (F.trim(F.col(col_name)) != F.col(col_name))
    ).count()

    variant_groups: list[CategoricalVariantGroup] = []
    if is_categorical_shape:
        normalized = (
            df.filter(F.col(col_name).isNotNull())
            .withColumn("_norm", F.lower(F.trim(F.col(col_name))))
            .groupBy("_norm", col_name)
            .count()
        )
        grouped = normalized.groupBy("_norm").agg(
            F.collect_list(F.struct(F.col(col_name).alias("val"), F.col("count").alias("cnt"))).alias("variants"),
            F.count("*").alias("variant_count"),
        ).filter(F.col("variant_count") > 1).collect()

        for row in grouped:
            variants = {str(v["val"]): v["cnt"] for v in row["variants"]}
            variant_groups.append(
                CategoricalVariantGroup(normalized_key=row["_norm"], variants=variants)
            )

    return CategoricalStats(
        distinct_count=distinct_count,
        top_values=top_values,
        variant_groups=variant_groups,
        whitespace_affected_count=whitespace_affected_count,
    )


def _profile_date_column(df: DataFrame, col_name: str) -> DateFormatStats:
    non_null_df = df.filter(F.col(col_name).isNotNull())
    total = non_null_df.count()

    detected: dict[str, int] = {}
    matched_total = 0
    for label, pattern in _DATE_PATTERNS.items():
        count = non_null_df.filter(F.col(col_name).rlike(pattern)).count()
        if count > 0:
            detected[label] = count
            matched_total += count

    unparseable = total - matched_total
    return DateFormatStats(detected_patterns=detected, unparseable_count=unparseable)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _detect_duplicates(
    df: DataFrame, original_columns: list[str], id_like_columns: list[str], row_count: int
) -> DuplicateSummary:
    if row_count == 0:
        return DuplicateSummary(exact_duplicate_row_count=0, id_like_columns=id_like_columns)

    distinct_row_count = df.select(*original_columns).distinct().count()
    exact_duplicate_row_count = row_count - distinct_row_count

    non_id_cols = [c for c in original_columns if c not in id_like_columns]
    duplicate_candidates: list[DuplicateGroup] = []

    if non_id_cols:
        grouped = (
            df.groupBy(*non_id_cols)
            .count()
            .filter(F.col("count") > 1)
            .orderBy(F.desc("count"))
            .limit(config.MAX_EVIDENCE_SAMPLE_ROWS)
            .collect()
        )

        ref_col = id_like_columns[0] if id_like_columns else _ROW_REF_COL

        for group_row in grouped:
            conditions = None
            for c in non_id_cols:
                val = group_row[c]
                cond = F.col(c).eqNullSafe(val)
                conditions = cond if conditions is None else (conditions & cond)

            matching_refs = [
                str(r[0])
                for r in df.filter(conditions).select(ref_col).limit(config.MAX_EVIDENCE_SAMPLE_ROWS).collect()
            ]
            duplicate_candidates.append(
                DuplicateGroup(
                    criteria_columns=non_id_cols,
                    row_refs=matching_refs,
                    occurrence_count=group_row["count"],
                )
            )

    return DuplicateSummary(
        exact_duplicate_row_count=exact_duplicate_row_count,
        id_like_columns=id_like_columns,
        duplicate_candidates=duplicate_candidates,
    )


# ---------------------------------------------------------------------------
# Evidence generation (the AI-facing bounded contract)
# ---------------------------------------------------------------------------

def _generate_evidence(
    df: DataFrame, profile: DatasetProfile, id_like_columns: list[str]
) -> list[EvidenceItem]:
    """
    Builds the bounded evidence bundle from an already-computed profile.
    Only issues that are actually present produce evidence items --
    nothing is fabricated to fill out a checklist.
    """
    evidence: list[EvidenceItem] = []

    for col in profile.columns:
        if col.null_count > 0:
            evidence.append(
                EvidenceItem(
                    issue_type=IssueType.MISSING_VALUES,
                    columns=[col.name],
                    description=(
                        f"{col.null_count} of {profile.row_count} rows have a "
                        f"missing value in '{col.name}' ({col.null_percentage}%)."
                    ),
                    supporting_data={
                        "null_count": col.null_count,
                        "null_percentage": col.null_percentage,
                        "semantic_type": col.semantic_type.value,
                    },
                )
            )

        if col.categorical_stats and col.categorical_stats.variant_groups:
            for group in col.categorical_stats.variant_groups:
                evidence.append(
                    EvidenceItem(
                        issue_type=IssueType.CATEGORICAL_INCONSISTENCY,
                        columns=[col.name],
                        description=(
                            f"Column '{col.name}' has {len(group.variants)} distinct "
                            f"raw values that normalize to '{group.normalized_key}'."
                        ),
                        supporting_data={"variants": group.variants},
                    )
                )

        if col.categorical_stats and col.categorical_stats.whitespace_affected_count > 0:
            evidence.append(
                EvidenceItem(
                    issue_type=IssueType.WHITESPACE_OR_CASING,
                    columns=[col.name],
                    description=(
                        f"{col.categorical_stats.whitespace_affected_count} values in "
                        f"'{col.name}' have leading/trailing whitespace."
                    ),
                    supporting_data={
                        "affected_count": col.categorical_stats.whitespace_affected_count,
                        "sample_values": col.sample_values,
                    },
                )
            )

        if col.date_format_stats and len(col.date_format_stats.detected_patterns) > 1:
            evidence.append(
                EvidenceItem(
                    issue_type=IssueType.DATE_FORMAT_INCONSISTENCY,
                    columns=[col.name],
                    description=(
                        f"Column '{col.name}' contains {len(col.date_format_stats.detected_patterns)} "
                        f"different date formats."
                    ),
                    supporting_data={"detected_patterns": col.date_format_stats.detected_patterns},
                )
            )

        if col.numeric_stats and col.numeric_stats.outlier_count > 0:
            evidence.append(
                EvidenceItem(
                    issue_type=IssueType.OUTLIER_CANDIDATE,
                    columns=[col.name],
                    description=(
                        f"{col.numeric_stats.outlier_count} values in '{col.name}' fall "
                        f"outside the IQR-based bounds "
                        f"[{col.numeric_stats.outlier_lower_bound}, {col.numeric_stats.outlier_upper_bound}]."
                    ),
                    supporting_data={
                        "outlier_count": col.numeric_stats.outlier_count,
                        "lower_bound": col.numeric_stats.outlier_lower_bound,
                        "upper_bound": col.numeric_stats.outlier_upper_bound,
                        "min": col.numeric_stats.min,
                        "max": col.numeric_stats.max,
                    },
                )
            )

    if profile.duplicate_summary.duplicate_candidates:
        for group in profile.duplicate_summary.duplicate_candidates:
            evidence.append(
                EvidenceItem(
                    issue_type=IssueType.DUPLICATE_ROWS,
                    columns=group.criteria_columns,
                    description=(
                        f"{group.occurrence_count} rows are identical across "
                        f"{len(group.criteria_columns)} column(s) "
                        f"(excluding id-like columns: {id_like_columns})."
                    ),
                    supporting_data={
                        "row_refs": group.row_refs,
                        "occurrence_count": group.occurrence_count,
                    },
                )
            )

    return evidence

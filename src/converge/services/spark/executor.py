"""
Spark executor.

SECURITY / SAFETY BOUNDARY: this module is the only code in CONVERGE
that is permitted to mutate a dataset. It is given exactly one input
type -- `ExecutionPlan` -- and nothing else. It never:

    - receives the mutable PipelineRun, a Recommendation's `reason`,
      `evidence_refs`, or `confidence` fields, or anything
      natural-language.
    - calls `spark.sql(...)` with any string built from
      recommendation-derived content (no SQL injection surface exists
      because no SQL string is ever constructed at all -- every
      operation is implemented purely with the DataFrame API).
    - uses `eval`, `exec`, or any dynamic code execution of any
      parameter value.
    - re-derives or re-checks user-approval or validator-eligibility
      logic. A PlannedOperation existing in the plan at all is the
      only fact the executor relies on for "was this authorized" --
      it has no access to `decision` or `validation` to even attempt
      re-checking them.
    - reorders, merges, optimizes, or otherwise reinterprets the plan.
      `ExecutionPlan.operations` is an explicit, ordered execution
      contract: the executor runs it in EXACT plan order, full stop.
      Any operation-order validity constraint (e.g. "type conversion
      should generally happen before a mean-fill on the same column")
      is a planning-time concern, decided when the plan is built
      (converge.services.pipeline.orchestrator.build_execution_plan),
      never a hidden, executor-side reordering the plan's author
      didn't ask for and can't see.

Every operation handler independently re-validates its own required
parameters (exact allowed values, required keys, column existence)
before touching Spark, even though the Validator (Step 2) already
performed equivalent checks upstream. This is deliberate defense in
depth: the executor does not assume anything reaching it was actually
validated -- it behaves safely even if given a hand-constructed,
never-validated ExecutionPlan (which the test suite exercises
directly).

Execution is deterministic and fail-fast: operations run in exactly
the order given in the plan, and the first failure stops all further
operations -- no partial, silently-incomplete transform is ever
written out as if it were a complete result.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from converge.core import config
from converge.core.logging import get_logger
from converge.models.execution import (
    ExecutionPlan,
    ExecutionSummary,
    OperationExecutionResult,
    OperationExecutionStatus,
    PlannedOperation,
)
from converge.services.spark.loader import load_csv

logger = get_logger(__name__)

_NUMERIC_DTYPES = ("int", "bigint", "smallint", "tinyint", "double", "float", "decimal")


class SparkExecutor:
    def __init__(self, spark: SparkSession):
        self._spark = spark

    def execute(self, plan: ExecutionPlan) -> ExecutionSummary:
        df = load_csv(self._spark, plan.dataset_path)
        schema_before = _schema_snapshot(df)

        # Executed in EXACT plan order -- no sorting, no precedence,
        # no reinterpretation. See module docstring.
        results: list[OperationExecutionResult] = []
        succeeded = 0
        failed = 0
        stop = False

        for planned_op in plan.operations:
            if stop:
                results.append(OperationExecutionResult(
                    recommendation_id=planned_op.recommendation_id,
                    operation=planned_op.operation,
                    target_columns=planned_op.target_columns,
                    status=OperationExecutionStatus.SKIPPED,
                ))
                continue

            try:
                df, rows_affected = _apply_operation(df, planned_op)
            except Exception as e:
                logger.error(
                    "Operation '%s' (recommendation_id=%s) failed: %s",
                    planned_op.operation, planned_op.recommendation_id, e,
                )
                results.append(OperationExecutionResult(
                    recommendation_id=planned_op.recommendation_id,
                    operation=planned_op.operation,
                    target_columns=planned_op.target_columns,
                    status=OperationExecutionStatus.FAILED,
                    error=str(e),
                ))
                failed += 1
                stop = True  # fail-fast: no further operations are applied
                continue

            results.append(OperationExecutionResult(
                recommendation_id=planned_op.recommendation_id,
                operation=planned_op.operation,
                target_columns=planned_op.target_columns,
                status=OperationExecutionStatus.SUCCEEDED,
                rows_affected=rows_affected,
            ))
            succeeded += 1

        schema_after = _schema_snapshot(df)
        skipped = sum(1 for r in results if r.status == OperationExecutionStatus.SKIPPED)

        output_path = None
        if failed == 0 and skipped == 0:
            output_path = _write_output(df, plan.run_id)

        return ExecutionSummary(
            operations_attempted=succeeded + failed,
            operations_succeeded=succeeded,
            operations_failed=failed,
            operations_skipped=skipped,
            operation_results=results,
            output_dataset_path=output_path,
            schema_before=schema_before,
            schema_after=schema_after,
            notes=(
                "All operations succeeded." if failed == 0 and skipped == 0
                else f"{failed} operation(s) failed; {skipped} skipped due to fail-fast stop."
            ),
        )


# ---------------------------------------------------------------------------
# Operation dispatch
# ---------------------------------------------------------------------------

def _apply_operation(df: DataFrame, planned_op: PlannedOperation) -> tuple[DataFrame, int]:
    handler = _OPERATION_HANDLERS.get(planned_op.operation)
    if handler is None:
        # Structurally should never happen: the orchestrator only ever
        # builds a PlannedOperation from a recommendation the Validator
        # classified as APPROVED_AUTO or NEEDS_REVIEW, and every
        # operation with those statuses is one of the 7 handlers below.
        # FLAG_OUTLIER (FINDING_ONLY) and any unregistered/forbidden
        # operation name land here and fail cleanly rather than being
        # silently ignored or executed.
        raise ValueError(
            f"'{planned_op.operation}' has no executor implementation. It is either "
            f"FINDING_ONLY (e.g. FLAG_OUTLIER, which must never execute) or not a "
            f"recognized P0 operation. This plan should never have reached the executor."
        )
    return handler(df, list(planned_op.target_columns), dict(planned_op.parameters))


def _require_column(df: DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset (columns: {df.columns}).")


# ---------------------------------------------------------------------------
# Operation implementations
#
# Every handler: (DataFrame, target_columns, parameters) -> (DataFrame, rows_affected)
# Pure DataFrame-API transformations only. No spark.sql(), no eval/exec,
# no dynamic attribute lookup driven by parameter content.
# ---------------------------------------------------------------------------

def _apply_trim_whitespace(df: DataFrame, columns: list[str], params: dict) -> tuple[DataFrame, int]:
    if not columns:
        raise ValueError("TRIM_WHITESPACE requires at least one target column.")
    for c in columns:
        _require_column(df, c)

    condition = None
    for c in columns:
        cond = F.col(c).isNotNull() & (F.trim(F.col(c)) != F.col(c))
        condition = cond if condition is None else (condition | cond)
    rows_affected = df.filter(condition).count() if condition is not None else 0

    result_df = df
    for c in columns:
        result_df = result_df.withColumn(c, F.trim(F.col(c)))
    return result_df, rows_affected


_ALLOWED_CASE_FNS = {"lower": F.lower, "upper": F.upper, "title": F.initcap}


def _apply_normalize_case(df: DataFrame, columns: list[str], params: dict) -> tuple[DataFrame, int]:
    if not columns:
        raise ValueError("NORMALIZE_CASE requires at least one target column.")
    for c in columns:
        _require_column(df, c)

    target_case = params.get("target_case")
    if target_case not in _ALLOWED_CASE_FNS:
        raise ValueError(
            f"NORMALIZE_CASE requires parameter 'target_case' to be one of "
            f"{sorted(_ALLOWED_CASE_FNS)}, got {target_case!r}."
        )
    case_fn = _ALLOWED_CASE_FNS[target_case]

    condition = None
    for c in columns:
        cond = F.col(c).isNotNull() & (case_fn(F.col(c)) != F.col(c))
        condition = cond if condition is None else (condition | cond)
    rows_affected = df.filter(condition).count() if condition is not None else 0

    result_df = df
    for c in columns:
        result_df = result_df.withColumn(c, case_fn(F.col(c)))
    return result_df, rows_affected


def _apply_standardize_category(df: DataFrame, columns: list[str], params: dict) -> tuple[DataFrame, int]:
    if len(columns) != 1:
        raise ValueError(f"STANDARDIZE_CATEGORY requires exactly one target column, got {columns}.")
    c = columns[0]
    _require_column(df, c)

    mapping = params.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "STANDARDIZE_CATEGORY requires a non-empty 'mapping' parameter "
            "(a dict of raw_value -> canonical_value)."
        )
    for k, v in mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                "STANDARDIZE_CATEGORY mapping keys and values must be plain strings; "
                f"got key {k!r} ({type(k).__name__}) -> value {v!r} ({type(v).__name__})."
            )

    original_col = F.col(c)
    mapped_col = original_col
    for raw_value, canonical_value in mapping.items():
        mapped_col = F.when(original_col == F.lit(raw_value), F.lit(canonical_value)).otherwise(mapped_col)

    changed_keys = [k for k, v in mapping.items() if k != v]
    rows_affected = df.filter(original_col.isin(changed_keys)).count() if changed_keys else 0

    result_df = df.withColumn(c, mapped_col)
    return result_df, rows_affected


# Documented P0 assumption: the ambiguous DD/MM vs MM/DD slash format
# detected by the profiler is interpreted as DD/MM/YYYY on input. This
# matches how the P0 sample dataset was constructed and is a known,
# explicit limitation for genuinely ambiguous real-world data -- not a
# general date-parsing solution.
_DATE_FORMAT_SPECS: dict[str, str] = {
    "YYYY-MM-DD": "yyyy-MM-dd",
    "DD/MM/YYYY": "dd/MM/yyyy",
    "MM/DD/YYYY": "MM/dd/yyyy",
}
_INPUT_DATE_FORMATS = ["yyyy-MM-dd", "dd/MM/yyyy"]


def _apply_standardize_date_format(df: DataFrame, columns: list[str], params: dict) -> tuple[DataFrame, int]:
    if len(columns) != 1:
        raise ValueError(f"STANDARDIZE_DATE_FORMAT requires exactly one target column, got {columns}.")
    c = columns[0]
    _require_column(df, c)

    target_format = params.get("target_format")
    if target_format not in _DATE_FORMAT_SPECS:
        raise ValueError(
            f"STANDARDIZE_DATE_FORMAT requires 'target_format' to be one of "
            f"{sorted(_DATE_FORMAT_SPECS)}, got {target_format!r}."
        )
    spark_pattern = _DATE_FORMAT_SPECS[target_format]

    parsed = None
    for fmt in _INPUT_DATE_FORMATS:
        candidate = F.to_date(F.col(c), fmt)
        parsed = candidate if parsed is None else F.coalesce(parsed, candidate)
    reformatted = F.date_format(parsed, spark_pattern)

    rows_affected = df.filter(F.col(c).isNotNull() & (reformatted != F.col(c))).count()
    result_df = df.withColumn(c, reformatted)
    return result_df, rows_affected


def _coerce_fill_value(raw, dtype: str):
    if not isinstance(raw, str):
        raise ValueError(f"fill_value must be a string, got {type(raw).__name__}.")
    if dtype in ("int", "bigint", "smallint", "tinyint"):
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"fill_value '{raw}' is not a valid integer for column dtype '{dtype}'.")
    if dtype in ("double", "float", "decimal"):
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"fill_value '{raw}' is not a valid number for column dtype '{dtype}'.")
    if dtype == "boolean":
        if raw.lower() not in ("true", "false"):
            raise ValueError(f"fill_value '{raw}' is not a valid boolean for column dtype '{dtype}'.")
        return raw.lower() == "true"
    return raw  # string column: used as-is


def _apply_fill_missing(df: DataFrame, columns: list[str], params: dict) -> tuple[DataFrame, int]:
    if len(columns) != 1:
        raise ValueError(f"FILL_MISSING requires exactly one target column, got {columns}.")
    c = columns[0]
    _require_column(df, c)

    strategy = params.get("strategy")
    if strategy not in ("mean", "median", "mode", "constant"):
        raise ValueError(
            f"FILL_MISSING requires 'strategy' to be one of mean/median/mode/constant, got {strategy!r}."
        )

    null_count = df.filter(F.col(c).isNull()).count()
    if null_count == 0:
        return df, 0

    dtype = dict(df.dtypes)[c]
    is_numeric = dtype in _NUMERIC_DTYPES

    if strategy == "constant":
        fill_value_raw = params.get("fill_value")
        if fill_value_raw is None:
            raise ValueError("FILL_MISSING strategy 'constant' requires parameter 'fill_value'.")
        fill_value = _coerce_fill_value(fill_value_raw, dtype)
    elif strategy == "mean":
        if not is_numeric:
            raise ValueError(f"FILL_MISSING strategy 'mean' requires a numeric column; '{c}' is '{dtype}'.")
        row = df.select(F.mean(c).alias("v")).first()
        fill_value = row["v"]
    elif strategy == "median":
        if not is_numeric:
            raise ValueError(f"FILL_MISSING strategy 'median' requires a numeric column; '{c}' is '{dtype}'.")
        quantiles = df.filter(F.col(c).isNotNull()).approxQuantile(c, [0.5], 0.01)
        if not quantiles:
            raise ValueError(f"Could not compute median for column '{c}': no non-null values.")
        fill_value = quantiles[0]
    else:  # mode
        mode_row = (
            df.filter(F.col(c).isNotNull())
            .groupBy(c).count()
            .orderBy(F.desc("count"))
            .first()
        )
        if mode_row is None:
            raise ValueError(f"Could not compute mode for column '{c}': no non-null values.")
        fill_value = mode_row[c]

    result_df = df.na.fill({c: fill_value})
    return result_df, null_count


_CONVERT_TYPE_MAP: dict[str, str] = {
    "integer": "int",
    "float": "double",
    "string": "string",
    "boolean": "boolean",
}


def _apply_convert_dtype(df: DataFrame, columns: list[str], params: dict) -> tuple[DataFrame, int]:
    if len(columns) != 1:
        raise ValueError(f"CONVERT_DTYPE requires exactly one target column, got {columns}.")
    c = columns[0]
    _require_column(df, c)

    target_type = params.get("target_type")
    if target_type not in _CONVERT_TYPE_MAP:
        raise ValueError(
            f"CONVERT_DTYPE requires 'target_type' to be one of {sorted(_CONVERT_TYPE_MAP)}, "
            f"got {target_type!r}."
        )
    spark_type = _CONVERT_TYPE_MAP[target_type]

    rows_affected = df.filter(F.col(c).isNotNull()).count()
    result_df = df.withColumn(c, F.col(c).cast(spark_type))
    return result_df, rows_affected


def _apply_remove_duplicates(df: DataFrame, columns: list[str], params: dict) -> tuple[DataFrame, int]:
    if not columns:
        raise ValueError("REMOVE_DUPLICATES requires at least one criteria column.")
    for c in columns:
        _require_column(df, c)

    before = df.count()
    result_df = df.dropDuplicates(subset=columns)
    after = result_df.count()
    return result_df, before - after


_OPERATION_HANDLERS = {
    "TRIM_WHITESPACE": _apply_trim_whitespace,
    "NORMALIZE_CASE": _apply_normalize_case,
    "STANDARDIZE_CATEGORY": _apply_standardize_category,
    "STANDARDIZE_DATE_FORMAT": _apply_standardize_date_format,
    "FILL_MISSING": _apply_fill_missing,
    "CONVERT_DTYPE": _apply_convert_dtype,
    "REMOVE_DUPLICATES": _apply_remove_duplicates,
}


# ---------------------------------------------------------------------------
# Schema snapshot + output writing
# ---------------------------------------------------------------------------

def _schema_snapshot(df: DataFrame) -> list[dict[str, str]]:
    return [{"name": name, "dtype": dtype} for name, dtype in df.dtypes]


def _write_output(df: DataFrame, run_id: str) -> str:
    """
    Writes the fully-cleaned dataset to a single, stable, predictable
    CSV path: data/runs/{run_id}/cleaned.csv. Spark's native writer
    produces a partitioned output directory; for P0's small datasets,
    coalescing to a single partition and renaming the resulting part
    file gives one plain CSV file rather than requiring the rest of
    the system to understand Spark's output directory layout.
    """
    output_dir = config.RUNS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned.csv"
    tmp_dir = output_dir / "_cleaned_tmp"

    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(tmp_dir))

    part_files = list(Path(tmp_dir).glob("part-*.csv"))
    if not part_files:
        raise RuntimeError("Spark did not produce an output part file for the cleaned dataset.")

    if output_path.exists():
        output_path.unlink()
    part_files[0].rename(output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return str(output_path)

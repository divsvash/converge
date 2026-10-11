"""
Spark executor tests.

Uses the REAL SparkExecutor against real, small, purpose-built CSV
datasets (plus the full sample_hr.csv for integration-style tests) --
no mocking of Spark itself. ExecutionPlan objects are constructed
directly in most tests (bypassing the orchestrator/validator entirely)
specifically to exercise the executor's own independent defenses: it
must behave safely even when given a plan that was never validated.

Per the architecture correction, the executor performs ZERO reordering
of ExecutionPlan.operations -- it runs exactly the sequence given.
Several tests below exist specifically to prove that (including one
where "wrong" order causes a real failure, and one where identical
operations in reversed order produce different, order-dependent
results).
"""
import csv
import uuid

import pytest

from converge.core.config import RAW_DATA_DIR
from converge.models.execution import ExecutionPlan, OperationExecutionStatus, PlannedOperation
from converge.services.spark.executor import SparkExecutor
from converge.services.spark.loader import load_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, header: list[str], rows: list[list]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def _plan(run_id: str, dataset_path: str, operations: list[PlannedOperation]) -> ExecutionPlan:
    return ExecutionPlan(run_id=run_id, dataset_path=dataset_path, operations=tuple(operations))


def _op(operation: str, target_columns: list[str], parameters: dict | None = None, rec_id: str | None = None) -> PlannedOperation:
    return PlannedOperation(
        recommendation_id=rec_id or str(uuid.uuid4()),
        operation=operation,
        target_columns=tuple(target_columns),
        parameters=parameters or {},
    )


@pytest.fixture()
def executor(spark):
    return SparkExecutor(spark)


@pytest.fixture()
def sample_hr_path():
    return str(RAW_DATA_DIR / "sample_hr.csv")


# ---------------------------------------------------------------------------
# TRIM_WHITESPACE
# ---------------------------------------------------------------------------

def test_trim_whitespace_operation(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["name"], [["  Alice "], ["Bob"], ["  Cara"]])
    plan = _plan("run1", path, [_op("TRIM_WHITESPACE", ["name"])])
    summary = executor.execute(plan)

    assert summary.fully_successful
    assert summary.operations_succeeded == 1
    assert summary.operation_results[0].rows_affected == 2  # Alice and Cara had whitespace

    result_df = load_csv(spark, summary.output_dataset_path)
    names = sorted(r["name"] for r in result_df.collect())
    assert names == ["Alice", "Bob", "Cara"]


# ---------------------------------------------------------------------------
# NORMALIZE_CASE
# ---------------------------------------------------------------------------

def test_normalize_case_operation(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["delhi"], ["DELHI"], ["Mumbai"]])
    plan = _plan("run2", path, [_op("NORMALIZE_CASE", ["city"], {"target_case": "title"})])
    summary = executor.execute(plan)

    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    cities = sorted(r["city"] for r in result_df.collect())
    assert cities == ["Delhi", "Delhi", "Mumbai"]
    assert summary.operation_results[0].rows_affected == 2  # "delhi" and "DELHI" both changed


def test_normalize_case_invalid_target_case_fails(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["delhi"]])
    plan = _plan("run3", path, [_op("NORMALIZE_CASE", ["city"], {"target_case": "sarcastic"})])
    summary = executor.execute(plan)

    assert not summary.fully_successful
    assert summary.operations_failed == 1
    assert "target_case" in summary.operation_results[0].error


# ---------------------------------------------------------------------------
# STANDARDIZE_CATEGORY
# ---------------------------------------------------------------------------

def test_standardize_category_operation(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["Delhi"], ["delhi"], ["DELHI"], ["Mumbai"]])
    plan = _plan("run4", path, [
        _op("STANDARDIZE_CATEGORY", ["city"], {"mapping": {"Delhi": "Delhi", "delhi": "Delhi", "DELHI": "Delhi"}}),
    ])
    summary = executor.execute(plan)

    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    cities = sorted(r["city"] for r in result_df.collect())
    assert cities == ["Delhi", "Delhi", "Delhi", "Mumbai"]
    assert summary.operation_results[0].rows_affected == 2  # "delhi" and "DELHI" changed, "Delhi" was already canonical


def test_standardize_category_missing_mapping_fails(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["Delhi"]])
    plan = _plan("run5", path, [_op("STANDARDIZE_CATEGORY", ["city"], {})])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "mapping" in summary.operation_results[0].error


# ---------------------------------------------------------------------------
# STANDARDIZE_DATE_FORMAT
# ---------------------------------------------------------------------------

def test_standardize_date_format_operation(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["join_date"], [["2023-01-05"], ["05/01/2023"], ["2023-03-15"]])
    plan = _plan("run6", path, [_op("STANDARDIZE_DATE_FORMAT", ["join_date"], {"target_format": "DD/MM/YYYY"})])
    summary = executor.execute(plan)

    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    dates = sorted(r["join_date"] for r in result_df.collect())
    # All three should now read as DD/MM/YYYY: Jan 5, Jan 5, Mar 15
    assert dates == ["05/01/2023", "05/01/2023", "15/03/2023"]


def test_standardize_date_format_invalid_target_format_fails(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["join_date"], [["2023-01-05"]])
    plan = _plan("run7", path, [_op("STANDARDIZE_DATE_FORMAT", ["join_date"], {"target_format": "not-a-format"})])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "target_format" in summary.operation_results[0].error


# ---------------------------------------------------------------------------
# FILL_MISSING
# ---------------------------------------------------------------------------

def test_fill_missing_mean_strategy(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["salary"], [["100"], ["200"], [""]])
    plan = _plan("run8", path, [_op("FILL_MISSING", ["salary"], {"strategy": "mean"})])
    summary = executor.execute(plan)

    assert summary.fully_successful
    assert summary.operation_results[0].rows_affected == 1
    result_df = load_csv(spark, summary.output_dataset_path)
    salaries = sorted(r["salary"] for r in result_df.collect())
    assert salaries == [100, 150, 200]  # mean of 100, 200


def test_fill_missing_constant_strategy(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["Delhi"], [""], ["Mumbai"]])
    plan = _plan("run9", path, [_op("FILL_MISSING", ["city"], {"strategy": "constant", "fill_value": "Unknown"})])
    summary = executor.execute(plan)

    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    cities = sorted(r["city"] for r in result_df.collect())
    assert cities == ["Delhi", "Mumbai", "Unknown"]


def test_fill_missing_mode_strategy(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["dept"], [["Eng"], ["Eng"], ["Sales"], [""]])
    plan = _plan("run10", path, [_op("FILL_MISSING", ["dept"], {"strategy": "mode"})])
    summary = executor.execute(plan)
    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    depts = [r["dept"] for r in result_df.collect()]
    assert depts.count("Eng") == 3  # original 2 + filled


def test_fill_missing_mean_on_non_numeric_column_fails(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["Delhi"], [""]])
    plan = _plan("run11", path, [_op("FILL_MISSING", ["city"], {"strategy": "mean"})])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "numeric" in summary.operation_results[0].error


def test_fill_missing_constant_without_fill_value_fails(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["Delhi"], [""]])
    plan = _plan("run12", path, [_op("FILL_MISSING", ["city"], {"strategy": "constant"})])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "fill_value" in summary.operation_results[0].error


def test_fill_missing_on_column_with_no_nulls_is_a_no_op(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["salary"], [["100"], ["200"]])
    plan = _plan("run13", path, [_op("FILL_MISSING", ["salary"], {"strategy": "mean"})])
    summary = executor.execute(plan)
    assert summary.fully_successful
    assert summary.operation_results[0].rows_affected == 0


# ---------------------------------------------------------------------------
# CONVERT_DTYPE
# ---------------------------------------------------------------------------

def test_convert_dtype_string_to_integer(executor, spark, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["age"], [["25"], ["30"]])
    # Force string dtype by adding a header-only trick isn't needed here;
    # PySpark's inferSchema will likely already read these as int. Use a
    # column with a mixed non-numeric marker instead to force string dtype.
    path = _write_csv(tmp_path / "d2.csv", ["code"], [["001"], ["002"]])
    plan = _plan("run14", path, [_op("CONVERT_DTYPE", ["code"], {"target_type": "integer"})])
    summary = executor.execute(plan)

    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    assert dict(result_df.dtypes)["code"] in ("int", "bigint")
    codes = sorted(r["code"] for r in result_df.collect())
    assert codes == [1, 2]


def test_convert_dtype_invalid_target_type_fails(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["age"], [["25"]])
    plan = _plan("run15", path, [_op("CONVERT_DTYPE", ["age"], {"target_type": "not-a-type"})])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "target_type" in summary.operation_results[0].error


# ---------------------------------------------------------------------------
# REMOVE_DUPLICATES
# ---------------------------------------------------------------------------

def test_remove_duplicates_operation(executor, spark, tmp_path):
    path = _write_csv(
        tmp_path / "d.csv", ["name", "city"],
        [["Alice", "Delhi"], ["Alice", "Delhi"], ["Bob", "Mumbai"]],
    )
    plan = _plan("run16", path, [_op("REMOVE_DUPLICATES", ["name", "city"])])
    summary = executor.execute(plan)

    assert summary.fully_successful
    assert summary.operation_results[0].rows_affected == 1  # one duplicate row removed
    result_df = load_csv(spark, summary.output_dataset_path)
    assert result_df.count() == 2


def test_remove_duplicates_with_no_criteria_columns_fails(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["name"], [["Alice"]])
    plan = _plan("run17", path, [_op("REMOVE_DUPLICATES", [])])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "criteria column" in summary.operation_results[0].error


# ---------------------------------------------------------------------------
# FLAG_OUTLIER must never reach execution
# ---------------------------------------------------------------------------

def test_flag_outlier_reaching_the_executor_fails_cleanly(executor, tmp_path):
    """
    FLAG_OUTLIER should never legitimately appear in a plan (the
    orchestrator's eligibility filter excludes it), but if it somehow
    does -- e.g. a hand-constructed, never-validated plan, exactly as
    built here -- the executor must fail that operation cleanly rather
    than silently execute it or crash unexpectedly.
    """
    path = _write_csv(tmp_path / "d.csv", ["salary"], [["100"], ["999999"]])
    plan = _plan("run18", path, [_op("FLAG_OUTLIER", ["salary"])])
    summary = executor.execute(plan)

    assert not summary.fully_successful
    assert summary.operations_failed == 1
    assert summary.operation_results[0].status == OperationExecutionStatus.FAILED
    assert "FINDING_ONLY" in summary.operation_results[0].error or "no executor implementation" in summary.operation_results[0].error
    assert summary.output_dataset_path is None  # nothing was ever written


# ---------------------------------------------------------------------------
# Unsupported / forbidden operations reaching the executor
# ---------------------------------------------------------------------------

def test_unsupported_operation_fails_cleanly(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["salary"], [["100"]])
    plan = _plan("run19", path, [_op("DELETE_COLUMN", ["salary"])])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert summary.output_dataset_path is None


def test_completely_unknown_operation_name_fails_cleanly(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["salary"], [["100"]])
    plan = _plan("run20", path, [_op("TOTALLY_MADE_UP", ["salary"])])
    summary = executor.execute(plan)
    assert not summary.fully_successful


# ---------------------------------------------------------------------------
# Nonexistent columns
# ---------------------------------------------------------------------------

def test_nonexistent_column_fails_cleanly(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["salary"], [["100"]])
    plan = _plan("run21", path, [_op("TRIM_WHITESPACE", ["column_that_does_not_exist"])])
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "does not exist" in summary.operation_results[0].error


# ---------------------------------------------------------------------------
# Arbitrary code / expression injection attempts
# ---------------------------------------------------------------------------

def test_code_injection_in_target_case_parameter_is_rejected_not_executed(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [["delhi"]])
    plan = _plan("run22", path, [
        _op("NORMALIZE_CASE", ["city"], {"target_case": "__import__('os').system('id')"}),
    ])
    summary = executor.execute(plan)
    assert not summary.fully_successful  # rejected as an invalid value, never evaluated as code


def test_code_injection_as_mapping_value_is_treated_as_literal_string(executor, spark, tmp_path):
    """
    A malicious-looking string used as a STANDARDIZE_CATEGORY mapping
    value must be written into the dataset as a literal string value
    (via F.lit), never interpreted, evaluated, or executed.
    """
    path = _write_csv(tmp_path / "d.csv", ["city"], [["Delhi"]])
    payload = "'; DROP TABLE users; --"
    plan = _plan("run23", path, [
        _op("STANDARDIZE_CATEGORY", ["city"], {"mapping": {"Delhi": payload}}),
    ])
    summary = executor.execute(plan)
    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    values = [r["city"] for r in result_df.collect()]
    assert values == [payload]  # stored as a literal string, nothing executed


def test_code_injection_as_unexpected_parameter_key_is_ignored(executor, spark, tmp_path):
    """
    TRIM_WHITESPACE declares zero parameters. An injected 'exec' key
    must simply be ignored (never read, never executed) -- the handler
    only ever looks at target_columns.
    """
    path = _write_csv(tmp_path / "d.csv", ["city"], [[" Delhi "]])
    plan = _plan("run24", path, [
        _op("TRIM_WHITESPACE", ["city"], {"exec": "import os; os.system('rm -rf /')"}),
    ])
    summary = executor.execute(plan)
    assert summary.fully_successful
    result_df = load_csv(spark, summary.output_dataset_path)
    assert result_df.collect()[0]["city"] == "Delhi"


def test_mapping_with_non_string_values_is_rejected(executor, tmp_path):
    """
    A structured/non-string mapping value (which could otherwise be a
    vector for smuggling non-literal content) is rejected outright.
    """
    path = _write_csv(tmp_path / "d.csv", ["city"], [["Delhi"]])
    plan = ExecutionPlan(
        run_id="run25", dataset_path=path,
        operations=(
            PlannedOperation(
                recommendation_id="r1", operation="STANDARDIZE_CATEGORY",
                target_columns=("city",),
                parameters={"mapping": {"Delhi": {"nested": "object"}}},
            ),
        ),
    )
    summary = executor.execute(plan)
    assert not summary.fully_successful
    assert "plain strings" in summary.operation_results[0].error


# ---------------------------------------------------------------------------
# Multiple operations in one plan (+ locked precedence order)
# ---------------------------------------------------------------------------

def test_multiple_operations_in_one_plan_all_succeed(executor, spark, tmp_path):
    path = _write_csv(
        tmp_path / "d.csv", ["city", "salary"],
        [[" Delhi ", "100"], ["delhi", "200"], ["Mumbai", ""]],
    )
    plan = _plan("run26", path, [
        _op("TRIM_WHITESPACE", ["city"]),
        _op("STANDARDIZE_CATEGORY", ["city"], {"mapping": {"Delhi": "Delhi", "delhi": "Delhi"}}),
        _op("FILL_MISSING", ["salary"], {"strategy": "mean"}),
    ])
    summary = executor.execute(plan)

    assert summary.fully_successful
    assert summary.operations_attempted == 3
    assert summary.operations_succeeded == 3
    result_df = load_csv(spark, summary.output_dataset_path)
    rows = sorted((r["city"], r["salary"]) for r in result_df.collect())
    assert rows == [("Delhi", 100), ("Delhi", 200), ("Mumbai", 150)]  # Mumbai filled with mean(100,200)


def test_executor_runs_operations_in_exact_plan_order_not_precedence(executor, spark, tmp_path):
    """
    The executor must NEVER silently reorder operations. This plan
    deliberately lists FILL_MISSING (requires a numeric column) BEFORE
    CONVERT_DTYPE -- an order that a hidden 'type conversion first'
    precedence would have fixed up automatically in the old design.
    Since the executor performs zero reordering, this must FAIL
    exactly as given (mean-fill attempted on a still-string column),
    proving the plan's literal order is what actually runs.

    The 'n/a' value forces Spark's CSV schema inference to read
    'score' as a string column (rather than being auto-inferred as
    numeric), while the empty cell still produces a real null --
    giving FILL_MISSING something to act on.
    """
    path = _write_csv(tmp_path / "d.csv", ["score"], [["10"], ["20"], [""], ["n/a"]])
    plan = _plan("run27", path, [
        _op("FILL_MISSING", ["score"], {"strategy": "mean"}, rec_id="rec-fill"),
        _op("CONVERT_DTYPE", ["score"], {"target_type": "integer"}, rec_id="rec-convert"),
    ])
    summary = executor.execute(plan)

    assert not summary.fully_successful
    fill_result = next(r for r in summary.operation_results if r.recommendation_id == "rec-fill")
    convert_result = next(r for r in summary.operation_results if r.recommendation_id == "rec-convert")
    assert fill_result.status == OperationExecutionStatus.FAILED
    assert "numeric" in fill_result.error
    assert convert_result.status == OperationExecutionStatus.SKIPPED  # never reached -- plan order preserved


def test_executor_runs_operations_in_exact_plan_order_when_that_order_is_valid(executor, spark, tmp_path):
    """
    The inverse of the above: list CONVERT_DTYPE before FILL_MISSING
    (the order that happens to work), and confirm the executor simply
    follows that given order -- it isn't "coincidentally correct" due
    to any internal reordering, it's just doing what the plan says.
    """
    path = _write_csv(tmp_path / "d.csv", ["score"], [["10"], ["20"], [""], ["n/a"]])
    plan = _plan("run27b", path, [
        _op("CONVERT_DTYPE", ["score"], {"target_type": "integer"}, rec_id="rec-convert"),
        _op("FILL_MISSING", ["score"], {"strategy": "mean"}, rec_id="rec-fill"),
    ])
    summary = executor.execute(plan)

    assert summary.fully_successful, summary.operation_results
    assert [r.recommendation_id for r in summary.operation_results] == ["rec-convert", "rec-fill"]
    result_df = load_csv(spark, summary.output_dataset_path)
    scores = sorted(r["score"] for r in result_df.collect())
    # "n/a" casts to null, empty cell is already null -> both filled with mean(10, 20) = 15
    assert scores == [10, 15, 15, 20]


def test_execution_order_matches_plan_order_and_affects_result(executor, spark, tmp_path):
    """
    Concrete proof that plan order (not any hidden rule) determines
    the outcome: with city='DELHI' and a mapping keyed on lowercase
    'delhi', running NORMALIZE_CASE(lower) before STANDARDIZE_CATEGORY
    produces a different final value than running them in the reverse
    order (case-sensitive mapping match only succeeds after lowering).
    """
    path_a = _write_csv(tmp_path / "a.csv", ["city"], [["DELHI"]])
    plan_a = _plan("orderA", path_a, [
        _op("NORMALIZE_CASE", ["city"], {"target_case": "lower"}),
        _op("STANDARDIZE_CATEGORY", ["city"], {"mapping": {"delhi": "Delhi"}}),
    ])
    summary_a = executor.execute(plan_a)
    assert summary_a.fully_successful
    result_a = load_csv(spark, summary_a.output_dataset_path).collect()[0]["city"]

    path_b = _write_csv(tmp_path / "b.csv", ["city"], [["DELHI"]])
    plan_b = _plan("orderB", path_b, [
        _op("STANDARDIZE_CATEGORY", ["city"], {"mapping": {"delhi": "Delhi"}}),
        _op("NORMALIZE_CASE", ["city"], {"target_case": "lower"}),
    ])
    summary_b = executor.execute(plan_b)
    assert summary_b.fully_successful
    result_b = load_csv(spark, summary_b.output_dataset_path).collect()[0]["city"]

    assert result_a == "Delhi"   # normalize-then-standardize: "DELHI"->"delhi"->matched->"Delhi"
    assert result_b == "delhi"   # standardize-then-normalize: "DELHI" never matches "delhi" key, then lowered
    assert result_a != result_b  # proves the executor actually respects the given order


def test_operation_results_are_in_plan_order(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [[" Delhi "]])
    plan = _plan("run27c", path, [
        _op("TRIM_WHITESPACE", ["city"], rec_id="op-1"),
        _op("NORMALIZE_CASE", ["city"], {"target_case": "upper"}, rec_id="op-2"),
        _op("NORMALIZE_CASE", ["city"], {"target_case": "lower"}, rec_id="op-3"),
    ])
    summary = executor.execute(plan)
    assert [r.recommendation_id for r in summary.operation_results] == ["op-1", "op-2", "op-3"]


# ---------------------------------------------------------------------------
# Deterministic repeated execution
# ---------------------------------------------------------------------------

def test_repeated_execution_of_same_plan_is_deterministic(executor, spark, tmp_path):
    path = _write_csv(
        tmp_path / "d.csv", ["city", "salary"],
        [[" Delhi ", "100"], ["delhi", ""], ["Mumbai", "300"]],
    )
    plan = _plan("run28", path, [
        _op("TRIM_WHITESPACE", ["city"]),
        _op("STANDARDIZE_CATEGORY", ["city"], {"mapping": {"Delhi": "Delhi", "delhi": "Delhi"}}),
        _op("FILL_MISSING", ["salary"], {"strategy": "mean"}),
    ])

    summary_a = executor.execute(plan)
    summary_b = executor.execute(plan)

    assert summary_a.fully_successful and summary_b.fully_successful
    df_a = load_csv(spark, summary_a.output_dataset_path)
    df_b = load_csv(spark, summary_b.output_dataset_path)
    rows_a = sorted((r["city"], r["salary"]) for r in df_a.collect())
    rows_b = sorted((r["city"], r["salary"]) for r in df_b.collect())
    assert rows_a == rows_b
    assert [r.rows_affected for r in summary_a.operation_results] == [r.rows_affected for r in summary_b.operation_results]


# ---------------------------------------------------------------------------
# Operation failure: fail-fast, no silent continuation
# ---------------------------------------------------------------------------

def test_operation_failure_stops_subsequent_operations(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city", "salary"], [["Delhi", "100"], ["Mumbai", "200"]])
    plan = _plan("run29", path, [
        _op("NORMALIZE_CASE", ["city"], {"target_case": "invalid-case"}, rec_id="rec-fail"),
        _op("TRIM_WHITESPACE", ["city"], rec_id="rec-never-runs"),
    ])
    summary = executor.execute(plan)

    assert not summary.fully_successful
    assert summary.operations_failed == 1
    assert summary.operations_skipped == 1

    failed = next(r for r in summary.operation_results if r.recommendation_id == "rec-fail")
    skipped = next(r for r in summary.operation_results if r.recommendation_id == "rec-never-runs")
    assert failed.status == OperationExecutionStatus.FAILED
    assert skipped.status == OperationExecutionStatus.SKIPPED
    assert summary.output_dataset_path is None  # no partial output written


# ---------------------------------------------------------------------------
# Traceability to recommendation IDs
# ---------------------------------------------------------------------------

def test_every_operation_result_is_traceable_to_its_recommendation_id(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [[" Delhi "], ["Mumbai"]])
    plan = _plan("run30", path, [
        _op("TRIM_WHITESPACE", ["city"], rec_id="rec-abc-123"),
    ])
    summary = executor.execute(plan)
    assert summary.operation_results[0].recommendation_id == "rec-abc-123"


# ---------------------------------------------------------------------------
# Execution summary correctness
# ---------------------------------------------------------------------------

def test_execution_summary_counts_are_internally_consistent(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [[" Delhi "], ["Mumbai"]])
    plan = _plan("run31", path, [
        _op("TRIM_WHITESPACE", ["city"]),
        _op("NORMALIZE_CASE", ["city"], {"target_case": "bad"}),  # fails
        _op("STANDARDIZE_CATEGORY", ["city"], {"mapping": {"Delhi": "Delhi"}}),  # skipped
    ])
    summary = executor.execute(plan)

    assert len(summary.operation_results) == 3
    assert summary.operations_attempted == summary.operations_succeeded + summary.operations_failed
    assert summary.operations_attempted + summary.operations_skipped == len(summary.operation_results)
    assert summary.operations_succeeded == 1
    assert summary.operations_failed == 1
    assert summary.operations_skipped == 1
    assert not summary.fully_successful


# ---------------------------------------------------------------------------
# Schema preservation where expected
# ---------------------------------------------------------------------------

def test_schema_before_and_after_are_captured(executor, tmp_path):
    # decimal-looking values: Spark's CSV inference reads these as
    # 'double', giving us a real, deterministic dtype change to assert
    # against when converting to 'integer'.
    path = _write_csv(tmp_path / "d.csv", ["city", "score"], [["Delhi", "1.5"], ["Mumbai", "2.5"]])
    plan = _plan("run32", path, [_op("CONVERT_DTYPE", ["score"], {"target_type": "integer"})])
    summary = executor.execute(plan)

    assert summary.fully_successful
    before_map = {c["name"]: c["dtype"] for c in summary.schema_before}
    after_map = {c["name"]: c["dtype"] for c in summary.schema_after}
    assert before_map["score"] == "double"
    assert after_map["score"] in ("int", "bigint")
    assert before_map["city"] == after_map["city"] == "string"  # unrelated column unaffected


def test_schema_unchanged_when_operation_does_not_alter_types(executor, tmp_path):
    path = _write_csv(tmp_path / "d.csv", ["city"], [[" Delhi "]])
    plan = _plan("run33", path, [_op("TRIM_WHITESPACE", ["city"])])
    summary = executor.execute(plan)
    assert summary.schema_before == summary.schema_after


# ---------------------------------------------------------------------------
# Integration: full sample_hr.csv dataset, multiple real operations
# ---------------------------------------------------------------------------

def test_end_to_end_on_sample_hr_dataset(executor, spark, sample_hr_path):
    plan = _plan("run34", sample_hr_path, [
        _op("STANDARDIZE_CATEGORY", ["city"], {
            "mapping": {"Delhi": "Delhi", "delhi": "Delhi", "DELHI": "Delhi", "Delhi ": "Delhi"}
        }),
        _op("FILL_MISSING", ["salary"], {"strategy": "mean"}),
        _op("REMOVE_DUPLICATES", [
            "full_name", "city", "department", "join_date", "salary", "age", "manager_id",
        ]),
    ])
    summary = executor.execute(plan)

    assert summary.fully_successful, summary.operation_results
    assert summary.operations_succeeded == 3

    result_df = load_csv(spark, summary.output_dataset_path)
    # No more nulls in salary
    assert result_df.filter(result_df.salary.isNull()).count() == 0
    # Duplicate (E001/E008) removed: 20 rows -> 19
    assert result_df.count() == 19

    dup_result = next(r for r in summary.operation_results if r.operation == "REMOVE_DUPLICATES")
    assert dup_result.rows_affected == 1

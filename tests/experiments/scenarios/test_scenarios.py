"""
Tests for controlled corruption scenarios and ground-truth evaluation.

Uses tmp_path for all file I/O — no persistent test artifacts.
"""
import csv
import hashlib
from pathlib import Path

import pytest

from experiments.scenarios.generator import (
    SCENARIO_SPECS,
    corrupt_dataset,
    generate_all_scenarios,
    get_clean_rows,
    write_clean_fixture,
)
from experiments.scenarios.ground_truth import evaluate
from experiments.scenarios.models import (
    CorruptionSpec,
    CorruptionType,
    GroundTruthManifest,
)


@pytest.fixture()
def clean_path(tmp_path):
    p = str(tmp_path / "clean.csv")
    write_clean_fixture(p)
    return p


@pytest.fixture()
def clean_checksum(clean_path):
    return hashlib.sha256(Path(clean_path).read_bytes()).hexdigest()


def _corrupt(clean_path, scenario_id, corruption_types, seed=42, tmp_path=None):
    out = str(tmp_path / f"{scenario_id}.csv") if tmp_path else clean_path.replace("clean", scenario_id)
    spec = CorruptionSpec(scenario_id=scenario_id, corruption_types=corruption_types, seed=seed)
    return corrupt_dataset(clean_path, spec, out), out


# ===================================================================
# CORRUPTION GENERATION — determinism
# ===================================================================

def test_same_clean_same_spec_same_seed_produces_identical_bytes(clean_path, tmp_path):
    spec = CorruptionSpec(scenario_id="det", corruption_types=[CorruptionType.WHITESPACE], seed=42)
    m1 = corrupt_dataset(clean_path, spec, str(tmp_path / "a.csv"))
    m2 = corrupt_dataset(clean_path, spec, str(tmp_path / "b.csv"))

    assert m1.corrupted_dataset_checksum == m2.corrupted_dataset_checksum
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_different_seed_produces_different_output(clean_path, tmp_path):
    spec_a = CorruptionSpec(scenario_id="s", corruption_types=[CorruptionType.WHITESPACE], seed=1)
    spec_b = CorruptionSpec(scenario_id="s", corruption_types=[CorruptionType.WHITESPACE], seed=2)
    m_a = corrupt_dataset(clean_path, spec_a, str(tmp_path / "a.csv"))
    m_b = corrupt_dataset(clean_path, spec_b, str(tmp_path / "b.csv"))
    assert m_a.corrupted_dataset_checksum != m_b.corrupted_dataset_checksum


def test_clean_reference_is_unchanged_after_corruption(clean_path, clean_checksum, tmp_path):
    spec = CorruptionSpec(scenario_id="s", corruption_types=[CorruptionType.WHITESPACE], seed=42)
    corrupt_dataset(clean_path, spec, str(tmp_path / "out.csv"))
    assert hashlib.sha256(Path(clean_path).read_bytes()).hexdigest() == clean_checksum


def test_every_corruption_has_a_ground_truth_record(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "s", [CorruptionType.WHITESPACE, CorruptionType.MISSING_VALUE], tmp_path=tmp_path)
    assert m.total_corruptions > 0
    # every cell corruption has a non-empty corruption_id, row_id, column, clean_value
    for cc in m.cell_corruptions:
        assert cc.corruption_id
        assert cc.row_id
        assert cc.column
        assert cc.clean_value is not None


def test_checksums_match_actual_files(clean_path, clean_checksum, tmp_path):
    m, out = _corrupt(clean_path, "s", [CorruptionType.WHITESPACE], tmp_path=tmp_path)
    assert m.clean_dataset_checksum == clean_checksum
    assert m.corrupted_dataset_checksum == hashlib.sha256(Path(out).read_bytes()).hexdigest()


# ===================================================================
# ISOLATED SCENARIOS — each contains only its designated corruption
# ===================================================================

@pytest.mark.parametrize("spec_name,expected_types", [
    ("S1_whitespace", {CorruptionType.WHITESPACE}),
    ("S2_duplicates", {CorruptionType.EXACT_DUPLICATE}),
    ("S3_categorical", {CorruptionType.CATEGORICAL_VARIANT}),
    ("S4_missing", {CorruptionType.MISSING_VALUE}),
    ("S5_dates", {CorruptionType.MIXED_DATE_FORMAT}),
])
def test_isolated_scenario_contains_only_its_designated_type(clean_path, tmp_path, spec_name, expected_types):
    spec = SCENARIO_SPECS[spec_name]
    m = corrupt_dataset(clean_path, spec, str(tmp_path / f"{spec_name}.csv"))
    cell_types = {cc.corruption_type for cc in m.cell_corruptions}
    dup_types = {dc.corruption_type for dc in m.duplicate_corruptions}
    all_types = cell_types | dup_types
    assert all_types == expected_types


def test_mixed_scenario_contains_all_five_types(clean_path, tmp_path):
    spec = SCENARIO_SPECS["S6_mixed"]
    m = corrupt_dataset(clean_path, spec, str(tmp_path / "S6_mixed.csv"))
    cell_types = {cc.corruption_type for cc in m.cell_corruptions}
    dup_types = {dc.corruption_type for dc in m.duplicate_corruptions}
    all_types = cell_types | dup_types
    assert all_types == {
        CorruptionType.WHITESPACE, CorruptionType.EXACT_DUPLICATE,
        CorruptionType.CATEGORICAL_VARIANT, CorruptionType.MISSING_VALUE,
        CorruptionType.MIXED_DATE_FORMAT,
    }


# ===================================================================
# WHITESPACE
# ===================================================================

def test_whitespace_injected_value_differs_from_clean(clean_path, tmp_path):
    m, out = _corrupt(clean_path, "ws", [CorruptionType.WHITESPACE], tmp_path=tmp_path)
    for cc in m.cell_corruptions:
        assert cc.corrupted_value != cc.clean_value
        assert cc.corrupted_value.strip() == cc.clean_value  # only whitespace was added


def test_whitespace_evaluator_recognizes_exact_restoration(clean_path, tmp_path):
    m, out = _corrupt(clean_path, "ws", [CorruptionType.WHITESPACE], tmp_path=tmp_path)
    gt = evaluate(clean_path, m, clean_path)  # clean = output = perfect repair
    assert gt.correctly_repaired == len(m.cell_corruptions)
    assert gt.unrepaired == 0


def test_whitespace_unrepaired_corrupted_data_shows_zero_correct(clean_path, tmp_path):
    m, out = _corrupt(clean_path, "ws", [CorruptionType.WHITESPACE], tmp_path=tmp_path)
    gt = evaluate(clean_path, m, out)
    assert gt.correctly_repaired == 0
    assert gt.unrepaired == len(m.cell_corruptions)


# ===================================================================
# CATEGORICAL VARIANTS
# ===================================================================

def test_categorical_canonical_clean_value_recorded(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "cat", [CorruptionType.CATEGORICAL_VARIANT], tmp_path=tmp_path)
    canonical = {"Engineering", "Sales", "Marketing"}
    for cc in m.cell_corruptions:
        assert cc.clean_value in canonical


def test_categorical_wrong_canonicalization_is_incorrect(clean_path, tmp_path):
    """Even if issue count decreases, using the wrong canonical form is not correct."""
    m, out = _corrupt(clean_path, "cat", [CorruptionType.CATEGORICAL_VARIANT], tmp_path=tmp_path)

    # Create a "repaired" output that uses a different capitalization
    rows = _read_csv(out)
    for cc in m.cell_corruptions:
        for row in rows:
            if row["_row_id"] == cc.row_id:
                # Set to a consistent-but-wrong form (all lowercase)
                row[cc.column] = cc.clean_value.lower()
    wrong_repair = str(tmp_path / "wrong_cat.csv")
    _write_csv(wrong_repair, rows)

    gt = evaluate(clean_path, m, wrong_repair)
    # If clean_value is "Engineering" and we wrote "engineering", that's NOT correct
    # (unless the clean value happened to already be lowercase, which it isn't)
    assert gt.incorrectly_repaired > 0 or gt.correctly_repaired < len(m.cell_corruptions)


# ===================================================================
# MISSING VALUES
# ===================================================================

def test_missing_original_value_recorded_before_nulling(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "miss", [CorruptionType.MISSING_VALUE], tmp_path=tmp_path)
    for cc in m.cell_corruptions:
        assert cc.clean_value not in (None, "")
        assert cc.corrupted_value == ""


def test_missing_wrong_imputation_is_incorrect(clean_path, tmp_path):
    m, out = _corrupt(clean_path, "miss", [CorruptionType.MISSING_VALUE], tmp_path=tmp_path)

    rows = _read_csv(out)
    for cc in m.cell_corruptions:
        for row in rows:
            if row["_row_id"] == cc.row_id:
                row[cc.column] = "99999"  # wrong fill value
    wrong_fill = str(tmp_path / "wrong_fill.csv")
    _write_csv(wrong_fill, rows)

    gt = evaluate(clean_path, m, wrong_fill)
    assert gt.incorrectly_repaired == len(m.cell_corruptions)
    assert gt.correctly_repaired == 0


def test_missing_exact_restoration_is_correct(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "miss", [CorruptionType.MISSING_VALUE], tmp_path=tmp_path)
    gt = evaluate(clean_path, m, clean_path)
    assert gt.correctly_repaired == len(m.cell_corruptions)


# ===================================================================
# MIXED DATE FORMATS
# ===================================================================

def test_dates_original_canonical_representation_recorded(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "dates", [CorruptionType.MIXED_DATE_FORMAT], tmp_path=tmp_path)
    for cc in m.cell_corruptions:
        assert "-" in str(cc.clean_value)  # YYYY-MM-DD has dashes
        assert "/" in str(cc.corrupted_value)  # DD/MM/YYYY has slashes


def test_dates_wrong_consistent_format_is_not_correct(clean_path, tmp_path):
    """All dates becoming MM/DD/YYYY is consistent but NOT correct if
    the clean reference uses YYYY-MM-DD."""
    m, out = _corrupt(clean_path, "dates", [CorruptionType.MIXED_DATE_FORMAT], tmp_path=tmp_path)

    rows = _read_csv(out)
    for cc in m.cell_corruptions:
        for row in rows:
            if row["_row_id"] == cc.row_id:
                # Rewrite to a consistent but WRONG format (MM/DD/YYYY instead of YYYY-MM-DD)
                parts = cc.clean_value.split("-")
                row[cc.column] = f"{parts[1]}/{parts[2]}/{parts[0]}"
    wrong_format = str(tmp_path / "wrong_date.csv")
    _write_csv(wrong_format, rows)

    gt = evaluate(clean_path, m, wrong_format)
    assert gt.correctly_repaired == 0
    assert gt.incorrectly_repaired == len(m.cell_corruptions)


# ===================================================================
# DUPLICATES
# ===================================================================

def test_duplicate_injection_recorded(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "dup", [CorruptionType.EXACT_DUPLICATE], tmp_path=tmp_path)
    assert len(m.duplicate_corruptions) > 0
    for dc in m.duplicate_corruptions:
        assert dc.source_row_id
        assert dc.duplicate_row_id
        assert dc.source_row_id != dc.duplicate_row_id


def test_removing_injected_duplicate_is_correct(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "dup", [CorruptionType.EXACT_DUPLICATE], tmp_path=tmp_path)
    # Clean data has no duplicates -> using clean as output = all injected dups removed
    gt = evaluate(clean_path, m, clean_path)
    assert gt.injected_duplicates_removed == len(m.duplicate_corruptions)
    assert gt.original_rows_lost == 0


def test_deleting_original_instead_of_duplicate_is_detected(clean_path, tmp_path):
    m, out = _corrupt(clean_path, "dup", [CorruptionType.EXACT_DUPLICATE], tmp_path=tmp_path)

    # Remove the original source row but keep the duplicate
    rows = _read_csv(out)
    source_ids = {dc.source_row_id for dc in m.duplicate_corruptions}
    rows_without_originals = [r for r in rows if r["_row_id"] not in source_ids]
    bad_output = str(tmp_path / "bad_dedup.csv")
    _write_csv(bad_output, rows_without_originals)

    gt = evaluate(clean_path, m, bad_output)
    assert gt.original_rows_lost == len(m.duplicate_corruptions)


def test_unrelated_row_deletion_is_not_rewarded(clean_path, tmp_path):
    """Deleting a row that wasn't a duplicate should not improve any metric."""
    m, out = _corrupt(clean_path, "dup", [CorruptionType.EXACT_DUPLICATE], tmp_path=tmp_path)

    rows = _read_csv(out)
    # Remove a row that is neither a source nor a duplicate
    dup_ids = {dc.source_row_id for dc in m.duplicate_corruptions} | {dc.duplicate_row_id for dc in m.duplicate_corruptions}
    unrelated = [r for r in rows if r["_row_id"] not in dup_ids][:1]
    if unrelated:
        rows_modified = [r for r in rows if r["_row_id"] != unrelated[0]["_row_id"]]
        partial_output = str(tmp_path / "partial_dedup.csv")
        _write_csv(partial_output, rows_modified)

        gt = evaluate(clean_path, m, partial_output)
        assert gt.injected_duplicates_removed == 0  # didn't remove any duplicates
        assert gt.originally_clean_values_modified == 0  # remaining rows are unchanged
        # But the unrelated row is just gone — not counted as "clean modified"
        # (it's not in the output, so no cell comparison is possible for it)


# ===================================================================
# PRESERVATION — detecting modifications to originally clean data
# ===================================================================

def test_modifying_originally_clean_value_is_detected(clean_path, tmp_path):
    m, out = _corrupt(clean_path, "ws", [CorruptionType.WHITESPACE], tmp_path=tmp_path)

    rows = _read_csv(out)
    # Modify a cell that was NOT corrupted
    for row in rows:
        if row["_row_id"] == "R015":
            row["city"] = "WRONGCITY"
            break
    damaged_output = str(tmp_path / "damaged.csv")
    _write_csv(damaged_output, rows)

    gt = evaluate(clean_path, m, damaged_output)
    assert gt.originally_clean_values_modified >= 1


def test_false_modification_rate_denominator_documented_and_tested(clean_path, tmp_path):
    """
    Denominator = cells in (common_rows × data_columns) that were NOT
    intentionally corrupted. _row_id is excluded from data_columns.
    """
    m, out = _corrupt(clean_path, "ws", [CorruptionType.WHITESPACE], tmp_path=tmp_path)
    gt = evaluate(clean_path, m, out)

    clean_rows = get_clean_rows()
    data_columns = [c for c in clean_rows[0] if c != "_row_id"]
    expected_total_clean = len(clean_rows) * len(data_columns) - len(m.cell_corruptions)

    assert gt.total_originally_clean_cells == expected_total_clean
    assert gt.false_modification_rate == 0.0  # corrupted file itself doesn't modify clean values


def test_perfect_output_has_zero_false_modification_rate(clean_path, tmp_path):
    m, _ = _corrupt(clean_path, "miss", [CorruptionType.MISSING_VALUE], tmp_path=tmp_path)
    gt = evaluate(clean_path, m, clean_path)
    assert gt.false_modification_rate == 0.0
    assert gt.originally_clean_values_modified == 0


# ===================================================================
# STRATEGY AGNOSTICISM
# ===================================================================

def test_evaluator_has_no_strategy_specific_branching():
    import ast
    from pathlib import Path as P
    from experiments.scenarios import ground_truth as gt_module
    import inspect

    source = inspect.getsource(gt_module)
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    converge_imports = {m for m in imported_modules if m.startswith("converge")}
    assert converge_imports == set(), f"ground_truth.py must not import converge: {converge_imports}"
    strategy_imports = {m for m in imported_modules if "strategy" in m.lower() or "strategyname" in m.lower()}
    assert strategy_imports == set(), f"ground_truth.py must not reference strategies: {strategy_imports}"


def test_evaluator_has_no_dependency_on_strategy_type():
    """The evaluate() function signature takes only paths and a manifest
    — no Configuration, no StrategyName, no AIProvider."""
    import inspect
    sig = inspect.signature(evaluate)
    param_names = list(sig.parameters.keys())
    assert param_names == ["clean_path", "manifest", "output_path"]


# ===================================================================
# BOUNDARY
# ===================================================================

def test_converge_still_never_imports_experiments():
    import ast
    from pathlib import Path as P
    import converge
    converge_root = P(converge.__file__).parent
    offending = []
    for py_file in converge_root.rglob("*.py"):
        if "/cli/" in str(py_file):
            continue  # CLI is a composition root, allowed to wire experiments
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "experiments":
                offending.append(str(py_file))
    assert offending == []


def test_scenario_models_have_no_converge_dependency():
    import ast
    from pathlib import Path as P
    from experiments.scenarios import models as models_module
    import inspect
    source_path = P(inspect.getfile(models_module))
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("converge"), f"models.py imports converge: {node.module}"


# ===================================================================
# Helpers
# ===================================================================

def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

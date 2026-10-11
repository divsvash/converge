"""
Strategy-agnostic ground-truth evaluator.

clean reference + corruption manifest + strategy output -> GroundTruthMetrics.

This module has zero knowledge of which strategy produced the output.
There is no branching on SPARK_ONLY/AI_ONLY/CONVERGE anywhere in this
file -- by construction, not just by convention (see
tests/experiments/scenarios/test_ground_truth.py's
test_evaluator_has_no_strategy_specific_branching).

METRIC DEFINITIONS

Cell-level corruptions (WHITESPACE, CATEGORICAL_VARIANT, MISSING_VALUE,
MIXED_DATE_FORMAT):

    correctly_repaired
        The output row (joined on _row_id) has the exact clean CSV field
        value. Whitespace and case are significant.

    incorrectly_repaired
        The output row exists and the column value differs from BOTH
        the corrupted value AND the clean value. The strategy tried
        to fix it but got the wrong answer.

    unrepaired
        The output row exists and the column value still matches the
        corrupted value (or is absent/null while the corrupted value
        was also absent). The strategy did not touch this corruption.

    repair_recall = correctly_repaired / total_corruptions (cell-level)
        0 if total_corruptions is 0.

Duplicate-level corruptions (EXACT_DUPLICATE):

    injected_duplicates_removed
        The _row_id assigned to the injected duplicate is absent
        from the output.

    original_rows_lost
        The _row_id of the SOURCE row (the one that was copied) is
        absent from the output. This detects the pathological case
        where a strategy removes the original instead of (or as well
        as) the duplicate.

Clean-data preservation:

    originally_clean_values_modified
        Number of (row, column) cells that were NOT corrupted by any
        CellCorruption, whose value in the output differs from the
        clean reference. Measures collateral damage: a strategy
        should not change cells that were already correct.

    false_modification_rate = originally_clean_values_modified /
        total_originally_clean_cells. 0 if denominator is 0.
        Denominator = (output_rows * data_columns) - cell_corruptions
        — i.e. all cells the evaluator can compare, minus the ones
        that were intentionally corrupted. _row_id itself is excluded
        from data_columns (it is infrastructure, not data).

repair_precision is NOT computed: "attempted_repairs" would require
knowing which cells the strategy intended to modify, which is not
reliably observable from the output alone (a strategy might modify a
cell to the same value it already had, which is indistinguishable from
not touching it). Pretending we can measure it would produce a number
with an undefined denominator — worse than not reporting it.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from experiments.scenarios.models import (
    CellCorruption,
    CorruptionType,
    DuplicateCorruption,
    GroundTruthManifest,
)


class GroundTruthMetrics(BaseModel):
    """Strategy-agnostic repair-correctness measurement."""
    scenario_id: str

    total_corruptions: int
    correctly_repaired: int
    incorrectly_repaired: int
    unrepaired: int
    repair_recall: float = Field(
        description="correctly_repaired / total_corruptions; 0 if total_corruptions is 0.",
    )

    injected_duplicates_removed: int
    original_rows_lost: int
    all_original_rows_lost: int = 0
    correctly_removed_duplicates: int = 0

    originally_clean_values_modified: int
    false_modification_rate: float = Field(
        description="originally_clean_values_modified / total_originally_clean_cells; "
                    "0 if denominator is 0.",
    )
    total_originally_clean_cells: int


def evaluate(
    clean_path: str,
    manifest: GroundTruthManifest,
    output_path: str,
) -> GroundTruthMetrics:
    clean_rows = _read_csv_by_row_id(clean_path)
    output_rows = _read_csv_by_row_id(output_path)

    data_columns = [c for c in _read_columns(clean_path) if c != "_row_id"]

    # --- Cell-level corruption evaluation ---
    correctly_repaired = 0
    incorrectly_repaired = 0
    unrepaired = 0

    for cc in manifest.cell_corruptions:
        out_row = output_rows.get(cc.row_id)
        if out_row is None:
            unrepaired += 1
            continue

        out_val = _normalize(out_row.get(cc.column))
        clean_val = _normalize(cc.clean_value)
        corrupted_val = _normalize(cc.corrupted_value)

        if cc.corruption_type == CorruptionType.WHITESPACE:
            # Whitespace corruption IS leading/trailing spaces, so
            # _normalize (which strips whitespace) would erase the
            # very difference we're testing. Compare raw values
            # instead, with only minimal newline stripping.
            raw_out = _strip_newlines_only(out_row.get(cc.column))
            raw_clean = _strip_newlines_only(cc.clean_value)
            raw_corrupted = _strip_newlines_only(cc.corrupted_value)
            if raw_out == raw_clean:
                correctly_repaired += 1
            elif raw_out == raw_corrupted:
                unrepaired += 1
            else:
                incorrectly_repaired += 1
        else:
            if out_val == clean_val:
                correctly_repaired += 1
            elif out_val == corrupted_val:
                unrepaired += 1
            else:
                incorrectly_repaired += 1

    # --- Duplicate-level evaluation ---
    injected_duplicates_removed = 0
    original_rows_lost = 0
    correctly_removed_duplicates = 0

    for dc in manifest.duplicate_corruptions:
        if dc.duplicate_row_id not in output_rows:
            injected_duplicates_removed += 1
        if dc.source_row_id not in output_rows:
            original_rows_lost += 1
        elif dc.duplicate_row_id not in output_rows:
            correctly_removed_duplicates += 1

    # --- Clean-data preservation ---
    corrupted_cells: set[tuple[str, str]] = set()
    for cc in manifest.cell_corruptions:
        corrupted_cells.add((cc.row_id, cc.column))

    # Only check rows that exist in both the clean reference and the output.
    # Injected duplicate rows are excluded (they don't exist in clean_rows,
    # so they won't appear in the intersection).
    all_original_rows_lost = len(set(clean_rows) - set(output_rows))
    common_row_ids = set(clean_rows.keys()) & set(output_rows.keys())

    originally_clean_modified = 0
    total_clean_cells = 0

    for row_id in common_row_ids:
        for col in data_columns:
            if (row_id, col) in corrupted_cells:
                continue
            total_clean_cells += 1
            clean_val = _normalize(clean_rows[row_id].get(col))
            out_val = _normalize(output_rows[row_id].get(col))
            if out_val != clean_val:
                originally_clean_modified += 1

    total_corruptions = len(manifest.cell_corruptions) + len(manifest.duplicate_corruptions)

    return GroundTruthMetrics(
        scenario_id=manifest.scenario_id,
        total_corruptions=total_corruptions,
        correctly_repaired=correctly_repaired,
        incorrectly_repaired=incorrectly_repaired,
        unrepaired=unrepaired,
        repair_recall=(correctly_repaired / total_corruptions) if total_corruptions else 0.0,
        injected_duplicates_removed=injected_duplicates_removed,
        original_rows_lost=original_rows_lost,
        all_original_rows_lost=all_original_rows_lost,
        correctly_removed_duplicates=correctly_removed_duplicates,
        originally_clean_values_modified=originally_clean_modified,
        false_modification_rate=(
            originally_clean_modified / total_clean_cells if total_clean_cells else 0.0
        ),
        total_originally_clean_cells=total_clean_cells,
    )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv_by_row_id(path: str) -> dict[str, dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["_row_id"]: row for row in reader}


def _read_columns(path: str) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def _strip_newlines_only(value) -> Optional[str]:
    """Strip only newline/carriage-return, preserving spaces (which ARE
    the corruption for whitespace scenarios)."""
    if value is None:
        return None
    return str(value).strip("\n\r")


def _normalize(value) -> Optional[str]:
    """Compare exact CSV field values; preserve spaces and case.

    CSV parsing already removes record delimiters. Missing CSV fields are
    represented as empty strings, equivalent to the manifest's None.
    """
    if value is None or value == "":
        return None
    return str(value)

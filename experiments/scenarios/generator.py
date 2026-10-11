"""
Deterministic corruption generator.

clean_dataset() -> a small, fully clean CSV with stable _row_id.
corrupt_dataset(clean_path, spec) -> corrupted CSV + GroundTruthManifest.

CLEAN REFERENCE DESIGN
The fixture has 15 rows and 7 columns, chosen so that every
corruption type has at least one suitable target column and enough
rows that injecting 2-3 corruptions of each type is a meaningful
fraction without overwhelming the profiler's statistics:

    _row_id      stable experiment-only identifier (R001..R015).
                 Triggers the profiler's id-like detection (name
                 contains "id", 100% distinct) — excluded from
                 duplicate criteria and categorical/numeric profiling.
    name         string — target for whitespace corruption.
    department   string, 3 distinct values — target for categorical
                 variant corruption.
    city         string — secondary string column.
    hire_date    date-like string, uniform YYYY-MM-DD — target for
                 date-format corruption.
    salary       integer — target for missing-value corruption.
    rating       float — secondary numeric column.

Every value is intentionally clean: no whitespace issues, no
categorical variants, no mixed date formats, no nulls, no duplicates.
sample_hr.csv is NOT used as ground truth because the audit showed it
already contains every corruption type we want to inject.

CORRUPTION ALGORITHMS (each is deterministic given the spec's seed):

WHITESPACE: prepend/append spaces to selected string cells.
EXACT_DUPLICATE: copy a selected row, assign a new _row_id.
CATEGORICAL_VARIANT: replace a categorical value with a case variant.
MISSING_VALUE: set a selected numeric cell to None/empty.
MIXED_DATE_FORMAT: rewrite a YYYY-MM-DD date to DD/MM/YYYY.
"""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from experiments.scenarios.models import (
    CellCorruption,
    CorruptionSpec,
    CorruptionType,
    DuplicateCorruption,
    GroundTruthManifest,
)

# ---------------------------------------------------------------------------
# Clean reference data
# ---------------------------------------------------------------------------

_CLEAN_ROWS: list[dict[str, str]] = [
    {"_row_id": "R001", "name": "Alice Martin",    "department": "Engineering", "city": "London",    "hire_date": "2022-01-15", "salary": "72000", "rating": "4.2"},
    {"_row_id": "R002", "name": "Bob Chen",         "department": "Engineering", "city": "Berlin",    "hire_date": "2022-03-20", "salary": "68000", "rating": "3.8"},
    {"_row_id": "R003", "name": "Carol Diaz",       "department": "Sales",       "city": "Madrid",    "hire_date": "2022-05-10", "salary": "55000", "rating": "4.5"},
    {"_row_id": "R004", "name": "David Kim",        "department": "Sales",       "city": "Seoul",     "hire_date": "2022-07-01", "salary": "58000", "rating": "3.9"},
    {"_row_id": "R005", "name": "Elena Rossi",      "department": "Marketing",   "city": "Rome",      "hire_date": "2022-09-12", "salary": "61000", "rating": "4.1"},
    {"_row_id": "R006", "name": "Frank Mueller",    "department": "Engineering", "city": "Berlin",    "hire_date": "2022-11-05", "salary": "75000", "rating": "4.7"},
    {"_row_id": "R007", "name": "Grace Tanaka",     "department": "Sales",       "city": "Tokyo",     "hire_date": "2023-01-18", "salary": "53000", "rating": "3.6"},
    {"_row_id": "R008", "name": "Henry Okonkwo",    "department": "Marketing",   "city": "Lagos",     "hire_date": "2023-03-22", "salary": "49000", "rating": "4.0"},
    {"_row_id": "R009", "name": "Ingrid Larsson",   "department": "Engineering", "city": "Stockholm", "hire_date": "2023-05-30", "salary": "71000", "rating": "4.4"},
    {"_row_id": "R010", "name": "James Wilson",     "department": "Sales",       "city": "London",    "hire_date": "2023-07-14", "salary": "56000", "rating": "3.7"},
    {"_row_id": "R011", "name": "Keiko Sato",       "department": "Marketing",   "city": "Tokyo",     "hire_date": "2023-09-01", "salary": "52000", "rating": "4.3"},
    {"_row_id": "R012", "name": "Liam Murphy",      "department": "Engineering", "city": "Dublin",    "hire_date": "2023-11-10", "salary": "69000", "rating": "3.5"},
    {"_row_id": "R013", "name": "Maria Santos",     "department": "Sales",       "city": "Madrid",    "hire_date": "2024-01-08", "salary": "54000", "rating": "4.6"},
    {"_row_id": "R014", "name": "Nikolai Petrov",   "department": "Marketing",   "city": "Berlin",    "hire_date": "2024-03-15", "salary": "60000", "rating": "4.0"},
    {"_row_id": "R015", "name": "Olivia Brown",     "department": "Engineering", "city": "London",    "hire_date": "2024-05-20", "salary": "73000", "rating": "4.8"},
]

_COLUMNS = list(_CLEAN_ROWS[0].keys())

# Canonical values for categorical variants — the generator will
# replace with variants of these, and ground truth expects restoration
# to exactly these.
_CANONICAL_DEPARTMENTS = {"Engineering", "Sales", "Marketing"}

# Canonical date format — the only "correct" format for ground truth.
_CANONICAL_DATE_FORMAT = "YYYY-MM-DD"


def write_clean_fixture(output_path: str) -> str:
    """Write the clean reference CSV and return its SHA-256 checksum."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(_CLEAN_ROWS)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_clean_rows() -> list[dict[str, str]]:
    """Return a copy of the clean reference rows (for tests that need
    the data without writing a file)."""
    return [dict(row) for row in _CLEAN_ROWS]


# ---------------------------------------------------------------------------
# Corruption generator
# ---------------------------------------------------------------------------

def corrupt_dataset(
    clean_path: str, spec: CorruptionSpec, output_path: str,
) -> GroundTruthManifest:
    """
    Read the clean CSV at `clean_path`, apply the corruptions declared
    in `spec`, write the corrupted CSV to `output_path`, and return
    the manifest recording exactly what was done.

    Deterministic: same clean data + same spec (including seed) =
    identical corrupted bytes.
    """
    clean_checksum = hashlib.sha256(Path(clean_path).read_bytes()).hexdigest()

    with open(clean_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    rng = random.Random(spec.seed)
    cell_corruptions: list[CellCorruption] = []
    dup_corruptions: list[DuplicateCorruption] = []
    corruption_counter = 0

    for ct in sorted(spec.corruption_types, key=lambda t: t.value):
        if ct == CorruptionType.WHITESPACE:
            corruption_counter, rows = _inject_whitespace(
                rows, rng, corruption_counter, cell_corruptions, spec.scenario_id,
            )
        elif ct == CorruptionType.EXACT_DUPLICATE:
            corruption_counter, rows = _inject_duplicates(
                rows, rng, corruption_counter, dup_corruptions, spec.scenario_id,
            )
        elif ct == CorruptionType.CATEGORICAL_VARIANT:
            corruption_counter, rows = _inject_categorical_variants(
                rows, rng, corruption_counter, cell_corruptions, spec.scenario_id,
            )
        elif ct == CorruptionType.MISSING_VALUE:
            corruption_counter, rows = _inject_missing_values(
                rows, rng, corruption_counter, cell_corruptions, spec.scenario_id,
            )
        elif ct == CorruptionType.MIXED_DATE_FORMAT:
            corruption_counter, rows = _inject_mixed_dates(
                rows, rng, corruption_counter, cell_corruptions, spec.scenario_id,
            )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    corrupted_checksum = hashlib.sha256(out.read_bytes()).hexdigest()

    return GroundTruthManifest(
        scenario_id=spec.scenario_id,
        clean_dataset_checksum=clean_checksum,
        corrupted_dataset_checksum=corrupted_checksum,
        corruption_seed=spec.seed,
        cell_corruptions=cell_corruptions,
        duplicate_corruptions=dup_corruptions,
    )


# ---------------------------------------------------------------------------
# Per-type injection (each deterministic given an rng instance)
# ---------------------------------------------------------------------------

def _inject_whitespace(
    rows, rng, counter, records, scenario_id,
) -> tuple[int, list[dict]]:
    """Prepend/append spaces to 3 cells in the 'name' column."""
    targets = rng.sample(range(len(rows)), min(3, len(rows)))
    patterns = ["  {v}", "{v}  ", "  {v}  "]
    for i, idx in enumerate(targets):
        row = rows[idx]
        clean_val = row["name"]
        corrupted_val = patterns[i % len(patterns)].format(v=clean_val)
        row["name"] = corrupted_val
        counter += 1
        records.append(CellCorruption(
            corruption_id=f"{scenario_id}::C{counter:03d}",
            corruption_type=CorruptionType.WHITESPACE,
            row_id=row["_row_id"], column="name",
            clean_value=clean_val, corrupted_value=corrupted_val,
        ))
    return counter, rows


def _inject_duplicates(
    rows, rng, counter, records, scenario_id,
) -> tuple[int, list[dict]]:
    """Copy 2 existing rows with new _row_id values, appended at the end."""
    sources = rng.sample(range(len(rows)), min(2, len(rows)))
    new_id_start = max(int(r["_row_id"][1:]) for r in rows) + 1
    for i, src_idx in enumerate(sources):
        src_row = rows[src_idx]
        dup_row = dict(src_row)
        dup_row_id = f"R{new_id_start + i:03d}"
        dup_row["_row_id"] = dup_row_id
        rows.append(dup_row)
        counter += 1
        records.append(DuplicateCorruption(
            corruption_id=f"{scenario_id}::C{counter:03d}",
            source_row_id=src_row["_row_id"],
            duplicate_row_id=dup_row_id,
        ))
    return counter, rows


_CASE_VARIANTS = {
    "Engineering": ["engineering", "ENGINEERING", "eNGINEERING"],
    "Sales": ["sales", "SALES", "sALES"],
    "Marketing": ["marketing", "MARKETING", "mARKETING"],
}


def _inject_categorical_variants(
    rows, rng, counter, records, scenario_id,
) -> tuple[int, list[dict]]:
    """Replace 3 department values with case-variant forms."""
    targets = rng.sample(range(len(rows)), min(3, len(rows)))
    for idx in targets:
        row = rows[idx]
        clean_val = row["department"]
        variants = _CASE_VARIANTS.get(clean_val)
        if not variants:
            continue
        corrupted_val = rng.choice(variants)
        row["department"] = corrupted_val
        counter += 1
        records.append(CellCorruption(
            corruption_id=f"{scenario_id}::C{counter:03d}",
            corruption_type=CorruptionType.CATEGORICAL_VARIANT,
            row_id=row["_row_id"], column="department",
            clean_value=clean_val, corrupted_value=corrupted_val,
        ))
    return counter, rows


def _inject_missing_values(
    rows, rng, counter, records, scenario_id,
) -> tuple[int, list[dict]]:
    """Null out 2 salary values."""
    targets = rng.sample(range(len(rows)), min(2, len(rows)))
    for idx in targets:
        row = rows[idx]
        clean_val = row["salary"]
        row["salary"] = ""
        counter += 1
        records.append(CellCorruption(
            corruption_id=f"{scenario_id}::C{counter:03d}",
            corruption_type=CorruptionType.MISSING_VALUE,
            row_id=row["_row_id"], column="salary",
            clean_value=clean_val, corrupted_value="",
        ))
    return counter, rows


def _inject_mixed_dates(
    rows, rng, counter, records, scenario_id,
) -> tuple[int, list[dict]]:
    """Rewrite 3 hire_date values from YYYY-MM-DD to DD/MM/YYYY."""
    targets = rng.sample(range(len(rows)), min(3, len(rows)))
    for idx in targets:
        row = rows[idx]
        clean_val = row["hire_date"]  # "YYYY-MM-DD"
        parts = clean_val.split("-")
        if len(parts) == 3:
            corrupted_val = f"{parts[2]}/{parts[1]}/{parts[0]}"
            row["hire_date"] = corrupted_val
            counter += 1
            records.append(CellCorruption(
                corruption_id=f"{scenario_id}::C{counter:03d}",
                corruption_type=CorruptionType.MIXED_DATE_FORMAT,
                row_id=row["_row_id"], column="hire_date",
                clean_value=clean_val, corrupted_value=corrupted_val,
            ))
    return counter, rows


# ---------------------------------------------------------------------------
# Convenience: generate all six scenarios
# ---------------------------------------------------------------------------

SCENARIO_SPECS: dict[str, CorruptionSpec] = {
    "S1_whitespace": CorruptionSpec(
        scenario_id="S1_whitespace", corruption_types=[CorruptionType.WHITESPACE], seed=42,
    ),
    "S2_duplicates": CorruptionSpec(
        scenario_id="S2_duplicates", corruption_types=[CorruptionType.EXACT_DUPLICATE], seed=42,
    ),
    "S3_categorical": CorruptionSpec(
        scenario_id="S3_categorical", corruption_types=[CorruptionType.CATEGORICAL_VARIANT], seed=42,
    ),
    "S4_missing": CorruptionSpec(
        scenario_id="S4_missing", corruption_types=[CorruptionType.MISSING_VALUE], seed=42,
    ),
    "S5_dates": CorruptionSpec(
        scenario_id="S5_dates", corruption_types=[CorruptionType.MIXED_DATE_FORMAT], seed=42,
    ),
    "S6_mixed": CorruptionSpec(
        scenario_id="S6_mixed",
        corruption_types=[
            CorruptionType.WHITESPACE, CorruptionType.EXACT_DUPLICATE,
            CorruptionType.CATEGORICAL_VARIANT, CorruptionType.MISSING_VALUE,
            CorruptionType.MIXED_DATE_FORMAT,
        ],
        seed=42,
    ),
}


def generate_all_scenarios(base_dir: str) -> dict[str, GroundTruthManifest]:
    """
    Write the clean reference and all six corrupted datasets into
    `base_dir`, returning a manifest per scenario.
    """
    base = Path(base_dir)
    clean_path = str(base / "clean.csv")
    write_clean_fixture(clean_path)

    manifests: dict[str, GroundTruthManifest] = {}
    for name, spec in SCENARIO_SPECS.items():
        output_path = str(base / f"{name}.csv")
        manifests[name] = corrupt_dataset(clean_path, spec, output_path)

    return manifests

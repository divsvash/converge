"""
Corruption scenario models: typed corruption specifications, records,
and the ground-truth manifest.

No dependency on converge -- these are pure experiment-side value
objects describing what was done to a clean dataset and what correct
repair looks like. The evaluator (ground_truth.py) reads these; the
generator (generator.py) writes them.

STABLE ROW IDENTITY
Every row in the clean research fixture carries a column named
`_row_id` (prefixed with underscore to signal "infrastructure, not
data"). The profiler's id-like detection
(converge.services.spark.profiler._detect_id_like_columns) matches
column names against a word-boundary-aware id pattern and checks
≥90% distinctness. `_row_id` will match the pattern (contains "id")
and will have 100% distinct values, so the profiler will classify it
as id-like and:
    - exclude it from duplicate detection criteria
    - skip categorical/numeric profiling for it
    - not emit evidence about it
This means the AI will never see `_row_id` in evidence, so it cannot
use it to infer the correct answer. It exists solely for ground-truth
evaluation: after a strategy runs, the evaluator joins the clean
reference and the strategy's output on `_row_id` to compare cell
values. If a strategy's output drops `_row_id` (unlikely given that
no operation in the catalog can delete columns), the evaluator falls
back to reporting what it can measure.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class CorruptionType(str, Enum):
    WHITESPACE = "whitespace"
    EXACT_DUPLICATE = "exact_duplicate"
    CATEGORICAL_VARIANT = "categorical_variant"
    MISSING_VALUE = "missing_value"
    MIXED_DATE_FORMAT = "mixed_date_format"


class CellCorruption(BaseModel):
    """One injected cell-level corruption."""
    model_config = ConfigDict(frozen=True)

    corruption_id: str
    corruption_type: CorruptionType
    row_id: str = Field(description="Value of _row_id for the affected row.")
    column: str
    clean_value: Any = Field(description="The original clean value before corruption.")
    corrupted_value: Any = Field(description="The value written into the corrupted dataset.")


class DuplicateCorruption(BaseModel):
    """One injected exact-duplicate row."""
    model_config = ConfigDict(frozen=True)

    corruption_id: str
    corruption_type: CorruptionType = CorruptionType.EXACT_DUPLICATE
    source_row_id: str = Field(description="_row_id of the original row that was copied.")
    duplicate_row_id: str = Field(description="_row_id assigned to the injected duplicate row.")


class CorruptionSpec(BaseModel):
    """
    Declarative specification for one corruption pass. The generator
    reads this and deterministically produces a corrupted dataset +
    manifest.
    """
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    corruption_types: list[CorruptionType]
    seed: int = Field(description="Determinism seed — same spec + same clean data + same seed = identical output.")


class GroundTruthManifest(BaseModel):
    """
    Complete record of every corruption injected into one scenario's
    dataset. This is what the evaluator reads to know what correct
    repair looks like.
    """
    scenario_id: str
    clean_dataset_checksum: str
    corrupted_dataset_checksum: str
    corruption_seed: int

    cell_corruptions: list[CellCorruption] = Field(default_factory=list)
    duplicate_corruptions: list[DuplicateCorruption] = Field(default_factory=list)

    @property
    def total_corruptions(self) -> int:
        return len(self.cell_corruptions) + len(self.duplicate_corruptions)

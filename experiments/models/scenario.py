"""
Dataset and Scenario -- the fixed experimental input side.

Dataset is an immutable descriptor of a concrete file on disk: a
stable id, a path, and a SHA-256 content checksum. It has no
experiment-specific behavior and no converge dependency -- it is a
plain fact about a file.

The checksum exists specifically to detect dataset drift: if a
dataset_id is reused across a long-running sweep, the checksum lets us
notice the underlying file changed on disk between runs, rather than
silently comparing results that were never actually produced from the
same input. Identity for comparison purposes is `dataset_id` (a
human-assigned stable label, e.g. "sample_hr"); the checksum's job is
drift detection, not identity.

Scenario is the fixed experimental input condition: a Dataset plus a
name and optional description -- nothing about which Configuration
ran against it. Held constant across every Configuration compared
within one Experiment (see experiment.py). Dataset is embedded
directly (composition, not a lookup-by-id reference) since there is no
dataset registry -- a registry was explicitly rejected during the
architecture design (nothing to register beyond what direct reference
already gives, for the current single-dataset scale) and embedding
keeps a Scenario fully self-contained for reproducibility: the
checksum that pins down "exactly this file" travels with the Scenario,
not through a separate lookup step.

Deliberately NOT included: ground-truth machinery (known-correct
answers for the dataset). Not designed yet; adding it needs its own
task, not a speculative field here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Dataset(BaseModel):
    """
    Immutable descriptor of one experiment input file.
    """
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    path: str
    checksum: str = Field(
        description="SHA-256 hex digest of the file's contents at the time this Dataset was constructed."
    )

    @field_validator("checksum")
    @classmethod
    def _validate_checksum_shape(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower()):
            raise ValueError("checksum must be a 64-character hex SHA-256 digest.")
        return v.lower()

    @classmethod
    def from_path(cls, dataset_id: str, path: str) -> "Dataset":
        """
        Computes the SHA-256 checksum of the file at `path` and
        constructs a Dataset. This is the normal way to create one;
        direct construction (e.g. when reloading a persisted
        Scenario) requires the checksum to already be known.
        """
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return cls(dataset_id=dataset_id, path=str(path), checksum=digest)


class Scenario(BaseModel):
    """
    The fixed experimental input condition: a Dataset plus a name and
    optional description. Held constant across every Configuration
    compared within one Experiment.
    """
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    name: str
    dataset: Dataset
    description: Optional[str] = None

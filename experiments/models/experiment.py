"""
Experiment -- one research question: a fixed Scenario, a set of
Configurations to compare, and how many times each should be run.

No execution behavior here. This is purely the declaration of what
comparison is being requested, not the machinery that runs it
(experiments/runner, not implemented in this phase).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from experiments.models.configuration import Configuration
from experiments.models.scenario import Scenario


class Experiment(BaseModel):
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    scenario: Scenario
    configurations: list[Configuration] = Field(min_length=1)
    repetition_count: int = Field(ge=1)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("configurations")
    @classmethod
    def _validate_unique_configurations(cls, configurations: list[Configuration]) -> list[Configuration]:
        """
        Configurations are compared by content (configuration_id), not
        by object identity -- listing the same configuration twice in
        one Experiment is almost certainly a mistake, not a legitimate
        way to request extra repetitions (that's what repetition_count
        is for).
        """
        ids = [c.configuration_id for c in configurations]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"configurations must be unique by content; duplicate configuration_id(s): {duplicates}"
            )
        return configurations

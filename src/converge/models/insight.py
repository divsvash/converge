"""
Insight models.

An Insight is explicitly tagged as FACT, INTERPRETATION, or
UNCERTAINTY so a reader (or a future UI) never has to guess which
kind of claim they're looking at. For P0, every Insight is produced
deterministically from a QualityComparison -- there is no LLM call
anywhere in this module or its generator (see
converge.services.insights.generator). See that module's docstring for
why, and for how an optional AI-phrased layer could be added later
without changing this contract.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from converge.core.time_utils import utcnow
from converge.models.profile import IssueType


class InsightKind(str, Enum):
    FACT = "fact"
    """A claim directly and fully supported by the QualityComparison --
    a number, a count, a direct before/after value. No inference."""

    INTERPRETATION = "interpretation"
    """A mild, explicitly-hedged reading of the facts (e.g. "associated
    with a net improvement"). Never asserts causation. Always
    generated from an explicit, documented rule -- never free-form."""

    UNCERTAINTY = "uncertainty"
    """Flags something that could NOT be determined -- an UNVERIFIED
    comparison, a failed/skipped operation, etc. Exists specifically so
    gaps in the evidence are surfaced, not silently omitted."""


class Insight(BaseModel):
    kind: InsightKind
    text: str
    related_issue_types: list[IssueType] = Field(default_factory=list)
    related_columns: list[str] = Field(default_factory=list)


class InsightArtifact(BaseModel):
    insights: list[Insight] = Field(default_factory=list)
    source: str = Field(
        default="deterministic",
        description="How these insights were produced. 'deterministic' for the P0 template-based "
                     "generator; reserved value 'ai' for a possible future AIProvider-backed phrasing "
                     "layer (not implemented) that would still consume only this same bounded "
                     "QualityComparison artifact, never the raw dataset.",
    )
    generated_at: datetime = Field(default_factory=utcnow)

"""
Structured recommendation contract.

This is what the AI provider (Gemini, later Phi/Ollama) is required to
produce. It is deliberately a data-only schema:

- `operation` is a plain string, not code, not a query, not a Spark
  expression -- it must match a key in the operation registry to mean
  anything, and the validator (not the type system) is the source of
  truth for whether it's supported.
- `parameters` is a plain JSON-shaped dict of primitives, validated
  structurally against the operation's declared ParamSpecs -- never
  evaluated, executed, or interpolated into a query string.
- There is no field anywhere in this schema for executable code,
  SQL, or a Spark expression. The AI has no channel through which to
  submit anything other than a structured, typed proposal.

`operation` is intentionally typed as `str` rather than a strict enum:
this lets a malformed, forbidden, or arbitrary value (e.g.
"ARBITRARY_CODE" or a nonsense string) construct successfully as a
Recommendation object, so the *validator* can reject it with a clear,
structured explanation -- rather than the object failing to even
construct and the failure being an opaque exception instead of an
auditable validation result.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from converge.models.profile import IssueType


class EvidenceReference(BaseModel):
    """
    A claim the AI makes about which profiler-generated evidence
    supports its recommendation. This is a *claim*, not a trusted
    fact -- the validator independently looks up the actual evidence
    in the profile and cross-checks it; the reference is never taken
    at face value.
    """
    issue_type: IssueType
    columns: list[str] = Field(default_factory=list)
    claim: str = Field(
        description="Free-text description of what the AI believes this evidence shows."
    )
    claimed_values: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Specific facts the AI claims (e.g. {'null_count': 2} or "
            "{'variant_count': 4}). The validator cross-checks any key "
            "present here against the corresponding field in the "
            "actual matched EvidenceItem.supporting_data and rejects "
            "the recommendation on mismatch."
        ),
    )


class Recommendation(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    operation: str = Field(
        description="Raw operation identifier. See module docstring for why this is a str, not an enum."
    )
    target_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Column(s) this recommendation affects. For row-level "
            "operations (REMOVE_DUPLICATES) this is the set of "
            "criteria columns defining a duplicate, not a column to "
            "transform."
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Plain JSON-shaped parameters, validated structurally -- never executed.",
    )
    reason: str = Field(
        default="",
        description="Free-text explanation from the AI. Never parsed or executed as instructions.",
    )
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

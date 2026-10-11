"""
Operation definition schema.

This is metadata *about* operations (safety tier, allowed column types,
declared parameters, evidence requirements) -- not the registry itself
(that's registry.py) and not a recommendation instance (that's
recommendation.py).

Deliberately a flat, static, Pydantic-validated data structure -- not a
plugin system. Extending the P0 operation set later means adding one
OperationDefinition entry to registry.py, not writing a new interface.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from converge.models.profile import IssueType, SemanticType


class SafetyTier(str, Enum):
    """
    Caps what validation_status an operation can ever reach,
    independent of how well individual checks pass. This is what
    correction #4/#5 ("safety class must be respected") is enforced
    through structurally, not just by convention:
      - SAFE            -> eligible for APPROVED_AUTO (still requires
                            manual Apply at the API/UI layer -- P0
                            never auto-executes anything)
      - REVIEW_REQUIRED  -> capped at NEEDS_REVIEW, never APPROVED_AUTO
      - FINDING_ONLY     -> capped at NOT_EXECUTABLE; can never become
                            an executable recommendation regardless of
                            how well every other check passes
    """
    SAFE = "safe"
    REVIEW_REQUIRED = "review_required"
    FINDING_ONLY = "finding_only"


ParamType = str  # one of: "str", "int", "float", "bool", "dict", "list"


class ParamSpec(BaseModel):
    """
    Declares one parameter an operation accepts. The validator uses
    this to structurally type-check AI-supplied parameters -- this is
    part of what keeps the AI's output data, not code: parameters are
    validated as plain JSON-shaped values (str/int/float/bool/dict/
    list) against a declared schema, never interpreted as expressions.
    """
    name: str
    type: ParamType
    required: bool = True
    allowed_values: Optional[list[Any]] = None
    description: str = ""


class OperationDefinition(BaseModel):
    operation: str
    safety_tier: SafetyTier
    description: str

    compatible_semantic_types: list[SemanticType] = Field(
        default_factory=list,
        description=(
            "Semantic types this operation may target. Empty list means "
            "the operation is not bound to a single column's type (e.g. "
            "a row-level operation like REMOVE_DUPLICATES)."
        ),
    )
    parameters: list[ParamSpec] = Field(default_factory=list)

    required_evidence_issue_types: list[IssueType] = Field(
        default_factory=list,
        description=(
            "If non-empty, the validator requires at least one profiler-"
            "generated EvidenceItem of one of these issue types, scoped "
            "to the recommendation's target columns, to exist "
            "independently in the profile. If empty, evidence is not "
            "structurally required for this operation in P0 (e.g. "
            "CONVERT_DTYPE, for which the profiler does not yet emit "
            "dedicated evidence) -- but the operation's safety tier "
            "ceiling (never APPROVED_AUTO for REVIEW_REQUIRED) still "
            "applies regardless."
        ),
    )

    is_row_level: bool = Field(
        default=False,
        description=(
            "True for operations whose target_columns represent "
            "row-matching criteria rather than a column to transform "
            "(currently only REMOVE_DUPLICATES). Affects how column "
            "evidence-matching and dtype checks are applied."
        ),
    )

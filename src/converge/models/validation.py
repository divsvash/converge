"""
Validation result contract.

CRITICAL DISTINCTION (per architecture correction #4): a ValidationStatus
of APPROVED_AUTO means the recommendation passed every automated check
and is *eligible* for execution. It does NOT mean it executes, and it
does NOT mean the user has approved it. That is tracked entirely
separately, by a `decision` field (PENDING / APPROVED_BY_USER /
REJECTED_BY_USER) that belongs to the future orchestrator/API layer,
not to this validator. The validator has no concept of, and no path
to, execution.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    APPROVED_AUTO = "approved_auto"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    # Reserved for FINDING_ONLY-tier operations (e.g. FLAG_OUTLIER).
    # Structurally distinct from the other three states so that a
    # finding-only recommendation can never be mistaken for, or
    # accidentally routed through, an execution path -- there is no
    # status value an executor could check that a finding-only
    # recommendation could ever hold in common with an executable one.
    NOT_EXECUTABLE = "not_executable"


class CheckResult(BaseModel):
    """One individual rule evaluation, always reported -- pass or fail."""
    rule: str
    passed: Optional[bool] = Field(
        description="True/False, or None if the check was skipped because a prior blocking check failed."
    )
    message: str


class ValidationResult(BaseModel):
    recommendation_id: str
    operation: str
    status: ValidationStatus
    checks: list[CheckResult]
    summary: str = Field(
        description="One-line human-readable explanation of the final status, for the UI validation checklist."
    )

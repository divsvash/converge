"""
Validator.

The validator is the trust boundary between the AI provider and Spark
execution. It runs entirely against the deterministic profile/evidence
contract produced in Step 1 (converge/models/profile.py,
converge/services/spark/profiler.py). It:

    - NEVER calls Gemini or any AI provider.
    - NEVER inspects the raw CSV or a live Spark DataFrame.
    - NEVER trusts a claim the AI makes about the data -- every claim
      (evidence reference, mapping key, criteria column) is looked up
      independently in the profile/evidence the validator was given,
      and cross-checked for contradiction where that's deterministically
      possible.

A ValidationStatus of APPROVED_AUTO means "eligible for execution
pending the user clicking Apply." It never means anything executes.
The validator has no concept of a `decision` field or of execution --
that distinction belongs entirely to the future orchestrator/API layer.
"""
from __future__ import annotations

from converge.models.operation import OperationDefinition, SafetyTier
from converge.models.profile import ColumnProfile, EvidenceItem, ProfileResult
from converge.models.recommendation import Recommendation
from converge.models.validation import CheckResult, ValidationResult, ValidationStatus
from converge.models.operation_catalog import FORBIDDEN_OPERATIONS, OPERATION_REGISTRY

# Tolerance for comparing AI-claimed numeric facts against the actual
# profile (protects against false "contradiction" flags from floating
# point noise, e.g. a rounded percentage).
_NUMERIC_TOLERANCE = 1e-6


class Validator:
    """
    Constructed once per profile/evidence bundle (i.e. once per
    pipeline run), then used to validate every recommendation produced
    for that run.
    """

    def __init__(self, profile_result: ProfileResult):
        self._profile_result = profile_result
        self._columns_by_name: dict[str, ColumnProfile] = {
            c.name: c for c in profile_result.profile.columns
        }
        self._evidence: list[EvidenceItem] = profile_result.evidence

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def validate(self, rec: Recommendation) -> ValidationResult:
        checks: list[CheckResult] = []

        # --- Rules 1 & 8: operation known / not forbidden -----------
        op_def = self._check_operation_supported(rec, checks)
        if op_def is None:
            return self._finalize(rec, checks, ValidationStatus.REJECTED)

        # --- Rule 2: target columns exist ----------------------------
        if not self._check_target_columns_exist(rec, checks):
            return self._finalize(rec, checks, ValidationStatus.REJECTED)

        # --- Rule 3: dtype compatibility -----------------------------
        if not self._check_dtype_compatible(rec, op_def, checks):
            return self._finalize(rec, checks, ValidationStatus.REJECTED)

        # --- Rule 4: parameters valid ---------------------------------
        if not self._check_parameters_valid(rec, op_def, checks):
            return self._finalize(rec, checks, ValidationStatus.REJECTED)

        # --- Rule 5: safety class noted (informational, always passes) ---
        checks.append(CheckResult(
            rule="safety_class_respected",
            passed=True,
            message=(
                f"Operation tier is '{op_def.safety_tier.value}'; final status "
                f"is capped accordingly regardless of other checks."
            ),
        ))

        # --- Rules 6 & 7: evidence exists / supports the claim --------
        evidence_ok = self._check_evidence(rec, op_def, checks)
        if not evidence_ok:
            return self._finalize(rec, checks, ValidationStatus.REJECTED)

        # --- Rule 9: finding-only operations can never be executable --
        if op_def.safety_tier == SafetyTier.FINDING_ONLY:
            checks.append(CheckResult(
                rule="finding_only_not_executable",
                passed=True,
                message=(
                    f"'{rec.operation}' is FINDING_ONLY: it produces a report, "
                    f"never an executable transformation."
                ),
            ))
            return self._finalize(rec, checks, ValidationStatus.NOT_EXECUTABLE)

        checks.append(CheckResult(
            rule="finding_only_not_executable",
            passed=True,
            message="Not a finding-only operation.",
        ))

        # --- Final status: capped by safety tier -----------------------
        if op_def.safety_tier == SafetyTier.SAFE:
            status = ValidationStatus.APPROVED_AUTO
        else:  # REVIEW_REQUIRED
            status = ValidationStatus.NEEDS_REVIEW

        return self._finalize(rec, checks, status)

    # ------------------------------------------------------------------
    # Rule 1 / 8: operation supported & not forbidden
    # ------------------------------------------------------------------

    def _check_operation_supported(
        self, rec: Recommendation, checks: list[CheckResult]
    ) -> OperationDefinition | None:
        if rec.operation in FORBIDDEN_OPERATIONS:
            checks.append(CheckResult(
                rule="operation_supported",
                passed=False,
                message=f"'{rec.operation}' is not permitted to execute.",
            ))
            checks.append(CheckResult(
                rule="forbidden_operation_rejected",
                passed=False,
                message=(
                    f"'{rec.operation}' matches the explicit P0 forbidden/unsupported "
                    f"list and is rejected regardless of parameters or evidence."
                ),
            ))
            return None

        if rec.operation not in OPERATION_REGISTRY:
            checks.append(CheckResult(
                rule="operation_supported",
                passed=False,
                message=f"'{rec.operation}' is not a recognized P0 operation.",
            ))
            checks.append(CheckResult(
                rule="forbidden_operation_rejected",
                passed=True,
                message="Not on the forbidden list (it is simply unknown).",
            ))
            return None

        checks.append(CheckResult(
            rule="operation_supported",
            passed=True,
            message=f"'{rec.operation}' is a registered P0 operation.",
        ))
        checks.append(CheckResult(
            rule="forbidden_operation_rejected",
            passed=True,
            message="Not on the forbidden list.",
        ))
        return OPERATION_REGISTRY[rec.operation]

    # ------------------------------------------------------------------
    # Rule 2: target columns exist
    # ------------------------------------------------------------------

    def _check_target_columns_exist(
        self, rec: Recommendation, checks: list[CheckResult]
    ) -> bool:
        if not rec.target_columns:
            checks.append(CheckResult(
                rule="target_columns_exist",
                passed=False,
                message="Recommendation specifies no target column(s).",
            ))
            return False

        missing = [c for c in rec.target_columns if c not in self._columns_by_name]
        if missing:
            checks.append(CheckResult(
                rule="target_columns_exist",
                passed=False,
                message=f"Column(s) not found in the dataset profile: {missing}.",
            ))
            return False

        checks.append(CheckResult(
            rule="target_columns_exist",
            passed=True,
            message=f"All target column(s) exist: {rec.target_columns}.",
        ))
        return True

    # ------------------------------------------------------------------
    # Rule 3: dtype / semantic-type compatibility
    # ------------------------------------------------------------------

    def _check_dtype_compatible(
        self, rec: Recommendation, op_def: OperationDefinition, checks: list[CheckResult]
    ) -> bool:
        if op_def.is_row_level or not op_def.compatible_semantic_types:
            checks.append(CheckResult(
                rule="dtype_compatible",
                passed=True,
                message="Operation is row-level / not bound to a single column's type.",
            ))
            return True

        incompatible = [
            c for c in rec.target_columns
            if self._columns_by_name[c].semantic_type not in op_def.compatible_semantic_types
        ]
        if incompatible:
            details = {
                c: self._columns_by_name[c].semantic_type.value for c in incompatible
            }
            checks.append(CheckResult(
                rule="dtype_compatible",
                passed=False,
                message=(
                    f"Column(s) have a semantic type incompatible with "
                    f"'{rec.operation}': {details}. Compatible types: "
                    f"{[t.value for t in op_def.compatible_semantic_types]}."
                ),
            ))
            return False

        checks.append(CheckResult(
            rule="dtype_compatible",
            passed=True,
            message="Target column(s) have a compatible semantic type.",
        ))
        return True

    # ------------------------------------------------------------------
    # Rule 4: parameters valid
    # ------------------------------------------------------------------

    def _check_parameters_valid(
        self, rec: Recommendation, op_def: OperationDefinition, checks: list[CheckResult]
    ) -> bool:
        errors: list[str] = []
        declared_names = {p.name for p in op_def.parameters}

        # Unexpected parameters are rejected outright -- this is part
        # of the structural boundary: the AI cannot smuggle arbitrary
        # extra fields (e.g. a "code" or "sql" key) into a recommendation
        # that otherwise looks like a legitimate, narrow operation.
        unexpected = [k for k in rec.parameters.keys() if k not in declared_names]
        if unexpected:
            errors.append(f"unexpected parameter(s): {unexpected}")

        for spec in op_def.parameters:
            if spec.name not in rec.parameters:
                if spec.required:
                    errors.append(f"missing required parameter '{spec.name}'")
                continue

            value = rec.parameters[spec.name]
            if not _matches_type(value, spec.type):
                errors.append(
                    f"parameter '{spec.name}' expected type '{spec.type}', "
                    f"got '{type(value).__name__}'"
                )
                continue

            if spec.allowed_values is not None and value not in spec.allowed_values:
                errors.append(
                    f"parameter '{spec.name}' has value {value!r}, "
                    f"which is not one of the allowed values {spec.allowed_values}"
                )

        # Operation-specific conditional requirement: FILL_MISSING with
        # strategy == 'constant' requires fill_value.
        if op_def.operation == "FILL_MISSING" and rec.parameters.get("strategy") == "constant":
            if "fill_value" not in rec.parameters:
                errors.append("strategy 'constant' requires parameter 'fill_value'")

        if errors:
            checks.append(CheckResult(
                rule="parameters_valid",
                passed=False,
                message="; ".join(errors),
            ))
            return False

        checks.append(CheckResult(
            rule="parameters_valid",
            passed=True,
            message="All declared parameters present, well-typed, and within allowed values.",
        ))
        return True

    # ------------------------------------------------------------------
    # Rules 6 & 7: evidence exists / supports the claim
    # ------------------------------------------------------------------

    def _check_evidence(
        self, rec: Recommendation, op_def: OperationDefinition, checks: list[CheckResult]
    ) -> bool:
        if not op_def.required_evidence_issue_types:
            checks.append(CheckResult(
                rule="evidence_exists",
                passed=True,
                message=f"'{rec.operation}' does not structurally require profiler evidence in P0.",
            ))
            checks.append(CheckResult(
                rule="evidence_supports_claim",
                passed=True,
                message="Not applicable: no evidence was required.",
            ))
            return True

        matches = self._find_matching_evidence(rec, op_def)
        if not matches:
            checks.append(CheckResult(
                rule="evidence_exists",
                passed=False,
                message=(
                    f"No matching evidence found in the Spark profile for issue "
                    f"type(s) {[t.value for t in op_def.required_evidence_issue_types]} "
                    f"on column(s) {rec.target_columns}. The recommendation is not "
                    f"independently supported by the profile."
                ),
            ))
            checks.append(CheckResult(
                rule="evidence_supports_claim",
                passed=False,
                message="Skipped: no evidence was found to check the claim against.",
            ))
            return False

        checks.append(CheckResult(
            rule="evidence_exists",
            passed=True,
            message=f"Found {len(matches)} matching evidence item(s) in the profile.",
        ))

        contradiction = self._find_contradiction(rec, op_def, matches)
        if contradiction:
            checks.append(CheckResult(
                rule="evidence_supports_claim",
                passed=False,
                message=contradiction,
            ))
            return False

        checks.append(CheckResult(
            rule="evidence_supports_claim",
            passed=True,
            message="Recommendation's claims are consistent with the actual profile evidence.",
        ))
        return True

    def _find_matching_evidence(
        self, rec: Recommendation, op_def: OperationDefinition
    ) -> list[EvidenceItem]:
        required_types = set(op_def.required_evidence_issue_types)
        target_set = set(rec.target_columns)

        matches = []
        for item in self._evidence:
            if item.issue_type not in required_types:
                continue
            evidence_cols = set(item.columns)
            if op_def.is_row_level:
                # REMOVE_DUPLICATES: criteria columns must match exactly.
                # A loose subset/overlap match would let the AI claim
                # duplicate support for a criteria set that was never
                # actually verified by Spark.
                if evidence_cols == target_set:
                    matches.append(item)
            else:
                # Column-scoped operations: every target column must be
                # covered by this evidence item's columns.
                if target_set.issubset(evidence_cols):
                    matches.append(item)
        return matches

    def _find_contradiction(
        self, rec: Recommendation, op_def: OperationDefinition, matches: list[EvidenceItem]
    ) -> str | None:
        # --- Generic check: any claimed_values the AI attached to its
        # evidence_refs must match the real evidence, where the same
        # key exists in both.
        for ref in rec.evidence_refs:
            if not ref.claimed_values:
                continue
            candidate = self._best_matching_evidence(ref, matches)
            if candidate is None:
                continue
            for key, claimed in ref.claimed_values.items():
                actual = candidate.supporting_data.get(key)
                if actual is None:
                    continue
                if not _values_match(claimed, actual):
                    return (
                        f"Claimed value for '{key}' ({claimed!r}) contradicts the "
                        f"actual profile evidence ({actual!r}) for column(s) {ref.columns}."
                    )

        # --- Operation-specific deterministic checks ---------------------
        if op_def.operation == "STANDARDIZE_CATEGORY":
            mapping = rec.parameters.get("mapping", {})
            known_variants: set[str] = set()
            for item in matches:
                known_variants.update(item.supporting_data.get("variants", {}).keys())
            unknown_keys = [k for k in mapping.keys() if k not in known_variants]
            if unknown_keys:
                return (
                    f"Mapping references raw value(s) {unknown_keys} that do not "
                    f"appear in the profiler's evidence for this column. The "
                    f"recommendation is not supported by the actual data."
                )

        if op_def.operation == "REMOVE_DUPLICATES":
            # Already enforced via exact-set matching in
            # _find_matching_evidence -- if we reach here, matches is
            # non-empty and criteria columns are already verified equal.
            pass

        return None

    @staticmethod
    def _best_matching_evidence(
        ref, candidates: list[EvidenceItem]
    ) -> EvidenceItem | None:
        for item in candidates:
            if item.issue_type == ref.issue_type and set(ref.columns).issubset(set(item.columns)):
                return item
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize(
        rec: Recommendation, checks: list[CheckResult], status: ValidationStatus
    ) -> ValidationResult:
        summaries = {
            ValidationStatus.APPROVED_AUTO: (
                "Passed all automated checks; eligible for execution once a "
                "user explicitly applies it."
            ),
            ValidationStatus.NEEDS_REVIEW: (
                "Passed automated checks but requires manual review before it "
                "can be applied (review-required operation tier)."
            ),
            ValidationStatus.REJECTED: "Failed one or more validation checks; not eligible for execution.",
            ValidationStatus.NOT_EXECUTABLE: "Finding-only: report generated, no transformation is possible.",
        }
        return ValidationResult(
            recommendation_id=rec.recommendation_id,
            operation=rec.operation,
            status=status,
            checks=checks,
            summary=summaries[status],
        )


# ---------------------------------------------------------------------------
# Parameter type checking helpers
# ---------------------------------------------------------------------------

def _matches_type(value, expected_type: str) -> bool:
    if expected_type == "str":
        return isinstance(value, str)
    if expected_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "bool":
        return isinstance(value, bool)
    if expected_type == "dict":
        return isinstance(value, dict)
    if expected_type == "list":
        return isinstance(value, list)
    return False


def _values_match(claimed, actual) -> bool:
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        return abs(claimed - actual) <= _NUMERIC_TOLERANCE
    return claimed == actual

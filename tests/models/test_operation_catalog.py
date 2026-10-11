from converge.models.operation import SafetyTier
from converge.models.operation_catalog import FORBIDDEN_OPERATIONS, OPERATION_REGISTRY

EXPECTED_SAFE = {"TRIM_WHITESPACE", "NORMALIZE_CASE", "STANDARDIZE_CATEGORY", "STANDARDIZE_DATE_FORMAT"}
EXPECTED_REVIEW_REQUIRED = {"FILL_MISSING", "CONVERT_DTYPE", "REMOVE_DUPLICATES"}
EXPECTED_FINDING_ONLY = {"FLAG_OUTLIER"}
EXPECTED_FORBIDDEN = {
    "DELETE_COLUMN", "DELETE_DATASET", "ARBITRARY_ROW_FILTER", "ARBITRARY_CODE",
    "SCHEMA_RESTRUCTURING", "JOIN", "PIVOT", "MERGE",
}


def test_registry_contains_exactly_the_p0_operations():
    expected = EXPECTED_SAFE | EXPECTED_REVIEW_REQUIRED | EXPECTED_FINDING_ONLY
    assert set(OPERATION_REGISTRY.keys()) == expected


def test_safe_tier_operations_are_tagged_correctly():
    for name in EXPECTED_SAFE:
        assert OPERATION_REGISTRY[name].safety_tier == SafetyTier.SAFE


def test_review_required_tier_operations_are_tagged_correctly():
    for name in EXPECTED_REVIEW_REQUIRED:
        assert OPERATION_REGISTRY[name].safety_tier == SafetyTier.REVIEW_REQUIRED


def test_finding_only_tier_operations_are_tagged_correctly():
    for name in EXPECTED_FINDING_ONLY:
        assert OPERATION_REGISTRY[name].safety_tier == SafetyTier.FINDING_ONLY


def test_forbidden_operations_match_spec():
    assert FORBIDDEN_OPERATIONS == frozenset(EXPECTED_FORBIDDEN)


def test_forbidden_and_registered_sets_never_overlap():
    """
    A forbidden operation must never also appear as a registry key --
    that would create a code path where a name lookup could resolve a
    forbidden operation to an executable definition.
    """
    assert FORBIDDEN_OPERATIONS.isdisjoint(OPERATION_REGISTRY.keys())


def test_every_safe_operation_declares_required_evidence():
    """
    Every SAFE-tier operation must require at least one evidence type --
    a safety class that can be auto-approved must be evidence-backed,
    never based purely on structural checks.
    """
    for name in EXPECTED_SAFE:
        assert len(OPERATION_REGISTRY[name].required_evidence_issue_types) > 0, (
            f"{name} is SAFE tier but declares no required evidence"
        )


def test_remove_duplicates_is_row_level_and_structural():
    op = OPERATION_REGISTRY["REMOVE_DUPLICATES"]
    assert op.is_row_level is True
    assert op.compatible_semantic_types == []


def test_flag_outlier_has_no_parameters_and_cannot_carry_execution_intent():
    op = OPERATION_REGISTRY["FLAG_OUTLIER"]
    assert op.parameters == []

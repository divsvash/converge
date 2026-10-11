"""
P0 operation catalog.

Lives in `models/` rather than `services/validation/` because it is
populated, static reference data (instances of `OperationDefinition`,
a type already defined in `models/operation.py`) consumed by multiple
otherwise-independent services -- `services.validation.validator`
(to check a recommendation against it), `services.ai.prompts` (to
tell the model what it may recommend), and
`services.comparison.traceability` (to map an operation back to the
evidence issue types it addresses). None of those services owns this
data; it's shared domain reference data, so it belongs with the other
domain models, not nested inside any one service that merely reads it.

A flat dict, not a plugin system: adding an operation later means
adding one OperationDefinition entry here. No dynamic loading, no
handler discovery, no registration decorators.

Three explicit sets govern how the validator treats an operation name:

    OPERATION_REGISTRY   -- the 8 supported P0 operations and their
                             declared contract (safety tier, column
                             type compatibility, parameters, evidence
                             requirements).
    FORBIDDEN_OPERATIONS  -- operation names the system explicitly
                             recognizes and refuses, ever, regardless
                             of parameters or evidence. Reported
                             distinctly from "unknown operation" so the
                             validator's explanation is precise ("this
                             is forbidden" vs "this doesn't exist").
    (anything else)       -- any operation string that is neither
                             registered nor explicitly forbidden is
                             rejected as unknown/unsupported.
"""
from converge.models.operation import OperationDefinition, ParamSpec, SafetyTier
from converge.models.profile import IssueType, SemanticType

_STRING_LIKE = [SemanticType.STRING, SemanticType.DATE_LIKE_STRING]

OPERATION_REGISTRY: dict[str, OperationDefinition] = {

    # ----------------------------------------------------------------
    # SAFE / LOW-RISK -- eligible for APPROVED_AUTO, but P0 still
    # requires a manual "Apply" click at the API/UI layer regardless.
    # ----------------------------------------------------------------
    "TRIM_WHITESPACE": OperationDefinition(
        operation="TRIM_WHITESPACE",
        safety_tier=SafetyTier.SAFE,
        description="Strip leading/trailing whitespace from a string column.",
        compatible_semantic_types=_STRING_LIKE,
        parameters=[],
        required_evidence_issue_types=[IssueType.WHITESPACE_OR_CASING],
    ),

    "NORMALIZE_CASE": OperationDefinition(
        operation="NORMALIZE_CASE",
        safety_tier=SafetyTier.SAFE,
        description="Normalize a string column to a consistent case.",
        compatible_semantic_types=_STRING_LIKE,
        parameters=[
            ParamSpec(
                name="target_case", type="str", required=True,
                allowed_values=["lower", "upper", "title"],
                description="Target casing to normalize to.",
            ),
        ],
        # Case inconsistency shows up in our profiler either as
        # categorical variant groups (e.g. Delhi/delhi/DELHI) or as
        # whitespace/casing evidence -- either satisfies this rule.
        required_evidence_issue_types=[
            IssueType.CATEGORICAL_INCONSISTENCY,
            IssueType.WHITESPACE_OR_CASING,
        ],
    ),

    "STANDARDIZE_CATEGORY": OperationDefinition(
        operation="STANDARDIZE_CATEGORY",
        safety_tier=SafetyTier.SAFE,
        description="Map inconsistent raw category values to a single canonical value.",
        compatible_semantic_types=[SemanticType.STRING],
        parameters=[
            ParamSpec(
                name="mapping", type="dict", required=True,
                description="raw_value -> canonical_value. Every key must actually appear in the profiler's evidence.",
            ),
        ],
        required_evidence_issue_types=[IssueType.CATEGORICAL_INCONSISTENCY],
    ),

    "STANDARDIZE_DATE_FORMAT": OperationDefinition(
        operation="STANDARDIZE_DATE_FORMAT",
        safety_tier=SafetyTier.SAFE,
        description="Reformat a date-like string column to a single target format.",
        compatible_semantic_types=[SemanticType.DATE_LIKE_STRING],
        parameters=[
            ParamSpec(
                name="target_format", type="str", required=True,
                allowed_values=["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"],
                description="Fixed P0 set of supported target date formats.",
            ),
        ],
        required_evidence_issue_types=[IssueType.DATE_FORMAT_INCONSISTENCY],
    ),

    # ----------------------------------------------------------------
    # REVIEW REQUIRED -- capped at NEEDS_REVIEW, never APPROVED_AUTO,
    # regardless of how well individual checks pass.
    # ----------------------------------------------------------------
    "FILL_MISSING": OperationDefinition(
        operation="FILL_MISSING",
        safety_tier=SafetyTier.REVIEW_REQUIRED,
        description="Fill missing values in a column using an explicit strategy.",
        compatible_semantic_types=[
            SemanticType.INTEGER, SemanticType.FLOAT,
            SemanticType.STRING, SemanticType.DATE_LIKE_STRING,
            SemanticType.BOOLEAN,
        ],
        parameters=[
            ParamSpec(
                name="strategy", type="str", required=True,
                allowed_values=["mean", "median", "mode", "constant"],
                description="Fill strategy. 'constant' requires fill_value.",
            ),
            ParamSpec(
                name="fill_value", type="str", required=False,
                description="Required only when strategy == 'constant'.",
            ),
        ],
        required_evidence_issue_types=[IssueType.MISSING_VALUES],
    ),

    "CONVERT_DTYPE": OperationDefinition(
        operation="CONVERT_DTYPE",
        safety_tier=SafetyTier.REVIEW_REQUIRED,
        description="Convert a column to a compatible target data type.",
        compatible_semantic_types=[
            SemanticType.STRING, SemanticType.INTEGER, SemanticType.FLOAT,
        ],
        parameters=[
            ParamSpec(
                name="target_type", type="str", required=True,
                allowed_values=["integer", "float", "string", "boolean"],
            ),
        ],
        # The P0 profiler (Step 1) does not yet emit a dedicated
        # TYPE_INCONSISTENCY evidence item, so evidence is not
        # structurally required here -- but the REVIEW_REQUIRED tier
        # ceiling still applies, so this can never reach APPROVED_AUTO
        # regardless.
        required_evidence_issue_types=[],
    ),

    "REMOVE_DUPLICATES": OperationDefinition(
        operation="REMOVE_DUPLICATES",
        safety_tier=SafetyTier.REVIEW_REQUIRED,
        description=(
            "Remove rows that are duplicates according to an explicit set "
            "of criteria columns. Structural/destructive: row count changes."
        ),
        compatible_semantic_types=[],  # row-level, not bound to one column's type
        parameters=[],
        required_evidence_issue_types=[IssueType.DUPLICATE_ROWS],
        is_row_level=True,
    ),

    # ----------------------------------------------------------------
    # FINDING ONLY -- can never reach an executable status, ever.
    # ----------------------------------------------------------------
    "FLAG_OUTLIER": OperationDefinition(
        operation="FLAG_OUTLIER",
        safety_tier=SafetyTier.FINDING_ONLY,
        description=(
            "Report suspected outlier values for a numeric column. P0: "
            "report/finding only -- never adds an annotation column or "
            "otherwise modifies the dataset or schema."
        ),
        compatible_semantic_types=[SemanticType.INTEGER, SemanticType.FLOAT],
        parameters=[],
        required_evidence_issue_types=[IssueType.OUTLIER_CANDIDATE],
    ),
}


# ----------------------------------------------------------------
# FORBIDDEN / UNSUPPORTED -- recognized by name, always rejected,
# regardless of any other field in the recommendation. These are
# deliberately NOT registry entries: there is no code path (e.g. a
# missing-key lookup) that could ever accidentally resolve one of
# these to an OperationDefinition and treat it as executable.
# ----------------------------------------------------------------
FORBIDDEN_OPERATIONS: frozenset[str] = frozenset({
    "DELETE_COLUMN",
    "DELETE_DATASET",
    "ARBITRARY_ROW_FILTER",
    "ARBITRARY_CODE",
    "SCHEMA_RESTRUCTURING",
    "JOIN",
    "PIVOT",
    "MERGE",
})

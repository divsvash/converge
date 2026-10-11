"""
Versioned prompt for the P0 recommendation stage.

PROMPT_VERSION is bumped whenever SYSTEM_PROMPT's substantive content
changes (not for typo fixes). The version string is attached to every
GeminiProvider call site (see gemini_provider.py) so that, once run
artifacts are persisted (a later step), every stored recommendation
can be traced back to the exact prompt that produced it -- important
for the "does validation reduce hallucination-related failures"
research question this prototype is meant to eventually support.

The model is given:
    1. The rules (SYSTEM_PROMPT) -- fixed, versioned, provider-agnostic
       in content (a future PhiProvider/OllamaProvider would reuse the
       same SYSTEM_PROMPT and build_user_prompt output; only the
       transport/schema-enforcement mechanism differs per provider).
    2. The operation catalog, generated directly from
       OPERATION_REGISTRY -- so the model's understanding of what it
       may recommend can never drift out of sync with what the
       Validator will actually accept.
    3. The dataset profile + evidence, generated directly from the
       Step 1 ProfileResult -- and nothing else.
"""
from __future__ import annotations

import json

from converge.models.profile import ProfileResult
from converge.models.operation_catalog import OPERATION_REGISTRY

PROMPT_VERSION = "recommendation-prompt-v1"

SYSTEM_PROMPT = """\
You are the recommendation stage of CONVERGE, a validation-guided data \
cleaning pipeline. You analyze a data-quality profile produced by Apache \
Spark and propose structured cleaning recommendations. You do not see, \
query, or execute anything yourself -- you only reason over the evidence \
you are given.

Follow these rules exactly. They are not suggestions.

1. SPARK EVIDENCE IS AUTHORITATIVE. Every number, count, value, and fact in \
the profile and evidence you are given was computed deterministically by \
Spark. Treat it as ground truth. Never contradict it.

2. DO NOT INVENT FACTS. Only reference values, counts, and column names \
that actually appear in the profile/evidence you were given. Never assume \
or guess at data you cannot see.

3. DO NOT CALCULATE. Do not compute statistics, counts, or aggregates \
yourself, even if you believe you could. If a fact is not already present \
in the profile/evidence, you do not have it -- do not derive it.

4. ONLY RECOMMEND OPERATIONS FROM THE PROVIDED OPERATION CATALOG. You will \
be given the exact list of operations you may recommend, each with its \
allowed parameters. Do not invent an operation name. Do not recommend an \
operation that is not in the catalog, no matter how appropriate it might \
seem.

5. NEVER PRODUCE CODE. Do not output Python, SQL, Spark expressions, shell \
commands, or any executable syntax, in any field, including the reason \
field. Your output is a structured recommendation, not a program.

6. NEVER RECOMMEND UNSUPPORTED OR DESTRUCTIVE OPERATIONS. Do not suggest \
deleting a column, deleting the dataset, arbitrary row filtering, joins, \
pivots, merges, or schema restructuring. These are outside your role \
entirely.

7. EVERY RECOMMENDATION MUST REFERENCE THE EVIDENCE THAT SUPPORTS IT. For \
each recommendation, cite the specific evidence item(s) (by issue type and \
column) that justify it, and state the specific facts you are relying on \
in evidence_refs.claimed_values so they can be checked against the actual \
profile. A recommendation without a real evidentiary basis will be \
rejected regardless of how it is phrased.

8. IF EVIDENCE IS INSUFFICIENT, DO NOT FABRICATE A RECOMMENDATION. If you \
do not see clear evidence of a data-quality issue for a column, do not \
propose an operation for it. It is correct and expected to return fewer \
recommendations, or none, when the evidence does not support more.

Respond with a single JSON object matching the schema you were given, \
containing a "recommendations" array. Each entry must include: operation, \
target_columns, parameters, reason, evidence_refs, and optionally \
confidence (0.0-1.0). Do not include any text outside the JSON object.
"""


def build_operation_catalog_text() -> str:
    """
    Serializes OPERATION_REGISTRY into the exact catalog the model is
    told it may choose from. Built directly from the registry (not a
    hand-maintained copy) so the prompt can never drift out of sync
    with what the Validator will actually accept.
    """
    catalog = []
    for op in OPERATION_REGISTRY.values():
        catalog.append({
            "operation": op.operation,
            "safety_tier": op.safety_tier.value,
            "description": op.description,
            "applies_to_column_types": [t.value for t in op.compatible_semantic_types] or "row-level (not column-type-bound)",
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "allowed_values": p.allowed_values,
                }
                for p in op.parameters
            ],
        })
    return json.dumps(catalog, indent=2)


def build_user_prompt(profile_result: ProfileResult) -> str:
    """
    Builds the full user-turn prompt: operation catalog + profile +
    evidence, as JSON. This -- and only this -- is what the AI
    provider is allowed to send as the dataset-facing portion of the
    request. No raw dataset content, no file paths, no Spark handles.

    Deliberate redaction: `ColumnProfile.sample_values` (Step 1's
    small per-column debug sample, up to 5 raw cell values) and the
    `top_values` / `variant_groups` fields inside `categorical_stats`
    are stripped before building the prompt. Those fields exist for
    display/debugging in the profile artifact itself, but they carry
    unscoped raw cell content -- e.g. for a free-text column like a
    person's name, `top_values` would otherwise leak individual raw
    names to the AI with no evidentiary purpose (the column isn't
    flagged as an issue at all; it just happens to be a string
    column). The AI already receives every raw value it actually
    needs for semantic reasoning through the bounded, issue-scoped
    `evidence` list (e.g. STANDARDIZE_CATEGORY reasoning needs the
    actual variant strings, and those are present in
    evidence[].supporting_data for exactly the columns that were
    flagged -- not for every string column indiscriminately).
    Aggregate statistics (null counts, dtype, distinct counts, numeric
    stats, whitespace-affected counts, date format pattern counts) are
    all kept -- only unscoped raw-value fields are removed.
    """
    profile_dict = json.loads(profile_result.profile.model_dump_json())
    for column in profile_dict.get("columns", []):
        column.pop("sample_values", None)
        cat_stats = column.get("categorical_stats")
        if cat_stats is not None:
            cat_stats.pop("top_values", None)
            cat_stats.pop("variant_groups", None)

    payload = {
        "operation_catalog": json.loads(build_operation_catalog_text()),
        "dataset_profile": profile_dict,
        "evidence": json.loads(json.dumps([
            e.model_dump(mode="json") for e in profile_result.evidence
        ])),
    }
    return (
        "Analyze the following Spark-generated data quality profile and "
        "evidence, and propose cleaning recommendations strictly according "
        "to the rules and operation catalog above.\n\n"
        + json.dumps(payload, indent=2)
    )

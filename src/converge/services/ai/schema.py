"""
Gemini structured-output schema.

This is a Gemini-specific implementation detail (its schema format is
an OpenAPI-3.0 subset with uppercase type names) -- it lives here,
inside services/ai/, rather than anywhere shared, precisely so a
future PhiProvider/OllamaProvider is free to enforce structure however
its own backend supports (or not at all, falling back to prompt-only
guidance + strict Pydantic parsing) without needing to touch this file.

Design note: `parameters` and `claimed_values` are declared as
type OBJECT with no `properties` key. Gemini's schema format has no
"any JSON object" primitive that's cleaner than this, and a fully
discriminated schema (different parameter shapes per operation) would
require one schema branch per operation and add real complexity for a
P0 prototype. This is a deliberate trade-off: response_schema narrows
the *shape* of the response (a JSON object with a top-level
"recommendations" array of objects with these field names and types)
but does not attempt to constrain the exact keys inside `parameters`
at the schema-enforcement layer. The actual, authoritative enforcement
of parameter shape happens where it always was intended to happen --
in the Validator (Step 2) -- which is why this being "best effort"
rather than airtight does not weaken the safety boundary.
"""

RECOMMENDATION_RESPONSE_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "recommendations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "operation": {
                        "type": "STRING",
                        "description": "Must be one of the operation names in the provided catalog.",
                    },
                    "target_columns": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "parameters": {
                        "type": "OBJECT",
                        "description": "Operation-specific parameters, per the catalog's parameter list for this operation.",
                    },
                    "reason": {
                        "type": "STRING",
                        "description": "Plain-language justification. Never code.",
                    },
                    "evidence_refs": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "issue_type": {"type": "STRING"},
                                "columns": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING"},
                                },
                                "claim": {"type": "STRING"},
                                "claimed_values": {
                                    "type": "OBJECT",
                                    "description": "Specific facts claimed, to be checked against the actual profile.",
                                },
                            },
                            "required": ["issue_type", "columns", "claim"],
                        },
                    },
                    "confidence": {
                        "type": "NUMBER",
                        "description": "Optional, 0.0 to 1.0.",
                    },
                },
                "required": ["operation", "target_columns", "reason", "evidence_refs"],
            },
        },
    },
    "required": ["recommendations"],
}

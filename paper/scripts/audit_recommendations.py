"""
Recommendation-level audit of the real LLM's output in an experiment.

For every CONVERGE run: loads the referenced PipelineRun artifact
(data/runs/<converge_run_id>/run.json) and records each LLM
recommendation's operation, targets, Validator status and failed rules.
For AI_ONLY runs: records each recommendation from the strategy trace
(operation, targets, structurally admitted or not). Failure reasons
for failed runs are also classified.

Usage: .venv/Scripts/python.exe paper/scripts/audit_recommendations.py <experiment_id>
Writes paper/results/recommendation_audit.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def classify_error(err: str) -> str:
    e = (err or "").lower()
    if "timed out" in e or "timeout" in e:
        return "LLM timeout"
    if "failed to parse as a recommendation" in e or "not valid json" in e or "missing the required" in e:
        return "Malformed LLM output (schema)"
    if "operation(s) failed" in e:
        return "Executor fail-fast (bad op admitted)"
    if "cannot reach" in e:
        return "Provider unreachable"
    return "Other: " + (err or "")[:60]


def revalidate_with_any_match(run_json: dict, rec_json: dict) -> str:
    """Offline ablation: the shipped validator compares claimed_values
    against only the FIRST evidence item of the matching type/columns.
    This re-runs the real Validator, but with the claim check relaxed to
    'consistent with ANY matching evidence item' -- the minimal fix for
    the ambiguity defect. Returns the resulting status."""
    from converge.models.profile import ProfileResult
    from converge.models.recommendation import Recommendation
    from converge.services.validation import validator as vmod

    pr = ProfileResult.model_validate(run_json["profile_result"])
    rec = Recommendation.model_validate(rec_json)
    V = vmod.Validator(pr)
    orig = V._find_contradiction

    def any_match(r, op_def, matches):
        for ref in r.evidence_refs:
            if not ref.claimed_values:
                continue
            cands = [m for m in matches if m.issue_type == ref.issue_type
                     and set(ref.columns).issubset(set(m.columns))] or matches
            def ok(m):
                return all(m.supporting_data.get(k) is None or vmod._values_match(v, m.supporting_data[k])
                           for k, v in ref.claimed_values.items())
            if not any(ok(m) for m in cands):
                return "contradiction (all candidates)"
        stripped = r.model_copy(update={"evidence_refs": []})
        return orig(stripped, op_def, matches)

    V._find_contradiction = any_match
    return V.validate(rec).status.value


def main(exp: str) -> None:
    runs = json.loads((ROOT / "data/research/results" / exp / "runs.json").read_text(encoding="utf-8"))
    recs, failures = [], []
    for run in runs:
        strat = run["strategy"]
        if strat == "spark_only":
            continue
        if run["status"] != "succeeded":
            failures.append({"strategy": strat, "scenario": run["scenario_id"],
                             "class": classify_error(run.get("error")), "error": (run.get("error") or "")[:400]})
        if strat == "converge" and run.get("converge_run_id"):
            p = ROOT / "data/runs" / run["converge_run_id"] / "run.json"
            if not p.exists():
                continue
            pr = json.loads(p.read_text(encoding="utf-8"))
            for rr in pr.get("recommendation_records") or []:
                v = rr.get("validation") or {}
                failed = [c["rule"] for c in v.get("checks", []) if not c["passed"]
                          and c["rule"] != "forbidden_operation_rejected"]
                status = v.get("status", "unknown")
                outcome = {"approved_auto": "Approved (safe tier)", "needs_review": "Approved (review tier)",
                           "not_executable": "Finding only"}.get(status)
                if outcome is None:
                    outcome = "Rejected: " + (failed[0] if failed else "unknown")
                recs.append({"strategy": "converge", "scenario": run["scenario_id"],
                             "rep": run["repetition_index"], "operation": rr["recommendation"]["operation"],
                             "targets": rr["recommendation"]["target_columns"],
                             "parameters": rr["recommendation"]["parameters"],
                             "evidence_refs": rr["recommendation"]["evidence_refs"],
                             "status": status, "failed_rules": failed, "outcome": outcome,
                             "failed_messages": [c["message"] for c in v.get("checks", []) if not c["passed"]
                                                 and c["rule"] != "forbidden_operation_rejected"],
                             "status_any_match_fix": revalidate_with_any_match(pr, rr["recommendation"])})
        elif strat == "ai_only" and run.get("trace") and "recommendations" in run["trace"]:
            for t in run["trace"]["recommendations"]:
                recs.append({"strategy": "ai_only", "scenario": run["scenario_id"], "rep": run["repetition_index"],
                             "operation": t["operation"], "targets": t["target_columns"],
                             "executable": t["executable"], "outcome": "AI_ONLY " +
                             ("admitted" if t["executable"] else "structurally dropped")})
    summary = {
        "converge_outcomes": Counter(r["outcome"] for r in recs if r["strategy"] == "converge"),
        "converge_ops": Counter(r["operation"] for r in recs if r["strategy"] == "converge"),
        "status_transitions_under_fix": Counter(f"{r['status']} -> {r['status_any_match_fix']}"
                                                for r in recs if r["strategy"] == "converge"),
        "ai_only_ops": Counter(r["operation"] for r in recs if r["strategy"] == "ai_only"),
        "failure_classes": Counter(f"{f['strategy']}: {f['class']}" for f in failures),
    }
    out = {"experiment": exp, "summary": summary,
           "recommendations": [r for r in recs if r["strategy"] == "converge"],
           "ai_only_recommendations": [r for r in recs if r["strategy"] == "ai_only"],
           "failures": failures}
    (ROOT / "paper/results/recommendation_audit.json").write_text(json.dumps(out, indent=1, default=dict))
    print(json.dumps(summary, indent=1, default=dict))


if __name__ == "__main__":
    main(sys.argv[1])

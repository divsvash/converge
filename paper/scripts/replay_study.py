"""
E5: Record-and-replay counterfactual study.

Every recommendation list the live model actually produced (E1 CONVERGE
artefacts for seed 42; every logged call from the E4 seed sweep) is
replayed, unchanged, through a set of decision pipelines:

    AI_ONLY            structural filter, model's own order (as in E1)
    AI_ONLY+prec       structural filter, CONVERGE's precedence order
    V0                 shipped Validator (as in E1)
    V0+any             claim check accepts ANY matching evidence item
    V0+conf            NORMALIZE_CASE requires categorical (not whitespace) evidence
    V0+coerce          FILL_MISSING params checked against column type
    V0+null            null-valued optional params treated as absent
    Vfix               V0 + any + conf + coerce   (all safety fixes)
    Vfix+null          Vfix + null
    V0-evidence        ablation: evidence rules disabled
    V0-claims          ablation: claimed-value cross-check disabled
    V0-params          ablation: parameter rule disabled
    V0-dtype           ablation: dtype rule disabled

Because every pipeline sees the SAME model output, differences between
pipelines are causal effects of the decision layer alone (paired design).
Each unique (dataset, plan) is executed once with the real SparkExecutor
and scored with the strategy-agnostic ground-truth evaluator.

Writes paper/results/replay.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import converge.core.config as cfg  # noqa: E402

cfg.RUNS_DIR = ROOT / "paper" / "results" / "_replay_runs"

from converge.models.execution import ExecutionPlan, PlannedOperation  # noqa: E402
from converge.models.profile import IssueType, ProfileResult, SemanticType  # noqa: E402
from converge.models.recommendation import Recommendation  # noqa: E402
from converge.models.validation import CheckResult  # noqa: E402
from converge.services.ai.ollama_provider import OllamaProvider  # noqa: E402
from converge.services.pipeline.orchestrator import OPERATION_PRECEDENCE  # noqa: E402
from converge.services.spark.executor import SparkExecutor  # noqa: E402
from converge.services.spark.loader import load_csv  # noqa: E402
from converge.services.spark.profiler import profile_dataset  # noqa: E402
from converge.services.spark.session import get_spark_session, stop_spark_session  # noqa: E402
from converge.services.validation import validator as vmod  # noqa: E402
from experiments.scenarios.ground_truth import evaluate  # noqa: E402
from experiments.scenarios.models import GroundTruthManifest  # noqa: E402
from experiments.strategies.ai_only import _structurally_translate  # noqa: E402

EXECUTABLE = {"approved_auto", "needs_review"}


# ---------------------------------------------------------------------------
# Validator variants (subclassing; src/converge is not modified)
# ---------------------------------------------------------------------------

class VariantValidator(vmod.Validator):
    def __init__(self, pr, any_match=False, conf=False, coerce=False, null_ok=False, sem=False, skip=()):
        super().__init__(pr)
        self.any_match, self.conf, self.coerce, self.null_ok, self.skip = any_match, conf, coerce, null_ok, set(skip)
        self.sem = sem

    def _semantic_guard(self, rec, op_def, matches):
        """Deterministic 'minimal-change' guards, checkable from evidence alone:
        (1) a category mapping may only map a value to a member of its own
            normalisation group (lower+trim key);
        (2) a case normalisation must not rewrite the dominant (most frequent)
            spelling of any evidenced variant group."""
        norm = lambda s: s.strip().lower()
        if op_def.operation == "STANDARDIZE_CATEGORY":
            for k, v in (rec.parameters.get("mapping") or {}).items():
                if isinstance(k, str) and isinstance(v, str) and norm(k) != norm(v):
                    return f"mapping '{k}'->'{v}' crosses normalisation groups"
        if op_def.operation == "NORMALIZE_CASE":
            fn = {"lower": str.lower, "upper": str.upper,
                  "title": lambda s: " ".join(w.capitalize() for w in s.split(" "))}.get(
                rec.parameters.get("target_case"))
            for e in matches:
                variants = e.supporting_data.get("variants") or {}
                if variants and fn:
                    dom = max(variants, key=lambda x: (variants[x], x))
                    if fn(dom) != dom:
                        return f"case '{rec.parameters.get('target_case')}' would rewrite dominant spelling '{dom}'"
        return None

    def _pass(self, checks, rule):
        checks.append(CheckResult(rule=rule, passed=True, message="ablated"))
        return True

    def validate(self, rec):
        if self.null_ok:
            rec = rec.model_copy(update={"parameters": {k: v for k, v in rec.parameters.items() if v is not None}})
        return super().validate(rec)

    def _check_dtype_compatible(self, rec, op_def, checks):
        if "dtype" in self.skip:
            return self._pass(checks, "dtype_compatible")
        return super()._check_dtype_compatible(rec, op_def, checks)

    def _check_parameters_valid(self, rec, op_def, checks):
        if "params" in self.skip:
            return self._pass(checks, "parameters_valid")
        ok = super()._check_parameters_valid(rec, op_def, checks)
        if ok and self.coerce and op_def.operation == "FILL_MISSING":
            col = self._columns_by_name[rec.target_columns[0]] if len(rec.target_columns) == 1 else None
            numeric = col is not None and col.semantic_type in (SemanticType.INTEGER, SemanticType.FLOAT)
            strat = rec.parameters.get("strategy")
            msg = None
            if col is None:
                msg = "FILL_MISSING requires exactly one target column"
            elif strat in ("mean", "median") and not numeric:
                msg = f"strategy '{strat}' requires a numeric column"
            elif strat == "constant" and numeric:
                try:
                    float(rec.parameters.get("fill_value"))
                except (TypeError, ValueError):
                    msg = "fill_value is not coercible to the column's numeric type"
            if msg:
                checks[-1] = CheckResult(rule="parameters_valid", passed=False, message=msg)
                return False
        return ok

    def _check_evidence(self, rec, op_def, checks):
        if "evidence" in self.skip:
            self._pass(checks, "evidence_exists")
            return self._pass(checks, "evidence_supports_claim")
        return super()._check_evidence(rec, op_def, checks)

    def _find_matching_evidence(self, rec, op_def):
        m = super()._find_matching_evidence(rec, op_def)
        if self.conf and op_def.operation == "NORMALIZE_CASE":
            m = [e for e in m if e.issue_type == IssueType.CATEGORICAL_INCONSISTENCY]
        return m

    def _find_contradiction(self, rec, op_def, matches):
        if self.sem:
            g = self._semantic_guard(rec, op_def, matches)
            if g:
                return g
        if "claims" in self.skip:
            return super()._find_contradiction(rec.model_copy(update={"evidence_refs": []}), op_def, matches)
        if not self.any_match:
            return super()._find_contradiction(rec, op_def, matches)
        for ref in rec.evidence_refs:
            if not ref.claimed_values:
                continue
            cands = [e for e in matches if e.issue_type == ref.issue_type
                     and set(ref.columns).issubset(set(e.columns))] or matches
            if not any(all(e.supporting_data.get(k) is None or vmod._values_match(v, e.supporting_data[k])
                           for k, v in ref.claimed_values.items()) for e in cands):
                return "claimed values contradict every matching evidence item"
        return super()._find_contradiction(rec.model_copy(update={"evidence_refs": []}), op_def, matches)


PIPELINES = {
    "V0": {}, "V0+any": {"any_match": True}, "V0+conf": {"conf": True}, "V0+coerce": {"coerce": True},
    "V0+null": {"null_ok": True}, "Vfix": {"any_match": True, "conf": True, "coerce": True},
    "Vfix+null": {"any_match": True, "conf": True, "coerce": True, "null_ok": True},
    "V0+sem": {"sem": True},
    "Vfix+sem": {"any_match": True, "conf": True, "coerce": True, "sem": True},
    "V0-evidence": {"skip": ["evidence"]}, "V0-claims": {"skip": ["claims"]},
    "V0-params": {"skip": ["params"]}, "V0-dtype": {"skip": ["dtype"]},
}


def plan_for(pipeline: str, recs: list[Recommendation], pr: ProfileResult):
    if pipeline.startswith("AI_ONLY"):
        ops, traces = _structurally_translate(recs)
        ops = [(o.operation, list(o.target_columns), dict(o.parameters)) for o in ops]
        if pipeline == "AI_ONLY+prec":
            ops.sort(key=lambda o: OPERATION_PRECEDENCE.get(o[0], 99))
        return ops, [t.executable for t in traces]
    v = VariantValidator(pr, **PIPELINES[pipeline])
    verdicts = [v.validate(r) for r in recs]
    chosen = []
    for r, vr in zip(recs, verdicts):
        if vr.status.value in EXECUTABLE:
            params = {k: x for k, x in r.parameters.items() if x is not None} if PIPELINES[pipeline].get("null_ok") \
                else dict(r.parameters)
            chosen.append((r.operation, list(r.target_columns), params))
    chosen.sort(key=lambda o: OPERATION_PRECEDENCE.get(o[0], 99))
    return chosen, [vr.status.value for vr in verdicts]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_corpus():
    items = []
    # (a) E1 seed 42: CONVERGE artefacts hold the full recommendation objects
    runs = json.loads((ROOT / "data/research/results/exp_20261104_qwen3b/runs.json").read_text(encoding="utf-8"))
    for run in runs:
        if run["strategy"] != "converge" or not run.get("converge_run_id"):
            continue
        p = json.loads((ROOT / "data/runs" / run["converge_run_id"] / "run.json").read_text(encoding="utf-8"))
        recs = [rr["recommendation"] for rr in p.get("recommendation_records") or []]
        sid = run["scenario_id"]
        items.append({"seed": 42, "scenario": sid, "source": "E1:converge", "rep": run["repetition_index"],
                      "recs": recs, "dataset": str(ROOT / f"data/research/{sid}.csv"),
                      "manifest": str(ROOT / f"data/research/{sid}.manifest.json")})
    # (c) E2 injection plans (oracle + one hallucination), seed 42 S6
    items.append({"_inject": True})  # expanded in main(), which needs a Spark profile of S6
    # (b) E4 seed sweep: raw logged outputs (both arms are just model calls on the same prompt)
    log = ROOT / "paper/results/llm_calls.jsonl"
    parser = OllamaProvider("replay")
    counters = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            seed = int(re.match(r"seed(\d+)", c["tag"]).group(1))
            sid = c["dataset_id"].split("__")[0]
            arm = c["dataset_id"].split("__")[-1]
            try:
                recs = [r.model_dump(mode="json") for r in parser._parse_recommendations(c["raw_output"])]
            except Exception as e:  # noqa: BLE001
                items.append({"seed": seed, "scenario": sid, "source": f"E4:{arm}", "parse_error": str(e)[:200]})
                continue
            k = (seed, sid, arm)
            counters[k] = counters.get(k, 0) + 1
            base = ROOT / f"data/research/seeds/seed{seed}"
            items.append({"seed": seed, "scenario": sid, "source": f"E4:{arm}", "rep": counters[k] - 1,
                          "recs": recs, "dataset": str(base / f"{sid}.csv"),
                          "manifest": str(base / f"{sid}.manifest.json")})
    return items


def main() -> None:
    corpus = load_corpus()
    spark = get_spark_session("converge-replay")
    profiles, exec_cache, out = {}, {}, []
    clean = str(ROOT / "data/research/clean.csv")
    expanded = []
    for it in corpus:
        if not it.get("_inject"):
            expanded.append(it)
            continue
        sys.path.insert(0, str(ROOT / "paper/scripts"))
        import hallucination_injection as H
        cfg.RUNS_DIR = ROOT / "paper" / "results" / "_replay_runs"  # importing H redirects it
        s6 =str(ROOT / "data/research/S6_mixed.csv")
        profiles[s6] = profile_dataset(load_csv(spark, s6), dataset_id="replay")
        oracle = H.oracle_recommendations(profiles[s6])
        for cid, (_, _, bad) in H.hallucinations(profiles[s6]).items():
            base = [r for r in oracle if not (cid in H.SUBSTITUTES and H.SUBSTITUTES[cid](r))]
            expanded.append({"seed": 42, "scenario": "S6_mixed", "source": f"E2:{cid}", "rep": 0,
                             "recs": [r.model_dump(mode="json") for r in base + [bad]], "dataset": s6,
                             "manifest": str(ROOT / "data/research/S6_mixed.manifest.json")})
    corpus = expanded
    for it in corpus:
        if "parse_error" in it:
            out.append(it)
            continue
        if it["dataset"] not in profiles:
            profiles[it["dataset"]] = profile_dataset(load_csv(spark, it["dataset"]), dataset_id="replay")
        pr = profiles[it["dataset"]]
        manifest = GroundTruthManifest.model_validate_json(Path(it["manifest"]).read_text())
        recs = [Recommendation.model_validate(r) for r in it["recs"]]
        row = {k: it[k] for k in ("seed", "scenario", "source", "rep")}
        row["n_recs"] = len(recs)
        row["ops"] = [f"{r.operation}:{','.join(r.target_columns)}" for r in recs]
        row["pipelines"] = {}
        for pipe in ["AI_ONLY", "AI_ONLY+prec", *PIPELINES]:
            ops, verdicts = plan_for(pipe, recs, pr)
            key = (it["dataset"], json.dumps(ops, sort_keys=True))
            if key not in exec_cache:
                plan = ExecutionPlan(run_id=f"replay_{len(exec_cache)}", dataset_path=it["dataset"],
                                     operations=tuple(PlannedOperation(recommendation_id=str(i), operation=o,
                                                                       target_columns=tuple(t), parameters=p)
                                                      for i, (o, t, p) in enumerate(ops)))
                s = SparkExecutor(spark).execute(plan)
                res = {"success": s.fully_successful, "n_ops": len(ops)}
                if s.fully_successful:
                    res.update(evaluate(clean, manifest, s.output_dataset_path).model_dump(exclude={"scenario_id"}))
                exec_cache[key] = res
            row["pipelines"][pipe] = {**exec_cache[key], "verdicts": verdicts}
        out.append(row)
        print(row["seed"], row["scenario"], row["source"], row["rep"], {
            p: (round(v["repair_recall"], 2), v["originally_clean_values_modified"] + v["all_original_rows_lost"])
            if v["success"] else "FAIL" for p, v in row["pipelines"].items()}, flush=True)
    (ROOT / "paper/results/replay.json").write_text(json.dumps(
        {"pipelines": ["AI_ONLY", "AI_ONLY+prec", *PIPELINES], "unique_executions": len(exec_cache),
         "items": out}, indent=1))
    print("unique executions:", len(exec_cache))
    stop_spark_session()


if __name__ == "__main__":
    main()

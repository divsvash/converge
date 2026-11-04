"""
E3: Scalability of the CONVERGE control plane vs. data volume.

Builds synthetic HR-style datasets of N rows (replicating the clean
reference with fresh _row_ids and the S6 corruption mix injected at a
fixed ~10% rate), then measures per stage:
    - Spark profiling wall time
    - size of the AI-facing prompt (characters; tokens ~ chars/4)
    - Validator wall time for the oracle recommendation set
    - SparkExecutor wall time for the validated plan

The claim under test: the AI/validation control plane is O(columns +
evidence), independent of row count, while Spark carries the O(rows) work.

Usage: .venv/Scripts/python.exe paper/scripts/scalability.py 100 1000 10000 100000 1000000
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import converge.core.config as cfg  # noqa: E402

cfg.RUNS_DIR = ROOT / "paper" / "results" / "_scale_runs"

from converge.models.execution import ExecutionPlan, PlannedOperation  # noqa: E402
from converge.services.ai.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from converge.services.spark.executor import SparkExecutor  # noqa: E402
from converge.services.spark.loader import load_csv  # noqa: E402
from converge.services.spark.profiler import profile_dataset  # noqa: E402
from converge.services.spark.session import get_spark_session, stop_spark_session  # noqa: E402
from converge.services.validation.validator import Validator  # noqa: E402
from experiments.scenarios.generator import get_clean_rows  # noqa: E402
from hallucination_injection import oracle_recommendations  # noqa: E402

DATA = ROOT / "paper" / "results" / "_scale_data"
DEPTS = {"Engineering": ["engineering", "ENGINEERING"], "Sales": ["sales", "SALES"],
         "Marketing": ["marketing", "mARKETING"]}


def make_dataset(n: int, path: Path, rate: float = 0.10, seed: int = 7) -> None:
    rng = random.Random(seed)
    base = get_clean_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base[0].keys()))
        w.writeheader()
        for i in range(n):
            r = dict(base[i % len(base)])
            r["_row_id"] = f"R{i:08d}"
            r["name"] = f"{r['name']} {i}"
            u = rng.random()
            if u < rate / 4:
                r["name"] = "  " + r["name"]
            elif u < 2 * rate / 4:
                r["department"] = rng.choice(DEPTS[r["department"]])
            elif u < 3 * rate / 4:
                y, m, d = r["hire_date"].split("-")
                r["hire_date"] = f"{d}/{m}/{y}"
            elif u < rate:
                r["salary"] = ""
            w.writerow(r)


def main(sizes: list[int]) -> None:
    spark = get_spark_session("converge-scale")
    out = []
    for n in sizes:
        p = DATA / f"scale_{n}.csv"
        make_dataset(n, p)
        df = load_csv(spark, str(p))
        df.count()  # warm file cache / JIT, excluded from timings
        t = time.perf_counter()
        pr = profile_dataset(df, dataset_id=f"scale_{n}")
        t_prof = time.perf_counter() - t
        prompt_chars = len(SYSTEM_PROMPT) + len(build_user_prompt(pr))
        recs = oracle_recommendations(pr)
        v = Validator(pr)
        t = time.perf_counter()
        for _ in range(100):
            verdicts = [v.validate(r) for r in recs]
        t_val = (time.perf_counter() - t) / 100
        ok = [r for r, vr in zip(recs, verdicts) if vr.status.value in ("approved_auto", "needs_review")]
        plan = ExecutionPlan(run_id=f"scale_{n}", dataset_path=str(p), operations=tuple(
            PlannedOperation(recommendation_id=r.recommendation_id, operation=r.operation,
                             target_columns=tuple(r.target_columns), parameters=dict(r.parameters)) for r in ok))
        t = time.perf_counter()
        summary = SparkExecutor(spark).execute(plan)
        t_exec = time.perf_counter() - t
        row = {"rows": n, "profile_s": t_prof, "prompt_chars": prompt_chars,
               "approx_prompt_tokens": prompt_chars // 4, "evidence_items": len(pr.evidence),
               "recommendations": len(recs), "approved": len(ok), "validate_ms": t_val * 1000,
               "execute_s": t_exec, "exec_ok": summary.fully_successful}
        print(json.dumps(row), flush=True)
        out.append(row)
        dest = ROOT / "paper" / "results" / "scalability.json"
        dest.write_text(json.dumps(out, indent=1))
    stop_spark_session()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 1000, 10000, 100000])

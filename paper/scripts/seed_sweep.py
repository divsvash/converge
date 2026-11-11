"""
E4: Corruption-seed sweep with full LLM call logging.

Re-runs S3/S4/S6 (the multi-decision scenarios) under additional
corruption seeds, with the same three arms and the same model as E1.
Every LLM call is logged (raw JSON output, prompt/eval token counts,
Ollama timings) to paper/results/llm_calls.jsonl. The raw outputs are the
corpus for the record-and-replay study (replay_study.py).

The logging provider subclasses OllamaProvider and reproduces its exact
request payload (same SYSTEM_PROMPT, user prompt, JSON schema, temperature
0); nothing in src/converge is modified.

Usage (repo root, WSL):
    python paper/scripts/seed_sweep.py --seeds 7 123 --repetitions 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from converge.core.exceptions import AIProviderAPIError, AIProviderResponseError, AIProviderTimeoutError  # noqa: E402
from converge.models.profile import ProfileResult  # noqa: E402
from converge.services.ai.ollama_provider import OLLAMA_RECOMMENDATION_SCHEMA, OllamaProvider  # noqa: E402
from converge.services.ai.prompts import SYSTEM_PROMPT  # noqa: E402
from converge.services.spark.session import get_spark_session, stop_spark_session  # noqa: E402
from experiments.models.experiment import Experiment  # noqa: E402
from experiments.models.scenario import Dataset, Scenario  # noqa: E402
from experiments.runner.experiment import build_configurations  # noqa: E402
from experiments.runner.export import export_pilot  # noqa: E402
from experiments.runner.runner import ExperimentRunner  # noqa: E402
from experiments.scenarios.generator import SCENARIO_SPECS, corrupt_dataset, write_clean_fixture  # noqa: E402

LOG = ROOT / "paper" / "results" / "llm_calls.jsonl"


class LoggingOllamaProvider(OllamaProvider):
    """OllamaProvider with identical request semantics plus call logging."""

    def __init__(self, *a, tag: str = "", **kw):
        super().__init__(*a, **kw)
        self._tag = tag
        self._dataset_id = None

    def generate_recommendations(self, profile_result: ProfileResult):
        self._dataset_id = profile_result.profile.dataset_id
        return super().generate_recommendations(profile_result)

    def _call_model(self, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_prompt}],
            "stream": False,
            "format": OLLAMA_RECOMMENDATION_SCHEMA,
            "options": {"temperature": 0.0},
        }
        req = urllib.request.Request(f"{self._base_url}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as e:
            raise AIProviderTimeoutError(f"Ollama request timed out: {e}") from e
        except Exception as e:  # noqa: BLE001
            if "timed out" in str(e).lower():
                raise AIProviderTimeoutError(f"Ollama request timed out: {e}") from e
            raise AIProviderAPIError(f"Cannot reach Ollama: {e}") from e
        wall = time.perf_counter() - t0
        text = (data.get("message") or {}).get("content", "")
        rec = {"tag": self._tag, "dataset_id": self._dataset_id, "wall_s": wall,
               "prompt_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count"),
               "prompt_eval_s": (data.get("prompt_eval_duration") or 0) / 1e9,
               "eval_s": (data.get("eval_duration") or 0) / 1e9,
               "load_s": (data.get("load_duration") or 0) / 1e9,
               "user_prompt_chars": len(user_prompt), "raw_output": text}
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if not text:
            raise AIProviderResponseError("Ollama response contained no message content.")
        return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["S3_categorical", "S4_missing", "S6_mixed"])
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    args = ap.parse_args()

    research = ROOT / "data" / "research"
    clean = str(research / "clean.csv")
    write_clean_fixture(clean)
    configs = build_configurations(ollama_model=args.model, ollama_model_class="slm")
    spark = get_spark_session("converge-seed-sweep")

    for seed in args.seeds:
        exp_id = f"exp_seed{seed}_qwen3b"
        out_dir = research / "seeds" / f"seed{seed}"
        scenarios, manifests = [], {}
        for name in args.scenarios:
            spec = SCENARIO_SPECS[name].model_copy(update={"seed": seed})
            path = out_dir / f"{name}.csv"
            m = corrupt_dataset(clean, spec, str(path))
            (out_dir / f"{name}.manifest.json").write_text(m.model_dump_json(indent=2))
            manifests[spec.scenario_id] = m
            scenarios.append(Scenario(scenario_id=spec.scenario_id, name=name,
                                      dataset=Dataset.from_path(f"{name}_s{seed}", str(path))))

        runner = ExperimentRunner(
            spark=spark,
            provider_resolver=lambda cfg, s=seed: LoggingOllamaProvider(
                model_name=cfg.model, tag=f"seed{seed}:{cfg.strategy.value}"),
            clean_reference_path=clean, ground_truth_manifests=manifests)
        all_runs, rel = [], {}
        for sc in scenarios:
            runs, r = runner.run_experiment(Experiment(name=f"{exp_id}_{sc.scenario_id}", scenario=sc,
                                                       configurations=configs,
                                                       repetition_count=args.repetitions))
            all_runs += runs
            rel.update({f"{sc.scenario_id}:{k}": v for k, v in r.items()})
            for run in runs:
                gt = run.result.ground_truth if run.result else None
                print(f"seed={seed} {sc.scenario_id:16s} {run.strategy.value:10s} rep={run.repetition_index} "
                      f"{run.status.value} recall={gt.repair_recall if gt else None} "
                      f"dmg={gt.originally_clean_values_modified if gt else None} err={(run.error or '')[:60]}",
                      flush=True)
        export_pilot(exp_id, all_runs, rel, str(research / "results"),
                     {"corruption_seed": seed,
                      "scenario_checksums": {k: m.corrupted_dataset_checksum for k, m in manifests.items()}})
    stop_spark_session()


if __name__ == "__main__":
    main()

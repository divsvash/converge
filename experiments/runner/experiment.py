"""
Full experiment matrix entry point.

Supports: S1-S6 × {SPARK_ONLY, AI_ONLY+Gemini, CONVERGE+Gemini,
AI_ONLY+Ollama, CONVERGE+Ollama} × N repetitions.

Usage:
    python -m experiments.runner.experiment [options]

All model/provider/deployment decisions are explicit Configuration
fields -- nothing is inferred.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from converge.services.spark.session import get_spark_session, stop_spark_session
from experiments.models.configuration import (
    AIPlacement, AIRole, Configuration, DeploymentMode, ModelClass, StrategyName,
)
from experiments.models.experiment import Experiment
from experiments.models.scenario import Dataset, Scenario
from experiments.providers import resolve_provider
from experiments.runner.export import compute_aggregate, compute_pilot_summary, export_pilot
from experiments.runner.runner import ExperimentRunner
from experiments.scenarios.generator import SCENARIO_SPECS, corrupt_dataset, write_clean_fixture


def build_configurations(
    gemini_model: str | None = None,
    ollama_model: str | None = None,
    gemini_model_class: str = "llm",
    ollama_model_class: str = "slm",
) -> list[Configuration]:
    """Build the explicit list of configurations to compare."""
    configs: list[Configuration] = [Configuration(strategy=StrategyName.SPARK_ONLY)]

    mc_map = {"llm": ModelClass.LLM, "slm": ModelClass.SLM}

    if gemini_model:
        for strategy in (StrategyName.AI_ONLY, StrategyName.CONVERGE):
            configs.append(Configuration(
                strategy=strategy, provider="gemini", model=gemini_model,
                model_class=mc_map.get(gemini_model_class, ModelClass.LLM),
                deployment_mode=DeploymentMode.REMOTE,
                ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
            ))

    if ollama_model:
        for strategy in (StrategyName.AI_ONLY, StrategyName.CONVERGE):
            configs.append(Configuration(
                strategy=strategy, provider="ollama", model=ollama_model,
                model_class=mc_map.get(ollama_model_class, ModelClass.SLM),
                deployment_mode=DeploymentMode.LOCAL,
                ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
            ))

    return configs


def build_scenarios(
    scenario_names: list[str], research_dir: str,
) -> tuple[list[Scenario], dict[str, object], str]:
    """Generate clean reference + selected scenario fixtures. Returns
    (scenarios, ground_truth_manifests, clean_path)."""
    clean_path = str(Path(research_dir) / "clean.csv")
    write_clean_fixture(clean_path)

    scenarios: list[Scenario] = []
    gt_manifests: dict[str, object] = {}

    for name in scenario_names:
        if name not in SCENARIO_SPECS:
            raise ValueError(f"Unknown scenario '{name}'. Available: {sorted(SCENARIO_SPECS.keys())}")
        spec = SCENARIO_SPECS[name]
        out_path = str(Path(research_dir) / f"{name}.csv")
        manifest = corrupt_dataset(clean_path, spec, out_path)
        gt_manifests[spec.scenario_id] = manifest
        ds = Dataset.from_path(f"{name}_ds", out_path)
        scenarios.append(Scenario(scenario_id=spec.scenario_id, name=name, dataset=ds))

    return scenarios, gt_manifests, clean_path


def run_experiment_matrix(
    experiment_id: str,
    scenarios: list[Scenario],
    configurations: list[Configuration],
    repetition_count: int,
    clean_path: str,
    gt_manifests: dict,
    output_dir: str = "data/research/results",
) -> Path:
    """Execute the full matrix and export results."""
    spark = get_spark_session(app_name=f"converge-{experiment_id}")

    runner = ExperimentRunner(
        spark=spark,
        provider_resolver=resolve_provider,
        clean_reference_path=clean_path,
        ground_truth_manifests=gt_manifests,
    )

    all_runs = []
    all_reliability = {}

    for scenario in scenarios:
        experiment = Experiment(
            name=f"{experiment_id}_{scenario.scenario_id}",
            scenario=scenario,
            configurations=configurations,
            repetition_count=repetition_count,
        )
        runs, reliability = runner.run_experiment(experiment)
        all_runs.extend(runs)
        all_reliability.update(reliability)

        for run in runs:
            status = run.status.value
            recall = ""
            if run.result and run.result.ground_truth:
                recall = f" recall={run.result.ground_truth.repair_recall:.2f}"
            err = f" err={run.error[:50]}..." if run.error else ""
            print(f"  {scenario.scenario_id:20s} {run.strategy.value:12s} rep={run.repetition_index} {status:9s}{recall}{err}")

    metadata_extra = {
        "clean_dataset_checksum": Dataset.from_path("clean", clean_path).checksum,
        "scenario_checksums": {sid: m.corrupted_dataset_checksum for sid, m in gt_manifests.items()},
    }
    pilot_dir = export_pilot(experiment_id, all_runs, all_reliability, output_dir, metadata_extra)

    print(f"\nResults exported to: {pilot_dir}")
    summary = compute_pilot_summary(all_runs)
    succeeded = sum(1 for r in all_runs if r.status.value == "succeeded")
    failed = sum(1 for r in all_runs if r.status.value == "failed")
    print(f"Total: {len(all_runs)} attempted, {succeeded} succeeded, {failed} failed")

    stop_spark_session()
    return pilot_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="CONVERGE controlled experiment")
    parser.add_argument("--experiment-id", required=True, help="Unique experiment identifier")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_SPECS.keys()),
                        help=f"Scenario names. Available: {sorted(SCENARIO_SPECS.keys())}")
    parser.add_argument("--gemini-model", default=None, help="Gemini model name (enables remote Gemini)")
    parser.add_argument("--gemini-model-class", default="llm", choices=["llm", "slm"])
    parser.add_argument("--ollama-model", default=None, help="Ollama model name (enables local Ollama)")
    parser.add_argument("--ollama-model-class", default="slm", choices=["llm", "slm"])
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per configuration")
    parser.add_argument("--output-dir", default="data/research/results")

    args = parser.parse_args()

    if not args.gemini_model and not args.ollama_model:
        print("At least one of --gemini-model or --ollama-model is required (SPARK_ONLY is always included).")
        sys.exit(1)

    configurations = build_configurations(
        gemini_model=args.gemini_model, ollama_model=args.ollama_model,
        gemini_model_class=args.gemini_model_class, ollama_model_class=args.ollama_model_class,
    )
    scenarios, gt_manifests, clean_path = build_scenarios(args.scenarios, "data/research")

    print(f"Experiment: {args.experiment_id}")
    print(f"Scenarios: {[s.scenario_id for s in scenarios]}")
    print(f"Configurations: {len(configurations)} ({[c.strategy.value + ('/' + c.provider if c.provider else '') for c in configurations]})")
    print(f"Repetitions: {args.repetitions}")
    print(f"Expected runs: {len(scenarios) * len(configurations) * args.repetitions}")
    print()

    run_experiment_matrix(
        experiment_id=args.experiment_id,
        scenarios=scenarios,
        configurations=configurations,
        repetition_count=args.repetitions,
        clean_path=clean_path,
        gt_manifests=gt_manifests,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

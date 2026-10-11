"""
Pilot 001: first real end-to-end experiment.

S1 (whitespace) × S2 (duplicates) × S3 (categorical)
×
SPARK_ONLY / AI_ONLY-Gemini / CONVERGE-Gemini
×
3 repetitions
= 27 attempted runs.

Usage:
    python -m experiments.runner.pilot_001

Requires GEMINI_API_KEY environment variable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from converge.services.ai.gemini_provider import GeminiProvider
from converge.services.spark.session import get_spark_session, stop_spark_session
from experiments.models.configuration import (
    AIPlacement, AIRole, Configuration, DeploymentMode, ModelClass, StrategyName,
)
from experiments.models.experiment import Experiment
from experiments.models.scenario import Dataset, Scenario
from experiments.runner.export import compute_pilot_summary, export_pilot
from experiments.runner.runner import ExperimentRunner
from experiments.scenarios.generator import corrupt_dataset, write_clean_fixture
from experiments.scenarios.models import CorruptionSpec, CorruptionType

PILOT_ID = "pilot_001"
REPETITION_COUNT = 3
RESEARCH_DIR = Path("data/research")
RESULTS_DIR = Path("data/research/results")


def main() -> None:
    print("=" * 60)
    print(f"CONVERGE Pilot {PILOT_ID}")
    print("=" * 60)

    # --- Build Gemini provider once and reuse ---
    try:
        gemini = GeminiProvider()
    except Exception as e:
        print(f"\nFATAL: Cannot create GeminiProvider: {e}")
        print("Set the GEMINI_API_KEY environment variable and retry.")
        sys.exit(1)

    print(f"  Provider:       {gemini.provider_name}")
    print(f"  Model:          {gemini.model_name}")
    print(f"  Prompt version: {gemini.prompt_version}")
    print()

    # --- Prepare clean reference + corrupted scenarios ---
    clean_path = str(RESEARCH_DIR / "clean.csv")
    write_clean_fixture(clean_path)

    scenarios_and_manifests: list[tuple[Scenario, dict]] = []
    scenario_specs = {
        "S1_whitespace": CorruptionSpec(
            scenario_id="S1_whitespace", corruption_types=[CorruptionType.WHITESPACE], seed=42,
        ),
        "S2_duplicates": CorruptionSpec(
            scenario_id="S2_duplicates", corruption_types=[CorruptionType.EXACT_DUPLICATE], seed=42,
        ),
        "S3_categorical": CorruptionSpec(
            scenario_id="S3_categorical", corruption_types=[CorruptionType.CATEGORICAL_VARIANT], seed=42,
        ),
    }

    gt_manifests: dict[str, object] = {}
    scenarios: list[Scenario] = []

    for name, spec in scenario_specs.items():
        out_path = str(RESEARCH_DIR / f"{name}.csv")
        manifest = corrupt_dataset(clean_path, spec, out_path)
        gt_manifests[spec.scenario_id] = manifest
        ds = Dataset.from_path(f"{name}_ds", out_path)
        sc = Scenario(scenario_id=spec.scenario_id, name=name, dataset=ds)
        scenarios.append(sc)
        print(f"  Scenario {name}: {manifest.total_corruptions} corruptions")

    # --- Build configurations ---
    cfg_spark = Configuration(strategy=StrategyName.SPARK_ONLY)
    cfg_ai = Configuration(
        strategy=StrategyName.AI_ONLY,
        provider=gemini.provider_name, model=gemini.model_name,
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )
    cfg_converge = Configuration(
        strategy=StrategyName.CONVERGE,
        provider=gemini.provider_name, model=gemini.model_name,
        model_class=ModelClass.LLM, deployment_mode=DeploymentMode.REMOTE,
        ai_placement=AIPlacement.DURING, ai_role=AIRole.RECOMMENDATION,
    )
    configurations = [cfg_spark, cfg_ai, cfg_converge]

    # --- Run experiments ---
    spark = get_spark_session(app_name="converge-pilot-001")
    runner = ExperimentRunner(
        spark=spark,
        provider_resolver=lambda cfg: gemini,
        clean_reference_path=clean_path,
        ground_truth_manifests=gt_manifests,
    )

    all_runs = []
    all_reliability = {}

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario.scenario_id} ---")
        experiment = Experiment(
            name=f"{PILOT_ID}_{scenario.scenario_id}",
            scenario=scenario,
            configurations=configurations,
            repetition_count=REPETITION_COUNT,
        )
        runs, reliability = runner.run_experiment(experiment)
        all_runs.extend(runs)
        all_reliability.update(reliability)

        for run in runs:
            status = run.status.value
            recall = ""
            if run.result and run.result.ground_truth:
                recall = f" recall={run.result.ground_truth.repair_recall:.2f}"
            time_s = ""
            if run.result and run.result.performance:
                time_s = f" time={run.result.performance.total_wall_seconds:.1f}s"
            err = f" error={run.error[:60]}..." if run.error else ""
            print(f"  {run.strategy.value:12s} rep={run.repetition_index} {status:9s}{recall}{time_s}{err}")

    # --- Export ---
    print(f"\n--- Exporting to {RESULTS_DIR / PILOT_ID} ---")
    metadata_extra = {
        "clean_dataset_checksum": Dataset.from_path("clean", clean_path).checksum,
        "scenario_checksums": {
            sid: m.corrupted_dataset_checksum for sid, m in gt_manifests.items()
        },
    }
    pilot_dir = export_pilot(
        PILOT_ID, all_runs, all_reliability, str(RESULTS_DIR), metadata_extra=metadata_extra,
    )
    print(f"  runs.json:     {pilot_dir / 'runs.json'}")
    print(f"  summary.csv:   {pilot_dir / 'summary.csv'}")
    print(f"  metadata.json: {pilot_dir / 'metadata.json'}")

    # --- Summary ---
    print("\n--- Pilot Summary ---")
    summary = compute_pilot_summary(all_runs)
    print(f"{'Scenario':<20s} {'Strategy':<12s} {'Att':>3s} {'OK':>3s} {'Fail':>4s} {'Recall':>7s} {'FalseMod':>9s} {'StratSec':>9s} {'TotalSec':>9s}")
    for row in summary:
        recall = f"{row['mean_repair_recall']:.3f}" if row['mean_repair_recall'] is not None else "N/A"
        fm = f"{row['mean_false_modification_rate']:.4f}" if row['mean_false_modification_rate'] is not None else "N/A"
        ss = f"{row['mean_strategy_execution_wall_seconds']:.1f}" if row['mean_strategy_execution_wall_seconds'] is not None else "N/A"
        ts = f"{row['mean_total_wall_seconds']:.1f}" if row['mean_total_wall_seconds'] is not None else "N/A"
        print(f"{row['scenario_id']:<20s} {row['strategy']:<12s} {row['attempted_runs']:>3d} {row['successful_runs']:>3d} {row['failed_runs']:>4d} {recall:>7s} {fm:>9s} {ss:>9s} {ts:>9s}")

    succeeded = sum(1 for r in all_runs if r.status.value == "succeeded")
    failed = sum(1 for r in all_runs if r.status.value == "failed")
    print(f"\nTotal: {len(all_runs)} attempted, {succeeded} succeeded, {failed} failed")

    stop_spark_session()


if __name__ == "__main__":
    main()

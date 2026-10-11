"""
CONVERGE CLI.

Two commands:
    run-pipeline   -- execute a single CONVERGE pipeline on a CSV
    run-experiment -- execute controlled comparison experiments

Usage:
    python -m converge.cli.main run-pipeline --csv data.csv --provider gemini --model gemini-2.0-flash
    python -m converge.cli.main run-experiment --experiment-id exp001 --gemini-model gemini-2.0-flash --scenarios S1_whitespace S3_categorical
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="converge",
        description="CONVERGE: Validation-Guided Generative AI for Big Data Pipelines",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run-pipeline ---
    pipe_parser = subparsers.add_parser("run-pipeline", help="Run CONVERGE pipeline on a CSV dataset")
    pipe_parser.add_argument("--csv", required=True, help="Path to input CSV file")
    pipe_parser.add_argument("--provider", required=True, choices=["gemini", "ollama"])
    pipe_parser.add_argument("--model", required=True, help="Model name (e.g. gemini-2.0-flash, phi3:mini)")
    pipe_parser.add_argument("--dataset-id", default=None, help="Dataset identifier (default: filename)")
    pipe_parser.add_argument("--auto-approve", action="store_true",
                             help="Auto-approve NEEDS_REVIEW recommendations (skip interactive approval)")

    # --- run-experiment ---
    exp_parser = subparsers.add_parser("run-experiment", help="Run controlled comparison experiments")
    exp_parser.add_argument("--experiment-id", required=True, help="Unique experiment identifier")
    exp_parser.add_argument("--scenarios", nargs="+", default=None,
                            help="Scenario names (default: all S1-S6)")
    exp_parser.add_argument("--gemini-model", default=None, help="Gemini model name")
    exp_parser.add_argument("--gemini-model-class", default="llm", choices=["llm", "slm"])
    exp_parser.add_argument("--ollama-model", default=None, help="Ollama model name")
    exp_parser.add_argument("--ollama-model-class", default="slm", choices=["llm", "slm"])
    exp_parser.add_argument("--repetitions", type=int, default=3)
    exp_parser.add_argument("--output-dir", default="data/research/results")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run-pipeline":
        _run_pipeline(args)
    elif args.command == "run-experiment":
        _run_experiment(args)


def _run_pipeline(args) -> None:
    from pathlib import Path
    from converge.services.spark.session import get_spark_session, stop_spark_session
    from converge.services.pipeline.orchestrator import PipelineOrchestrator
    from converge.models.pipeline import PipelineStatus, RecommendationDecision
    from converge.models.validation import ValidationStatus

    csv_path = args.csv
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    dataset_id = args.dataset_id or Path(csv_path).stem

    # Build provider
    try:
        ai_provider = _build_provider(args.provider, args.model)
    except Exception as e:
        print(f"Error creating AI provider: {e}")
        sys.exit(1)

    print(f"CONVERGE Pipeline")
    print(f"  Dataset:  {csv_path}")
    print(f"  Provider: {ai_provider.provider_name}")
    print(f"  Model:    {ai_provider.model_name}")
    print(f"  Prompt:   {ai_provider.prompt_version}")
    print()

    spark = get_spark_session(app_name="converge-cli")
    orchestrator = PipelineOrchestrator(spark=spark, ai_provider=ai_provider)

    try:
        run = orchestrator.start_run(dataset_id=dataset_id, dataset_path=csv_path)
        run = orchestrator.run_profiling(run)
        print(f"Profiled: {run.profile_result.profile.row_count} rows, "
              f"{run.profile_result.profile.column_count} columns, "
              f"{len(run.profile_result.evidence)} evidence items")

        run = orchestrator.run_ai_analysis(run)
        run = orchestrator.run_validation(run)

        print(f"\nRecommendations ({len(run.recommendation_records)}):")
        for i, rec in enumerate(run.recommendation_records):
            status = rec.validation.status.value if rec.validation else "unknown"
            print(f"  [{i+1}] {rec.recommendation.operation} on {rec.recommendation.target_columns} "
                  f"| validation: {status}")
            print(f"      reason: {rec.recommendation.reason[:80]}")

        # Approval
        for rec in run.recommendation_records:
            val_status = rec.validation.status if rec.validation else None
            if val_status == ValidationStatus.APPROVED_AUTO:
                run = orchestrator.submit_decision(
                    run, rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER
                )
            elif val_status == ValidationStatus.NEEDS_REVIEW:
                if args.auto_approve:
                    run = orchestrator.submit_decision(
                        run, rec.recommendation.recommendation_id, RecommendationDecision.APPROVED_BY_USER
                    )
                    print(f"  [auto-approved] {rec.recommendation.operation}")
                else:
                    answer = input(f"  Approve {rec.recommendation.operation} on "
                                   f"{rec.recommendation.target_columns}? [y/N] ").strip().lower()
                    decision = (RecommendationDecision.APPROVED_BY_USER if answer == "y"
                                else RecommendationDecision.REJECTED_BY_USER)
                    run = orchestrator.submit_decision(
                        run, rec.recommendation.recommendation_id, decision
                    )
            else:
                run = orchestrator.submit_decision(
                    run, rec.recommendation.recommendation_id, RecommendationDecision.REJECTED_BY_USER
                )

        run = orchestrator.execute(run)
        print(f"\nExecution: {run.execution_summary.operations_succeeded}/{run.execution_summary.operations_attempted} operations succeeded")
        print(f"Output: {run.execution_summary.output_dataset_path}")

        run = orchestrator.run_post_execution_analysis(run)
        m = run.quality_comparison.metrics
        print(f"\nQuality comparison:")
        print(f"  Issues: {m.total_issues_before} before -> {m.total_issues_after} after")
        print(f"  Resolved: {m.resolved_count}, Improved: {m.improved_count}, "
              f"Unchanged: {m.unchanged_count}, Worsened: {m.worsened_count}, "
              f"Introduced: {m.introduced_count}")

        run = orchestrator.generate_insights(run)
        print(f"\nInsights ({len(run.insight_artifact.insights)}):")
        for insight in run.insight_artifact.insights:
            print(f"  [{insight.kind.value}] {insight.text[:100]}")

        run = orchestrator.complete_run(run)
        print(f"\nRun {run.run_id}: {run.status.value}")

    except Exception as e:
        print(f"\nPipeline error: {e}")
        sys.exit(1)
    finally:
        stop_spark_session()


def _run_experiment(args) -> None:
    # Research harness is intentionally repo-only and is not installed with
    # the production package. Give a useful error outside a source checkout.
    try:
        import experiments.scenarios.generator
    except ModuleNotFoundError as exc:
        if exc.name == "experiments" or exc.name.startswith("experiments."):
            raise SystemExit(
                "run-experiment requires the CONVERGE source checkout. "
                "Run this command from the repository root (with its "
                "experiments/ directory available)."
            ) from exc
        raise
    from experiments.scenarios.generator import SCENARIO_SPECS
    from experiments.runner.experiment import (
        build_configurations, build_scenarios, run_experiment_matrix,
    )

    scenarios_list = args.scenarios or list(SCENARIO_SPECS.keys())

    if not args.gemini_model and not args.ollama_model:
        print("Error: at least one of --gemini-model or --ollama-model is required.")
        sys.exit(1)

    configurations = build_configurations(
        gemini_model=args.gemini_model, ollama_model=args.ollama_model,
        gemini_model_class=args.gemini_model_class, ollama_model_class=args.ollama_model_class,
    )
    scenarios, gt_manifests, clean_path = build_scenarios(scenarios_list, "data/research")

    run_experiment_matrix(
        experiment_id=args.experiment_id,
        scenarios=scenarios,
        configurations=configurations,
        repetition_count=args.repetitions,
        clean_path=clean_path,
        gt_manifests=gt_manifests,
        output_dir=args.output_dir,
    )


def _build_provider(provider_name: str, model: str):
    if provider_name == "gemini":
        from converge.services.ai.gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    elif provider_name == "ollama":
        from converge.services.ai.ollama_provider import OllamaProvider
        return OllamaProvider(model_name=model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


if __name__ == "__main__":
    main()

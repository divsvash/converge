"""
Experiment execution harness.

Drives the common run procedure (see module docstring sections below
and the Phase 9 design) uniformly across all three strategies, for
every Configuration x repetition in an Experiment. Coordination only
-- no strategy-specific or metric-specific logic duplicated here; it
calls experiments.strategies.* and experiments.metrics.* exactly as
they're already defined.

COMMON RUN PROCEDURE, one repetition, in order:
    1. Resolve the AI provider for this Configuration (SPARK_ONLY
       needs none; AI_ONLY/CONVERGE fail clearly without one).
    2. Verify the Scenario's dataset SHA-256 checksum. Any drift fails
       the run BEFORE the strategy is ever invoked -- no silent
       checksum update, ever.
    3. Profile the ORIGINAL dataset using converge's own loader +
       profiler (independently of whatever the strategy does
       internally -- this is the runner's own "before" measurement,
       not borrowed from the strategy, so no strategy ever grades
       itself).
    4. Execute the selected strategy, timed.
    5. If the strategy succeeded: profile its OUTPUT dataset with the
       SAME profiler, then compare before/after with the SAME
       converge.services.comparison.comparator.QualityComparator used
       in steps 3+5 for every strategy. Timed, added to the
       evaluation-time bucket together with step 2's checksum work and
       step 3's before-profiling.
    6. Build PerformanceMetrics from the three timers (see
       experiments/metrics/performance.py for the exact boundaries).
    7. Build CostMetrics (see experiments/metrics/cost.py -- zero for
       SPARK_ONLY, None for AI_ONLY/CONVERGE under the current
       AIProvider interface).
    8. Build the final Result (quality + performance + cost).
    9. Construct the final Run, status=SUCCEEDED.

If any stage fails, the attempt is preserved as a Run with
status=FAILED, a recorded error, result=None (the existing Phase 7
Run model forbids a Result on a FAILED run -- not worked around here),
and whatever reproducibility/trace/converge_run_id information was
legitimately available by the point of failure. started_at/completed_at
are always set, even on failure -- those are the "run-level timing
legitimately available" for a failed run; the fine-grained
PerformanceMetrics breakdown lives only inside Result, which stays
None for failures by the model's own existing rule.

One failed repetition never aborts the rest of the sweep: _run_one()
catches everything internally and always returns a Run, never raises.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pyspark.sql import SparkSession

from converge.services.ai.base import AIProvider
from converge.services.comparison.comparator import QualityComparator
from converge.services.spark.loader import load_csv
from converge.services.spark.profiler import profile_dataset
from experiments.metrics.cost import compute_cost_metrics
from experiments.metrics.performance import build_performance_metrics
from experiments.metrics.quality import build_quality_metrics
from experiments.metrics.reliability import ReliabilityMetrics, compute_reliability_metrics
from experiments.models.configuration import Configuration, StrategyName
from experiments.models.experiment import Experiment
from experiments.models.result import GroundTruthResult, Result
from experiments.models.run import ReproducibilityMetadata, Run, RunStatus
from experiments.strategies.ai_only import run_ai_only
from experiments.strategies.common import Strategy, StrategyOutcome
from experiments.strategies.converge import run_converge
from experiments.strategies.spark_only import run_spark_only

# Explicit mapping, not a registry framework -- see Phase 9 instructions.
STRATEGY_REGISTRY: dict[StrategyName, Strategy] = {
    StrategyName.SPARK_ONLY: run_spark_only,
    StrategyName.AI_ONLY: run_ai_only,
    StrategyName.CONVERGE: run_converge,
}

# Caller-supplied: given a Configuration, return a ready-to-use
# AIProvider. The runner does not know how to build one itself (no
# provider registry in this phase -- see Part I).
ProviderResolver = Callable[[Configuration], AIProvider]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_of(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package_version() -> Optional[str]:
    try:
        return importlib.metadata.version("converge")
    except importlib.metadata.PackageNotFoundError:
        return None


class ExperimentRunner:
    def __init__(
        self,
        spark: SparkSession,
        provider_resolver: Optional[ProviderResolver] = None,
        clean_reference_path: Optional[str] = None,
        ground_truth_manifests: Optional[dict[str, "GroundTruthManifest"]] = None,
    ):
        """
        `clean_reference_path` and `ground_truth_manifests` are
        optional: when both are provided, the runner will also produce
        GroundTruthResult for every successful run whose scenario_id
        appears in the manifests dict. When absent (the default, and
        the state for all Phase 9 tests), the runner behaves exactly
        as before — no ground-truth evaluation is attempted.
        """
        self._spark = spark
        self._provider_resolver = provider_resolver
        self._clean_reference_path = clean_reference_path
        self._ground_truth_manifests = ground_truth_manifests or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_experiment(self, experiment: Experiment) -> tuple[list[Run], dict[str, ReliabilityMetrics]]:
        """
        Executes every Configuration x repetition_count repetition in
        `experiment`, in order. Returns (all Runs, reliability per
        configuration_id) -- both purely in-memory; nothing is
        persisted by this phase.
        """
        runs: list[Run] = []
        for configuration in experiment.configurations:
            for repetition_index in range(experiment.repetition_count):
                runs.append(self._run_one(experiment, configuration, repetition_index))

        reliability: dict[str, ReliabilityMetrics] = {}
        for configuration in experiment.configurations:
            config_runs = [r for r in runs if r.configuration_id == configuration.configuration_id]
            if config_runs:
                reliability[configuration.configuration_id] = compute_reliability_metrics(config_runs)

        return runs, reliability

    # ------------------------------------------------------------------
    # One repetition
    # ------------------------------------------------------------------

    def _run_one(self, experiment: Experiment, configuration: Configuration, repetition_index: int) -> Run:
        run_id = str(uuid.uuid4())
        attempt_start_time = _utcnow()
        attempt_start_perf = time.perf_counter()

        def fail(error: str, converge_run_id: Optional[str] = None,
                  reproducibility: Optional[ReproducibilityMetadata] = None,
                  trace: Optional[dict] = None) -> Run:
            return Run(
                run_id=run_id, experiment_id=experiment.experiment_id,
                scenario_id=experiment.scenario.scenario_id, configuration_id=configuration.configuration_id,
                strategy=configuration.strategy, repetition_index=repetition_index,
                status=RunStatus.FAILED, started_at=attempt_start_time, completed_at=_utcnow(),
                converge_run_id=converge_run_id, error=error, result=None, trace=trace,
                reproducibility=reproducibility or ReproducibilityMetadata(
                    dataset_checksum=experiment.scenario.dataset.checksum
                ),
            )

        strategy_fn = STRATEGY_REGISTRY.get(configuration.strategy)
        if strategy_fn is None:
            return fail(f"No strategy implementation registered for '{configuration.strategy.value}'.")

        # Step 1 (of this method; the Common Run Procedure's own step
        # numbering starts at checksum verification): resolve the AI
        # provider, if this strategy needs one.
        ai_provider: Optional[AIProvider] = None
        if configuration.strategy != StrategyName.SPARK_ONLY:
            if self._provider_resolver is None:
                return fail(
                    f"strategy={configuration.strategy.value} requires an AI provider, but this "
                    f"ExperimentRunner was constructed without a provider_resolver."
                )
            try:
                ai_provider = self._provider_resolver(configuration)
            except Exception as e:
                return fail(f"Provider resolution failed: {e}")

        reproducibility = ReproducibilityMetadata(
            dataset_checksum=experiment.scenario.dataset.checksum,
            provider=configuration.provider,
            model=configuration.model,
            model_class=configuration.model_class.value if configuration.model_class else None,
            deployment_mode=configuration.deployment_mode.value if configuration.deployment_mode else None,
            prompt_version=ai_provider.prompt_version if ai_provider is not None else None,
            package_version=_package_version(),
        )

        # Common Run Procedure step 1: checksum verification.
        try:
            actual_checksum = _sha256_of(experiment.scenario.dataset.path)
        except Exception as e:
            return fail(f"Failed to read dataset for checksum verification: {e}", reproducibility=reproducibility)

        if actual_checksum != experiment.scenario.dataset.checksum:
            return fail(
                f"Dataset drift detected for scenario '{experiment.scenario.scenario_id}': "
                f"expected checksum {experiment.scenario.dataset.checksum}, got {actual_checksum}. "
                f"Refusing to execute -- the Scenario no longer means the same experimental input.",
                reproducibility=reproducibility,
            )

        evaluation_seconds = 0.0

        # Common Run Procedure step 3: profile the ORIGINAL dataset
        # (independent of whatever the strategy does internally).
        eval_start = time.perf_counter()
        try:
            before_df = load_csv(self._spark, experiment.scenario.dataset.path)
            before_profile = profile_dataset(
                before_df, dataset_id=f"{experiment.scenario.scenario_id}__{run_id}__before"
            )
        except Exception as e:
            return fail(f"Failed to profile the original dataset: {e}", reproducibility=reproducibility)
        evaluation_seconds += time.perf_counter() - eval_start

        # Common Run Procedure step 4: execute the strategy, timed.
        strategy_start = time.perf_counter()
        try:
            outcome: StrategyOutcome = strategy_fn(
                experiment.scenario, configuration, run_id, self._spark, ai_provider=ai_provider
            )
        except Exception as e:
            return fail(f"Strategy raised an unexpected exception: {e}", reproducibility=reproducibility)
        strategy_seconds = time.perf_counter() - strategy_start

        if not outcome.success:
            return fail(
                outcome.error or "Strategy failed with no error message.",
                converge_run_id=outcome.converge_run_id, reproducibility=reproducibility, trace=outcome.trace,
            )

        # Common Run Procedure step 5: profile the OUTPUT dataset with
        # the SAME profiler, compare with the SAME QualityComparator,
        # and (when a ground-truth manifest is available) also evaluate
        # repair correctness. All of this is "evaluation" time.
        eval_start2 = time.perf_counter()
        try:
            after_df = load_csv(self._spark, outcome.output_dataset_path)
            after_profile = profile_dataset(
                after_df, dataset_id=f"{experiment.scenario.scenario_id}__{run_id}__after"
            )
            comparison = QualityComparator().compare(before_profile, after_profile)
        except Exception as e:
            return fail(
                f"Post-execution measurement failed: {e}",
                converge_run_id=outcome.converge_run_id, reproducibility=reproducibility, trace=outcome.trace,
            )

        gt_result: Optional[GroundTruthResult] = None
        manifest = self._ground_truth_manifests.get(experiment.scenario.scenario_id)
        if manifest is not None and self._clean_reference_path is not None:
            try:
                from experiments.scenarios.ground_truth import evaluate as gt_evaluate

                gt_metrics = gt_evaluate(self._clean_reference_path, manifest, outcome.output_dataset_path)
                gt_result = GroundTruthResult(
                    scenario_id=gt_metrics.scenario_id,
                    total_corruptions=gt_metrics.total_corruptions,
                    correctly_repaired=gt_metrics.correctly_repaired,
                    incorrectly_repaired=gt_metrics.incorrectly_repaired,
                    unrepaired=gt_metrics.unrepaired,
                    repair_recall=gt_metrics.repair_recall,
                    injected_duplicates_removed=gt_metrics.injected_duplicates_removed,
                    original_rows_lost=gt_metrics.original_rows_lost,
                    all_original_rows_lost=gt_metrics.all_original_rows_lost,
                    correctly_removed_duplicates=gt_metrics.correctly_removed_duplicates,
                    originally_clean_values_modified=gt_metrics.originally_clean_values_modified,
                    false_modification_rate=gt_metrics.false_modification_rate,
                    total_originally_clean_cells=gt_metrics.total_originally_clean_cells,
                )
            except Exception as e:
                return fail(
                    f"Ground-truth evaluation failed: {e}",
                    converge_run_id=outcome.converge_run_id, reproducibility=reproducibility, trace=outcome.trace,
                )

        evaluation_seconds += time.perf_counter() - eval_start2

        quality = build_quality_metrics(comparison)
        performance = build_performance_metrics(
            strategy_execution_wall_seconds=strategy_seconds,
            evaluation_wall_seconds=evaluation_seconds,
            total_wall_seconds=time.perf_counter() - attempt_start_perf,
        )
        cost = compute_cost_metrics(configuration.strategy)
        result = Result(quality=quality, performance=performance, cost=cost, ground_truth=gt_result)

        return Run(
            run_id=run_id, experiment_id=experiment.experiment_id,
            scenario_id=experiment.scenario.scenario_id, configuration_id=configuration.configuration_id,
            strategy=configuration.strategy, repetition_index=repetition_index,
            status=RunStatus.SUCCEEDED, started_at=attempt_start_time, completed_at=_utcnow(),
            converge_run_id=outcome.converge_run_id, result=result, trace=outcome.trace,
            reproducibility=reproducibility,
        )

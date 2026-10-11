"""
Execution/runtime performance metrics.

Measured externally and uniformly by experiments/runner/runner.py,
using time.perf_counter() (monotonic, high-resolution, immune to
wall-clock adjustments) around three distinct, non-overlapping
boundaries. converge is never modified to insert timers -- every
measurement here wraps a call INTO converge/strategy code from the
outside.

EXACT TIMING BOUNDARIES (the runner is the only caller of
build_performance_metrics(); see experiments/runner/runner.py for
where each perf_counter() call actually sits):

    strategy_execution_wall_seconds
        Starts immediately before calling the selected strategy
        function (experiments.strategies.*.run_*()) and ends
        immediately after it returns (or raises). Includes
        everything that strategy does internally -- its own
        profiling, its own AI call (if any), its own Spark execution
        -- since all of that genuinely IS "executing the strategy".

    evaluation_wall_seconds
        The sum of: (a) dataset checksum verification, (b) the
        runner's OWN before-profiling of the original dataset (step 3
        of the common run procedure), and (c) the runner's OWN
        after-profiling of the strategy's output dataset plus the
        QualityComparator comparison (step 5). This is the runner's
        independent measurement work, deliberately separate from
        strategy_execution_wall_seconds so a strategy's own internal
        profiling (e.g. CONVERGE profiles the same dataset itself, to
        decide what to do) is not confused with the external,
        uniform measurement every strategy is subjected to
        identically.

    total_wall_seconds
        Wall time for the entire Run attempt, from the moment the
        runner begins working on that repetition to the moment the
        Run record is finalized. This will be slightly larger than
        strategy_execution_wall_seconds + evaluation_wall_seconds,
        since it also includes runner-level bookkeeping (constructing
        objects, provider resolution, building the final Run) that
        isn't attributed to either narrower bucket.

HONESTY ABOUT WHAT THIS DOES NOT ISOLATE: Spark runs as a process-wide
singleton JVM session (converge.services.spark.session), and Spark's
lazy evaluation means a "quick" call can trigger deferred computation
whose actual cost lands on whatever later call forces materialization
(e.g. a .count() or .collect()). JVM/Spark warm-up effects on the
first repetition of a sweep are real and not corrected for here. These
timings measure wall-clock elapsed time around well-defined code
boundaries -- they do NOT claim to isolate "pure" CPU-bound strategy
cost from Spark scheduling/warm-up noise. Repeated runs (reliability
aggregation) are the intended way to see this variance, not a reason
to attempt artificial isolation here.
"""
from __future__ import annotations

from typing import Optional

from experiments.models.result import PerformanceMetrics


def build_performance_metrics(
    strategy_execution_wall_seconds: Optional[float],
    evaluation_wall_seconds: Optional[float],
    total_wall_seconds: Optional[float],
) -> PerformanceMetrics:
    return PerformanceMetrics(
        strategy_execution_wall_seconds=strategy_execution_wall_seconds,
        evaluation_wall_seconds=evaluation_wall_seconds,
        total_wall_seconds=total_wall_seconds,
    )

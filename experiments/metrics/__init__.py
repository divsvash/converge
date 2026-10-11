"""
Metrics computation for experiment results.

Split into four modules because the research requirements name four
genuinely orthogonal measurement axes -- unlike the deferred models/
split, these don't depend on an undesigned taxonomy; "data quality
improved" and "it took 40 seconds" and "it cost $0.02 in API calls"
and "results varied across repetitions" are different questions no
matter how the surrounding domain model ends up shaped.

    quality.py      -- build_quality_metrics(): maps converge's own
                        QualityComparison into QualityMetrics, field
                        by field, by name. Never recomputed
                        independently.
    performance.py  -- build_performance_metrics(): three wall-clock
                        timing boundaries (strategy execution /
                        evaluation / total), measured externally by
                        the runner around converge/strategy calls.
    cost.py         -- compute_cost_metrics(): zero for SPARK_ONLY
                        (semantically correct -- no AI call is ever
                        made), None for AI_ONLY/CONVERGE (the current
                        AIProvider interface exposes no token/cost
                        data -- None means unknown, not zero).
    reliability.py  -- compute_reliability_metrics(): aggregate
                        attempted/successful/failed counts and
                        success_rate over sibling Runs sharing
                        (experiment_id, scenario_id, configuration_id),
                        plus an optional AI_ONLY-only recommendation-
                        consistency measure where the trace data
                        genuinely supports it.

experiments/runner/runner.py is the only caller of all four modules.
"""

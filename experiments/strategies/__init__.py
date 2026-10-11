"""
Baseline and proposed pipeline strategies for controlled comparison.

Three strategies, matching the research's three comparison arms:
    spark_only.run_spark_only   -- deterministic, non-AI baseline
    ai_only.run_ai_only         -- AI-directed cleaning without CONVERGE's
                                    evidence-validation / safety architecture
    converge.run_converge       -- thin adapter around the real
                                    PipelineOrchestrator

All three share the uniform callable surface defined in common.py
(`Strategy` Protocol, `StrategyOutcome` return type) so a future runner
(Phase 9, not implemented here) can invoke any of them identically.
`StrategyOutcome` carries only execution facts -- success/failure,
output dataset path, error, strategy-specific trace, and (CONVERGE
only) a reference to the underlying PipelineRun. No strategy computes
its own quality/performance/cost metrics; that is Phase 9's common
measurement layer, applied identically and externally to every
strategy's before/output dataset so no strategy grades its own work.
"""

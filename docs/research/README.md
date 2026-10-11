# Research Design

## Research Questions
1. Does AI improve data-cleaning outcomes compared to deterministic-only processing?
2. Does CONVERGE's validation boundary reduce hallucination-driven errors compared to unvalidated AI?
3. How do different model classes (LLM vs SLM) and deployment modes (remote vs local) affect repair quality, reliability, and performance?

## Experimental Arms
- A: SPARK_ONLY (deterministic baseline — no AI)
- B: AI_ONLY (AI decisions without validator/evidence gate)
- C: CONVERGE (AI + validator + evidence-gated execution)

## Controlled Variables
- Same dataset/scenario across all strategies in a comparison
- Same profiler, comparator, and ground-truth evaluator
- Same operation catalog and executor
- Same prompt and evidence representation for AI strategies

## Scenarios (S1-S6)
Deterministic corruption injection with known ground truth. See `experiments/scenarios/generator.py` for exact algorithms.

## Metrics
- **Quality**: deterministic before/after issue counts (from converge's own comparator)
- **Ground truth**: repair_recall, false_modification_rate, per-corruption accuracy
- **Performance**: wall-clock timing with three documented boundaries
- **Cost**: token usage where available (None for unknown, 0 for SPARK_ONLY)
- **Reliability**: success_rate + recommendation_consistency across repetitions

## Reproducibility Metadata
Per-run: dataset checksum, provider, model, model_class, deployment_mode, prompt_version, package_version. No fabricated seed/temperature.

## Limitations
- Small fixture (15 rows) — correctness first, scale later
- No ground truth for dtype/outlier corruptions
- Cost data unavailable under current AIProvider interface
- 3 repetitions is a pilot, not a final experiment

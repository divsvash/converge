# Architecture Decision Records

## ADR-001: AI Recommends, Spark Executes
AI proposes structured recommendations against Spark-produced evidence. The Validator gates. Spark executes. AI never touches data directly.

## ADR-002: Explicit Operation Catalog
8 P0 operations with declared safety tiers, parameter schemas, and required evidence types. Lives in `models/operation_catalog.py` (reference data, not service logic) because three independent services consume it.

## ADR-003: Validator Independence
The Validator has no dependency on AIProvider, SparkExecutor, or the orchestrator. It can be tested in complete isolation against hand-built recommendations and evidence.

## ADR-004: Human Approval Boundary
The pipeline hard-pauses at `AWAITING_APPROVAL`. The experiment layer automates this with a documented approval policy (`default_approval_policy`) that only approves what the Validator already deemed executable — it can only narrow, never widen.

## ADR-005: Experiment Layer Outside Converge
`experiments/` is not part of the installed `converge` package. `converge` never imports `experiments`. Baseline strategies (deliberately less safe than CONVERGE) live here to prevent blurring the safety guarantees the paper demonstrates.

## ADR-006: ModelClass Separate From DeploymentMode
`LLM`/`SLM` is a size classification. `REMOTE`/`LOCAL` is a deployment characteristic. Neither is inferred from the model name — both are explicit Configuration fields.

## ADR-007: Same Evaluator Across All Strategies
Every strategy's output is measured by the same external profiler/comparator path and the same ground-truth evaluator. No strategy computes its own quality — Phase 9's fairness rule, enforced structurally.

## ADR-008: Ground-Truth Controlled Corruption
Clean reference + deterministic corruption generator + manifest recording every injected change. Evaluator joins on stable `_row_id` column. This is what makes "correct repair" measurable rather than assumed.

## ADR-009: No Overall Quality Score
Before/after issue counts are observed signals, not proof of correctness. Ground-truth repair_recall is the correctness measure. They answer different questions and are never merged.

## ADR-010: Provider Resolution Without a Registry
A simple conditional factory (`experiments/providers.py`) maps (provider, deployment_mode) to the correct provider class. No plugin system needed for two providers.

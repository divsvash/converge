# Architecture

## Dependency Direction

```
converge.core            (leaf: stdlib only)
       ▲
converge.models           (+ operation_catalog.py)
       ▲
converge.services.*       (spark, validation, ai, comparison, insights)
       ▲
converge.services.pipeline (orchestrator — depends on interfaces, not concrete providers)
       ▲
converge.storage           (RunStore)
       ▲
converge.cli               (composition root only)

═══════ hard boundary — one direction only ═══════

experiments                (imports converge as a library; converge NEVER imports experiments)
```

## Pipeline State Machine

```
INGESTED → PROFILED → AI_ANALYZED → VALIDATED → AWAITING_APPROVAL
→ EXECUTING → CLEANED → STATS_COMPUTED → INSIGHTS_GENERATED → COMPLETED
                                                            (or FAILED at any stage)
```

Hard pause at `AWAITING_APPROVAL` — nothing executes without explicit user approval (or an experiment's documented approval policy).

## Safety Boundaries

1. **AI sees only bounded evidence** — `ProfileResult` (aggregate stats + typed evidence items), never raw dataset rows or file paths.
2. **Validator is independent** — deterministic, evidence-based, no AI dependency. Checks operation validity, evidence support, parameter correctness.
3. **ExecutionPlan is immutable** — the executor receives a frozen plan; it cannot reorder, modify, or add operations.
4. **Executor re-validates parameters** — defense-in-depth; even if a malformed recommendation somehow reached execution, the executor's own parameter checks would catch it.
5. **No code execution** — AI output is structured data (operation name + target columns + parameters), never executable code.

## Experiment Layer

The experiment layer (`experiments/`) sits entirely outside the installed `converge` package. It invokes converge as a library through its public interfaces (AIProvider abstraction, PipelineOrchestrator constructor injection). Baseline strategies (SPARK_ONLY, AI_ONLY) live here because they represent the *absence* of CONVERGE's safety guarantees — mixing them into the production package would blur the very boundaries the paper demonstrates.

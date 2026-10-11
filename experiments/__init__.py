"""
CONVERGE experiment harness.

This package is OUTSIDE the production `converge` system. It exists to
configure and invoke `converge` for controlled comparison research --
it must never modify `converge`'s internals, and `converge` must never
import anything from here.

DEPENDENCY RULE (structural, not just convention):
    experiments -> converge   (allowed: experiments imports converge as a library)
    converge -> experiments   (FORBIDDEN, always)

WHY THIS PACKAGE EXISTS
CONVERGE's own orchestrator (converge.services.pipeline.orchestrator)
implements exactly one strategy: AI-assisted, validator-gated,
Spark-executed data cleaning. That IS the thing objective 2 of the
research is trying to demonstrate is safe. It is not itself a
framework for comparing alternatives to itself.

The research question is comparative:
    A. Traditional Spark-only processing (no AI at all)
    B. Pure AI-based processing (no Spark determinism, no validator)
    C. CONVERGE's proposed hybrid architecture

...evaluated across model class (LLM/SLM/local), provider, AI
placement (before/during/after Spark), AI role (recommendation/
validation/transformation/insight), and whether AI should be used at
all for a given stage -- all while holding dataset, scenario, and
measurement methodology constant.

Baselines A and B are NOT things CONVERGE's orchestrator can express
(A has no AI step at all; B has no validator/executor step at all).
Their implementations belong here, in experiments/strategies/, NOT as
new modes bolted onto converge.services.pipeline.orchestrator. This is
the concrete meaning of "CONVERGE should not contain experiment-
specific logic": alternative/baseline strategies -- including ones
that are deliberately less safe or less rigorous, since that's the
point of comparison -- are experiment concerns, never modifications to
the production package.

STATUS: skeleton only. No experiment behavior is implemented yet.
See docs/decisions/ for the record of which design questions (the
Experiment/Configuration/Scenario/Run/Result split in particular) are
still open and deliberately deferred.
"""

"""
Common strategy contract.

The smallest shared shape genuinely required for a future runner
(Phase 9, not implemented here) to invoke any of the three strategies
identically and interpret their result the same way.

`StrategyOutcome` intentionally contains ONLY execution facts -- no
quality/performance/cost metrics. Measuring what happened is Phase 9's
job (a common measurement layer, applied externally and identically to
every strategy's output dataset, so no strategy grades its own work --
see each strategy module's docstring for how this is enforced
structurally, not just by convention). Each strategy's own
strategy-specific traceability (see spark_only.DeterministicDecision,
ai_only.RecommendationTrace) flows through `StrategyOutcome.trace` as a
plain, JSON-serializable dict, so this contract itself never needs to
know what shape a particular strategy's trace takes.

`Strategy` is a `typing.Protocol` (not an ABC, no class hierarchy, no
registration) -- used only because a future runner genuinely benefits
from one type hint that works for all three strategy callables, not
because an abstraction is architecturally fashionable. Each strategy
is a plain function; none of them subclass anything, and this module
introduces no framework, no dependency-injection container, and no
registry.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field
from pyspark.sql import SparkSession

from converge.services.ai.base import AIProvider
from experiments.models.configuration import Configuration
from experiments.models.scenario import Scenario


class StrategyOutcome(BaseModel):
    """
    Execution facts only. No quality/performance/cost fields here --
    that is Phase 9's common measurement layer, applied identically
    and externally to every strategy's output dataset.
    """
    success: bool
    output_dataset_path: Optional[str] = Field(
        default=None, description="Set only when success is True."
    )
    error: Optional[str] = Field(
        default=None, description="Set only when success is False."
    )
    converge_run_id: Optional[str] = Field(
        default=None,
        description="Only ever set by the CONVERGE strategy -- a reference to the underlying "
                    "converge.models.pipeline.PipelineRun.run_id, set whenever a PipelineRun "
                    "was actually created, whether the run succeeded or failed (failure is "
                    "research data too).",
    )
    trace: Optional[dict[str, Any]] = Field(
        default=None,
        description="Strategy-specific traceability detail, serialized as a plain dict. See "
                    "each strategy module's own trace model for its real, typed shape.",
    )


class Strategy(Protocol):
    """
    Uniform callable surface every strategy implements -- a plain
    function, not a class. `ai_provider` is accepted by all three for
    signature uniformity; spark_only's implementation never reads it
    (see its own module docstring for how that is enforced
    structurally), since SPARK_ONLY never invokes AI.
    """
    def __call__(
        self,
        scenario: Scenario,
        configuration: Configuration,
        run_id: str,
        spark: SparkSession,
        ai_provider: Optional[AIProvider] = None,
    ) -> StrategyOutcome:
        ...

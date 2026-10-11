"""
Run -- one attempted execution of one Configuration against one
Scenario, at a given repetition_index.

A Run is always representable once execution is attempted, whether it
succeeds or fails: failure is research data (reliability/error-rate
are explicit metrics this project cares about -- see
experiments/metrics/reliability.py's placeholder), so RunStatus.FAILED
is a first-class outcome, not something callers are expected to
discard or special-case away.

No execution behavior lives here -- this is the record of what
happened, produced by a (not yet implemented) runner/strategy, not the
thing that produces it. Unlike Dataset/Scenario/Configuration (true
value objects, frozen), Run is left mutable, mirroring the same
convention converge.models.pipeline.PipelineRun already uses: a run
is naturally built up over its own lifecycle (started -> completed or
failed) by whatever drives it.

PHASE 9 EXTENSION NOTE: a `trace` field was added below, discovered
necessary while implementing the runner (experiments/runner/runner.py).
experiments.strategies.common.StrategyOutcome already carried a
strategy-specific `trace` dict (spark_only's deterministic decisions,
ai_only's per-recommendation executability), but Phase 7's Run model
had nowhere to carry it forward -- without this field, that
traceability would be silently discarded the moment the runner
finished building a Run. The field mirrors StrategyOutcome.trace
exactly (same optional dict-of-Any shape) and is populated whenever a
strategy actually ran (success or failure), left None when a Run
failed before ever reaching strategy execution (e.g. checksum drift).

LIFECYCLE NOTE: RunStatus intentionally remains SUCCEEDED | FAILED
only (no PENDING/RUNNING added in Phase 9). The runner never exposes
or returns an in-progress Run to any caller -- there is no polling, no
concurrent observer, nothing in this phase's actual behavior that
needs an intermediate state to be representable. A Run is only ever
constructed once, in its terminal state, after the full attempt (steps
1-9 or an early failure) has concluded. Adding states nothing reads
would be exactly the "generic state-machine framework" complexity the
Phase 9 instructions warn against building prematurely.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from experiments.models.configuration import StrategyName
from experiments.models.result import Result


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReproducibilityMetadata(BaseModel):
    """
    Everything about a Run that can be known without modifying
    src/converge -- i.e. strictly what AIProvider/converge already
    expose, never fabricated.

    Deliberately does NOT include a sampling seed or temperature:
    converge.services.ai.base.AIProvider exposes only provider_name,
    model_name, and prompt_version as identity properties -- no
    sampling-parameter property exists on the interface (GeminiProvider
    happens to hardcode temperature=0.0 internally, but that is not
    something the AIProvider interface reports, so it is not
    recorded here either). Recording a seed/temperature value here
    would mean inventing data the system does not actually report,
    which is exactly what this field set must not do.
    """
    dataset_checksum: str
    provider: Optional[str] = None
    model: Optional[str] = None
    model_class: Optional[str] = None
    deployment_mode: Optional[str] = None
    prompt_version: Optional[str] = None
    package_version: Optional[str] = Field(
        default=None,
        description="converge package version (e.g. via importlib.metadata.version('converge')), "
                    "populated by the runner -- not computed here, since this model has no behavior.",
    )


class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str
    scenario_id: str
    configuration_id: str
    strategy: StrategyName
    repetition_index: int = Field(ge=0)

    status: RunStatus
    started_at: datetime
    completed_at: Optional[datetime] = None

    converge_run_id: Optional[str] = Field(
        default=None,
        description="Foreign reference to a persisted converge.models.pipeline.PipelineRun.run_id. "
                    "Only ever applicable when strategy=CONVERGE -- spark_only and ai_only have no "
                    "PipelineRun equivalent.",
    )
    error: Optional[str] = None
    result: Optional[Result] = None
    trace: Optional[dict[str, Any]] = Field(
        default=None,
        description="Mirrors experiments.strategies.common.StrategyOutcome.trace -- strategy-specific "
                    "traceability, carried forward from the strategy's own outcome. None if the Run "
                    "failed before the strategy ever ran.",
    )

    reproducibility: ReproducibilityMetadata

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Run":
        if self.converge_run_id is not None and self.strategy != StrategyName.CONVERGE:
            raise ValueError(
                f"converge_run_id may only be set when strategy=CONVERGE; got strategy={self.strategy.value}."
            )

        if self.status == RunStatus.FAILED:
            if not self.error:
                raise ValueError("A FAILED run must include error information.")
            if self.result is not None:
                raise ValueError("A FAILED run must not carry a Result -- nothing succeeded to measure.")

        if self.status == RunStatus.SUCCEEDED and self.error is not None:
            raise ValueError("A SUCCEEDED run must not carry error information.")

        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at.")

        return self

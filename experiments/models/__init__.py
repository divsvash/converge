"""
Experiment domain models.

Implements the minimum vocabulary needed to describe a controlled
comparison:

    Dataset, Scenario   -- the fixed experimental input side
    Configuration        -- the treatment (strategy + AI axes)
    Experiment            -- the comparison definition (one Scenario,
                              several Configurations, a repetition count)
    Run                   -- one attempted execution
    Result                -- typed, not-yet-computed measurement containers

Explicitly rejected during design (recorded here so these aren't
silently reintroduced later): a generic `Provider` class or registry,
a generic `Metric` class, a `Repetition` class (repetition_index is
just a field on Run), a generic `Baseline` class (StrategyName already
represents the experimental arms), and automatic inference of
ModelClass or DeploymentMode from a model/provider name (both are
always explicit fields).

No execution behavior anywhere in this package -- strategies, the
runner, and metric computation are implemented in later phases. These
models also have no dependency on `converge` at all (verified by
tests/experiments/test_dependency_boundary.py): everything they need
to represent is expressible without importing converge's code.
"""
from experiments.models.configuration import (
    AIPlacement,
    AIRole,
    Configuration,
    DeploymentMode,
    ModelClass,
    StrategyName,
)
from experiments.models.experiment import Experiment
from experiments.models.result import CostMetrics, GroundTruthResult, PerformanceMetrics, QualityMetrics, Result
from experiments.models.run import ReproducibilityMetadata, Run, RunStatus
from experiments.models.scenario import Dataset, Scenario

__all__ = [
    "AIPlacement",
    "AIRole",
    "Configuration",
    "CostMetrics",
    "Dataset",
    "DeploymentMode",
    "Experiment",
    "ModelClass",
    "PerformanceMetrics",
    "GroundTruthResult",
    "QualityMetrics",
    "ReproducibilityMetadata",
    "Result",
    "Run",
    "RunStatus",
    "Scenario",
    "StrategyName",
]

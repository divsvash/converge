"""
Reliability / reproducibility metrics.

Aggregate data over SIBLING Runs sharing
(experiment_id, scenario_id, configuration_id) -- never attached to an
individual Run or Result (see experiments/models/result.py's own
docstring for why). compute_reliability_metrics() is the only function
here; it takes a list of Runs and enforces that they actually all
share that triple before computing anything, so a caller can't
accidentally mix groups.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from pydantic import BaseModel, Field

from experiments.models.configuration import StrategyName
from experiments.models.run import Run, RunStatus


class ReliabilityMetrics(BaseModel):
    experiment_id: str
    scenario_id: str
    configuration_id: str

    attempted_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float = Field(description="successful_runs / attempted_runs; 0.0 if attempted_runs is 0.")

    recommendation_consistency: Optional[float] = Field(
        default=None,
        description="Average pairwise Jaccard similarity of the (operation, target_columns) "
                    "recommendation-identity set across successful AI_ONLY runs in this group. "
                    "None whenever it cannot be honestly computed -- see "
                    "_compute_recommendation_consistency's docstring for exactly when that is.",
    )


def compute_reliability_metrics(runs: list[Run]) -> ReliabilityMetrics:
    if not runs:
        raise ValueError("compute_reliability_metrics requires at least one Run.")

    experiment_ids = {r.experiment_id for r in runs}
    scenario_ids = {r.scenario_id for r in runs}
    configuration_ids = {r.configuration_id for r in runs}
    if len(experiment_ids) != 1 or len(scenario_ids) != 1 or len(configuration_ids) != 1:
        raise ValueError(
            "compute_reliability_metrics requires every Run to share the same "
            "(experiment_id, scenario_id, configuration_id); got "
            f"{len(experiment_ids)} experiment_id(s), {len(scenario_ids)} scenario_id(s), "
            f"{len(configuration_ids)} configuration_id(s)."
        )

    attempted = len(runs)
    succeeded = sum(1 for r in runs if r.status == RunStatus.SUCCEEDED)
    failed = attempted - succeeded

    return ReliabilityMetrics(
        experiment_id=runs[0].experiment_id,
        scenario_id=runs[0].scenario_id,
        configuration_id=runs[0].configuration_id,
        attempted_runs=attempted,
        successful_runs=succeeded,
        failed_runs=failed,
        success_rate=(succeeded / attempted) if attempted else 0.0,
        recommendation_consistency=_compute_recommendation_consistency(runs),
    )


def _compute_recommendation_consistency(runs: list[Run]) -> Optional[float]:
    """
    Deliberately conservative, per the Phase 9 instructions: implement
    only if the strategy traces provide sufficient uniform information
    to compute it honestly; defer otherwise rather than guess.

    Computed ONLY when every run in the group has strategy=AI_ONLY:
    ai_only's trace (experiments.strategies.ai_only.RecommendationTrace,
    carried onto Run.trace) is a flat, uniformly-shaped
    list[{"operation": str, "target_columns": list[str], ...}] -- a
    direct, reliable source of (operation, target_columns) identity
    with zero extra I/O.

    NOT computed for CONVERGE groups: converge.py's strategy trace
    deliberately does NOT duplicate recommendation detail (it is just
    an explanatory note pointing at converge_run_id, by design -- see
    that module's docstring). Extracting the same identity information
    would require an additional RunStore load of the referenced
    PipelineRun, which this function was not given access to and which
    this phase does not add. Deferred honestly, not guessed.

    NOT computed for SPARK_ONLY groups: its decisions are deterministic
    by construction (Phase 8's own tests prove identical input always
    produces identical decisions), so a variability/consistency metric
    is not a meaningful question for it -- there is nothing to measure
    variation in.

    Within the eligible AI_ONLY runs: only SUCCEEDED runs with a
    well-formed trace are used (a FAILED run's trace, if present, may
    reflect a partial/aborted attempt and is not a fair comparison
    point). Returns None if fewer than 2 such runs exist -- a
    consistency measure needs at least a pair to compare.
    """
    if any(r.strategy != StrategyName.AI_ONLY for r in runs):
        return None

    identity_sets: list[frozenset[tuple[str, tuple[str, ...]]]] = []
    for run in runs:
        if run.status != RunStatus.SUCCEEDED:
            continue
        if not run.trace or "recommendations" not in run.trace:
            continue
        recs = run.trace["recommendations"]
        identity_set = frozenset(
            (rec["operation"], tuple(rec["target_columns"]))
            for rec in recs
            if isinstance(rec, dict) and "operation" in rec and "target_columns" in rec
        )
        identity_sets.append(identity_set)

    if len(identity_sets) < 2:
        return None

    similarities = [_jaccard(a, b) for a, b in combinations(identity_sets, 2)]
    return sum(similarities) / len(similarities)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0  # both empty -- perfectly consistent (consistently recommended nothing)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)

"""
Pilot result export.

Writes three files into a pilot directory:
    runs.json       — complete machine-readable run records
    summary.csv     — one row per attempted Run, flattened
    metadata.json   — pilot identity, scenarios, configurations, etc.

No database. No overwriting an existing completed pilot directory.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from experiments.metrics.reliability import ReliabilityMetrics
from experiments.models.run import Run, RunStatus


def export_pilot(
    pilot_id: str,
    runs: list[Run],
    reliability: dict[str, ReliabilityMetrics],
    output_dir: str,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Write runs.json, summary.csv, metadata.json into
    `output_dir/pilot_id/`. Raises if that directory already exists
    (collision protection — do not silently overwrite).
    """
    pilot_dir = Path(output_dir) / pilot_id
    if pilot_dir.exists():
        raise FileExistsError(
            f"Pilot directory '{pilot_dir}' already exists. "
            f"Choose a different pilot_id or remove the existing directory."
        )
    pilot_dir.mkdir(parents=True)

    _write_runs_json(runs, pilot_dir / "runs.json")
    _write_summary_csv(runs, pilot_dir / "summary.csv")
    write_aggregate_csv(compute_aggregate(runs), pilot_dir / "aggregate.csv")
    _write_metadata_json(pilot_id, runs, reliability, pilot_dir / "metadata.json", metadata_extra)

    return pilot_dir


def _write_runs_json(runs: list[Run], path: Path) -> None:
    data = [json.loads(r.model_dump_json()) for r in runs]
    path.write_text(json.dumps(data, indent=2, default=str))


def _write_summary_csv(runs: list[Run], path: Path) -> None:
    fieldnames = [
        "run_id", "scenario_id", "strategy", "repetition_index", "success",
        "provider", "model", "prompt_version",
        "total_corruptions", "correctly_repaired", "incorrectly_repaired",
        "unrepaired", "repair_recall",
        "originally_clean_values_modified", "false_modification_rate",
        "injected_duplicates_removed", "original_rows_lost",
        "all_original_rows_lost", "correctly_removed_duplicates",
        "issues_before", "issues_after",
        "resolved", "improved", "unchanged", "worsened", "introduced", "unverified",
        "strategy_execution_wall_seconds", "evaluation_wall_seconds", "total_wall_seconds",
        "input_tokens", "output_tokens", "total_tokens", "monetary_cost", "currency",
        "error",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            row = _flatten_run(run)
            writer.writerow(row)


def _flatten_run(run: Run) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "scenario_id": run.scenario_id,
        "strategy": run.strategy.value,
        "repetition_index": run.repetition_index,
        "success": run.status == RunStatus.SUCCEEDED,
        "provider": run.reproducibility.provider,
        "model": run.reproducibility.model,
        "prompt_version": run.reproducibility.prompt_version,
        "error": run.error,
    }

    if run.result and run.result.ground_truth:
        gt = run.result.ground_truth
        row.update({
            "total_corruptions": gt.total_corruptions,
            "correctly_repaired": gt.correctly_repaired,
            "incorrectly_repaired": gt.incorrectly_repaired,
            "unrepaired": gt.unrepaired,
            "repair_recall": gt.repair_recall,
            "originally_clean_values_modified": gt.originally_clean_values_modified,
            "false_modification_rate": gt.false_modification_rate,
            "injected_duplicates_removed": gt.injected_duplicates_removed,
            "original_rows_lost": gt.original_rows_lost,
            "all_original_rows_lost": gt.all_original_rows_lost,
            "correctly_removed_duplicates": gt.correctly_removed_duplicates,
        })

    if run.result and run.result.quality:
        q = run.result.quality
        row.update({
            "issues_before": q.total_issues_before,
            "issues_after": q.total_issues_after,
            "resolved": q.resolved_count,
            "improved": q.improved_count,
            "unchanged": q.unchanged_count,
            "worsened": q.worsened_count,
            "introduced": q.introduced_count,
            "unverified": q.unverified_count,
        })

    if run.result and run.result.performance:
        p = run.result.performance
        row.update({
            "strategy_execution_wall_seconds": p.strategy_execution_wall_seconds,
            "evaluation_wall_seconds": p.evaluation_wall_seconds,
            "total_wall_seconds": p.total_wall_seconds,
        })

    if run.result and run.result.cost:
        c = run.result.cost
        row.update({
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            "total_tokens": c.total_tokens,
            "monetary_cost": c.monetary_cost,
            "currency": c.currency,
        })

    return row


def _write_metadata_json(
    pilot_id: str,
    runs: list[Run],
    reliability: dict[str, ReliabilityMetrics],
    path: Path,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    scenarios = sorted({r.scenario_id for r in runs})
    strategies = sorted({r.strategy.value for r in runs})
    config_ids = sorted({r.configuration_id for r in runs})

    sample_repro = next((r.reproducibility for r in runs if r.reproducibility.provider), None)

    metadata = {
        "pilot_id": pilot_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "strategies": strategies,
        "configuration_ids": config_ids,
        "repetition_count": max((r.repetition_index for r in runs), default=0) + 1,
        "total_attempted_runs": len(runs),
        "total_succeeded": sum(1 for r in runs if r.status == RunStatus.SUCCEEDED),
        "total_failed": sum(1 for r in runs if r.status == RunStatus.FAILED),
        "provider": sample_repro.provider if sample_repro else None,
        "model": sample_repro.model if sample_repro else None,
        "prompt_version": sample_repro.prompt_version if sample_repro else None,
        "package_version": sample_repro.package_version if sample_repro else None,
        "reliability": {
            cfg_id: json.loads(rel.model_dump_json())
            for cfg_id, rel in reliability.items()
        },
    }
    if extra:
        metadata.update(extra)

    path.write_text(json.dumps(metadata, indent=2, default=str))


# ---------------------------------------------------------------------------
# Pilot summary aggregation
# ---------------------------------------------------------------------------

def compute_pilot_summary(runs: list[Run]) -> list[dict[str, Any]]:
    """
    Grouped summary: one row per (scenario_id, strategy).
    Means are computed over successful runs only; attempted/failed
    counts include all runs.
    """
    from collections import defaultdict

    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.scenario_id, run.strategy.value)].append(run)

    summary = []
    for (scenario_id, strategy), group_runs in sorted(groups.items()):
        succeeded = [r for r in group_runs if r.status == RunStatus.SUCCEEDED]
        summary.append({
            "scenario_id": scenario_id,
            "strategy": strategy,
            "attempted_runs": len(group_runs),
            "successful_runs": len(succeeded),
            "failed_runs": len(group_runs) - len(succeeded),
            "mean_repair_recall": _mean([r.result.ground_truth.repair_recall for r in succeeded if r.result and r.result.ground_truth]),
            "mean_false_modification_rate": _mean([r.result.ground_truth.false_modification_rate for r in succeeded if r.result and r.result.ground_truth]),
            "mean_strategy_execution_wall_seconds": _mean([r.result.performance.strategy_execution_wall_seconds for r in succeeded if r.result and r.result.performance and r.result.performance.strategy_execution_wall_seconds is not None]),
            "mean_total_wall_seconds": _mean([r.result.performance.total_wall_seconds for r in succeeded if r.result and r.result.performance and r.result.performance.total_wall_seconds is not None]),
        })
    return summary


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def compute_aggregate(runs: list[Run]) -> list[dict[str, Any]]:
    """
    Grouped aggregate: one row per (scenario_id, strategy, provider, model).
    Extends compute_pilot_summary with provider/model grouping and std_repair_recall.
    """
    from collections import defaultdict

    groups: dict[tuple, list[Run]] = defaultdict(list)
    for run in runs:
        provider = run.reproducibility.provider or ""
        model = run.reproducibility.model or ""
        groups[(run.scenario_id, run.strategy.value, provider, model)].append(run)

    rows = []
    for (scenario_id, strategy, provider, model), group_runs in sorted(groups.items()):
        succeeded = [r for r in group_runs if r.status == RunStatus.SUCCEEDED]

        recalls = [r.result.ground_truth.repair_recall for r in succeeded if r.result and r.result.ground_truth]
        fmrs = [r.result.ground_truth.false_modification_rate for r in succeeded if r.result and r.result.ground_truth]
        ibs = [r.result.quality.total_issues_before for r in succeeded if r.result and r.result.quality and r.result.quality.total_issues_before is not None]
        ias = [r.result.quality.total_issues_after for r in succeeded if r.result and r.result.quality and r.result.quality.total_issues_after is not None]
        secs = [r.result.performance.strategy_execution_wall_seconds for r in succeeded if r.result and r.result.performance and r.result.performance.strategy_execution_wall_seconds is not None]
        tots = [r.result.performance.total_wall_seconds for r in succeeded if r.result and r.result.performance and r.result.performance.total_wall_seconds is not None]
        inp_tok = [r.result.cost.input_tokens for r in succeeded if r.result and r.result.cost and r.result.cost.input_tokens is not None]
        out_tok = [r.result.cost.output_tokens for r in succeeded if r.result and r.result.cost and r.result.cost.output_tokens is not None]
        tot_tok = [r.result.cost.total_tokens for r in succeeded if r.result and r.result.cost and r.result.cost.total_tokens is not None]

        rows.append({
            "scenario_id": scenario_id,
            "strategy": strategy,
            "provider": provider or None,
            "model": model or None,
            "attempted_runs": len(group_runs),
            "successful_runs": len(succeeded),
            "failed_runs": len(group_runs) - len(succeeded),
            "success_rate": len(succeeded) / len(group_runs) if group_runs else None,
            "mean_repair_recall": _mean(recalls),
            "std_repair_recall": _std(recalls),
            "mean_false_modification_rate": _mean(fmrs),
            "mean_issues_before": _mean(ibs),
            "mean_issues_after": _mean(ias),
            "mean_strategy_execution_wall_seconds": _mean(secs),
            "mean_total_wall_seconds": _mean(tots),
            "mean_input_tokens": _mean(inp_tok) if inp_tok else None,
            "mean_output_tokens": _mean(out_tok) if out_tok else None,
            "mean_total_tokens": _mean(tot_tok) if tot_tok else None,
        })
    return rows


def write_aggregate_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scenario_id", "strategy", "provider", "model",
        "attempted_runs", "successful_runs", "failed_runs", "success_rate",
        "mean_repair_recall", "std_repair_recall", "mean_false_modification_rate",
        "mean_issues_before", "mean_issues_after",
        "mean_strategy_execution_wall_seconds", "mean_total_wall_seconds",
        "mean_input_tokens", "mean_output_tokens", "mean_total_tokens",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

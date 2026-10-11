"""
File-based run storage.

Per the approved P0 architecture: no database, no Redis -- a run is a
single JSON file at data/runs/{run_id}/run.json. This is what makes a
CONVERGE run "inspectable": at any point during or after a run, the
full artifact (profile, evidence, recommendations, validation results,
decisions, timeline) can be read directly off disk.

This module is intentionally tiny. It has no knowledge of pipeline
stages or state transitions -- it only knows how to serialize and
deserialize a PipelineRun.
"""
from __future__ import annotations

from pathlib import Path

from converge.core import config
from converge.models.pipeline import PipelineRun


class RunStore:
    def __init__(self, base_dir: Path | None = None):
        self._base_dir = base_dir or config.RUNS_DIR

    def _run_path(self, run_id: str) -> Path:
        return self._base_dir / run_id / "run.json"

    def save(self, run: PipelineRun) -> Path:
        path = self._run_path(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run.model_dump_json(indent=2))
        return path

    def load(self, run_id: str) -> PipelineRun:
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"No run artifact found for run_id '{run_id}' at {path}")
        return PipelineRun.model_validate_json(path.read_text())

    def exists(self, run_id: str) -> bool:
        return self._run_path(run_id).exists()

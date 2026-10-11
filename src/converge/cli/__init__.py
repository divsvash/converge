"""
CLI entry point for CONVERGE.

Not implemented yet. Reserved so the CLI-only usage path (frontend is
optional, per the project's constraints) has a clear, obvious home
once it's built.

Rules for when this is implemented (recorded now so the boundary is
decided before any code exists, not invented while coding):

    - This package is a composition root: argument parsing, wiring a
      concrete AIProvider + SparkSession + RunStore, constructing a
      PipelineOrchestrator, calling it, formatting output.
    - It must contain NO pipeline logic of its own. Any decision about
      validity, eligibility, or data quality belongs in
      converge.services.*, never here.
    - It depends on converge.services.pipeline (orchestrator) and
      converge.services.ai (to build a provider), converge.storage,
      and converge.core -- nothing else.
"""

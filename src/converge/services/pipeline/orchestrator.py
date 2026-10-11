"""
Pipeline orchestrator.

Coordination logic ONLY. This module's job is to call the existing,
already-tested stages (loader/profiler, AIProvider, Validator) in the
right order, enforce the state machine, and persist the run artifact
after every step. It does not:

    - reimplement profiling, AI reasoning, or validation logic,
    - know what a "duplicate row" or a "categorical variant" is,
    - know anything about Gemini specifically (it depends on the
      AIProvider interface, not GeminiProvider),
    - execute any data transformation itself.

Execution boundary: this orchestrator defines and enforces exactly
which recommendations are allowed to reach the executor -- user-
approved AND validator-eligible, never a finding-only operation --
via `build_execution_plan()`, which returns the immutable
ExecutionPlan. This keeps "only recommendations that are user-approved
AND validator-eligible may reach the executor" true by construction:
`build_execution_plan()` is the *only* path by which an ExecutionPlan
is ever created, and it is built from `RecommendationRecord.decision`
and `.validation` alone -- never from either field in isolation, and
the SparkExecutor never sees either field at all.

Post-execution (Step 6): after a successful CLEANED transition, the
orchestrator re-profiles the executor's output with the SAME Step 1
profiler (no second profiler implementation), hands both profiles to
the deterministic QualityComparator (no LLM), and only then -- as a
separate, later, optional step -- generates a deterministic insight
artifact. A failed comparison never reaches STATS_COMPUTED; a failed
insight generation never touches the already-persisted comparison.
"""
from __future__ import annotations

from pyspark.sql import SparkSession

from converge.core.exceptions import (
    InvalidStateTransitionError,
    PipelineStageError,
    RecommendationNotEligibleError,
    RecommendationNotFoundError,
)
from converge.core.logging import get_logger
from converge.models.comparison import QualityComparison
from converge.models.execution import ExecutionPlan, ExecutionSummary, PlannedOperation
from converge.models.insight import InsightArtifact
from converge.models.pipeline import (
    VALID_TRANSITIONS,
    AIStageMetadata,
    PipelineRun,
    PipelineStatus,
    RecommendationDecision,
    RecommendationRecord,
    TimelineEvent,
    utcnow,
)
from converge.models.recommendation import Recommendation
from converge.models.validation import ValidationStatus
from converge.services.ai.base import AIProvider
from converge.services.comparison.comparator import QualityComparator
from converge.services.comparison.traceability import build_recommendation_outcomes
from converge.services.insights.generator import InsightGenerator
from converge.services.spark.executor import SparkExecutor
from converge.services.spark.loader import load_csv
from converge.services.spark.profiler import profile_dataset
from converge.services.validation.validator import Validator
from converge.storage.run_store import RunStore

logger = get_logger(__name__)

# The only validator outcomes that make a recommendation eligible to
# ever be executed. Deliberately does NOT include NOT_EXECUTABLE
# (FLAG_OUTLIER and any future finding-only operation) or REJECTED --
# this set is the single source of truth for "validator-eligible", and
# every eligibility check in this file (approval guard AND execution
# eligibility filter) is written in terms of membership in this set,
# rather than duplicated per-call-site logic.
EXECUTABLE_VALIDATION_STATUSES = frozenset({
    ValidationStatus.APPROVED_AUTO,
    ValidationStatus.NEEDS_REVIEW,
})

# Locked P0 execution precedence, applied HERE -- at plan-construction
# time, not inside the executor. This is the explicit "planning rule"
# the architecture correction calls for: the resulting ExecutionPlan
# already contains operations in their final, correct order, visible
# and inspectable (it's part of the persisted run artifact) before
# execution ever begins. The executor itself performs zero reordering
# -- it runs exactly the sequence handed to it, whatever that sequence
# is, which is what makes ExecutionPlan.operations a true, explicit
# execution contract rather than a hint.
#   1. type conversions
#   2. missing-value handling
#   3. text/category normalization
#   4. date formatting
#   5. duplicate removal
# No dependency graph -- a flat precedence ranking is sufficient for
# the fixed P0 operation set.
OPERATION_PRECEDENCE: dict[str, int] = {
    "CONVERT_DTYPE": 0,
    "FILL_MISSING": 1,
    "TRIM_WHITESPACE": 2,
    "NORMALIZE_CASE": 2,
    "STANDARDIZE_CATEGORY": 2,
    "STANDARDIZE_DATE_FORMAT": 3,
    "REMOVE_DUPLICATES": 4,
}


class PipelineOrchestrator:
    def __init__(
        self,
        spark: SparkSession,
        ai_provider: AIProvider,
        run_store: RunStore | None = None,
        executor: SparkExecutor | None = None,
        comparator: QualityComparator | None = None,
        insight_generator: InsightGenerator | None = None,
    ):
        self._spark = spark
        self._ai_provider = ai_provider
        self._run_store = run_store or RunStore()
        self._executor = executor or SparkExecutor(spark)
        self._comparator = comparator or QualityComparator()
        self._insight_generator = insight_generator or InsightGenerator()

    # ------------------------------------------------------------------
    # Stage 1: ingest
    # ------------------------------------------------------------------

    def start_run(self, dataset_id: str, dataset_path: str) -> PipelineRun:
        run = PipelineRun(
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            status=PipelineStatus.INGESTED,
        )
        run.timeline.append(TimelineEvent(
            status=PipelineStatus.INGESTED,
            message=f"Dataset '{dataset_id}' ingested from {dataset_path}.",
        ))
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # Stage 2: profiling (delegates entirely to Step 1's loader/profiler)
    # ------------------------------------------------------------------

    def run_profiling(self, run: PipelineRun) -> PipelineRun:
        self._require_status(run, PipelineStatus.INGESTED)

        def work() -> None:
            df = load_csv(self._spark, run.dataset_path)
            run.profile_result = profile_dataset(df, dataset_id=run.dataset_id)

        self._execute_stage(run, "profiling", work)

        pr = run.profile_result
        self._transition(
            run, PipelineStatus.PROFILED,
            message=(
                f"Profiled dataset: {pr.profile.row_count} rows, "
                f"{pr.profile.column_count} columns, {len(pr.evidence)} evidence item(s)."
            ),
        )
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # Stage 3: AI analysis (delegates entirely to Step 3's AIProvider)
    # ------------------------------------------------------------------

    def run_ai_analysis(self, run: PipelineRun) -> PipelineRun:
        self._require_status(run, PipelineStatus.PROFILED)

        def work() -> None:
            recommendations: list[Recommendation] = self._ai_provider.generate_recommendations(
                run.profile_result
            )
            run.recommendation_records = [
                RecommendationRecord(recommendation=r) for r in recommendations
            ]
            run.ai_metadata = AIStageMetadata(
                provider=self._ai_provider.provider_name,
                model=self._ai_provider.model_name,
                prompt_version=self._ai_provider.prompt_version,
                recommendation_count=len(recommendations),
            )

        self._execute_stage(run, "ai_analysis", work)

        self._transition(
            run, PipelineStatus.AI_ANALYZED,
            message=f"AI stage ({run.ai_metadata.provider}/{run.ai_metadata.model}) "
                    f"returned {run.ai_metadata.recommendation_count} recommendation(s).",
        )
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # Stage 4: validation (delegates entirely to Step 2's Validator),
    # then immediately hands off to AWAITING_APPROVAL -- there is no
    # automated action between "validated" and "waiting for a human",
    # so both transitions happen within this one call, each recorded
    # as its own explicit timeline entry.
    # ------------------------------------------------------------------

    def run_validation(self, run: PipelineRun) -> PipelineRun:
        self._require_status(run, PipelineStatus.AI_ANALYZED)

        def work() -> None:
            validator = Validator(run.profile_result)
            for record in run.recommendation_records:
                record.validation = validator.validate(record.recommendation)

        self._execute_stage(run, "validation", work)

        self._transition(
            run, PipelineStatus.VALIDATED,
            message=f"Validated {len(run.recommendation_records)} recommendation(s).",
        )
        self._transition(
            run, PipelineStatus.AWAITING_APPROVAL,
            message="Pipeline is awaiting user approval. No recommendation executes until "
                    "the user explicitly applies it and execution is explicitly triggered.",
        )
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # User decisions -- explicitly does NOT change pipeline status and
    # does NOT execute anything. Only mutates one RecommendationRecord.
    # ------------------------------------------------------------------

    def submit_decision(
        self, run: PipelineRun, recommendation_id: str, decision: RecommendationDecision
    ) -> PipelineRun:
        self._require_status(run, PipelineStatus.AWAITING_APPROVAL)

        if decision == RecommendationDecision.PENDING:
            raise ValueError("decision must be APPROVED_BY_USER or REJECTED_BY_USER, not PENDING.")

        record = self._find_record(run, recommendation_id)

        if decision == RecommendationDecision.APPROVED_BY_USER:
            if record.validation is None or record.validation.status not in EXECUTABLE_VALIDATION_STATUSES:
                actual = record.validation.status.value if record.validation else "unvalidated"
                raise RecommendationNotEligibleError(
                    f"Cannot approve recommendation '{recommendation_id}': validator status "
                    f"is '{actual}', which is never eligible for execution."
                )

        record.decision = decision
        run.updated_at = utcnow()
        run.timeline.append(TimelineEvent(
            status=run.status,
            message=f"User set decision={decision.value} for recommendation '{recommendation_id}' "
                    f"(operation={record.recommendation.operation}). Pipeline status unchanged.",
        ))
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # Execution boundary:
    #     PipelineRun -> orchestrator determines eligibility -> ExecutionPlan
    #         -> SparkExecutor -> Spark
    #
    # begin_execution() is the ONLY place an ExecutionPlan is ever
    # constructed, and it is built exclusively from
    # RecommendationRecord.decision and .validation -- the two fields
    # that respectively represent "the user approved this" and "the
    # validator independently judged this eligible". A PlannedOperation
    # existing in the resulting plan is proof both checks already
    # passed; nothing downstream (the executor) has any way to
    # re-derive, bypass, or need to re-check either fact, because it
    # never receives the RecommendationRecord, RecommendationDecision,
    # or ValidationResult objects at all -- only the plan.
    # ------------------------------------------------------------------

    def build_execution_plan(self, run: PipelineRun) -> ExecutionPlan:
        """
        Transitions AWAITING_APPROVAL -> EXECUTING and returns the
        immutable ExecutionPlan -- the only artifact the executor is
        ever given. Finding-only operations (FLAG_OUTLIER) can never
        appear here: their validation.status is NOT_EXECUTABLE, which
        is not in EXECUTABLE_VALIDATION_STATUSES, so they are excluded
        by this filter regardless of any decision ever recorded on
        them (submit_decision() additionally refuses to let one be
        approved in the first place -- this is the second, independent
        enforcement of the same guarantee).

        Eligible operations are ordered here, at plan-construction
        time, according to OPERATION_PRECEDENCE (a stable sort, so
        operations within the same precedence tier keep their
        relative order). This ordering is final and explicit: it is
        part of the persisted ExecutionPlan the run artifact stores,
        inspectable before execution ever runs, and the executor
        performs no further reordering of its own.
        """
        self._require_status(run, PipelineStatus.AWAITING_APPROVAL)

        eligible_records = [
            record for record in run.recommendation_records
            if record.decision == RecommendationDecision.APPROVED_BY_USER
            and record.validation is not None
            and record.validation.status in EXECUTABLE_VALIDATION_STATUSES
        ]
        eligible_records.sort(
            key=lambda r: OPERATION_PRECEDENCE.get(r.recommendation.operation, len(OPERATION_PRECEDENCE))
        )

        planned_operations = tuple(
            PlannedOperation(
                recommendation_id=record.recommendation.recommendation_id,
                operation=record.recommendation.operation,
                target_columns=tuple(record.recommendation.target_columns),
                parameters=dict(record.recommendation.parameters),
            )
            for record in eligible_records
        )

        plan = ExecutionPlan(
            run_id=run.run_id,
            dataset_path=run.dataset_path,
            operations=planned_operations,
        )
        run.execution_plan = plan

        self._transition(
            run, PipelineStatus.EXECUTING,
            message=f"Execution plan built with {len(planned_operations)} user-approved, "
                    f"validator-eligible operation(s).",
        )
        self._persist(run)
        return plan

    def complete_execution(self, run: PipelineRun, execution_summary: ExecutionSummary) -> PipelineRun:
        """
        Called once the executor has finished processing a plan.
        Transitions EXECUTING -> CLEANED only if every planned
        operation succeeded. If any operation failed (or was skipped
        due to fail-fast), the run is transitioned to FAILED instead
        -- CLEANED must never be reached for a run whose execution did
        not fully complete.

        IMPORTANT SEMANTIC NOTE: CLEANED means "all approved,
        executable transformations completed successfully." It does
        NOT mean "the dataset now has no remaining data-quality
        issues." Execution can only apply what was actually approved
        -- rejected recommendations, operations the user chose not to
        approve, and issues no recommendation ever covered are all
        still present in the output dataset. Measuring what actually
        changed requires re-profiling the output dataset and comparing
        it against the original profile -- that comparison is Step 6
        (post-execution profiling / quality comparison), not this
        step. CLEANED here is a pipeline-progress state, not a claim
        about data quality.
        """
        self._require_status(run, PipelineStatus.EXECUTING)
        run.execution_summary = execution_summary

        if not execution_summary.fully_successful:
            message = (
                f"Execution did not complete successfully: "
                f"{execution_summary.operations_failed} failed, "
                f"{execution_summary.operations_skipped} skipped."
            )
            self._fail(run, stage="execution", error=RuntimeError(message))
            raise PipelineStageError(message)

        self._transition(
            run, PipelineStatus.CLEANED,
            message=(
                f"Execution completed: {execution_summary.operations_succeeded} approved "
                f"operation(s) applied successfully. This means the approved transformations "
                f"completed -- it does not mean the dataset is free of remaining data-quality "
                f"issues; re-profiling the output is required to measure that."
            ),
        )
        self._persist(run)
        return run

    def execute(self, run: PipelineRun) -> PipelineRun:
        """
        Convenience wrapper: build the plan, hand it to the injected
        SparkExecutor, and complete execution with the result -- all
        in one call. build_execution_plan()/complete_execution() remain
        available separately for callers (and tests) that want to
        inspect the plan before it runs, or supply their own executor
        result.
        """
        plan = self.build_execution_plan(run)
        summary = self._executor.execute(plan)
        return self.complete_execution(run, summary)

    # ------------------------------------------------------------------
    # Stage: post-execution profiling + deterministic comparison (Step 6)
    #
    # Reuses the exact Step 1 profiler function against the executor's
    # output dataset -- no second profiler implementation. The before
    # profile (`run.profile_result`) is never touched or overwritten;
    # the after profile is stored separately on `run.after_profile_result`.
    # Comparison is 100% deterministic (QualityComparator, no LLM).
    # ------------------------------------------------------------------

    def run_post_execution_analysis(self, run: PipelineRun) -> PipelineRun:
        """
        Transitions CLEANED -> STATS_COMPUTED. Re-profiles the
        executor's output dataset, computes the deterministic
        before/after QualityComparison, and builds recommendation ->
        outcome traceability. If any part of this fails, the run goes
        to FAILED -- STATS_COMPUTED is never reached for an incomplete
        or failed comparison (this mirrors the exact same fail-fast
        discipline used for every other stage).

        IMPORTANT: reaching STATS_COMPUTED is a statement about the
        comparison having been computed successfully. It says nothing
        about whether the comparison's *findings* were favorable --
        even a comparison that finds issues WORSENED or were newly
        INTRODUCED is a "successful" comparison in this sense.
        """
        self._require_status(run, PipelineStatus.CLEANED)

        def work() -> None:
            if run.execution_summary is None or not run.execution_summary.output_dataset_path:
                raise RuntimeError(
                    "No output dataset path recorded on execution_summary; cannot re-profile."
                )
            after_dataset_id = f"{run.dataset_id}__after_execution"
            df = load_csv(self._spark, run.execution_summary.output_dataset_path)
            run.after_profile_result = profile_dataset(df, dataset_id=after_dataset_id)

            comparison = self._comparator.compare(run.profile_result, run.after_profile_result)
            run.quality_comparison = comparison
            run.recommendation_outcomes = build_recommendation_outcomes(
                run.recommendation_records, run.execution_summary, comparison
            )

        self._execute_stage(run, "post_execution_analysis", work)

        m = run.quality_comparison.metrics
        self._transition(
            run, PipelineStatus.STATS_COMPUTED,
            message=(
                f"Post-execution comparison computed: {m.resolved_count} resolved, "
                f"{m.improved_count} improved, {m.unchanged_count} unchanged, "
                f"{m.worsened_count} worsened, {m.introduced_count} introduced, "
                f"{m.unverified_count} unverified. This reflects execution completeness and "
                f"measured change -- not a claim that the dataset is now fully clean."
            ),
        )
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # Stage: deterministic insight generation (Step 6)
    #
    # Consumes ONLY the already-persisted QualityComparison and
    # recommendation outcomes -- never the raw dataset, never the
    # original profile fields. See InsightGenerator's own docstring
    # for why this is deterministic (no LLM) for P0.
    # ------------------------------------------------------------------

    def generate_insights(self, run: PipelineRun) -> PipelineRun:
        """
        Transitions STATS_COMPUTED -> INSIGHTS_GENERATED. A failure
        here goes to FAILED but never touches `run.quality_comparison`
        or `run.recommendation_outcomes` -- those were already
        computed and persisted by the prior stage and are untouched by
        anything this method does.
        """
        self._require_status(run, PipelineStatus.STATS_COMPUTED)

        def work() -> None:
            run.insight_artifact = self._insight_generator.generate(
                run.quality_comparison, run.recommendation_outcomes
            )

        self._execute_stage(run, "insight_generation", work)

        self._transition(
            run, PipelineStatus.INSIGHTS_GENERATED,
            message=f"Generated {len(run.insight_artifact.insights)} insight(s) "
                    f"(source={run.insight_artifact.source}).",
        )
        self._persist(run)
        return run

    def complete_run(self, run: PipelineRun) -> PipelineRun:
        """Transitions INSIGHTS_GENERATED -> COMPLETED. Trivial, no new work."""
        self._require_status(run, PipelineStatus.INSIGHTS_GENERATED)
        self._transition(run, PipelineStatus.COMPLETED, message="Pipeline run completed.")
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_status(self, run: PipelineRun, expected: PipelineStatus) -> None:
        if run.status != expected:
            raise InvalidStateTransitionError(
                f"This operation requires run '{run.run_id}' to be in status "
                f"'{expected.value}', but it is currently '{run.status.value}'."
            )

    def _transition(self, run: PipelineRun, new_status: PipelineStatus, message: str) -> None:
        allowed = VALID_TRANSITIONS.get(run.status, set())
        if new_status not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid state transition for run '{run.run_id}': "
                f"'{run.status.value}' -> '{new_status.value}' is not permitted."
            )
        run.status = new_status
        run.updated_at = utcnow()
        run.timeline.append(TimelineEvent(status=new_status, message=message))
        logger.info("Run '%s' transitioned to %s: %s", run.run_id, new_status.value, message)

    def _execute_stage(self, run: PipelineRun, stage_name: str, work) -> None:
        """
        Runs `work()` (which mutates `run` in place) and, on any
        exception, transitions the run to FAILED with the error
        recorded -- persisting that failure -- before re-raising a
        PipelineStageError. No stage failure is ever left unrecorded
        or silently treated as success; the caller cannot proceed past
        a failed stage because the run's status is no longer the
        status any subsequent method requires.
        """
        try:
            work()
        except Exception as e:
            self._fail(run, stage_name, e)
            raise PipelineStageError(f"Stage '{stage_name}' failed: {e}") from e

    def _fail(self, run: PipelineRun, stage: str, error: Exception) -> None:
        run.status = PipelineStatus.FAILED
        run.error = f"[{stage}] {type(error).__name__}: {error}"
        run.updated_at = utcnow()
        run.timeline.append(TimelineEvent(status=PipelineStatus.FAILED, message=run.error))
        logger.error("Run '%s' failed at stage '%s': %s", run.run_id, stage, error)
        self._persist(run)

    def _find_record(self, run: PipelineRun, recommendation_id: str) -> RecommendationRecord:
        for record in run.recommendation_records:
            if record.recommendation.recommendation_id == recommendation_id:
                return record
        raise RecommendationNotFoundError(
            f"No recommendation with id '{recommendation_id}' exists on run '{run.run_id}'."
        )

    def _persist(self, run: PipelineRun) -> None:
        self._run_store.save(run)

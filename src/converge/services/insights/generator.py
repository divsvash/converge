"""
Insight generator.

P0 SCOPE DECISION: this generator is entirely deterministic,
template-based Python -- there is NO LLM call anywhere in this module.

Why, given Part F allows for an AI-generated natural-language layer:
Part F frames that layer as conditional ("if implemented") and
explicitly subordinate to the deterministic comparison ("the
deterministic comparison is more fundamental than the optional
natural-language insight layer... keep the system usable without the
AI insight generator"). Building the deterministic version first, and
shipping it as a complete, self-sufficient P0 capability, satisfies
every stated requirement (fact/interpretation/uncertainty separation,
no dataset access, no independent metric calculation, cannot override
the comparator) without adding an AI dependency this step doesn't
strictly need. A future AIProvider-backed rephrasing layer is a
natural extension -- it would consume this same InsightArtifact/
QualityComparison (never the raw dataset or original profile fields),
following the exact provider-abstraction pattern already established
in Step 3 -- but is not implemented here. `InsightArtifact.source`
already reserves the value "ai" for that future addition so the
artifact shape doesn't need to change when/if it's built.

Every Insight is explicitly tagged FACT / INTERPRETATION / UNCERTAINTY
(see converge.models.insight.InsightKind). FACT insights state a number
or comparison outcome directly present in the QualityComparison.
INTERPRETATION insights are generated from one narrow, explicit,
non-causal rule (never free-form). UNCERTAINTY insights exist
specifically to surface UNVERIFIED comparisons and execution gaps
rather than omit them.
"""
from __future__ import annotations

from converge.models.comparison import IssueOutcome, QualityComparison, RecommendationOutcome
from converge.models.insight import Insight, InsightArtifact, InsightKind


class InsightGenerator:
    """Stateless -- generate() takes the comparison/outcomes as arguments."""

    def generate(
        self, comparison: QualityComparison, outcomes: list[RecommendationOutcome]
    ) -> InsightArtifact:
        insights: list[Insight] = []
        insights.extend(self._headline_facts(comparison))
        insights.extend(self._per_issue_facts(comparison))
        insights.extend(self._execution_facts(outcomes))
        insights.extend(self._uncertainty(comparison, outcomes))
        insights.extend(self._interpretation(comparison))
        return InsightArtifact(insights=insights, source="deterministic")

    # ------------------------------------------------------------------
    # FACT insights
    # ------------------------------------------------------------------

    def _headline_facts(self, comparison: QualityComparison) -> list[Insight]:
        m = comparison.metrics
        facts = [
            Insight(
                kind=InsightKind.FACT,
                text=f"{m.total_issues_before} data-quality issue(s) were detected before execution; "
                     f"{m.total_issues_after} are detected after.",
            ),
            Insight(
                kind=InsightKind.FACT,
                text=(
                    f"Of the issues compared: {m.resolved_count} resolved, {m.improved_count} improved, "
                    f"{m.unchanged_count} unchanged, {m.worsened_count} worsened, "
                    f"{m.introduced_count} newly introduced, {m.unverified_count} could not be verified."
                ),
            ),
        ]
        if m.row_count_before != m.row_count_after:
            facts.append(Insight(
                kind=InsightKind.FACT,
                text=f"Row count changed from {m.row_count_before} to {m.row_count_after}.",
            ))
        return facts

    def _per_issue_facts(self, comparison: QualityComparison) -> list[Insight]:
        facts = []
        for ic in comparison.issue_comparisons:
            if ic.outcome in (IssueOutcome.RESOLVED, IssueOutcome.IMPROVED, IssueOutcome.WORSENED, IssueOutcome.INTRODUCED):
                facts.append(Insight(
                    kind=InsightKind.FACT,
                    text=(
                        f"{ic.issue_type.value.replace('_', ' ').capitalize()} in "
                        f"{', '.join(repr(c) for c in ic.columns)} was {ic.outcome.value.upper()} "
                        f"({ic.before_value} -> {ic.after_value})."
                    ),
                    related_issue_types=[ic.issue_type],
                    related_columns=ic.columns,
                ))
        return facts

    def _execution_facts(self, outcomes: list[RecommendationOutcome]) -> list[Insight]:
        approved = [o for o in outcomes if o.decision == "approved_by_user"]
        if not approved:
            return [Insight(kind=InsightKind.FACT, text="No recommendations were approved for execution.")]
        succeeded = sum(1 for o in approved if o.execution_status == "succeeded")
        return [Insight(
            kind=InsightKind.FACT,
            text=f"{succeeded} of {len(approved)} approved operation(s) executed successfully.",
        )]

    # ------------------------------------------------------------------
    # UNCERTAINTY insights
    # ------------------------------------------------------------------

    def _uncertainty(
        self, comparison: QualityComparison, outcomes: list[RecommendationOutcome]
    ) -> list[Insight]:
        insights = []
        for ic in comparison.issue_comparisons:
            if ic.outcome == IssueOutcome.UNVERIFIED:
                insights.append(Insight(
                    kind=InsightKind.UNCERTAINTY,
                    text=f"Could not verify the outcome for {ic.issue_type.value} in "
                         f"{', '.join(repr(c) for c in ic.columns)}: {ic.detail}",
                    related_issue_types=[ic.issue_type],
                    related_columns=ic.columns,
                ))
        for o in outcomes:
            if o.decision == "approved_by_user" and o.execution_status == "succeeded" and not o.linked_issue_comparisons:
                insights.append(Insight(
                    kind=InsightKind.UNCERTAINTY,
                    text=(
                        f"'{o.operation}' on {o.target_columns} executed successfully, but no directly "
                        f"comparable before/after evidence exists to confirm its effect."
                    ),
                    related_columns=o.target_columns,
                ))
        return insights

    # ------------------------------------------------------------------
    # INTERPRETATION insights (one narrow, explicit, non-causal rule)
    # ------------------------------------------------------------------

    def _interpretation(self, comparison: QualityComparison) -> list[Insight]:
        m = comparison.metrics
        positive = m.resolved_count + m.improved_count
        negative = m.worsened_count + m.introduced_count

        if positive == 0 and negative == 0:
            return []

        if positive > negative:
            text = (
                f"The approved transformations were associated with a net improvement in the issue "
                f"categories they targeted ({positive} resolved/improved vs. {negative} worsened/introduced). "
                f"This describes the categories the recommendations addressed -- it is not a claim about "
                f"overall dataset quality, and issues nothing was recommended for are not reflected here."
            )
        elif negative > positive:
            text = (
                f"The approved transformations were associated with a net regression in the issue "
                f"categories they targeted ({negative} worsened/introduced vs. {positive} resolved/improved). "
                f"This warrants review before trusting these transformations on similar data."
            )
        else:
            text = (
                f"The approved transformations show an even mix of improvement and regression "
                f"({positive} resolved/improved vs. {negative} worsened/introduced) in the categories they targeted."
            )

        return [Insight(kind=InsightKind.INTERPRETATION, text=text)]

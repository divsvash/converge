"""
Configuration -- the treatment being compared: which strategy, and
(for AI-involving strategies) which model, how it is deployed, where
in the pipeline it acts, and what role it performs.

StrategyName selects which experiments/strategies/*.py implementation
would run (not implemented in this phase).

ModelClass and DeploymentMode are deliberately separate axes, per
explicit correction: "local" is a deployment/execution characteristic,
not a model class. A model can be an LLM or an SLM (ModelClass), and
*independently* can run REMOTE (an API call) or LOCAL (e.g. an
Ollama-hosted model) (DeploymentMode). Neither is ever inferred from
the model/provider name -- both are explicit fields a Configuration
must state, never guessed from a string like "gemini-2.0-flash".

AIPlacement and AIRole are kept intentionally minimal: only the values
actually representable by the strategies this phase anticipates
(spark_only, ai_only, converge -- none implemented yet). See each
enum's own docstring for exactly what is supported now and the
explicit extension point for values this phase deliberately does not
add.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class StrategyName(str, Enum):
    SPARK_ONLY = "spark_only"
    AI_ONLY = "ai_only"
    CONVERGE = "converge"


class ModelClass(str, Enum):
    """
    A coarse size/capability classification of a model, used only for
    grouping results in research analysis. converge itself has no
    notion of this -- AIProvider exposes provider_name/model_name/
    prompt_version only, nothing about size class. Never inferred from
    the model name; always an explicit field stated by the
    Configuration.
    """
    LLM = "llm"
    SLM = "slm"


class DeploymentMode(str, Enum):
    """
    Where the model actually executes -- independent of its size
    class. A model can be, for example, an LLM served remotely (the
    Gemini API) or, in principle, an LLM or SLM hosted locally (e.g.
    via Ollama). This is a genuinely separate axis from ModelClass; a
    value here is never derived from ModelClass or from the
    model/provider name.
    """
    REMOTE = "remote"
    LOCAL = "local"


class AIPlacement(str, Enum):
    """
    Where in the pipeline timeline the AI acts, relative to Spark.

    Minimal vocabulary for what this phase's planned strategies can
    actually represent:

        DURING -- the AI acts as part of the Spark pipeline itself,
                  consuming Spark-produced evidence and producing
                  output the pipeline consumes further. This is the
                  only placement converge.services.pipeline.
                  orchestrator implements today: AI analysis happens
                  strictly between Spark profiling and Spark
                  execution.

    Deliberately NOT added here: BEFORE / AFTER placements. These were
    raised as a future research axis, but no strategy planned for this
    phase actually produces a "before Spark" or "after Spark" AI
    placement yet. Extend this enum when a strategy that genuinely
    needs a new value is being implemented -- not speculatively now,
    to avoid inventing a placement merely to fill the enum out.
    """
    DURING = "during"


class AIRole(str, Enum):
    """
    What the AI is asked to do.

    Minimal vocabulary for what this phase's planned strategies can
    actually represent:

        RECOMMENDATION -- the AI proposes structured cleaning
                           recommendations, which converge's Validator
                           and executor may or may not act on. This is
                           the only role converge.services.pipeline.
                           orchestrator implements today.

    Deliberately NOT added here, even though the research design
    anticipates them eventually: VALIDATION, TRANSFORMATION,
    INSIGHT_GENERATION.
        - converge's validator is, and must remain, deterministic --
          an AI performing a VALIDATION role is only something the
          not-yet-implemented ai_only baseline could ever represent,
          never a CONVERGE configuration.
        - converge's insight stage is deterministic, not AI-driven
          (services.insights.generator), so INSIGHT_GENERATION is not
          representable by any currently planned strategy.
        - TRANSFORMATION (AI directly rewriting data, bypassing
          converge's validator/executor) is only meaningful for the
          not-yet-implemented ai_only baseline.
    Add each of these when the strategy that actually needs it is
    being implemented, with its semantics pinned down by that
    strategy's real requirements -- not speculatively here.
    """
    RECOMMENDATION = "recommendation"


# Single source of truth for which strategies require AI fields to be
# populated (and, by exclusion, which require them to be absent).
_STRATEGIES_REQUIRING_AI_FIELDS = frozenset({StrategyName.AI_ONLY, StrategyName.CONVERGE})

# Which (strategy, ai_placement, ai_role) combinations are currently
# valid. Deliberately explicit and closed rather than a general rule:
# CONVERGE's valid combination is defined by what
# converge.services.pipeline.orchestrator actually implements, and
# ai_only has no implementation yet to define further combinations
# against, so only the one combination each strategy is currently
# capable of is accepted. Extend this set -- not the enums -- when a
# new strategy implementation actually supports a new combination.
_VALID_STRATEGY_PLACEMENT_ROLE: frozenset[tuple[StrategyName, AIPlacement, AIRole]] = frozenset({
    (StrategyName.CONVERGE, AIPlacement.DURING, AIRole.RECOMMENDATION),
    (StrategyName.AI_ONLY, AIPlacement.DURING, AIRole.RECOMMENDATION),
})


class Configuration(BaseModel):
    """
    One treatment: which strategy, and (for AI-involving strategies)
    which model, how deployed, where placed, what role.

    Identity is content-derived and stable (see `configuration_id`),
    not a random UUID: two Configurations with identical field values
    ARE the same configuration for comparison purposes.
    """
    model_config = ConfigDict(frozen=True)

    strategy: StrategyName
    provider: Optional[str] = None
    model: Optional[str] = None
    model_class: Optional[ModelClass] = None
    deployment_mode: Optional[DeploymentMode] = None
    ai_placement: Optional[AIPlacement] = None
    ai_role: Optional[AIRole] = None

    @model_validator(mode="after")
    def _validate_ai_fields(self) -> "Configuration":
        ai_fields = {
            "provider": self.provider,
            "model": self.model,
            "model_class": self.model_class,
            "deployment_mode": self.deployment_mode,
            "ai_placement": self.ai_placement,
            "ai_role": self.ai_role,
        }

        if self.strategy == StrategyName.SPARK_ONLY:
            populated = [name for name, value in ai_fields.items() if value is not None]
            if populated:
                raise ValueError(
                    f"strategy=SPARK_ONLY must not set any AI-specific field; found: {populated}."
                )
            return self

        if self.strategy in _STRATEGIES_REQUIRING_AI_FIELDS:
            missing = [name for name, value in ai_fields.items() if value is None]
            if missing:
                raise ValueError(
                    f"strategy={self.strategy.value} requires all AI-specific fields "
                    f"to be set; missing: {missing}."
                )
            if not self.provider.strip():
                raise ValueError("provider must not be an empty string.")
            if not self.model.strip():
                raise ValueError("model must not be an empty string.")

            combo = (self.strategy, self.ai_placement, self.ai_role)
            if combo not in _VALID_STRATEGY_PLACEMENT_ROLE:
                raise ValueError(
                    f"The combination strategy={self.strategy.value}, "
                    f"ai_placement={self.ai_placement.value}, ai_role={self.ai_role.value} "
                    f"is not currently supported. Supported combinations: "
                    f"{sorted((s.value, p.value, r.value) for s, p, r in _VALID_STRATEGY_PLACEMENT_ROLE)}."
                )
        return self

    @property
    def configuration_id(self) -> str:
        """
        Stable, content-derived identity: a SHA-256 hash of the
        canonical JSON representation of this Configuration's fields.
        Two Configurations with identical field values always produce
        the same configuration_id, by construction -- there is no
        separate identity to keep in sync with the content.
        """
        canonical = json.dumps(
            {
                "strategy": self.strategy.value,
                "provider": self.provider,
                "model": self.model,
                "model_class": self.model_class.value if self.model_class else None,
                "deployment_mode": self.deployment_mode.value if self.deployment_mode else None,
                "ai_placement": self.ai_placement.value if self.ai_placement else None,
                "ai_role": self.ai_role.value if self.ai_role else None,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

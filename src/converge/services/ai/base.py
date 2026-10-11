"""
AIProvider abstraction.

This is the single seam between CONVERGE and any specific AI backend.
Nothing outside `services/ai/` should ever import `GeminiProvider`
directly in application logic that isn't explicitly wiring up which
provider to use -- callers should depend on `AIProvider`, so that
adding PhiProvider or OllamaProvider later (out of scope for P0) means
writing one new class here, not touching the pipeline, validator, or
API layer.

Hard boundaries every implementation must respect (enforced by the
method signature itself, not just documentation):

    - The only input is a `ProfileResult` -- the bounded, Spark-
      generated profile/evidence contract from Step 1. There is no
      parameter through which a raw dataset, file path, or Spark
      handle could be passed in.
    - The only output is a `list[Recommendation]` -- the structured,
      data-only contract from Step 2. There is no return type through
      which executable code could flow out.
    - Implementations must never call the Validator, never touch
      Spark, and never write to the dataset. Enforcing this is a
      matter of implementation discipline (the interface can't stop a
      misbehaving implementation from importing the validator), but no
      P0 implementation does so, and it is called out here explicitly
      as an implementation requirement.
"""
from abc import ABC, abstractmethod

from converge.models.profile import ProfileResult
from converge.models.recommendation import Recommendation


class AIProvider(ABC):
    """Base interface for all AI recommendation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Short, stable identifier for the provider (e.g. "gemini").
        Used only for run-artifact metadata (provider/model/prompt
        version) so future model comparisons are possible -- never
        used for branching logic elsewhere in the pipeline.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The specific model identifier used (e.g. "gemini-2.0-flash")."""
        raise NotImplementedError

    @property
    @abstractmethod
    def prompt_version(self) -> str:
        """The versioned prompt identifier this provider is currently using."""
        raise NotImplementedError

    @abstractmethod
    def generate_recommendations(self, profile_result: ProfileResult) -> list[Recommendation]:
        """
        Given a Spark-generated profile/evidence bundle, return a list
        of structured Recommendation objects.

        Must raise a subclass of `AIProviderError`
        (converge.core.exceptions) on any failure -- config, transport,
        timeout, rate limiting, or malformed output. Must never raise
        an unhandled/unexpected exception type, and must never return
        anything other than a list of well-formed Recommendation
        objects (an empty list is a valid, expected result when the
        evidence doesn't support any recommendation).
        """
        raise NotImplementedError

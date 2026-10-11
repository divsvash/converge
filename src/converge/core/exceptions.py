"""
Controlled exceptions for CONVERGE.

The AI stage in particular must never let an unhandled exception (a
network error, a malformed response, a timeout) crash the pipeline.
Every failure mode the AI provider can encounter is mapped to one of
these typed exceptions, so a future orchestrator can catch
`AIProviderError` once and record a controlled pipeline failure
instead of the process dying.
"""


class ConvergeError(Exception):
    """Base class for all CONVERGE application errors."""


class AIProviderError(ConvergeError):
    """Base class for every error the AI stage can raise."""


class AIProviderConfigError(AIProviderError):
    """The provider is missing required configuration (e.g. no API key)."""


class AIProviderAPIError(AIProviderError):
    """The upstream AI API returned an error response (4xx/5xx, not a rate limit)."""


class AIProviderRateLimitError(AIProviderAPIError):
    """The upstream AI API rejected the request due to rate limiting (HTTP 429)."""


class AIProviderTimeoutError(AIProviderError):
    """The request to the upstream AI API did not complete in time."""


class AIProviderResponseError(AIProviderError):
    """
    The upstream AI API responded successfully at the transport level,
    but the content was not usable: invalid JSON, missing the expected
    top-level structure, or a recommendation that fails to parse into
    the Recommendation schema (e.g. a missing required field).

    This is distinct from a *validation* failure -- a recommendation
    that parses successfully but is later rejected by the Validator
    (e.g. an unknown operation, fabricated evidence) is NOT an error;
    it is a normal, expected Validator outcome. AIProviderResponseError
    means the AI stage itself could not even produce a well-formed
    Recommendation object.
    """


class PipelineError(ConvergeError):
    """Base class for every error the pipeline orchestrator can raise."""


class InvalidStateTransitionError(PipelineError):
    """
    Raised when an orchestrator method is called while the run is not
    in the status that method requires, or when a computed transition
    is not present in the state machine's transition table. Distinct
    from PipelineStageError: this means the *caller* did something out
    of order, not that a stage's underlying work failed.
    """


class PipelineStageError(PipelineError):
    """
    Raised when a stage's underlying work (profiling, AI analysis,
    validation) raises an exception. The run is transitioned to
    FAILED with the error recorded before this is raised -- so the
    failure is both structurally recorded on the run artifact AND
    surfaced to the caller as a real exception. Nothing is ever
    silently swallowed.
    """


class RecommendationNotFoundError(PipelineError):
    """Raised when a decision is submitted for a recommendation_id that doesn't exist on the run."""


class RecommendationNotEligibleError(PipelineError):
    """
    Raised when attempting to APPROVE (not reject) a recommendation
    whose validator status is REJECTED or NOT_EXECUTABLE. Approving
    something the validator has already ruled out is never a valid
    action -- failing loudly here (rather than silently accepting a
    no-op decision) is what makes the finding-only /
    never-auto-executable guarantee (FLAG_OUTLIER) and the
    rejected-stays-rejected guarantee structurally enforced, not just
    conventionally true.
    """

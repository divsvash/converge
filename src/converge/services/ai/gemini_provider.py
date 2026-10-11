"""
GeminiProvider: the concrete, P0-only AIProvider implementation.

Boundary discipline (see also base.py):
    - generate_recommendations() takes a ProfileResult and nothing else.
      There is no code path in this file that opens a file, queries
      Spark, or accepts a raw dataset.
    - The Gemini client is called exactly once per
      generate_recommendations() call, with exactly two inputs: the
      fixed SYSTEM_PROMPT and the JSON built by build_user_prompt()
      (operation catalog + profile + evidence). Nothing else is ever
      sent.
    - Every failure mode -- config, transport error, rate limit,
      timeout, malformed/unparseable output -- is caught here and
      re-raised as a typed AIProviderError subclass. Nothing escapes
      as a raw SDK exception or an unhandled exception of any kind.
    - Output parsing is strict: JSON is parsed, the top-level shape is
      checked, and every entry is constructed as a `Recommendation`
      Pydantic model. A Recommendation that fails to construct (e.g. a
      missing required field) fails the whole call with
      AIProviderResponseError -- it is not silently dropped, since
      that would hide a real problem with the model's output.
      A Recommendation that constructs successfully but has, e.g., an
      unknown operation name or fabricated evidence is NOT an error
      here -- that is exactly the Validator's job (Step 2), not this
      stage's.
"""
from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from converge.core.exceptions import (
    AIProviderAPIError,
    AIProviderConfigError,
    AIProviderRateLimitError,
    AIProviderResponseError,
    AIProviderTimeoutError,
)
from converge.core.logging import get_logger
from converge.models.profile import ProfileResult
from converge.models.recommendation import Recommendation
from converge.services.ai.base import AIProvider
from converge.services.ai.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from converge.services.ai.schema import RECOMMENDATION_RESPONSE_SCHEMA

logger = get_logger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TIMEOUT_SECONDS = 30.0


class GeminiProvider(AIProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: Any = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        """
        `client` is an injection point for tests (or an alternate
        pre-configured google-genai Client) -- when provided, no API
        key handling or real client construction happens at all. This
        is what lets the test suite exercise GeminiProvider's prompt
        building, error mapping, and output parsing without ever
        touching the network or requiring a real Gemini API key.
        """
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._prompt_version = PROMPT_VERSION

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise AIProviderConfigError(
                "GeminiProvider requires an api_key argument, a GEMINI_API_KEY "
                "environment variable, or an injected client."
            )

        # Imported lazily so importing this module never requires
        # network access or credentials unless a real client is
        # actually being constructed.
        from google import genai
        self._client = genai.Client(api_key=resolved_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def generate_recommendations(self, profile_result: ProfileResult) -> list[Recommendation]:
        user_prompt = build_user_prompt(profile_result)

        logger.info(
            "Calling Gemini (model=%s, prompt_version=%s) for dataset '%s'",
            self._model, self._prompt_version, profile_result.profile.dataset_id,
        )

        raw_text = self._call_model(user_prompt)
        return self._parse_recommendations(raw_text)

    # ------------------------------------------------------------------
    # Transport call + error mapping
    # ------------------------------------------------------------------

    def _call_model(self, user_prompt: str) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RECOMMENDATION_RESPONSE_SCHEMA,
            temperature=0.0,
            http_options=types.HttpOptions(timeout=int(self._timeout_seconds * 1000)),
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=config,
            )
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                raise AIProviderRateLimitError(f"Gemini rate limit exceeded: {e}") from e
            raise AIProviderAPIError(f"Gemini rejected the request (client error): {e}") from e
        except errors.ServerError as e:
            raise AIProviderAPIError(f"Gemini server error: {e}") from e
        except TimeoutError as e:
            raise AIProviderTimeoutError(f"Gemini request timed out: {e}") from e
        except Exception as e:
            # Catches transport-level timeouts (e.g. httpx.TimeoutException)
            # and anything else the SDK might raise that isn't one of the
            # typed cases above -- we never let a raw, unmapped exception
            # escape this boundary.
            if "timeout" in type(e).__name__.lower():
                raise AIProviderTimeoutError(f"Gemini request timed out: {e}") from e
            raise AIProviderAPIError(f"Unexpected error calling Gemini: {e}") from e

        text = getattr(response, "text", None)
        if not text:
            raise AIProviderResponseError("Gemini response contained no text content.")
        return text

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_recommendations(self, raw_text: str) -> list[Recommendation]:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise AIProviderResponseError(
                f"Gemini output was not valid JSON: {e}"
            ) from e

        if not isinstance(data, dict) or "recommendations" not in data:
            raise AIProviderResponseError(
                "Gemini output is missing the required top-level 'recommendations' field."
            )

        raw_list = data["recommendations"]
        if not isinstance(raw_list, list):
            raise AIProviderResponseError("'recommendations' must be a JSON array.")

        recommendations: list[Recommendation] = []
        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                raise AIProviderResponseError(
                    f"recommendations[{i}] is not a JSON object."
                )
            try:
                recommendations.append(Recommendation(**item))
            except ValidationError as e:
                raise AIProviderResponseError(
                    f"recommendations[{i}] failed to parse as a Recommendation: {e}"
                ) from e

        logger.info("Parsed %d recommendation(s) from Gemini output.", len(recommendations))
        return recommendations

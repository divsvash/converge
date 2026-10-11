"""
OllamaProvider -- AIProvider implementation for locally-hosted models
served by Ollama (https://ollama.com).

Reuses the EXACT same versioned prompt (SYSTEM_PROMPT, build_user_prompt,
PROMPT_VERSION) and the EXACT same Recommendation parsing logic as
GeminiProvider -- the only difference is the transport layer (HTTP POST
to Ollama's /api/chat endpoint instead of the Google GenAI SDK) and the
absence of structured-output schema enforcement (Ollama does not support
Gemini's response_schema; structure is enforced by the prompt itself
plus strict Pydantic parsing, which is the same fallback any provider
without native schema enforcement would use).

No provider-specific behavior leaks into the orchestrator: the
orchestrator depends only on AIProvider, never on this module directly.

Configuration:
    model_name   -- the Ollama model tag (e.g. "phi3:mini", "llama3.2")
    base_url     -- Ollama API base URL; defaults to http://localhost:11434
    timeout_seconds -- request timeout; defaults to 120s (local models
                        can be slow on first load)

Does NOT auto-download/pull models. If the requested model is not
already pulled, the Ollama API returns an error and we surface it
cleanly as AIProviderAPIError.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from pydantic import ValidationError

from converge.core.exceptions import (
    AIProviderAPIError,
    AIProviderConfigError,
    AIProviderResponseError,
    AIProviderTimeoutError,
)
from converge.core.logging import get_logger
from converge.models.profile import ProfileResult
from converge.models.recommendation import Recommendation
from converge.services.ai.base import AIProvider
from converge.services.ai.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaProvider(AIProvider):
    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        _http_post: Any = None,
    ):
        """
        `_http_post` is an injection point for tests -- when provided,
        it replaces the real HTTP call entirely. Same pattern
        GeminiProvider uses with its `client` parameter.
        """
        if not model_name or not model_name.strip():
            raise AIProviderConfigError("OllamaProvider requires a non-empty model_name.")
        self._model = model_name.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._prompt_version = PROMPT_VERSION
        self._http_post = _http_post

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def generate_recommendations(self, profile_result: ProfileResult) -> list[Recommendation]:
        user_prompt = build_user_prompt(profile_result)

        logger.info(
            "Calling Ollama (model=%s, prompt_version=%s, base_url=%s) for dataset '%s'",
            self._model, self._prompt_version, self._base_url,
            profile_result.profile.dataset_id,
        )

        raw_text = self._call_model(user_prompt)
        return self._parse_recommendations(raw_text)

    def _call_model(self, user_prompt: str) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        body = json.dumps(payload).encode("utf-8")

        if self._http_post is not None:
            try:
                return self._http_post(url, body)
            except (AIProviderAPIError, AIProviderResponseError, AIProviderTimeoutError):
                raise
            except urllib.error.URLError as e:
                if "timed out" in str(e).lower():
                    raise AIProviderTimeoutError(f"Ollama request timed out: {e}") from e
                raise AIProviderAPIError(f"Cannot reach Ollama: {e}") from e
            except TimeoutError as e:
                raise AIProviderTimeoutError(f"Ollama request timed out: {e}") from e
            except Exception as e:
                raise AIProviderAPIError(f"Unexpected error calling Ollama: {e}") from e

        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                raise AIProviderTimeoutError(f"Ollama request timed out: {e}") from e
            raise AIProviderAPIError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e
        except TimeoutError as e:
            raise AIProviderTimeoutError(f"Ollama request timed out: {e}") from e
        except json.JSONDecodeError as e:
            raise AIProviderResponseError(f"Ollama response was not valid JSON: {e}") from e
        except Exception as e:
            raise AIProviderAPIError(f"Unexpected error calling Ollama: {e}") from e

        message = data.get("message", {})
        text = message.get("content", "")
        if not text:
            error_msg = data.get("error", "")
            if error_msg:
                raise AIProviderAPIError(f"Ollama returned an error: {error_msg}")
            raise AIProviderResponseError("Ollama response contained no message content.")
        return text

    def _parse_recommendations(self, raw_text: str) -> list[Recommendation]:
        """Same parsing logic as GeminiProvider -- deliberately not
        shared as a base-class method to avoid coupling the two
        providers' internals, but intentionally identical in behavior."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise AIProviderResponseError(
                f"Ollama output was not valid JSON: {e}"
            ) from e

        if not isinstance(data, dict) or "recommendations" not in data:
            raise AIProviderResponseError(
                "Ollama output is missing the required top-level 'recommendations' field."
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

        logger.info("Parsed %d recommendation(s) from Ollama output.", len(recommendations))
        return recommendations

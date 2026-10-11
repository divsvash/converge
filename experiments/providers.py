"""
Provider resolution for experiments and CLI.

Small, explicit factory -- not a plugin system, not a registry
framework. Given a Configuration, returns a ready-to-use AIProvider.

SPARK_ONLY configurations never call this (the runner skips provider
resolution for them). Unknown provider/deployment combinations fail
clearly.
"""
from __future__ import annotations

from converge.services.ai.base import AIProvider
from converge.services.ai.gemini_provider import GeminiProvider
from converge.services.ai.ollama_provider import OllamaProvider
from experiments.models.configuration import Configuration, DeploymentMode


def resolve_provider(configuration: Configuration) -> AIProvider:
    """
    Build an AIProvider from a Configuration's explicit fields.

    Does NOT infer LLM vs SLM or LOCAL vs REMOTE -- those are already
    explicit fields on Configuration by design. This function only
    maps (provider, deployment_mode) to the correct provider class.
    """
    if configuration.deployment_mode == DeploymentMode.REMOTE:
        if configuration.provider == "gemini":
            return GeminiProvider(model=configuration.model)
        raise ValueError(
            f"Unknown remote provider '{configuration.provider}'. "
            f"Supported remote providers: gemini."
        )

    if configuration.deployment_mode == DeploymentMode.LOCAL:
        if configuration.provider == "ollama":
            return OllamaProvider(model_name=configuration.model)
        raise ValueError(
            f"Unknown local provider '{configuration.provider}'. "
            f"Supported local providers: ollama."
        )

    raise ValueError(f"Unsupported deployment_mode '{configuration.deployment_mode}'.")

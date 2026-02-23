"""
infrastructure.llm.llm_builder - Centralized LLM construction.

Single source of truth for building LLM instances across all components
(agent, RAGs, intent parser, safety filter). The provider is controlled
by the LLM_PROVIDER environment variable.

Supported providers:
    - "openai"  → langchain_openai.ChatOpenAI
    - "groq"    → langchain_groq.ChatGroq
    - "ollama"  → langchain_ollama.ChatOllama / OllamaLLM
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from langchain_core.language_models import BaseChatModel, BaseLLM

logger = logging.getLogger(__name__)


def build_llm(
    *,
    provider: str,
    model: str,
    temperature: float = 0,
    ollama_base_url: str = "http://localhost:11434/",
    openai_api_key: str = "",
    groq_api_key: str = "",
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    chat_model: bool = False,
) -> Union[BaseChatModel, BaseLLM]:
    """Build an LLM instance for the given provider.

    Args:
        provider: One of "openai", "groq", "ollama".
        model: Model name for the selected provider.
        temperature: Sampling temperature.
        ollama_base_url: Ollama server URL (only used when provider="ollama").
        openai_api_key: API key for OpenAI.
        groq_api_key: API key for Groq.
        json_mode: Enable structured JSON output mode.
        max_tokens: Maximum tokens. Defaults to 512 for Groq if not set.
        chat_model: When True **and** provider="ollama", use ChatOllama
                    (required for tool-calling agents) instead of OllamaLLM.

    Returns:
        A configured LangChain LLM instance.

    Raises:
        ValueError: If the provider is unknown or required credentials are missing.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER='openai'")

        kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "openai_api_key": openai_api_key,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        logger.info("Building OpenAI LLM (model=%s, json_mode=%s)", model, json_mode)
        return ChatOpenAI(**kwargs)

    elif provider == "groq":
        from langchain_groq import ChatGroq

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER='groq'")

        kwargs = {
            "model": model,
            "temperature": temperature,
            "groq_api_key": groq_api_key,
            "max_tokens": max_tokens if max_tokens is not None else 512,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        logger.info("Building Groq LLM (model=%s, json_mode=%s)", model, json_mode)
        return ChatGroq(**kwargs)

    elif provider == "ollama":
        if chat_model:
            from langchain_ollama import ChatOllama

            kwargs = {
                "model": model,
                "temperature": temperature,
                "base_url": ollama_base_url,
            }
            if json_mode:
                kwargs["format"] = "json"

            logger.info("Building ChatOllama (model=%s, json_mode=%s)", model, json_mode)
            return ChatOllama(**kwargs)
        else:
            from langchain_ollama import OllamaLLM

            kwargs = {
                "model": model,
                "temperature": temperature,
                "base_url": ollama_base_url,
            }
            if json_mode:
                kwargs["format"] = "json"

            logger.info("Building OllamaLLM (model=%s, json_mode=%s)", model, json_mode)
            return OllamaLLM(**kwargs)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Must be 'openai', 'groq', or 'ollama'."
        )

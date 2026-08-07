"""Extractor strategy API."""

from __future__ import annotations

from congreso_open_data.models import ExtractionSpec
from congreso_open_data.protocols import ExtractionContext, ExtractionResult, ExtractorBackend
from congreso_open_data.registry import ExtractorRegistry, default_registry


def create_extractor(
    spec: ExtractionSpec,
    *,
    registry: ExtractorRegistry = default_registry,
    api_key: str | None = None,
    client: object | None = None,
) -> ExtractorBackend:
    if api_key is not None or client is not None:
        from congreso_open_data.extractors.llm import (
            AnthropicExtractor,
            OpenAICompatibleExtractor,
            OpenAIExtractor,
        )

        options = spec.options
        common = {
            "model": spec.model,
            "instructions": options.get("instructions"),
            "timeout": float(options.get("timeout", 60.0)),
            "api_key": api_key,
            "client": client,
        }
        if spec.backend == "openai":
            return OpenAIExtractor(**common)
        if spec.backend == "anthropic":
            return AnthropicExtractor(
                **common,
                max_tokens=int(options.get("max_tokens", 2048)),
            )
        if spec.backend in {"ollama", "openai-compatible"}:
            return OpenAICompatibleExtractor(
                **common,
                base_url=str(options.get("base_url", "http://localhost:11434/v1")),
            )
        raise ValueError("api_key/client injection is supported only for LLM backends")
    return registry.create(spec)


def register_extractor(name: str, backend: ExtractorBackend) -> None:
    default_registry.register_instance(name, backend)


def extract_with_fallback(
    spec: ExtractionSpec,
    content: bytes,
    context: ExtractionContext,
    *,
    registry: ExtractorRegistry = default_registry,
) -> ExtractionResult:
    """Run only the explicitly declared chain, stopping after the first success."""

    attempts = (spec, *spec.fallback)
    failures: list[str] = []
    for attempt in attempts:
        try:
            return registry.create(attempt).extract(content, context)
        except Exception as exc:
            failures.append(f"{attempt.backend}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "All explicitly configured extraction backends failed: " + " | ".join(failures)
    )


__all__ = [
    "ExtractionContext",
    "ExtractionResult",
    "ExtractorBackend",
    "ExtractorRegistry",
    "create_extractor",
    "extract_with_fallback",
    "register_extractor",
]

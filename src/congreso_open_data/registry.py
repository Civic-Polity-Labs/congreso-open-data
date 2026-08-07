"""Collision-safe extractor registry with Python entry-point discovery."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from threading import RLock
from typing import Any, cast

from congreso_open_data.models import ExtractionSpec
from congreso_open_data.protocols import ExtractorBackend

ExtractorFactory = Callable[[ExtractionSpec], ExtractorBackend]

BUILTIN_EXTRACTORS = frozenset(
    {
        "json",
        "csv",
        "xml",
        "html",
        "pypdf",
        "pymupdf",
        "regex",
        "spacy",
        "rapidocr",
        "paddleocr",
        "transformers",
        "openai",
        "anthropic",
        "ollama",
        "openai-compatible",
    }
)


class ExtractorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ExtractorFactory] = {}
        self._instances: dict[str, ExtractorBackend] = {}
        self._lock = RLock()

    def register_factory(
        self,
        name: str,
        factory: ExtractorFactory,
        *,
        builtin: bool = False,
    ) -> None:
        key = self._key(name)
        with self._lock:
            if key in self._factories or key in self._instances:
                raise ValueError(f"Extractor backend already registered: {key}")
            if key in BUILTIN_EXTRACTORS and not builtin:
                raise ValueError(f"Extractor name is reserved by the package: {key}")
            self._factories[key] = factory

    def register_instance(self, name: str, backend: ExtractorBackend) -> None:
        key = self._key(name)
        with self._lock:
            if key in BUILTIN_EXTRACTORS:
                raise ValueError(f"Extractor name is reserved by the package: {key}")
            if key in self._factories or key in self._instances:
                raise ValueError(f"Extractor backend already registered: {key}")
            self._instances[key] = backend

    def create(self, spec: ExtractionSpec) -> ExtractorBackend:
        key = self._key(spec.backend)
        with self._lock:
            instance = self._instances.get(key)
            factory = self._factories.get(key)
        if instance is not None:
            return instance
        if factory is None:
            raise LookupError(
                f"Unknown extractor backend {key!r}. Install its optional extra, "
                "register an instance, or install a congreso_open_data.extractors plugin."
            )
        backend = factory(spec)
        if backend.engine != spec.engine:
            raise ValueError(
                f"Backend {key!r} belongs to engine {backend.engine!r}, not {spec.engine!r}"
            )
        return backend

    def discover_entry_points(self) -> tuple[str, ...]:
        loaded: list[str] = []
        entry_points = metadata.entry_points(group="congreso_open_data.extractors")
        for entry_point in sorted(entry_points, key=lambda item: item.name):
            key = self._key(entry_point.name)
            if key in BUILTIN_EXTRACTORS:
                # The distribution's own declarations document discovery; built-ins are
                # registered directly so optional imports stay lazy.
                continue
            factory_or_type = entry_point.load()

            def factory(
                spec: ExtractionSpec, loaded_object: Any = factory_or_type
            ) -> ExtractorBackend:
                try:
                    return cast(ExtractorBackend, loaded_object(spec))
                except TypeError:
                    return cast(ExtractorBackend, loaded_object())

            self.register_factory(key, factory)
            loaded.append(key)
        return tuple(loaded)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(set(self._factories) | set(self._instances)))

    @staticmethod
    def _key(name: str) -> str:
        key = name.strip().casefold()
        if not key:
            raise ValueError("Extractor backend name cannot be empty")
        return key


default_registry = ExtractorRegistry()


def register_builtins(registry: ExtractorRegistry = default_registry) -> None:
    from congreso_open_data.extractors.llm import (
        AnthropicExtractor,
        OpenAICompatibleExtractor,
        OpenAIExtractor,
    )
    from congreso_open_data.extractors.native import (
        CsvExtractor,
        HtmlExtractor,
        JsonExtractor,
        PyMuPDFExtractor,
        PyPdfExtractor,
        XmlExtractor,
    )
    from congreso_open_data.extractors.nlp import SpacyExtractor
    from congreso_open_data.extractors.ocr import (
        PaddleOcrExtractor,
        RapidOcrExtractor,
        TransformersOcrExtractor,
    )
    from congreso_open_data.extractors.rules import RegexExtractor

    factories: dict[str, Callable[[ExtractionSpec], Any]] = {
        "json": JsonExtractor.from_spec,
        "csv": CsvExtractor.from_spec,
        "xml": XmlExtractor.from_spec,
        "html": HtmlExtractor.from_spec,
        "pypdf": PyPdfExtractor.from_spec,
        "pymupdf": PyMuPDFExtractor.from_spec,
        "regex": RegexExtractor.from_spec,
        "spacy": SpacyExtractor.from_spec,
        "rapidocr": RapidOcrExtractor.from_spec,
        "paddleocr": PaddleOcrExtractor.from_spec,
        "transformers": TransformersOcrExtractor.from_spec,
        "openai": OpenAIExtractor.from_spec,
        "anthropic": AnthropicExtractor.from_spec,
        "ollama": OpenAICompatibleExtractor.from_spec,
        "openai-compatible": OpenAICompatibleExtractor.from_spec,
    }
    for name, factory in factories.items():
        if name not in registry.names():
            registry.register_factory(name, cast(ExtractorFactory, factory), builtin=True)


register_builtins()

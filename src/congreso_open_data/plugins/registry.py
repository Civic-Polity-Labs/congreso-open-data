"""Thread-safe runtime registry with lazy Python entry-point discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import metadata
from threading import RLock
from typing import Any, Literal, cast

from congreso_open_data.models import ExtractionSpec
from congreso_open_data.plugins.models import (
    CallableModelBackend,
    ModelBackend,
    ModelCallable,
    ModelUnavailableError,
)

ModelFactory = Callable[[ExtractionSpec], ModelBackend]
MODEL_ENTRY_POINT_GROUP = "congreso_open_data.models"


class ModelRegistry:
    def __init__(self, models: Mapping[str, ModelBackend] | None = None) -> None:
        self._instances: dict[str, ModelBackend] = {}
        self._factories: dict[str, ModelFactory] = {}
        self._lock = RLock()
        for name, backend in (models or {}).items():
            self.register(name, backend)

    def register(self, name: str, backend: ModelBackend) -> None:
        key = self._key(name)
        if not isinstance(backend, ModelBackend):
            raise TypeError("Model backend does not implement ModelBackend")
        with self._lock:
            self._ensure_available(key)
            self._instances[key] = backend

    def register_factory(self, name: str, factory: ModelFactory) -> None:
        key = self._key(name)
        with self._lock:
            self._ensure_available(key)
            self._factories[key] = factory

    def register_callable(
        self,
        name: str,
        *,
        model: str,
        version: str,
        function: ModelCallable,
        provider: str | None = None,
        engine: Literal["llm", "nlp", "ocr", "custom"] = "llm",
    ) -> CallableModelBackend:
        """Register an ordinary synchronous callable with explicit provenance."""

        if engine not in {"llm", "nlp", "ocr", "custom"}:
            raise ValueError(f"Unsupported model engine: {engine}")
        backend = CallableModelBackend(
            name=name,
            model=model,
            version=version,
            function=function,
            provider=provider,
            engine=engine,
        )
        self.register(name, backend)
        return backend

    def contains(self, name: str) -> bool:
        key = self._key(name)
        with self._lock:
            if key in self._instances or key in self._factories:
                return True
        return any(item.name.casefold() == key for item in self._entry_points())

    def create(self, spec: ExtractionSpec) -> ModelBackend:
        key = self._key(spec.backend)
        with self._lock:
            instance = self._instances.get(key)
            factory = self._factories.get(key)
        if instance is not None:
            if instance.descriptor.model != spec.model:
                raise ValueError(
                    f"Registered backend {key!r} identifies model "
                    f"{instance.descriptor.model!r}, not {spec.model!r}"
                )
            return instance
        if factory is None:
            factory = self._load_entry_point(key)
        if factory is None:
            raise ModelUnavailableError(
                f"Unknown model backend {key!r}. Register a runtime backend or install "
                f"a {MODEL_ENTRY_POINT_GROUP} plugin."
            )
        backend = factory(spec)
        if not isinstance(backend, ModelBackend):
            raise TypeError(f"Model factory {key!r} did not return ModelBackend")
        if backend.descriptor.model != spec.model:
            raise ValueError(
                f"Model factory {key!r} identifies model "
                f"{backend.descriptor.model!r}, not {spec.model!r}"
            )
        return backend

    def names(self) -> tuple[str, ...]:
        with self._lock:
            local = set(self._instances) | set(self._factories)
        discovered = {item.name.casefold() for item in self._entry_points()}
        return tuple(sorted(local | discovered))

    def _load_entry_point(self, key: str) -> ModelFactory | None:
        matches = [item for item in self._entry_points() if item.name.casefold() == key]
        if not matches:
            return None
        if len(matches) > 1:
            distributions = sorted(str(item.dist) for item in matches)
            raise ModelUnavailableError(f"Multiple model plugins use name {key!r}: {distributions}")
        loaded = matches[0].load()
        if not callable(loaded):
            raise TypeError(f"Model plugin {key!r} must expose a callable factory")
        factory = cast(ModelFactory, loaded)
        with self._lock:
            self._ensure_available(key)
            self._factories[key] = factory
        return factory

    @staticmethod
    def _entry_points() -> tuple[Any, ...]:
        return tuple(metadata.entry_points(group=MODEL_ENTRY_POINT_GROUP))

    def _ensure_available(self, key: str) -> None:
        if key in self._instances or key in self._factories:
            raise ValueError(f"Model backend already registered: {key}")

    @staticmethod
    def _key(name: str) -> str:
        key = str(name).strip().casefold()
        if not key:
            raise ValueError("Model backend name cannot be empty")
        return key

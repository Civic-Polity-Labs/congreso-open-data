from __future__ import annotations

from datetime import UTC, datetime

import pytest

from congreso_open_data import ExtractionSpec, SourceRef
from congreso_open_data.plugins import (
    CallableModelBackend,
    ExtractionContext,
    ExtractionLimits,
    ExtractionTask,
    ModelRegistry,
    ModelResponseError,
    StructuredModelExtractor,
)


def _source() -> SourceRef:
    return SourceRef(
        requested_url="https://example.test/transcript",
        sha256="a" * 64,
        fetched_at=datetime.now(UTC),
        adapter="test",
        adapter_version="1",
        normalization_version="1",
    )


def test_plain_callable_is_provider_agnostic_and_retains_all_literal_spans() -> None:
    registry = ModelRegistry()
    registry.register_callable(
        "mine",
        model="unit-model",
        version="2026-08-09",
        provider="self-hosted",
        function=lambda request: {
            "candidates": [
                {
                    "kind": "topic",
                    "value": {"label": "vivienda"},
                    "quote": "vivienda",
                    "confidence": 0.9,
                }
            ]
        },
    )
    task = ExtractionTask(
        name="topics",
        instructions="Extract topics supported by literal quotes.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="unit-model"),
    )
    backend = registry.create(task.backend)

    result = StructuredModelExtractor(backend, task).extract(
        "vivienda pública y vivienda asequible".encode(),
        ExtractionContext(source=_source(), mime_type="text/plain"),
    )

    assert result.candidates[0].status == "review_required"
    assert result.candidates[0].source.model == "unit-model"
    assert [(item.span_start, item.span_end) for item in result.candidates[0].evidence] == [
        (0, 8),
        (19, 27),
    ]
    assert all(item.literal for item in result.candidates[0].evidence)
    assert result.diagnostics["provider_diagnostics"] == {}


def test_non_literal_model_quote_is_explicitly_reviewable() -> None:
    registry = ModelRegistry()
    registry.register_callable(
        "mine",
        model="unit-model",
        version="1",
        function=lambda request: {
            "candidates": [{"kind": "claim", "value": True, "quote": "invented"}]
        },
    )
    task = ExtractionTask(
        name="claims",
        instructions="Extract claims.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="unit-model"),
    )

    result = StructuredModelExtractor(registry.create(task.backend), task).extract(
        b"literal source",
        ExtractionContext(source=_source(), mime_type="text/plain"),
    )

    evidence = result.candidates[0].evidence[0]
    assert evidence.literal is False
    assert evidence.span_start is None
    assert evidence.diagnostics["quote_occurrences"] == 0


def test_invalid_model_payload_fails_closed() -> None:
    registry = ModelRegistry()
    registry.register_callable(
        "mine",
        model="unit-model",
        version="1",
        function=lambda request: "not-json",
    )
    task = ExtractionTask(
        name="claims",
        instructions="Extract claims.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="unit-model"),
    )

    with pytest.raises(ModelResponseError):
        StructuredModelExtractor(registry.create(task.backend), task).extract(
            b"source",
            ExtractionContext(source=_source(), mime_type="text/plain"),
        )


def test_serializable_task_rejects_runtime_objects_in_legacy_spec_options() -> None:
    spec = ExtractionSpec(
        engine="llm",
        backend="mine",
        model="unit-model",
        options={"client": object()},
    )

    with pytest.raises(ValueError, match="ModelRegistry"):
        ExtractionTask(
            name="invalid-runtime",
            instructions="Extract.",
            backend=spec,
        )


def test_registry_requires_exact_model_identity() -> None:
    registry = ModelRegistry()
    registry.register_callable(
        "mine",
        model="unit-model",
        version="1",
        function=lambda request: {"candidates": []},
    )

    with pytest.raises(ValueError, match="other-model"):
        registry.create(ExtractionSpec(engine="llm", backend="mine", model="other-model"))


def test_registry_factory_requires_exact_model_identity() -> None:
    registry = ModelRegistry()
    registry.register_factory(
        "factory",
        lambda spec: CallableModelBackend(
            name="factory",
            model="factory-chose-a-different-model",
            version="1",
            function=lambda request: {"candidates": []},
        ),
    )

    with pytest.raises(ValueError, match="requested-model"):
        registry.create(ExtractionSpec(engine="llm", backend="factory", model="requested-model"))


def test_long_inputs_are_bounded_chunked_and_overlap_candidates_are_deduplicated() -> None:
    requests: list[tuple[int, str]] = []

    def model(request):
        requests.append((request.metadata["chunk_start"], request.text))
        candidates = []
        if "evidence" in request.text:
            candidates.append({"kind": "marker", "value": "found", "quote": "evidence"})
        return {"candidates": candidates}

    registry = ModelRegistry()
    registry.register_callable(
        "mine",
        model="unit-model",
        version="1",
        function=model,
    )
    task = ExtractionTask(
        name="bounded",
        instructions="Find the marker.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="unit-model"),
        limits=ExtractionLimits(
            max_input_characters=100,
            chunk_characters=20,
            chunk_overlap_characters=10,
            max_chunks=20,
        ),
    )
    source = "0123456789evidence0123456789"

    result = StructuredModelExtractor(registry.create(task.backend), task).extract(
        source.encode(),
        ExtractionContext(source=_source(), mime_type="text/plain"),
    )

    assert len(requests) > 1
    assert all(len(text) <= 20 for _, text in requests)
    assert len(result.candidates) == 1
    assert [(item.span_start, item.span_end) for item in result.candidates[0].evidence] == [
        (10, 18)
    ]
    assert result.diagnostics["chunk_count"] == len(requests)

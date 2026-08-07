from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from congreso_open_data.extractors import create_extractor, extract_with_fallback
from congreso_open_data.extractors.llm import (
    AnthropicExtractor,
    OpenAICompatibleExtractor,
    OpenAIExtractor,
)
from congreso_open_data.extractors.native import (
    CsvExtractor,
    HtmlExtractor,
    JsonExtractor,
    XmlExtractor,
)
from congreso_open_data.models import ExtractionSpec, SourceRef
from congreso_open_data.protocols import ExtractionContext, ExtractionResult
from congreso_open_data.registry import ExtractorRegistry


def context() -> ExtractionContext:
    return ExtractionContext(
        source=SourceRef(
            requested_url="https://example.test/source",
            sha256="c" * 64,
            adapter="test",
            adapter_version="1",
            normalization_version="1",
        )
    )


def test_native_json_is_literal() -> None:
    backend = JsonExtractor(model="stdlib-json")
    result = backend.extract(b'{"nombre":"Ana"}', context())

    assert result.texts == ('{"nombre":"Ana"}',)
    assert result.evidence[0].literal is True
    assert result.evidence[0].confidence == 1.0


def test_csv_xml_and_html_native_backends_are_literal() -> None:
    csv_result = CsvExtractor().extract(b"name,value\nAna,1\nLuis,2\n", context())
    xml_result = XmlExtractor().extract(b"<root><name>Ana</name><value>1</value></root>", context())
    html_result = HtmlExtractor().extract(
        b"<html><script>ignored()</script><body><p>Ana</p></body></html>", context()
    )

    assert len(csv_result.texts) == 2
    assert csv_result.diagnostics["row_count"] == 2
    assert xml_result.texts == ("Ana", "1")
    assert html_result.texts == ("Ana",)


def test_regex_backend_returns_reviewable_candidate_with_span() -> None:
    backend = create_extractor(
        ExtractionSpec(
            engine="rules",
            backend="regex",
            model="rules-2026-08",
            options={
                "rules": [{"name": "amount", "pattern": r"(?P<amount>\d+) EUR", "group": "amount"}]
            },
        )
    )

    result = backend.extract(b"Importe 125 EUR.", context())

    assert result.candidates[0].value == "125"
    assert result.candidates[0].status == "review_required"
    assert result.candidates[0].inferred is False
    assert result.candidates[0].evidence[0].span_start == 8


class FakeResponses:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output})()


class FakeOpenAI:
    def __init__(self, output: str) -> None:
        self.responses = FakeResponses(output)


class FakeMessages:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = type("Block", (), {"text": self.output})()
        return type("Message", (), {"content": [block]})()


class FakeAnthropic:
    def __init__(self, output: str) -> None:
        self.messages = FakeMessages(output)


class FakeCompletions:
    def __init__(self, output: str) -> None:
        self.output = output

    def create(self, **kwargs):
        message = type("Message", (), {"content": self.output})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


class FakeCompatible:
    def __init__(self, output: str) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions(output)})()


def test_openai_uses_responses_structured_output_and_never_promotes_candidate() -> None:
    client = FakeOpenAI(
        json.dumps(
            {"candidates": [{"kind": "person", "value": "Ana", "quote": "Ana", "confidence": 0.8}]}
        )
    )
    backend = OpenAIExtractor(model="chosen-by-user", api_key="not-logged", client=client)

    result = backend.extract(b"Interviene Ana.", context())

    call = client.responses.calls[0]
    assert call["model"] == "chosen-by-user"
    assert call["text"]["format"]["type"] == "json_schema"
    assert result.candidates[0].status == "review_required"
    assert result.candidates[0].evidence[0].literal is True
    assert "not-logged" not in repr(backend)


def test_llm_non_literal_quote_is_flagged_and_invalid_payload_fails() -> None:
    non_literal = OpenAIExtractor(
        model="test",
        client=FakeOpenAI(
            json.dumps(
                {
                    "candidates": [
                        {
                            "kind": "person",
                            "value": "Inventado",
                            "quote": "Inventado",
                            "confidence": 0.2,
                        }
                    ]
                }
            )
        ),
    )
    assert non_literal.extract(b"Solo evidencia real", context()).evidence[0].literal is False

    invalid = OpenAIExtractor(model="test", client=FakeOpenAI("{}"))
    with pytest.raises(ValueError, match="candidates"):
        invalid.extract(b"source", context())


def test_api_key_can_be_injected_without_entering_spec() -> None:
    backend = create_extractor(
        ExtractionSpec(engine="llm", backend="openai", model="chosen"),
        api_key="runtime-only",
        client=FakeOpenAI('{"candidates":[]}'),
    )

    assert isinstance(backend, OpenAIExtractor)
    assert "runtime-only" not in repr(backend)


def test_anthropic_and_openai_compatible_clients_are_selectable() -> None:
    payload = json.dumps(
        {"candidates": [{"kind": "person", "value": "Ana", "quote": "Ana", "confidence": 1.0}]}
    )
    anthropic = AnthropicExtractor(model="user-model", client=FakeAnthropic(payload))
    compatible = OpenAICompatibleExtractor(model="local-model", client=FakeCompatible(payload))

    assert anthropic.extract(b"Ana", context()).candidates[0].value == "Ana"
    assert compatible.extract(b"Ana", context()).candidates[0].value == "Ana"


@dataclass
class FailingBackend:
    model: str = "fail"
    name: str = "custom-fail"
    engine: str = "native"
    version: str = "1"

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        raise TimeoutError("expected")


@dataclass
class WorkingBackend:
    model: str = "ok"
    name: str = "custom-ok"
    engine: str = "native"
    version: str = "1"

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        return ExtractionResult(texts=(content.decode(),))


def test_only_declared_fallback_chain_runs() -> None:
    registry = ExtractorRegistry()
    registry.register_factory("custom-fail", lambda spec: FailingBackend())
    registry.register_factory("custom-ok", lambda spec: WorkingBackend())
    spec = ExtractionSpec(
        engine="native",
        backend="custom-fail",
        model="fail",
        fallback=(ExtractionSpec(engine="native", backend="custom-ok", model="ok"),),
    )

    result = extract_with_fallback(spec, b"bounded", context(), registry=registry)

    assert result.texts == ("bounded",)


def test_builtin_names_are_reserved() -> None:
    registry = ExtractorRegistry()
    with pytest.raises(ValueError, match="reserved"):
        registry.register_factory("openai", lambda spec: WorkingBackend())


def test_unknown_and_duplicate_custom_backends_fail_explicitly() -> None:
    registry = ExtractorRegistry()
    registry.register_factory("custom", lambda spec: WorkingBackend())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_factory("custom", lambda spec: WorkingBackend())
    with pytest.raises(LookupError, match="Unknown extractor"):
        registry.create(ExtractionSpec(engine="native", backend="absent", model="none"))

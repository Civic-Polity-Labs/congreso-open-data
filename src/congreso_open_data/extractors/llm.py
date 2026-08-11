"""Explicit cloud/local LLM adapters. No provider is called implicitly."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from congreso_open_data.models import ExtractionCandidate, ExtractionEvidence, ExtractionSpec
from congreso_open_data.protocols import ExtractionContext, ExtractionResult

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "value": {},
                    "quote": {"type": "string"},
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                },
                "required": ["kind", "value", "quote", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def _prompt(text: str, instructions: str | None) -> str:
    task = instructions or "Extract only the requested fields from the supplied source."
    return (
        f"{task}\n"
        "Return JSON matching the schema. Every candidate must include a verbatim quote. "
        "Do not invent absent values.\n\nSOURCE:\n"
        f"{text}"
    )


def _payload(raw: Any) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else json.loads(str(raw))
    if not isinstance(value.get("candidates"), list):
        raise ValueError("Provider response does not contain a candidates list")
    return value


def _result(
    *,
    payload: dict[str, Any],
    source_text: str,
    context: ExtractionContext,
    backend: str,
    model: str,
    version: str,
) -> ExtractionResult:
    candidates: list[ExtractionCandidate] = []
    evidence: list[ExtractionEvidence] = []
    for index, raw in enumerate(payload["candidates"]):
        if not isinstance(raw, dict) or not raw.get("kind"):
            raise ValueError(f"Invalid provider candidate at index {index}")
        quote = str(raw.get("quote") or "")
        spans = (
            [(match.start(), match.end()) for match in re.finditer(re.escape(quote), source_text)]
            if quote
            else []
        )
        item_evidence = tuple(
            ExtractionEvidence(
                text=quote or None,
                span_start=start if spans else None,
                span_end=end if spans else None,
                confidence=raw.get("confidence"),
                backend=backend,
                model=model,
                version=version,
                literal=bool(spans),
                diagnostics={
                    "provider_index": index,
                    "quote_occurrences": len(spans),
                    "occurrence_index": occurrence if spans else None,
                },
            )
            for occurrence, (start, end) in enumerate(spans or [(-1, -1)])
        )
        digest = hashlib.sha256(
            f"{context.source.sha256}:{backend}:{model}:{index}:{raw.get('kind')}".encode()
        ).hexdigest()
        candidates.append(
            ExtractionCandidate(
                candidate_id=f"{backend}:{digest[:24]}",
                kind=str(raw["kind"]),
                value=raw.get("value"),
                evidence=item_evidence,
                source=context.source.model_copy(update={"method": backend, "model": model}),
            )
        )
        evidence.extend(item_evidence)
    return ExtractionResult(
        texts=(source_text,),
        candidates=candidates,
        evidence=evidence,
        diagnostics={
            "candidate_count": len(candidates),
            "non_literal_evidence": sum(not item.literal for item in evidence),
        },
    )


@dataclass(frozen=True)
class OpenAIExtractor:
    model: str
    instructions: str | None = None
    timeout: float = 60.0
    api_key: str | None = field(default=None, repr=False, compare=False)
    client: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "openai"
    engine: ClassVar[str] = "llm"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> OpenAIExtractor:
        return cls(
            model=spec.model,
            instructions=spec.options.get("instructions"),
            timeout=float(spec.options.get("timeout", 60.0)),
            client=spec.options.get("client"),
        )

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        source_text = content.decode(
            str(context.metadata.get("encoding", "utf-8")), errors="replace"
        )
        client = self.client
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install congreso-open-data[openai] for OpenAI") from exc
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for the OpenAI backend")
            client = OpenAI(api_key=api_key, timeout=self.timeout)
        response = client.responses.create(
            model=self.model,
            input=_prompt(source_text, self.instructions),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "congreso_extraction",
                    "schema": _SCHEMA,
                    "strict": True,
                }
            },
        )
        return _result(
            payload=_payload(response.output_text),
            source_text=source_text,
            context=context,
            backend=self.name,
            model=self.model,
            version=self.version,
        )


@dataclass(frozen=True)
class AnthropicExtractor:
    model: str
    instructions: str | None = None
    timeout: float = 60.0
    max_tokens: int = 2048
    api_key: str | None = field(default=None, repr=False, compare=False)
    client: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "anthropic"
    engine: ClassVar[str] = "llm"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> AnthropicExtractor:
        return cls(
            model=spec.model,
            instructions=spec.options.get("instructions"),
            timeout=float(spec.options.get("timeout", 60.0)),
            max_tokens=int(spec.options.get("max_tokens", 2048)),
            client=spec.options.get("client"),
        )

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        source_text = content.decode(
            str(context.metadata.get("encoding", "utf-8")), errors="replace"
        )
        client = self.client
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("Install congreso-open-data[anthropic] for Anthropic") from exc
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic backend")
            client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": _prompt(source_text, self.instructions)}],
            system=(
                "Return only a JSON object matching this JSON Schema: "
                + json.dumps(_SCHEMA, separators=(",", ":"))
            ),
        )
        raw = "".join(str(block.text) for block in response.content if hasattr(block, "text"))
        return _result(
            payload=_payload(raw),
            source_text=source_text,
            context=context,
            backend=self.name,
            model=self.model,
            version=self.version,
        )


@dataclass(frozen=True)
class OpenAICompatibleExtractor:
    model: str
    base_url: str = "http://localhost:11434/v1"
    instructions: str | None = None
    timeout: float = 60.0
    api_key: str | None = field(default=None, repr=False, compare=False)
    client: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "openai-compatible"
    engine: ClassVar[str] = "llm"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> OpenAICompatibleExtractor:
        return cls(
            model=spec.model,
            base_url=str(spec.options.get("base_url", "http://localhost:11434/v1")),
            instructions=spec.options.get("instructions"),
            timeout=float(spec.options.get("timeout", 60.0)),
            client=spec.options.get("client"),
        )

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        source_text = content.decode(
            str(context.metadata.get("encoding", "utf-8")), errors="replace"
        )
        client = self.client
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install congreso-open-data[local] for local LLMs") from exc
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "local-not-a-secret",
                timeout=self.timeout,
            )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": _prompt(source_text, self.instructions)}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return _result(
            payload=_payload(raw),
            source_text=source_text,
            context=context,
            backend=self.name,
            model=self.model,
            version=self.version,
        )

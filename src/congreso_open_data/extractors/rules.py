"""Configurable deterministic regular-expression extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, ClassVar

from congreso_open_data.models import (
    ExtractionCandidate,
    ExtractionEvidence,
    ExtractionSpec,
)
from congreso_open_data.protocols import ExtractionContext, ExtractionResult


@dataclass(frozen=True)
class RegexRule:
    name: str
    pattern: str
    flags: int = re.IGNORECASE | re.MULTILINE
    group: str | int = 0


@dataclass(frozen=True)
class RegexExtractor:
    model: str
    rules: tuple[RegexRule, ...]
    encoding: str = "utf-8"
    name: ClassVar[str] = "regex"
    engine: ClassVar[str] = "rules"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> RegexExtractor:
        raw_rules = spec.options.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("Regex backend requires options.rules with at least one rule")
        rules: list[RegexRule] = []
        for raw in raw_rules:
            if not isinstance(raw, dict) or not raw.get("name") or not raw.get("pattern"):
                raise ValueError("Every regex rule requires name and pattern")
            rules.append(
                RegexRule(
                    name=str(raw["name"]),
                    pattern=str(raw["pattern"]),
                    flags=int(raw.get("flags", re.IGNORECASE | re.MULTILINE)),
                    group=raw.get("group", 0),
                )
            )
        return cls(
            model=spec.model,
            rules=tuple(rules),
            encoding=str(spec.options.get("encoding", "utf-8")),
        )

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        text = content.decode(self.encoding, errors="replace")
        candidates: list[ExtractionCandidate] = []
        evidence: list[ExtractionEvidence] = []
        for rule in self.rules:
            expression = re.compile(rule.pattern, rule.flags)
            for match in expression.finditer(text):
                value: Any = match.group(rule.group)
                start, end = match.span(rule.group)
                item_evidence = ExtractionEvidence(
                    text=text[start:end],
                    span_start=start,
                    span_end=end,
                    confidence=1.0,
                    backend=self.name,
                    model=self.model,
                    version=self.version,
                    literal=True,
                    diagnostics={"rule": rule.name},
                )
                digest = hashlib.sha256(
                    f"{context.source.sha256}:{rule.name}:{start}:{end}".encode()
                ).hexdigest()
                candidates.append(
                    ExtractionCandidate(
                        candidate_id=f"regex:{digest[:24]}",
                        kind=rule.name,
                        value=value,
                        evidence=(item_evidence,),
                        source=context.source,
                        inferred=False,
                    )
                )
                evidence.append(item_evidence)
        return ExtractionResult(
            texts=(text,),
            candidates=candidates,
            evidence=evidence,
            diagnostics={"rules": len(self.rules), "matches": len(candidates)},
        )

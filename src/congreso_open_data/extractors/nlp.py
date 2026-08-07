"""Optional NLP extraction. All model output remains review-gated."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, ClassVar

from congreso_open_data.models import ExtractionCandidate, ExtractionEvidence, ExtractionSpec
from congreso_open_data.protocols import ExtractionContext, ExtractionResult


@dataclass(frozen=True)
class SpacyExtractor:
    model: str
    labels: tuple[str, ...] = ()
    _pipeline: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "spacy"
    engine: ClassVar[str] = "nlp"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> SpacyExtractor:
        labels = tuple(str(item) for item in spec.options.get("labels", ()))
        return cls(model=spec.model, labels=labels, _pipeline=spec.options.get("pipeline"))

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        text = content.decode(str(context.metadata.get("encoding", "utf-8")), errors="replace")
        pipeline = self._pipeline
        if pipeline is None:
            try:
                import spacy
            except ImportError as exc:
                raise RuntimeError("Install congreso-open-data[nlp] for the spaCy backend") from exc
            pipeline = spacy.load(self.model)
        document = pipeline(text)
        candidates: list[ExtractionCandidate] = []
        evidence: list[ExtractionEvidence] = []
        for entity in document.ents:
            label = str(entity.label_)
            if self.labels and label not in self.labels:
                continue
            item_evidence = ExtractionEvidence(
                text=str(entity.text),
                span_start=int(entity.start_char),
                span_end=int(entity.end_char),
                confidence=None,
                backend=self.name,
                model=self.model,
                version=self.version,
                literal=True,
                diagnostics={"label": label},
            )
            digest = hashlib.sha256(
                f"{context.source.sha256}:{label}:{entity.start_char}:{entity.end_char}".encode()
            ).hexdigest()
            candidates.append(
                ExtractionCandidate(
                    candidate_id=f"spacy:{digest[:24]}",
                    kind=label,
                    value=str(entity.text),
                    evidence=(item_evidence,),
                    source=context.source,
                )
            )
            evidence.append(item_evidence)
        return ExtractionResult(
            texts=(text,),
            candidates=candidates,
            evidence=evidence,
            diagnostics={"entities": len(candidates)},
        )

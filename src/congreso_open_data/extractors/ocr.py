"""Optional OCR strategies with explicit model and backend selection."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, ClassVar

from congreso_open_data.models import ExtractionEvidence, ExtractionSpec
from congreso_open_data.protocols import ExtractionContext, ExtractionResult


def _image(content: bytes) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install an OCR extra to decode images") from exc
    return Image.open(io.BytesIO(content)).convert("RGB")


def _confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, number))


@dataclass(frozen=True)
class RapidOcrExtractor:
    model: str
    _engine: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "rapidocr"
    engine: ClassVar[str] = "ocr"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> RapidOcrExtractor:
        return cls(model=spec.model, _engine=spec.options.get("engine_instance"))

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        engine = self._engine
        if engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise RuntimeError("Install congreso-open-data[ocr] for RapidOCR") from exc
            engine = RapidOCR()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install congreso-open-data[ocr] for RapidOCR") from exc
        raw, _ = engine(np.asarray(_image(content)))
        lines = raw or []
        texts: list[str] = []
        evidence: list[ExtractionEvidence] = []
        for item in lines:
            bbox, text, score = item[0], str(item[1]), item[2]
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
            texts.append(text)
            evidence.append(
                ExtractionEvidence(
                    text=text,
                    page=int(context.metadata.get("page", 1)),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=_confidence(score),
                    backend=self.name,
                    model=self.model,
                    version=self.version,
                    literal=True,
                )
            )
        return ExtractionResult(
            texts=("\n".join(texts),),
            evidence=evidence,
            diagnostics={"lines": len(texts)},
        )


@dataclass(frozen=True)
class PaddleOcrExtractor:
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    _engine: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "paddleocr"
    engine: ClassVar[str] = "ocr"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> PaddleOcrExtractor:
        options = {key: value for key, value in spec.options.items() if key != "engine_instance"}
        return cls(model=spec.model, options=options, _engine=spec.options.get("engine_instance"))

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        engine = self._engine
        if engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("Install congreso-open-data[paddle] for PaddleOCR") from exc
            engine = PaddleOCR(**self.options)
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install congreso-open-data[paddle] for PaddleOCR") from exc
        result = engine.ocr(np.asarray(_image(content)))
        rows = result[0] if result and isinstance(result[0], list) else result or []
        texts: list[str] = []
        evidence: list[ExtractionEvidence] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            bbox, value = row[0], row[1]
            text = str(value[0] if isinstance(value, (list, tuple)) else value)
            score = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else None
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
            texts.append(text)
            evidence.append(
                ExtractionEvidence(
                    text=text,
                    page=int(context.metadata.get("page", 1)),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=_confidence(score),
                    backend=self.name,
                    model=self.model,
                    version=self.version,
                    literal=True,
                )
            )
        return ExtractionResult(
            texts=("\n".join(texts),),
            evidence=evidence,
            diagnostics={"lines": len(texts)},
        )


@dataclass(frozen=True)
class TransformersOcrExtractor:
    model: str
    task: str = "image-to-text"
    _pipeline: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "transformers"
    engine: ClassVar[str] = "ocr"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> TransformersOcrExtractor:
        return cls(
            model=spec.model,
            task=str(spec.options.get("task", "image-to-text")),
            _pipeline=spec.options.get("pipeline"),
        )

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        pipeline = self._pipeline
        if pipeline is None:
            try:
                from transformers import pipeline as create_pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Install congreso-open-data[transformers] for local Transformers OCR"
                ) from exc
            pipeline = create_pipeline(self.task, model=self.model)
        output = pipeline(_image(content))
        if isinstance(output, dict):
            output = [output]
        texts = tuple(
            str(item.get("generated_text") or item.get("text") or "")
            for item in output
            if isinstance(item, dict)
        )
        evidence = tuple(
            ExtractionEvidence(
                text=text,
                page=int(context.metadata.get("page", 1)),
                confidence=None,
                backend=self.name,
                model=self.model,
                version=self.version,
                literal=True,
            )
            for text in texts
        )
        return ExtractionResult(texts=texts, evidence=evidence, diagnostics={"outputs": len(texts)})

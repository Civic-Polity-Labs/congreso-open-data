"""Optional OCR strategies with explicit model and backend selection."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from congreso_open_data.models import ExtractionEvidence, ExtractionSpec
from congreso_open_data.protocols import ExtractionContext, ExtractionResult
from congreso_open_data.rapidocr_compat import create_rapidocr_engine, rapidocr_lines


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
            engine = create_rapidocr_engine()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install congreso-open-data[ocr] for RapidOCR") from exc
        lines = rapidocr_lines(engine(np.asarray(_image(content))))
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
    task: Literal["image-text-to-text"] = "image-text-to-text"
    prompt: str = "Transcribe all visible text exactly. Do not summarize or infer missing text."
    _pipeline: Any = field(default=None, repr=False, compare=False)
    name: ClassVar[str] = "transformers"
    engine: ClassVar[str] = "ocr"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> TransformersOcrExtractor:
        task = str(spec.options.get("task", "image-text-to-text"))
        # Transformers 5 removed the legacy image-to-text pipeline. Accept the
        # old spelling while configurations migrate, but always execute the
        # supported image-text-to-text contract.
        if task == "image-to-text":
            task = "image-text-to-text"
        if task != "image-text-to-text":
            raise ValueError("Transformers OCR task must be 'image-text-to-text'")
        return cls(
            model=spec.model,
            task="image-text-to-text",
            prompt=str(
                spec.options.get(
                    "prompt",
                    "Transcribe all visible text exactly. Do not summarize or infer missing text.",
                )
            ),
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
        output = pipeline(images=_image(content), text=self.prompt)
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
                # Generative vision output can hallucinate even when prompted
                # for transcription. It is evidence to review, not a literal
                # source span.
                literal=False,
                diagnostics={"review_required": True, "task": self.task},
            )
            for text in texts
        )
        return ExtractionResult(texts=texts, evidence=evidence, diagnostics={"outputs": len(texts)})

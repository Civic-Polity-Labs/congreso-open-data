from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from congreso_open_data.extractors.nlp import SpacyExtractor
from congreso_open_data.extractors.ocr import (
    PaddleOcrExtractor,
    RapidOcrExtractor,
    TransformersOcrExtractor,
)
from congreso_open_data.models import ExtractionSpec, SourceRef
from congreso_open_data.protocols import ExtractionContext


def _png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 4), color="white").save(stream, format="PNG")
    return stream.getvalue()


def _context() -> ExtractionContext:
    return ExtractionContext(
        source=SourceRef(
            requested_url="https://example.test/evidence",
            sha256="a" * 64,
            adapter="test",
            adapter_version="1",
            normalization_version="1",
        ),
        metadata={"page": 3},
    )


def test_rapid_and_paddle_ocr_preserve_page_bbox_and_confidence() -> None:
    box = [[0, 0], [3, 0], [3, 2], [0, 2]]
    rapid = RapidOcrExtractor(
        model="rapid-test",
        _engine=lambda image: ([[box, "literal rapid", 0.87]], None),
    )

    class PaddleEngine:
        def ocr(self, image):
            return [[(box, ("literal paddle", 1.4))]]

    paddle = PaddleOcrExtractor(model="paddle-test", _engine=PaddleEngine())

    rapid_result = rapid.extract(_png(), _context())
    paddle_result = paddle.extract(_png(), _context())

    assert rapid_result.evidence[0].page == 3
    assert rapid_result.evidence[0].bbox == (0.0, 0.0, 3.0, 2.0)
    assert rapid_result.evidence[0].confidence == 0.87
    assert paddle_result.evidence[0].confidence == 1.0
    assert paddle_result.evidence[0].literal is True


def test_current_rapidocr_output_object_is_normalized() -> None:
    box = [[0, 0], [3, 0], [3, 2], [0, 2]]

    def engine(image):
        return SimpleNamespace(
            boxes=(box,),
            txts=("current rapid",),
            scores=(0.93,),
        )

    result = RapidOcrExtractor(model="rapid-current", _engine=engine).extract(
        _png(),
        _context(),
    )

    assert result.texts == ("current rapid",)
    assert result.evidence[0].bbox == (0.0, 0.0, 3.0, 2.0)
    assert result.evidence[0].confidence == 0.93


def test_transformers_ocr_is_non_literal_and_spacy_is_review_gated() -> None:
    calls = []

    def vision_pipeline(**kwargs):
        calls.append(kwargs)
        return [{"generated_text": "texto transcrito por modelo"}]

    transformer = TransformersOcrExtractor(
        model="vision-test",
        _pipeline=vision_pipeline,
    )

    entity = SimpleNamespace(
        label_="PERSON",
        text="Ana",
        start_char=11,
        end_char=14,
    )
    spacy = SpacyExtractor(
        model="nlp-test",
        labels=("PERSON",),
        _pipeline=lambda text: SimpleNamespace(ents=(entity,)),
    )

    ocr_result = transformer.extract(_png(), _context())
    nlp_result = spacy.extract(b"Interviene Ana", _context())

    assert calls[0]["text"].startswith("Transcribe all visible text")
    assert ocr_result.evidence[0].literal is False
    assert ocr_result.evidence[0].diagnostics["review_required"] is True
    assert nlp_result.candidates[0].status == "review_required"
    assert nlp_result.candidates[0].inferred is True
    assert nlp_result.candidates[0].evidence[0].span_start == 11


def test_transformers_ocr_migrates_legacy_task_and_rejects_unrelated_tasks() -> None:
    legacy = TransformersOcrExtractor.from_spec(
        ExtractionSpec(
            engine="ocr",
            backend="transformers",
            model="vision-test",
            options={"task": "image-to-text", "pipeline": lambda **kwargs: []},
        )
    )
    assert legacy.task == "image-text-to-text"

    with pytest.raises(ValueError, match="image-text-to-text"):
        TransformersOcrExtractor.from_spec(
            ExtractionSpec(
                engine="ocr",
                backend="transformers",
                model="vision-test",
                options={"task": "text-generation"},
            )
        )

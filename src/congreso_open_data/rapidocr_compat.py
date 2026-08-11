"""Compatibility boundary for the maintained RapidOCR distribution."""

from __future__ import annotations

from inspect import signature
from typing import Any


def create_rapidocr_engine(*, providers: tuple[str, ...] = ()) -> Any:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Install congreso-open-data[ocr] for RapidOCR and ONNX Runtime") from exc
    return RapidOCR(**rapidocr_runtime_kwargs(RapidOCR, providers=providers))


def rapidocr_runtime_kwargs(
    rapidocr_cls: Any,
    *,
    providers: tuple[str, ...],
) -> dict[str, Any]:
    """Map the selected ONNX providers to both current and legacy constructors."""

    try:
        parameter_names = set(signature(rapidocr_cls).parameters)
    except (TypeError, ValueError):
        parameter_names = set()
    if "params" in parameter_names:
        selected = set(providers)
        return {
            "params": {
                "EngineConfig.onnxruntime.use_cuda": "CUDAExecutionProvider" in selected,
                "EngineConfig.onnxruntime.use_dml": "DmlExecutionProvider" in selected,
                "EngineConfig.onnxruntime.use_cann": "CANNExecutionProvider" in selected,
                "EngineConfig.onnxruntime.use_coreml": "CoreMLExecutionProvider" in selected,
            }
        }
    kwargs: dict[str, Any] = {}
    if "providers" in parameter_names:
        kwargs["providers"] = list(providers)
    if "use_cuda" in parameter_names:
        kwargs["use_cuda"] = "CUDAExecutionProvider" in providers
    if "use_gpu" in parameter_names:
        kwargs["use_gpu"] = "CUDAExecutionProvider" in providers
    return kwargs


def rapidocr_lines(output: Any) -> list[Any]:
    """Normalize RapidOCR 3.x output and the former tuple format to line triples."""

    if (
        isinstance(output, tuple)
        and len(output) == 2
        and (output[0] is None or isinstance(output[0], list))
    ):
        output = output[0]
    boxes = getattr(output, "boxes", None)
    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    if boxes is not None and texts is not None and scores is not None:
        return [
            [box.tolist() if hasattr(box, "tolist") else box, str(text), score]
            for box, text, score in zip(boxes, texts, scores, strict=True)
        ]
    if output is None:
        return []
    if isinstance(output, list):
        return output
    raise ValueError(f"Unsupported RapidOCR output type: {type(output).__name__}")

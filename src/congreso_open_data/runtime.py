from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_QUALITY_GRAPH_PATH = Path(__file__).with_name("quality_graph.yaml")


@dataclass(frozen=True)
class RuntimeSettings:
    quality_graph_path: Path
    require_quality_graph: bool
    require_gpu: bool
    llm_review_enabled: bool


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        quality_graph_path=Path(
            os.getenv("CONGRESO_QUALITY_GRAPH_PATH", str(DEFAULT_QUALITY_GRAPH_PATH))
        ),
        require_quality_graph=_env_bool("CONGRESO_REQUIRE_QUALITY_GRAPH", True),
        require_gpu=_env_bool("CONGRESO_REQUIRE_GPU", False),
        llm_review_enabled=_env_bool("CONGRESO_ENABLE_LLM_REVIEW", False),
    )


@lru_cache(maxsize=4)
def load_quality_graph(path: str | None = None) -> dict[str, Any]:
    graph_path = Path(path) if path else runtime_settings().quality_graph_path
    if not graph_path.exists():
        if runtime_settings().require_quality_graph:
            raise FileNotFoundError(f"Quality graph is required but was not found: {graph_path}")
        return {"models": {}, "quality_gates": {}}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Quality graph loading requires PyYAML.") from exc
    graph = yaml.safe_load(graph_path.read_text(encoding="utf-8")) or {}
    if not isinstance(graph, dict):
        raise ValueError(f"Quality graph must be a mapping: {graph_path}")
    graph.setdefault("models", {})
    graph.setdefault("quality_gates", {})
    return graph


def require_model_contract(model_key: str) -> dict[str, Any]:
    settings = runtime_settings()
    graph = load_quality_graph(str(settings.quality_graph_path))
    models = graph.get("models") or {}
    if model_key not in models:
        if settings.require_quality_graph:
            raise ValueError(
                f"Quality graph {settings.quality_graph_path} does not define model {model_key!r}"
            )
        return {}
    contract = models[model_key] or {}
    if contract.get("requires_gpu") and settings.require_gpu:
        require_gpu_provider()
    return contract


def model_name_from_graph(model_key: str, default: str) -> str:
    contract = require_model_contract(model_key)
    return str(contract.get("implementation") or contract.get("model") or default)


def _required_contract_value(
    contract: dict[str, Any],
    *,
    model_key: str,
    field: str,
) -> Any:
    value = contract.get(field)
    if value in (None, "", ()):
        raise ValueError(f"Quality graph model {model_key!r} must define {field!r}")
    return value


def llm_review_runtime_config() -> dict[str, Any]:
    contract = require_model_contract("llm_review")
    base_url_env = str(
        _required_contract_value(contract, model_key="llm_review", field="base_url_env")
    )
    api_key_env = str(
        _required_contract_value(contract, model_key="llm_review", field="api_key_env")
    )
    endpoint_path = str(contract.get("endpoint_path") or "/chat/completions")
    return {
        "implementation": str(
            _required_contract_value(
                contract,
                model_key="llm_review",
                field="implementation",
            )
        ),
        "provider": str(
            _required_contract_value(contract, model_key="llm_review", field="provider")
        ),
        "provider_model": str(
            _required_contract_value(
                contract,
                model_key="llm_review",
                field="provider_model",
            )
        ),
        "base_url": os.getenv(base_url_env, ""),
        "base_url_env": base_url_env,
        "api_key": os.getenv(api_key_env, ""),
        "api_key_env": api_key_env,
        "endpoint_path": endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}",
        "timeout_seconds": float(contract.get("timeout_seconds", 60)),
        "temperature": float(contract.get("temperature", 0)),
        "response_format": contract.get("response_format") or _default_llm_response_format(),
    }


def _default_llm_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "congreso_llm_review_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "accepted",
                            "corrected",
                            "rejected",
                            "needs_human_review",
                        ],
                    },
                    "corrected_fields": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "explanation": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": [
                    "status",
                    "corrected_fields",
                    "confidence",
                    "explanation",
                ],
            },
        },
    }


def ocr_runtime_config(*, require_gpu: bool | None = None) -> dict[str, Any]:
    settings = runtime_settings()
    contract = require_model_contract("document_ocr")
    gpu_required = settings.require_gpu if require_gpu is None else require_gpu
    providers = contract.get("onnx_providers") or contract.get("providers") or ()
    if isinstance(providers, str):
        provider_values = _env_list_from_raw(providers)
    else:
        provider_values = tuple(str(provider) for provider in providers if provider)
    if not provider_values:
        raise ValueError("Quality graph model 'document_ocr' must define 'onnx_providers'")
    if gpu_required and "CUDAExecutionProvider" not in provider_values:
        raise ValueError(
            "Quality graph model 'document_ocr' must include CUDAExecutionProvider "
            "when GPU execution is required"
        )
    implementation = _required_contract_value(
        contract,
        model_key="document_ocr",
        field="implementation",
    )
    return {
        "implementation": str(implementation),
        "pipeline_version": str(contract.get("pipeline_version") or implementation),
        "use_angle_classifier": bool(contract.get("use_angle_classifier", False)),
        "onnx_providers": list(provider_values),
    }


def quality_gate_for_model(model_key: str) -> dict[str, Any]:
    settings = runtime_settings()
    graph = load_quality_graph(str(settings.quality_graph_path))
    contract = require_model_contract(model_key)
    gate_name = contract.get("quality_gate")
    if not gate_name:
        return {}
    gate = (graph.get("quality_gates") or {}).get(gate_name, {})
    if not isinstance(gate, dict):
        raise ValueError(f"Quality gate {gate_name!r} for model {model_key!r} must be a mapping")
    return gate


def min_confidence_for_model(model_key: str, default: float = 0.7) -> float:
    gate = quality_gate_for_model(model_key)
    value = gate.get("min_confidence", default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"min_confidence for model {model_key!r} must be numeric, got {value!r}"
        ) from exc


def require_gpu_provider() -> None:
    providers = available_onnx_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "GPU execution was required but CUDAExecutionProvider is not available. "
            f"Available ONNX providers: {providers}"
        )


def desired_ocr_onnx_providers() -> list[str]:
    return list(ocr_runtime_config()["onnx_providers"])


def available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed.") from exc
    return sorted(ort.get_available_providers())


def runtime_diagnostics(
    *,
    quality_graph_path: str | None = None,
    require_gpu: bool | None = None,
    validate_gpu: bool = True,
) -> dict[str, Any]:
    settings = runtime_settings()
    graph_path = quality_graph_path or str(settings.quality_graph_path)
    graph = load_quality_graph(graph_path)
    models = graph.get("models") or {}
    gpu_required = settings.require_gpu if require_gpu is None else require_gpu
    diagnostics: dict[str, Any] = {
        "quality_graph_path": graph_path,
        "quality_graph_name": graph.get("name"),
        "quality_graph_version": graph.get("version"),
        "require_quality_graph": settings.require_quality_graph,
        "require_gpu": gpu_required,
        "models": {},
        "quality_gates": sorted((graph.get("quality_gates") or {}).keys()),
        "coverage_targets": graph.get("coverage_targets") or {},
        "onnx_providers": [],
        "ocr_onnx_providers": ocr_runtime_config(require_gpu=gpu_required)["onnx_providers"],
        "gpu_ok": False,
        "gpu_error": None,
    }
    for key, contract in models.items():
        if not isinstance(contract, dict):
            continue
        diagnostics["models"][key] = {
            "implementation": contract.get("implementation") or contract.get("model"),
            "provider": contract.get("provider"),
            "provider_model": contract.get("provider_model"),
            "requires_gpu": bool(contract.get("requires_gpu")),
            "quality_gate": contract.get("quality_gate"),
            "outputs": contract.get("outputs") or [],
            "inputs": contract.get("inputs") or [],
        }
    try:
        diagnostics["onnx_providers"] = available_onnx_providers()
        diagnostics["gpu_ok"] = "CUDAExecutionProvider" in diagnostics["onnx_providers"]
    except RuntimeError as exc:
        diagnostics["gpu_error"] = str(exc)
    if validate_gpu and gpu_required and not diagnostics["gpu_ok"]:
        provider_details = diagnostics["gpu_error"] or diagnostics["onnx_providers"]
        raise RuntimeError(
            "GPU execution was required but runtime diagnostics did not find "
            f"CUDAExecutionProvider. Details: {provider_details}"
        )
    return diagnostics


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_list_from_raw(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())

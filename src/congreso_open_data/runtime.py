"""Runtime contracts for package-owned PDF/OCR extraction only."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_EXTRACTION_CONTRACT_PATH = Path(__file__).with_name("extraction_contracts.toml")


@dataclass(frozen=True)
class RuntimeSettings:
    extraction_contract_path: Path
    require_extraction_contract: bool
    require_gpu: bool


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        extraction_contract_path=Path(
            os.getenv(
                "CONGRESO_EXTRACTION_CONTRACT_PATH",
                str(DEFAULT_EXTRACTION_CONTRACT_PATH),
            )
        ),
        require_extraction_contract=_env_bool("CONGRESO_REQUIRE_EXTRACTION_CONTRACT", True),
        require_gpu=_env_bool("CONGRESO_REQUIRE_GPU", False),
    )


@lru_cache(maxsize=4)
def load_extraction_contracts(path: str | None = None) -> dict[str, Any]:
    contract_path = Path(path) if path else runtime_settings().extraction_contract_path
    if not contract_path.exists():
        if runtime_settings().require_extraction_contract:
            raise FileNotFoundError(
                f"Extraction contract is required but was not found: {contract_path}"
            )
        return {"models": {}}
    with contract_path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("models", {}), dict):
        raise ValueError(f"Extraction contract must define a models mapping: {contract_path}")
    forbidden = {"quality_gates", "coverage_targets", "publication", "serving"}
    present = forbidden.intersection(payload)
    if present:
        raise ValueError(
            "Extraction contract contains foundry-owned policy keys: " + ", ".join(sorted(present))
        )
    payload.setdefault("models", {})
    return payload


def require_model_contract(model_key: str) -> dict[str, Any]:
    settings = runtime_settings()
    contracts = load_extraction_contracts(str(settings.extraction_contract_path))
    models = contracts.get("models") or {}
    if model_key not in models:
        if settings.require_extraction_contract:
            raise ValueError(
                f"Extraction contract {settings.extraction_contract_path} "
                f"does not define model {model_key!r}"
            )
        return {}
    contract = models[model_key] or {}
    if not isinstance(contract, dict):
        raise ValueError(f"Extraction model contract {model_key!r} must be a mapping")
    if contract.get("requires_gpu") and settings.require_gpu:
        require_gpu_provider()
    return contract


def model_name_from_contract(model_key: str, default: str) -> str:
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
        raise ValueError(f"Extraction model {model_key!r} must define {field!r}")
    return value


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
        raise ValueError("Extraction model 'document_ocr' must define 'onnx_providers'")
    if gpu_required and "CUDAExecutionProvider" not in provider_values:
        raise ValueError(
            "Extraction model 'document_ocr' must include CUDAExecutionProvider "
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


def review_threshold_for_model(model_key: str, default: float = 0.7) -> float:
    """Return an extraction warning threshold, never a publication threshold."""

    value = require_model_contract(model_key).get("review_below_confidence", default)
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"review_below_confidence for {model_key!r} must be numeric, got {value!r}"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"review_below_confidence for {model_key!r} must be within 0..1, got {threshold}"
        )
    return threshold


def require_gpu_provider() -> None:
    providers = available_onnx_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "GPU execution was required but CUDAExecutionProvider is not available. "
            f"Available ONNX providers: {providers}"
        )


def desired_ocr_onnx_providers() -> list[str]:
    return list(ocr_runtime_config()["onnx_providers"])


def selected_ocr_onnx_providers() -> list[str]:
    """Use configured providers that exist, falling back to CPU when allowed."""

    configured = desired_ocr_onnx_providers()
    available = available_onnx_providers()
    selected = [provider for provider in configured if provider in available]
    if selected:
        return selected
    if runtime_settings().require_gpu:
        require_gpu_provider()
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return []


def available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed.") from exc
    return sorted(ort.get_available_providers())


def runtime_diagnostics(
    *,
    extraction_contract_path: str | None = None,
    require_gpu: bool | None = None,
    validate_gpu: bool = True,
) -> dict[str, Any]:
    settings = runtime_settings()
    contract_path = extraction_contract_path or str(settings.extraction_contract_path)
    contracts = load_extraction_contracts(contract_path)
    models = contracts.get("models") or {}
    gpu_required = settings.require_gpu if require_gpu is None else require_gpu
    diagnostics: dict[str, Any] = {
        "extraction_contract_path": contract_path,
        "extraction_contract_name": contracts.get("name"),
        "extraction_contract_version": contracts.get("version"),
        "require_extraction_contract": settings.require_extraction_contract,
        "require_gpu": gpu_required,
        "models": {},
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
            "requires_gpu": bool(contract.get("requires_gpu")),
            "review_below_confidence": contract.get("review_below_confidence"),
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

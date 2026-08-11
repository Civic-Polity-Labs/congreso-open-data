from __future__ import annotations

from pathlib import Path

import pytest

from congreso_open_data import runtime as runtime_module
from congreso_open_data.runtime import (
    DEFAULT_EXTRACTION_CONTRACT_PATH,
    load_extraction_contracts,
    review_threshold_for_model,
    runtime_diagnostics,
    selected_ocr_onnx_providers,
)


def test_default_runtime_contract_contains_only_extraction_configuration() -> None:
    payload = load_extraction_contracts(str(DEFAULT_EXTRACTION_CONTRACT_PATH))

    assert payload["name"] == "congreso-open-data-extraction-contracts"
    assert "document_ocr" in payload["models"]
    assert not {"quality_gates", "coverage_targets", "publication", "serving"}.intersection(payload)
    assert review_threshold_for_model("deputy_document_rules") == 0.7


def test_runtime_contract_rejects_foundry_policy_keys(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        'name = "invalid"\nquality_gates = ["publish"]\n[models.document_ocr]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="foundry-owned policy keys"):
        load_extraction_contracts(str(path))


def test_ocr_provider_selection_falls_back_to_cpu_unless_gpu_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "available_onnx_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.delenv("CONGRESO_REQUIRE_GPU", raising=False)

    assert selected_ocr_onnx_providers() == ["CPUExecutionProvider"]

    monkeypatch.setenv("CONGRESO_REQUIRE_GPU", "true")
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        selected_ocr_onnx_providers()


def test_review_threshold_is_bounded_and_diagnostics_use_extraction_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "contracts.toml"
    path.write_text(
        """
name = "test"
[models.document_ocr]
implementation = "test"
onnx_providers = ["CPUExecutionProvider"]
[models.deputy_document_rules]
implementation = "rules"
review_below_confidence = 2
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONGRESO_EXTRACTION_CONTRACT_PATH", str(path))

    with pytest.raises(ValueError, match=r"within 0\.\.1"):
        review_threshold_for_model("deputy_document_rules")

    diagnostics = runtime_diagnostics(
        extraction_contract_path=str(DEFAULT_EXTRACTION_CONTRACT_PATH),
        require_gpu=False,
        validate_gpu=False,
    )
    assert "extraction_contract_path" in diagnostics
    assert not any("quality_gate" in key for key in diagnostics)

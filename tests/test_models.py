from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from congreso_open_data.models import ArtifactManifest, ExtractionSpec, SourceRef


def test_source_ref_redacts_nested_credentials() -> None:
    source = SourceRef(
        requested_url="https://example.test/data",
        sha256="a" * 64,
        adapter="test",
        adapter_version="1",
        normalization_version="1",
        parameters={"page": 1, "api_key": "secret", "nested": {"token": "secret-2"}},
    )

    assert source.parameters == {
        "page": 1,
        "api_key": "[REDACTED]",
        "nested": {"token": "[REDACTED]"},
    }
    assert "secret-2" not in source.model_dump_json()


def test_unversioned_manifest_is_backward_compatible() -> None:
    manifest = ArtifactManifest.from_legacy(
        {
            "family": "diputados",
            "dataset": "Diputados",
            "format": "json",
            "source_url": "https://example.test/data.json",
            "run_date": "2026-08-07",
            "fetched_at": datetime.now(UTC).isoformat(),
            "sha256": "b" * 64,
            "bytes": 12,
            "payload_path": "bronze/file.json",
        }
    )

    assert manifest.schema_version == "legacy-0"
    assert manifest.adapter == "congreso.legacy"
    assert manifest.effective_url == manifest.source_url


def test_extraction_spec_requires_explicit_engine_backend_and_model() -> None:
    with pytest.raises(ValidationError):
        ExtractionSpec.model_validate({"engine": "llm", "backend": "openai"})

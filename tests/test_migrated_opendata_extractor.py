from __future__ import annotations

from pathlib import Path

import pytest

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import FetchResult


class _Client:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def get(self, url: str) -> FetchResult:
        return FetchResult(url=url, status_code=200, headers={}, content=self.content)


def _resource() -> DatasetResource:
    return DatasetResource(
        family="intervenciones",
        dataset="IntervencionesCronologicamente",
        format="json",
        url="https://example.test/data.json",
        snapshot_token="one",
    )


def test_extract_resource_rejects_non_json_http_200(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        extract_resource(
            resource=_resource(),
            run_date="2026-07-14",
            output_root=tmp_path,
            client=_Client(b"<html>temporarily unavailable</html>"),
        )


def test_extract_resource_accepts_json_object_or_array(tmp_path: Path) -> None:
    manifest = extract_resource(
        resource=_resource(),
        run_date="2026-07-14",
        output_root=tmp_path,
        client=_Client(b'{"items":[]}'),
    )

    assert (tmp_path / manifest.bronze_path).read_bytes() == b'{"items":[]}'


def test_extract_resource_rejects_non_pdf_http_200(tmp_path: Path) -> None:
    resource = DatasetResource(
        family="documents",
        dataset="official-record",
        format="pdf",
        url="https://example.test/document.pdf",
    )
    with pytest.raises(ValueError, match="invalid PDF"):
        extract_resource(
            resource=resource,
            run_date="2026-07-14",
            output_root=tmp_path,
            client=_Client(b"<html>temporarily unavailable</html>"),
        )

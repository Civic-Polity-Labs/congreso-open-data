from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import requests

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import CongresoHttpClient, FetchResult


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


def test_extract_resource_streams_get_payloads_to_staging(tmp_path: Path) -> None:
    response = requests.Response()
    response.status_code = 200
    response.url = "https://example.test/data.json"
    response.headers.update({"Content-Type": "application/json"})
    response.raw = BytesIO(b'[{"id":1},{"id":2}]')

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, *args, **kwargs):
            return response

        def close(self) -> None:
            return None

    manifest = extract_resource(
        resource=_resource(),
        run_date="2026-08-08",
        output_root=tmp_path,
        client=CongresoHttpClient(session=Session()),
    )

    assert (tmp_path / manifest.bronze_path).read_bytes() == b'[{"id":1},{"id":2}]'
    assert not list((tmp_path / ".staging" / "downloads").glob("*.part"))

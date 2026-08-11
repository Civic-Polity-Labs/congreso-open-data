import hashlib

import pytest
import requests

from congreso_open_data.http import (
    CongresoHttpClient,
    ResponseTooLargeError,
    official_diary_pdf_fallback_url,
)


class _FakeRateLimiter:
    def __init__(self) -> None:
        self.waits = 0
        self.deferred: list[float] = []

    def wait(self) -> None:
        self.waits += 1

    def defer(self, seconds: float) -> None:
        self.deferred.append(seconds)


class _FakeSession:
    def __init__(self, responses: list[requests.Response]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}

    def request(self, *args, **kwargs) -> requests.Response:
        return self.responses.pop(0)


class _DownloadSession(_FakeSession):
    def get(self, *args, **kwargs) -> requests.Response:
        return self.responses.pop(0)


def _response(status: int, body: bytes, **headers: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.url = "https://www.congreso.es/test"
    response.headers.update(headers)
    return response


def test_http_client_retries_403_with_shared_cooldown() -> None:
    limiter = _FakeRateLimiter()
    client = CongresoHttpClient(
        max_retries=3,
        rate_limiter=limiter,
        throttle_backoff_seconds=30,
    )
    client.session = _FakeSession(
        [
            _response(403, b"forbidden", **{"Retry-After": "45"}),
            _response(200, b"ok"),
        ]
    )

    result = client.get("https://www.congreso.es/test")

    assert result.content == b"ok"
    assert limiter.waits == 2
    assert limiter.deferred == [45.0]


def test_http_client_falls_back_for_mislabelled_official_diary_pdf() -> None:
    original = "https://www.congreso.es/public_oficiales/L9/CORT/DS/CM/CM_601.PDF"
    fallback = "https://www.congreso.es/public_oficiales/L9/CONG/DS/CO/CO_601.PDF"
    missing = _response(404, b"missing")
    missing.url = original
    found = _response(200, b"%PDF-1.7")
    found.url = fallback
    client = CongresoHttpClient(max_retries=1)
    client.session = _FakeSession([missing, found])

    result = client.get(original)

    assert result.url == fallback
    assert result.content == b"%PDF-1.7"


def test_official_diary_fallback_is_narrow() -> None:
    assert (
        official_diary_pdf_fallback_url(
            "https://www.congreso.es/public_oficiales/L2/CONG/DS/CO/CI_303.PDF"
        )
        == "https://www.congreso.es/public_oficiales/L2/CONG/DS/CO/CO_303.PDF"
    )


def test_http_client_validates_retry_timeout_and_byte_budgets() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        CongresoHttpClient(max_retries=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        CongresoHttpClient(timeout_seconds=(0, 1))

    client = CongresoHttpClient(max_retries=1, max_response_bytes=2)
    client.session = _FakeSession([_response(200, b"three")])
    with pytest.raises(ResponseTooLargeError, match="configured limit"):
        client.get("https://www.congreso.es/test")
    assert official_diary_pdf_fallback_url("https://example.test/L2/CONG/DS/CO/CI_303.PDF") is None
    assert (
        official_diary_pdf_fallback_url(
            "https://www.congreso.es/public_oficiales/L9/CONG/DS/CO/CO_601.PDF"
        )
        is None
    )


def test_http_post_forwards_form_data_and_uses_bounded_streaming_defaults() -> None:
    response = _response(200, b"ok")

    class RecordingSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__([response])
            self.call = None

        def request(self, *args, **kwargs):
            self.call = (args, kwargs)
            return super().request(*args, **kwargs)

    session = RecordingSession()
    client = CongresoHttpClient(session=session, max_retries=1)

    result = client.post("https://www.congreso.es/test", data={"speaker": "Ana"})

    assert result.content == b"ok"
    assert session.call == (
        ("POST", "https://www.congreso.es/test"),
        {
            "data": {"speaker": "Ana"},
            "timeout": (10.0, 60.0),
            "stream": True,
        },
    )
    assert session.headers["User-Agent"].startswith("congreso-open-data/1.1")


def test_stream_download_is_atomic_hashed_and_preserves_existing_file_on_oversize(
    tmp_path,
) -> None:
    destination = tmp_path / "source.pdf"
    good = _response(200, b"bounded", **{"Content-Length": "7"})
    good._content_consumed = True
    client = CongresoHttpClient(session=_DownloadSession([good]), max_retries=1)

    result = client.download_to_file(
        "https://www.congreso.es/source.pdf",
        destination,
        max_bytes=7,
    )

    assert destination.read_bytes() == b"bounded"
    assert result.bytes == 7
    assert result.sha256 == hashlib.sha256(b"bounded").hexdigest()
    assert not destination.with_suffix(".pdf.tmp").exists()

    destination.write_bytes(b"previous")
    oversized = _response(200, b"too-large", **{"Content-Length": "9"})
    oversized._content_consumed = True
    client = CongresoHttpClient(session=_DownloadSession([oversized]), max_retries=1)
    with pytest.raises(ResponseTooLargeError, match="Content-Length"):
        client.download_to_file(
            "https://www.congreso.es/source.pdf",
            destination,
            max_bytes=8,
        )
    assert destination.read_bytes() == b"previous"
    assert not destination.with_suffix(".pdf.tmp").exists()

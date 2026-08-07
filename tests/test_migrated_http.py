import requests

from congreso_open_data.http import (
    CongresoHttpClient,
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
    assert official_diary_pdf_fallback_url("https://example.test/L2/CONG/DS/CO/CI_303.PDF") is None
    assert (
        official_diary_pdf_fallback_url(
            "https://www.congreso.es/public_oficiales/L9/CONG/DS/CO/CO_601.PDF"
        )
        is None
    )

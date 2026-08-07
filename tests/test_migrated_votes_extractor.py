import json

import pytest

from congreso_open_data.extractors.votes import (
    discover_historical_vote_resources,
    discover_vote_dates,
    vote_source_resources_from_html,
    vote_supporting_source_resources,
)
from congreso_open_data.http import FetchResult


def test_discover_vote_dates_reads_calendar_array() -> None:
    client = _FakeVoteClient(
        """
        <script>
        var diasVotaciones = [20200107, 20200108, 20200107]
        </script>
        """
    )

    assert discover_vote_dates(client=client, legislature="14") == ("20200107", "20200108")


@pytest.mark.parametrize("html", ["<html>upstream error</html>", "var diasVotaciones = []"])
def test_discover_vote_dates_fails_closed_on_missing_or_empty_calendar(html: str) -> None:
    with pytest.raises(ValueError, match="vote calendar"):
        discover_vote_dates(client=_FakeVoteClient(html), legislature="14")


def test_historical_vote_discovery_reports_progress_and_deduplicates() -> None:
    client = _FakeHistoricalVoteClient()
    events: list[dict] = []

    resources = discover_historical_vote_resources(
        client=client,
        legislatures=("14",),
        progress=events.append,
    )

    assert len(resources) == 2
    assert resources[0].legislature == "Leg14"
    assert resources[0].session == "10"
    assert events[0]["event"] == "legislature_started"
    assert events[-1] == {
        "event": "date_discovered",
        "legislature": "14",
        "vote_date": "20200108",
        "dates_completed": 2,
        "dates_planned": 2,
        "resources": 2,
    }


def test_historical_vote_discovery_samples_latest_dates_but_freezes_full_calendar(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "votes.discovery.state.json"

    resources = discover_historical_vote_resources(
        client=_FakeHistoricalVoteClient(),
        legislatures=("14",),
        checkpoint_path=checkpoint,
        sample_dates_per_legislature=1,
    )
    state = json.loads(checkpoint.read_text(encoding="utf-8"))

    assert len(resources) == 1
    assert "20200108" in resources[0].url
    assert state["calendar_dates_by_legislature"]["14"] == [
        "20200107",
        "20200108",
    ]
    assert state["dates_by_legislature"]["14"] == ["20200108"]
    assert state["completed_dates"] == ["14|20200108"]


def test_historical_vote_discovery_rejects_date_page_without_matching_vote() -> None:
    class OffDateClient:
        def get(self, url: str) -> FetchResult:
            html = (
                "var diasVotaciones = [20200108]"
                if "targetDate=" not in url
                else (
                    '<a href="/webpublica/opendata/votaciones/Leg14/'
                    'Sesion10/20200107/Votacion2/Votacion.json">voto</a>'
                )
            )
            return FetchResult(url=url, status_code=200, headers={}, content=html.encode())

    with pytest.raises(ValueError, match="neither matching official JSON"):
        discover_historical_vote_resources(
            client=OffDateClient(),
            legislatures=("14",),
        )


def test_historical_vote_discovery_resumes_without_repeating_requests(tmp_path) -> None:
    checkpoint = tmp_path / "votes.discovery.state.json"
    client = _FakeHistoricalVoteClient()
    expected = discover_historical_vote_resources(
        client=client,
        legislatures=("14",),
        checkpoint_path=checkpoint,
        checkpoint_interval=1,
    )
    resumed_client = _NoRequestClient()

    actual = discover_historical_vote_resources(
        client=resumed_client,
        legislatures=("14",),
        checkpoint_path=checkpoint,
    )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert actual == expected
    assert resumed_client.requests == 0
    assert state["status"] == "completed"
    assert state["completed_dates"] == ["14|20200107", "14|20200108"]


def test_historical_vote_discovery_rejects_corrupt_completed_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "votes.discovery.state.json"
    discover_historical_vote_resources(
        client=_FakeHistoricalVoteClient(),
        legislatures=("14",),
        checkpoint_path=checkpoint,
        checkpoint_interval=1,
    )
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    state["resources"] = state["resources"][:-1]
    checkpoint.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="completed dates without coverage"):
        discover_historical_vote_resources(
            client=_NoRequestClient(),
            legislatures=("14",),
            checkpoint_path=checkpoint,
        )


def test_historical_vote_discovery_accepts_single_quoted_resource_links() -> None:
    class SingleQuoteClient:
        def get(self, url: str) -> FetchResult:
            html = (
                "var diasVotaciones = [20200107]"
                if "targetDate=" not in url
                else (
                    "<a href='/webpublica/opendata/votaciones/Leg14/"
                    "Sesion10/20200107/Votacion2/Votacion.json'>voto</a>"
                )
            )
            return FetchResult(url=url, status_code=200, headers={}, content=html.encode())

    resources = discover_historical_vote_resources(
        client=SingleQuoteClient(),
        legislatures=("14",),
    )

    assert len(resources) == 1
    assert resources[0].vote_number == "2"


def test_vote_date_page_discovers_all_official_source_variants() -> None:
    stem = "/webpublica/opendata/votaciones/Leg15/Sesion193/20260723/Votacion001/VOT_20260723211927"
    html = f"""
    <html><body>
      <a href="/webpublica/opendata/votaciones/Leg15/Sesion193/20260723/VOT.zip">ZIP</a>
      <a href="{stem}.pdf">Detalle</a>
      <a href="{stem}.xml">XML</a>
      <a href="{stem}.json">JSON</a>
      <img src="{stem}.png" />
    </body></html>
    """.encode()

    resources = vote_source_resources_from_html(html, legislature="15")

    assert {(resource.dataset, resource.format) for resource in resources} == {
        ("SesionVotaciones", "zip"),
        ("Votacion", "pdf"),
        ("Votacion", "xml"),
        ("Votacion", "json"),
        ("Votacion", "png"),
    }
    assert all(resource.legislature == "Leg15" for resource in resources)
    assert all(
        resource.session == "193" and resource.vote_number == "001"
        for resource in resources
        if resource.dataset == "Votacion"
    )


def test_discovery_freezes_calendar_date_pages_and_non_json_variants(tmp_path) -> None:
    checkpoint = tmp_path / "votes.discovery.state.json"
    discover_historical_vote_resources(
        client=_AllVariantVoteClient(),
        legislatures=("15",),
        checkpoint_path=checkpoint,
    )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    supporting = vote_supporting_source_resources(state)

    assert state["version"] == 4
    assert set(state["calendar_pages"]) == {"15"}
    assert set(state["date_pages"]) == {"15|20260723"}
    assert len(state["source_variants"]) == 5
    assert {(resource.dataset, resource.format) for resource in supporting} == {
        ("VoteCalendarPage", "html"),
        ("VoteDatePage", "html"),
        ("SesionVotaciones", "zip"),
        ("Votacion", "pdf"),
        ("Votacion", "xml"),
        ("Votacion", "png"),
    }


def test_historical_vote_discovery_profiles_official_unstructured_summary(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "votes.discovery.state.json"

    class SummaryOnlyClient:
        def get(self, url: str) -> FetchResult:
            html = (
                "var diasVotaciones = [20230927]"
                if "targetDate=" not in url
                else """
                <html><body><div class="cuerpo-votaciones">
                  <h3>Sesión Plenaria número 5</h3>
                  <a href="/votes/session.zip">ZIP</a>
                  <div id="accordionEst1">
                    <h5>Propuesta de investidura. (Pública por llamamiento)</h5>
                    <a class="n_exp">(Núm. expte. 080/000001)</a>
                    <div class="result_vot">
                      <p>Si: 172</p><p>No: 178</p><p>Abstenciones: 0</p>
                      <a href="/detail.PDF">Detalle</a>
                    </div>
                    <img src="/webpublica/opendata/votaciones/vote.png" />
                  </div>
                </div></body></html>
                """
            )
            return FetchResult(url=url, status_code=200, headers={}, content=html.encode())

    resources = discover_historical_vote_resources(
        client=SummaryOnlyClient(),
        legislatures=("15",),
        checkpoint_path=checkpoint,
    )
    state = json.loads(checkpoint.read_text(encoding="utf-8"))

    assert resources == []
    assert state["status"] == "completed"
    assert state["completed_dates"] == ["15|20230927"]
    assert len(state["unstructured_dates"]) == 1
    summary = state["unstructured_dates"][0]
    assert summary["session_number"] == 5
    assert summary["summary_events"][0]["yes_votes"] == 172
    assert summary["summary_events"][0]["no_votes"] == 178
    assert {item["format"] for item in summary["alternative_resources"]} == {
        "zip",
        "pdf",
        "png",
    }


class _FakeVoteClient:
    def __init__(self, html: str) -> None:
        self.html = html

    def get(self, url: str) -> FetchResult:
        return FetchResult(url=url, status_code=200, headers={}, content=self.html.encode())


class _FakeHistoricalVoteClient:
    def get(self, url: str) -> FetchResult:
        if "targetDate=" not in url:
            html = "var diasVotaciones = [20200107, 20200108]"
        else:
            target = "20200108" if "08/01/2020" in url else "20200107"
            html = (
                '<a href="/webpublica/opendata/votaciones/Leg14/'
                f'Sesion10/{target}/Votacion2/Votacion-{target}.json">voto</a>'
            )
        return FetchResult(url=url, status_code=200, headers={}, content=html.encode())


class _NoRequestClient:
    requests = 0

    def get(self, url: str) -> FetchResult:
        self.requests += 1
        raise AssertionError(f"unexpected request: {url}")


class _AllVariantVoteClient:
    def get(self, url: str) -> FetchResult:
        if "targetDate=" not in url:
            content = b"var diasVotaciones = [20260723]"
        else:
            stem = (
                "/webpublica/opendata/votaciones/Leg15/Sesion193/20260723/"
                "Votacion001/VOT_20260723211927"
            )
            content = f"""
            <html><body>
              <a href="/webpublica/opendata/votaciones/Leg15/Sesion193/20260723/VOT.zip">ZIP</a>
              <a href="{stem}.pdf">Detalle</a>
              <a href="{stem}.xml">XML</a>
              <a href="{stem}.json">JSON</a>
              <img src="{stem}.png" />
            </body></html>
            """.encode()
        return FetchResult(url=url, status_code=200, headers={}, content=content)

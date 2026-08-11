import json

import pytest

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.extractors.initiatives import (
    GENERAL_INITIATIVE_LEGISLATURES,
    INITIATIVE_LEGISLATURES,
    HistoricalInitiativeScope,
    _resource_key,
    discover_approved_law_resources,
    discover_historical_initiative_resources,
    discover_historical_initiative_scope_resources,
    historical_initiative_scopes,
)
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import FetchResult
from congreso_open_data.normalizers import initiatives as normalize_initiatives
from congreso_open_data.normalizers import public_manifest
from congreso_open_data.storage import read_bronze_manifest


def test_initiative_resource_identity_preserves_distinct_post_requests() -> None:
    first = DatasetResource(
        family="iniciativas",
        dataset="GeneralInitiatives",
        format="json",
        url="https://example.test/same-endpoint",
        post_data={"page": "1", "legislature": "XV"},
    )
    second = DatasetResource(
        family="iniciativas",
        dataset="GeneralInitiatives",
        format="json",
        url=first.url,
        post_data={"page": "2", "legislature": "XV"},
    )

    planned = {_resource_key(resource): resource for resource in (first, second)}

    assert len(planned) == 2


def test_discover_historical_initiative_resources_expands_pages() -> None:
    client = _FakeInitiativeClient(
        first_payload={
            "iniciativas_encontradas": "26",
            "lista_iniciativas": {"iniciativa1": {"id_iniciativa": "121/000001"}},
        },
    )

    resources = discover_historical_initiative_resources(
        client=client,
        legislatures=("XIV",),
        include_approved_laws=False,
        datasets={"ProyectosDeLey"},
    )

    assert len(resources) == 2
    assert resources[0].dataset == "ProyectosDeLey"
    assert resources[0].post_data["_iniciativas_paginaActual"] == "1"
    assert resources[1].post_data["_iniciativas_paginaActual"] == "2"
    assert resources[0].legislature == "Leg.14"


def test_default_historical_initiative_scope_stops_before_current_catalog() -> None:
    assert INITIATIVE_LEGISLATURES[-1] == "XIV"
    assert "XV" not in INITIATIVE_LEGISLATURES
    assert GENERAL_INITIATIVE_LEGISLATURES[-1] == "XV"


def test_historical_initiative_scopes_are_bounded_and_network_free() -> None:
    scopes = historical_initiative_scopes(
        legislatures=("XIV",),
        general_legislatures=("XIV", "XV"),
    )

    assert scopes == (
        HistoricalInitiativeScope("ProyectosDeLey", "XIV"),
        HistoricalInitiativeScope("ProposicionesDeLey", "XIV"),
        HistoricalInitiativeScope("PropuestasDeReforma", "XIV"),
        HistoricalInitiativeScope("GeneralInitiatives", "XIV"),
        HistoricalInitiativeScope("GeneralInitiatives", "XV"),
        HistoricalInitiativeScope("IniciativasLegislativasAprobadas", None),
    )


def test_discover_general_initiative_scope_never_expands_other_scopes() -> None:
    client = _FakeInitiativeClient(first_payload={"iniciativas_encontradas": "26"})

    resources = discover_historical_initiative_scope_resources(
        scope=HistoricalInitiativeScope("GeneralInitiatives", "XV"),
        client=client,
    )

    assert len(resources) == 2
    assert {resource.dataset for resource in resources} == {"GeneralInitiatives"}
    assert all(resource.legislature == "Leg.15" for resource in resources)


def test_general_initiative_discovery_covers_current_and_expands_pages() -> None:
    client = _FakeInitiativeClient(first_payload={"iniciativas_encontradas": "26"})

    resources = discover_historical_initiative_resources(
        client=client,
        legislatures=(),
        general_legislatures=("XV",),
        include_approved_laws=False,
        datasets={"GeneralInitiatives"},
    )

    assert len(resources) == 2
    assert {resource.dataset for resource in resources} == {"GeneralInitiatives"}
    assert len({resource.url for resource in resources}) == 2
    assert [resource.post_data["_iniciativas_paginaActual"] for resource in resources] == [
        "1",
        "2",
    ]
    assert all(resource.post_data["_iniciativas_legislatura"] == "XV" for resource in resources)
    assert all("_iniciativas_tipo" in resource.post_data for resource in resources)


def test_historical_initiative_discovery_fails_closed_on_missing_total() -> None:
    client = _FakeInitiativeClient(first_payload={"lista_iniciativas": {}})

    with pytest.raises(ValueError, match="missing iniciativas_encontradas"):
        discover_historical_initiative_resources(
            client=client,
            legislatures=(),
            general_legislatures=("XV",),
            include_approved_laws=False,
            datasets={"GeneralInitiatives"},
        )


def test_historical_initiative_discovery_persists_zero_result_evidence() -> None:
    client = _FakeInitiativeClient(
        first_payload={"iniciativas_encontradas": "0", "lista_iniciativas": {}},
    )

    resources = discover_historical_initiative_resources(
        client=client,
        legislatures=(),
        general_legislatures=("XV",),
        include_approved_laws=False,
        datasets={"GeneralInitiatives"},
    )

    assert len(resources) == 1
    assert resources[0].snapshot_token == "historical-GeneralInitiatives-XV-0001"


def test_historical_initiative_discovery_accepts_only_allowlisted_empty_object() -> None:
    client = _FakeInitiativeClient(first_payload={})

    resources = discover_historical_initiative_resources(
        client=client,
        legislatures=("XI",),
        general_legislatures=(),
        include_approved_laws=False,
        datasets={"ProyectosDeLey"},
    )

    assert len(resources) == 1
    assert resources[0].snapshot_token == "historical-ProyectosDeLey-XI-0001"

    with pytest.raises(ValueError, match="missing iniciativas_encontradas"):
        discover_historical_initiative_resources(
            client=client,
            legislatures=("XII",),
            general_legislatures=(),
            include_approved_laws=False,
            datasets={"ProyectosDeLey"},
        )


def test_historical_initiative_discovery_tracks_approved_law_years(tmp_path) -> None:
    checkpoint = tmp_path / "initiatives.discovery.state.json"
    client = _FakeInitiativeClient(
        html="""
        <select id="_iniciativasLegislativasAprobadas_anyoSelec">
          <option value="2026">2026</option>
          <option value="2025">2025</option>
        </select>
        """,
    )

    resources = discover_historical_initiative_resources(
        client=client,
        legislatures=(),
        datasets={"IniciativasLegislativasAprobadas"},
        checkpoint_path=checkpoint,
    )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert [resource.snapshot_token for resource in resources] == [
        "historical-IniciativasLegislativasAprobadas-2026",
        "historical-IniciativasLegislativasAprobadas-2025",
    ]
    assert state["approved_laws_max_year"] is None
    assert state["completed_scopes"] == [
        "IniciativasLegislativasAprobadas|2025",
        "IniciativasLegislativasAprobadas|2026",
    ]


def test_discover_approved_law_resources_reads_year_options() -> None:
    client = _FakeInitiativeClient(
        html="""
        <select id="_iniciativasLegislativasAprobadas_anyoSelec">
          <option value="2026">2026</option>
          <option value="2025">2025</option>
        </select>
        """,
    )

    resources = discover_approved_law_resources(client=client)

    years = [
        resource.post_data["_iniciativasLegislativasAprobadas_anyoSelec"] for resource in resources
    ]
    assert years == [
        "2026",
        "2025",
    ]
    assert resources[0].dataset == "IniciativasLegislativasAprobadas"


def test_discover_approved_law_resources_fails_closed_on_schema_drift() -> None:
    client = _FakeInitiativeClient(html="<html><body>upstream error</body></html>")

    with pytest.raises(ValueError, match="year selector is missing"):
        discover_approved_law_resources(client=client)


def test_historical_initiative_discovery_resumes_without_requests(tmp_path) -> None:
    checkpoint = tmp_path / "initiatives.discovery.state.json"
    client = _FakeInitiativeClient(
        first_payload={"iniciativas_encontradas": "26"},
    )
    expected = discover_historical_initiative_resources(
        client=client,
        legislatures=("XIV",),
        include_approved_laws=False,
        datasets={"ProyectosDeLey"},
        checkpoint_path=checkpoint,
    )
    resumed_client = _NoRequestInitiativeClient()

    actual = discover_historical_initiative_resources(
        client=resumed_client,
        legislatures=("XIV",),
        include_approved_laws=False,
        datasets={"ProyectosDeLey"},
        checkpoint_path=checkpoint,
    )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert actual == expected
    assert resumed_client.requests == 0
    assert state["status"] == "completed"
    assert state["completed_scopes"] == ["ProyectosDeLey|XIV"]


def test_general_scope_migrates_legacy_unused_legislature_identity(tmp_path) -> None:
    checkpoint = tmp_path / "general.discovery.state.json"
    client = _FakeInitiativeClient(first_payload={"iniciativas_encontradas": "1"})
    expected = discover_historical_initiative_resources(
        client=client,
        legislatures=("0",),
        general_legislatures=("0",),
        include_approved_laws=False,
        datasets={"GeneralInitiatives"},
        checkpoint_path=checkpoint,
    )
    legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
    legacy["legislatures"] = ["0"]
    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")
    resumed_client = _NoRequestInitiativeClient()

    actual = discover_historical_initiative_scope_resources(
        scope=HistoricalInitiativeScope("GeneralInitiatives", "0"),
        client=resumed_client,
        checkpoint_path=checkpoint,
    )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert actual == expected
    assert resumed_client.requests == 0
    assert state["legislatures"] == []


def test_extract_resource_uses_post_data(tmp_path) -> None:
    client = _FakeInitiativeClient(
        first_payload={"data": {"L": [{"numLey": "1", "titulo": "Ley 1/2026"}]}},
    )
    resource = DatasetResource(
        family="iniciativas",
        dataset="IniciativasLegislativasAprobadas",
        format="json",
        url="https://example.test/search",
        snapshot_token="approved-2026",
        post_data={"year": "2026"},
    )

    manifest = extract_resource(
        resource=resource,
        run_date="2026-07-05",
        output_root=tmp_path,
        client=client,
    )

    manifest_path = (tmp_path / manifest.bronze_path).with_suffix(".json.manifest.json")
    saved = read_bronze_manifest(manifest_path)
    assert saved.source_url == "https://example.test/search"
    assert client.posts == [("https://example.test/search", {"year": "2026"})]


def test_historical_initiative_manifest_normalizes_to_public_model(tmp_path) -> None:
    client = _FakeInitiativeClient(
        first_payload={
            "iniciativas_encontradas": "1",
            "lista_iniciativas": {
                "iniciativa1": {
                    "legislatura": "XIV",
                    "titulo": "Proyecto de Ley de prueba.",
                    "id_iniciativa": "121/000001",
                    "fecha_presentado": "01/01/2023",
                    "autores": {"autor01": {"nombre": "Gobierno"}},
                }
            },
        },
    )
    resource = DatasetResource(
        family="iniciativas",
        dataset="ProyectosDeLey",
        format="json",
        url="https://example.test/list",
        snapshot_token="historical-ProyectosDeLey-XIV-0001",
        post_data={"page": "1"},
    )

    manifest = extract_resource(
        resource=resource,
        run_date="2026-07-05",
        output_root=tmp_path,
        client=client,
    )
    rows = tuple(normalize_initiatives((public_manifest(manifest),), root=tmp_path))

    assert rows[0].legislature == "Leg.14"
    assert rows[0].file_number == "121/000001/0000"


class _FakeInitiativeClient:
    def __init__(
        self,
        *,
        first_payload: dict | None = None,
        html: str = "",
    ) -> None:
        self.first_payload = first_payload or {}
        self.html = html
        self.posts = []

    def get(self, url: str) -> FetchResult:
        return FetchResult(url=url, status_code=200, headers={}, content=self.html.encode())

    def post(self, url: str, *, data: dict | None = None) -> FetchResult:
        self.posts.append((url, data or {}))
        return FetchResult(
            url=url,
            status_code=200,
            headers={},
            content=json.dumps(self.first_payload).encode(),
        )


class _NoRequestInitiativeClient:
    requests = 0

    def get(self, url: str) -> FetchResult:
        self.requests += 1
        raise AssertionError(f"unexpected request: {url}")

    def post(self, url: str, *, data: dict | None = None) -> FetchResult:
        self.requests += 1
        raise AssertionError(f"unexpected request: {url} {data}")

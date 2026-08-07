import pytest

from congreso_open_data.extractors.profiles import (
    deputy_profile_resources_from_payload,
    discover_deputy_profile_resources,
)
from congreso_open_data.html import parse_visible_html
from congreso_open_data.http import FetchResult
from congreso_open_data.transforms import (
    deputy_financial_document_rows_from_profile,
    deputy_profile_row,
    document_asset_rows_from_links,
)


class FakeProfileClient:
    def post(self, url: str, *, data: dict[str, str] | None = None) -> FetchResult:
        assert data
        assert data["_diputadomodule_tipo"] == "2"
        if data["_diputadomodule_idLegislatura"] == "-1":
            content = (
                '{"data":['
                '{"codParlamentario":342,"idLegislatura":0,'
                '"apellidosNombre":"Acevedo Bisshopp, Manuel"},'
                '{"codParlamentario":35,"idLegislatura":15,'
                '"apellidosNombre":"Zaragoza Alonso, Jose"}'
                "]}"
            )
        else:
            assert "_diputadomodule_idLegislatura=15" in url
            assert data["_diputadomodule_idLegislatura"] == "15"
            content = '{"data":[{"codParlamentario":35,"apellidosNombre":"Zaragoza Alonso, Jose"}]}'
        return FetchResult(
            url=url,
            status_code=200,
            headers={},
            content=content.encode(),
        )


def test_discover_deputy_profile_resources_from_search_endpoint() -> None:
    resources = discover_deputy_profile_resources(client=FakeProfileClient())
    assert len(resources) == 1
    assert resources[0].dataset == "DeputyProfile"
    assert resources[0].snapshot_token == "15_35"
    assert "codParlamentario=35" in resources[0].url
    assert "idLegislatura=XV" in resources[0].url


def test_discover_deputy_profile_resources_all_legislatures_uses_row_legislature() -> None:
    resources = discover_deputy_profile_resources(
        client=FakeProfileClient(),
        legislature_number=-1,
    )
    assert len(resources) == 2
    assert resources[0].snapshot_token == "0_342"
    assert "idLegislatura=0" in resources[0].url
    assert resources[1].snapshot_token == "15_35"
    assert "idLegislatura=XV" in resources[1].url


def test_profile_discovery_rejects_missing_duplicate_or_wrong_legislature_rows() -> None:
    with pytest.raises(ValueError, match="Missing deputy code"):
        deputy_profile_resources_from_payload(
            payload={"data": [{"codParlamentario": ""}]},
        )
    with pytest.raises(ValueError, match="Duplicate deputy profile"):
        deputy_profile_resources_from_payload(
            payload={
                "data": [
                    {"codParlamentario": "35", "idLegislatura": 15},
                    {"codParlamentario": "35", "idLegislatura": 15},
                ]
            },
        )
    with pytest.raises(ValueError, match="outside the requested legislature"):
        deputy_profile_resources_from_payload(
            payload={"data": [{"codParlamentario": "35", "idLegislatura": 14}]},
        )


def test_profile_html_extracts_birth_date_and_financial_documents() -> None:
    html = """
    <html><body>
      <main>
        <p>Zaragoza Alonso, Jose</p>
        <p>XV Legislatura (2023-Actualidad)</p>
        <p>Diputado por Barcelona</p>
        <p>G.P. Socialista</p><p>(</p><p>GS</p><p>)</p>
        <a href="https://www.congreso.es/docinte/registro_intereses_diputado_35.pdf">
          Declaracion de Actividades
        </a>
        <a href="https://www.congreso.es/docbienes/leg15/000027/asset.pdf">
          Declaracion de Bienes y Rentas (Wed Aug 02 00:00:00 CEST 2023)
        </a>
        <a href="https://www.congreso.es/docacteco/leg15/000030/economic.pdf">
          Declaracion de Intereses Economicos (Wed Aug 02 00:00:00 CEST 2023)
        </a>
        <p>jzaragoza@congreso.es</p>
        <p>PSC-PSOE</p>
        <p>Ficha personal</p>
        <p>Nacido el Tue Sep 12 00:00:00 CET 1961</p>
        <p>en Molins de Rei</p>
        <p>Condicion plena: Thu Aug 17 00:00:00 CEST 2023</p>
      </main>
    </body></html>
    """
    source_url = (
        "https://www.congreso.es/es/busqueda-de-diputados?"
        "_diputadomodule_mostrarFicha=true&codParlamentario=35&idLegislatura=XV"
    )
    parsed = parse_visible_html(html, base_url=source_url)
    profile = deputy_profile_row(
        visible_text=parsed.visible_text,
        source_url=source_url,
        source_sha256="abc",
        snapshot_date="2026-06-29",
    )
    documents = deputy_financial_document_rows_from_profile(
        links=parsed.links,
        profile=profile,
        snapshot_date="2026-06-29",
    )
    assets = document_asset_rows_from_links(
        links=parsed.links,
        family="diputados",
        dataset="DeputyProfile",
        entity_id=profile["person_id"],
        snapshot_date="2026-06-29",
    )
    assert profile["full_name"] == "Zaragoza Alonso, Jose"
    assert str(profile["birth_date"]) == "1961-09-12"
    assert profile["age_at_snapshot"] == 64
    assert profile["birth_place"] == "Molins de Rei"
    assert profile["constituency"] == "Barcelona"
    assert profile["parliamentary_group_code"] == "GS"
    assert profile["electoral_party"] == "PSC-PSOE"
    assert str(profile["full_condition_date"]) == "2023-08-17"
    assert [row["document_kind"] for row in documents] == [
        "activities",
        "assets_income",
        "economic_interests",
    ]
    assert documents[0]["legislature"] == "XV"
    assert documents[0]["parliamentary_group"] == "G.P. Socialista"
    assert len(assets) == 3

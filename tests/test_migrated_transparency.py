import json
from urllib.parse import parse_qs, urlparse

from congreso_open_data.extractors.transparency import discover_composition_resources
from congreso_open_data.html import HtmlLink, parse_visible_html
from congreso_open_data.http import FetchResult
from congreso_open_data.transforms import (
    document_asset_rows_from_records,
    organ_membership_rows,
    organ_rows_from_links,
    salary_rows_from_text,
)


def test_parse_visible_html_collects_links() -> None:
    html = "<html><body><script>noise()</script><a href='/es/mesa'>Mesa</a></body></html>"
    parsed = parse_visible_html(html, base_url="https://www.congreso.es/es/comisiones")
    assert "noise" not in parsed.visible_text
    assert parsed.links[0].url == "https://www.congreso.es/es/mesa"


def test_organ_rows_from_commission_links() -> None:
    rows = organ_rows_from_links(
        [
            HtmlLink(
                text="Comisión Constitucional",
                url="https://www.congreso.es/es/comisiones?_organos_codComision=301",
            )
        ],
        snapshot_date="2026-06-29",
    )
    assert rows[0]["organ_type"] == "commission"


def test_discover_composition_resources_all_legislatures_filters_empty_payloads() -> None:
    client = _FakeCompositionClient(
        non_empty={
            ("0", "1"),
            ("0", "300"),
            ("XIV", "301"),
            ("XV", "371201"),
        }
    )

    resources = discover_composition_resources(
        client=client,
        all_legislatures=True,
        require_non_empty=True,
    )

    keys = {
        (
            parse_qs(urlparse(resource.url).query)["_organos_selectedLegislatura"][0],
            parse_qs(urlparse(resource.url).query)["_organos_selectedSuborgano"][0],
        )
        for resource in resources
    }
    assert keys == {("0", "1"), ("0", "300"), ("XIV", "301"), ("XV", "371201")}
    assert {resource.legislature for resource in resources} == {"0", "XIV", "XV"}


def test_salary_rows_from_retrib_text() -> None:
    text = """
    Secretario General
    Sueldo y complemento de jornada
    6.708,77 €
    Complemento de destino Secretario General
    5.970,92 €
    Información actualizada el 30 de enero de 2026
    """
    rows = salary_rows_from_text(
        text,
        source_url="https://www.congreso.es/es/cem/retrib",
        snapshot_date="2026-06-29",
    )
    assert rows[0]["role"] == "Secretario General"
    assert rows[0]["amount_eur"] == 6708.77
    assert str(rows[0]["valid_from"]) == "2026-01-30"


def test_document_assets_from_initiative_links() -> None:
    rows = document_asset_rows_from_records(
        [
            {
                "NUMEXPEDIENTE": "121/000001/0000",
                "ENLACESBOCG": "https://www.congreso.es/a.pdf#page=3",
            }
        ],
        family="iniciativas",
        dataset="ProyectosDeLey",
        snapshot_date="2026-06-29",
    )
    assert rows[0]["mime_type"] == "application/pdf"
    assert rows[0]["page_hint"] == 3


def test_organ_membership_rows_from_search_organo_payload() -> None:
    payload = {
        "data": [
            {
                "idCargo": 2,
                "apellidosNombre": "Zaragoza Alonso, José",
                "fechaAltaFormat": "04/12/2023",
                "urlFichaDiputado": "/busqueda-de-diputados?codParlamentario=35",
                "descCargo": "Presidente",
                "siglas": "GS",
                "fechaBajaFormat": "",
            }
        ]
    }
    url = (
        "https://www.congreso.es/es/organos/composicion-en-la-legislatura"
        "?_organos_selectedLegislatura=XV&_organos_selectedOrganoSup=1"
        "&_organos_selectedSuborgano=301"
    )
    rows = organ_membership_rows(
        payload,
        source_url=url,
        source_sha256="abc",
        snapshot_date="2026-06-29",
    )
    assert rows[0]["organ_code"] == "301"
    assert rows[0]["organ_type"] == "commission"
    assert rows[0]["role"] == "Presidente"
    assert str(rows[0]["started_at"]) == "2023-12-04"


def test_organ_membership_id_distinguishes_legislatures() -> None:
    payload = {
        "data": [
            {
                "apellidosNombre": "Diputada, Una",
                "fechaAltaFormat": "01/01/2024",
                "fechaBajaFormat": "",
                "descCargo": "Vocal",
                "siglas": "GP",
            }
        ]
    }
    base = "https://www.congreso.es/es/organos/composicion-en-la-legislatura"
    first = organ_membership_rows(
        payload,
        source_url=f"{base}?_organos_selectedLegislatura=XIV&_organos_selectedSuborgano=301",
        source_sha256="sha1",
        snapshot_date="2026-07-04",
    )[0]
    second = organ_membership_rows(
        payload,
        source_url=f"{base}?_organos_selectedLegislatura=XV&_organos_selectedSuborgano=301",
        source_sha256="sha2",
        snapshot_date="2026-07-04",
    )[0]
    assert first["membership_id"] != second["membership_id"]


class _FakeCompositionClient:
    def __init__(self, *, non_empty: set[tuple[str, str]]) -> None:
        self.non_empty = non_empty

    def get(self, url: str) -> FetchResult:
        if "/comisiones" in url:
            content = (
                b"<html><body>"
                b"<a href='/es/comisiones?_organos_codComision=301'>Comision</a>"
                b"<a href='/es/comisiones?_organos_codComision=371201'>Subcomision</a>"
                b"</body></html>"
            )
            return FetchResult(url=url, status_code=200, headers={}, content=content)
        params = parse_qs(urlparse(url).query)
        key = (
            params.get("_organos_selectedLegislatura", [""])[0],
            params.get("_organos_selectedSuborgano", [""])[0],
        )
        payload = {
            "data": [
                {
                    "apellidosNombre": "Diputada, Una",
                    "descCargo": "Vocal",
                    "fechaAltaFormat": "01/01/2024",
                    "fechaBajaFormat": "",
                    "siglas": "GP",
                }
            ]
            if key in self.non_empty
            else []
        }
        return FetchResult(
            url=url,
            status_code=200,
            headers={},
            content=json.dumps(payload).encode(),
        )

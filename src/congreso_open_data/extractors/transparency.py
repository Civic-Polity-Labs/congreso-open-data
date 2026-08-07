from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlencode

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.html import parse_visible_html
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.storage import BronzeManifest

TRANSPARENCY_RESOURCES = (
    DatasetResource(
        family="organos",
        dataset="OrganosIndex",
        format="html",
        url="https://www.congreso.es/es/opendata/organos",
    ),
    DatasetResource(
        family="organos",
        dataset="Mesa",
        format="html",
        url="https://www.congreso.es/es/mesa",
    ),
    DatasetResource(
        family="organos",
        dataset="JuntaPortavoces",
        format="html",
        url="https://www.congreso.es/es/junta-de-portavoces",
    ),
    DatasetResource(
        family="organos",
        dataset="DiputacionPermanente",
        format="html",
        url="https://www.congreso.es/es/diputacion-permanente",
    ),
    DatasetResource(
        family="organos",
        dataset="Comisiones",
        format="html",
        url="https://www.congreso.es/es/comisiones",
    ),
    DatasetResource(
        family="transparencia",
        dataset="RetribucionesCargosMesa",
        format="html",
        url="https://www.congreso.es/es/cem/retrib",
    ),
)

COMPOSITION_BASE_URL = "https://www.congreso.es/es/organos/composicion-en-la-legislatura"
COMPOSITION_FIXED_CODES = ("1", "300", "401", "402")
COMPOSITION_LEGISLATURES = (
    "0",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
)
COMPOSITION_HISTORICAL_CODE_RANGE = range(300, 403)


def extract_transparency_resources(
    *,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
    all_legislatures: bool = False,
) -> list[BronzeManifest]:
    client = client or CongresoHttpClient()
    manifests = [
        extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )
        for resource in TRANSPARENCY_RESOURCES
    ]
    manifests.extend(
        extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )
        for resource in discover_composition_resources(
            client=client,
            all_legislatures=all_legislatures,
            require_non_empty=all_legislatures,
        )
    )
    return manifests


def discover_composition_resources(
    *,
    client: CongresoHttpClient | None = None,
    legislature: str = "XV",
    all_legislatures: bool = False,
    require_non_empty: bool = False,
) -> list[DatasetResource]:
    client = client or CongresoHttpClient()
    resources: dict[str, DatasetResource] = {}
    legislatures = COMPOSITION_LEGISLATURES if all_legislatures else (legislature,)
    codes = _composition_candidate_codes(client=client)
    for selected_legislature in legislatures:
        for code in codes:
            resource = _composition_resource(
                legislature=selected_legislature,
                organ_sup="1",
                organ_code=code,
            )
            if require_non_empty and not _composition_resource_has_rows(resource, client=client):
                continue
            resources[resource.url] = resource
    return list(resources.values())


def _composition_candidate_codes(*, client: CongresoHttpClient) -> tuple[str, ...]:
    codes = set(COMPOSITION_FIXED_CODES)
    codes.update(str(code) for code in COMPOSITION_HISTORICAL_CODE_RANGE)

    html = client.get("https://www.congreso.es/es/comisiones").content
    parsed = parse_visible_html(html, base_url="https://www.congreso.es/es/comisiones")
    for link in parsed.links:
        match = re.search(r"_organos_codComision=(\d+)", link.url)
        if match:
            codes.add(match.group(1))
    return tuple(sorted(codes, key=lambda value: int(value)))


def _composition_resource_has_rows(
    resource: DatasetResource,
    *,
    client: CongresoHttpClient,
) -> bool:
    try:
        payload = json.loads(client.get(resource.url).content.decode("utf-8-sig"))
    except Exception:
        return False
    data = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(data, list) and bool(data)


def _composition_resource(
    *,
    legislature: str,
    organ_sup: str,
    organ_code: str,
) -> DatasetResource:
    query = {
        "p_p_id": "organos",
        "p_p_lifecycle": "2",
        "p_p_state": "normal",
        "p_p_mode": "view",
        "p_p_resource_id": "searchOrgano",
        "p_p_cacheability": "cacheLevelPage",
        "_organos_selectedLegislatura": legislature,
        "_organos_selectedOrganoSup": organ_sup,
        "_organos_selectedSuborgano": organ_code,
    }
    return DatasetResource(
        family="organos",
        dataset="OrganMemberships",
        format="json",
        url=f"{COMPOSITION_BASE_URL}?{urlencode(query)}",
        legislature=legislature,
    )

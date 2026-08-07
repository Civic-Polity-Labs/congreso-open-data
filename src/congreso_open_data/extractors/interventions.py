from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.interventions import (
    canonical_intervention_pdf_url,
    canonical_intervention_text_url,
    intervention_document_id_from_urls,
)
from congreso_open_data.normalization import load_records, normalize_record_keys
from congreso_open_data.storage import BronzeManifest, bronze_manifest_from_dict

HISTORICAL_INTERVENTION_LEGISLATURES = tuple(str(number) for number in range(16))
INTERVENTION_EXPORT_PAGE_SIZE = 100
INTERVENTION_LIST_ENDPOINT = "https://www.congreso.es:443/es/busqueda-de-intervenciones"
INTERVENTION_LIST_PARAMS = {
    "p_p_id": "intervenciones",
    "p_p_lifecycle": "2",
    "p_p_state": "normal",
    "p_p_mode": "view",
    "p_p_resource_id": "filtrarListado",
    "p_p_cacheability": "cacheLevelPage",
}
INTERVENTION_EXPORT_PARAMS = INTERVENTION_LIST_PARAMS | {
    "p_p_resource_id": "resourceIDopendataExport",
}


def discover_intervention_text_resources_from_manifest(
    *,
    lake_root: Path,
    manifest: BronzeManifest,
) -> list[DatasetResource]:
    if manifest.family != "intervenciones" or manifest.format not in {"json", "csv"}:
        return []
    source_path = lake_root / manifest.bronze_path
    rows = load_records(source_path.read_bytes(), manifest.format)
    resources: dict[str, DatasetResource] = {}
    for row in rows:
        item = normalize_record_keys(row)
        full_text_url = item.get("enlacetextointegro")
        document_id = intervention_document_id_from_urls(
            full_text_url=full_text_url,
            pdf_url=item.get("enlacepdf"),
        )
        canonical_url = canonical_intervention_text_url(full_text_url)
        if not document_id or not canonical_url:
            continue
        resources[document_id] = DatasetResource(
            family="intervention_documents",
            dataset="InterventionFullText",
            format="html",
            url=canonical_url,
            snapshot_token=document_id,
            legislature=manifest.legislature or item.get("legislatura"),
        )
    return list(resources.values())


def discover_intervention_pdf_resources_from_manifest(
    *,
    lake_root: Path,
    manifest: BronzeManifest,
) -> list[DatasetResource]:
    if manifest.family != "intervenciones" or manifest.format not in {"json", "csv"}:
        return []
    source_path = lake_root / manifest.bronze_path
    rows = load_records(source_path.read_bytes(), manifest.format)
    resources: dict[str, DatasetResource] = {}
    for row in rows:
        item = normalize_record_keys(row)
        pdf_url = canonical_intervention_pdf_url(item.get("enlacepdf"))
        document_id = intervention_document_id_from_urls(
            full_text_url=item.get("enlacetextointegro"),
            pdf_url=pdf_url,
        )
        if not document_id or not pdf_url:
            continue
        canonical_pdf_url = canonical_intervention_pdf_url(pdf_url)
        if not canonical_pdf_url:
            continue
        resources[document_id] = DatasetResource(
            family="intervention_documents",
            dataset="InterventionFullTextPdf",
            format="pdf",
            url=canonical_pdf_url,
            snapshot_token=document_id,
            legislature=manifest.legislature or item.get("legislatura"),
        )
    return list(resources.values())


def discover_intervention_document_resources_from_index(
    *,
    lake_root: Path,
    manifest_index_path: Path,
    include_html: bool = True,
    include_pdf: bool = True,
) -> list[DatasetResource]:
    """Discover one globally deduplicated document plan for an index snapshot."""

    raw = json.loads(manifest_index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Intervention manifest index must contain a JSON list")
    resources: dict[tuple[str, str], DatasetResource] = {}
    for item in raw:
        manifest = bronze_manifest_from_dict(item)
        if include_html:
            for resource in discover_intervention_text_resources_from_manifest(
                lake_root=lake_root,
                manifest=manifest,
            ):
                resources[(resource.dataset, resource.url)] = resource
        if include_pdf:
            for resource in discover_intervention_pdf_resources_from_manifest(
                lake_root=lake_root,
                manifest=manifest,
            ):
                resources[(resource.dataset, resource.url)] = resource
    return sorted(
        resources.values(),
        key=lambda resource: (
            str(resource.snapshot_token or ""),
            resource.dataset,
            resource.url,
        ),
    )


def discover_historical_intervention_resources(
    *,
    client: CongresoHttpClient | None = None,
    legislatures: tuple[str, ...] = HISTORICAL_INTERVENTION_LEGISLATURES,
) -> list[DatasetResource]:
    client = client or CongresoHttpClient()
    resources: list[DatasetResource] = []
    list_url = _url_with_query(INTERVENTION_LIST_ENDPOINT, INTERVENTION_LIST_PARAMS)
    for legislature in legislatures:
        first_page = client.post(
            list_url,
            data=_historical_intervention_list_post_data(legislature=legislature, page=1),
        )
        total = _intervention_total(_json_payload(first_page))
        if total <= 0:
            continue
        pages = max(1, math.ceil(total / INTERVENTION_EXPORT_PAGE_SIZE))
        resources.extend(
            _historical_intervention_export_resource(
                legislature=legislature,
                file_index=file_index,
                total=total,
            )
            for file_index in range(1, pages + 1)
        )
    return resources


def extract_historical_intervention_resources(
    *,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
    legislatures: tuple[str, ...] = HISTORICAL_INTERVENTION_LEGISLATURES,
) -> list[BronzeManifest]:
    client = client or CongresoHttpClient()
    return [
        extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )
        for resource in discover_historical_intervention_resources(
            client=client,
            legislatures=legislatures,
        )
    ]


def extract_intervention_text_resources_from_manifest(
    *,
    lake_root: Path,
    manifest: BronzeManifest,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
) -> list[BronzeManifest]:
    client = client or CongresoHttpClient()
    return [
        extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )
        for resource in discover_intervention_text_resources_from_manifest(
            lake_root=lake_root,
            manifest=manifest,
        )
    ]


def extract_intervention_pdf_resources_from_manifest(
    *,
    lake_root: Path,
    manifest: BronzeManifest,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
) -> list[BronzeManifest]:
    client = client or CongresoHttpClient()
    return [
        extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )
        for resource in discover_intervention_pdf_resources_from_manifest(
            lake_root=lake_root,
            manifest=manifest,
        )
    ]


def _historical_intervention_export_resource(
    *,
    legislature: str,
    file_index: int,
    total: int,
) -> DatasetResource:
    last_result = min(file_index * INTERVENTION_EXPORT_PAGE_SIZE, total)
    post_data = _historical_intervention_list_post_data(legislature=legislature, page=1) | {
        "_intervenciones_fileIndex": str(file_index),
        "_intervenciones_fileType": "json",
        "_intervenciones_lastResult": str(last_result),
    }
    return DatasetResource(
        family="intervenciones",
        dataset="IntervencionesCronologicamente",
        format="json",
        url=_url_with_query(
            INTERVENTION_LIST_ENDPOINT,
            INTERVENTION_EXPORT_PARAMS
            | {
                "_intervenciones_legislatura": legislature,
                "_intervenciones_fileIndex": str(file_index),
                "_intervenciones_fileType": "json",
            },
        ),
        snapshot_token=(
            f"historical-IntervencionesCronologicamente-Leg{legislature}-{file_index:05d}"
        ),
        legislature=f"Leg.{legislature}",
        post_data=post_data,
    )


def _historical_intervention_list_post_data(
    *,
    legislature: str,
    page: int,
) -> dict[str, str]:
    return {
        "_intervenciones_legislatura": legislature,
        "_intervenciones_orador": "",
        "_intervenciones_cargo": "",
        "_intervenciones_titulo": "",
        "_intervenciones_texto": "",
        "_intervenciones_tipoIniciativa": "",
        "_intervenciones_fechaDesde": "",
        "_intervenciones_fechaHasta": "",
        "_intervenciones_expedientes": "",
        "_intervenciones_hasta": "",
        "_intervenciones_fase": "",
        "_intervenciones_organo": "",
        "_intervenciones_autor": "",
        "_intervenciones_modoListado": "1",
        "_intervenciones_paginaActual": str(page),
        "_intervenciones_id_iniciativa": "",
    }


def _json_payload(result: Any) -> dict[str, Any]:
    content = result.content.decode("utf-8-sig")
    if not content.strip():
        return {}
    payload = json.loads(content)
    return payload if isinstance(payload, dict) else {}


def _intervention_total(payload: dict[str, Any]) -> int:
    value = payload.get("intervenciones_encontradas")
    if value in (None, ""):
        return 0
    return int(re.sub(r"\D+", "", str(value)) or "0")


def _url_with_query(base_url: str, params: dict[str, Any]) -> str:
    return f"{base_url}?{urlencode(params)}"

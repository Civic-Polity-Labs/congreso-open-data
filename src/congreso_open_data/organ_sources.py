from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from lxml import etree

from congreso_open_data.batch_extract import extract_resource_batch
from congreso_open_data.catalog import DatasetResource
from congreso_open_data.durable_io import append_jsonl_durably, write_json_atomically
from congreso_open_data.storage import (
    BronzeManifest,
    bronze_payload_is_valid,
    content_matches_format_contract,
    content_type_matches_format,
    decode_official_csv_value,
)

ORGAN_SOURCE_VERSION = "leg15-organ-sources-v3-official-csv-quote-semantics"
_BASE = "https://www.congreso.es"
_INVENTORY_URL = (
    f"{_BASE}/es/opendata/organos?p_p_id=opendata&p_p_lifecycle=2&"
    "p_p_state=normal&p_p_mode=view&p_p_resource_id=resourceIDOrganos&"
    "p_p_cacheability=cacheLevelPage"
)
_SUPERIOR_PAGES = {
    "mesa": "/es/mesa",
    "junta-portavoces": "/es/junta-de-portavoces",
    "diputacion-permanente": "/es/diputacion-permanente",
}


def extract_leg15_organ_sources(
    *,
    run_date: str,
    lake_root: Path,
    workers: int = 2,
    request_interval_seconds: float = 0.5,
    throttle_backoff_seconds: float = 60.0,
    resume: bool = True,
) -> dict[str, Any]:
    lake_root = lake_root.resolve()
    if not lake_root.is_absolute():
        raise ValueError("lake_root must be absolute")
    if workers < 1 or workers > 3:
        raise ValueError("workers must be between 1 and 3")

    plans = lake_root / "plans"
    audit_path = lake_root / "audit" / f"organ-sources-Leg.15-{run_date}.json"
    event_log = plans / f"organ-sources-Leg.15-{run_date}.progress.jsonl"
    inventory_index = plans / f"organ-sources-Leg.15-{run_date}.inventory.json"
    data_index = plans / f"organ-sources-Leg.15-{run_date}.data.json"
    resource_plan = plans / f"organ-sources-Leg.15-{run_date}.resources.json"

    def progress(payload: dict[str, Any]) -> None:
        append_jsonl_durably(event_log, payload)

    inventory_result = extract_resource_batch(
        resources=organ_inventory_resources(),
        run_date=run_date,
        output_root=lake_root,
        manifest_index_path=inventory_index,
        max_workers=workers,
        resume=resume,
        continue_on_error=False,
        progress=progress,
        request_interval_seconds=request_interval_seconds,
        throttle_backoff_seconds=throttle_backoff_seconds,
    )
    inventory_payloads = _read_inventory_payloads(
        lake_root=lake_root,
        manifests=inventory_result.manifests,
    )
    resources, descriptors = organ_data_resources(inventory_payloads)
    write_json_atomically(
        resource_plan,
        {
            "version": ORGAN_SOURCE_VERSION,
            "run_date": run_date,
            "legislature": "Leg.15",
            "resources": [asdict(resource) for resource in resources],
            "descriptors": descriptors,
        },
    )
    data_result = extract_resource_batch(
        resources=resources,
        run_date=run_date,
        output_root=lake_root,
        manifest_index_path=data_index,
        max_workers=workers,
        resume=resume,
        continue_on_error=False,
        progress=progress,
        request_interval_seconds=request_interval_seconds,
        throttle_backoff_seconds=throttle_backoff_seconds,
    )
    audit = audit_organ_sources(
        lake_root=lake_root,
        run_date=run_date,
        inventory_manifests=inventory_result.manifests,
        data_manifests=data_result.manifests,
        planned_resources=resources,
        descriptors=descriptors,
    )
    write_json_atomically(audit_path, audit)
    append_jsonl_durably(event_log, {"event": "terminal_audit", **audit})
    if not audit["passed"]:
        raise ValueError(f"Legislature XV organ source audit failed: {audit_path}")
    return {**audit, "audit_path": str(audit_path)}


def organ_inventory_resources() -> list[DatasetResource]:
    resources = [
        DatasetResource(
            family="organos",
            dataset="OrganOpenDataIndex",
            format="html",
            url=f"{_BASE}/es/opendata/organos",
            snapshot_token="Leg15-index",
            legislature="Leg.15",
        )
    ]
    resources.extend(
        DatasetResource(
            family="organos",
            dataset="OrganSuperiorPage",
            format="html",
            url=(
                f"{_BASE}{path}?p_p_id=organos&_organos_statusOpenData=true&selectedLegislatura=XV"
            ),
            snapshot_token=f"Leg15-{slug}-page",
            legislature="Leg.15",
        )
        for slug, path in _SUPERIOR_PAGES.items()
    )
    resources.extend(
        DatasetResource(
            family="organos",
            dataset="OrganDynamicInventory",
            format="json",
            url=_INVENTORY_URL,
            snapshot_token=f"Leg15-type{organ_type}-inventory",
            legislature="Leg.15",
            post_data={
                "_opendata_legislatura": "15",
                "_opendata_tipoConsulta": organ_type,
            },
        )
        for organ_type in ("3", "4")
    )
    return resources


def organ_data_resources(
    inventory_payloads: dict[str, dict[str, Any]],
) -> tuple[list[DatasetResource], list[dict[str, str]]]:
    descriptors: dict[tuple[str, str, str], dict[str, str]] = {}
    for slug, path in _SUPERIOR_PAGES.items():
        descriptors[(slug, path, "")] = {
            "slug": slug,
            "path": path,
            "organ_sup": "",
            "suborgan": "",
            "description": slug,
        }
    for inventory_name, payload in sorted(inventory_payloads.items()):
        rows = payload.get("datosOrganos") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Official organ inventory is empty: {inventory_name}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Malformed organ inventory row: {inventory_name}")
            raw_url = str(row.get("urlExport") or "")
            query = parse_qs(urlparse(raw_url).query)
            organ_sup = _single_query_value(query, "_organos_selectedOrganoSup")
            suborgan = _single_query_value(query, "_organos_selectedSuborgano")
            legislature = _single_query_value(query, "_organos_selectedLegislatura")
            if legislature != "XV" or not organ_sup.isdigit() or not suborgan.isdigit():
                raise ValueError(f"Malformed XV organ URL in {inventory_name}: {raw_url}")
            slug = f"{organ_sup}-{suborgan}"
            descriptors[(slug, "/es/organos/composicion-en-la-legislatura", suborgan)] = {
                "slug": slug,
                "path": "/es/organos/composicion-en-la-legislatura",
                "organ_sup": organ_sup,
                "suborgan": suborgan,
                "description": re.sub(r"\s+", " ", str(row.get("descOrgano") or "")).strip(),
            }

    resources: list[DatasetResource] = []
    ordered_descriptors = sorted(descriptors.values(), key=lambda item: item["slug"])
    for descriptor in ordered_descriptors:
        resources.extend(_resources_for_descriptor(descriptor))
    return resources, ordered_descriptors


def _resources_for_descriptor(descriptor: dict[str, str]) -> list[DatasetResource]:
    slug = descriptor["slug"]
    query = {
        "p_p_id": "organos",
        "p_p_lifecycle": "2",
        "p_p_state": "normal",
        "p_p_mode": "view",
        "p_p_cacheability": "cacheLevelPage",
        "_organos_selectedLegislatura": "XV",
        "_organos_statusOpenData": "true",
    }
    post_data = {
        "_organos_selectedLegislatura": "XV",
        "_organos_compoHistorica": "false",
    }
    if descriptor["organ_sup"]:
        query["_organos_selectedOrganoSup"] = descriptor["organ_sup"]
        query["_organos_selectedSuborgano"] = descriptor["suborgan"]
        post_data["_organos_selectedOrganoSup"] = descriptor["organ_sup"]
        post_data["_organos_selectedSuborgano"] = descriptor["suborgan"]
    base_url = f"{_BASE}{descriptor['path']}"
    search_url = f"{base_url}?{urlencode(query | {'p_p_resource_id': 'searchOrgano'})}"
    export_url = f"{base_url}?{urlencode(query | {'p_p_resource_id': 'opendataExport'})}"
    resources = [
        DatasetResource(
            family="organos",
            dataset="OrganMembershipsAjax",
            format="json",
            url=search_url,
            snapshot_token=f"Leg15-{slug}-search",
            legislature="Leg.15",
            post_data=post_data,
        )
    ]
    resources.extend(
        DatasetResource(
            family="organos",
            dataset="OrganCompositionExport",
            format=format_name,
            url=export_url,
            snapshot_token=f"Leg15-{slug}-export",
            legislature="Leg.15",
            post_data=post_data | {"_organos_fileType": format_name},
        )
        for format_name in ("csv", "json", "xml")
    )
    return resources


def audit_organ_sources(
    *,
    lake_root: Path,
    run_date: str,
    inventory_manifests: tuple[BronzeManifest, ...],
    data_manifests: tuple[BronzeManifest, ...],
    planned_resources: list[DatasetResource],
    descriptors: list[dict[str, str]],
) -> dict[str, Any]:
    all_manifests = (*inventory_manifests, *data_manifests)
    invalid_payloads = 0
    invalid_hashes = 0
    invalid_request_lineage = 0
    invalid_http_statuses = 0
    mime_mismatches = 0
    official_mime_exceptions = 0
    for manifest in all_manifests:
        path = lake_root / manifest.bronze_path
        if not bronze_payload_is_valid(root=lake_root, manifest=manifest):
            invalid_payloads += 1
        if _sha256_path(path) != manifest.sha256:
            invalid_hashes += 1
        if not 200 <= manifest.status_code < 300:
            invalid_http_statuses += 1
        if not content_type_matches_format(manifest.content_type, manifest.format):
            if _official_organ_mime_exception(manifest):
                official_mime_exceptions += 1
            else:
                mime_mismatches += 1
        if manifest.request_method == "POST":
            request_lineage_valid = bool(
                manifest.request_parameters_json
                and manifest.request_parameters_sha256
                and hashlib.sha256(manifest.request_parameters_json.encode()).hexdigest()
                == manifest.request_parameters_sha256
            )
        else:
            request_lineage_valid = (
                manifest.request_method == "GET"
                and manifest.request_parameters_json is None
                and manifest.request_parameters_sha256 is None
            )
        if not request_lineage_valid:
            invalid_request_lineage += 1

    expected_inventory = {_resource_identity(resource) for resource in organ_inventory_resources()}
    actual_inventory = {_manifest_identity(manifest) for manifest in inventory_manifests}
    expected_keys = {_resource_identity(resource) for resource in planned_resources}
    actual_keys = {_manifest_identity(manifest) for manifest in data_manifests}
    duplicate_inventory_manifests = len(inventory_manifests) - len(actual_inventory)
    duplicate_data_manifests = len(data_manifests) - len(actual_keys)
    missing_resources = len(expected_keys - actual_keys)
    unexpected_resources = len(actual_keys - expected_keys)
    format_groups: dict[str, dict[str, BronzeManifest]] = defaultdict(dict)
    for manifest in data_manifests:
        if manifest.dataset == "OrganCompositionExport":
            format_groups[str(manifest.snapshot_token)][manifest.format] = manifest
    incomplete_format_groups = 0
    cross_format_mismatches = 0
    for group in format_groups.values():
        if set(group) != {"csv", "json", "xml"}:
            incomplete_format_groups += 1
            continue
        canonical = {
            format_name: _canonical_export_rows(
                (lake_root / manifest.bronze_path).read_bytes(),
                format_name=format_name,
            )
            for format_name, manifest in group.items()
        }
        if not _canonical_export_groups_match(canonical):
            cross_format_mismatches += 1

    checks = {
        "inventory_complete": (
            expected_inventory == actual_inventory
            and len(inventory_manifests) == len(expected_inventory)
            and duplicate_inventory_manifests == 0
        ),
        "resource_reconciliation": (
            missing_resources == 0
            and unexpected_resources == 0
            and len(data_manifests) == len(expected_keys)
            and duplicate_data_manifests == 0
        ),
        "payload_contracts": invalid_payloads == 0,
        "deep_hashes": invalid_hashes == 0,
        "http_statuses": invalid_http_statuses == 0,
        "mime_types": mime_mismatches == 0,
        "post_request_lineage": invalid_request_lineage == 0,
        "three_format_coverage": (
            len(format_groups) == len(descriptors) and incomplete_format_groups == 0
        ),
        "cross_format_semantics": cross_format_mismatches == 0,
    }
    return {
        "version": ORGAN_SOURCE_VERSION,
        "run_date": run_date,
        "legislature": "Leg.15",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "inventory_manifests": len(inventory_manifests),
            "descriptors": len(descriptors),
            "planned_data_resources": len(planned_resources),
            "data_manifests": len(data_manifests),
            "missing_resources": missing_resources,
            "unexpected_resources": unexpected_resources,
            "missing_inventory_resources": len(expected_inventory - actual_inventory),
            "unexpected_inventory_resources": len(actual_inventory - expected_inventory),
            "duplicate_inventory_manifests": duplicate_inventory_manifests,
            "duplicate_data_manifests": duplicate_data_manifests,
            "invalid_payloads": invalid_payloads,
            "invalid_hashes": invalid_hashes,
            "invalid_http_statuses": invalid_http_statuses,
            "mime_mismatches": mime_mismatches,
            "official_mime_exceptions": official_mime_exceptions,
            "invalid_request_lineage": invalid_request_lineage,
            "format_groups": len(format_groups),
            "incomplete_format_groups": incomplete_format_groups,
            "cross_format_mismatches": cross_format_mismatches,
        },
    }


def _read_inventory_payloads(
    *,
    lake_root: Path,
    manifests: tuple[BronzeManifest, ...],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        if manifest.dataset != "OrganDynamicInventory":
            continue
        payload = json.loads((lake_root / manifest.bronze_path).read_text("utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"Organ inventory root must be an object: {manifest.source_url}")
        payloads[str(manifest.snapshot_token)] = payload
    if set(payloads) != {"Leg15-type3-inventory", "Leg15-type4-inventory"}:
        raise ValueError("Both XV organ dynamic inventories are required")
    return payloads


def _canonical_export_rows(content: bytes, *, format_name: str) -> list[dict[str, str]]:
    if not content_matches_format_contract(content=content, format_name=format_name):
        raise ValueError(f"Invalid organ {format_name} export")
    if format_name == "json":
        payload = json.loads(content.decode("utf-8-sig"))
        rows = payload if isinstance(payload, list) else payload.get("data")
        if not isinstance(rows, list):
            raise ValueError("Organ JSON export must be a list")
        raw_rows = rows
    elif format_name == "csv":
        raw_rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig")), delimiter=";"))
    elif format_name == "xml":
        root = etree.fromstring(
            content,
            parser=etree.XMLParser(resolve_entities=False, no_network=True, recover=False),
        )
        raw_rows = [
            {etree.QName(child).localname: child.text or "" for child in row} for row in root
        ]
    else:
        raise ValueError(f"Unsupported organ export format: {format_name}")
    canonical = [
        {
            str(key): (
                decode_official_csv_value(
                    str(value or "").replace("\r\n", "\n").replace("\r", "\n")
                )
                if format_name == "csv"
                else str(value or "").replace("\r\n", "\n").replace("\r", "\n")
            )
            for key, value in row.items()
        }
        for row in raw_rows
        if isinstance(row, dict)
    ]
    return sorted(canonical, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))


def _canonical_export_groups_match(
    groups: dict[str, list[dict[str, str]]],
) -> bool:
    fingerprints = {
        json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for rows in groups.values()
    }
    return len(fingerprints) == 1


def _official_organ_mime_exception(manifest: BronzeManifest) -> bool:
    content_type = str(manifest.content_type or "").split(";", 1)[0].strip().casefold()
    return (
        manifest.format == "json"
        and manifest.dataset in {"OrganDynamicInventory", "OrganMembershipsAjax"}
        and content_type == "text/html"
    )


def _single_query_value(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key) or []
    if len(values) != 1:
        raise ValueError(f"Expected one {key} query value")
    return values[0]


def _resource_identity(resource: DatasetResource) -> tuple[str, ...]:
    params = json.dumps(resource.post_data or {}, ensure_ascii=False, sort_keys=True, default=str)
    return (
        resource.family,
        resource.dataset,
        resource.format,
        resource.url,
        str(resource.snapshot_token),
        str(resource.legislature),
        "POST" if resource.post_data is not None else "GET",
        params,
    )


def _manifest_identity(manifest: BronzeManifest) -> tuple[str, ...]:
    params = json.loads(manifest.request_parameters_json or "{}")
    normalized = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    return (
        manifest.family,
        manifest.dataset,
        manifest.format,
        manifest.requested_url or manifest.source_url,
        str(manifest.snapshot_token),
        str(manifest.legislature),
        manifest.request_method,
        normalized,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

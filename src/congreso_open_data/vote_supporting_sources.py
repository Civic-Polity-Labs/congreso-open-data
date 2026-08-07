from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from congreso_open_data.durable_io import write_json_atomically
from congreso_open_data.extractors.votes import (
    _vote_dates_from_content,
    vote_source_resources_from_html,
    vote_supporting_source_resources,
)
from congreso_open_data.storage import (
    BronzeManifest,
    bronze_manifest_from_dict,
    bronze_payload_is_valid,
)

_MIME_TYPES = {
    "html": ("text/html", "application/xhtml+xml"),
    "json": ("application/json", "text/json", "application/octet-stream"),
    "xml": ("application/xml", "text/xml", "application/octet-stream"),
    "pdf": ("application/pdf", "application/octet-stream"),
    "png": ("image/png", "application/octet-stream"),
    "zip": (
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ),
}


def vote_supporting_manifest_payload_is_valid(
    *,
    lake_root: Path,
    manifest: BronzeManifest,
    discovery_checkpoint: dict[str, Any],
) -> bool:
    """Validate every frozen vote page or non-JSON artifact against discovery."""

    lake_root = lake_root.resolve()
    expected = {
        resource.url: resource
        for resource in vote_supporting_source_resources(discovery_checkpoint)
    }
    requested_url = manifest.requested_url or manifest.source_url
    resource = expected.get(requested_url)
    if resource is None or not _manifest_matches_resource(manifest, resource):
        return False
    if not _mime_is_valid(manifest):
        return False
    if not bronze_payload_is_valid(
        root=lake_root,
        manifest=manifest,
        verify_checksum=True,
    ):
        return False
    path = (lake_root / manifest.bronze_path).resolve()
    if not path.is_relative_to(lake_root):
        return False
    try:
        content = path.read_bytes()
    except OSError:
        return False
    if manifest.dataset == "VoteCalendarPage":
        legislature = str(manifest.legislature or "").removeprefix("Leg")
        page = discovery_checkpoint.get("calendar_pages", {}).get(legislature)
        expected_dates = discovery_checkpoint.get(
            "calendar_dates_by_legislature",
            discovery_checkpoint.get("dates_by_legislature", {}),
        ).get(legislature)
        if not isinstance(page, dict) or not isinstance(expected_dates, list):
            return False
        try:
            observed_dates = _vote_dates_from_content(
                content,
                legislature=legislature,
            )
        except ValueError:
            return False
        return (
            manifest.sha256 == page.get("page_sha256")
            and manifest.bytes == page.get("page_bytes")
            and list(observed_dates) == expected_dates
        )
    if manifest.dataset == "VoteDatePage":
        page = next(
            (
                item
                for item in discovery_checkpoint.get("date_pages", {}).values()
                if item.get("page_url") == requested_url
            ),
            None,
        )
        if not isinstance(page, dict):
            return False
        observed_resources = vote_source_resources_from_html(
            content,
            legislature=str(page["legislature"]),
        )
        return (
            manifest.sha256 == page.get("page_sha256")
            and manifest.bytes == page.get("page_bytes")
            and sorted(resource.url for resource in observed_resources)
            == sorted(str(value) for value in page["resource_urls"])
        )
    return True


def read_vote_discovery_for_supporting_sources(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise ValueError("Vote discovery checkpoint is not completed")
    vote_supporting_source_resources(payload)
    return payload


def audit_vote_supporting_manifest_index(
    *,
    lake_root: Path,
    manifest_index_path: Path,
    discovery_checkpoint_path: Path,
    extraction_state_path: Path | None = None,
) -> dict[str, Any]:
    """Reconcile the frozen calendar/date pages and every offered source variant."""

    lake_root = lake_root.resolve()
    manifest_index_path = manifest_index_path.resolve()
    discovery_checkpoint_path = discovery_checkpoint_path.resolve()
    extraction_state_path = (
        extraction_state_path.resolve()
        if extraction_state_path is not None
        else manifest_index_path.with_suffix(manifest_index_path.suffix + ".state.json")
    )
    discovery = read_vote_discovery_for_supporting_sources(discovery_checkpoint_path)
    raw_index = json.loads(manifest_index_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_state_path.read_text(encoding="utf-8"))
    if not isinstance(raw_index, list) or not isinstance(extraction, dict):
        raise ValueError("Vote supporting-source checkpoints are malformed")
    expected = {resource.url: resource for resource in vote_supporting_source_resources(discovery)}
    observed: dict[str, BronzeManifest] = {}
    invalid_records = 0
    invalid_payloads = 0
    duplicate_urls = 0
    payload_bytes = 0
    fingerprint = hashlib.sha256()
    fingerprint.update(discovery_checkpoint_path.read_bytes())
    for raw in raw_index:
        try:
            manifest = bronze_manifest_from_dict(raw)
        except (KeyError, TypeError, ValueError):
            invalid_records += 1
            continue
        url = manifest.requested_url or manifest.source_url
        if url in observed:
            duplicate_urls += 1
            continue
        observed[url] = manifest
        if not vote_supporting_manifest_payload_is_valid(
            lake_root=lake_root,
            manifest=manifest,
            discovery_checkpoint=discovery,
        ):
            invalid_payloads += 1
            continue
        payload_bytes += manifest.bytes
        fingerprint.update(url.encode())
        fingerprint.update(manifest.sha256.encode())
    missing_urls = sorted(set(expected) - set(observed))
    unexpected_urls = sorted(set(observed) - set(expected))
    checkpoint_complete = _extraction_checkpoint_passes(
        extraction,
        manifest_index_path=manifest_index_path,
        manifest_count=len(raw_index),
    )
    format_counts: dict[str, int] = {}
    for resource in expected.values():
        format_counts[resource.format] = format_counts.get(resource.format, 0) + 1
    gates = {
        "supporting_manifest_index_non_empty": bool(raw_index),
        "supporting_manifest_records_valid": invalid_records == 0,
        "supporting_extraction_checkpoint_complete": checkpoint_complete,
        "supporting_inventory_reconciles": (
            not missing_urls
            and not unexpected_urls
            and not duplicate_urls
            and len(raw_index) == len(expected)
        ),
        "supporting_payloads_hash_mime_and_contract_valid": invalid_payloads == 0,
        "calendar_and_date_pages_frozen_exactly": (
            format_counts.get("html", 0)
            == len(discovery["calendar_pages"]) + len(discovery["date_pages"])
        ),
    }
    return {
        "audit_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "lake_root": str(lake_root),
        "manifest_index_path": str(manifest_index_path),
        "discovery_checkpoint_path": str(discovery_checkpoint_path),
        "extraction_state_path": str(extraction_state_path),
        "source_fingerprint": fingerprint.hexdigest(),
        "metrics": {
            "expected_resources": len(expected),
            "manifests": len(raw_index),
            "payload_bytes": payload_bytes,
            "invalid_records": invalid_records,
            "invalid_payloads": invalid_payloads,
            "duplicate_urls": duplicate_urls,
            "missing_urls": len(missing_urls),
            "unexpected_urls": len(unexpected_urls),
            "format_counts": format_counts,
        },
        "examples": {
            "missing_urls": missing_urls[:50],
            "unexpected_urls": unexpected_urls[:50],
        },
        "gates": gates,
        "promotion_passed": all(gates.values()),
    }


def write_vote_supporting_source_audit(
    *,
    lake_root: Path,
    manifest_index_path: Path,
    discovery_checkpoint_path: Path,
    extraction_state_path: Path | None,
    output_path: Path,
) -> Path:
    report = audit_vote_supporting_manifest_index(
        lake_root=lake_root,
        manifest_index_path=manifest_index_path,
        discovery_checkpoint_path=discovery_checkpoint_path,
        extraction_state_path=extraction_state_path,
    )
    write_json_atomically(output_path, report)
    return output_path


def _manifest_matches_resource(manifest: BronzeManifest, resource: Any) -> bool:
    return (
        manifest.family == resource.family
        and manifest.dataset == resource.dataset
        and manifest.format == resource.format
        and manifest.snapshot_token == resource.snapshot_token
        and manifest.legislature == resource.legislature
        and str(manifest.session) == str(resource.session)
        and str(manifest.vote_number) == str(resource.vote_number)
        and manifest.status_code == 200
    )


def _mime_is_valid(manifest: BronzeManifest) -> bool:
    content_type = str(manifest.content_type or "").split(";", 1)[0].strip().casefold()
    return content_type in _MIME_TYPES.get(manifest.format.casefold(), ())


def _extraction_checkpoint_passes(
    payload: dict[str, Any],
    *,
    manifest_index_path: Path,
    manifest_count: int,
) -> bool:
    try:
        recorded_path = Path(str(payload["manifest_index_path"])).resolve()
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return (
        payload.get("status") == "completed"
        and int(payload.get("planned", -1)) == manifest_count
        and int(payload.get("completed", -1)) == manifest_count
        and int(payload.get("failed", -1)) == 0
        and recorded_path == manifest_index_path
        and payload.get("failures") == []
    )

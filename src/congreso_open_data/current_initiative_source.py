from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from congreso_open_data.batch_extract import extract_resource_batch
from congreso_open_data.catalog import (
    DatasetResource,
    discover_catalog,
    read_catalog,
)
from congreso_open_data.durable_io import append_jsonl_durably, write_json_atomically
from congreso_open_data.initiative_ownership import (
    INITIATIVE_OWNER_DATASETS,
    INITIATIVE_OWNER_PREFIXES,
    canonical_initiative_owner_keys,
    initiative_prefix_is_owned,
)
from congreso_open_data.normalization import load_records
from congreso_open_data.storage import (
    bronze_manifest_from_dict,
    bronze_payload_is_valid,
    content_type_matches_format,
)
from congreso_open_data.transforms import initiative_row

STATE_VERSION = 1
AUDIT_VERSION = "2.0.0"
CURRENT_INITIATIVE_OWNER_PREFIXES = INITIATIVE_OWNER_PREFIXES
CURRENT_INITIATIVE_OWNER_DATASETS = INITIATIVE_OWNER_DATASETS
_STRUCTURED_FILE_NUMBER = re.compile(r"^[0-9]{3}/[0-9]{6}/[0-9]{4}$")


def select_current_initiative_owner_resources(
    resources: list[DatasetResource],
) -> list[DatasetResource]:
    """Select exactly one official JSON resource for every current owner."""

    selected = [
        resource
        for resource in resources
        if resource.family == "iniciativas"
        and resource.format.casefold() == "json"
        and resource.dataset in CURRENT_INITIATIVE_OWNER_PREFIXES
    ]
    counts = {
        dataset: sum(resource.dataset == dataset for resource in selected)
        for dataset in CURRENT_INITIATIVE_OWNER_DATASETS
    }
    if any(count != 1 for count in counts.values()) or len(selected) != len(counts):
        raise RuntimeError(
            "Current initiative catalog must expose exactly one JSON resource per "
            f"owner dataset; observed={counts}"
        )
    by_dataset = {resource.dataset: resource for resource in selected}
    return [by_dataset[dataset] for dataset in CURRENT_INITIATIVE_OWNER_DATASETS]


def audit_current_initiative_owner_index(
    *,
    lake_root: Path,
    manifest_index_path: Path,
) -> dict[str, Any]:
    """Audit the daily detailed initiative owners before they enter a bundle."""

    lake_root = lake_root.resolve()
    manifest_index_path = manifest_index_path.resolve()
    index_bytes = manifest_index_path.read_bytes()
    raw_manifests = json.loads(index_bytes.decode("utf-8"))
    if not isinstance(raw_manifests, list):
        raise ValueError("Current initiative manifest index must contain a JSON list")

    manifests = [bronze_manifest_from_dict(item) for item in raw_manifests]
    dataset_counts = {
        dataset: sum(manifest.dataset == dataset for manifest in manifests)
        for dataset in CURRENT_INITIATIVE_OWNER_DATASETS
    }
    metrics: dict[str, Any] = {
        "manifests": len(manifests),
        "records": 0,
        "invalid_payloads": 0,
        "invalid_records": 0,
        "invalid_structured_ids": 0,
        "wrong_prefix_records": 0,
        "missing_legislatures": 0,
        "duplicate_structured_keys": 0,
        "invalid_http_statuses": 0,
        "mime_mismatches": 0,
        "invalid_request_lineage": 0,
        "invalid_source_urls": 0,
    }
    owner_rows = {dataset: 0 for dataset in CURRENT_INITIATIVE_OWNER_DATASETS}
    owner_prefix_rows = {
        dataset: {prefix: 0 for prefix in sorted(prefixes)}
        for dataset, prefixes in CURRENT_INITIATIVE_OWNER_PREFIXES.items()
    }
    observed_legislatures: set[str] = set()
    structured_keys: set[tuple[str, str]] = set()
    duplicate_examples: list[dict[str, str]] = []
    anomaly_examples: list[dict[str, Any]] = []
    fingerprint = hashlib.sha256()
    fingerprint.update(hashlib.sha256(index_bytes).digest())
    payload_bytes = 0

    for manifest in manifests:
        requested_url = manifest.requested_url or manifest.source_url
        if not 200 <= manifest.status_code < 300:
            metrics["invalid_http_statuses"] += 1
        if not content_type_matches_format(manifest.content_type, manifest.format):
            metrics["mime_mismatches"] += 1
        if (
            manifest.request_method != "GET"
            or manifest.request_parameters_json is not None
            or manifest.request_parameters_sha256 is not None
        ):
            metrics["invalid_request_lineage"] += 1
        if not requested_url.startswith(
            ("https://www.congreso.es/", "https://www.congreso.es:443/")
        ):
            metrics["invalid_source_urls"] += 1
        valid_contract = (
            manifest.family == "iniciativas"
            and manifest.format.casefold() == "json"
            and manifest.dataset in CURRENT_INITIATIVE_OWNER_PREFIXES
            and bronze_payload_is_valid(
                root=lake_root,
                manifest=manifest,
                verify_checksum=True,
            )
        )
        if not valid_contract:
            metrics["invalid_payloads"] += 1
            _add_example(
                anomaly_examples,
                dataset=manifest.dataset,
                error="invalid_manifest_or_payload_contract",
            )
            continue
        payload_path = lake_root / manifest.bronze_path
        content = payload_path.read_bytes()
        payload_bytes += len(content)
        fingerprint.update(manifest.dataset.encode())
        fingerprint.update(manifest.sha256.encode())
        try:
            records = load_records(content, "json")
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            metrics["invalid_payloads"] += 1
            _add_example(
                anomaly_examples,
                dataset=manifest.dataset,
                error="invalid_json_record_shape",
            )
            continue
        metrics["records"] += len(records)
        owner_rows[manifest.dataset] += len(records)
        for record in records:
            if not isinstance(record, dict):
                metrics["invalid_records"] += 1
                _add_example(
                    anomaly_examples,
                    dataset=manifest.dataset,
                    error="record_is_not_an_object",
                )
                continue
            try:
                transformed = initiative_row(
                    record,
                    source_sha256=manifest.sha256,
                    snapshot_date=manifest.run_date,
                    source_dataset=manifest.dataset,
                )
            except (TypeError, ValueError) as exc:
                metrics["invalid_records"] += 1
                _add_example(
                    anomaly_examples,
                    dataset=manifest.dataset,
                    error=f"transform_failed:{type(exc).__name__}",
                )
                continue
            legislature = transformed.get("legislature")
            file_number = transformed.get("file_number")
            if not isinstance(legislature, str) or not legislature:
                metrics["missing_legislatures"] += 1
                _add_example(
                    anomaly_examples,
                    dataset=manifest.dataset,
                    file_number=file_number,
                    error="missing_legislature",
                )
                continue
            observed_legislatures.add(legislature)
            if not isinstance(file_number, str) or not _STRUCTURED_FILE_NUMBER.fullmatch(
                file_number
            ):
                metrics["invalid_structured_ids"] += 1
                _add_example(
                    anomaly_examples,
                    dataset=manifest.dataset,
                    legislature=legislature,
                    file_number=file_number,
                    error="invalid_structured_id",
                )
                continue
            prefix = file_number.split("/", 1)[0]
            if not initiative_prefix_is_owned(
                dataset=manifest.dataset,
                prefix=prefix,
            ):
                metrics["wrong_prefix_records"] += 1
                _add_example(
                    anomaly_examples,
                    dataset=manifest.dataset,
                    legislature=legislature,
                    file_number=file_number,
                    error="wrong_owner_prefix",
                )
            else:
                owner_prefix_rows[manifest.dataset][prefix] += 1
            key = (legislature, file_number)
            if key in structured_keys:
                metrics["duplicate_structured_keys"] += 1
                if len(duplicate_examples) < 50:
                    duplicate_examples.append(
                        {"legislature": legislature, "file_number": file_number}
                    )
            structured_keys.add(key)

    gates = {
        "owner_manifest_set_complete": (
            len(manifests) == len(CURRENT_INITIATIVE_OWNER_DATASETS)
            and all(count == 1 for count in dataset_counts.values())
        ),
        "payloads_exist_and_match_manifests": metrics["invalid_payloads"] == 0,
        "http_statuses_valid": metrics["invalid_http_statuses"] == 0,
        "mime_types_valid": metrics["mime_mismatches"] == 0,
        "request_lineage_valid": metrics["invalid_request_lineage"] == 0,
        "source_urls_official": metrics["invalid_source_urls"] == 0,
        "owner_datasets_non_empty": all(owner_rows.values()),
        "records_are_objects": metrics["invalid_records"] == 0,
        "structured_ids_canonical": metrics["invalid_structured_ids"] == 0,
        "owner_prefixes_exact": metrics["wrong_prefix_records"] == 0,
        "exact_current_legislature": (
            metrics["missing_legislatures"] == 0 and observed_legislatures == {"Leg.15"}
        ),
        "structured_keys_unique": metrics["duplicate_structured_keys"] == 0,
    }
    owner_keys = canonical_initiative_owner_keys(
        [f"{legislature}|{file_number}" for legislature, file_number in structured_keys]
    )
    owner_keys_sha256 = hashlib.sha256(
        json.dumps(owner_keys, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "lake_root": str(lake_root),
        "manifest_index_path": str(manifest_index_path),
        "source_fingerprint": {
            "sha256": fingerprint.hexdigest(),
            "manifest_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "manifest_count": len(manifests),
            "payload_bytes": payload_bytes,
        },
        "dataset_manifest_counts": dataset_counts,
        "owner_rows": owner_rows,
        "owner_prefix_rows": owner_prefix_rows,
        "owner_keys": owner_keys,
        "owner_keys_sha256": owner_keys_sha256,
        "observed_legislatures": sorted(observed_legislatures),
        "metrics": metrics,
        "duplicate_examples": duplicate_examples,
        "anomaly_examples": anomaly_examples,
        "gates": gates,
        "promotion_passed": all(gates.values()),
        "review_required": not all(gates.values()),
    }


def extract_current_initiative_owner_sources(
    *,
    run_date: str,
    lake_root: Path,
    catalog_path: Path | None = None,
    discovery_plan_path: Path | None = None,
    manifest_index_path: Path | None = None,
    checkpoint_path: Path | None = None,
    event_log_path: Path | None = None,
    workers: int = 2,
    request_interval_seconds: float = 0.5,
    throttle_backoff_seconds: float = 60.0,
    resume: bool = True,
) -> dict[str, Any]:
    """Discover, extract and audit the three daily-owned initiative datasets."""

    if not run_date:
        raise ValueError("run_date is required")
    lake_root = lake_root.resolve()
    discovery_plan_path = (
        discovery_plan_path
        or lake_root / "plans" / "discovery" / f"current-initiative-owners-{run_date}.json"
    ).resolve()
    manifest_index_path = (
        manifest_index_path
        or lake_root / "manifests" / f"current-initiative-owners-{run_date}.json"
    ).resolve()
    checkpoint_path = (
        checkpoint_path or lake_root / "plans" / f"current-initiative-owners-{run_date}.state.json"
    ).resolve()
    event_log_path = (
        event_log_path or lake_root / "plans" / f"current-initiative-owners-{run_date}.jsonl"
    ).resolve()
    audit_path = (lake_root / "audit" / f"current_initiative_owners_{run_date}.json").resolve()

    if discovery_plan_path.exists() and resume:
        raw_plan = json.loads(discovery_plan_path.read_text(encoding="utf-8"))
        if not isinstance(raw_plan, list):
            raise ValueError("Current initiative discovery plan must contain a JSON list")
        resources = select_current_initiative_owner_resources(
            [DatasetResource(**item) for item in raw_plan]
        )
    else:
        catalog = read_catalog(catalog_path) if catalog_path else discover_catalog()
        resources = select_current_initiative_owner_resources(catalog)
        write_json_atomically(discovery_plan_path, [asdict(item) for item in resources])

    plan_bytes = discovery_plan_path.read_bytes()
    expected_identity = {
        "version": STATE_VERSION,
        "run_date": run_date,
        "lake_root": str(lake_root),
        "discovery_plan_path": str(discovery_plan_path),
        "discovery_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "manifest_index_path": str(manifest_index_path),
        "checkpoint_path": str(checkpoint_path),
        "audit_path": str(audit_path),
        "owner_datasets": list(CURRENT_INITIATIVE_OWNER_DATASETS),
    }
    existing = None
    if checkpoint_path.exists() and resume:
        existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"expected": value, "actual": existing.get(key)}
            for key, value in expected_identity.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Current initiative source checkpoint does not match this run: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        if existing.get("status") == "completed":
            _validate_completed_source_state(existing)
            _emit(event_log_path, "already_completed", status="completed")
            return existing

    write_json_atomically(
        checkpoint_path,
        expected_identity
        | {
            "status": "extracting",
            "error": None,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    _emit(event_log_path, "plan", planned_resources=len(resources))

    try:
        result = extract_resource_batch(
            resources=resources,
            run_date=run_date,
            output_root=lake_root,
            manifest_index_path=manifest_index_path,
            max_workers=workers,
            resume=resume,
            continue_on_error=False,
            request_interval_seconds=request_interval_seconds,
            throttle_backoff_seconds=throttle_backoff_seconds,
            progress=lambda event: _emit(
                event_log_path,
                "extraction_progress",
                extraction_event=event["event"],
                planned=int(event["planned"]),
                completed=int(event["completed"]),
                failed=int(event["failed"]),
            ),
        )
        if result.failed or result.completed != result.planned:
            raise RuntimeError("Current initiative owner extraction did not reconcile")
        audit = audit_current_initiative_owner_index(
            lake_root=lake_root,
            manifest_index_path=manifest_index_path,
        )
        write_json_atomically(audit_path, audit)
        if not audit["promotion_passed"]:
            raise RuntimeError(f"Current initiative owner source audit failed: {audit_path}")
        state = expected_identity | {
            "status": "completed",
            "planned": result.planned,
            "completed": result.completed,
            "reused": result.reused,
            "downloaded": result.downloaded,
            "failed": result.failed,
            "source_fingerprint": audit["source_fingerprint"],
            "owner_rows": audit["owner_rows"],
            "observed_legislatures": audit["observed_legislatures"],
            "error": None,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        write_json_atomically(checkpoint_path, state)
        _emit(
            event_log_path,
            "completed",
            records=int(audit["metrics"]["records"]),
            owner_rows=audit["owner_rows"],
            legislatures=audit["observed_legislatures"],
        )
        return state
    except Exception as exc:
        failed_state = expected_identity | {
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "updated_at": datetime.now(UTC).isoformat(),
        }
        write_json_atomically(checkpoint_path, failed_state)
        _emit(
            event_log_path,
            "failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def _validate_completed_source_state(state: dict[str, Any]) -> None:
    audit_path = Path(str(state.get("audit_path") or "")).resolve()
    manifest_index_path = Path(str(state.get("manifest_index_path") or "")).resolve()
    if not audit_path.exists() or not manifest_index_path.exists():
        raise RuntimeError("Completed current initiative source artifact is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("promotion_passed"):
        raise RuntimeError("Completed current initiative source audit no longer passes")
    if audit.get("source_fingerprint") != state.get("source_fingerprint"):
        raise RuntimeError("Completed current initiative source fingerprint changed")


def _add_example(target: list[dict[str, Any]], **values: Any) -> None:
    if len(target) < 50:
        target.append(values)


def _emit(path: Path, event: str, **fields: Any) -> None:
    append_jsonl_durably(
        path,
        {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields},
    )

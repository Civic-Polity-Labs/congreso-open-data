from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.durable_io import write_json_atomically
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import CongresoHttpClient, RequestRateLimiter
from congreso_open_data.storage import (
    BronzeManifest,
    bronze_manifest_from_dict,
    bronze_payload_is_valid,
    canonical_request_parameters,
)

STATE_VERSION = 1
DEFAULT_MAX_WORKERS = 4


@dataclass(frozen=True)
class ExtractionFailure:
    resource_key: str
    family: str
    dataset: str
    snapshot_token: str | None
    source_url: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class BatchExtractionResult:
    planned: int
    completed: int
    reused: int
    downloaded: int
    failed: int
    manifest_index_path: str
    state_path: str
    manifests: tuple[BronzeManifest, ...]
    failures: tuple[ExtractionFailure, ...]


ProgressCallback = Callable[[dict[str, Any]], None]
ExtractOne = Callable[[DatasetResource], BronzeManifest]
ManifestValidator = Callable[[BronzeManifest], bool]


def extract_resource_batch(
    *,
    resources: Iterable[DatasetResource],
    run_date: str,
    output_root: Path,
    manifest_index_path: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
    submission_batch_size: int = 32,
    resume: bool = True,
    continue_on_error: bool = True,
    progress: ProgressCallback | None = None,
    extract_one: ExtractOne | None = None,
    manifest_validator: ManifestValidator | None = None,
    checkpoint_interval: int = 1,
    request_interval_seconds: float = 0.2,
    throttle_backoff_seconds: float = 60.0,
) -> BatchExtractionResult:
    """Extract a bounded resource plan with atomic, resumable checkpoints.

    The manifest index is compatible with ``materialize manifest-index``. A
    sibling ``.state.json`` records failures and progress without mixing them
    with materializable manifests.
    """

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if submission_batch_size <= 0:
        raise ValueError("submission_batch_size must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if not run_date:
        raise ValueError("run_date is required")
    if request_interval_seconds < 0:
        raise ValueError("request_interval_seconds must be non-negative")
    if throttle_backoff_seconds <= 0:
        raise ValueError("throttle_backoff_seconds must be positive")

    plan = _deduplicate_resources(resources, run_date=run_date)
    manifest_index_path = manifest_index_path.resolve()
    state_path = manifest_index_path.with_suffix(manifest_index_path.suffix + ".state.json")
    existing = (
        _existing_manifests(
            resources=plan,
            run_date=run_date,
            output_root=output_root,
            manifest_index_path=manifest_index_path,
            manifest_validator=manifest_validator,
        )
        if resume
        else {}
    )
    manifests_by_key = {
        key: existing[key]
        for key in (_resource_key(resource, run_date=run_date) for resource in plan)
        if key in existing
    }
    pending = [
        resource
        for resource in plan
        if _resource_key(resource, run_date=run_date) not in manifests_by_key
    ]
    failures: list[ExtractionFailure] = []
    reused = len(manifests_by_key)
    downloaded = 0
    checkpoint_lock = threading.Lock()
    extractor = extract_one or _thread_local_extractor(
        run_date=run_date,
        output_root=output_root,
        request_interval_seconds=request_interval_seconds,
        throttle_backoff_seconds=throttle_backoff_seconds,
    )

    _write_checkpoint(
        plan=plan,
        run_date=run_date,
        manifest_index_path=manifest_index_path,
        state_path=state_path,
        manifests_by_key=manifests_by_key,
        failures=failures,
        reused=reused,
        downloaded=downloaded,
    )
    if progress:
        progress(_progress_event("planned", len(plan), reused, downloaded, failures))

    first_error: Exception | None = None
    results_since_checkpoint = 0
    worker_count = min(max_workers, max(1, min(len(pending), submission_batch_size)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for batch_start in range(0, len(pending), submission_batch_size):
            batch = pending[batch_start : batch_start + submission_batch_size]
            futures: dict[Future[BronzeManifest], DatasetResource] = {
                pool.submit(extractor, resource): resource for resource in batch
            }
            stop_after_batch = False
            for future in as_completed(futures):
                resource = futures[future]
                key = _resource_key(resource, run_date=run_date)
                stop_after_event = False
                try:
                    manifest = future.result()
                    if manifest_validator is not None and not manifest_validator(manifest):
                        raise ValueError(
                            "Extracted resource failed its dataset-specific content contract: "
                            f"{resource.url}"
                        )
                    manifests_by_key[key] = manifest
                    downloaded += 1
                    event = "downloaded"
                except Exception as exc:  # noqa: BLE001 - persist arbitrary transport failures
                    failures.append(_failure(resource, run_date=run_date, exc=exc))
                    first_error = first_error or exc
                    event = "failed"
                    if not continue_on_error:
                        for pending_future in futures:
                            if not pending_future.done():
                                pending_future.cancel()
                        stop_after_event = True
                        stop_after_batch = True
                results_since_checkpoint += 1
                if (
                    event == "failed"
                    or results_since_checkpoint >= checkpoint_interval
                    or reused + downloaded == len(plan)
                ):
                    with checkpoint_lock:
                        _write_checkpoint(
                            plan=plan,
                            run_date=run_date,
                            manifest_index_path=manifest_index_path,
                            state_path=state_path,
                            manifests_by_key=manifests_by_key,
                            failures=failures,
                            reused=reused,
                            downloaded=downloaded,
                        )
                    results_since_checkpoint = 0
                if progress:
                    progress(
                        _progress_event(
                            event,
                            len(plan),
                            reused,
                            downloaded,
                            failures,
                            resource,
                        )
                    )
                if stop_after_event:
                    break
            if stop_after_batch:
                break

    if first_error is not None and not continue_on_error:
        raise first_error

    if results_since_checkpoint:
        _write_checkpoint(
            plan=plan,
            run_date=run_date,
            manifest_index_path=manifest_index_path,
            state_path=state_path,
            manifests_by_key=manifests_by_key,
            failures=failures,
            reused=reused,
            downloaded=downloaded,
        )

    ordered_manifests = tuple(
        manifests_by_key[key]
        for key in (_resource_key(resource, run_date=run_date) for resource in plan)
        if key in manifests_by_key
    )
    return BatchExtractionResult(
        planned=len(plan),
        completed=len(ordered_manifests),
        reused=reused,
        downloaded=downloaded,
        failed=len(failures),
        manifest_index_path=str(manifest_index_path),
        state_path=str(state_path),
        manifests=ordered_manifests,
        failures=tuple(failures),
    )


def _thread_local_extractor(
    *,
    run_date: str,
    output_root: Path,
    request_interval_seconds: float,
    throttle_backoff_seconds: float,
) -> ExtractOne:
    local = threading.local()
    rate_limiter = RequestRateLimiter(min_interval_seconds=request_interval_seconds)

    def run(resource: DatasetResource) -> BronzeManifest:
        client = getattr(local, "client", None)
        if client is None:
            client = CongresoHttpClient(
                max_retries=5,
                rate_limiter=rate_limiter,
                throttle_backoff_seconds=throttle_backoff_seconds,
            )
            local.client = client
        return extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )

    return run


def _deduplicate_resources(
    resources: Iterable[DatasetResource],
    *,
    run_date: str,
) -> list[DatasetResource]:
    by_key: dict[str, DatasetResource] = {}
    for resource in resources:
        by_key.setdefault(_resource_key(resource, run_date=run_date), resource)
    return list(by_key.values())


def _existing_manifests(
    *,
    resources: list[DatasetResource],
    run_date: str,
    output_root: Path,
    manifest_index_path: Path,
    manifest_validator: ManifestValidator | None = None,
) -> dict[str, BronzeManifest]:
    candidates: list[BronzeManifest] = []
    if manifest_index_path.exists():
        payload = json.loads(manifest_index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Manifest index must contain a JSON list: {manifest_index_path}")
        candidates.extend(bronze_manifest_from_dict(item) for item in payload)

    roots = {
        output_root / "bronze" / resource.family / resource.dataset / f"snapshot_date={run_date}"
        for resource in resources
    }
    indexed_paths = {
        (output_root / manifest.bronze_path)
        .with_suffix((output_root / manifest.bronze_path).suffix + ".manifest.json")
        .resolve()
        for manifest in candidates
    }
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.manifest.json"):
            if path.resolve() in indexed_paths:
                continue
            try:
                candidates.append(bronze_manifest_from_dict(json.loads(path.read_text("utf-8"))))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    allowed = {_resource_key(resource, run_date=run_date) for resource in resources}
    existing: dict[str, BronzeManifest] = {}
    for manifest in sorted(candidates, key=lambda item: item.extracted_at):
        key = _manifest_key(manifest)
        if (
            key in allowed
            and bronze_payload_is_valid(
                root=output_root,
                manifest=manifest,
                # Historical PDFs can span many GiB. Resume validates size and the PDF
                # header in O(1); the separate resumable deep audit streams every SHA-256.
                verify_checksum=manifest.format.casefold() != "pdf",
            )
            and (manifest_validator is None or manifest_validator(manifest))
        ):
            existing[key] = manifest
    return existing


def _write_checkpoint(
    *,
    plan: list[DatasetResource],
    run_date: str,
    manifest_index_path: Path,
    state_path: Path,
    manifests_by_key: dict[str, BronzeManifest],
    failures: list[ExtractionFailure],
    reused: int,
    downloaded: int,
) -> None:
    ordered = [
        manifests_by_key[key]
        for key in (_resource_key(resource, run_date=run_date) for resource in plan)
        if key in manifests_by_key
    ]
    _atomic_json_write(manifest_index_path, [asdict(manifest) for manifest in ordered])
    _atomic_json_write(
        state_path,
        {
            "version": STATE_VERSION,
            "run_date": run_date,
            "planned": len(plan),
            "completed": len(ordered),
            "reused": reused,
            "downloaded": downloaded,
            "failed": len(failures),
            "status": (
                "failed" if failures else "completed" if len(ordered) == len(plan) else "running"
            ),
            "manifest_index_path": str(manifest_index_path),
            "failures": [asdict(failure) for failure in failures],
        },
    )


def _atomic_json_write(path: Path, payload: Any) -> None:
    write_json_atomically(path, payload, default=str)


def _resource_key(resource: DatasetResource, *, run_date: str) -> str:
    return "|".join(
        str(value or "")
        for value in (
            resource.family,
            resource.dataset,
            resource.format,
            resource.snapshot_token,
            resource.url,
            canonical_request_parameters(resource),
            run_date,
        )
    )


def _manifest_key(manifest: BronzeManifest) -> str:
    return "|".join(
        str(value or "")
        for value in (
            manifest.family,
            manifest.dataset,
            manifest.format,
            manifest.snapshot_token,
            manifest.requested_url or manifest.source_url,
            manifest.request_parameters_json,
            manifest.run_date,
        )
    )


def _failure(
    resource: DatasetResource,
    *,
    run_date: str,
    exc: Exception,
) -> ExtractionFailure:
    return ExtractionFailure(
        resource_key=_resource_key(resource, run_date=run_date),
        family=resource.family,
        dataset=resource.dataset,
        snapshot_token=resource.snapshot_token,
        source_url=resource.url,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def _progress_event(
    event: str,
    planned: int,
    reused: int,
    downloaded: int,
    failures: list[ExtractionFailure],
    resource: DatasetResource | None = None,
) -> dict[str, Any]:
    return {
        "event": event,
        "planned": planned,
        "completed": reused + downloaded,
        "reused": reused,
        "downloaded": downloaded,
        "failed": len(failures),
        "snapshot_token": resource.snapshot_token if resource else None,
    }

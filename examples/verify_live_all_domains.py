"""Bounded, resumable live verification of every public Congress data domain.

This is an operational verification runner, not boilerplate an application must copy.
It intentionally acquires at most one representative source page/file per case.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from congreso_open_data import (
    CatalogResource,
    Congress,
    CongressClient,
    ExtractionLimits,
    ExtractionPlan,
    ExtractionSpec,
    ExtractionTask,
    ModelRequest,
)
from congreso_open_data.adapters import CongressSourceAdapter
from congreso_open_data.extractors.initiatives import (
    HistoricalInitiativeScope,
    discover_historical_initiative_scope_resources,
)
from congreso_open_data.extractors.profiles import (
    deputy_profile_resources_from_payload,
    deputy_profile_search_rows,
)
from congreso_open_data.extractors.transparency import (
    TRANSPARENCY_RESOURCES,
    discover_composition_resources,
)
from congreso_open_data.http import CongresoHttpClient, RequestRateLimiter

MAX_ARTIFACT_BYTES = 32 * 1024**2


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_or_create_report(
    *,
    data_dir: Path,
    report_path: Path,
    run_date: date,
    speaker: str,
) -> dict[str, Any]:
    identity = {
        "version": 1,
        "data_dir": str(data_dir),
        "run_date": run_date.isoformat(),
        "speaker": speaker,
    }
    if not report_path.exists():
        return {
            **identity,
            "status": "running",
            "started_at": _now(),
            "limits": {"max_artifact_bytes": MAX_ARTIFACT_BYTES, "max_workers": 1},
            "cases": {},
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_identity = {key: report.get(key) for key in identity}
    if actual_identity != identity:
        raise RuntimeError(
            "Existing report belongs to another verification run: "
            f"expected={identity!r}, actual={actual_identity!r}"
        )
    report.setdefault("resumed_at", []).append(_now())
    report["status"] = "running"
    return report


def _public_resource(resource: Any) -> CatalogResource:
    if isinstance(resource, CatalogResource):
        return resource
    if is_dataclass(resource) and not isinstance(resource, type):
        return CatalogResource.model_validate(asdict(resource))
    return CatalogResource.model_validate(resource)


def _extract_one(
    client: CongressClient,
    resource: Any,
    *,
    run_date: str,
) -> Any:
    manifests = tuple(
        client.extract(
            ExtractionPlan(
                resources=(_public_resource(resource),),
                output_root=client.output_root,
                run_date=run_date,
                batch_size=1,
                max_resources=1,
                max_workers=1,
                request_interval_seconds=0.0,
                resume=True,
                continue_on_error=False,
            )
        )
    )
    if len(manifests) != 1 or client.last_run is None or client.last_run.failed:
        raise RuntimeError(f"Expected one successful manifest, got {client.last_run!r}")
    manifest = manifests[0]
    if manifest.bytes > MAX_ARTIFACT_BYTES:
        raise RuntimeError(f"Artifact exceeded 32 MiB: {manifest.source_url}")
    return manifest


def _catalog_resource(
    resources: Iterable[CatalogResource],
    *,
    family: str,
    dataset: str,
    format_name: str,
) -> CatalogResource:
    matches = [
        item
        for item in resources
        if item.family == family and item.dataset == dataset and item.format == format_name
    ]
    if not matches:
        raise RuntimeError(f"Catalog lacks {family}/{dataset}.{format_name}")
    return matches[0]


def _record_summary(records: list[Any], *, sample_fields: tuple[str, ...]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Representative source normalized to zero rows")
    hashes = {item.source.sha256 for item in records}
    if not hashes or any(len(value) != 64 for value in hashes):
        raise RuntimeError("Normalized rows do not preserve valid source hashes")
    sample = {
        field: getattr(records[0], field)
        for field in sample_fields
        if getattr(records[0], field, None) is not None
    }
    return {
        "records": len(records),
        "types": dict(sorted(Counter(type(item).__name__ for item in records).items())),
        "source_hashes": sorted(hashes),
        "sample": sample,
    }


def _keyword_model(request: ModelRequest) -> dict[str, Any]:
    match = re.search(r"\b[\wáéíóúüñ]{5,}\b", request.text, flags=re.IGNORECASE)
    if match is None:
        return {"candidates": []}
    quote = match.group(0)
    return {
        "candidates": [
            {
                "kind": "keyword",
                "value": quote.casefold(),
                "quote": quote,
                "confidence": 1.0,
            }
        ]
    }


def verify(
    *,
    data_dir: Path,
    report_path: Path,
    run_date: date,
    speaker: str,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    report_path = report_path.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    report = _load_or_create_report(
        data_dir=data_dir,
        report_path=report_path,
        run_date=run_date,
        speaker=speaker,
    )
    _atomic_json(report_path, report)

    limiter = RequestRateLimiter(min_interval_seconds=0.2)
    with CongresoHttpClient(
        timeout_seconds=(10.0, 45.0),
        max_retries=3,
        sleep_seconds=0.4,
        rate_limiter=limiter,
        throttle_backoff_seconds=5.0,
        max_response_bytes=MAX_ARTIFACT_BYTES,
        max_download_bytes=MAX_ARTIFACT_BYTES,
    ) as transport:
        adapter = CongressSourceAdapter(output_root=data_dir, transport=transport)
        client = CongressClient(output_root=data_dir, adapter=adapter)
        catalog = tuple(client.catalog())

        def record_case(name: str, operation: Callable[[], dict[str, Any]]) -> None:
            previous = report["cases"].get(name, {})
            if previous.get("status") == "passed":
                print(f"[{_now()}] {name}: reused", flush=True)
                return
            print(f"[{_now()}] {name}: running", flush=True)
            started_at = _now()
            try:
                details = operation()
                case = {
                    "status": "passed",
                    "started_at": started_at,
                    "finished_at": _now(),
                    "details": details,
                }
            except Exception as exc:
                case = {
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": _now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            report["cases"][name] = case
            _atomic_json(report_path, report)
            print(f"[{_now()}] {name}: {case['status']}", flush=True)

        def catalog_case() -> dict[str, Any]:
            required = {
                "diputados",
                "iniciativas",
                "intervenciones",
                "organos",
                "transparencia",
                "votaciones",
            }
            families = Counter(item.family for item in catalog)
            missing = required.difference(families)
            if missing:
                raise RuntimeError(f"Missing catalog families: {sorted(missing)}")
            return {
                "resources": len(catalog),
                "families": dict(sorted(families.items())),
                "formats": dict(sorted(Counter(item.format for item in catalog).items())),
            }

        def deputies_case() -> dict[str, Any]:
            resource = _catalog_resource(
                catalog,
                family="diputados",
                dataset="DiputadosActivos",
                format_name="json",
            )
            manifest = _extract_one(client, resource, run_date=run_date.isoformat())
            return {
                **_record_summary(
                    list(client.deputies((manifest,))),
                    sample_fields=("deputy_id", "full_name", "constituency", "legislature"),
                ),
                "artifact_bytes": manifest.bytes,
            }

        def interests_case() -> dict[str, Any]:
            resource = _catalog_resource(
                catalog,
                family="diputados",
                dataset="docacteco",
                format_name="json",
            )
            manifest = _extract_one(client, resource, run_date=run_date.isoformat())
            return {
                **_record_summary(
                    list(client.interests((manifest,))),
                    sample_fields=("declaration_id", "full_name", "declaration_kind"),
                ),
                "artifact_bytes": manifest.bytes,
            }

        def initiatives_case() -> dict[str, Any]:
            resources = discover_historical_initiative_scope_resources(
                scope=HistoricalInitiativeScope("GeneralInitiatives", "XV"),
                client=transport,
                checkpoint_path=data_dir / "example-checkpoints" / "initiatives-xv.json",
                resume=True,
            )
            if not resources:
                raise RuntimeError("Initiative discovery returned no representative page")
            manifest = _extract_one(client, resources[0], run_date=run_date.isoformat())
            return {
                **_record_summary(
                    list(client.initiatives((manifest,))),
                    sample_fields=("initiative_id", "title", "file_number", "legislature"),
                ),
                "discovered_pages": len(resources),
                "artifact_bytes": manifest.bytes,
            }

        def interventions_and_model_case() -> dict[str, Any]:
            with Congress(
                data_dir=data_dir,
                transport=transport,
                adapter=adapter,
                today=run_date,
            ) as congress:
                congress.models.register_callable(
                    "example-callable",
                    model="example-keyword-model",
                    version="1",
                    provider="user-code",
                    function=_keyword_model,
                )
                task = ExtractionTask(
                    name="keywords",
                    instructions="Return one exact keyword from the intervention.",
                    backend=ExtractionSpec(
                        engine="llm",
                        backend="example-callable",
                        model="example-keyword-model",
                    ),
                    limits=ExtractionLimits(max_input_characters=250_000),
                )
                result = congress.interventions.search(
                    speaker=speaker,
                    legislatures=("XV",),
                    last_months=3,
                    text_policy="native",
                    extractions=(task,),
                    max_results=100,
                )
                rows = result.collect(max_items=100)
                primary = tuple(
                    item for item in result.manifests if item.family == "intervenciones"
                )
                document_html = tuple(
                    item
                    for item in result.manifests
                    if item.family == "intervention_documents" and item.format == "html"
                )
                occurrences = list(congress.raw.intervention_occurrences(primary))
                speech_blocks = list(congress.raw.speech_blocks(document_html))
            if not result.run.complete:
                raise RuntimeError(f"Intervention result is incomplete: {result.run.failures}")
            if any(row.text_status != "matched" for row in rows):
                raise RuntimeError("At least one intervention lacks matched native text")
            candidates = [candidate for row in rows for candidate in row.extractions]
            if not candidates or any(
                candidate.status != "review_required" for candidate in candidates
            ):
                raise RuntimeError("Callable output did not remain review-gated")
            if any(not evidence.literal for item in candidates for evidence in item.evidence):
                raise RuntimeError("Deterministic example quote was not preserved as literal")
            return {
                **_record_summary(
                    rows,
                    sample_fields=("intervention_id", "session_date", "speaker", "text_method"),
                ),
                "occurrences": len(occurrences),
                "speech_blocks_from_html": len(speech_blocks),
                "candidates": len(candidates),
                "resolved_entities": result.run.resolved_entities,
                "query_fingerprint": result.run.query_fingerprint,
                "complete": result.run.complete,
            }

        def votes_case() -> dict[str, Any]:
            resource = _catalog_resource(
                catalog,
                family="votaciones",
                dataset="Votacion",
                format_name="json",
            )
            manifest = _extract_one(client, resource, run_date=run_date.isoformat())
            rows = list(client.votes((manifest,)))
            kinds = Counter(type(item).__name__ for item in rows)
            if not kinds["VoteEvent"] or not kinds["NominalVote"]:
                raise RuntimeError(f"Vote JSON lacks expected typed rows: {dict(kinds)}")
            return {
                **_record_summary(
                    rows,
                    sample_fields=("vote_id", "session", "vote_number", "vote_date"),
                ),
                "artifact_bytes": manifest.bytes,
            }

        def profiles_documents_pdf_case() -> dict[str, Any]:
            search_rows = deputy_profile_search_rows(client=transport, legislature_number=15)
            matches = [row for row in search_rows if str(row.get("codParlamentario")) == "189"]
            if len(matches) != 1:
                raise RuntimeError(f"Expected deputy code 189 once, got {len(matches)}")
            resources = deputy_profile_resources_from_payload(
                payload={"data": matches},
                legislature_number=15,
            )
            profile_manifest = _extract_one(client, resources[0], run_date=run_date.isoformat())
            profiles = list(client.profiles((profile_manifest,)))
            financial = list(client.financial_documents((profile_manifest,)))
            assets = list(client.documents((profile_manifest,)))
            _record_summary(profiles, sample_fields=("deputy_id", "full_name", "legislature"))
            _record_summary(
                financial,
                sample_fields=("document_id", "document_kind", "full_name"),
            )
            pdf_assets = [item for item in assets if item.mime_type == "application/pdf"]
            if not pdf_assets:
                raise RuntimeError("Profile contains no PDF asset")
            asset = pdf_assets[0]
            pdf_manifest = _extract_one(
                client,
                CatalogResource(
                    family="documents",
                    dataset=asset.document_kind or "DeputyFinancialDocument",
                    format="pdf",
                    url=asset.url,
                    snapshot_token=asset.document_id,
                    legislature="XV",
                ),
                run_date=run_date.isoformat(),
            )
            texts = list(client.document_texts((pdf_manifest,), use_ocr=True))
            return {
                "profiles": _record_summary(
                    profiles,
                    sample_fields=("deputy_id", "full_name", "legislature"),
                ),
                "financial_documents": _record_summary(
                    financial,
                    sample_fields=("document_id", "document_kind", "full_name"),
                ),
                "document_assets": _record_summary(
                    assets,
                    sample_fields=("document_id", "document_kind", "mime_type"),
                ),
                "document_text": _record_summary(
                    texts,
                    sample_fields=(
                        "document_id",
                        "extraction_method",
                        "model",
                        "page_count",
                        "extraction_status",
                    ),
                ),
                "pdf_artifact_bytes": pdf_manifest.bytes,
                "pdf_text_characters": len(texts[0].text),
            }

        def organs_case() -> dict[str, Any]:
            manifests = [
                _extract_one(
                    client,
                    _catalog_resource(
                        catalog,
                        family="organos",
                        dataset="OrganosIndex",
                        format_name="html",
                    ),
                    run_date=run_date.isoformat(),
                )
            ]
            composition = discover_composition_resources(
                client=transport,
                legislature="XV",
                all_legislatures=False,
            )
            membership_count = 0
            attempted = 0
            for resource in composition[:8]:
                attempted += 1
                manifest = _extract_one(client, resource, run_date=run_date.isoformat())
                manifests.append(manifest)
                membership_count += sum(
                    type(item).__name__ == "OrganMembership" for item in client.organs((manifest,))
                )
                if membership_count:
                    break
            rows = list(client.organs(manifests))
            return {
                **_record_summary(
                    rows,
                    sample_fields=("organ_id", "name", "organ_type", "legislature"),
                ),
                "composition_resources": len(composition),
                "composition_resources_attempted": attempted,
            }

        def salary_case() -> dict[str, Any]:
            resource = next(
                item for item in TRANSPARENCY_RESOURCES if item.dataset == "RetribucionesCargosMesa"
            )
            manifest = _extract_one(client, resource, run_date=run_date.isoformat())
            return {
                **_record_summary(
                    list(client.salary_entitlements((manifest,))),
                    sample_fields=("entitlement_id", "label", "amount_eur", "role"),
                ),
                "artifact_bytes": manifest.bytes,
            }

        def formats_case() -> dict[str, Any]:
            selected = {
                "csv": _catalog_resource(
                    catalog,
                    family="diputados",
                    dataset="DiputadosActivos",
                    format_name="csv",
                ),
                "json": _catalog_resource(
                    catalog,
                    family="diputados",
                    dataset="DiputadosActivos",
                    format_name="json",
                ),
                "xml": _catalog_resource(
                    catalog,
                    family="diputados",
                    dataset="DiputadosActivos",
                    format_name="xml",
                ),
                "html": _catalog_resource(
                    catalog,
                    family="transparencia",
                    dataset="RetribucionesCargosMesa",
                    format_name="html",
                ),
                "pdf": _catalog_resource(
                    catalog,
                    family="votaciones",
                    dataset="Votacion",
                    format_name="pdf",
                ),
                "zip": _catalog_resource(
                    catalog,
                    family="votaciones",
                    dataset="SesionVotaciones",
                    format_name="zip",
                ),
            }
            manifests = {
                name: _extract_one(client, resource, run_date=run_date.isoformat())
                for name, resource in selected.items()
            }
            return {
                name: {
                    "bytes": manifest.bytes,
                    "sha256": manifest.sha256,
                    "content_type": manifest.content_type,
                }
                for name, manifest in manifests.items()
            }

        for name, operation in (
            ("catalog", catalog_case),
            ("deputies", deputies_case),
            ("interests_and_assets", interests_case),
            ("initiatives", initiatives_case),
            ("interventions_text_and_callable_model", interventions_and_model_case),
            ("votes", votes_case),
            ("profiles_financial_documents_pdf_ocr_policy", profiles_documents_pdf_case),
            ("organs_and_memberships", organs_case),
            ("salary_entitlements", salary_case),
            ("raw_formats", formats_case),
        ):
            record_case(name, operation)

    report["finished_at"] = _now()
    report["status"] = (
        "passed"
        if report["cases"] and all(case["status"] == "passed" for case in report["cases"].values())
        else "failed"
    )
    _atomic_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--speaker", default="Pedro Sánchez")
    args = parser.parse_args(argv)
    report = verify(
        data_dir=args.data_dir,
        report_path=args.report,
        run_date=args.run_date,
        speaker=args.speaker,
    )
    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

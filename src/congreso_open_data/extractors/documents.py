from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.normalizers import discover_document_assets
from congreso_open_data.storage import BronzeManifest


def discover_document_resources_from_manifest(
    *,
    lake_root: Path,
    manifest: BronzeManifest,
    mime_types: set[str] | None = None,
) -> list[DatasetResource]:
    selected_mime_types = mime_types or {"application/pdf"}
    assets = discover_document_assets(root=lake_root, manifest=manifest, mime_types=mime_types)
    resources: dict[str, DatasetResource] = {}
    for asset in assets:
        if asset.get("mime_type") not in selected_mime_types:
            continue
        url = asset.get("url")
        if not url:
            continue
        resource = DatasetResource(
            family="documents",
            dataset=_dataset_for_asset(asset),
            format=_format_from_url(url),
            url=url,
            snapshot_token=asset.get("document_id"),
        )
        resources[resource.url] = resource
    return list(resources.values())


def extract_document_resources_from_manifest(
    *,
    lake_root: Path,
    manifest: BronzeManifest,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
    mime_types: set[str] | None = None,
) -> list[BronzeManifest]:
    client = client or CongresoHttpClient()
    return [
        extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )
        for resource in discover_document_resources_from_manifest(
            lake_root=lake_root,
            manifest=manifest,
            mime_types=mime_types,
        )
    ]


def _dataset_for_asset(asset: dict) -> str:
    if asset.get("mime_type") == "application/pdf":
        return "PdfDocument"
    return "Document"


def _format_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if path.endswith(".xml"):
        return "xml"
    if path.endswith(".mp4"):
        return "mp4"
    return "bin"

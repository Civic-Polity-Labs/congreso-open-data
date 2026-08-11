"""Official Congress catalog/acquisition adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from congreso_open_data.catalog import DatasetResource, discover_catalog
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.extractors.transparency import TRANSPARENCY_RESOURCES
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.models import ArtifactManifest, CatalogResource
from congreso_open_data.normalizers import public_manifest


class CongressSourceAdapter:
    name = "congreso.catalog"
    version = "1.0.0"

    def __init__(
        self,
        *,
        output_root: Path,
        transport: CongresoHttpClient | None = None,
    ) -> None:
        self.output_root = output_root
        self.transport = transport or CongresoHttpClient()

    def catalog(self) -> Iterator[CatalogResource]:
        seen: set[tuple[str, str, str]] = set()
        for resource in (*discover_catalog(client=self.transport), *TRANSPARENCY_RESOURCES):
            identity = (resource.family, resource.url, resource.format)
            if identity in seen:
                continue
            seen.add(identity)
            yield CatalogResource.model_validate(resource.__dict__)

    def acquire(self, resource: CatalogResource, *, run_date: str) -> ArtifactManifest:
        legacy = DatasetResource(**resource.model_dump())
        manifest = extract_resource(
            resource=legacy,
            run_date=run_date,
            output_root=self.output_root,
            client=self.transport,
        )
        return public_manifest(manifest)

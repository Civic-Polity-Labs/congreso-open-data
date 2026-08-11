from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from congreso_open_data import CatalogResource, CongressClient, ExtractionPlan
from congreso_open_data import adapters as adapters_module
from congreso_open_data.adapters import CongressSourceAdapter
from congreso_open_data.models import ArtifactManifest


class FakeAdapter:
    name = "fake"
    version = "1"

    def __init__(self, root) -> None:
        self.root = root
        self.calls = 0

    def catalog(self):
        yield CatalogResource(
            family="diputados",
            dataset="Diputados",
            format="json",
            url="https://example.test/deputies.json",
        )

    def acquire(self, resource: CatalogResource, *, run_date: str) -> ArtifactManifest:
        self.calls += 1
        content = json.dumps(
            [{"nombre": "Ana", "apellidos": "García", "legislatura": "XV"}],
            ensure_ascii=False,
        ).encode()
        relative = "bronze/diputados/deputies.json"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return ArtifactManifest(
            family=resource.family,
            dataset=resource.dataset,
            format=resource.format,
            source_url=resource.url,
            effective_url=resource.url,
            run_date=run_date,
            fetched_at=datetime.now(UTC),
            sha256=hashlib.sha256(content).hexdigest(),
            bytes=len(content),
            payload_path=relative,
            http_status=200,
        )


def test_client_injects_adapter_checkpoints_and_normalizes(tmp_path) -> None:
    adapter = FakeAdapter(tmp_path)
    client = CongressClient(output_root=tmp_path, adapter=adapter)
    plan = ExtractionPlan(families=("diputados",), output_root=tmp_path, max_workers=1)

    manifests = tuple(client.extract(plan))
    deputies = tuple(client.deputies(manifests))

    assert adapter.calls == 1
    assert len(manifests) == 1
    assert deputies[0].full_name == "García, Ana"
    assert client.last_run is not None
    assert client.last_run.planned == client.last_run.succeeded == 1
    assert Path(client.last_run.checkpoint_path).exists()
    assert (tmp_path / "extraction-runs" / plan.run_date).is_dir()


def test_public_catalog_includes_known_organ_and_transparency_pages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(adapters_module, "discover_catalog", lambda client: [])
    adapter = CongressSourceAdapter(output_root=tmp_path, transport=object())

    resources = list(adapter.catalog())

    assert any(item.family == "organos" for item in resources)
    assert any(item.family == "transparencia" for item in resources)
    assert len({(item.family, item.url, item.format) for item in resources}) == len(resources)

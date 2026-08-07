import json
from dataclasses import asdict
from pathlib import Path

from congreso_open_data.catalog import (
    DatasetResource,
    _resource_from_link,
    dataset_name_from_url,
    read_catalog,
)


def test_dataset_name_from_snapshot_url() -> None:
    url = "https://www.congreso.es/webpublica/opendata/diputados/DiputadosActivos__20260629050006.json"
    assert dataset_name_from_url(url) == "DiputadosActivos"


def test_votacion_link_extracts_context() -> None:
    resource = _resource_from_link(
        "votaciones",
        "JSON",
        "/webpublica/opendata/votaciones/Leg15/Sesion191/20260625/Votacion001/VOT_20260625150739.json",
    )
    assert resource is not None
    assert resource.dataset == "Votacion"
    assert resource.legislature == "Leg15"
    assert resource.session == "191"
    assert resource.vote_number == "001"
    assert resource.snapshot_token == "20260625150739"


def test_catalog_preserves_vote_png_variant() -> None:
    resource = _resource_from_link(
        "votaciones",
        "PNG",
        (
            "/webpublica/opendata/votaciones/Leg15/Sesion191/20260625/"
            "Votacion001/VOT_20260625150739.png"
        ),
    )

    assert resource is not None
    assert resource.dataset == "Votacion"
    assert resource.format == "png"
    assert resource.snapshot_token == "20260625150739"


def test_read_catalog_accepts_frozen_snapshot_envelope(tmp_path: Path) -> None:
    resource = DatasetResource(
        family="iniciativas",
        dataset="ProposicionesDeLey",
        format="json",
        url="https://www.congreso.es/webpublica/opendata/iniciativas/example.json",
        snapshot_token="20260802",
        legislature="Leg.15",
    )
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"version": "test", "resources": [asdict(resource)]}),
        encoding="utf-8",
    )

    assert read_catalog(path) == [resource]

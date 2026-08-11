"""Regression tests owned by the public acquisition package."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from congreso_open_data.batch_extract import BatchExtractionResult
from congreso_open_data.catalog import DatasetResource, write_catalog
from congreso_open_data.current_initiative_source import (
    CURRENT_INITIATIVE_OWNER_PREFIXES,
    audit_current_initiative_owner_index,
    extract_current_initiative_owner_sources,
    select_current_initiative_owner_resources,
)
from congreso_open_data.http import FetchResult
from congreso_open_data.storage import persist_bronze


def _resources() -> list[DatasetResource]:
    return [
        DatasetResource(
            family="iniciativas",
            dataset=dataset,
            format="json",
            url=f"https://www.congreso.es/webpublica/opendata/{dataset}.json",
            snapshot_token="20260715060000",
        )
        for dataset in CURRENT_INITIATIVE_OWNER_PREFIXES
    ]


def _record(dataset: str, *, prefix: str | None = None) -> dict[str, str]:
    effective_prefix = prefix or sorted(CURRENT_INITIATIVE_OWNER_PREFIXES[dataset])[0]
    return {
        "LEGISLATURA": "Leg.15",
        "NUMEXPEDIENTE": f"{effective_prefix}/000001/0000",
        "OBJETO": f"Registro {dataset}",
    }


def _write_owner_index(
    root: Path,
    *,
    datasets: tuple[str, ...] = tuple(CURRENT_INITIATIVE_OWNER_PREFIXES),
    wrong_prefix_dataset: str | None = None,
) -> Path:
    manifests = []
    resources = {resource.dataset: resource for resource in _resources()}
    for dataset in datasets:
        record = _record(
            dataset,
            prefix="999" if dataset == wrong_prefix_dataset else None,
        )
        manifest = persist_bronze(
            root=root,
            resource=resources[dataset],
            run_date="2026-07-15",
            result=FetchResult(
                url=resources[dataset].url,
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=json.dumps([record]).encode(),
            ),
        )
        manifests.append(asdict(manifest))
    index = root / "manifests" / "current-owners.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(manifests), encoding="utf-8")
    return index


def test_current_initiative_owner_audit_requires_all_exact_owners(tmp_path: Path) -> None:
    index = _write_owner_index(tmp_path)

    report = audit_current_initiative_owner_index(
        lake_root=tmp_path,
        manifest_index_path=index,
    )

    assert report["promotion_passed"] is True
    assert report["owner_rows"] == {
        "ProyectosDeLey": 1,
        "ProposicionesDeLey": 1,
        "PropuestasDeReforma": 1,
    }
    assert report["observed_legislatures"] == ["Leg.15"]
    assert report["metrics"]["duplicate_structured_keys"] == 0
    assert len(report["owner_keys"]) == 3
    assert report["owner_keys_sha256"]


def test_current_proposition_owner_accepts_all_official_prefix_families(
    tmp_path: Path,
) -> None:
    resources = {resource.dataset: resource for resource in _resources()}
    manifests = []
    for dataset, resource in resources.items():
        records = (
            [
                _record(dataset, prefix=prefix) | {"NUMEXPEDIENTE": f"{prefix}/{index:06d}/0000"}
                for index, prefix in enumerate(("122", "123", "124", "125"), 1)
            ]
            if dataset == "ProposicionesDeLey"
            else [_record(dataset)]
        )
        manifest = persist_bronze(
            root=tmp_path,
            resource=resource,
            run_date="2026-07-15",
            result=FetchResult(
                url=resource.url,
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=json.dumps(records).encode(),
            ),
        )
        manifests.append(asdict(manifest))
    index = tmp_path / "manifests" / "all-owner-prefixes.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(manifests), encoding="utf-8")

    report = audit_current_initiative_owner_index(
        lake_root=tmp_path,
        manifest_index_path=index,
    )

    assert report["promotion_passed"] is True
    assert report["owner_prefix_rows"]["ProposicionesDeLey"] == {
        "122": 1,
        "123": 1,
        "124": 1,
        "125": 1,
    }


def test_current_initiative_owner_audit_rejects_missing_owner_and_wrong_prefix(
    tmp_path: Path,
) -> None:
    missing = _write_owner_index(
        tmp_path / "missing",
        datasets=("ProyectosDeLey", "ProposicionesDeLey"),
    )
    wrong = _write_owner_index(
        tmp_path / "wrong",
        wrong_prefix_dataset="PropuestasDeReforma",
    )

    missing_report = audit_current_initiative_owner_index(
        lake_root=tmp_path / "missing",
        manifest_index_path=missing,
    )
    wrong_report = audit_current_initiative_owner_index(
        lake_root=tmp_path / "wrong",
        manifest_index_path=wrong,
    )

    assert missing_report["promotion_passed"] is False
    assert missing_report["gates"]["owner_manifest_set_complete"] is False
    assert wrong_report["promotion_passed"] is False
    assert wrong_report["gates"]["owner_prefixes_exact"] is False


def test_current_initiative_resource_selection_rejects_ambiguous_catalog() -> None:
    resources = _resources()
    with pytest.raises(RuntimeError, match="exactly one JSON resource"):
        select_current_initiative_owner_resources([*resources, resources[0]])


def test_current_initiative_source_runner_resumes_without_reextracting_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    write_catalog(_resources(), catalog_path)
    calls = 0

    def fake_batch(*, resources, run_date, output_root, manifest_index_path, **_kwargs):
        nonlocal calls
        calls += 1
        manifests = []
        for resource in resources:
            manifest = persist_bronze(
                root=output_root,
                resource=resource,
                run_date=run_date,
                result=FetchResult(
                    url=resource.url,
                    status_code=200,
                    headers={"Content-Type": "application/json"},
                    content=json.dumps([_record(resource.dataset)]).encode(),
                ),
            )
            manifests.append(manifest)
        manifest_index_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_index_path.write_text(
            json.dumps([asdict(item) for item in manifests]), encoding="utf-8"
        )
        return BatchExtractionResult(
            planned=3,
            completed=3,
            reused=0,
            downloaded=3,
            failed=0,
            manifest_index_path=str(manifest_index_path),
            state_path=str(manifest_index_path.with_suffix(".state.json")),
            manifests=tuple(manifests),
            failures=(),
        )

    monkeypatch.setattr(
        "congreso_open_data.current_initiative_source.extract_resource_batch",
        fake_batch,
    )
    first = extract_current_initiative_owner_sources(
        run_date="2026-07-15",
        lake_root=tmp_path,
        catalog_path=catalog_path,
    )
    resumed = extract_current_initiative_owner_sources(
        run_date="2026-07-15",
        lake_root=tmp_path,
        catalog_path=catalog_path,
    )

    assert first["status"] == "completed"
    assert resumed["source_fingerprint"] == first["source_fingerprint"]
    assert calls == 1

    plan_path = tmp_path / "plans" / "discovery" / "current-initiative-owners-2026-07-15.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan[0]["url"] += "?changed=true"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        extract_current_initiative_owner_sources(
            run_date="2026-07-15",
            lake_root=tmp_path,
            catalog_path=catalog_path,
        )

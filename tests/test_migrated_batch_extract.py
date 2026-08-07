from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import congreso_open_data.batch_extract as batch_extract_module
from congreso_open_data.batch_extract import extract_resource_batch
from congreso_open_data.catalog import DatasetResource
from congreso_open_data.http import FetchResult
from congreso_open_data.storage import persist_bronze


def test_extract_resource_batch_checkpoints_in_plan_order_and_resumes(tmp_path: Path) -> None:
    resources = [_resource("2"), _resource("1"), _resource("3")]
    calls: list[str] = []

    def extract_one(resource: DatasetResource):
        calls.append(str(resource.snapshot_token))
        return persist_bronze(
            root=tmp_path,
            resource=resource,
            run_date="2026-07-14",
            result=FetchResult(
                url=resource.url,
                status_code=200,
                headers={},
                content=json.dumps([{"token": resource.snapshot_token}]).encode(),
            ),
        )

    index = tmp_path / "manifests" / "historical.json"
    first = extract_resource_batch(
        resources=resources,
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        max_workers=2,
        extract_one=extract_one,
    )

    assert first.planned == 3
    assert first.completed == 3
    assert first.downloaded == 3
    assert [item["snapshot_token"] for item in json.loads(index.read_text("utf-8"))] == [
        "2",
        "1",
        "3",
    ]

    calls.clear()
    second = extract_resource_batch(
        resources=resources,
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        max_workers=2,
        extract_one=extract_one,
    )

    assert calls == []
    assert second.reused == 3
    assert second.downloaded == 0
    assert second.failed == 0


def test_extract_resource_batch_persists_failures_and_can_resume_them(tmp_path: Path) -> None:
    resources = [_resource("ok"), _resource("retry")]
    should_fail = True

    def extract_one(resource: DatasetResource):
        if resource.snapshot_token == "retry" and should_fail:
            raise RuntimeError("temporary failure")
        return persist_bronze(
            root=tmp_path,
            resource=resource,
            run_date="2026-07-14",
            result=FetchResult(
                url=resource.url,
                status_code=200,
                headers={},
                content=json.dumps({"token": resource.snapshot_token}).encode(),
            ),
        )

    index = tmp_path / "manifests" / "historical.json"
    first = extract_resource_batch(
        resources=resources,
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )

    state = json.loads(Path(first.state_path).read_text("utf-8"))
    assert first.completed == 1
    assert first.failed == 1
    assert state["status"] == "failed"
    assert state["failures"][0]["snapshot_token"] == "retry"

    should_fail = False
    second = extract_resource_batch(
        resources=resources,
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )

    assert second.reused == 1
    assert second.downloaded == 1
    assert second.failed == 0
    assert json.loads(Path(second.state_path).read_text("utf-8"))["status"] == "completed"


def test_batch_identity_distinguishes_post_parameters(tmp_path: Path) -> None:
    resources = [
        DatasetResource(
            family="organos",
            dataset="OrganInventory",
            format="json",
            url="https://example.test/inventory",
            snapshot_token="Leg15",
            post_data={"type": value},
        )
        for value in ("3", "4")
    ]
    calls: list[str] = []

    def extract_one(resource: DatasetResource):
        value = str((resource.post_data or {})["type"])
        calls.append(value)
        return persist_bronze(
            root=tmp_path,
            resource=resource,
            run_date="2026-08-02",
            result=FetchResult(
                url=resource.url,
                status_code=200,
                headers={},
                content=json.dumps({"type": value}).encode(),
            ),
        )

    index = tmp_path / "post-manifests.json"
    first = extract_resource_batch(
        resources=resources,
        run_date="2026-08-02",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )
    second = extract_resource_batch(
        resources=resources,
        run_date="2026-08-02",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )

    assert first.planned == first.completed == 2
    assert sorted(calls) == ["3", "4"]
    assert second.reused == 2


def test_batch_resume_matches_requested_url_after_official_fallback(tmp_path: Path) -> None:
    resource = DatasetResource(
        family="intervention_documents",
        dataset="InterventionFullTextPdf",
        format="pdf",
        url="https://www.congreso.es/public_oficiales/L9/CORT/DS/CM/CM_601.PDF",
        snapshot_token="L9-CM_601",
        legislature="Leg.9",
    )
    calls = 0

    def extract_one(item: DatasetResource):
        nonlocal calls
        calls += 1
        return persist_bronze(
            root=tmp_path,
            resource=item,
            run_date="2026-07-14",
            result=FetchResult(
                url="https://www.congreso.es/public_oficiales/L9/CONG/DS/CO/CO_601.PDF",
                status_code=200,
                headers={},
                content=b"%PDF-1.7\n",
            ),
        )

    index = tmp_path / "manifests" / "pdfs.json"
    first = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )
    second = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )

    assert first.manifests[0].source_url.endswith("/CONG/DS/CO/CO_601.PDF")
    assert first.manifests[0].requested_url == resource.url
    assert calls == 1
    assert second.reused == 1


def test_extract_resource_batch_validates_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        extract_resource_batch(
            resources=[_resource("1")],
            run_date="2026-07-14",
            output_root=tmp_path,
            manifest_index_path=tmp_path / "index.json",
            max_workers=0,
        )
    with pytest.raises(ValueError, match="checkpoint_interval"):
        extract_resource_batch(
            resources=[_resource("1")],
            run_date="2026-07-14",
            output_root=tmp_path,
            manifest_index_path=tmp_path / "index.json",
            checkpoint_interval=0,
        )


def test_extract_resource_batch_batches_aggregate_checkpoint_rewrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[Path] = []
    original = batch_extract_module._atomic_json_write

    def tracked_write(path: Path, payload) -> None:
        writes.append(path)
        original(path, payload)

    monkeypatch.setattr(batch_extract_module, "_atomic_json_write", tracked_write)

    def extract_one(resource: DatasetResource):
        return persist_bronze(
            root=tmp_path,
            resource=resource,
            run_date="2026-07-14",
            result=FetchResult(
                url=resource.url,
                status_code=200,
                headers={},
                content=json.dumps({"token": resource.snapshot_token}).encode(),
            ),
        )

    result = extract_resource_batch(
        resources=[_resource("1"), _resource("2"), _resource("3")],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=tmp_path / "index.json",
        extract_one=extract_one,
        checkpoint_interval=2,
    )

    assert result.completed == 3
    assert len(writes) == 6  # initial index/state + one interval + terminal index/state


def test_extract_resource_batch_fail_fast_does_not_record_cancelled_plan_as_failures(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fail_first(resource: DatasetResource):
        calls.append(str(resource.snapshot_token))
        if resource.snapshot_token == "0":
            raise RuntimeError("source failed")
        return persist_bronze(
            root=tmp_path,
            resource=resource,
            run_date="2026-07-14",
            result=FetchResult(
                url=resource.url,
                status_code=200,
                headers={},
                content=b"{}",
            ),
        )

    index = tmp_path / "index.json"
    with pytest.raises(RuntimeError, match="source failed"):
        extract_resource_batch(
            resources=[_resource(str(value)) for value in range(20)],
            run_date="2026-07-14",
            output_root=tmp_path,
            manifest_index_path=index,
            max_workers=1,
            extract_one=fail_first,
            continue_on_error=False,
            checkpoint_interval=10,
        )

    state = json.loads(index.with_suffix(".json.state.json").read_text("utf-8"))
    assert state["failed"] == 1
    assert state["failures"][0]["snapshot_token"] == "0"
    assert len(calls) < 20


def test_extract_resource_batch_redownloads_a_corrupted_reusable_blob(
    tmp_path: Path,
) -> None:
    resource = _resource("corrupt")
    calls = 0

    def extract_one(item: DatasetResource):
        nonlocal calls
        calls += 1
        return persist_bronze(
            root=tmp_path,
            resource=item,
            run_date="2026-07-14",
            result=FetchResult(
                url=item.url,
                status_code=200,
                headers={},
                content=b'[{"value":"official"}]',
            ),
        )

    index = tmp_path / "manifests" / "historical.json"
    first = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )
    blob = tmp_path / first.manifests[0].bronze_path
    blob.write_bytes(bytes(first.manifests[0].bytes))

    second = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )

    assert calls == 2
    assert second.reused == 0
    assert second.downloaded == 1
    assert json.loads(blob.read_text(encoding="utf-8")) == [{"value": "official"}]


def test_extract_resource_batch_redownloads_a_non_pdf_reusable_blob(
    tmp_path: Path,
) -> None:
    resource = DatasetResource(
        family="documents",
        dataset="official-record",
        format="pdf",
        url="https://example.test/document.pdf",
        snapshot_token="document-1",
        legislature="Leg.0",
    )
    calls = 0

    def extract_one(item: DatasetResource):
        nonlocal calls
        calls += 1
        return persist_bronze(
            root=tmp_path,
            resource=item,
            run_date="2026-07-14",
            result=FetchResult(
                url=item.url,
                status_code=200,
                headers={},
                content=(b"<html>not a PDF</html>" if calls == 1 else b"%PDF-1.7\n"),
            ),
        )

    index = tmp_path / "manifests" / "pdfs.json"
    first = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )
    second = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
    )

    assert first.downloaded == 1
    assert second.reused == 0
    assert second.downloaded == 1
    assert calls == 2


def test_extract_resource_batch_dataset_validator_retries_http_200_error_json(
    tmp_path: Path,
) -> None:
    resource = _resource("semantic-error")
    calls = 0

    def extract_one(item: DatasetResource):
        nonlocal calls
        calls += 1
        content = (
            b'{"error":"temporary official query error"}'
            if calls == 1
            else b'{"records":[{"value":"official"}]}'
        )
        return persist_bronze(
            root=tmp_path,
            resource=item,
            run_date="2026-07-14",
            result=FetchResult(
                url=item.url,
                status_code=200,
                headers={},
                content=content,
            ),
        )

    def validator(manifest) -> bool:
        payload = json.loads((tmp_path / manifest.bronze_path).read_text("utf-8"))
        return "records" in payload

    index = tmp_path / "manifests" / "historical.json"
    first = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
        manifest_validator=validator,
    )
    second = extract_resource_batch(
        resources=[resource],
        run_date="2026-07-14",
        output_root=tmp_path,
        manifest_index_path=index,
        extract_one=extract_one,
        manifest_validator=validator,
    )

    assert first.completed == 0
    assert first.failed == 1
    assert second.reused == 0
    assert second.downloaded == 1
    assert second.failed == 0
    assert calls == 2


def _resource(token: str) -> DatasetResource:
    base = DatasetResource(
        family="intervenciones",
        dataset="IntervencionesCronologicamente",
        format="json",
        url=f"https://example.test/{token}",
        snapshot_token=token,
        legislature="Leg.0",
    )
    return replace(base, post_data={"page": token})

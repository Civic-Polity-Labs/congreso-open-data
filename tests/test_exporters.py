from __future__ import annotations

import json

import pytest

from congreso_open_data.exporters import export_records


def test_jsonl_export_is_bounded_and_physically_multifile(tmp_path) -> None:
    paths = export_records(
        ({"id": index, "text": "bounded"} for index in range(5)),
        tmp_path / "export",
        format_name="jsonl",
        batch_size=2,
    )

    assert len(paths) == 3
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert [row["id"] for row in rows] == list(range(5))


def test_export_refuses_to_overwrite(tmp_path) -> None:
    output = tmp_path / "export"
    export_records(({"id": 1},), output, format_name="json")

    try:
        export_records(({"id": 2},), output, format_name="json")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing immutable export was overwritten")


def test_json_and_csv_exports_preserve_nested_values(tmp_path) -> None:
    json_paths = export_records(
        ({"id": 1, "nested": {"ok": True}},),
        tmp_path / "json",
        format_name="json",
    )
    csv_paths = export_records(
        ({"id": 1, "nested": {"ok": True}},),
        tmp_path / "csv",
        format_name="csv",
    )

    assert json.loads(json_paths[0].read_text())[0]["nested"] == {"ok": True}
    assert '""ok"": true' in csv_paths[0].read_text()


def test_export_rejects_unknown_format_and_non_mapping(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        export_records(({"id": 1},), tmp_path / "bad", format_name="xlsx")
    with pytest.raises(TypeError, match="expected a model or mapping"):
        export_records((object(),), tmp_path / "object", format_name="json")

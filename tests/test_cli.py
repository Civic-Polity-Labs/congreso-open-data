from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from typer.testing import CliRunner

from congreso_open_data import (
    ArtifactManifest,
    CatalogResource,
    InterventionRecord,
    QueryRun,
    SearchResult,
    SourceRef,
)
from congreso_open_data import cli as cli_module

runner = CliRunner()


def _source() -> SourceRef:
    return SourceRef(
        requested_url="https://example.test/intervention",
        sha256="a" * 64,
        adapter="test",
        adapter_version="1",
        normalization_version="1",
    )


class _FakeInterventions:
    def __init__(self, observed: list[object]) -> None:
        self.observed = observed

    def execute(self, query):
        self.observed.append(query)
        record = InterventionRecord(
            intervention_id="intervention-1",
            title="Intervencion de prueba",
            speaker="Ana Garcia",
            source=_source(),
        )
        return SearchResult(
            query=query,
            records=lambda: iter((record,)),
            run=QueryRun(
                run_id="run-1",
                query_fingerprint=query.fingerprint(),
                started_at=datetime.now(UTC),
            ),
        )


class _FakeModels:
    @staticmethod
    def names() -> tuple[str, ...]:
        return ("local-model", "remote-model")


class _FakeCongress:
    observed: ClassVar[list[object]] = []

    def __init__(self, **kwargs) -> None:
        self.interventions = _FakeInterventions(self.observed)
        self.models = _FakeModels()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _FakeClient:
    plans: ClassVar[list[object]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def catalog(self):
        return iter(
            (
                CatalogResource(
                    family="diputados",
                    dataset="Diputados",
                    format="json",
                    url="https://example.test/deputies.json",
                ),
            )
        )

    def extract(self, plan):
        self.plans.append(plan)
        return iter(
            (
                ArtifactManifest(
                    family="diputados",
                    dataset="Diputados",
                    format="json",
                    source_url="https://example.test/deputies.json",
                    effective_url="https://example.test/deputies.json",
                    run_date=plan.run_date,
                    fetched_at=datetime.now(UTC),
                    sha256="b" * 64,
                    bytes=2,
                    payload_path="bronze/deputies.json",
                ),
            )
        )


def test_cli_help_catalog_backends_and_models(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "Congress", _FakeCongress)
    monkeypatch.setattr(cli_module, "CongressClient", _FakeClient)
    monkeypatch.setattr(cli_module.default_registry, "discover_entry_points", lambda: ())

    help_result = runner.invoke(cli_module.app, ["--help"])
    catalog_result = runner.invoke(cli_module.app, ["catalog"])
    backends_result = runner.invoke(cli_module.app, ["backends", "--no-discover-plugins"])
    models_result = runner.invoke(cli_module.app, ["models"])

    assert help_result.exit_code == 0
    assert "interventions" in help_result.stdout
    assert json.loads(catalog_result.stdout)["family"] == "diputados"
    assert backends_result.exit_code == 0
    assert "json" in backends_result.stdout
    assert models_result.stdout.splitlines() == ["local-model", "remote-model"]


def test_cli_json_falls_back_to_ascii_for_a_limited_console(monkeypatch) -> None:
    monkeypatch.setattr(cli_module.sys, "stdout", SimpleNamespace(encoding="cp1252"))

    rendered = cli_module._console_json({"text": "línea ― oficial"})

    assert "\\u2015" in rendered
    assert json.loads(rendered) == {"text": "línea ― oficial"}


def test_cli_interventions_streams_records_and_run_metadata(monkeypatch, tmp_path: Path) -> None:
    _FakeCongress.observed.clear()
    monkeypatch.setattr(cli_module, "Congress", _FakeCongress)

    result = runner.invoke(
        cli_module.app,
        [
            "interventions",
            "--speaker",
            "Ana Garcia",
            "--date-from",
            "2026-01-01",
            "--date-to",
            "2026-02-01",
            "--text-policy",
            "none",
            "--data-dir",
            str(tmp_path),
            "--max-results",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    stdout_row = json.loads(result.stdout.splitlines()[0])
    stderr_row = json.loads(result.stderr)
    assert stdout_row["intervention_id"] == "intervention-1"
    assert stderr_row["query_run"]["complete"] is True
    query = _FakeCongress.observed[0]
    assert str(query.date_from) == "2026-01-01"
    assert str(query.date_to) == "2026-02-01"
    assert query.max_results == 10


def test_cli_rejects_non_iso_dates_and_builds_bounded_extract_plan(
    monkeypatch, tmp_path: Path
) -> None:
    invalid = runner.invoke(
        cli_module.app,
        ["interventions", "--date-from", "01/02/2026"],
    )
    assert invalid.exit_code == 2
    assert "expected YYYY-MM-DD" in invalid.output

    _FakeClient.plans.clear()
    monkeypatch.setattr(cli_module, "CongressClient", _FakeClient)
    extracted = runner.invoke(
        cli_module.app,
        [
            "extract",
            "--output-root",
            str(tmp_path),
            "--family",
            "diputados",
            "--format",
            "json",
            "--max-workers",
            "1",
            "--no-resume",
        ],
    )

    assert extracted.exit_code == 0, extracted.output
    assert json.loads(extracted.stdout)["dataset"] == "Diputados"
    plan = _FakeClient.plans[0]
    assert plan.families == ("diputados",)
    assert plan.formats == ("json",)
    assert plan.max_workers == 1
    assert plan.resume is False

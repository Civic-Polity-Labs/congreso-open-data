from __future__ import annotations

import importlib.util
import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

import congreso_open_data
from congreso_open_data import CatalogResource, ModelRequest, SourceRef

ROOT = Path(__file__).resolve().parents[1]


def _load_example(name: str) -> ModuleType:
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ExampleResult:
    def __init__(self) -> None:
        source = SimpleNamespace(sha256="a" * 64)
        self.rows = [
            SimpleNamespace(
                session_date=date(2026, 6, 24),
                title="Debate",
                speaker="Sánchez Pérez-Castejón, Pedro",
                text="Respuesta oficial.",
                text_method="pypdf_text",
                source=source,
            )
        ]
        self.run = SimpleNamespace(
            complete=True,
            resolved_entities={"speaker.XV": "Sánchez Pérez-Castejón, Pedro"},
            reused_resources=1,
            failures=(),
        )

    def collect(self, *, max_items: int) -> list[Any]:
        assert max_items == 100
        return self.rows


class _ExampleCongress:
    observed: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.interventions = self

    def __enter__(self) -> _ExampleCongress:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def search(self, **filters: Any) -> _ExampleResult:
        type(self).observed = filters
        return _ExampleResult()


def test_interventions_example_runs_through_the_public_facade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_example("interventions_last_three_months")
    monkeypatch.setattr(module, "Congress", _ExampleCongress)

    summary = module.run(speaker="Pedro Sánchez", data_dir=tmp_path)

    assert summary["records"] == 1
    assert summary["complete"] is True
    assert summary["sample"][0]["source_sha256"] == "a" * 64
    assert _ExampleCongress.observed == {
        "speaker": "Pedro Sánchez",
        "legislatures": ("XV",),
        "last_months": 3,
        "text_policy": "native",
        "max_results": 100,
    }


def test_interventions_example_escapes_text_unsupported_by_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example("interventions_last_three_months")
    monkeypatch.setattr(module.sys, "stdout", SimpleNamespace(encoding="cp1252"))

    rendered = module._console_json({"text": "línea ― oficial"})

    assert "\\u2015" in rendered
    assert json.loads(rendered) == {"text": "línea ― oficial"}


def test_callable_example_returns_a_literal_contract() -> None:
    module = _load_example("custom_model_callable")
    response = module.keyword_model(
        ModelRequest(
            text="La vivienda requiere una respuesta pública.",
            instructions="Return a keyword.",
            output_schema={},
            source=SourceRef(
                requested_url="https://example.test/intervention",
                sha256="b" * 64,
                fetched_at=datetime.now(UTC),
                adapter="test",
                adapter_version="1",
                normalization_version="1",
            ),
        )
    )

    candidate = response["candidates"][0]
    assert candidate["quote"] in "La vivienda requiere una respuesta pública."
    assert candidate["value"] == candidate["quote"].casefold()


def test_public_api_contract_example_covers_and_executes_every_export(tmp_path: Path) -> None:
    module = _load_example("public_api_contracts")

    assert set(module.PUBLIC_API_EXAMPLES) == set(congreso_open_data.__all__)
    summary = module.run(tmp_path)

    assert summary["status"] == "passed"
    assert summary["public_symbols"] == len(congreso_open_data.__all__) == 63
    assert summary["queries"] == 11
    assert summary["normalized_record_examples"] == 17
    assert summary["streaming_query_complete"] is True
    assert summary["low_level_run"]["planned"] == 1
    assert summary["low_level_run"]["succeeded"] == 1
    assert summary["low_level_run"]["failed"] == 0
    assert summary["low_level_deputy"]["full_name"] == "García, Ana"
    assert summary["facade_deputy"]["full_name"] == "García, Ana"
    assert summary["model_candidate"]["status"] == "review_required"
    assert summary["model_candidate"]["evidence"][0]["literal"] is True
    assert set(summary["caught_errors"]) == {
        "CongressError",
        "QueryValidationError",
        "EntityNotFoundError",
        "AmbiguousEntityError",
        "SourceUnavailableError",
        "SourceContractError",
        "IncompleteResultError",
        "ResultConsumedError",
    }
    checkpoint = Path(summary["low_level_run"]["checkpoint_path"])
    assert checkpoint.is_file()


def test_live_verifier_helpers_are_bounded_and_require_lineage() -> None:
    module = _load_example("verify_live_all_domains")
    resource = CatalogResource(
        family="votaciones",
        dataset="Votacion",
        format="json",
        url="https://example.test/vote.json",
    )

    assert (
        module._catalog_resource(
            (resource,),
            family="votaciones",
            dataset="Votacion",
            format_name="json",
        )
        is resource
    )
    assert module.MAX_ARTIFACT_BYTES == 32 * 1024**2
    with pytest.raises(RuntimeError, match="Catalog lacks"):
        module._catalog_resource(
            (resource,),
            family="votaciones",
            dataset="Votacion",
            format_name="xml",
        )


def test_live_verifier_resumes_only_the_same_run(tmp_path: Path) -> None:
    module = _load_example("verify_live_all_domains")
    data_dir = (tmp_path / "bronze").resolve()
    report_path = tmp_path / "report.json"
    report = module._load_or_create_report(
        data_dir=data_dir,
        report_path=report_path,
        run_date=date(2026, 8, 9),
        speaker="Pedro Sánchez",
    )
    report["cases"]["catalog"] = {"status": "passed"}
    report_path.write_text(module.json.dumps(report), encoding="utf-8")

    resumed = module._load_or_create_report(
        data_dir=data_dir,
        report_path=report_path,
        run_date=date(2026, 8, 9),
        speaker="Pedro Sánchez",
    )

    assert resumed["cases"]["catalog"]["status"] == "passed"
    assert len(resumed["resumed_at"]) == 1
    with pytest.raises(RuntimeError, match="another verification run"):
        module._load_or_create_report(
            data_dir=data_dir,
            report_path=report_path,
            run_date=date(2026, 8, 10),
            speaker="Pedro Sánchez",
        )


@pytest.mark.parametrize(
    "name",
    [
        "public_api_contracts",
        "interventions_last_three_months",
        "custom_model_callable",
        "search_all_domains",
        "verify_live_all_domains",
    ],
)
def test_every_example_has_working_help(name: str) -> None:
    module = _load_example(name)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0


def test_examples_guide_names_every_public_domain_and_runner() -> None:
    guide = (ROOT / "docs" / "EXAMPLES.md").read_text(encoding="utf-8")
    for token in (
        "congress.deputies.search",
        "congress.profiles.search",
        "congress.interests.search",
        "congress.financial_documents.search",
        "congress.initiatives.search",
        "congress.interventions.search",
        "congress.votes.search",
        "congress.organs.search",
        "congress.salary_entitlements.search",
        "congress.documents.search",
        "verify_live_all_domains.py",
        "CongressClient",
        "search_all_domains.py",
    ):
        assert token in guide


def test_api_cookbook_covers_every_supported_layer_and_client_normalizer() -> None:
    cookbook = (ROOT / "docs" / "API_COOKBOOK.md").read_text(encoding="utf-8")
    for token in (
        "public_api_contracts.py",
        "congreso_open_data.__all__",
        "CongressQuery",
        "SearchResult",
        "QueryRun",
        "SourceRef",
        "ModelBackend",
        "ModelRegistry",
        "StructuredModelExtractor",
        "ExtractionPlan",
        "ArtifactManifest",
        "ExtractionRun",
        "export_records",
        "CongressError",
        "congreso-open-data interventions",
    ):
        assert token in cookbook
    for method in (
        "deputies",
        "profiles",
        "interests",
        "financial_documents",
        "initiatives",
        "interventions",
        "intervention_occurrences",
        "votes",
        "organs",
        "documents",
        "document_texts",
        "speech_blocks",
        "salary_entitlements",
    ):
        assert f"client.{method}(" in cookbook

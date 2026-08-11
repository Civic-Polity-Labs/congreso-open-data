from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from congreso_open_data import (
    AmbiguousEntityError,
    CatalogResource,
    Congress,
    Deputy,
    DeputyQuery,
    ExtractionSpec,
    ExtractionTask,
    InterventionQuery,
    OrganQuery,
    QueryValidationError,
    ResultConsumedError,
    SourceRef,
    VoteQuery,
)
from congreso_open_data.api.congress import (
    _catalog_resources_for_query,
    _deduplicate_records,
    _filter_normalized_records,
    _record_matches,
)
from congreso_open_data.extractors.interventions import (
    discover_filtered_intervention_resources,
)
from congreso_open_data.http import FetchResult
from congreso_open_data.models import (
    ArtifactManifest,
    NominalVote,
    Organ,
    OrganMembership,
    VoteEvent,
    VoteItem,
)


class FakeCongressTransport:
    def __init__(self, *, ambiguous: bool = False, total: int = 2) -> None:
        self.ambiguous = ambiguous
        self.total = total
        self.post_calls: list[tuple[str, dict[str, str]]] = []

    def post(self, url: str, *, data: dict[str, str] | None = None) -> FetchResult:
        body = dict(data or {})
        self.post_calls.append((url, body))
        if "searchDiputados" in url:
            rows: list[dict[str, Any]] = [
                {
                    "codParlamentario": "189",
                    "idLegislatura": 15,
                    "apellidosNombre": "Sánchez Pérez-Castejón, Pedro",
                }
            ]
            if self.ambiguous:
                rows.append(
                    {
                        "codParlamentario": "999",
                        "idLegislatura": 15,
                        "apellidosNombre": "Sánchez García, Pedro",
                    }
                )
            return self._json(url, {"data": rows})
        if "resourceIDopendataExport" in url:
            return self._json(
                url,
                [
                    {
                        "tipo": "Pregunta oral en Pleno.",
                        "fase": "Pregunta-contestación",
                        "legislatura": "XV",
                        "hora_inicio": "09:00:00",
                        "hora_fin": "09:01:00",
                        "nombre_sesion": "Pleno",
                        "objeto_iniciativa": "Pregunta uno",
                        "fecha": "20/05/2026",
                        "orador": "Sánchez Pérez-Castejón, Pedro (GS)",
                        "numero_expediente": "180/000001/0000",
                        "enlace_texto_integro": (
                            "https://www.congreso.es/es/busqueda-de-intervenciones?"
                            "_intervenciones_id_texto=(DSCD-15-PL-1.CODI.)"
                        ),
                    },
                    {
                        "tipo": "Pregunta oral en Pleno.",
                        "fase": "Pregunta-contestación",
                        "legislatura": "XV",
                        "hora_inicio": "09:02:00",
                        "hora_fin": "09:03:00",
                        "nombre_sesion": "Pleno",
                        "objeto_iniciativa": "Pregunta dos",
                        "fecha": "24/06/2026",
                        "orador": "Sánchez Pérez-Castejón, Pedro (GS)",
                        "numero_expediente": "180/000002/0000",
                        "enlace_texto_integro": (
                            "https://www.congreso.es/es/busqueda-de-intervenciones?"
                            "_intervenciones_id_texto=(DSCD-15-PL-1.CODI.)"
                        ),
                    },
                ],
            )
        if "filtrarListado" in url:
            return self._json(url, {"intervenciones_encontradas": str(self.total)})
        raise AssertionError(f"Unexpected POST: {url}")

    def get(self, url: str) -> FetchResult:
        content = (
            '<html><body><div class="textoIntegro">'
            "El señor SÁNCHEZ PÉREZ-CASTEJÓN: Primera respuesta oficial.<br>"
            "El señor SÁNCHEZ PÉREZ-CASTEJÓN: Segunda respuesta oficial."
            "</div></body></html>"
        ).encode()
        return FetchResult(
            url=url,
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=content,
        )

    @staticmethod
    def _json(url: str, payload: Any) -> FetchResult:
        return FetchResult(
            url=url,
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(payload, ensure_ascii=False).encode(),
        )

    def close(self) -> None:
        raise AssertionError("An injected transport must not be closed by Congress")


class FakeDeputyAdapter:
    name = "fake-deputies"
    version = "1"

    def __init__(self, root: Path) -> None:
        self.root = root

    def catalog(self):
        yield CatalogResource(
            family="diputados",
            dataset="Diputados",
            format="json",
            url="https://example.test/deputies.json",
            legislature="Leg.15",
        )

    def acquire(self, resource: CatalogResource, *, run_date: str) -> ArtifactManifest:
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
            legislature="Leg.15",
        )


def test_filtered_intervention_plan_uses_official_filters_and_pagination() -> None:
    transport = FakeCongressTransport(total=201)

    plan = discover_filtered_intervention_resources(
        client=transport,
        legislature="15",
        speaker="Sánchez Pérez-Castejón, Pedro",
        date_from=date(2026, 5, 9),
        date_to=date(2026, 8, 9),
        max_results=500,
    )

    assert plan.official_total == 201
    assert len(plan.resources) == 3
    assert plan.resources[-1].post_data["_intervenciones_lastResult"] == "201"
    assert plan.resources[0].post_data["_intervenciones_fechaDesde"] == "09/05/2026"
    assert plan.resources[0].post_data["_intervenciones_fechaHasta"] == "09/08/2026"
    assert plan.resources[0].post_data["_intervenciones_orador"].startswith("Sánchez")


def test_congress_facade_resolves_identity_reconciles_and_preserves_post(
    tmp_path: Path,
) -> None:
    transport = FakeCongressTransport()
    congress = Congress(data_dir=tmp_path, transport=transport, today=date(2026, 8, 9))

    result = congress.interventions.search(
        speaker="Pedro Sanchez",
        last_months=3,
        text_policy="none",
        max_results=10,
    )
    rows = list(result)

    assert [str(item.session_date) for item in rows] == ["2026-05-20", "2026-06-24"]
    assert result.run.raw_records == result.run.normalized_records == 2
    assert result.run.complete is True
    assert result.run.resolved_entities == {
        "speaker.XV": "Sánchez Pérez-Castejón, Pedro",
        "speaker_id.XV": "189",
    }
    assert result.manifests[0].request_method == "POST"
    assert result.manifests[0].request_parameters_sha256
    count_post = next(data for url, data in transport.post_calls if "filtrarListado" in url)
    assert count_post["_intervenciones_fechaDesde"] == "09/05/2026"
    assert count_post["_intervenciones_fechaHasta"] == "09/08/2026"
    with pytest.raises(ResultConsumedError):
        list(result)


def test_congress_facade_rejects_ambiguous_speaker_before_export(tmp_path: Path) -> None:
    transport = FakeCongressTransport(ambiguous=True)
    congress = Congress(data_dir=tmp_path, transport=transport, today=date(2026, 8, 9))

    with pytest.raises(AmbiguousEntityError, match="Sánchez Pérez-Castejón"):
        congress.interventions.search(
            speaker="Pedro Sanchez",
            last_months=3,
            text_policy="none",
        )

    assert not any("resourceIDopendataExport" in url for url, _ in transport.post_calls)


def test_congress_facade_matches_repeated_native_speech_occurrences(tmp_path: Path) -> None:
    congress = Congress(
        data_dir=tmp_path,
        transport=FakeCongressTransport(),
        today=date(2026, 8, 9),
    )

    result = congress.interventions.search(
        speaker="Pedro Sanchez",
        last_months=3,
        text_policy="native",
        max_results=10,
    )
    rows = result.collect(max_items=10)

    assert [item.text_status for item in rows] == ["matched", "matched"]
    assert rows[0].text == "Primera respuesta oficial."
    assert rows[1].text == "Segunda respuesta oficial."
    assert all(item.text_method == "official_html_transcript" for item in rows)
    assert result.run.unmatched_text_records == 0
    assert result.run.complete is True


def test_congress_facade_runs_a_user_callable_as_reviewable_extraction(
    tmp_path: Path,
) -> None:
    congress = Congress(
        data_dir=tmp_path,
        transport=FakeCongressTransport(),
        today=date(2026, 8, 9),
    )
    congress.models.register_callable(
        "mine",
        model="my-local-model",
        version="1",
        function=lambda request: {
            "candidates": [
                {
                    "kind": "topic",
                    "value": "respuesta",
                    "quote": "respuesta oficial",
                }
            ]
        },
    )
    task = ExtractionTask(
        name="topics",
        instructions="Extract literal topics.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="my-local-model"),
    )

    rows = congress.interventions.search(
        speaker="Pedro Sanchez",
        last_months=3,
        extractions=(task,),
        max_results=10,
    ).collect(max_items=10)

    assert [len(item.extractions) for item in rows] == [1, 1]
    assert all(item.extractions[0].status == "review_required" for item in rows)
    assert all(item.extractions[0].source.model == "my-local-model" for item in rows)
    assert all(item.extractions[0].evidence[0].literal for item in rows)


def test_queries_are_serializable_and_do_not_contain_runtime_clients() -> None:
    query = InterventionQuery(
        speaker="Pedro Sanchez",
        last_months=3,
        extractions=(
            {
                "name": "topics",
                "instructions": "Return literal topics.",
                "backend": ExtractionSpec(engine="llm", backend="mine", model="unit"),
            },
        ),
    )

    payload = query.model_dump_json()

    assert json.loads(payload)["extractions"][0]["backend"]["backend"] == "mine"
    assert "client" not in payload.casefold()
    assert len(query.fingerprint()) == 64

    upper_bound = InterventionQuery(date_to=date(2026, 1, 31)).resolved(today=date(2026, 8, 9))
    lower_bound = InterventionQuery(date_from=date(2026, 1, 1)).resolved(today=date(2026, 8, 9))
    assert upper_bound.date_from is None
    assert upper_bound.date_to == date(2026, 1, 31)
    assert lower_bound.date_from == date(2026, 1, 1)
    assert lower_bound.date_to is None


def test_friendly_search_wraps_query_validation_errors(tmp_path: Path) -> None:
    congress = Congress(
        data_dir=tmp_path,
        transport=FakeCongressTransport(),
        today=date(2026, 8, 9),
    )

    with pytest.raises(QueryValidationError, match="last_months"):
        congress.interventions.search(
            speaker="Pedro Sanchez",
            last_months=3,
            date_from=date(2026, 1, 1),
            text_policy="none",
        )


def test_refresh_policy_reuses_or_redownloads_bronze_explicitly(tmp_path: Path) -> None:
    transport = FakeCongressTransport()
    congress = Congress(data_dir=tmp_path, transport=transport, today=date(2026, 8, 9))

    for refresh in ("auto", "auto", "always"):
        congress.interventions.search(
            speaker="Pedro Sanchez",
            last_months=3,
            text_policy="none",
            refresh=refresh,
            max_results=10,
        ).collect(max_items=10)

    export_calls = [url for url, _ in transport.post_calls if "resourceIDopendataExport" in url]
    assert len(export_calls) == 2


def test_generic_domain_services_use_the_same_bounded_result_contract(tmp_path: Path) -> None:
    congress = Congress(data_dir=tmp_path, adapter=FakeDeputyAdapter(tmp_path))

    result = congress.deputies.search(name="Ana", legislatures=(15,), max_results=10)
    rows = result.collect(max_items=10)

    assert len(rows) == 1
    assert rows[0].full_name == "García, Ana"
    assert rows[0].source.adapter
    assert result.run.complete is True
    assert result.run.planned_resources == result.run.succeeded_resources == 1


def test_facade_reuses_catalog_within_session_and_refreshes_explicitly(tmp_path: Path) -> None:
    adapter = FakeDeputyAdapter(tmp_path)
    catalog_calls = 0
    original_catalog = adapter.catalog

    def counted_catalog():
        nonlocal catalog_calls
        catalog_calls += 1
        yield from original_catalog()

    adapter.catalog = counted_catalog  # type: ignore[method-assign]
    congress = Congress(data_dir=tmp_path, adapter=adapter)

    congress.deputies.search(name="Ana", max_results=10).collect(max_items=10)
    congress.deputies.search(name="Ana", max_results=10).collect(max_items=10)
    assert catalog_calls == 1

    congress.deputies.search(name="Ana", refresh="always", max_results=10).collect(max_items=10)
    assert catalog_calls == 2


def test_heterogeneous_domain_filters_exclude_records_without_the_requested_field() -> None:
    source = SourceRef(
        requested_url="https://example.test/source",
        sha256="a" * 64,
        adapter="test",
        adapter_version="1",
        normalization_version="1",
    )
    event = VoteEvent(vote_id="vote-1", session="10", source=source)
    item = VoteItem(vote_item_id="item-1", vote_id="vote-1", source=source)
    nominal = NominalVote(
        nominal_vote_id="nominal-1",
        vote_id="vote-1",
        deputy_name="Ana Garcia",
        position="yes",
        source=source,
    )

    by_session = VoteQuery(session="10")
    assert _record_matches(event, by_session) is True
    assert _record_matches(item, by_session) is False
    assert _record_matches(nominal, by_session) is False

    by_deputy = VoteQuery(deputy="Ana")
    assert _record_matches(nominal, by_deputy) is True
    assert _record_matches(event, by_deputy) is False
    assert _record_matches(item, by_deputy) is False

    organ = Organ(organ_id="organ-1", name="Comision", organ_type="commission", source=source)
    membership = OrganMembership(
        membership_id="membership-1",
        organ_id="organ-1",
        source=source,
    )
    by_type = OrganQuery(organ_type="commission")
    assert _record_matches(organ, by_type) is True
    assert _record_matches(membership, by_type) is False


def test_person_filters_are_order_and_accent_insensitive() -> None:
    source = SourceRef(
        requested_url="https://example.test/source",
        sha256="a" * 64,
        adapter="test",
        adapter_version="1",
        normalization_version="1",
    )
    deputy = Deputy(
        source=source,
        deputy_id="person-1",
        full_name="Sánchez Pérez-Castejón, Pedro",
    )

    assert _record_matches(deputy, DeputyQuery(name="Pedro Sanchez")) is True


def test_catalog_planner_selects_only_formats_supported_by_each_domain() -> None:
    resources = tuple(
        CatalogResource(
            family="diputados",
            dataset="DiputadosActivos",
            format=format_name,
            url=f"https://example.test/deputies.{format_name}",
        )
        for format_name in ("csv", "json", "xml")
    ) + tuple(
        CatalogResource(
            family="votaciones",
            dataset="Votacion",
            format=format_name,
            url=f"https://example.test/vote.{format_name}",
            legislature="Leg15",
            session="193",
            vote_number="1",
        )
        for format_name in ("json", "pdf", "xml")
    )

    deputies = _catalog_resources_for_query(DeputyQuery(), resources)
    votes = _catalog_resources_for_query(VoteQuery(session="193", vote_number="1"), resources)

    assert [(item.dataset, item.format) for item in deputies] == [("DiputadosActivos", "json")]
    assert [(item.dataset, item.format) for item in votes] == [("Votacion", "json")]


def test_vote_scope_keeps_event_items_and_nominals_without_collapsing_children() -> None:
    source = SourceRef(
        requested_url="https://example.test/source",
        sha256="a" * 64,
        adapter="test",
        adapter_version="1",
        normalization_version="1",
    )
    records = [
        VoteEvent(vote_id="vote-1", session="193", vote_number="1", source=source),
        VoteItem(vote_item_id="item-1", vote_id="vote-1", source=source),
        NominalVote(
            nominal_vote_id="nominal-1",
            vote_id="vote-1",
            deputy_name="Diputada, Ana",
            position="Sí",
            source=source,
        ),
        NominalVote(
            nominal_vote_id="nominal-2",
            vote_id="vote-1",
            deputy_name="Diputado, Luis",
            position="No",
            source=source,
        ),
    ]

    scoped = _filter_normalized_records(records, VoteQuery(session="193", vote_number="1"))
    deduplicated, duplicates = _deduplicate_records(scoped)

    assert len(deduplicated) == 4
    assert duplicates == 0
    by_deputy = _filter_normalized_records(
        records,
        VoteQuery(session="193", vote_number="1", deputy="Ana"),
    )
    assert [item.nominal_vote_id for item in by_deputy] == ["nominal-1"]


def test_every_search_shortcut_has_an_explicit_typed_signature(tmp_path: Path) -> None:
    congress = Congress(data_dir=tmp_path, adapter=FakeDeputyAdapter(tmp_path))

    for service in (
        congress.deputies,
        congress.profiles,
        congress.interests,
        congress.financial_documents,
        congress.initiatives,
        congress.interventions,
        congress.votes,
        congress.organs,
        congress.salary_entitlements,
        congress.documents,
    ):
        parameters = inspect.signature(service.search).parameters.values()
        assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in parameters)

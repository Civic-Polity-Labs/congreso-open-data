from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from congreso_open_data import CongressClient, ExtractionPlan
from congreso_open_data.models import ArtifactManifest, CatalogResource, ExtractionSpec
from congreso_open_data.normalizers import (
    document_texts,
    financial_documents,
    initiatives,
    interests,
    intervention_occurrences,
    legacy_manifest,
    public_manifest,
    salary_entitlements,
    speech_blocks,
    votes,
)
from congreso_open_data.storage import BronzeManifest

MINIMAL_TEXT_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
    b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    b"5 0 obj << /Length 44 >> stream\n"
    b"BT /F1 12 Tf 72 100 Td (Declaracion test) Tj ET\n"
    b"endstream endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n0000000241 00000 n \n"
    b"0000000311 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\n"
    b"startxref\n405\n%%EOF\n"
)


def _manifest(
    root: Path,
    *,
    family: str,
    dataset: str,
    format_name: str,
    content: bytes,
    url: str = "https://example.test/source",
    legislature: str | None = None,
) -> ArtifactManifest:
    digest = hashlib.sha256(content).hexdigest()
    path = root / f"{dataset}-{digest[:8]}.{format_name}"
    path.write_bytes(content)
    return ArtifactManifest(
        family=family,
        dataset=dataset,
        format=format_name,
        source_url=url,
        effective_url=url,
        run_date="2026-08-08",
        fetched_at=datetime.now(UTC),
        sha256=digest,
        bytes=len(content),
        payload_path=str(path),
        legislature=legislature,
    )


def test_public_normalizers_cover_interests_and_historical_initiatives(tmp_path: Path) -> None:
    interest_manifest = _manifest(
        tmp_path,
        family="diputados",
        dataset="docacteco",
        format_name="json",
        content=json.dumps([{"NOMBRE": "Diputada, Ana", "DESCRIPCION": "Docencia"}]).encode(),
    )
    initiative_manifest = _manifest(
        tmp_path,
        family="iniciativas",
        dataset="ProyectosDeLey",
        format_name="json",
        legislature="Leg.14",
        content=json.dumps(
            {
                "iniciativas_encontradas": "1",
                "lista_iniciativas": {
                    "iniciativa1": {
                        "legislatura": "XIV",
                        "titulo": "Proyecto de Ley de prueba",
                        "id_iniciativa": "121/000001",
                    }
                },
            }
        ).encode(),
    )

    interest = next(interests((interest_manifest,), root=tmp_path))
    initiative = next(initiatives((initiative_manifest,), root=tmp_path))

    assert interest.full_name == "Diputada, Ana"
    assert interest.description == "Docencia"
    assert initiative.legislature == "Leg.14"
    assert initiative.file_number == "121/000001/0000"
    assert initiative.title == "Proyecto de Ley de prueba"


def test_public_vote_normalizer_accepts_official_numeric_identifiers(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        family="votaciones",
        dataset="Votacion",
        format_name="json",
        legislature="Leg10",
        content=json.dumps(
            {
                "informacion": {
                    "fecha": "14/03/2013",
                    "sesion": 91,
                    "numeroVotacion": 38,
                    "titulo": "Dictamen",
                    "textoExpediente": "Suplicatorio",
                    "votacionesConjuntas": [],
                },
                "totales": {
                    "presentes": 1,
                    "afavor": 1,
                    "enContra": 0,
                    "abstenciones": 0,
                    "noVotan": 0,
                },
                "votaciones": [],
            }
        ).encode(),
    )

    event = next(votes((manifest,), root=tmp_path))

    assert event.session == "91"
    assert event.vote_number == "38"


def test_public_manifest_round_trip_preserves_post_provenance() -> None:
    bronze = BronzeManifest(
        family="intervenciones",
        dataset="IntervencionesCronologicamente",
        format="json",
        source_url="https://www.congreso.es/export",
        snapshot_token="filtered-1",
        run_date="2026-08-09",
        extracted_at="2026-08-09T12:00:00+00:00",
        sha256="a" * 64,
        bytes=10,
        bronze_path="bronze/interventions.json",
        status_code=200,
        request_method="POST",
        request_parameters_json='{"speaker":"Pedro"}',
        request_parameters_sha256="b" * 64,
    )

    public = public_manifest(bronze)
    restored = legacy_manifest(public)

    assert public.request_method == "POST"
    assert public.request_parameters_sha256 == "b" * 64
    assert restored.request_method == "POST"
    assert restored.request_parameters_sha256 == "b" * 64


def test_public_normalizers_cover_profile_financial_documents(tmp_path: Path) -> None:
    html = b"""
    <html><body><main>
      <p>Diputada Ejemplo, Ana</p><p>XV Legislatura (2023-Actualidad)</p>
      <a href="https://www.congreso.es/docbienes/leg15/asset.pdf">
        Declaracion de Bienes y Rentas
      </a>
    </main></body></html>
    """
    manifest = _manifest(
        tmp_path,
        family="diputados",
        dataset="DeputyProfile",
        format_name="html",
        content=html,
        url=(
            "https://www.congreso.es/es/busqueda-de-diputados?codParlamentario=1&idLegislatura=XV"
        ),
        legislature="Leg.15",
    )

    item = next(financial_documents((manifest,), root=tmp_path))

    assert item.document_kind == "assets_income"
    assert item.url.endswith("asset.pdf")
    assert item.deputy_id
    assert item.source.method == "official_html"


def test_public_normalizers_cover_occurrences_pdf_text_speech_and_salary(
    tmp_path: Path,
) -> None:
    intervention_manifest = _manifest(
        tmp_path,
        family="intervenciones",
        dataset="IntervencionesCronologicamente",
        format_name="json",
        content=json.dumps(
            [
                {
                    "LEGISLATURA": "XV",
                    "SESION": "01/08/2026",
                    "ORADOR": "Diputada, Ana",
                    "ENLACETEXTOINTEGRO": "https://www.congreso.es/test?p=1",
                }
            ]
        ).encode(),
        legislature="Leg.15",
    )
    pdf_manifest = _manifest(
        tmp_path,
        family="documents",
        dataset="PdfDocument",
        format_name="pdf",
        content=MINIMAL_TEXT_PDF,
        url="https://www.congreso.es/docbienes/leg15/test.pdf",
    )
    speech_manifest = _manifest(
        tmp_path,
        family="intervention_documents",
        dataset="InterventionFullText",
        format_name="html",
        content=(
            '<html><body><div class="textoIntegro">'
            "La señora PRESIDENTA: Tiene la palabra.<br>"
            "La señora DIPUTADA EJEMPLO: Muchas gracias."
            "</div></body></html>"
        ).encode(),
        url="https://www.congreso.es/public_oficiales/L15/CONG/DS/PL/DSCD-15-PL-1.PDF",
        legislature="Leg.15",
    )
    salary_manifest = _manifest(
        tmp_path,
        family="transparencia",
        dataset="RetribucionesCargosMesa",
        format_name="html",
        content=(
            "<html><body><main><p>Secretario General</p><p>Sueldo base:</p>"
            "<p>1.234,56 €</p></main></body></html>"
        ).encode(),
    )

    occurrence = next(intervention_occurrences((intervention_manifest,), root=tmp_path))
    document = next(document_texts((pdf_manifest,), root=tmp_path))
    speech = next(speech_blocks((speech_manifest,), root=tmp_path))
    salary = next(salary_entitlements((salary_manifest,), root=tmp_path))

    assert str(occurrence.date) == "2026-08-01"
    assert document.text == "Declaracion test"
    assert document.evidence[0].page == 1
    assert speech.document_id and speech.text
    assert salary.amount_eur == 1234.56
    assert salary.label == "Sueldo base"


class _NoopAdapter:
    name = "noop"
    version = "1"

    def catalog(self):
        return iter(())

    def acquire(self, resource, *, run_date):
        raise AssertionError("not reached")


def test_client_rejects_root_mismatch_and_oversized_plans(tmp_path: Path) -> None:
    client = CongressClient(output_root=tmp_path / "client", adapter=_NoopAdapter())
    with pytest.raises(ValueError, match="must match"):
        tuple(client.extract(ExtractionPlan(output_root=tmp_path / "plan")))

    resources = tuple(
        CatalogResource(
            family="diputados",
            dataset=f"dataset-{index}",
            format="json",
            url=f"https://example.test/{index}.json",
        )
        for index in range(2)
    )
    with pytest.raises(ValueError, match="configured maximum"):
        tuple(
            CongressClient(
                output_root=tmp_path,
                adapter=_NoopAdapter(),
            ).extract(
                ExtractionPlan(
                    output_root=tmp_path,
                    resources=resources,
                    max_resources=1,
                )
            )
        )


def test_client_refuses_unbounded_in_memory_backend_input(tmp_path: Path) -> None:
    content = b'{"name":"too-large-for-test-budget"}'

    class Adapter(_NoopAdapter):
        def catalog(self):
            yield CatalogResource(
                family="diputados",
                dataset="bounded",
                format="json",
                url="https://example.test/bounded.json",
            )

        def acquire(self, resource, *, run_date):
            return _manifest(
                tmp_path,
                family=resource.family,
                dataset=resource.dataset,
                format_name=resource.format,
                content=content,
            )

    client = CongressClient(output_root=tmp_path, adapter=Adapter())
    plan = ExtractionPlan(
        output_root=tmp_path,
        specs=(ExtractionSpec(engine="native", backend="native-json", model="json"),),
        max_artifact_bytes=len(content) - 1,
        max_workers=1,
    )

    with pytest.raises(ValueError, match="configured limit"):
        tuple(client.extract(plan))

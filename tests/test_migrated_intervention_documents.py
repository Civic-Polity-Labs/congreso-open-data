"""Regression tests owned by the public acquisition package."""

import json
from dataclasses import asdict

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.extractors.interventions import (
    discover_intervention_document_resources_from_index,
    discover_intervention_pdf_resources_from_manifest,
    discover_intervention_text_resources_from_manifest,
)
from congreso_open_data.http import FetchResult
from congreso_open_data.storage import persist_bronze


def test_discover_intervention_text_resources_deduplicates_by_document(tmp_path) -> None:
    resource = DatasetResource(
        family="intervenciones",
        dataset="IntervencionesCronologicamente",
        format="json",
        url="https://example.test/intervenciones.json",
        snapshot_token="20260630050000",
    )
    text_url = (
        "https://www.congreso.es/busqueda-de-intervenciones?"
        "_intervenciones_id_texto=(DSCD-15-PL-1.CODI.)#(Pagina15)"
    )
    rows = [
        {
            "LEGISLATURA": "Leg.15",
            "ORADOR": "A",
            "ENLACETEXTOINTEGRO": text_url,
            "ENLACEPDF": "https://www.congreso.es/public/DSCD-15-PL-1.PDF#page=15",
        },
        {
            "LEGISLATURA": "Leg.15",
            "ORADOR": "B",
            "ENLACETEXTOINTEGRO": text_url.replace("Pagina15", "Pagina16"),
            "ENLACEPDF": "https://www.congreso.es/public/DSCD-15-PL-1.PDF#page=16",
        },
    ]
    manifest = persist_bronze(
        root=tmp_path,
        resource=resource,
        run_date="2026-06-30",
        result=FetchResult(
            url=resource.url,
            status_code=200,
            headers={},
            content=json.dumps(rows).encode(),
        ),
    )
    resources = discover_intervention_text_resources_from_manifest(
        lake_root=tmp_path,
        manifest=manifest,
    )
    assert len(resources) == 1
    assert resources[0].family == "intervention_documents"
    assert resources[0].dataset == "InterventionFullText"
    assert resources[0].format == "html"
    assert resources[0].snapshot_token == "DSCD-15-PL-1.CODI."
    assert "#" not in resources[0].url

    pdf_resources = discover_intervention_pdf_resources_from_manifest(
        lake_root=tmp_path,
        manifest=manifest,
    )
    assert len(pdf_resources) == 1
    assert pdf_resources[0].family == "intervention_documents"
    assert pdf_resources[0].dataset == "InterventionFullTextPdf"
    assert pdf_resources[0].format == "pdf"
    assert pdf_resources[0].snapshot_token == "DSCD-15-PL-1.CODI."
    assert "#" not in pdf_resources[0].url


def test_document_plan_deduplicates_globally_across_export_pages(tmp_path) -> None:
    manifests = []
    for page in (1, 2):
        resource = DatasetResource(
            family="intervenciones",
            dataset="IntervencionesCronologicamente",
            format="json",
            url=f"https://example.test/intervenciones-{page}.json",
            snapshot_token=f"page-{page}",
            legislature="Leg.15",
        )
        rows = [
            {
                "LEGISLATURA": "XV",
                "ENLACETEXTOINTEGRO": (
                    "https://www.congreso.es/busqueda-de-intervenciones?"
                    "_intervenciones_id_texto=(DSCD-15-PL-1.CODI.)"
                ),
                "ENLACEPDF": "https://www.congreso.es/public/DSCD-15-PL-1.PDF",
            }
        ]
        manifests.append(
            persist_bronze(
                root=tmp_path,
                resource=resource,
                run_date="2026-08-01",
                result=FetchResult(
                    url=resource.url,
                    status_code=200,
                    headers={},
                    content=json.dumps(rows).encode(),
                ),
            )
        )
    index = tmp_path / "index.json"
    index.write_text(json.dumps([asdict(manifest) for manifest in manifests]), encoding="utf-8")

    resources = discover_intervention_document_resources_from_index(
        lake_root=tmp_path,
        manifest_index_path=index,
    )

    assert [(resource.dataset, resource.legislature) for resource in resources] == [
        ("InterventionFullText", "Leg.15"),
        ("InterventionFullTextPdf", "Leg.15"),
    ]

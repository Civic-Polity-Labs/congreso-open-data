import io
import zipfile

from PIL import Image

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.http import FetchResult
from congreso_open_data.storage import (
    bronze_payload_is_valid,
    content_matches_format_contract,
    content_type_matches_format,
    persist_bronze,
)


def test_content_type_contract_requires_declared_compatible_mime() -> None:
    assert content_type_matches_format("application/json; charset=UTF-8", "json")
    assert content_type_matches_format("application/octet-stream", "zip")
    assert content_type_matches_format("application/download", "xml")
    assert not content_type_matches_format(None, "json")
    assert not content_type_matches_format("text/html", "json")
    assert not content_type_matches_format("text/html", "pdf")


def test_persist_bronze_is_content_addressed(tmp_path) -> None:
    resource = DatasetResource(
        family="diputados",
        dataset="DiputadosActivos",
        format="json",
        url="https://example.test/file.json",
        snapshot_token="20260629050006",
    )
    result = FetchResult(url=resource.url, status_code=200, headers={}, content=b'[{"x":1}]')
    first = persist_bronze(root=tmp_path, resource=resource, run_date="2026-06-29", result=result)
    second = persist_bronze(root=tmp_path, resource=resource, run_date="2026-06-29", result=result)
    assert first.bronze_path == second.bronze_path
    assert (tmp_path / first.bronze_path).exists()


def test_persist_bronze_records_exact_post_request_lineage(tmp_path) -> None:
    resource = DatasetResource(
        family="organos",
        dataset="OrganInventory",
        format="json",
        url="https://example.test/inventory",
        snapshot_token="Leg15-type3",
        post_data={"_opendata_legislatura": 15, "_opendata_tipoConsulta": 3},
    )
    manifest = persist_bronze(
        root=tmp_path,
        resource=resource,
        run_date="2026-08-02",
        result=FetchResult(
            url=resource.url,
            status_code=200,
            headers={},
            content=b'{"datosOrganos":[]}',
        ),
    )

    assert manifest.request_method == "POST"
    assert manifest.request_parameters_json == (
        '{"_opendata_legislatura":"15","_opendata_tipoConsulta":"3"}'
    )
    assert len(manifest.request_parameters_sha256 or "") == 64
    assert (manifest.request_parameters_sha256 or "")[:12] in manifest.bronze_path


def test_pdf_resume_can_be_fast_while_deep_checksum_still_detects_tampering(
    tmp_path,
) -> None:
    resource = DatasetResource(
        family="documents",
        dataset="official-record",
        format="pdf",
        url="https://example.test/file.pdf",
        snapshot_token="document-1",
    )
    manifest = persist_bronze(
        root=tmp_path,
        resource=resource,
        run_date="2026-07-14",
        result=FetchResult(
            url=resource.url,
            status_code=200,
            headers={},
            content=b"%PDF-1.7\noriginal-payload",
        ),
    )
    path = tmp_path / manifest.bronze_path
    path.write_bytes(b"%PDF-1.7\ntampered-payload")

    assert path.stat().st_size == manifest.bytes
    assert bronze_payload_is_valid(root=tmp_path, manifest=manifest, verify_checksum=False)
    assert not bronze_payload_is_valid(root=tmp_path, manifest=manifest)


def test_html_contract_rejects_http_200_error_body() -> None:
    assert not content_matches_format_contract(
        content=(b"<html><body><h1>Error 404</h1><p>Pagina no encontrada</p></body></html>"),
        format_name="html",
    )


def test_html_contract_accepts_substantive_document() -> None:
    assert content_matches_format_contract(
        content=(
            b"<!doctype html><html><body><div class='textoIntegro'>"
            + b"Texto parlamentario oficial suficientemente largo. " * 4
            + b"</div></body></html>"
        ),
        format_name="html",
    )


def test_xml_contract_rejects_html_and_accepts_well_formed_xml() -> None:
    assert not content_matches_format_contract(
        content=b"<html><body>upstream error</body></html>",
        format_name="xml",
    )
    assert content_matches_format_contract(
        content=b'<?xml version="1.0"?><Resultado><Totales/></Resultado>',
        format_name="xml",
    )
    assert not content_matches_format_contract(
        content=b'<?xml version="1.0"?><Error>temporarily unavailable</Error>',
        format_name="xml",
    )


def test_csv_contract_rejects_error_pages_and_accepts_rectangular_rows() -> None:
    assert not content_matches_format_contract(
        content=b"<html><body>temporary,error</body></html>",
        format_name="csv",
    )
    assert not content_matches_format_contract(
        content=b"error,message\nservice,unavailable,unexpected\n",
        format_name="csv",
    )
    assert content_matches_format_contract(
        content="id;nombre\n1;Garc\u00eda\n".encode(),
        format_name="csv",
    )


def test_zip_contract_rejects_error_page_and_path_traversal() -> None:
    assert not content_matches_format_contract(
        content=b"<html><body>upstream error</body></html>",
        format_name="zip",
    )
    unsafe = io.BytesIO()
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.json", "{}")
    assert not content_matches_format_contract(
        content=unsafe.getvalue(),
        format_name="zip",
    )


def test_zip_contract_accepts_non_empty_safe_archive() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("Votacion001/VOT.json", "{}")
    assert content_matches_format_contract(
        content=payload.getvalue(),
        format_name="zip",
    )


def test_png_contract_verifies_the_complete_image() -> None:
    payload = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(payload, format="PNG")

    assert content_matches_format_contract(
        content=payload.getvalue(),
        format_name="png",
    )
    assert not content_matches_format_contract(
        content=payload.getvalue()[:16],
        format_name="png",
    )

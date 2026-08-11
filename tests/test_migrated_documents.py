import json
import sys
from types import SimpleNamespace

import congreso_open_data.documents as documents
from congreso_open_data.catalog import DatasetResource
from congreso_open_data.documents import clean_repeated_page_margins, pdf_document_text_row
from congreso_open_data.extractors.documents import discover_document_resources_from_manifest
from congreso_open_data.http import FetchResult
from congreso_open_data.interventions import speech_block_rows
from congreso_open_data.storage import persist_bronze

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
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000241 00000 n \n"
    b"0000000311 00000 n \n"
    b"trailer << /Root 1 0 R /Size 6 >>\n"
    b"startxref\n"
    b"405\n"
    b"%%EOF\n"
)


def _ocr_item(x1, y1, x2, y2, text, confidence=0.95):
    return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, confidence]


def test_rapidocr_reads_detected_column_crops_left_then_right(monkeypatch) -> None:
    monkeypatch.setattr(documents, "_ocr_image_inputs", lambda path: ["left", "right"])

    calls = []

    def engine(image, **kwargs):
        calls.append((image, kwargs))
        if image == "left":
            return [
                _ocr_item(0, 20, 40, 30, "LEFT SECOND"),
                _ocr_item(0, 0, 40, 10, "LEFT FIRST"),
            ], None
        return [
            _ocr_item(0, 20, 40, 30, "RIGHT SECOND"),
            _ocr_item(0, 0, 40, 10, "RIGHT FIRST"),
        ], None

    text, confidence = documents.rapidocr_image_text_with_engine("unused", engine)

    assert text.splitlines() == [
        "LEFT FIRST",
        "LEFT SECOND",
        "RIGHT FIRST",
        "RIGHT SECOND",
    ]
    assert confidence == 0.95
    assert calls == [
        ("left", {"use_cls": False}),
        ("right", {"use_cls": False}),
    ]


def test_rapidocr_reads_mixed_layout_header_before_columns(monkeypatch) -> None:
    monkeypatch.setattr(
        documents,
        "_ocr_image_inputs",
        lambda path: ["header", "left", "right"],
    )

    def engine(image, **kwargs):
        return [_ocr_item(0, 0, 40, 10, image.upper())], None

    text, confidence = documents.rapidocr_image_text_with_engine("unused", engine)

    assert text.splitlines() == ["HEADER", "LEFT", "RIGHT"]
    assert abs(confidence - 0.95) < 1e-12


def test_ocr_reading_order_keeps_single_column_vertical_order() -> None:
    lines = [
        _ocr_item(100, 40, 900, 50, "THIRD"),
        _ocr_item(100, 0, 900, 10, "FIRST"),
        _ocr_item(100, 20, 900, 30, "SECOND"),
        _ocr_item(100, 60, 900, 70, "FOURTH"),
    ]

    ordered = documents._ocr_lines_in_reading_order(lines)

    assert [line[1] for line in ordered] == ["FIRST", "SECOND", "THIRD", "FOURTH"]


def test_ocr_column_split_detects_rule_and_rejects_single_column_lines() -> None:
    import numpy as np

    ruled = np.full((400, 300), 255, dtype=np.uint8)
    ruled[40:370, 150:152] = 0
    assert documents._two_column_split(ruled) in {150, 151}

    single_column = np.full((400, 300), 255, dtype=np.uint8)
    for y in range(40, 360, 25):
        single_column[y : y + 3, 30:270] = 0
    assert documents._two_column_split(single_column) is None


def test_ocr_column_split_detects_skewed_rule_among_dense_text_lines() -> None:
    """Regression for L2-PL_101 page 6 and L4-CO_131 ruled scans."""

    import numpy as np

    page = np.full((1_576, 1_155), 255, dtype=np.uint8)
    for y in range(135, 1_500, 35):
        page[y : y + 12, 95:595] = 0
        page[y : y + 12, 615:1_070] = 0
    for y in range(135, 1_500):
        x = 595 + (y - 135) * 14 // (1_500 - 135)
        page[y, x : x + 2] = 0

    split = documents._two_column_split(page)

    assert split is not None
    assert 590 <= split <= 615


def test_ocr_mixed_layout_preserves_header_before_bottom_columns() -> None:
    """Regression for L4-CO_131 page 1's masthead plus short column section."""

    import numpy as np

    page = np.full((800, 600), 255, dtype=np.uint8)
    for y in range(80, 360, 45):
        page[y : y + 10, 70:530] = 0
    for y in range(620, 770, 28):
        page[y : y + 9, 45:292] = 0
        page[y : y + 9, 310:555] = 0
    page[600:775, 300:302] = 0

    layout = documents._mixed_two_column_layout(page)

    assert layout is not None
    split, column_start = layout
    assert 295 <= split <= 305
    assert 580 <= column_start <= 610


def test_ocr_mixed_layout_rejects_large_centered_header_glyph() -> None:
    import numpy as np

    page = np.full((800, 600), 255, dtype=np.uint8)
    page[440:475, 290:310] = 0
    for y in range(520, 760, 30):
        page[y : y + 8, 60:540] = 0

    assert documents._mixed_two_column_layout(page) is None


def test_discover_document_resources_from_profile_manifest(tmp_path) -> None:
    resource = DatasetResource(
        family="diputados",
        dataset="DeputyProfile",
        format="html",
        url=(
            "https://www.congreso.es/es/busqueda-de-diputados?"
            "_diputadomodule_mostrarFicha=true&codParlamentario=35&idLegislatura=XV"
        ),
        snapshot_token="35",
    )
    content = """
    <html><body>
      <p>Diputados</p><p>Zaragoza Alonso, José</p>
      <a href="https://www.congreso.es/docbienes/leg15/a.pdf">
        Declaración de Bienes y Rentas
      </a>
    </body></html>
    """.encode()
    manifest = persist_bronze(
        root=tmp_path,
        resource=resource,
        run_date="2026-06-29",
        result=FetchResult(url=resource.url, status_code=200, headers={}, content=content),
    )
    resources = discover_document_resources_from_manifest(lake_root=tmp_path, manifest=manifest)
    assert len(resources) == 1
    assert resources[0].family == "documents"
    assert resources[0].dataset == "PdfDocument"
    assert resources[0].format == "pdf"
    assert resources[0].snapshot_token


def test_pdf_document_text_row_extracts_text() -> None:
    row = pdf_document_text_row(
        MINIMAL_TEXT_PDF,
        source_url="https://www.congreso.es/docbienes/leg15/a.pdf",
        source_sha256="abc",
        snapshot_date="2026-06-29",
    )
    assert row["document_kind"] == "assets_income"
    assert row["page_count"] == 1
    assert row["text"] == "Declaracion test"
    assert row["extraction_method"] == "pypdf_text"
    assert row["model_name"] == "pypdf"
    assert row["extraction_status"] == "ok"
    assert row["_page_methods"] == ["pypdf_text"]
    assert row["_page_confidences"] == [0.92]
    geometry = json.loads(row["_page_diagnostics"][0])["pymupdf_geometry"]
    assert geometry["reading_order"] == "pymupdf_blocks_sort_true"
    assert geometry["blocks"][0]["bbox"]
    assert len(geometry["blocks"][0]["text_sha256"]) == 64

    strict_row = pdf_document_text_row(
        MINIMAL_TEXT_PDF,
        source_url="https://www.congreso.es/public_oficiales/L15/DSCD-15-PL-1.PDF",
        source_sha256="strict",
        snapshot_date="2026-08-01",
        strict_validation=True,
    )
    assert strict_row["extraction_status"] == "ok"


def test_pdf_document_text_row_marks_invalid_pdf_as_error() -> None:
    row = pdf_document_text_row(
        json.dumps({"not": "a pdf"}).encode(),
        source_url="https://www.congreso.es/docacteco/leg15/a.pdf",
        source_sha256="abc",
        snapshot_date="2026-06-29",
    )
    assert row["document_kind"] == "economic_interests"
    assert row["extraction_method"] == "pypdf_text"
    assert row["model_name"] == "pypdf"
    assert row["extraction_status"] == "error"
    assert row["extraction_error"]


def test_pdf_text_extraction_falls_back_to_pymupdf_with_provenance(monkeypatch) -> None:
    class FakePage:
        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return "Texto recuperado"

    class FakeDocument:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def __iter__(self):
            return iter([FakePage()])

    def fail_pypdf(*args, **kwargs):
        raise ValueError("malformed content stream")

    monkeypatch.setattr("pypdf.PdfReader", fail_pypdf)
    monkeypatch.setattr("pymupdf.open", lambda **kwargs: FakeDocument())

    pages, method, warning = documents._extract_pdf_page_texts_with_method(b"%PDF")

    assert pages == ["Texto recuperado"]
    assert method == "pymupdf_text"
    assert warning == "pypdf recovery: ValueError: malformed content stream"

    strict = documents._extract_pdf_text(b"%PDF", strict_validation=True)
    assert strict.status == "needs_review"


def test_rapidocr_engine_receives_gpu_runtime_providers(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, providers=None, use_cuda=False) -> None:
            created["providers"] = providers
            created["use_cuda"] = use_cuda

    monkeypatch.setattr(documents, "_RAPID_OCR_ENGINE", None)
    monkeypatch.setenv("CONGRESO_REQUIRE_GPU", "true")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
            ]
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(RapidOCR=FakeRapidOCR),
    )

    documents.rapidocr_engine()

    assert created["providers"] == ["CUDAExecutionProvider"]
    assert created["use_cuda"] is True


def test_rapidocr_engine_maps_gpu_runtime_to_current_params(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, config_path=None, params=None) -> None:
            created["config_path"] = config_path
            created["params"] = params

    monkeypatch.setattr(documents, "_RAPID_OCR_ENGINE", None)
    monkeypatch.setenv("CONGRESO_REQUIRE_GPU", "true")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"]),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapidocr",
        SimpleNamespace(RapidOCR=FakeRapidOCR),
    )

    documents.rapidocr_engine()

    assert created["params"] == {
        "EngineConfig.onnxruntime.use_cuda": True,
        "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False,
        "EngineConfig.onnxruntime.use_coreml": False,
    }


def test_amount_parser_handles_mixed_thousands_and_decimal_separators() -> None:
    assert documents._structured_amount_to_float("113,423.11") == 113423.11
    assert documents._structured_amount_to_float("1.234,56 EUR") == 1234.56
    assert documents._structured_amount_to_float("1.380,7 EUR") == 1380.7
    assert documents._amount_to_float("importe: 300.000,00 EUR") == 300000.0
    assert documents._structured_income_category("Dietas Grupo Parlamentario") == "public_salary"


def test_generic_extractor_does_not_duplicate_structured_mortgage_rows() -> None:
    pages = [
        {
            "page_number": 1,
            "text": (
                "BIENES PATRIMONIALES DEL PARLAMENTARIO\n"
                "DEUDAS Y OBLIGACIONES PATRIMONIALES\n"
                "PRESTAMO HIPOTECARIO 113,423.11 26/03/2014 115.000"
            ),
        }
    ]
    rows = documents._extraction_rows(
        pages=pages,
        source_url="https://www.congreso.es/docbienes/leg15/example.pdf",
        source_sha256="mortgage-test",
        snapshot_date="2026-06-30",
        document_kind="assets_income",
    )
    assert len([row for row in rows if row["item_category"] == "mortgage"]) == 1


def test_repeated_pdf_margins_are_removed_with_exact_audit_spans() -> None:
    pages = [
        f"DIARIO DE SESIONES\nDiscurso diferente {page}\nCongreso de los Diputados"
        for page in range(1, 5)
    ]

    cleaned, removals = clean_repeated_page_margins(pages)

    assert cleaned == [f"\nDiscurso diferente {page}\n" for page in range(1, 5)]
    assert len(removals) == 8
    parsed = [json.loads(removal) for removal in removals]
    assert {item["kind"] for item in parsed} == {
        "repeated_header",
        "repeated_footer",
    }
    assert {item["page_number"] for item in parsed} == {1, 2, 3, 4}
    assert {item["offset_basis"] for item in parsed} == {"selected_page_text_unicode_codepoints"}
    assert all(item["source_page_length"] == len(pages[item["page_number"] - 1]) for item in parsed)


def test_dynamic_page_header_and_geometric_legal_footer_are_removed() -> None:
    pages = [
        (
            "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS\n"
            "COMISIONES\n"
            f"Núm. 606 23 de julio de 2026 Pág. {page}\n"
            f"Texto parlamentario de la página {page}."
            + (
                "\ncve: DSCD-15-CO-606\n"
                "http://www.congreso.es  Calle Floridablanca, s/n. 28071 Madrid\n"
                "D. L.: M-12.580/1961 CONGRESO DE LOS DIPUTADOS Teléf.: 91 390 60 00\n"
                "Edición electrónica preparada por la Agencia Estatal "
                "Boletín Oficial del Estado – http://boe.es"
                if page == 4
                else "\ncve: DSCD-15-CO-606"
            )
        )
        for page in range(1, 5)
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 842.0,
                    "page_width": 595.0,
                    "blocks": (
                        [
                            {
                                "bbox": [38.0, 20.0, 556.0, 85.0],
                                "lines": [
                                    "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS",
                                    "COMISIONES",
                                    f"Núm. 606 23 de julio de 2026 Pág. {page}",
                                ],
                            },
                            {"bbox": [38.0, 791.0, 556.0, 823.0]},
                            {"bbox": [561.0, 708.0, 572.0, 780.0]},
                        ]
                        if page == 4
                        else [
                            {
                                "bbox": [38.0, 20.0, 556.0, 85.0],
                                "lines": [
                                    "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS",
                                    "COMISIONES",
                                    f"Núm. 606 23 de julio de 2026 Pág. {page}",
                                ],
                            }
                        ]
                    ),
                }
            }
        )
        for page in range(1, 5)
    ]

    cleaned, removals = clean_repeated_page_margins(pages, diagnostics)

    assert all("Pág." not in page for page in cleaned)
    assert all("DIARIO DE SESIONES" not in page for page in cleaned)
    assert "Calle Floridablanca" not in cleaned[3]
    assert "Edición electrónica" not in cleaned[3]
    assert all("Texto parlamentario" in page for page in cleaned)
    kinds = {json.loads(item)["kind"] for item in removals}
    assert "repeated_header" in kinds
    assert "geometric_footer" in kinds
    assert "geometric_side_margin" in kinds
    for raw in removals:
        removal = json.loads(raw)
        source_page = pages[removal["page_number"] - 1]
        assert removal["text"] == source_page[removal["source_start"] : removal["source_end"]]


def test_proper_unicode_issue_header_is_removed_from_content_projection() -> None:
    pages = [
        (
            "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS\n"
            "COMISIONES\n"
            f"N\u00fam. 16 19 de noviembre de 2024 P\u00e1g. {page}\n"
            f"Texto oficial {page}"
        )
        for page in range(1, 4)
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 842.0,
                    "page_width": 595.0,
                    "blocks": [
                        {
                            "bbox": [38.0, 20.0, 556.0, 85.0],
                            "lines": [
                                "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS",
                                "COMISIONES",
                                f"N\u00fam. 16 19 de noviembre de 2024 P\u00e1g. {page}",
                            ],
                        }
                    ],
                }
            }
        )
        for page in range(1, 4)
    ]

    cleaned, removals = clean_repeated_page_margins(pages, diagnostics)

    assert all("N\u00fam." not in page for page in cleaned)
    assert all("P\u00e1g." not in page for page in cleaned)
    assert all("Texto oficial" in page for page in cleaned)
    assert len(removals) == 9


def test_flattened_running_header_matches_separate_geometry_lines() -> None:
    pages = [
        (
            "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS\n"
            "COMISIONES\n"
            f"Núm. 16 19 de noviembre de 2024\u2002 Pág.\u2002{page}\n"
            f"El señor ORADOR: Texto parlamentario {page}.\n"
            "cve: DSCD-15-CI-16"
        )
        for page in range(1, 5)
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 842.0,
                    "page_width": 595.0,
                    "blocks": [
                        {
                            "bbox": [38.0, 20.0, 556.0, 85.0],
                            "lines": [
                                "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS",
                                "COMISIONES",
                                "Núm. 16",
                                "19 de noviembre de 2024",
                                f"Pág. {page}",
                            ],
                        },
                        {
                            "bbox": [561.0, 708.0, 572.0, 780.0],
                            "lines": ["cve: DSCD-15-CI-16"],
                        },
                    ],
                }
            }
        )
        for page in range(1, 5)
    ]

    cleaned, removals = clean_repeated_page_margins(pages, diagnostics)

    assert all("Núm. 16" not in page for page in cleaned)
    assert all("Pág." not in page for page in cleaned)
    assert all("Texto parlamentario" in page for page in cleaned)
    assert sum(json.loads(item)["kind"] == "repeated_header" for item in removals) == 12
    for raw in removals:
        removal = json.loads(raw)
        source_page = pages[removal["page_number"] - 1]
        assert removal["text"] == source_page[removal["source_start"] : removal["source_end"]]


def test_five_line_issue_header_is_removed_from_short_contents_page() -> None:
    proper_page = "Pág. 2"
    mojibake_page = "PÃ¡g. 2"
    assert documents._margin_line_signature(proper_page) == documents._margin_line_signature(
        mojibake_page
    ), (
        [hex(ord(char)) for char in proper_page],
        [hex(ord(char)) for char in mojibake_page],
    )
    pages = [
        (
            "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS\n"
            "PLENO Y DIPUTACIÃ“N PERMANENTE\n"
            "NÃºm. 181\t\n"
            "28\u202fde\u202fabril\u202fde\u202f2026\t\n"
            f"PÃ¡g. {page}\n"
            f"Toma en consideraciÃ³n. (VotaciÃ³n) {page}\n"
            "cve: DSCD-15-PL-181"
        )
        for page in range(2, 5)
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 841.89,
                    "page_width": 595.276,
                    "blocks": [
                        {
                            "bbox": [21.924, 20.061, 573.348, 47.269],
                            "lines": ["DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS"],
                        },
                        {
                            "bbox": [144.148, 49.147, 451.119, 73.491],
                            "lines": ["PLENO Y DIPUTACIÃ“N PERMANENTE"],
                        },
                        {
                            "bbox": [68.032, 78.369, 527.248, 92.689],
                            "lines": [
                                "NÃºm. 181",
                                "28 de abril de 2026",
                                f"PÃ¡g. {page}",
                            ],
                        },
                        {
                            "bbox": [561.713, 709.899, 571.737, 779.528],
                            "lines": ["cve: DSCD-15-PL-181"],
                        },
                    ],
                }
            }
        )
        for page in range(2, 5)
    ]

    cleaned, removals = clean_repeated_page_margins(pages, diagnostics)

    assert all("NÃºm. 181" not in page for page in cleaned)
    assert all("28\u202fde\u202fabril" not in page for page in cleaned)
    assert all("PÃ¡g." not in page for page in cleaned)
    assert all("Toma en consideraciÃ³n" in page for page in cleaned)
    assert sum(json.loads(item)["kind"] == "repeated_header" for item in removals) == 15


def test_short_document_headers_are_removed_from_all_content_pages() -> None:
    pages = [
        "PORTADA\nDocumento de constitución de la comisión.",
        *[
            (
                "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS\n"
                "COMISIONES\n"
                f"Núm. 4 2 de abril de 2024 Pág. {page}\n"
                f"El señor ORADOR: Texto parlamentario {page}."
            )
            for page in (2, 3)
        ],
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 842.0,
                    "page_width": 595.0,
                    "blocks": (
                        [{"bbox": [38.0, 100.0, 556.0, 180.0], "lines": ["PORTADA"]}]
                        if page == 1
                        else [
                            {
                                "bbox": [38.0, 20.0, 556.0, 85.0],
                                "lines": [
                                    "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS",
                                    "COMISIONES",
                                    "Núm. 4",
                                    "2 de abril de 2024",
                                    f"Pág. {page}",
                                ],
                            }
                        ]
                    ),
                }
            }
        )
        for page in (1, 2, 3)
    ]

    cleaned, removals = clean_repeated_page_margins(pages, diagnostics)

    assert cleaned[0] == pages[0]
    assert all("DIARIO DE SESIONES" not in page for page in cleaned[1:])
    assert all("COMISIONES" not in page for page in cleaned[1:])
    assert all("Núm. 4" not in page for page in cleaned[1:])
    assert all("Texto parlamentario" in page for page in cleaned[1:])
    assert len(removals) == 6


def test_single_content_page_uses_exact_geometric_header_anatomy() -> None:
    pages = [
        "PORTADA",
        (
            "CORTES GENERALES\n"
            "DIARIO DE SESIONES DEL\n"
            "CONGRESO DE LOS DIPUTADOS\n"
            "El señor ORADOR: Texto oficial."
        ),
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 842.0,
                    "page_width": 595.0,
                    "blocks": (
                        [{"bbox": [38.0, 100.0, 556.0, 180.0], "lines": ["PORTADA"]}]
                        if page == 1
                        else [
                            {
                                "bbox": [38.0, 20.0, 556.0, 85.0],
                                "lines": [
                                    "CORTES GENERALES",
                                    "DIARIO DE SESIONES DEL",
                                    "CONGRESO DE LOS DIPUTADOS",
                                ],
                            }
                        ]
                    ),
                }
            }
        )
        for page in (1, 2)
    ]

    cleaned, removals = clean_repeated_page_margins(pages, diagnostics)

    assert cleaned[0] == "PORTADA"
    assert cleaned[1].strip() == "El señor ORADOR: Texto oficial."
    assert len(removals) == 3
    assert all(json.loads(item)["kind"] == "geometric_header" for item in removals)


def test_cleaned_page_boundary_recovers_split_parenthetical_interruption() -> None:
    headers = [
        "DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS",
        "COMISIONES",
    ]
    pages = [
        "PORTADA",
        "\n".join(
            [
                *headers,
                "Núm. 6 30 de abril de 2024 Pág. 2",
                "El señor PRESIDENTE: Continúo. (La señora Vázquez Blanco: "
                "Nos ha faltado al respeto. Nos ha llamado",
            ]
        ),
        "\n".join(
            [
                *headers,
                "Núm. 6 30 de abril de 2024 Pág. 3",
                "coro). Le pido que no intervenga.",
                "El señor BENDODO BENASAYAG: Le pido que retire sus palabras.",
            ]
        ),
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 842.0,
                    "page_width": 595.0,
                    "blocks": (
                        [{"bbox": [38.0, 100.0, 556.0, 180.0], "lines": ["PORTADA"]}]
                        if page == 1
                        else [
                            {
                                "bbox": [38.0, 20.0, 556.0, 85.0],
                                "lines": [
                                    *headers,
                                    "Núm. 6",
                                    "30 de abril de 2024",
                                    f"Pág. {page}",
                                ],
                            }
                        ]
                    ),
                }
            }
        )
        for page in (1, 2, 3)
    ]

    cleaned, _ = clean_repeated_page_margins(pages, diagnostics)
    rows = speech_block_rows(
        visible_text="",
        page_texts=cleaned,
        page_diagnostics=diagnostics,
        source_url="https://www.congreso.es/DSCD-15-CI-6.PDF",
        source_sha256="a" * 64,
        snapshot_date="2026-08-01",
        legislature="Leg.15",
        document_id="DSCD-15-CI-6",
        source_kind="pdf",
        extraction_method="pypdf_text",
    )
    interruption = next(row for row in rows if row["normalized_speaker"] == "vazquez blanco")

    assert interruption["content_text"] == ("Nos ha faltado al respeto. Nos ha llamado coro")
    assert interruption["page_hint"] == 2
    assert interruption["page_end"] == 3
    assert interruption["turn_kind"] == "parenthetical_interruption"


def test_repeated_discourse_line_is_not_removed_without_margin_geometry() -> None:
    pages = [
        f"DIARIO DE SESIONES\nIntervenci\u00f3n {page}.\nMuchas gracias." for page in range(1, 5)
    ]
    diagnostics = [
        json.dumps(
            {
                "pymupdf_geometry": {
                    "page_height": 842.0,
                    "page_width": 595.0,
                    "blocks": [
                        {
                            "bbox": [38.0, 20.0, 556.0, 50.0],
                            "lines": ["DIARIO DE SESIONES"],
                        },
                        {
                            "bbox": [38.0, 680.0, 556.0, 720.0],
                            "lines": ["Muchas gracias."],
                        },
                    ],
                }
            }
        )
        for _ in pages
    ]

    cleaned, _ = clean_repeated_page_margins(pages, diagnostics)

    assert all("DIARIO DE SESIONES" not in page for page in cleaned)
    assert all("Muchas gracias." in page for page in cleaned)

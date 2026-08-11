"""Regression tests owned by the public acquisition package."""

from congreso_open_data.normalization import normalize_record_keys
from congreso_open_data.transforms import (
    approved_law_rows_from_payload,
    deputy_row,
    document_asset_rows_from_records,
    historical_initiative_rows_from_list_payload,
    initiative_row,
    interest_row,
    intervention_row,
    split_links,
    vote_rows,
)


def test_split_links_from_multiline_field() -> None:
    assert split_links("https://a.test/x.pdf \n https://b.test/y.pdf") == [
        "https://a.test/x.pdf",
        "https://b.test/y.pdf",
    ]


def test_split_links_repairs_concatenated_official_origins_and_preserves_page() -> None:
    assert split_links(
        "https://www.congreso.es:443https://static.congreso.es/video.mp4 "
        "https://www.congreso.es:443/public/diario.PDF#page=7"
    ) == [
        "https://static.congreso.es/video.mp4",
        "https://www.congreso.es/public/diario.PDF#page=7",
    ]


def test_document_assets_accept_filtered_intervention_file_number_alias() -> None:
    rows = document_asset_rows_from_records(
        [
            {
                "numero_expediente": "180/001050/0000",
                "enlace_pdf": "https://www.congreso.es/diario.PDF#page=6",
            }
        ],
        family="intervenciones",
        dataset="IntervencionesCronologicamente",
        snapshot_date="2026-08-10",
    )

    assert rows[0]["entity_id"] == "180/001050/0000"
    assert rows[0]["page_hint"] == 6


def test_vote_rows_preserve_primary_and_joint_targets_without_inventing_links() -> None:
    event, nominal, vote_items = vote_rows(
        {
            "informacion": {
                "fecha": "15/07/2026",
                "sesion": "12",
                "numeroVotacion": "3",
                "titulo": "Votacion conjunta",
                "textoExpediente": "Proyecto que modifica la Ley 35/2006.",
                "tituloSubGrupo": "",
                "votacionesConjuntas": [
                    {"textoExpediente": "Otra iniciativa sobre la Ley 5/2004."},
                    {"textoExpediente": "Otra iniciativa sobre la Ley 5/2004."},
                ],
            },
            "totales": {
                "presentes": 1,
                "afavor": 1,
                "enContra": 0,
                "abstenciones": 0,
                "noVotan": 0,
            },
            "votaciones": [
                {
                    "asiento": "1",
                    "diputado": "Diputada, Ana",
                    "grupo": "GS",
                    "voto": "Si",
                }
            ],
        },
        source_sha256="sha",
        snapshot_date="2026-07-15",
        legislature="Leg15",
        source_url="https://www.congreso.es/vote.json",
    )

    assert len(nominal) == 1
    assert event["file_text"] == "Proyecto que modifica la Ley 35/2006."
    assert event["initiative_file_number"] is None
    assert len(vote_items) == 3
    assert [row["item_order"] for row in vote_items] == [1, 2, 3]
    assert [row["source_item_kind"] for row in vote_items] == [
        "primary",
        "joint",
        "joint",
    ]
    assert len({row["vote_item_id"] for row in vote_items}) == 3
    assert all(row["initiative_file_number"] is None for row in vote_items)


def test_vote_rows_marks_empty_official_nominal_array_as_totals_only() -> None:
    event, nominal, vote_items = vote_rows(
        {
            "informacion": {
                "fecha": "14/03/2013",
                "sesion": 91,
                "numeroVotacion": 38,
                "titulo": "Dictamen",
                "textoExpediente": "Suplicatorio",
                "tituloSubGrupo": "",
                "votacionesConjuntas": [],
            },
            "totales": {
                "asentimiento": "No",
                "presentes": 289,
                "afavor": 278,
                "enContra": 2,
                "abstenciones": 9,
                "noVotan": 61,
            },
            "votaciones": [],
        },
        source_sha256="sha",
        snapshot_date="2026-07-15",
        legislature="Leg10",
        source_url="https://www.congreso.es/vote.json",
    )

    assert nominal == []
    assert len(vote_items) == 1
    assert event["source_mode"] == "official_json_totals_only"
    assert event["decision_method"] == "recorded_vote"
    assert event["nominal_data_available"] is False
    assert event["nominal_group_data_available"] is False
    assert event["nominal_seat_data_available"] is False
    assert all(
        row["initiative_link_status"] == "unresolved_no_structured_reference" for row in vote_items
    )


def test_vote_rows_preserves_official_assent_as_decision_method() -> None:
    event, nominal, _ = vote_rows(
        {
            "informacion": {
                "fecha": "13/09/2012",
                "sesion": 53,
                "numeroVotacion": 33,
                "titulo": "Propuesta",
                "textoExpediente": "Solicitud de subcomisiÃ³n",
                "tituloSubGrupo": "",
                "votacionesConjuntas": [],
            },
            "totales": {
                "asentimiento": "Si",
                "presentes": 0,
                "afavor": 0,
                "enContra": 0,
                "abstenciones": 0,
                "noVotan": 0,
            },
            "votaciones": [],
        },
        source_sha256="sha",
        snapshot_date="2026-07-15",
        legislature="Leg10",
        source_url="https://www.congreso.es/assent.json",
    )

    assert nominal == []
    assert event["source_mode"] == "official_json_assent"
    assert event["decision_method"] == "assent"
    assert event["nominal_data_available"] is False


def test_deputy_row_derives_ids() -> None:
    row = {
        "NOMBRE": "Abades Martínez, Cristina",
        "CIRCUNSCRIPCION": "Lugo",
        "FORMACIONELECTORAL": "PP",
        "FECHAALTA": "08/08/2023",
        "GENERO": "2",
    }
    out = deputy_row(row, source_sha256="abc", snapshot_date="2026-06-29")
    assert out["person_id"]
    assert out["constituency"] == "Lugo"
    assert out["gender"] == 2
    assert str(out["term_start_date"]) == "2023-08-08"


def test_normalize_record_keys_handles_accents_spaces_and_aliases() -> None:
    out = normalize_record_keys({"Fecha de nacimiento": "04/05/1970", "FORMACIÓN ELECTORAL": "PP"})
    assert out["fecha_de_nacimiento"] == "04/05/1970"
    assert out["fechanacimiento"] == "04/05/1970"
    assert out["formacionelectoral"] == "PP"


def test_deputy_row_combines_split_name_fields() -> None:
    row = {
        "NOMBRE": "Cristina",
        "APELLIDOS": "Abades Martínez",
        "FECHAALTA": "08/08/2023",
    }
    out = deputy_row(row, source_sha256="abc", snapshot_date="2026-06-29")
    assert out["full_name"] == "Abades Martínez, Cristina"
    assert out["first_name"] == "Cristina"
    assert out["last_names"] == "Abades Martínez"


def test_deputy_row_identity_distinguishes_group_segments() -> None:
    base = {
        "NOMBRE": "Acevedo Bisshopp, Manuel",
        "CIRCUNSCRIPCION": "Santa Cruz de Tenerife",
        "FECHAALTA": "11/07/1977",
        "FECHABAJA": "02/01/1979",
    }
    first = deputy_row(
        {
            **base,
            "GRUPOPARLAMENTARIO": "Grupo Parlamentario de Union de Centro Democratico",
            "FECHAALTAENGRUPOPARLAMENTARIO": "26/07/1977",
            "FECHABAJAENGRUPOPARLAMENTARIO": "23/09/1978",
        },
        source_sha256="abc",
        snapshot_date="2026-07-04",
        default_legislature="Leg.0",
    )
    second = deputy_row(
        {
            **base,
            "GRUPOPARLAMENTARIO": "Grupo Parlamentario Mixto",
            "FECHAALTAENGRUPOPARLAMENTARIO": "23/09/1978",
            "FECHABAJAENGRUPOPARLAMENTARIO": "02/01/1979",
        },
        source_sha256="abc",
        snapshot_date="2026-07-04",
        default_legislature="Leg.0",
    )
    assert first["deputy_term_id"] != second["deputy_term_id"]


def test_deputy_row_derives_age_when_birth_date_is_available() -> None:
    row = {
        "NOMBRE": "Vedrina Conesa, Maria Elisa",
        "FECHANACIMIENTO": "04/05/1970",
        "URLFICHADIPUTADO": "https://www.congreso.es/ficha/123",
    }
    out = deputy_row(row, source_sha256="abc", snapshot_date="2026-06-29")
    assert str(out["birth_date"]) == "1970-05-04"
    assert out["age_at_snapshot"] == 56
    assert out["profile_url"] == "https://www.congreso.es/ficha/123"


def test_interest_row_maps_docacteco() -> None:
    row = {
        "NOMBRE": "Abades Martínez,Cristina",
        "FECHAREGISTRO": "08/08/2023",
        "DECLARACION": "Declaración inicial",
        "TIPO": "ACTIVIDAD",
    }
    out = interest_row(row, source_sha256="abc", snapshot_date="2026-06-29")
    assert out["item_type"] == "ACTIVIDAD"
    assert str(out["registered_at"]) == "2023-08-08"


def test_interest_row_distinguishes_foundation_recipient() -> None:
    base = {
        "NOMBRE": "Abades Martinez,Cristina",
        "FECHAREGISTRO": "08/08/2023",
        "DECLARACION": "Declaracion inicial",
        "TIPO": "FUNDACIONES",
        "DESCRIPCION": "APORTACION ANUAL",
    }
    first = interest_row(
        {**base, "DESTINATARIO": "CRUZ ROJA"},
        source_sha256="abc",
        snapshot_date="2026-06-29",
    )
    second = interest_row(
        {**base, "DESTINATARIO": "ASOCIACION CONTRA EL CANCER"},
        source_sha256="abc",
        snapshot_date="2026-06-29",
    )
    assert first["recipient"] == "CRUZ ROJA"
    assert first["interest_id"] != second["interest_id"]


def test_initiative_row_maps_approved_law_fields() -> None:
    row = {
        "TIPO": "Leyes",
        "NUMERO_LEY": "1",
        "TITULO_LEY": "Ley 1/2026, de prueba.",
        "NUMERO_BOLETIN": "87",
        "FECHA_BOLETIN": "09/04/2026",
        "FECHA_LEY": "08/04/2026",
        "PDF": "https://www.congreso.es/ley.pdf",
    }
    out = initiative_row(
        row,
        source_sha256="abc",
        snapshot_date="2026-07-04",
        source_dataset="IniciativasLegislativasAprobadas",
    )
    assert out["source_dataset"] == "IniciativasLegislativasAprobadas"
    assert out["legislature"] == "Leg.15"
    assert out["file_number"] == "1"
    assert out["law_number"] == "1"
    assert str(out["law_date"]) == "2026-04-08"
    assert out["pdf_links"] == ["https://www.congreso.es/ley.pdf"]


def test_initiative_row_maps_related_file_numbers() -> None:
    row = {
        "LEGISLATURA": "Leg.15",
        "TIPO": "Proyecto de ley",
        "OBJETO": "Proyecto de Ley de prueba.",
        "NUMEXPEDIENTE": "121/000001/0000",
        "INICIATIVASRELACIONADAS": "025/000031 059/000020/0000",
    }
    out = initiative_row(
        row,
        source_sha256="abc",
        snapshot_date="2026-07-04",
        source_dataset="ProyectosDeLey",
    )
    assert out["file_number"] == "121/000001/0000"
    assert out["related_file_numbers"] == ["025/000031/0000", "059/000020/0000"]


def test_historical_initiative_rows_from_list_payload_map_ajax_fields() -> None:
    payload = {
        "iniciativas_encontradas": "1",
        "lista_iniciativas": {
            "iniciativa1": {
                "legislatura": "XIV",
                "resultado_tram": "Caducado sin calificacion previa 16/06/2023",
                "titulo": "Proyecto de Ley de prueba.",
                "fecha_calificado": "01/06/2023",
                "fecha_presentado": "27/05/2023",
                "autores": {"autor01": {"nombre": "Gobierno"}},
                "id_iniciativa": "121/000155",
                "enlace": {"url": "/wc/servidorCGI", "cmd": "VERLST", "base": "IW14"},
            }
        },
    }

    rows = historical_initiative_rows_from_list_payload(payload, source_dataset="ProyectosDeLey")
    out = initiative_row(
        rows[0],
        source_sha256="abc",
        snapshot_date="2026-07-05",
        source_dataset="ProyectosDeLey",
    )

    assert out["legislature"] == "Leg.14"
    assert out["file_number"] == "121/000155/0000"
    assert out["author"] == "Gobierno"
    assert str(out["presented_at"]) == "2023-05-27"
    assert (
        out["processing_history"] == "https://www.congreso.es/wc/servidorCGI?cmd=VERLST&base=IW14"
    )


def test_structured_initiative_identity_ignores_mutable_title_and_source_dataset() -> None:
    original = initiative_row(
        {
            "LEGISLATURA": "Leg.14",
            "NUMEXPEDIENTE": "121/000155",
            "OBJETO": "Titulo original",
        },
        source_sha256="source-a",
        snapshot_date="2023-01-01",
        source_dataset="ProyectosDeLey",
    )
    corrected = initiative_row(
        {
            "LEGISLATURA": "Leg.14",
            "NUMEXPEDIENTE": "121/000155/0000",
            "OBJETO": "Titulo oficial corregido",
        },
        source_sha256="source-b",
        snapshot_date="2026-07-15",
        source_dataset="GeneralInitiatives",
    )

    assert original["initiative_id"] == corrected["initiative_id"]


def test_historical_initiative_rows_map_constituent_legislature_code() -> None:
    payload = {
        "lista_iniciativas": {
            "iniciativa1": {
                "legislatura": "C",
                "titulo": "Derecho de asilo.",
                "id_iniciativa": "122/000001",
            }
        }
    }

    rows = historical_initiative_rows_from_list_payload(
        payload,
        source_dataset="ProposicionesDeLey",
    )
    out = initiative_row(
        rows[0],
        source_sha256="abc",
        snapshot_date="2026-07-05",
        source_dataset="ProposicionesDeLey",
    )

    assert out["legislature"] == "Leg.0"


def test_general_initiative_rows_exclude_dedicated_sources_and_map_metadata() -> None:
    payload = {
        "lista_iniciativas": {
            "general": {
                "atis": "Funcion de control",
                "tipo": "Pregunta oral en Pleno",
                "legislatura": "XV",
                "titulo": "Pregunta de prueba.",
                "id_iniciativa": "180/000001",
                "autor": "Diputada Uno",
            },
            "dedicated": {
                "tipo": "Proyecto de ley",
                "legislatura": "XV",
                "titulo": "Proyecto duplicado.",
                "id_iniciativa": "121/000001",
            },
            "dedicated_proposition_family": {
                "tipo": "Proposición de ley",
                "legislatura": "XV",
                "titulo": "Proposición duplicada.",
                "id_iniciativa": "125/000001",
            },
        }
    }

    records = historical_initiative_rows_from_list_payload(
        payload,
        source_dataset="GeneralInitiatives",
    )
    rows = [
        initiative_row(
            record,
            source_sha256="abc",
            snapshot_date="2026-07-15",
            source_dataset="GeneralInitiatives",
        )
        for record in records
    ]

    assert len(rows) == 1
    assert rows[0]["file_number"] == "180/000001/0000"
    assert rows[0]["grouping"] == "Funcion de control"
    assert rows[0]["type"] == "Pregunta oral en Pleno"
    assert rows[0]["author"] == "Diputada Uno"


def test_approved_law_rows_from_payload_map_pdf_tramitacion_and_date() -> None:
    payload = {
        "data": {
            "L": [
                {
                    "numLey": "2",
                    "pdf": "l_002_2023.pdf",
                    "pdf2": "",
                    "descrTipoTexto": "Ley",
                    "tramitacion": (
                        "/busqueda-de-iniciativas?_iniciativas_legislatura=XIV"
                        "&_iniciativas_id=121/000123"
                    ),
                    "titulo": ("Ley 2/2023, de 20 de febrero, reguladora de la proteccion."),
                }
            ]
        }
    }

    rows = approved_law_rows_from_payload(payload, default_year="2023")
    out = initiative_row(
        rows[0],
        source_sha256="abc",
        snapshot_date="2026-07-05",
        source_dataset="IniciativasLegislativasAprobadas",
    )

    assert out["legislature"] == "Leg.14"
    assert out["file_number"] == "2"
    assert out["law_number"] == "2"
    assert str(out["law_date"]) == "2023-02-20"
    assert out["pdf_links"] == [
        "https://www.congreso.es/constitucion/ficheros/leyes_espa/l_002_2023.pdf"
    ]


def test_approved_law_rows_parse_titles_without_second_de_in_date() -> None:
    payload = {
        "data": {
            "LO": [
                {
                    "numLey": "2",
                    "pdf": "lo_002_2003.pdf",
                    "descrTipoTexto": "Ley",
                    "titulo": "Ley Organica 2/2003, de 14 marzo, complementaria.",
                }
            ]
        }
    }

    rows = approved_law_rows_from_payload(payload, default_year="2003")
    out = initiative_row(
        rows[0],
        source_sha256="abc",
        snapshot_date="2026-07-05",
        source_dataset="IniciativasLegislativasAprobadas",
    )

    assert out["legislature"] == "Leg.7"
    assert str(out["law_date"]) == "2003-03-14"


def test_intervention_row_identity_is_case_and_separator_insensitive() -> None:
    row = {
        "legislatura": "Leg.15",
        "num expediente": "130/000001",
        "sesión": "30/06/2026",
        "órgano": "Pleno",
        "orador": "García Pérez, Ana",
        "inicio intervención": "10:00",
        "fin intervención": "10:05",
        "enlace pdf": "https://www.congreso.es/DSCD-15-PL-1.PDF#page=15",
    }
    out = intervention_row(row, source_sha256="abc", snapshot_date="2026-06-30")
    assert out["intervention_id"]
    assert out["speaker_name"] == "García Pérez, Ana"
    assert out["page_hint"] == 15


def test_intervention_row_identity_distinguishes_same_time_different_subject() -> None:
    base = {
        "LEGISLATURA": "Leg.15",
        "SESION": "30/06/2026",
        "ORGANO": "Pleno",
        "ORADOR": "García Pérez, Ana",
        "INICIOINTERVENCION": "10:00",
        "FININTERVENCION": "10:05",
        "ENLACEPDF": "https://www.congreso.es/DSCD-15-PL-1.PDF#page=15",
    }
    first = intervention_row(
        {**base, "OBJETOINICIATIVA": "Cuestión fuera del orden del día"},
        source_sha256="abc",
        snapshot_date="2026-06-30",
    )
    second = intervention_row(
        {**base, "OBJETOINICIATIVA": "Ordenación del debate"},
        source_sha256="abc",
        snapshot_date="2026-06-30",
    )
    assert first["intervention_id"] != second["intervention_id"]


def test_intervention_row_maps_historical_export_fields() -> None:
    row = {
        "tipo": "Proposicion no de ley ante el Pleno.",
        "fase": "",
        "legislatura": "C",
        "hora_inicio": "",
        "enlace_emision": "",
        "autores": "Grupo Parlamentario Socialista del Congreso",
        "nombre_sesion": "Pleno",
        "objeto_iniciativa": "Derecho de asilo.",
        "fecha": "27/07/1977",
        "enlace_descarga": "https://static.congreso.es/audio.mp3",
        "orador": "Alvarez de Miranda y Torres, Fernando (GUCD)",
        "hora_fin": "",
        "enlace_pdf": "https://www.congreso.es/public_oficiales/L0/CONG/DS/C_1977_005.PDF",
        "numero_expediente": "162/000052/0000",
    }

    out = intervention_row(row, source_sha256="abc", snapshot_date="2026-07-05")

    assert out["legislature"] == "Leg.0"
    assert out["initiative_file_number"] == "162/000052/0000"
    assert out["initiative_reference_raw"] == "162/000052/0000"
    assert out["initiative_reference_qualifier"] is None
    assert str(out["session_date"]) == "1977-07-27"
    assert out["session_year"] == 1977
    assert out["body"] == "Pleno"
    assert out["intervention_type"] == "Proposicion no de ley ante el Pleno."
    assert out["direct_video_url"] == "https://static.congreso.es/audio.mp3"
    assert out["document_id"] == "C_1977_005"


def test_intervention_row_splits_official_qualified_initiative_reference() -> None:
    out = intervention_row(
        {
            "legislatura": "XIV",
            "fecha": "16/11/2022",
            "orador": "Bal Francés, Edmundo (GCs)",
            "numero_expediente": "121/000125/0000 43424",
        },
        source_sha256="abc",
        snapshot_date="2026-07-14",
    )

    assert out["initiative_file_number"] == "121/000125/0000"
    assert out["initiative_reference_raw"] == "121/000125/0000 43424"
    assert out["initiative_reference_qualifier"] == "43424"


def test_intervention_row_repairs_concatenated_and_repeated_video_urls() -> None:
    out = intervention_row(
        {
            "legislatura": "14",
            "fecha": "08/11/2022",
            "orador": "Ejemplo, Ana",
            "enlace_emision": ("https://app.congreso.es/v1/old https://app.congreso.es/v1/new"),
            "enlace_descarga": ("https://www.congreso.es:443https://static.congreso.es/video.mp4"),
        },
        source_sha256="abc",
        snapshot_date="2026-07-14",
    )

    assert out["video_url"] == "https://app.congreso.es/v1/new"
    assert out["direct_video_url"] == "https://static.congreso.es/video.mp4"


def test_intervention_row_canonicalizes_known_official_pdf_sentinel() -> None:
    out = intervention_row(
        {
            "legislatura": "C",
            "fecha": "27/12/1978",
            "orador": "Hernandez Gil, Antonio",
            "enlace_pdf": (
                "https://www.congreso.es:443/public_oficiales/L0/CONG/DS/C_1978_999.PDF"
            ),
        },
        source_sha256="abc",
        snapshot_date="2026-07-14",
    )

    assert out["document_id"] == "SC_000"
    assert out["pdf_url"].endswith("/SC_000.PDF")

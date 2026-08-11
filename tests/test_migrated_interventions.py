"""Regression tests owned by the public acquisition package."""

import json

from congreso_open_data.html import parse_visible_html
from congreso_open_data.identifiers import parse_initiative_reference
from congreso_open_data.interventions import (
    canonical_intervention_pdf_url,
    canonical_intervention_text_url,
    canonical_official_resource_url,
    extract_id_texto,
    has_embedded_speaker_heading,
    html_to_visible_text,
    intervention_document_id_from_urls,
    intervention_id,
    match_intervention_text,
    normalize_speaker_name,
    page_hint_from_url,
    speech_block_rows,
    split_speech_blocks,
)


def test_intervention_id_canonicalizes_initiative_file_number() -> None:
    base = {
        "LEGISLATURA": "XV",
        "SESION": "01/01/2026",
        "ORADOR": "Diputada Uno",
    }

    assert intervention_id(base | {"NUMEXPEDIENTE": "121/000001"}) == intervention_id(
        base | {"NUMEXPEDIENTE": "121/000001/0000"}
    )


def test_qualified_initiative_reference_links_parent_and_preserves_identity() -> None:
    reference = parse_initiative_reference("121/000125/0000 43424")

    assert reference.file_number == "121/000125/0000"
    assert reference.raw_value == "121/000125/0000 43424"
    assert reference.qualifier == "43424"

    base = {
        "LEGISLATURA": "XIV",
        "SESION": "16/11/2022",
        "ORADOR": "Diputada Uno",
    }
    assert intervention_id(base | {"NUMEXPEDIENTE": "121/000125/0000 43424"}) != intervention_id(
        base | {"NUMEXPEDIENTE": "121/000125/0000 43425"}
    )


def test_canonical_official_resource_url_uses_last_absolute_revision() -> None:
    assert (
        canonical_official_resource_url(
            "https://www.congreso.es:443https://static.congreso.es/video.mp4"
        )
        == "https://static.congreso.es/video.mp4"
    )
    assert (
        canonical_official_resource_url(
            "https://app.congreso.es/v1/old https://app.congreso.es/v1/new"
        )
        == "https://app.congreso.es/v1/new"
    )
    assert canonical_official_resource_url("/public/document.pdf#page=4") == (
        "https://www.congreso.es/public/document.pdf"
    )
    assert (
        canonical_official_resource_url("https://www.congreso.es:443/public/document.pdf")
        == "https://www.congreso.es/public/document.pdf"
    )


def test_historical_document_ids_are_scoped_when_filenames_repeat() -> None:
    assert (
        intervention_document_id_from_urls(
            full_text_url=None,
            pdf_url="https://www.congreso.es/public_oficiales/L3/CONG/DS/CO/CO_058.PDF",
        )
        == "L3-CO_058"
    )
    assert (
        intervention_document_id_from_urls(
            full_text_url=None,
            pdf_url="https://www.congreso.es/public_oficiales/L14/CONG/DS/CO/DSCD-14-CO-804.PDF",
        )
        == "DSCD-14-CO-804"
    )
    assert (
        intervention_document_id_from_urls(
            full_text_url=None,
            pdf_url="https://www.congreso.es/public_oficiales/L0/CONG/DS/C_1977_005.PDF",
        )
        == "C_1977_005"
    )


def test_extract_id_texto_from_url() -> None:
    url = (
        "https://www.congreso.es/busqueda-de-intervenciones?p_p_id=intervenciones"
        "&_intervenciones_id_texto=(DSCD-15-PL-28.CODI.)#(Pagina22)"
    )
    assert extract_id_texto(url) == "DSCD-15-PL-28.CODI."


def test_official_html_projection_maps_visible_text_to_dom_nodes() -> None:
    parsed = parse_visible_html(
        """
        <html><body><nav>Excluir</nav><div class="textoIntegro">
          <p>El se\u00f1or GARCIA: Texto oficial.</p>
          <p>Continuaci\u00f3n parlamentaria.</p>
        </div></body></html>
        """
    )
    projection = json.loads(parsed.source_projection_json or "[]")

    assert parsed.content_selector == ".textoIntegro"
    assert "Excluir" not in parsed.visible_text
    assert [
        parsed.visible_text[item["content_start"] : item["content_end"]] for item in projection
    ] == [
        "El se\u00f1or GARCIA: Texto oficial.",
        "Continuaci\u00f3n parlamentaria.",
    ]
    assert all("/p" in item["xpath"] for item in projection)


def test_intervention_document_identity_and_page_hint() -> None:
    url = (
        "https://www.congreso.es/busqueda-de-intervenciones?"
        "_intervenciones_id_texto=(DSCD-15-PL-1.CODI.)#(P%C3%A1gina15)"
    )
    pdf = "https://www.congreso.es/public_oficiales/L15/CONG/DS/PL/DSCD-15-PL-1.PDF#page=15"
    assert canonical_intervention_text_url(url).endswith("(DSCD-15-PL-1.CODI.)")
    assert intervention_document_id_from_urls(full_text_url=url, pdf_url=pdf) == (
        "DSCD-15-PL-1.CODI."
    )
    assert page_hint_from_url(url) == 15
    assert page_hint_from_url(pdf) == 15


def test_historical_pdf_sentinels_resolve_to_verified_official_documents() -> None:
    assert (
        canonical_intervention_pdf_url(
            "https://www.congreso.es:443/public_oficiales/L0/CONG/DS/C_1977_0..CCAL: 37 37.PDF"
        )
        == "https://www.congreso.es/public_oficiales/L0/CONG/DS/C_1977_022.PDF"
    )
    assert (
        canonical_intervention_pdf_url(
            "https://www.congreso.es:443/public_oficiales/L0/CONG/DS/C_1978_999.PDF"
        )
        == "https://www.congreso.es/public_oficiales/L0/CONG/DS/SC_000.PDF"
    )


def test_placeholder_document_urls_are_not_treated_as_documents() -> None:
    assert canonical_intervention_text_url("https://www.congreso.es") is None
    assert canonical_intervention_pdf_url("https://www.congreso.es/public_oficiales/L15/") is None


def test_split_and_match_speech_blocks() -> None:
    html = """
    <html><body>
      <p>La senora Garcia Perez: Muchas gracias, presidenta.</p>
      <p>Continua mi intervencion.</p>
      <p>El senor Lopez Ruiz: Gracias.</p>
    </body></html>
    """
    text = html_to_visible_text(html)
    blocks = split_speech_blocks(text)
    match = match_intervention_text(speaker="Garcia Perez, Ana (GS)", blocks=blocks)
    assert len(blocks) == 2
    assert match.confidence > 0.7
    assert "Continua" in (match.text_fragment or "")


def test_split_speech_blocks_accepts_spanish_accents() -> None:
    html = """
    <html><body>
      <p>La se\u00f1ora Garc\u00eda P\u00e9rez: Muchas gracias.</p>
      <p>P\u00e1gina 12</p>
      <p>El se\u00f1or L\u00f3pez Ruiz: Gracias.</p>
    </body></html>
    """
    blocks = split_speech_blocks(html_to_visible_text(html))
    assert len(blocks) == 2
    assert blocks[0].normalized_speaker == "garcia perez"
    assert blocks[0].page_hint == 11
    assert blocks[1].normalized_speaker == "lopez ruiz"
    assert blocks[1].page_hint == 12


def test_split_speech_blocks_supports_wrapped_formal_session_headings() -> None:
    text = """
    Una vez cesados los aplausos, el señor PRE-
    SIDENTE DE LAS CORTES (Hernández Gil),
    pronunció el siguiente discurso:
    Majestades: las Cortes les dan la bienvenida.
    A continuación, SU MAJESTAD EL REY
    BALDUINO DE B,ELGICA leyó en español el
    siguiente mensaje:
    Señoras y señores: agradezco la invitación.
    """

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert match_intervention_text(
        speaker="Hernández Gil, Antonio",
        blocks=blocks,
    ).text_fragment.startswith("Majestades")
    assert match_intervention_text(
        speaker="Balduino de Bélgica",
        blocks=blocks,
    ).text_fragment.startswith("Señoras")


def test_split_speech_blocks_supports_early_formal_narrative_speeches() -> None:
    text = """
    Acto seguido, el señor PRESIDENTE DEL
    CONGRESO DE LOS DIPUTADOS (Lavilla
    Alsina) ley6 el siguiente discurso:
    Majestades, nuestra Constitución proclama la Monarquía parlamentaria.
    A continuación, SU MAJESTAD EL REY ley6 el siguiente discurso:
    Señor Presidente, inicio esta Legislatura con esperanza.
    """

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[0].normalized_speaker == "presidente lavilla alsina"
    assert blocks[0].text.startswith("Majestades")
    assert blocks[1].normalized_speaker == "su majestad el rey"
    assert blocks[1].text.startswith("Señor Presidente")


def test_split_speech_blocks_preserves_normalizacion_and_senado_inside_lines() -> None:
    text = (
        "La señora PRESIDENTA: Queda rechazado.\n"
        "• PROPOSICIÓN DE LEY ORGÁNICA DE AMNISTÍA PARA LA NORMALIZACIÓN "
        "INSTITUCIONAL.\n"
        "PRESENTADA POR EL GRUPO PARLAMENTARIO SOCIALISTA EN EL SENADO."
    )

    block = split_speech_blocks(text)[0]

    assert "NORMALIZACIÓN" in block.text
    assert "NOR MALIZACIÓN" not in block.text
    assert "SENADO" in block.text
    assert "SEN ADO" not in block.text
    for projection in json.loads(block.source_map_json):
        source = block.raw_text[
            projection["source_start"] - block.source_char_start : projection["source_end"]
            - block.source_char_start
        ]
        content = block.text[projection["content_start"] : projection["content_end"]]
        if projection["kind"] == "line_separator_projection":
            assert content == "\n"
            assert source and not source.strip()
        elif projection["kind"] == "synthetic_line_separator":
            assert content == "\n"
            assert source == ""
        else:
            assert " ".join(source.split()) == content


def test_split_speech_blocks_removes_standalone_agenda_heading_reversibly() -> None:
    text = (
        "La seÃ±ora PRESIDENTA: Muchas gracias. (Aplausos).\n"
        "REAL DECRETO DE CONVOCATORIA DE ELECCIONES.\n"
        "— ELECCIÓN DE LOS SECRETARIOS.\n"
        "El seÃ±or SECRETARIO: Lectura."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[0].text == "Muchas gracias. (Aplausos)."
    assert "REAL DECRETO DE CONVOCATORIA DE ELECCIONES." in (blocks[0].raw_text or "")
    removed = json.loads(blocks[0].removed_spans_json)
    assert sum(span["kind"] == "agenda_heading" for span in removed) == 2
    assert blocks[1].text == "Lectura."


def test_agenda_metadata_lines_are_classified_only_in_structural_style() -> None:
    from congreso_open_data.interventions import _looks_like_agenda_heading

    assert _looks_like_agenda_heading("— PRESENTADA POR EL GRUPO PARLAMENTARIO SOCIALISTA.")
    assert _looks_like_agenda_heading("— RELATIVA AL PARQUE PÚBLICO DE VIVIENDA.")
    assert not _looks_like_agenda_heading("presentada por el Grupo Parlamentario Socialista.")


def test_split_speech_blocks_supports_direct_royal_heading() -> None:
    text = (
        "La señora PRESIDENTA: Majestad, las Cortes esperan sus palabras.\n"
        "SU MAJESTAD EL REY DON FELIPE VI: Gracias, presidenta.\n"
        "La señora PRESIDENTA: Se levanta la sesión."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert blocks[1].speaker_heading == "SU MAJESTAD EL REY DON FELIPE VI"
    assert blocks[1].normalized_speaker == "su majestad el rey don felipe vi"
    assert blocks[1].text == "Gracias, presidenta."


def test_split_speech_blocks_supports_princess_of_asturias_heading() -> None:
    text = (
        "La señora PRESIDENTA: Alteza, puede prestar juramento.\n"
        "SU ALTEZA REAL LA PRINCESA DE ASTURIAS, DOÑA LEONOR DE BORBÓN Y ORTIZ: "
        "Juro desempeñar fielmente mis funciones.\n"
        "La señora PRESIDENTA: Muchas gracias."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert blocks[1].speaker_heading == (
        "SU ALTEZA REAL LA PRINCESA DE ASTURIAS, DOÑA LEONOR DE BORBÓN Y ORTIZ"
    )
    assert blocks[1].normalized_speaker.endswith("leonor de borbon y ortiz")
    assert blocks[1].text == "Juro desempeñar fielmente mis funciones."


def test_split_speech_blocks_joins_observed_early_ocr_headings() -> None:
    text = """
    El ~ñot PRESIDENTE DEL CONGRESO DE LOS DI-
    PUTADOS: Se abre la sesión.
    81 scnor PRESIDENTE DE LA REPUBLICA ARGEN-
    TINA (Raúl R. Alfonsín): Excelentísimos parlamentarios.
    El scnor PRESIDENTE DEL CONGRESO DE LOS DI-
    PUTADOS: Se levanta la sesión.
    """

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert [block.normalized_speaker for block in blocks] == [
        "presidente del congreso de los diputados",
        "presidente raul r alfonsin",
        "presidente del congreso de los diputados",
    ]
    assert blocks[0].text == "Se abre la sesión."
    assert blocks[1].text == "Excelentísimos parlamentarios."
    assert blocks[2].text == "Se levanta la sesión."


def test_speaker_match_tolerates_one_ocr_error_in_two_token_surname() -> None:
    text = """
    Acto seguido, el señor Presidente de las
    Ciortes (Hmández Gil) leyó el siguiente dis-
    curso:
    Majestades, Alteza Real: comienza el discurso.
    """

    blocks = split_speech_blocks(text)
    match = match_intervention_text(speaker="Hernández Gil, Antonio", blocks=blocks)

    assert len(blocks) == 1
    assert match.text_fragment.startswith("Majestades")


def test_ocr_speaker_variants_terminate_the_previous_block() -> None:
    text = """
    El señor GALEOTE JIMENEZ: Retiramos la enmienda.
    E1 ceflor PRESIDENTE: Tiene la palabra el siguiente orador.
    (El Mor MARTIN VILLA (don Emilio): Pido la palabra.
    """

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert blocks[0].text == "Retiramos la enmienda."
    assert blocks[1].text == "Tiene la palabra el siguiente orador."
    assert blocks[2].text == "Pido la palabra."


def test_generic_ocr_honorific_boundary_requires_an_uppercase_name() -> None:
    text = """
    El señor FUENTES QUINTANA: Termina la explicación.
    El s&or PRESDENTE: Se suspende la sesión.
    La seÍícrita CALVET PUIG: Formula una pregunta.
    El sistema debe ser de mercado: abierto y competitivo.
    """

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert blocks[0].text == "Termina la explicación."
    assert blocks[1].text == "Se suspende la sesión."
    assert blocks[2].text.endswith("El sistema debe ser de mercado: abierto y competitivo.")


def test_procedural_sentence_with_ocr_colon_does_not_truncate_speech() -> None:
    text = """
    El señor PRESIDENTE: Se abre la sesión.
    El señor Ministro de Asuntos Exteriores pide la palabra y, de acuerdo con el
    artículo 60 del Reglamento, se la: concedemos. Tiene la palabra.
    El señor LOPEZ RUIZ: Muchas gracias.
    """

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert "artículo 60" in blocks[0].text
    assert blocks[1].normalized_speaker == "lopez ruiz"


def test_speech_block_rows_from_visible_text() -> None:
    html = """
    <html><body>
      <p>La senora Garcia Perez: Muchas gracias.</p>
      <p>Continua mi intervencion.</p>
      <p>El senor Lopez Ruiz: Gracias.</p>
    </body></html>
    """
    rows = speech_block_rows(
        visible_text=html_to_visible_text(html),
        source_url=(
            "https://www.congreso.es/busqueda-de-intervenciones?"
            "_intervenciones_id_texto=(DSCD-15-PL-1.CODI.)"
        ),
        source_sha256="abc",
        snapshot_date="2026-06-30",
        legislature="Leg.15",
    )
    assert len(rows) == 2
    assert rows[0]["document_id"] == "DSCD-15-PL-1.CODI."
    assert rows[0]["ordinal"] == 0
    assert rows[0]["normalized_speaker"]


def test_split_speech_blocks_preserves_parenthetical_speaker_and_page_hint() -> None:
    html = """
    <html><body>
      <p>La senora PRESIDENTA DE LA MESA DE EDAD (Narbona Ruiz): Comenzamos.</p>
      <p>Pagina 4</p>
      <p>La senora PRESIDENTA: Continuamos.</p>
    </body></html>
    """
    blocks = split_speech_blocks(html_to_visible_text(html))
    assert blocks[0].normalized_speaker == "presidenta narbona ruiz"
    assert blocks[0].page_hint == 3
    assert blocks[1].page_hint == 4


def test_split_speech_blocks_uses_parenthetical_for_generic_candidate_heading() -> None:
    html = """
    <html><body>
      <p>El senor CANDIDATO (Lopez Sanchez): Comparezco.</p>
    </body></html>
    """
    blocks = split_speech_blocks(html_to_visible_text(html))
    assert blocks[0].normalized_speaker == "lopez sanchez"


def test_split_speech_blocks_uses_parenthetical_for_secretary_heading() -> None:
    html = """
    <html><body>
      <p>El senor SECRETARIO (Pau Pernau): Leo el texto.</p>
    </body></html>
    """
    blocks = split_speech_blocks(html_to_visible_text(html))
    assert blocks[0].normalized_speaker == "secretario pau pernau"


def test_split_speech_blocks_uses_parenthetical_for_minister_heading() -> None:
    text = (
        "El se\u00f1or MINISTRO DE ASUNTOS EXTERIORES "
        "(Garc\u00eda- Margallo Marfil): Muchas gracias."
    )
    blocks = split_speech_blocks(text)
    assert blocks[0].normalized_speaker == "ministro garcia margallo marfil"


def test_split_speech_blocks_keeps_parenthetical_period_heading_when_body_wraps() -> None:
    text = (
        "El señor SECRETARIO DE ACCIÓN SINDICAL DE UGT FICA "
        "(Pasadas Muñoz).\n"
        "Quiero aprovechar esta comparecencia para explicar el acuerdo.\n"
        "La señora PRESIDENTA: Muchas gracias."
    )
    blocks = split_speech_blocks(text)
    assert [block.normalized_speaker for block in blocks] == [
        "secretario pasadas munoz",
        "presidenta",
    ]
    assert blocks[0].text == ("Quiero aprovechar esta comparecencia para explicar el acuerdo.")
    assert "La señora PRESIDENTA" not in blocks[0].text


def test_split_speech_blocks_accepts_diputado_role_heading() -> None:
    blocks = split_speech_blocks(
        "La diputada ALMODÓVAR SÁNCHEZ: Gracias, presidenta.\nLa señora PRESIDENTA: Continúo."
    )
    assert blocks[0].normalized_speaker == "almodovar sanchez"
    assert blocks[0].text == "Gracias, presidenta."


def test_role_heading_lineage_anchors_repeated_body_article_after_colon() -> None:
    text = (
        "La señora PRESIDENTA DE REDEIA CORPORACIÓN, S.A. (Corredor Sierra): La "
        "política energética."
    )

    rows = speech_block_rows(
        visible_text=text,
        source_url="https://www.congreso.es/diario.pdf",
        source_sha256="f" * 64,
        snapshot_date="2026-08-01",
        legislature="Leg.15",
        document_id="DSCD-15-CI-62",
        source_kind="pdf",
        extraction_method="pymupdf_text",
    )

    assert len(rows) == 1
    assert rows[0]["content_text"] == "La política energética."
    source_map = json.loads(rows[0]["source_map_json"])
    removals = json.loads(rows[0]["removed_spans_json"])
    expected_body_start = text.index(": La") + 2
    assert source_map == [
        {
            "content_start": 0,
            "content_end": len("La política energética."),
            "source_start": expected_body_start,
            "source_end": len(text),
            "kind": "line_identity_projection",
        }
    ]
    assert removals == [
        {
            "source_start": 0,
            "source_end": expected_body_start,
            "kind": "speaker_heading_wrapper",
        }
    ]


def test_source_map_records_character_exact_whitespace_transformations() -> None:
    text = "El seÃ±or GARCIA: Hola   mundo.\nSegunda\tlÃ­nea."

    block = split_speech_blocks(text)[0]
    projections = json.loads(block.source_map_json)

    assert block.text == "Hola mundo.\nSegunda lÃ­nea."
    assert any(
        item["kind"] == "whitespace_collapse_projection"
        and text[item["source_start"] : item["source_end"]] == "   "
        and block.text[item["content_start"] : item["content_end"]] == " "
        for item in projections
    )
    assert any(
        item["kind"] == "line_separator_projection"
        and text[item["source_start"] : item["source_end"]] == "\n"
        and block.text[item["content_start"] : item["content_end"]] == "\n"
        for item in projections
    )
    assert any(
        item["kind"] == "whitespace_collapse_projection"
        and text[item["source_start"] : item["source_end"]] == "\t"
        for item in projections
    )


def test_split_speech_blocks_recovers_role_heading_without_honorific() -> None:
    text = "El MINISTRO DE ECONOMÍA, COMERCIO Y EMPRESA (Cuerpo Caballero): Muchísimas gracias."

    blocks = split_speech_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].speaker_heading == (
        "El MINISTRO DE ECONOMÍA, COMERCIO Y EMPRESA (Cuerpo Caballero)"
    )
    assert blocks[0].normalized_speaker == "ministro cuerpo caballero"
    assert blocks[0].text == "Muchísimas gracias."


def test_split_speech_blocks_recovers_official_dot_delimiter_typo() -> None:
    text = (
        "La señora PRESIDENTA: Tiene la palabra.\n"
        "El señor MINISTRO DEL INTERIOR (Grande-Marlaska Gómez). "
        "Muchas gracias, señora presidenta.\n"
        "La señora PRESIDENTA: Muchas gracias, señor ministro."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert blocks[1].speaker_heading == ("El señor MINISTRO DEL INTERIOR (Grande-Marlaska Gómez)")
    assert blocks[1].normalized_speaker == "ministro grande marlaska gomez"
    assert blocks[1].text == "Muchas gracias, señora presidenta."


def test_dot_delimited_role_heading_does_not_swallow_next_inline_speaker() -> None:
    text = (
        "El señor VICEPRESIDENTE (Rodríguez Gómez de Celis). Por el Grupo "
        "Parlamentario\nSocialista, tiene la palabra el señor Sarrià Morell.\n"
        "El señor SARRIÀ MORELL: Gracias, señor presidente."
    )

    blocks = split_speech_blocks(text)

    assert [block.normalized_speaker for block in blocks] == [
        "vicepresidente rodriguez gomez de celis",
        "sarria morell",
    ]
    assert blocks[0].text == (
        "Por el Grupo Parlamentario\nSocialista, tiene la palabra el señor Sarrià Morell."
    )
    assert blocks[1].text == "Gracias, señor presidente."


def test_split_speech_blocks_uses_parenthetical_name_for_occupational_role() -> None:
    text = (
        "El señor DIRECTOR DE LA AGENCIA ESPAÑOLA (Leis García): Comparezco.\n"
        "El señor CATEDRÁTICO DE DERECHO CIVIL (Nasarre Aznar): Respondo.\n"
        "La señora REPRESENTANTE DEL PARLAMENTO (Jurado Fernández de Córdoba): "
        "Defiendo la propuesta."
    )

    blocks = split_speech_blocks(text)

    assert [block.normalized_speaker for block in blocks] == [
        "director leis garcia",
        "catedratico nasarre aznar",
        "representante jurado fernandez de cordoba",
    ]


def test_split_speech_blocks_uses_parenthetical_name_for_public_office_roles() -> None:
    text = (
        "El señor GOBERNADOR DEL BANCO DE ESPAÑA (Hernández de Cos): Comparezco.\n"
        "El señor DEFENSOR DEL PUEBLO (Gabilondo Pujol): Respondo.\n"
        "La señora DELEGADA DEL GOBIERNO (Sureda Llull): Continúo."
    )

    blocks = split_speech_blocks(text)

    assert [block.normalized_speaker for block in blocks] == [
        "gobernador hernandez de cos",
        "defensor gabilondo pujol",
        "delegada sureda llull",
    ]


def test_split_speech_blocks_uses_final_parenthetical_for_extended_roles() -> None:
    text = (
        "El señor SÍNDICO MAYOR (Salazar Canalda): Comparezco.\n"
        "El señor PROFESOR DE LA UNED (Olivas Osuna): Respondo.\n"
        "La señora PUBLIC POLICY MANAGER (Verbrugghe): Continúo.\n"
        "El señor DECANO (Mattioli Jacobs): Informo.\n"
        "El señor VICEDECANO (Tabarés Cuadrado): Explico.\n"
        "La señora DIRECTORA DEL PROGRAMA P(A)T (Domènech Moral): Detallo.\n"
        "El señor OFICIAL JURÍDICO (Solomon): Concluyo."
    )

    blocks = split_speech_blocks(text)

    assert [block.normalized_speaker for block in blocks] == [
        "sindico salazar canalda",
        "profesor olivas osuna",
        "manager verbrugghe",
        "decano mattioli jacobs",
        "vicedecano tabares cuadrado",
        "directora domenech moral",
        "oficial solomon",
    ]


def test_normalize_speaker_name_keeps_lexical_hyphen_as_token_boundary() -> None:
    assert normalize_speaker_name("Sánchez Pérez-Castejón, Pedro") == (
        "sanchez perez castejon pedro"
    )


def test_normalize_speaker_name_treats_pdf_format_control_as_token_boundary() -> None:
    assert normalize_speaker_name("López\u200cSánchez, José Pablo") == ("lopez sanchez jose pablo")


def test_normalize_speaker_name_splits_missing_camelcase_separator() -> None:
    assert normalize_speaker_name("GarcíaPage Sánchez, Emiliano") == (
        "garcia page sanchez emiliano"
    )


def test_speech_block_rows_records_official_presidency_identity_evidence() -> None:
    presidency = "PRESIDENCIA DE LA EXCMA. SRA. D.ª FRANCINA ARMENGOL SOCIAS"
    page = f"{presidency}\nSesión plenaria núm. 6\nLa señora PRESIDENTA: Se abre la sesión."

    rows = speech_block_rows(
        visible_text="",
        page_texts=[page],
        source_url="https://www.congreso.es/diario.pdf",
        source_sha256="a" * 64,
        snapshot_date="2026-08-01",
        legislature="Leg.15",
        document_id="DSCD-15-PL-6",
        source_kind="pdf",
        extraction_method="pymupdf",
    )

    evidence = json.loads(rows[0]["speaker_identity_evidence_json"])
    assert len(evidence) == 1
    assert evidence[0]["kind"] == "official_document_presidency"
    assert evidence[0]["role"] == "presidenta"
    assert evidence[0]["normalized_speaker"] == "francina armengol socias"
    assert evidence[0]["source_text"] == presidency
    assert page[evidence[0]["source_char_start"] : evidence[0]["source_char_end"]] == presidency


def test_speech_block_rows_recovers_presidency_after_pypdf_ordinal_replacement() -> None:
    presidency = "PRESIDENCIA DE LA EXCMA. SRA. D.� FRANCINA ARMENGOL SOCIAS"
    page = f"{presidency}\nSUMARIO\nLa señora PRESIDENTA: Se abre la sesión."

    rows = speech_block_rows(
        visible_text="",
        page_texts=[page],
        source_url="https://www.congreso.es/DSCG-15-SC-1.PDF",
        source_sha256="c" * 64,
        snapshot_date="2026-08-02",
        legislature="Leg.15",
        document_id="L15-DSCG-15-SC-1",
        source_kind="pdf",
        extraction_method="pypdf_text",
    )

    evidence = json.loads(rows[0]["speaker_identity_evidence_json"])
    assert evidence[0]["speaker_name"] == "FRANCINA ARMENGOL SOCIAS"
    assert evidence[0]["source_text"] == presidency


def test_split_speech_blocks_projects_named_floor_denial() -> None:
    text = "El señor PRESIDENTE: Tiene la palabra la señora Marcos Ortega. (Denegación). Muy bien."

    blocks = split_speech_blocks(text)
    response = next(block for block in blocks if block.turn_kind == "nonverbal_response")

    assert response.speaker_heading == "la señora Marcos Ortega"
    assert response.normalized_speaker == "marcos ortega"
    assert response.text == "Denegación"
    assert text[response.source_char_start : response.source_char_end] == response.raw_text
    projection = json.loads(response.source_map_json)[0]
    assert text[projection["source_start"] : projection["source_end"]] == "Denegación"


def test_split_speech_blocks_recovers_named_floor_continuation_without_heading() -> None:
    text = (
        "La señora PRESIDENTA: Tiene la palabra la señora Vázquez.\n"
        "Señorías, continúo con la pregunta.\n"
        "La señora PRESIDENTA: Gracias."
    )

    blocks = split_speech_blocks(text)
    continuation = next(block for block in blocks if block.normalized_speaker == "vazquez")

    assert continuation.text == "Señorías, continúo con la pregunta."
    assert continuation.turn_kind == "formal_turn"
    assert "floor continuation" in continuation.speaker_heading
    assert text[continuation.source_char_start : continuation.source_char_end] == (
        "La señora PRESIDENTA: Tiene la palabra la señora Vázquez.\n"
        "Señorías, continúo con la pregunta."
    )
    removed = json.loads(continuation.removed_spans_json)
    assert removed[0]["kind"] == "floor_call_context"


def test_speech_block_rows_projects_unique_named_role_to_later_generic_heading() -> None:
    page = (
        "El señor PRESIDENTE DEL GOBIERNO (Sánchez Pérez-Castejón): Primera respuesta.\n"
        "La señora PRESIDENTA: Continúe.\n"
        "El señor PRESIDENTE DEL GOBIERNO: Segunda respuesta."
    )

    rows = speech_block_rows(
        visible_text="",
        page_texts=[page],
        source_url="https://www.congreso.es/DSCD-15-PL-30.PDF",
        source_sha256="b" * 64,
        snapshot_date="2026-08-02",
        legislature="Leg.15",
        document_id="DSCD-15-PL-30",
        source_kind="pdf",
        extraction_method="pypdf_text",
    )

    generic = rows[2]
    evidence = json.loads(generic["speaker_identity_evidence_json"])
    assert generic["normalized_speaker"] == "presidente del gobierno"
    assert evidence == [
        {
            "kind": "official_document_unique_role_identity",
            "role": "presidente del gobierno",
            "speaker_name": "Sánchez Pérez-Castejón",
            "normalized_speaker": "sanchez perez castejon",
            "source_char_start": evidence[0]["source_char_start"],
            "source_char_end": evidence[0]["source_char_end"],
            "source_text": evidence[0]["source_text"],
            "source_offset_basis": "document_content_text_unicode_codepoints",
        }
    ]
    assert (
        page[evidence[0]["source_char_start"] : evidence[0]["source_char_end"]]
        == evidence[0]["source_text"]
    )
    assert "Sánchez Pérez-Castejón" in evidence[0]["source_text"]


def test_split_speech_blocks_joins_corporate_role_heading_after_abbreviation() -> None:
    text = (
        "La señora PRESIDENTA: Tiene la palabra.\n"
        "El señor PRESIDENTE EJECUTIVO DEL CONSEJO DE ADMINISTRACIÓN DE "
        "TELEFÓNICA, S.A.\n"
        "(Murtra Millar): Muchísimas gracias, señora presidenta."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[1].speaker_heading.endswith("S.A. (Murtra Millar)")
    assert blocks[1].normalized_speaker == "presidente murtra millar"
    assert blocks[1].text == "Muchísimas gracias, señora presidenta."


def test_split_speech_blocks_joins_long_multiline_compareciente_heading() -> None:
    text = (
        "El señor PRESIDENTE: Tiene la palabra.\n"
        "El señor PRESIDENTE DE LA SOCIEDAD ESTATAL CORREOS Y TELÉGRAFOS, S.A., S.M.E.\n"
        "Y RESPONSABLE DE LA DIRECCIÓN GENERAL DE OPERACIONES Y SERVICIOS\n"
        "PÚBLICOS DE LA COMPAÑÍA ESTATAL\n"
        "(Saura García): Gracias, señor presidente."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[1].speaker_heading.endswith("(Saura García)")
    assert blocks[1].normalized_speaker == "presidente saura garcia"
    assert blocks[1].text == "Gracias, señor presidente."


def test_split_speech_blocks_joins_long_candidate_role_parenthetical_heading() -> None:
    text = (
        "La señora PRESIDENTA: Tiene la palabra.\n"
        "El señor GUILLÉN ESPEJO‑SAAVEDRA (candidato propuesto por el Gobierno como\n"
        "presidente de la Autoridad Administrativa Independiente para la Investigación "
        "Técnica de\n"
        "Accidentes e Incidentes Ferroviarios, Marítimos y de Aviación Civil): "
        "Muchas gracias."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[1].normalized_speaker == "guillen espejo saavedra"
    assert blocks[1].speaker_heading.startswith("El señor GUILLÉN ESPEJO‑SAAVEDRA")
    assert blocks[1].text == "Muchas gracias."


def test_split_speech_blocks_supports_de_la_senora_compareciente_heading() -> None:
    text = (
        "El señor PRESIDENTE: Tiene la palabra.\n"
        "De la señora TRÍAS GIL (profesora de Antropología y Ética): Buenos días."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[1].speaker_heading.startswith("De la señora TRÍAS GIL")
    assert blocks[1].normalized_speaker == "trias gil"
    assert blocks[1].text == "Buenos días."


def test_split_speech_blocks_keeps_name_before_parenthetical_role() -> None:
    html = """
    <html><body>
      <p>La senora DIAZ PEREZ (vicepresidenta segunda y ministra): Gracias.</p>
    </body></html>
    """
    blocks = split_speech_blocks(html_to_visible_text(html))
    assert blocks[0].normalized_speaker == "diaz perez"


def test_split_speech_blocks_detects_inline_historical_heading() -> None:
    text = (
        "El senor PRESIDENTE: Tiene la palabra el senor De la Fuente. "
        "El senor DE LA FUEN- TE Y DE LA FUENTE: Muchas gracias."
    )
    blocks = split_speech_blocks(text)
    assert len(blocks) == 2
    assert blocks[1].normalized_speaker == "de la fuente y de la fuente"
    assert "Muchas gracias" in blocks[1].text


def test_split_speech_blocks_keeps_annotations_and_extracts_single_interruption() -> None:
    text = (
        "La señora PRESIDENTA: Tiene la palabra. (Continúan las protestas.—\n"
        "El señor Ibáñez Mezquita: ¡Esto es intolerable!—El señor Figaredo "
        "Álvarez-Sala: ¡Fascistas, vosotros!—Varios señores diputados pronuncian "
        "palabras que no se perciben).\n"
        "Por favor, guarden silencio.\n"
        "El señor RALLO LOMBARTE: Señora presidenta, señoras y señores diputados "
        "(el señor Ortega Smith-Molina: ¡Vendido!), Europa ha hablado."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert blocks[0].speaker_heading == "La señora PRESIDENTA"
    assert "El señor Ibáñez Mezquita: ¡Esto es intolerable!" in blocks[0].text
    assert "Por favor, guarden silencio." in blocks[0].text
    assert blocks[1].speaker_heading == "El señor RALLO LOMBARTE"
    assert "el señor Ortega Smith-Molina: ¡Vendido!" in blocks[1].text
    assert blocks[2].normalized_speaker == "ortega smith molina"
    assert blocks[2].text == "¡Vendido!"
    assert blocks[2].turn_kind == "parenthetical_interruption"


def test_split_speech_blocks_extracts_closed_parenthetical_interruption() -> None:
    text = (
        "El señor PRESIDENTE: Las votaremos agrupadas. "
        "(El señor Santos Maraver: De una en una). Una a una, sí.\n"
        "El señor CATALÁN HIGUERAS: Muchas gracias."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert "(El señor Santos Maraver: De una en una). Una a una, sí." in blocks[0].text
    assert blocks[1].normalized_speaker == "santos maraver"
    assert blocks[1].text == "De una en una"
    assert blocks[1].turn_kind == "parenthetical_interruption"
    assert blocks[2].normalized_speaker == "catalan higueras"


def test_uppercase_parenthetical_interruption_is_not_promoted_to_formal_turn() -> None:
    text = "El se\u00f1or PRESIDENTE: Silencio. (El se\u00f1or BRAVO: Primera). Contin\u00fae."

    blocks = split_speech_blocks(text)

    assert [(block.turn_kind, block.normalized_speaker) for block in blocks] == [
        ("formal_turn", "presidente"),
        ("parenthetical_interruption", "bravo"),
    ]
    assert "(El se\u00f1or BRAVO: Primera). Contin\u00fae." in blocks[0].text
    assert blocks[1].text == "Primera"


def test_split_speech_blocks_projects_named_nonverbal_responses() -> None:
    text = (
        "El señor PRESIDENTE: Señora Jordà, ¿acepta usted alguna de las "
        "enmiendas? (Denegaciones). No.\n"
        "El señor PRESIDENTE: Señora Sagastizabal, entiendo que han presentado "
        "una transaccional. (Asentimiento.―La señora presidenta ocupa la "
        "Presidencia)."
    )

    blocks = [
        block for block in split_speech_blocks(text) if block.turn_kind == "nonverbal_response"
    ]

    assert [(block.speaker_heading, block.normalized_speaker, block.text) for block in blocks] == [
        ("Señora Jordà", "jorda", "Denegaciones"),
        ("Señora Sagastizabal", "sagastizabal", "Asentimiento"),
    ]
    for block in blocks:
        projection = json.loads(block.source_map_json)[0]
        assert text[projection["source_start"] : projection["source_end"]] == block.text


def test_nonverbal_response_preserves_multipage_source_range() -> None:
    rows = speech_block_rows(
        visible_text="",
        page_texts=[
            (
                "El se\u00f1or PRESIDENTE: Se\u00f1ora Jord\u00e0, "
                "\u00bfacepta usted alguna de las enmiendas?"
            ),
            "(Denegaciones). No.",
        ],
        source_url="https://www.congreso.es/DSCD-15-CO-1.PDF",
        source_sha256="a" * 64,
        snapshot_date="2026-08-01",
        legislature="Leg.15",
        document_id="DSCD-15-CO-1",
        source_kind="pdf",
        extraction_method="pypdf_text",
    )

    response = next(row for row in rows if row["turn_kind"] == "nonverbal_response")

    assert response["content_text"] == "Denegaciones"
    assert response["page_hint"] == 1
    assert response["page_end"] == 2


def test_split_speech_blocks_projects_explicit_named_negative_gesture() -> None:
    text = (
        "El señor PRESIDENTE: Le pregunto al señor Clavell si admite la "
        "enmienda. (El señor Clavell López hace signos negativos). No la admite."
    )

    block = next(
        item for item in split_speech_blocks(text) if item.turn_kind == "nonverbal_response"
    )

    assert block.speaker_heading == "El señor Clavell López"
    assert block.normalized_speaker == "clavell lopez"
    assert block.text == "hace signos negativos"
    projection = json.loads(block.source_map_json)[0]
    assert text[projection["source_start"] : projection["source_end"]] == block.text


def test_nonverbal_response_does_not_cross_to_a_different_named_addressee() -> None:
    text = (
        "La señora PRESIDENTA: Señor Arribas, no entre en debate. "
        "¿Ha finalizado su intervención, señor Hernando? "
        "(Asentimiento.―El señor Gil Lázaro pide la palabra)."
    )

    blocks = split_speech_blocks(text)

    assert all(block.turn_kind != "nonverbal_response" for block in blocks)


def test_split_speech_blocks_does_not_promote_roll_call_narration() -> None:
    text = (
        "La señora PRESIDENTA: Comienza la votación.\n"
        "El señor Hernández Quero dijo: No a la traición.\n"
        "El señor Abascal Conde dijo: No a la corrupción. No.\n"
        "La señora PRESIDENTA: Termina la votación."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert "El señor Hernández Quero dijo: No a la traición." in blocks[0].text
    assert "El señor Abascal Conde dijo: No a la corrupción. No." in blocks[0].text
    assert blocks[1].text == "Termina la votación."


def test_split_speech_blocks_does_not_promote_lowercase_narrative_colons() -> None:
    text = (
        "La señora JORDÀ I ROURA: Com deia el president, el senyor Vidal ho deia: "
        "Sí a la llengua. También citó el señor Sánchez y dice: querida Giorgia.\n"
        "La señora PRESIDENTA: Muchas gracias."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert "el senyor Vidal ho deia: Sí a la llengua" in blocks[0].text
    assert "el señor Sánchez y dice: querida Giorgia" in blocks[0].text
    assert blocks[1].speaker_heading == "La señora PRESIDENTA"


def test_split_speech_blocks_does_not_promote_role_led_narrative_colons() -> None:
    text = (
        "La señora ORADORA: El presidente Pedro Sánchez hace referencia a esta "
        "legislatura como la legislatura de la vivienda. Por eso le pregunto: "
        "¿por qué la llaman así?\n"
        "La señora PRESIDENTA: Muchas gracias."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert "El presidente Pedro Sánchez hace referencia" in blocks[0].text
    assert blocks[1].speaker_heading == "La señora PRESIDENTA"


def test_agenda_metadata_does_not_promote_named_narrative_to_heading() -> None:
    text = (
        "El señor VICEPRESIDENTE (Sahuquillo García): Muchas gracias.\n"
        "La señora Santana ya nos ha anunciado que hay una enmienda transaccional y, "
        "cuando la tengamos, se la pasaremos a los portavoces.\n"
        "— PARA QUE SE RECIBA LA SUBVENCIÓN COMPROMETIDA. "
        "(Número de expediente 161/001777).\n"
        "El señor VICEPRESIDENTE (Sahuquillo García): Pasamos al siguiente punto."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert [block.normalized_speaker for block in blocks] == [
        "vicepresidente sahuquillo garcia",
        "vicepresidente sahuquillo garcia",
    ]
    assert "La señora Santana ya nos ha anunciado" in blocks[0].text
    assert blocks[1].text == "Pasamos al siguiente punto."


def test_split_speech_blocks_does_not_swallow_narrative_before_real_heading() -> None:
    text = (
        "El señor PRESIDENTE: Muchísimas gracias, señora Vaquero.\n"
        "El señor Sánchez Serna, del Grupo Podemos, no está. No sé si hay alguien "
        "en el Grupo Mixto que quiera intervenir. (Pausa). ¿No? En ese caso, aquí "
        "termina el turno de intervenciones.\n"
        "Le pregunto a la portavoz del Partido Popular si acepta las enmiendas.\n"
        "La señora VÁZQUEZ JIMÉNEZ: Estamos justamente firmando las enmiendas "
        "transaccionales para entregarlas a la Mesa.\n"
        "El señor PRESIDENTE: Perfecto."
    )

    blocks = split_speech_blocks(text)

    assert [block.normalized_speaker for block in blocks] == [
        "presidente",
        "vazquez jimenez",
        "presidente",
    ]
    assert blocks[0].text.endswith("si acepta las enmiendas.")
    assert blocks[1].text == (
        "Estamos justamente firmando las enmiendas transaccionales para entregarlas a la Mesa."
    )


def test_unbalanced_parenthesis_cannot_hide_later_formal_turns() -> None:
    text = (
        "El señor MINISTRO (Bolaños García): Respondo. (Una anotación rota\n"
        "La señora MUÑOZ DE LA IGLESIA: Formula la pregunta.\n"
        "El señor FIGAREDO ÁLVAREZ-SALA: Continúa el debate."
    )

    blocks = split_speech_blocks(text)

    assert [block.speaker_heading for block in blocks] == [
        "El señor MINISTRO (Bolaños García)",
        "La señora MUÑOZ DE LA IGLESIA",
        "El señor FIGAREDO ÁLVAREZ-SALA",
    ]
    assert blocks[0].text == "Respondo. (Una anotación rota"


def test_unclosed_native_parenthetical_speaker_is_not_a_formal_turn() -> None:
    text = (
        "La señora PRESIDENTA: Guarden silencio.\n"
        "(El señor Asarta Cuevas: He pedido la palabra\n"
        "La señora PORTAVOZ: Continúo con mi intervención."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert "(El señor Asarta Cuevas: He pedido la palabra" in blocks[0].text
    assert blocks[1].speaker_heading == "La señora PORTAVOZ"


def test_lowercase_parenthetical_speaker_is_an_explicit_interruption_turn() -> None:
    text = (
        "La señora PRESIDENTA: Mantenga el orden.\n"
        "(el señor Rodríguez Serra: ¡Sí está!) y debemos respetar a las "
        "personas comparecientes.\n"
        "El señor PORTAVOZ: Continúo."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 3
    assert "(el señor Rodríguez Serra: ¡Sí está!)" in blocks[0].text
    assert blocks[1].speaker_heading == "el señor Rodríguez Serra"
    assert blocks[1].text == "¡Sí está!"
    assert blocks[1].turn_kind == "parenthetical_interruption"
    assert blocks[2].speaker_heading == "El señor PORTAVOZ"


def test_parenthetical_interruption_preserves_exact_source_projection() -> None:
    text = (
        "El señor CAMINO MIÑANA: Ustedes son indistinguibles. "
        "(El señor Bravo Baena: El INE). Porque son ustedes lo mismo."
    )

    rows = speech_block_rows(
        visible_text=text,
        source_url="https://www.congreso.es/DSCD-15-PL-43.PDF",
        source_sha256="a" * 64,
        snapshot_date="2026-08-01",
        legislature="Leg.15",
        document_id="DSCD-15-PL-43",
        source_kind="pdf",
        extraction_method="pypdf",
    )
    interruption = next(row for row in rows if row["turn_kind"] == "parenthetical_interruption")

    assert interruption["speaker_heading"] == "El señor Bravo Baena"
    assert interruption["normalized_speaker"] == "bravo baena"
    assert interruption["content_text"] == "El INE"
    assert interruption["raw_text"] == "(El señor Bravo Baena: El INE)"
    projection = json.loads(interruption["source_map_json"])[0]
    assert text[projection["source_start"] : projection["source_end"]] == "El INE"
    removed = json.loads(interruption["removed_spans_json"])
    assert [item["kind"] for item in removed] == [
        "parenthetical_interruption_heading_wrapper",
        "parenthetical_interruption_closing_wrapper",
    ]
    assert not has_embedded_speaker_heading(interruption["raw_text"])


def test_wrapped_parenthetical_interruption_normalizes_layout_whitespace() -> None:
    text = "La señora PRESIDENTA: Silencio. (El señor Bravo Baena: Primera\nsegunda). Continúe."

    block = next(
        item for item in split_speech_blocks(text) if item.turn_kind == "parenthetical_interruption"
    )
    projection = json.loads(block.source_map_json)[0]

    assert block.text == "Primera segunda"
    source_body = text[projection["source_start"] : projection["source_end"]]
    assert "\n" in source_body
    assert " ".join(source_body.split()) == block.text


def test_split_speech_blocks_repairs_collapsed_ocr_speaker_prefixes() -> None:
    text = (
        "ElsenorPRESIDENTE: Tiene la palabra.\n"
        "ElsenorCAMUNASSOLIS: Muchas gracias, senor Presidente."
    )

    blocks = split_speech_blocks(text)

    assert [block.speaker_heading for block in blocks] == [
        "El senor PRESIDENTE",
        "El senor CAMUNASSOLIS",
    ]
    assert blocks[1].text == "Muchas gracias, senor Presidente."


def test_split_speech_blocks_repairs_ei_without_promoting_procedural_mentions() -> None:
    text = (
        "El senor PRESIDENTE: Gracias.\n"
        "EI\n"
        "senorMinistrodeJusticiatienelapalabra.\n"
        "EISenorMINISTRODEJUSTICIA (Ledesma Bartret): Respondo a la pregunta."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[0].speaker_heading == "El senor PRESIDENTE"
    assert "senorMinistrodeJusticiatienelapalabra" in blocks[0].text
    assert blocks[1].speaker_heading == ("EI Senor MINISTRODEJUSTICIA (Ledesma Bartret)")
    assert blocks[1].text == "Respondo a la pregunta."


def test_split_speech_blocks_repairs_observed_collapsed_ocr_honorifics() -> None:
    text = (
        "El senor PRESIDENTE: Abre la sesion.\n"
        "Elscnor BARRERO LOPEZ: Primera respuesta.\n"
        "Elenor CORTE MIER: Segunda respuesta.\n"
        "ElseAorHERRERO RODRIGUEZDE MINON: Tercera respuesta.\n"
        "ElscniorMOLINACABRERA: Cuarta respuesta."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 5
    assert [block.normalized_speaker for block in blocks] == [
        "presidente",
        "barrero lopez",
        "corte mier",
        "herrero rodriguezde minon",
        "molinacabrera",
    ]
    assert has_embedded_speaker_heading("Texto.\nElzzzzzPRESIDENTE: Incidencia.")
    assert not has_embedded_speaker_heading("Texto parlamentario ordinario.")


def test_split_speech_blocks_joins_wrapped_senior_ocr_heading() -> None:
    text = (
        "El senior PRESIDENTE: Tiene la palabra.\n"
        "El senior DIRECTORGENERALDELENTEPUBLICO\n"
        "RTVE (Garcia Candau): Respondo a la pregunta."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 2
    assert blocks[0].speaker_heading == "El senior PRESIDENTE"
    assert blocks[1].speaker_heading == (
        "El senior DIRECTORGENERALDELENTEPUBLICO RTVE (Garcia Candau)"
    )
    assert blocks[1].text == "Respondo a la pregunta."


def test_split_speech_blocks_detects_wrapped_inline_heading() -> None:
    text = (
        "El se\u00f1or PRESIDENTE: Tiene la palabra el se\u00f1or candidato.\n"
        "El se\u00f1or RAJOY BREY (Candidato a la Presidencia\n"
        "del Gobierno): Muchas gracias."
    )
    blocks = split_speech_blocks(text)
    assert len(blocks) == 2
    assert blocks[1].normalized_speaker == "rajoy brey"
    assert blocks[1].text == "Muchas gracias."


def test_native_speaker_heading_preserves_observed_diacritics() -> None:
    text = "La señora DÍAZ ÁLVAREZ DE TOLEDO: Gracias.\nEl señor PANIAGUA NÚÑEZ: Buenos días."

    blocks = split_speech_blocks(text)

    assert [block.speaker_heading for block in blocks] == [
        "La señora DÍAZ ÁLVAREZ DE TOLEDO",
        "El señor PANIAGUA NÚÑEZ",
    ]


def test_wrapped_heading_source_span_keeps_the_complete_body() -> None:
    text = (
        "El senor PRESIDENTE: Tiene la palabra.\n"
        "El senor RAJOY BREY (Candidato a la Presidencia\n"
        "del Gobierno): Muchas gracias."
    )
    rows = speech_block_rows(
        visible_text=text,
        source_url="https://www.congreso.es/texto",
        source_sha256="a" * 64,
        snapshot_date="2026-08-01",
    )

    assert rows[1]["raw_text"].endswith("Muchas gracias.")
    source_map = json.loads(rows[1]["source_map_json"])[0]
    relative_start = source_map["source_start"] - rows[1]["source_char_start"]
    relative_end = source_map["source_end"] - rows[1]["source_char_start"]
    assert rows[1]["raw_text"][relative_start:relative_end] == "Muchas gracias."


def test_match_intervention_text_handles_repeated_single_surname() -> None:
    blocks = split_speech_blocks("El senor DE LA FUEN- TE Y DE LA FUENTE: Intervengo.")
    match = match_intervention_text(
        speaker="Fuente y de la Fuente, Licinio de la (GAP)",
        blocks=blocks,
    )
    assert match.confidence > 0.7
    assert match.text_fragment == "Intervengo."


def test_normalize_speaker_name_removes_group() -> None:
    assert normalize_speaker_name("Requena Ruiz, Juan Diego (GP)") == "requena ruiz juan diego"


def test_normalize_speaker_name_is_accent_insensitive() -> None:
    assert normalize_speaker_name("Belmonte G\u00f3mez, Rafael") == "belmonte gomez rafael"


def test_normalize_speaker_name_accepts_official_ogu_heading_alias() -> None:
    assert normalize_speaker_name("OGU I CORBI") == "ogou i corbi"


def test_split_speech_blocks_accepts_lowercase_role_parenthetical_heading() -> None:
    blocks = split_speech_blocks(
        "El ministro de TRANSPORTES Y MOVILIDAD SOSTENIBLE (Puente Santiago): "
        "Muchas gracias, senora presidenta."
    )

    assert len(blocks) == 1
    assert blocks[0].normalized_speaker == "ministro puente santiago"


def test_page_markers_are_lineage_not_content() -> None:
    text = (
        "P\u00e1gina 7\n"
        "El se\u00f1or GARCIA: Primera l\u00ednea.\n"
        "P\u00e1gina 8\n"
        "Segunda l\u00ednea."
    )

    blocks = split_speech_blocks(text)

    assert len(blocks) == 1
    assert blocks[0].page_hint == 7
    assert blocks[0].page_end == 8
    assert blocks[0].text == "Primera l\u00ednea.\nSegunda l\u00ednea."
    assert "P\u00e1gina" not in blocks[0].text
    assert '"kind":"page_marker"' in blocks[0].removed_spans_json
    assert blocks[0].source_char_start is not None
    assert blocks[0].source_char_end is not None


def test_pdf_page_arrays_assign_pages_without_injecting_markers() -> None:
    rows = speech_block_rows(
        visible_text="ignored",
        page_texts=[
            "El se\u00f1or GARCIA: Primera parte.",
            "Continuaci\u00f3n en la p\u00e1gina siguiente.",
        ],
        source_url="https://www.congreso.es/DSCD-15-PL-1.PDF",
        source_sha256="a" * 64,
        snapshot_date="2026-08-01",
        legislature="Leg.15",
        document_id="DSCD-15-PL-1",
        source_kind="pdf",
        extraction_method="pymupdf_text",
    )

    assert len(rows) == 1
    assert rows[0]["page_hint"] == 1
    assert rows[0]["page_end"] == 2
    assert rows[0]["content_text"] == (
        "Primera parte.\nContinuaci\u00f3n en la p\u00e1gina siguiente."
    )
    assert "P\u00e1gina 2" not in rows[0]["content_text"]
    assert rows[0]["source_document_sha256"] == "a" * 64


def test_pdf_blocks_link_page_local_layout_removals_without_rewriting_them() -> None:
    removals = [
        json.dumps(
            {
                "page_number": page,
                "source_start": 0,
                "source_end": 6,
                "source_page_length": 50,
                "offset_basis": "selected_page_text_unicode_codepoints",
                "kind": "repeated_header",
                "text": f"Pág. {page}",
            }
        )
        for page in (1, 2)
    ]
    rows = speech_block_rows(
        visible_text="",
        page_texts=[
            "El señor GARCIA: Primer turno.",
            "La señora DÍAZ: Segundo turno.",
        ],
        layout_removals=removals,
        source_url="https://www.congreso.es/DSCD-15-PL-1.PDF",
        source_sha256="a" * 64,
        snapshot_date="2026-08-01",
        legislature="Leg.15",
        document_id="DSCD-15-PL-1",
        source_kind="pdf",
        extraction_method="pymupdf_text",
    )

    assert [
        [item["page_number"] for item in json.loads(row["document_layout_removals_json"])]
        for row in rows
    ] == [[1], [2]]

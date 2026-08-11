"""Regression tests owned by the public acquisition package."""

import pytest

from congreso_open_data.vote_summary_pdf_nominal import (
    parse_vote_summary_roll_call_texts,
)


def test_pdf_roll_call_parser_recovers_all_categories_and_footnote_lineage() -> None:
    result = parse_vote_summary_roll_call_texts(
        page_texts=_roll_call_pages(),
        target_page=1,
        expected_yes_votes=2,
        expected_no_votes=1,
        expected_abstentions=0,
        expected_member_count=5,
    )

    assert len(result.votes) == 5
    assert (result.yes_votes, result.no_votes, result.abstentions) == (2, 1, 0)
    assert result.null_votes == 1
    assert result.not_voting == 1
    assert result.emitted_votes == 4
    assert result.page_start == 1
    assert result.page_end == 2
    assert result.footnote_markers == ("1",)
    assert result.votes[0].raw_deputy_name.endswith(" 1")
    assert result.votes[0].deputy_name == "Álvarez Uno, Ana"
    assert {vote.roll_call_section for vote in result.votes} == {
        "floor",
        "telematic",
        "bureau",
    }


def test_pdf_roll_call_parser_rejects_duplicate_deputy_identity() -> None:
    pages = _roll_call_pages()
    pages[1] = pages[1].replace("Bravo Dos, Berta", "Álvarez Uno, Ana")

    with pytest.raises(ValueError, match="duplicate deputy names"):
        parse_vote_summary_roll_call_texts(
            page_texts=pages,
            target_page=1,
            expected_yes_votes=2,
            expected_no_votes=1,
            expected_abstentions=0,
            expected_member_count=5,
        )


def test_pdf_roll_call_parser_rejects_missing_absence() -> None:
    pages = _roll_call_pages()
    pages[1] = pages[1].replace("Echo Cinco, Eva\n", "")

    with pytest.raises(ValueError, match="absence total"):
        parse_vote_summary_roll_call_texts(
            page_texts=pages,
            target_page=1,
            expected_yes_votes=2,
            expected_no_votes=1,
            expected_abstentions=0,
            expected_member_count=5,
        )


def test_pdf_roll_call_parser_rejects_unpaired_footnote_marker() -> None:
    pages = _roll_call_pages()
    pages[0] = pages[0].replace("1 Realizado", "2 Realizado")

    with pytest.raises(ValueError, match="footnote markers do not reconcile"):
        parse_vote_summary_roll_call_texts(
            page_texts=pages,
            target_page=1,
            expected_yes_votes=2,
            expected_no_votes=1,
            expected_abstentions=0,
            expected_member_count=5,
        )


def test_pdf_roll_call_parser_accepts_geometry_parser_split_page_header() -> None:
    pages = _roll_call_pages()
    pages[1] = pages[1].replace(
        "Núm. 5 27 de septiembre de 2023 Pág. 2",
        "Núm. 5\n27\u202fde\u202fseptiembre\u202fde\u202f2023\nPág. 2",
    )

    result = parse_vote_summary_roll_call_texts(
        page_texts=pages,
        target_page=1,
        expected_yes_votes=2,
        expected_no_votes=1,
        expected_abstentions=0,
        expected_member_count=5,
    )

    assert len(result.votes) == 5


def _roll_call_pages() -> list[str]:
    return [
        """
        Por los secretarios se procede al llamamiento.
        Señoras y señores diputados que dijeron «sí»:
        Álvarez Uno, Ana 1
        cve: DSCD-15-PL-5
        1 Realizado por la secretaria el llamamiento del diputado,
        este corrigió el sentido de su voto.
        """,
        """
        DIARIO DE SESIONES DEL CONGRESO DE LOS DIPUTADOS
        CONGRESO DE LOS DIPUTADOS
        PLENO Y DIPUTACIÓN PERMANENTE
        Núm. 5 27 de septiembre de 2023 Pág. 2
        Señoras y señores diputados que votaron «sí» telemáticamente:
        Bravo Dos, Berta
        Señoras y señores diputados que dijeron «no»:
        Cano Tres, Carlos
        Votos nulos:
        Delta Cuatro, Diana
        Miembros de la Mesa ausentes:
        Echo Cinco, Eva
        La señora PRESIDENTA: El resultado de la votación ha sido el siguiente:
        votos emitidos, 4; sí, 2; no, 1; un voto nulo.
        """,
    ]

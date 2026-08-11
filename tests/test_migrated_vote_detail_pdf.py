"""Regression tests owned by the public acquisition package."""

import pytest

from congreso_open_data.vote_detail_pdf import parse_vote_detail_page_texts


def test_vote_detail_pdf_parses_and_reconciles_nominal_rows() -> None:
    result = parse_vote_detail_page_texts(
        [
            """Expediente\n1Votación:Sesión: 193 Fecha: 23-7-2026
            PRESENTES\nSÍ\nNO\nABSTENCIONES\nNO VOTAN
            3\n1\n1\n1\n1\nRESULTADO DE LA VOTACIÓN""",
            """SÍ\nGrupo A\n - Pérez Uno, Ana1201\nTotal: 1
            NO\nGrupo B\n - López Dos, LuisTELEMÁTICO\nTotal: 1""",
            """ABSTENCIONES\nGrupo C\n - Ruiz Tres, María3\nTotal: 1
            NO VOTAN\nGrupo D\n - Sanz Cuatro, José1234\nTotal: 1""",
        ]
    )

    assert result.session_number == 193
    assert result.vote_number == 1
    assert result.vote_date == "20260723"
    assert (result.present, result.yes_votes, result.no_votes) == (3, 1, 1)
    assert [row.vote for row in result.nominal_rows] == [
        "Sí",
        "No",
        "Abstención",
        "No vota",
    ]
    assert result.nominal_rows[1].seat == "TELEMÁTICO"


def test_vote_detail_pdf_allows_explicit_totals_only_source() -> None:
    result = parse_vote_detail_page_texts(
        [
            """Expediente\n5Votación:Sesión: 89 Fecha: 22-1-2025
            PRESENTES\nSÍ\nNO\nABSTENCIONES\nNO VOTAN
            345\n345\n0\n0\n5\nRESULTADO DE LA VOTACIÓN"""
        ]
    )

    assert result.nominal_rows == ()
    assert result.present == 345


def test_vote_detail_pdf_finds_totals_on_second_page_after_long_joint_title() -> None:
    result = parse_vote_detail_page_texts(
        [
            """Expediente conjunto muy largo
            56Votacion:Sesion: 40 Fecha: 23-5-2024""",
            """TOTAL
            PRESENTES
            SI
            NO
            ABSTENCIONES
            NO VOTAN
            1
            1
            0
            0
            0
            -
            RESULTADO DE LA VOTACION
            SI
            Grupo A
             - Perez Uno, Ana1201""",
        ]
    )

    assert result.vote_number == 56
    assert result.present == 1
    assert len(result.nominal_rows) == 1


def test_vote_detail_pdf_rejects_nominal_total_mismatch() -> None:
    with pytest.raises(ValueError, match="nominal rows"):
        parse_vote_detail_page_texts(
            [
                """1Votación:Sesión: 1 Fecha: 1-1-2026
                PRESENTES\nSÍ\nNO\nABSTENCIONES\nNO VOTAN
                1\n1\n0\n0\n0""",
                "SÍ\nGrupo A\n - Uno, Ana1\n - Dos, Bea2",
            ]
        )

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import parse_qs, unquote, urldefrag, urljoin, urlparse

from congreso_open_data.html import parse_visible_html
from congreso_open_data.identifiers import initiative_reference_identity
from congreso_open_data.normalization import normalize_record_keys, normalize_text, stable_id

_OCR_ARTICLE_PATTERN = r"(?:El|E1|Ei|81|8I|8l|La)"

SPEAKER_PATTERNS = (
    re.compile(
        r"^(?P<name>su majestad el rey(?: don [^:]{2,100})?):\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(
        r"^(?P<name>su alteza real la princesa de asturias,\s*"
        r"do\u00f1a [^:]{2,120}):\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(
        "^(El|La) "
        "(se\\u00f1or|se\\u00f1ora|senor|senora|se\\u00c3\\u00b1or|se\\u00c3\\u00b1ora) "
        "(?P<name>[^:]{3,500}):\\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(
        "^(El|La) "
        "(se\\u00f1or|se\\u00f1ora|senor|senora|se\\u00c3\\u00b1or|se\\u00c3\\u00b1ora) "
        "(?P<name>[^:\\n]{3,450}\\([^)]+\\))\\.\\s+(?P<body>.+)$",
        re.I,
    ),
    # Some official PDFs put the speech body on the following physical line:
    # ``El señor SECRETARIO (Apellido).`` followed by ``Texto...``.  Keep this
    # deliberately narrow (a parenthetical identity is required) so ordinary
    # prose ending in a full stop is never promoted to a speaker heading.
    re.compile(
        "^(El|La) "
        "(se\\u00f1or|se\\u00f1ora|senor|senora|se\\u00c3\\u00b1or|se\\u00c3\\u00b1ora) "
        "(?P<name>[^:\\n]{3,450}\\([^)]+\\))\\.\\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(
        r"^(El|La) (?P<title>diputado|diputada) "
        r"(?P<name>[^:\n]{3,500}):\s*(?P<body>.*)$",
        re.I,
    ),
    # A number of official pages use a full stop after an all-caps named
    # speaker and put the body on the same line (``El señor ARGÜELLES GARCÍA.
    # Gracias...``).  Requiring an uppercase initial token keeps narrative
    # prose from being promoted while covering mojibake/OCR capitals.
    re.compile(
        r"^(El|La) "
        r"(se\u00f1or|se\u00f1ora|senor|senora|se\u00c3\u00b1or|se\u00c3\u00b1ora) "
        r"(?P<name>(?=(?-i:[A-ZÃ�\ufffdÀ-ÖØ-Þ0-9 .,'’\-]+)\.\s)"
        r"[^:.\n]{3,450})\.\s+"
        r"(?P<body>.+)$",
        re.I,
    ),
    re.compile(
        r"^(El|La) (?P<name>(?:MINISTR[OA]|SECRETARI[OA](?:\s+DE\s+ESTADO)?|"
        r"PRESIDENT[EA]|VICEPRESIDENT[EA]|DIRECTOR(?:A)?|FISCAL|DEFENSOR(?:A)?)"
        r"\b[^:\n]{0,420}\([^)]+\)):\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(r"^(El|La) (senor|senora) (?P<name>[^:]{3,500}):\s*(?P<body>.*)$", re.I),
    re.compile(r"^(El|La) (señor|señora) (?P<name>[^:]{3,500}):\s*(?P<body>.*)$", re.I),
    re.compile(
        r"^De (?:el|la) (?:señor|señora|senor|senora|seÃ±or|seÃ±ora) "
        r"(?P<name>[^:]{3,500}):\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(
        r"^(El|La) (?P<role>presidente|presidenta|vicepresidente|vicepresidenta)"
        r"[^:]*:\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(
        r"^.*?(?P<name>presidente (?:de las|del) [^:]{2,100}?\([^)]+\)|"
        r"su majestad el rey(?=\s+(?:pronunci[oó]|ley[oó6])\b)|"
        r"su majestad el rey(?: don)? [^:]{2,100}?)[,;]?\s+"
        r"(?:pronunci[oó]|ley[oó6])\s+(?:en\s+\w+\s+)?"
        r"(?:el\s+)?siguiente\s+(?:discurso|mensaje):\s*(?P<body>.*)$",
        re.I,
    ),
    re.compile(
        rf"^\(?{_OCR_ARTICLE_PATTERN}\s+(?:[sc]e\S{{1,6}}r(?:a)?|Mor)\s+"
        r"(?P<name>[^:]{3,500}):\s*(?P<body>.*)$",
        re.I,
    ),
)
PAGE_PATTERN = re.compile("P(?:\\u00e1|a|\\u00c3\\u00a1)gina\\s+(?P<page>\\d+)", re.I)
INLINE_SPEAKER_PATTERN = re.compile(
    r"(?<!^)(?<!\()(?<!De )(?=(?:El|La) "
    r"(?:se\u00f1or|se\u00f1ora|senor|senora|seÃ±or|seÃ±ora) "
    r"[^:\n]{3,500}:\s*)",
    re.I,
)
OCR_INLINE_SPEAKER_PATTERN = re.compile(
    rf"(?<!^)(?<!\()(?<!De )(?=\(?{_OCR_ARTICLE_PATTERN}\s+"
    r"(?:[sc]e\S{1,6}r(?:a)?|Mor)\s+"
    r"[^:\n]{3,500}:\s*)",
    re.I,
)
OCR_GENERIC_SPEAKER_PATTERN = re.compile(
    rf"^\(?{_OCR_ARTICLE_PATTERN}\s+(?P<title>\S{{2,15}})\s+"
    r"(?P<name>[^:]{2,500}):\s*(?P<body>.*)$"
)
OCR_COLLAPSED_SPEAKER_PREFIX = re.compile(
    r"(?m)^\(?(?P<article>(?i:El|E1|Ei|81|8I|8l|La))\s*"
    r"(?P<title>[^\s:\n]{3,10}?)\s*"
    r"(?=[A-ZÁÉÍÓÚÜÑ]{2})"
)
OCR_EMBEDDED_SPEAKER_CANDIDATE = re.compile(
    rf"(?m)^\(?{_OCR_ARTICLE_PATTERN}[^\s:\n]{{2,24}}\s*"
    r"[A-ZÁÉÍÓÚÜÑ]{2}[^:\n]{0,140}:"
)
NAMED_FLOOR_CONTINUATION_PATTERN = re.compile(
    r"^(?P<article>El|La)\s+"
    r"(?P<honorific>se\u00f1or|se\u00f1ora|senor|senora|se\u00c3\u00b1or|se\u00c3\u00b1ora)\s+"
    r"[^:]{2,240}:\s*"
    r"Tiene\s+la\s+palabra\s+(?P<target_article>el|la)\s+"
    r"(?P<target_honorific>se\u00f1or|se\u00f1ora|senor|senora|se\u00c3\u00b1or|se\u00c3\u00b1ora)\s+"
    r"(?P<name>[^.:\n]{2,160})\.\s*$",
    re.I,
)

# The historical export contains two known sentinel/corrupt document names.
# Canonical targets are verified against the Congress publication catalogue:
# C_1977_022 (7 October 1977) and SC_000 (27 December 1978).
_OFFICIAL_PDF_FILENAME_OVERRIDES = {
    "C_1977_0..CCAL: 37 37.PDF": "C_1977_022.PDF",
    "C_1978_999.PDF": "SC_000.PDF",
}
_OFFICIAL_PDF_BASE = "https://www.congreso.es/public_oficiales/L0/CONG/DS"


WRAPPED_SPEAKER_START_PATTERN = re.compile(
    rf"^\(?{_OCR_ARTICLE_PATTERN}\s+\S{{2,15}}\s+[^:]{{3,500}}$",
    re.I,
)
NARRATIVE_SPEAKER_START_PATTERN = re.compile(
    r"(?:\bpre-\s*$|presi-?\s*dente (?:de las|del)\b|su majestad el rey)",
    re.I,
)
INTERVENTION_SPEECH_BLOCK_PARSER_VERSION = (
    "deterministic_turn_state_v34_named_floor_continuations_role_case_agenda_metadata"
)

# Standalone section titles are emitted inside the official transcript text
# stream, usually immediately after a chair's closing sentence.  They are
# document structure, not words spoken by the current speaker.  Keep this
# allow-list deliberately narrow: generic uppercase lines (roll calls,
# quoted titles and spelled-out names) remain speech content unless they match
# a known parliamentary section family.
_AGENDA_HEADING_PREFIXES = (
    "REAL DECRETO DE CONVOCATORIA DE ELECCIONES",
    "REAL DECRETO DE CREACIÓN Y AUTORIZACIÓN DE UNIVERSIDADES Y CENTROS",
    "RECURSOS CONTENCIOSO-ELECTORALES INTERPUESTOS",
    "JURAMENTO O PROMESA DE ACATAMIENTO DE LA CONSTITUCIÓN",
    "DISCURSO DE LA SEÑORA PRESIDENTA DEL CONGRESO DE LOS DIPUTADOS",
    "ELECCIÓN DE LA MESA DEL CONGRESO DE LOS DIPUTADOS",
    "ELECCIÓN DE VACANTES EN LA MESA DE LA COMISIÓN",
    "ELECCIÓN DE UN SEÑOR DIPUTADO, DE CONFORMIDAD CON EL PUNTO TERCERO DE LA",
    "ELECCIÓN DE LOS DIPUTADOS A LOS QUE SE REFIERE EL PUNTO TERCERO DE LA",
    "ELECCIÓN DE MIEMBROS DEL CONSEJO DE ADMINISTRACIÓN DE LA CORPORACIÓN RTVE",
    "ELECCIÓN DEL PRESIDENTE DE LA CORPORACIÓN RTVE",
    "ELECCIÓN PARA CUBRIR LA VACANTE EXISTENTE EN LA SECRETARÍA",
    "CELEBRACIÓN DE LAS COMPARECENCIAS DE CANDIDATOS PARA LA ELECCIÓN",
    "MODIFICACIÓN DEL ORDEN DEL DÍA",
    "COMPARECENCIA DEL GOBIERNO ANTE EL PLENO",
    "COMPARECENCIA DE LA ",
    "COMPARECENCIA DEL ",
    "COMPARECENCIA CONJUNTA ",
    "COMPARECENCIA PERIÓDICA ",
    "COMPARECENCIA EN LA COMISIÓN ",
    "COMPARECENCIAS EN RELACIÓN ",
    "PROPOSICIONES NO DE LEY",
    "PROPOSICIONES DE NO LEY",
    "PREGUNTAS",
    "INTERPELACIONES",
    "MOCIONES CONSECUENCIA DE INTERPELACIONES",
    "VOTACIONES",
    "ENMIENDAS DEL SENADO",
    "DICTAMEN",
    "PETICIÓN FORMULADA POR UN NÚMERO SUFICIENTE DE DIPUTADOS MIEMBROS",
    "TOMA EN CONSIDERACIÓN DE PROPOSICIONES DE LEY",
    "INFORME DE FISCALIZACIÓN DE LAS CONTABILIDADES DE LAS ELECCIONES",
)
_AGENDA_TITLE_CASE_EXACT = frozenset(
    {
        (
            "COMPARECENCIA ANTE LAS COMISIONES DE INVESTIGACIÓN DEL CONGRESO Y "
            "DEL SENADO O DE AMBAS CÁMARAS."
        ),
        "COMPARECENCIA ANTE LAS COMISIONES DE INVESTIGACIÓN DEL CONGRESO Y DEL SENADO.",
    }
)


def _agenda_fold(value: str) -> str:
    """Fold accents only for conservative agenda-title comparisons."""

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).upper()


def _looks_like_agenda_heading(line: str) -> bool:
    """Return true only for a standalone, known parliamentary section title."""

    value = normalize_text(line.strip())
    if not value or value.startswith(("•", "*")):
        return False
    if value in _AGENDA_TITLE_CASE_EXACT:
        return True
    folded = _agenda_fold(value)
    # The native PDFs print agenda titles in uppercase.  Requiring that style
    # prevents ordinary prose such as ``Real Decreto 6/2023...`` from being
    # removed from a speech.  Em-dash election items are equally distinctive.
    uppercase_style = value == value.upper()
    if value.startswith(("—", "–", "-")):
        uppercase_style = value.lstrip("—–- ") == value.lstrip("—–- ").upper()
    if not uppercase_style:
        return False
    return any(folded.startswith(_agenda_fold(prefix)) for prefix in _AGENDA_HEADING_PREFIXES)


# Keep the production comparison independent of source-file encoding artefacts
# in historical constants above.  Prefixes are intentionally ASCII-folded;
# the input is folded by ``_agenda_fold`` before comparison.
_AGENDA_ASCII_PREFIXES = (
    "REAL DECRETO DE CONVOCATORIA DE ELECCIONES",
    "REAL DECRETO DE CREACION Y AUTORIZACION DE UNIVERSIDADES Y CENTROS",
    "RECURSOS CONTENCIOSO-ELECTORALES INTERPUESTOS",
    "JURAMENTO O PROMESA DE ACATAMIENTO DE LA CONSTITUCION",
    "DISCURSO DE LA SENORA PRESIDENTA DEL CONGRESO DE LOS DIPUTADOS",
    "ELECCION DE LA MESA DEL CONGRESO DE LOS DIPUTADOS",
    "ELECCION DE VACANTES EN LA MESA DE LA COMISION",
    "ELECCION DE UN SENOR DIPUTADO, DE CONFORMIDAD CON EL PUNTO TERCERO DE LA",
    "ELECCION DE LOS DIPUTADOS A LOS QUE SE REFIERE EL PUNTO TERCERO DE LA",
    "ELECCION DE MIEMBROS DEL CONSEJO DE ADMINISTRACION DE LA CORPORACION RTVE",
    "ELECCION DEL PRESIDENTE DE LA CORPORACION RTVE",
    "ELECCION DE ",
    "ELECCION DEL ",
    "ELECCION PARA CUBRIR LA VACANTE EXISTENTE EN LA SECRETARIA",
    "CELEBRACION DE LAS COMPARECENCIAS DE CANDIDATOS PARA LA ELECCION",
    "MODIFICACION DEL ORDEN DEL DIA",
    "COMPARECENCIA DEL GOBIERNO ANTE EL PLENO",
    "COMPARECENCIA DE LA ",
    "COMPARECENCIA DEL ",
    "COMPARECENCIA CONJUNTA ",
    "COMPARECENCIA PERIODICA ",
    "COMPARECENCIA EN LA COMISION ",
    "COMPARECENCIAS EN RELACION ",
    "PROPOSICIONES NO DE LEY",
    "PROPOSICIONES DE NO LEY",
    "PREGUNTAS",
    "INTERPELACIONES",
    "MOCIONES CONSECUENCIA DE INTERPELACIONES",
    "VOTACIONES",
    "ENMIENDAS DEL SENADO",
    "DICTAMEN",
    "PETICION FORMULADA POR UN NUMERO SUFICIENTE DE DIPUTADOS MIEMBROS",
    "TOMA EN CONSIDERACION DE PROPOSICIONES DE LEY",
    "INFORME DE FISCALIZACION DE LAS CONTABILIDADES DE LAS ELECCIONES",
    "INCLUSION EN EL ORDEN DEL DIA",
    "EXCLUSION DEL ORDEN DEL DIA",
    "DECAIDA DEL ORDEN DEL DIA",
    "SIGUIENTE ORDEN DEL DIA",
    "PALABRAS DE LA PRESIDENCIA",
    "BLOQUE ",
    "DICTAMENES DE LA COMISION",
    "DICTAMENES DE COMISIONES",
    "DEBATE SOBRE CONTROL DE SUBSIDIARIEDAD",
    "DEBATES DE TOTALIDAD DE INICIATIVAS LEGISLATIVAS",
    "CONVALIDACION O DEROGACION DE REALES DECRETOS LEYES",
    "AVOCACION DE INICIATIVAS LEGISLATIVAS",
    "RESOLUCION DE LA PRESIDENCIA DEL CONGRESO DE LOS DIPUTADOS",
    "EMITIR DICTAMEN",
    "DELEGACION EN LA MESA DE LA COMISION",
    "CELEBRACION, EN SU CASO, DE LAS COMPARECENCIAS",
    "PREGUNTA SOBRE ",
    "INFORMAR SOBRE LOS SIGUIENTES EXTREMOS",
    "ARTICULO 44 DEL REGLAMENTO",
    "SOLICITUD DE LOS GRUPOS PARLAMENTARIOS",
    "PROPUESTA DE ",
    "VOTACION",
    "AUTOR:",
)
_AGENDA_ASCII_EXACT = frozenset(
    {
        (
            "COMPARECENCIA ANTE LAS COMISIONES DE INVESTIGACION DEL CONGRESO Y "
            "DEL SENADO O DE AMBAS CAMARAS."
        ),
        "COMPARECENCIA ANTE LAS COMISIONES DE INVESTIGACION DEL CONGRESO Y DEL SENADO.",
    }
)
_AGENDA_ASCII_EXTRA_PREFIXES = (
    "INCLUSION EN EL ORDEN DEL DIA",
    "EXCLUSION DEL ORDEN DEL DIA",
    "DECAIDA DEL ORDEN DEL DIA",
    "SIGUIENTE ORDEN DEL DIA",
    "PALABRAS DE LA PRESIDENCIA",
    "BLOQUE ",
    "DICTAMENES DE LA COMISION",
    "DICTAMENES DE COMISIONES",
    "DEBATE SOBRE CONTROL DE SUBSIDIARIEDAD",
    "DEBATES DE TOTALIDAD DE INICIATIVAS LEGISLATIVAS",
    "CONVALIDACION O DEROGACION DE REALES DECRETOS LEYES",
    "AVOCACION DE INICIATIVAS LEGISLATIVAS",
    "RESOLUCION DE LA PRESIDENCIA DEL CONGRESO DE LOS DIPUTADOS",
    "EMITIR DICTAMEN",
    "DELEGACION EN LA MESA DE LA COMISION",
    "CELEBRACION, EN SU CASO, DE LAS COMPARECENCIAS",
    "PREGUNTA SOBRE ",
    "INFORMAR SOBRE LOS SIGUIENTES EXTREMOS",
    "ARTICULO 44 DEL REGLAMENTO",
    "SOLICITUD DE LOS GRUPOS PARLAMENTARIOS",
    "PROPUESTA DE ",
    "VOTACION",
    "AUTOR:",
    "CELEBRACION DE LAS SIGUIENTES COMPARECENCIAS",
    "DECLARACION INSTITUCIONAL",
    "SOLICITUD DE PRORROGA DE SUBCOMISIONES",
    "RATIFICACION DE LA PONENCIA DESIGNADA",
    "RATIFICACION DEL ACUERDO DE LA MESA",
    "DEBATE Y VOTACION DE LAS PROPUESTAS DE RESOLUCION",
    "MINUTO DE SILENCIO",
    "ACUERDO DEL GOBIERNO POR EL QUE",
    "QUE FORMULA ",
    "POPULAR EN EL CONGRESO, QUE FORMULA",
    "PARLAMENTARIO ",
    "DEL DIPUTADO ",
    "DE LA DIPUTADA ",
    "DEL GRUPO PARLAMENTARIO ",
    "A PETICION ",
    "PARA INFORMAR ",
    "INTERNACIONALES",
    "RATIFICACION ",
    "POR ACUERDO DE ",
)
_AGENDA_ITEM_PREFIXES = (
    "DEL DIPUTADO ",
    "DE LA DIPUTADA ",
    "DEL GRUPO PARLAMENTARIO ",
    "A PETICION ",
    "PARA INFORMAR ",
    "PROPUESTA DE ",
    "INFORME DE ",
    "QUE FORMULA ",
    "PRESENTADA POR ",
    "RELATIVA ",
)


def _looks_like_agenda_heading(line: str) -> bool:
    """Conservatively identify a known standalone parliamentary section title."""

    value = normalize_text(line.strip())
    if not value or value.startswith(("\u2022", "*")):
        return False
    comparison_value = (
        value.lstrip("\u2014\u2013- ") if value.startswith(("\u2014", "\u2013", "-")) else value
    )
    folded = _agenda_fold(comparison_value)
    if folded in _AGENDA_ASCII_EXACT:
        return True
    if folded.startswith(("AUTOR:", "NUMERO DE EXPEDIENTE")):
        return True
    if value.startswith(("\u2014", "\u2013", "-")) and any(
        folded.startswith(prefix) for prefix in _AGENDA_ITEM_PREFIXES
    ):
        return True
    if comparison_value != comparison_value.upper():
        return False
    return any(
        folded.startswith(prefix)
        for prefix in (*_AGENDA_ASCII_PREFIXES, *_AGENDA_ASCII_EXTRA_PREFIXES)
    )


PARENTHETICAL_INTERRUPTION_PATTERN = re.compile(
    r"\((?P<heading>(?:El|La)\s+"
    r"(?:señor|señora|senor|senora|seÃ±or|seÃ±ora)\s+"
    r"(?P<name>[^:()\n]{3,160})):\s*(?P<body>[^()]{1,500}?)\)",
    re.I,
)

NONVERBAL_QUESTION_RESPONSE_PATTERN = re.compile(
    r"(?P<label>Se\u00f1or(?P<gender>a)?\s+"
    r"(?P<name>[A-Z\u00c0-\u00d6\u00d8-\u00de][^?().,;:\n]{1,90}?))"
    r",(?:(?!\b[Ss]e\u00f1or(?:a)?\s+"
    r"[A-Z\u00c0-\u00d6\u00d8-\u00de])[^?:]){0,520}?[?.]\s*"
    r"\((?P<body>Asentimiento|Denegaci(?:\u00f3n|ones))"
    r"(?P<tail>[^)]{0,180})\)",
    re.DOTALL,
)
NONVERBAL_NAMED_GESTURE_PATTERN = re.compile(
    r"\((?P<label>El\s+se\u00f1or\s+"
    r"(?P<name>[A-Z\u00c0-\u00d6\u00d8-\u00de][^():\n]{1,90}?))\s+"
    r"(?P<body>hace\s+signos\s+negativos)\)",
    re.I,
)
NONVERBAL_NAMED_FLOOR_RESPONSE_PATTERN = re.compile(
    r"\btiene\s+la\s+palabra\s+"
    r"(?P<label>(?:el|la)\s+se\u00f1or(?:a)?\s+"
    r"(?P<name>[A-Z\u00c0-\u00d6\u00d8-\u00de][^?().,;:\n]{1,90}?))\.\s*"
    r"\((?P<body>Asentimiento|Denegaci(?:\u00f3n|ones))\)",
    re.I,
)


@dataclass(frozen=True)
class SpeechBlock:
    speaker_heading: str
    normalized_speaker: str
    text: str
    ordinal: int
    page_hint: int | None = None
    page_end: int | None = None
    raw_text: str | None = None
    source_char_start: int | None = None
    source_char_end: int | None = None
    source_map_json: str = "[]"
    removed_spans_json: str = "[]"
    turn_kind: str = "formal_turn"
    parser_version: str = INTERVENTION_SPEECH_BLOCK_PARSER_VERSION


@dataclass(frozen=True)
class InterventionMatch:
    text_fragment: str | None
    confidence: float
    reason: str


def extract_id_texto(url: str) -> str | None:
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("_intervenciones_id_texto", [])
    if not values:
        return None
    return values[0].strip("()")


def canonical_intervention_text_url(url: str | None) -> str | None:
    canonical = canonical_official_resource_url(url)
    if not canonical:
        return None
    clean_url, _ = urldefrag(canonical)
    parsed = urlparse(clean_url)
    if parsed.path in {"", "/"} and not parsed.query:
        return None
    return clean_url


def canonical_intervention_pdf_url(url: str | None) -> str | None:
    canonical = canonical_official_resource_url(url)
    if not canonical:
        return None
    clean_url, _ = urldefrag(canonical)
    filename = unquote(urlparse(clean_url).path.rsplit("/", 1)[-1])
    override = _OFFICIAL_PDF_FILENAME_OVERRIDES.get(filename)
    if override:
        return f"{_OFFICIAL_PDF_BASE}/{override}"
    if not filename.casefold().endswith(".pdf"):
        return None
    return clean_url


def canonical_official_resource_url(url: str | None) -> str | None:
    """Repair official exports that concatenate or repeat absolute URLs.

    Some historical rows contain an origin immediately followed by the real
    static URL, while others contain two space-separated revisions. The final
    absolute URL is the usable resource in both observed forms.
    """

    if not url or not str(url).strip():
        return None
    value = str(url).strip()
    starts = [match.start() for match in re.finditer(r"https?://", value, re.I)]
    value = value[starts[-1] :] if starts else urljoin("https://www.congreso.es", value)
    value = re.sub(
        r"^https://www\.congreso\.es:443(?=/)",
        "https://www.congreso.es",
        value,
        flags=re.I,
    )
    clean_url, _ = urldefrag(value)
    return clean_url or None


def intervention_document_id_from_urls(
    *,
    full_text_url: str | None,
    pdf_url: str | None,
) -> str | None:
    text_id = extract_id_texto(full_text_url or "")
    if text_id:
        return text_id
    if pdf_url:
        parsed_path = urlparse(pdf_url).path
        filename = parsed_path.rsplit("/", 1)[-1]
        stem = re.sub(r"\.pdf$", "", filename, flags=re.I)
        if not stem:
            return None
        legislature_match = re.search(r"/L(?P<number>\d+)/", parsed_path, re.I)
        if not legislature_match:
            return stem
        legislature = legislature_match.group("number")
        if legislature == "0" or re.match(
            rf"^DSCD-{re.escape(legislature)}(?:-|$)",
            stem,
            re.I,
        ):
            return stem
        return f"L{legislature}-{stem}"
    return None


def page_hint_from_url(url: str | None) -> int | None:
    if not url:
        return None
    decoded = unquote(url)
    match = re.search(r"(?:#page=|Página|Pagina)(\d+)", decoded, re.I)
    return int(match.group(1)) if match else None


def normalize_speaker_name(value: str | None) -> str:
    if not value:
        return ""
    # Native PDFs occasionally encode the visual separator between two surnames
    # as a zero-width format character (for example ``López\u200cSánchez``).
    # Dropping it fuses both identity tokens and makes an otherwise exact official
    # speaker impossible to retrieve.  Preserve the observed text elsewhere, but
    # treat format controls as token boundaries in this comparison-only key.
    value = "".join(" " if unicodedata.category(char) == "Cf" else char for char in value)
    # A few native PDFs omit a visible separator at a lowercase-to-uppercase
    # boundary (for example ``Garc\u00edaPage`` for ``Garc\u00eda-Page``). Split only
    # that unambiguous camel boundary in the comparison key; source text and
    # offsets remain untouched.
    value = re.sub(
        r"(?<=[a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]{4})"
        r"(?=[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1])",
        " ",
        value,
    )
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = re.sub(r"(?<=\w)[-‐‑‒–—](?=\w)", " ", value)
    value = re.sub(r"\([^)]*\)", "", value)
    value = normalize_text(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.replace(",", " ")
    value = re.sub(r"[^a-z ]+", "", value)
    value = normalize_text(value)
    # The official CO-304 PDF consistently prints this surname as ``OGU``
    # while the index and the registered name are ``OGOU``.  This is a
    # comparison-only alias; the literal heading and source map remain intact.
    if value == "ogu i corbi":
        return "ogou i corbi"
    return value


def html_to_visible_text(html: str) -> str:
    return parse_visible_html(html).visible_text


def split_speech_blocks(
    document_text: str,
    *,
    page_spans: list[tuple[int, int, int]] | None = None,
) -> list[SpeechBlock]:
    blocks: list[SpeechBlock] = []
    current_heading: str | None = None
    current_speaker = ""
    current_page: int | None = None
    current_block_page: int | None = None
    current_block_page_end: int | None = None
    current_lines: list[str] = []
    current_line_spans: list[tuple[int, int]] = []
    current_heading_start: int | None = None
    current_source_end: int | None = None
    current_removed_spans: list[dict[str, Any]] = []
    search_cursor = 0
    pending_floor_continuation: dict[str, Any] | None = None
    pending_agenda_continuation = False

    def flush() -> None:
        if current_heading and current_lines:
            content_text = "\n".join(current_lines).strip()
            content_map: list[dict[str, Any]] = []
            content_cursor = 0
            for index, (line_text, source_span) in enumerate(
                zip(current_lines, current_line_spans, strict=True)
            ):
                if index:
                    previous_source_end = current_line_spans[index - 1][1]
                    source_gap = document_text[previous_source_end : source_span[0]]
                    separator_has_literal_source = bool(source_gap) and not source_gap.strip()
                    content_map.append(
                        {
                            "content_start": content_cursor,
                            "content_end": content_cursor + 1,
                            "source_start": (
                                previous_source_end
                                if separator_has_literal_source
                                else source_span[0]
                            ),
                            "source_end": (
                                source_span[0] if separator_has_literal_source else source_span[0]
                            ),
                            "kind": (
                                "line_separator_projection"
                                if separator_has_literal_source
                                else "synthetic_line_separator"
                            ),
                        }
                    )
                    content_cursor += 1
                content_map.extend(
                    _character_exact_projection_map(
                        source_text=document_text[source_span[0] : source_span[1]],
                        content_text=line_text,
                        content_start=content_cursor,
                        source_start=source_span[0],
                    )
                )
                content_cursor += len(line_text)
            source_start = current_heading_start
            source_end = current_source_end
            blocks.append(
                SpeechBlock(
                    speaker_heading=current_heading,
                    normalized_speaker=current_speaker,
                    text=content_text,
                    ordinal=len(blocks),
                    page_hint=current_block_page,
                    page_end=current_block_page_end or current_block_page,
                    raw_text=(
                        document_text[source_start:source_end]
                        if source_start is not None and source_end is not None
                        else None
                    ),
                    source_char_start=source_start,
                    source_char_end=source_end,
                    source_map_json=json.dumps(
                        content_map,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    removed_spans_json=json.dumps(
                        current_removed_spans,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

    for line in _iter_speech_lines(document_text):
        if not line:
            continue
        line_search_start = search_cursor
        located_line = _locate_normalized_source_text(
            document_text,
            line,
            start=search_cursor,
        )
        if located_line is None:
            next_content = re.search(r"\S", document_text[search_cursor:])
            line_start = (
                search_cursor + next_content.start() if next_content is not None else search_cursor
            )
            line_end = min(len(document_text), line_start + len(line))
        else:
            line_start, line_end = located_line
        search_cursor = max(search_cursor, line_end)
        page_from_span = _page_for_offset(page_spans, line_start)
        if page_from_span is not None:
            current_page = page_from_span
        page_matches = list(PAGE_PATTERN.finditer(line))
        is_page_marker = bool(page_matches) and bool(
            re.fullmatch(r"\s*P(?:á|a|Ã¡)gina\s+\d+\s*", line, re.I)
        )
        if is_page_marker:
            next_page = int(page_matches[-1].group("page"))
            if current_heading and current_block_page is None and next_page > 1:
                current_block_page = next_page - 1
            current_page = next_page
            if current_heading:
                current_removed_spans.append(
                    {
                        "source_start": line_start,
                        "source_end": line_end,
                        "kind": "page_marker",
                    }
                )
                current_source_end = line_end
            continue
        heading_match = None
        for pattern in SPEAKER_PATTERNS:
            candidate_match = pattern.match(line)
            if candidate_match and _plausible_speaker_match(candidate_match):
                heading_match = candidate_match
                break
        if heading_match is None:
            heading_match = _generic_ocr_speaker_match(line)
        floor_continuation_match = NAMED_FLOOR_CONTINUATION_PATTERN.match(line)
        if heading_match:
            # Some official commission PDFs omit the second speaker heading after
            # a chair's floor call and start the answer directly on the next
            # physical line (for example ``Tiene la palabra la señora Vázquez.``
            # followed by the question).  Keep the chair's call as its own turn,
            # then project the following literal lines into a synthetic metadata
            # heading.  No text is invented: the body and source map point only
            # to the observed following line, while the call is retained as
            # reversible context.
            if floor_continuation_match:
                pending_floor_continuation = {
                    "article": floor_continuation_match.group("target_article"),
                    "honorific": floor_continuation_match.group("target_honorific"),
                    "name": normalize_text(floor_continuation_match.group("name")),
                    "source_start": line_start,
                    "source_end": line_end,
                    "page": current_page,
                }
            else:
                pending_floor_continuation = None
            flush()
            current_heading = _speaker_heading_text(line, heading_match)
            name = _speaker_name_from_heading(heading_match, current_heading)
            current_speaker = normalize_speaker_name(name)
            current_block_page = current_page
            current_block_page_end = None
            current_lines = []
            current_line_spans = []
            current_heading_start = line_start
            current_source_end = line_end
            current_removed_spans = []
            body = heading_match.groupdict().get("body")
            if body:
                current_lines.append(body)
                body_group_start = heading_match.start("body")
                heading_prefix = line[:body_group_start]
                heading_prefix_span = _locate_normalized_source_text(
                    document_text,
                    heading_prefix,
                    start=line_start,
                )
                body_search_start = (
                    heading_prefix_span[1]
                    if heading_prefix_span is not None and heading_prefix_span[0] == line_start
                    else _heading_body_source_search_start(
                        document_text,
                        line_start=line_start,
                        line_search_start=line_search_start,
                    )
                )
                body_span = _locate_normalized_source_text(
                    document_text,
                    body,
                    start=body_search_start,
                )
                if body_span is None:
                    body_offset = line.rfind(body)
                    body_span = (line_start + max(0, body_offset), line_end)
                current_line_spans.append(body_span)
                if line_start < body_span[0]:
                    current_removed_spans.append(
                        {
                            "source_start": line_start,
                            "source_end": body_span[0],
                            "kind": "speaker_heading_wrapper",
                        }
                    )
                current_block_page_end = current_page or current_block_page
                current_source_end = max(current_source_end or 0, body_span[1])
                search_cursor = max(search_cursor, body_span[1])
            else:
                current_removed_spans.append(
                    {
                        "source_start": line_start,
                        "source_end": line_end,
                        "kind": "speaker_heading_wrapper",
                    }
                )
            pending_agenda_continuation = False
        elif current_heading and _looks_like_agenda_heading(line):
            current_removed_spans.append(
                {
                    "source_start": line_start,
                    "source_end": line_end,
                    "kind": "agenda_heading",
                }
            )
            current_source_end = line_end
            pending_agenda_continuation = not bool(re.search(r"[.:;!?)]\s*$", line.strip()))
        elif current_heading and pending_agenda_continuation:
            continuation_value = normalize_text(line.strip())
            if continuation_value and continuation_value == continuation_value.upper():
                current_removed_spans.append(
                    {
                        "source_start": line_start,
                        "source_end": line_end,
                        "kind": "agenda_heading",
                    }
                )
                current_source_end = line_end
                pending_agenda_continuation = False
            else:
                pending_agenda_continuation = False
                current_lines.append(line)
                current_line_spans.append((line_start, line_end))
                current_block_page_end = current_page or current_block_page
                current_source_end = line_end
        elif pending_floor_continuation is not None:
            continuation = pending_floor_continuation
            pending_floor_continuation = None
            flush()
            current_heading = (
                f"{continuation['article']} {continuation['honorific']} "
                f"{continuation['name']} [floor continuation]"
            )
            current_speaker = normalize_speaker_name(continuation["name"])
            current_block_page = continuation["page"] or current_page
            current_block_page_end = current_page or current_block_page
            current_lines = [line]
            current_line_spans = [(line_start, line_end)]
            current_heading_start = continuation["source_start"]
            current_source_end = line_end
            current_removed_spans = [
                {
                    "source_start": continuation["source_start"],
                    "source_end": continuation["source_end"],
                    "kind": "floor_call_context",
                }
            ]
            pending_agenda_continuation = False
        elif current_heading:
            current_lines.append(line)
            current_line_spans.append((line_start, line_end))
            current_block_page_end = current_page or current_block_page
            current_source_end = line_end
            pending_agenda_continuation = False
        if page_match := page_matches:
            next_page = int(page_match[-1].group("page"))
            if current_heading and current_block_page is None and next_page > 1:
                current_block_page = next_page - 1
            current_page = next_page
    flush()
    blocks.extend(
        _parenthetical_interruption_blocks(
            document_text,
            page_spans=page_spans,
        )
    )
    blocks.extend(
        _nonverbal_response_blocks(
            document_text,
            page_spans=page_spans,
        )
    )
    blocks.sort(
        key=lambda block: (
            block.source_char_start if block.source_char_start is not None else len(document_text),
            0 if block.turn_kind == "formal_turn" else 1,
        )
    )
    return [replace(block, ordinal=ordinal) for ordinal, block in enumerate(blocks)]


def _parenthetical_interruption_blocks(
    document_text: str,
    *,
    page_spans: list[tuple[int, int, int]] | None,
) -> list[SpeechBlock]:
    """Extract bounded single-speaker interruptions without losing their source."""

    interruptions: list[SpeechBlock] = []
    for match in PARENTHETICAL_INTERRUPTION_PATTERN.finditer(document_text):
        body = match.group("body")
        if INLINE_SPEAKER_PATTERN.search(body) or OCR_INLINE_SPEAKER_PATTERN.search(body):
            continue
        if body.count("\n") > 4:
            continue
        content_text = normalize_text(body)
        if not content_text:
            continue
        leading = len(body) - len(body.lstrip())
        trailing = len(body.rstrip())
        body_start = match.start("body") + leading
        body_end = match.start("body") + trailing
        heading = normalize_text(match.group("heading"))
        normalized_speaker = normalize_speaker_name(match.group("name"))
        if re.search(
            r"\b(?:dijo|dijeron|manifesto|manifestaron|respondio|respondieron|"
            r"pregunto|preguntaron|anadio|anadieron|exclamo|exclamaron)$",
            normalized_speaker,
        ):
            continue
        removed_spans = [
            {
                "source_start": match.start(),
                "source_end": body_start,
                "kind": "parenthetical_interruption_heading_wrapper",
            },
            {
                "source_start": body_end,
                "source_end": match.end(),
                "kind": "parenthetical_interruption_closing_wrapper",
            },
        ]
        page, page_end = _page_range_for_offsets(
            page_spans,
            match.start(),
            match.end(),
        )
        interruptions.append(
            SpeechBlock(
                speaker_heading=heading,
                normalized_speaker=normalized_speaker,
                text=content_text,
                ordinal=-1,
                page_hint=page,
                page_end=page_end,
                raw_text=document_text[match.start() : match.end()],
                source_char_start=match.start(),
                source_char_end=match.end(),
                source_map_json=json.dumps(
                    [
                        {
                            "content_start": 0,
                            "content_end": len(content_text),
                            "source_start": body_start,
                            "source_end": body_end,
                            "kind": "parenthetical_interruption_body_projection",
                        }
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                removed_spans_json=json.dumps(
                    removed_spans,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                turn_kind="parenthetical_interruption",
            )
        )
    return interruptions


def _nonverbal_response_blocks(
    document_text: str,
    *,
    page_spans: list[tuple[int, int, int]] | None,
) -> list[SpeechBlock]:
    """Project explicit named assent, denial and gesture evidence as turns."""

    responses: list[SpeechBlock] = []
    for pattern, projection_kind in (
        (NONVERBAL_QUESTION_RESPONSE_PATTERN, "named_question_nonverbal_response"),
        (NONVERBAL_NAMED_GESTURE_PATTERN, "named_nonverbal_gesture"),
        (
            NONVERBAL_NAMED_FLOOR_RESPONSE_PATTERN,
            "named_floor_response_nonverbal_response",
        ),
    ):
        for match in pattern.finditer(document_text):
            body = match.group("body")
            content_text = normalize_text(body)
            if not content_text:
                continue
            body_start = match.start("body")
            body_end = match.end("body")
            source_start = match.start()
            source_end = match.end()
            label = normalize_text(match.group("label"))
            normalized_speaker = normalize_speaker_name(match.group("name"))
            if not label or not normalized_speaker:
                continue
            removed_spans = []
            if source_start < body_start:
                removed_spans.append(
                    {
                        "source_start": source_start,
                        "source_end": body_start,
                        "kind": f"{projection_kind}_context",
                    }
                )
            if body_end < source_end:
                removed_spans.append(
                    {
                        "source_start": body_end,
                        "source_end": source_end,
                        "kind": f"{projection_kind}_wrapper",
                    }
                )
            page, page_end = _page_range_for_offsets(
                page_spans,
                source_start,
                source_end,
            )
            responses.append(
                SpeechBlock(
                    speaker_heading=label,
                    normalized_speaker=normalized_speaker,
                    text=content_text,
                    ordinal=-1,
                    page_hint=page,
                    page_end=page_end,
                    raw_text=document_text[source_start:source_end],
                    source_char_start=source_start,
                    source_char_end=source_end,
                    source_map_json=json.dumps(
                        [
                            {
                                "content_start": 0,
                                "content_end": len(content_text),
                                "source_start": body_start,
                                "source_end": body_end,
                                "kind": projection_kind,
                            }
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    removed_spans_json=json.dumps(
                        removed_spans,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    turn_kind="nonverbal_response",
                )
            )
    return responses


def _locate_normalized_source_text(
    source: str,
    normalized_text: str,
    *,
    start: int,
) -> tuple[int, int] | None:
    tokens = re.split(r"\s+", str(normalized_text or "").strip())
    if not tokens or not tokens[0]:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source[start:])
    if match is None:
        return None
    return start + match.start(), start + match.end()


def _heading_body_source_search_start(
    document_text: str,
    *,
    line_start: int,
    line_search_start: int,
) -> int:
    """Anchor repaired OCR headings to their literal source colon when possible."""

    bounded_start = max(0, min(line_start, line_search_start))
    bounded_end = min(len(document_text), max(line_start, line_search_start) + 1_000)
    colon = document_text.find(":", bounded_start, bounded_end)
    return colon + 1 if colon >= 0 else bounded_start


def _character_exact_projection_map(
    *,
    source_text: str,
    content_text: str,
    content_start: int,
    source_start: int,
) -> list[dict[str, Any]]:
    """Map normalized content characters to exact source spans without ambiguity."""

    if source_text == content_text:
        return [
            {
                "content_start": content_start,
                "content_end": content_start + len(content_text),
                "source_start": source_start,
                "source_end": source_start + len(source_text),
                "kind": "line_identity_projection",
            }
        ]
    if normalize_text(source_text) != content_text:
        raise ValueError("Source slice does not normalize to parsed content")

    items: list[dict[str, Any]] = []
    source_index = 0
    content_index = 0
    while source_index < len(source_text):
        if source_text[source_index].isspace():
            whitespace_start = source_index
            while source_index < len(source_text) and source_text[source_index].isspace():
                source_index += 1
            if content_index < len(content_text) and content_text[content_index] == " ":
                items.append(
                    {
                        "content_start": content_start + content_index,
                        "content_end": content_start + content_index + 1,
                        "source_start": source_start + whitespace_start,
                        "source_end": source_start + source_index,
                        "kind": "whitespace_collapse_projection",
                    }
                )
                content_index += 1
            continue

        identity_source_start = source_index
        identity_content_start = content_index
        while source_index < len(source_text) and not source_text[source_index].isspace():
            if (
                content_index >= len(content_text)
                or source_text[source_index] != content_text[content_index]
            ):
                raise ValueError("Non-whitespace source character changed during parsing")
            source_index += 1
            content_index += 1
        items.append(
            {
                "content_start": content_start + identity_content_start,
                "content_end": content_start + content_index,
                "source_start": source_start + identity_source_start,
                "source_end": source_start + source_index,
                "kind": "identity_projection",
            }
        )
    if content_index != len(content_text):
        raise ValueError("Source projection did not account for every content character")
    return items


def _page_for_offset(
    page_spans: list[tuple[int, int, int]] | None,
    offset: int,
) -> int | None:
    if not page_spans:
        return None
    for page_number, start, end in page_spans:
        if start <= offset < end:
            return page_number
    return page_spans[-1][0] if offset == page_spans[-1][2] else None


def _page_range_for_offsets(
    page_spans: list[tuple[int, int, int]] | None,
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    """Project a non-empty half-open source span onto its first and last pages."""

    page_start = _page_for_offset(page_spans, start)
    last_source_offset = max(start, end - 1)
    page_end = _page_for_offset(page_spans, last_source_offset)
    return page_start, page_end if page_end is not None else page_start


def has_embedded_speaker_heading(text: str) -> bool:
    """Fail closed on parsed or still-unknown OCR speaker headings inside a block."""

    parsed_blocks = split_speech_blocks(text)
    return bool(
        any(block.turn_kind == "formal_turn" for block in parsed_blocks)
        or OCR_EMBEDDED_SPEAKER_CANDIDATE.search(text)
    )


def _iter_speech_lines(document_text: str) -> Iterable[str]:
    document_text = OCR_COLLAPSED_SPEAKER_PREFIX.sub(
        _repair_collapsed_speaker_prefix,
        document_text,
    )
    raw_lines = _split_outside_parenthetical_annotations(document_text).splitlines()
    index = 0
    while index < len(raw_lines):
        line = normalize_text(raw_lines[index])
        if not line:
            index += 1
            continue
        should_join_heading = WRAPPED_SPEAKER_START_PATTERN.match(
            line
        ) or NARRATIVE_SPEAKER_START_PATTERN.search(line)
        next_line = normalize_text(raw_lines[index + 1]) if index + 1 < len(raw_lines) else ""
        corporate_role_continuation = bool(
            WRAPPED_SPEAKER_START_PATTERN.match(line)
            and re.search(
                r"\bS\.(?:A|L)\.(?:U\.)?(?:,\s*S\.M\.E\.)?\s*$",
                line,
                re.I,
            )
            and (next_line.startswith("(") or next_line[:1].isupper())
        )
        # A few official diaries delimit a role heading with a full stop instead
        # of a colon (for example ``El señor VICEPRESIDENTE (Name). Por ...``).
        # That line is already a complete, valid heading/body pair.  Joining it
        # forward until the next colon would swallow the following real speaker.
        if _matches_speaker_heading(line):
            yield line
            index += 1
            continue
        if should_join_heading and (
            not re.search(r"[.!?;]\s*$", line) or corporate_role_continuation
        ):
            combined = line
            lookahead_index = index + 1
            while ":" not in combined and lookahead_index < min(index + 10, len(raw_lines)):
                next_line = normalize_text(raw_lines[lookahead_index])
                if not next_line:
                    break
                combined = normalize_text(f"{combined} {next_line}")
                if len(combined) > 1_200:
                    break
                lookahead_index += 1
            combined = re.sub(r"(?<=\w)-\s+(?=\w)", "", combined)
            if _matches_speaker_heading(combined):
                yield combined
                index = lookahead_index
                continue
        yield line
        index += 1


def _split_outside_parenthetical_annotations(document_text: str) -> str:
    """Expose inline turn headings without promoting parenthetical interruptions.

    Diario de Sesiones records shouts and stage directions inside parentheses, and
    those annotations frequently contain text shaped exactly like a turn heading.
    They may span physical PDF lines.  Flattening only the line breaks inside an
    open parenthesis keeps the annotation attached to the active turn, while
    inserting separators for heading candidates at depth zero still recovers real
    same-line speaker changes.  Whitespace-only changes remain reversible through
    the source projection built by ``split_speech_blocks``.
    """

    candidate_offsets = sorted(
        {
            match.start()
            for pattern in (INLINE_SPEAKER_PATTERN, OCR_INLINE_SPEAKER_PATTERN)
            for match in pattern.finditer(document_text)
        }
    )
    candidate_offset_set = set(candidate_offsets)
    annotation_intervals = _parenthetical_speaker_annotation_intervals(
        document_text,
        candidate_offsets,
    )
    output: list[str] = []
    interval_index = 0
    for offset, char in enumerate(document_text):
        while (
            interval_index < len(annotation_intervals)
            and offset >= annotation_intervals[interval_index][1]
        ):
            interval_index += 1
        in_annotation = (
            interval_index < len(annotation_intervals)
            and annotation_intervals[interval_index][0]
            <= offset
            < annotation_intervals[interval_index][1]
        )
        if offset in candidate_offset_set and not in_annotation and output and output[-1] != "\n":
            output.append("\n")
        if char == "\n" and in_annotation:
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _parenthetical_speaker_annotation_intervals(
    document_text: str,
    candidate_offsets: list[int],
    *,
    max_characters: int = 2_000,
    max_line_breaks: int = 12,
) -> list[tuple[int, int]]:
    """Return only bounded, balanced parentheses containing speaker-like text."""

    stack: list[int] = []
    intervals: list[tuple[int, int]] = []
    for offset, char in enumerate(document_text):
        if char == "(":
            stack.append(offset)
            continue
        if char != ")" or not stack:
            continue
        start = stack.pop()
        end = offset + 1
        if end - start > max_characters:
            continue
        if document_text.count("\n", start, end) > max_line_breaks:
            continue
        candidate_index = bisect_left(candidate_offsets, start)
        if candidate_index < len(candidate_offsets) and candidate_offsets[candidate_index] < end:
            intervals.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _repair_collapsed_speaker_prefix(match: re.Match[str]) -> str:
    title = normalize_speaker_name(match.group("title"))
    similarity = max(
        SequenceMatcher(None, title, candidate).ratio()
        for candidate in ("senor", "senora", "senior", "not")
    )
    if similarity < 0.8:
        return match.group(0)
    raw_prefix = match.group(0)
    article = match.group("article")
    title = match.group("title")
    # This repair exists only to recover missing layout separators in OCR. Keep
    # every observed glyph (including ñ, accents and OCR errors) so the displayed
    # heading is never silently rewritten into a plausible but unsupported form.
    if re.fullmatch(
        rf"{re.escape(article)}\s+{re.escape(title)}\s+",
        raw_prefix,
    ):
        return raw_prefix
    return f"{article} {title} "


def _matches_speaker_heading(line: str) -> bool:
    return any(
        match and _plausible_speaker_match(match)
        for pattern in SPEAKER_PATTERNS
        if (match := pattern.match(line)) is not None
    ) or bool(_generic_ocr_speaker_match(line))


def _plausible_speaker_match(match: re.Match[str]) -> bool:
    name = match.groupdict().get("name")
    matched_text = match.group(0)
    stripped_text = matched_text.lstrip()
    if stripped_text.startswith(("el ", "la ")):
        # Native Legislature XV headings begin with the capitalized article.
        # Lower-case forms occur inside narrative speech (including Catalan
        # quotations such as ``el senyor X ho deia:``), not at a turn boundary.
        return False
    if not name:
        heading = matched_text.split(":", 1)[0]
        heading_words = re.findall(r"[^\W\d_]+", heading, flags=re.UNICODE)
        if "?" in heading or "¿" in heading or len(heading_words) > 24:
            return False
        return not re.search(
            r"\b(?:hace|hizo|hacía|dice|decía|dijo|afirma|afirmó|señala|"
            r"señaló|pregunta|preguntó|pregunto|parece|resulta|recuerda|"
            r"recordó|exige|exigió|garantiza|garantizó)\b",
            heading,
            re.I,
        )
    parenthetical_values = re.findall(r"\(([^()]*)\)", name)
    if parenthetical_values and normalize_speaker_name(parenthetical_values[-1]).startswith(
        "numero de expediente"
    ):
        # An agenda title following narrative prose can be joined forward until
        # the next real colon. Its final ``(Número de expediente ...)`` is metadata,
        # not a personal-name parenthetical and must never promote the narrative
        # ``La señora X ya nos ha anunciado...`` to a turn heading.
        return False
    if re.match(
        r"^\((?:el|la)\s+se\S{1,6}r(?:a)?\b",
        stripped_text,
        re.I,
    ):
        # In native XV text an opening parenthesis denotes an interruption or
        # stage direction even when its closing glyph is missing or far away.
        return False
    colon_offset = matched_text.find(":")
    if (
        matched_text.lstrip().startswith("(")
        and colon_offset >= 0
        and matched_text.find(")", colon_offset + 1) >= 0
    ):
        # This is a closed stage direction or shouted interruption, not a new
        # top-level turn.  It remains verbatim in the active speaker's content.
        return False
    folded_name = normalize_speaker_name(name)
    if re.search(
        r"\b(?:dijo|dijeron|manifesto|manifestaron|respondio|respondieron|"
        r"pregunto|preguntaron|anadio|anadieron|exclamo|exclamaron)$",
        folded_name,
    ):
        # Roll-call prose such as ``El señor X dijo: No`` describes a vote; it
        # is not a parliamentary turn heading.
        return False
    words = re.findall(r"[^\W\d_]+", name, flags=re.UNICODE)
    if "?" in name or "¿" in name:
        return False
    if re.search(r"\([^)]+\)", name):
        prefix_words = re.findall(
            r"[^\W\d_]+",
            name.split("(", 1)[0],
            flags=re.UNICODE,
        )
        if len(prefix_words) <= 7:
            return True
        letters = [char for char in name if char.isalpha()]
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters) if letters else 0.0
        return uppercase_ratio >= 0.45
    if len(words) > 24:
        return False
    if len(words) <= 7:
        return True
    letters = [char for char in name if char.isalpha()]
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters) if letters else 0.0
    return uppercase_ratio >= 0.45


def _generic_ocr_speaker_match(line: str) -> re.Match[str] | None:
    match = OCR_GENERIC_SPEAKER_PATTERN.match(line)
    if match is None:
        return None
    folded_title = normalize_speaker_name(match.group("title"))
    title_similarity = max(
        SequenceMatcher(None, folded_title, candidate).ratio()
        for candidate in ("senor", "senora", "senorita", "not")
    )
    name = match.group("name")
    letters = [char for char in name if char.isalpha()]
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters) if letters else 0.0
    return match if title_similarity >= 0.55 and uppercase_ratio >= 0.55 else None


def _speaker_heading_text(line: str, match: re.Match[str]) -> str:
    body_group = match.groupdict().get("body")
    if body_group is None:
        return line.split(":", 1)[0]
    body_start = match.start("body")
    heading_with_delimiter = line[:body_start].rstrip()
    return re.sub(r"(?:[:.]\s*)$", "", heading_with_delimiter).rstrip()


def _speaker_name_from_heading(match: re.Match[str], heading: str) -> str:
    # Corporate abbreviations can contain their own parentheses (for example
    # ``P(A)T``). The personal identity in an official role heading is the final
    # parenthetical, never the first arbitrary one.
    parentheticals = list(re.finditer(r"\((?P<value>[^()]{1,160})\)", heading))
    parenthetical = parentheticals[-1] if parentheticals else None
    if name := match.groupdict().get("name"):
        clean_name = re.sub(r"\([^)]*\)", "", name).strip()
        role_match = re.search(
            r"president[ea]|vicepresident[ea]|secretari[oa]|ministr[oa]|"
            r"director(?:a)?|catedr[aá]tic[oa]|representante|comisari[oa]|"
            r"portavoz|coordinador(?:a)?|responsable|fundador(?:a)?|"
            r"defensor(?:a)?|gobernador(?:a)?|delegad[oa]|consejer[oa]|"
            r"comisionad[oa]|vocal|magistrad[oa]|alcald[ea]|jef[ea]|"
            r"fiscal|rector(?:a)?|embajador(?:a)?|letrad[oa]|"
            r"profesor(?:a)?|decano|decana|vicedecano|vicedecana|"
            r"s[i\u00ed]ndic[oa]|oficial|manager|gerente",
            clean_name,
            re.I,
        )
        if parenthetical and role_match:
            return f"{role_match.group(0)} {_speaker_parenthetical_value(parenthetical)}"
        if parenthetical and _is_generic_speaker_label(clean_name):
            return _speaker_parenthetical_value(parenthetical)
        return clean_name
    role = match.groupdict().get("role") or heading
    if parenthetical:
        return f"{role} {_speaker_parenthetical_value(parenthetical)}"
    return role


def _speaker_parenthetical_value(parenthetical: re.Match[str]) -> str:
    return re.sub(r"(?<=\w)-\s+(?=\w)", " ", parenthetical.group("value"))


def _is_generic_speaker_label(value: str) -> bool:
    clean = re.sub(r"\([^)]*\)", "", value)
    clean = normalize_speaker_name(clean)
    generic_tokens = {
        "candidato",
        "candidata",
        "compareciente",
        "interviniente",
        "ponente",
        "secretario",
        "secretaria",
        "ministro",
        "ministra",
        "invitado",
        "invitada",
    }
    return bool(clean) and set(clean.split()) <= generic_tokens


def speech_block_rows(
    *,
    visible_text: str,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    legislature: str | None = None,
    document_id: str | None = None,
    page_texts: list[str] | None = None,
    page_diagnostics: list[str] | None = None,
    layout_removals: list[str] | None = None,
    source_kind: str | None = None,
    extraction_method: str | None = None,
) -> list[dict[str, Any]]:
    resolved_document_id = (
        document_id
        or intervention_document_id_from_urls(
            full_text_url=source_url,
            pdf_url=None,
        )
        or stable_id(source_url)
    )
    page_spans: list[tuple[int, int, int]] | None = None
    if page_texts is not None:
        rebuilt_parts: list[str] = []
        page_spans = []
        cursor = 0
        for page_number, page_text in enumerate(page_texts, start=1):
            if rebuilt_parts:
                rebuilt_parts.append("\n")
                cursor += 1
            value = str(page_text or "")
            start = cursor
            rebuilt_parts.append(value)
            cursor += len(value)
            page_spans.append((page_number, start, cursor))
        visible_text = "".join(rebuilt_parts)
    presidency_evidence = _document_presidency_evidence(visible_text)
    blocks = split_speech_blocks(visible_text, page_spans=page_spans)
    role_identity_evidence = _document_unique_role_identity_evidence(
        visible_text,
        blocks,
    )
    return [
        {
            "speech_block_id": stable_id(
                resolved_document_id,
                block.ordinal,
                block.speaker_heading,
                source_url,
            ),
            "document_id": resolved_document_id,
            "legislature": legislature,
            "speaker_heading": block.speaker_heading,
            "normalized_speaker": block.normalized_speaker,
            "ordinal": block.ordinal,
            "page_hint": block.page_hint,
            "page_end": block.page_end,
            "text": block.text,
            "raw_text": block.raw_text,
            "content_text": block.text,
            "source_char_start": block.source_char_start,
            "source_char_end": block.source_char_end,
            "source_offset_basis": (
                "pdf_content_page_texts_unicode_codepoints"
                if (source_kind or "").casefold() == "pdf"
                else "official_html_visible_text_unicode_codepoints"
            ),
            "source_map_json": block.source_map_json,
            "removed_spans_json": block.removed_spans_json,
            "turn_kind": block.turn_kind,
            "speaker_identity_evidence_json": _speaker_identity_evidence_for_block(
                block,
                presidency_evidence,
                role_identity_evidence,
            ),
            "document_layout_removals_json": _layout_removals_for_block(
                block,
                layout_removals,
            ),
            "source_page_geometry_json": _source_page_geometry_json(
                block,
                page_diagnostics,
            ),
            "parser_version": block.parser_version,
            "source_kind": source_kind
            or ("pdf" if source_url.lower().endswith(".pdf") else "html"),
            "extraction_method": extraction_method,
            "source_url": source_url,
            "source_file_sha256": source_sha256,
            "source_document_sha256": source_sha256,
            "snapshot_date": snapshot_date,
        }
        for block in blocks
    ]


def _document_presidency_evidence(document_text: str) -> dict[str, Any] | None:
    """Extract the named presiding officer from the official document preamble."""

    lines = document_text.splitlines(keepends=True)
    cursor = 0
    for index, line in enumerate(lines):
        line_start = cursor
        cursor += len(line)
        if not re.match(r"^\s*PRESIDENCIA\s+(?:DE LA|DEL)\b", line, re.I):
            continue
        selected = [line]
        for continuation in lines[index + 1 : index + 5]:
            if re.match(r"^\s*(?:Sesión|SUMARIO)\b", continuation, re.I):
                break
            selected.append(continuation)
        raw = "".join(selected).rstrip("\r\n")
        # XV uses proper Unicode ``D.\u00aa``; legacy decoding can expose
        # ``D.\u00c2\u00aa`` and pypdf emits one literal U+FFFD for this glyph in
        # DSCG-15-SC-1. These are marker variants only: the personal name remains
        # an exact slice of the selected official page text.
        marker = re.search(
            r"\bD\.(?:(?:\u00aa|\u00c2\u00aa|\ufffd))?\s+",
            raw,
            re.I,
        )
        if marker is None:
            return None
        name = normalize_text(raw[marker.end() :])
        name = re.sub(
            r",?\s+(?:VICEPRESIDENTE|VICEPRESIDENTA|PRESIDENTE|PRESIDENTA)\b.*$",
            "",
            name,
            flags=re.I,
        ).strip(" ,.;")
        words = re.findall(r"[^\W\d_]+", name, flags=re.UNICODE)
        if not 2 <= len(words) <= 8:
            return None
        role = "presidenta" if re.search(r"\bSRA\.", raw, re.I) else "presidente"
        source_end = line_start + len(raw)
        return {
            "kind": "official_document_presidency",
            "role": role,
            "speaker_name": name,
            "normalized_speaker": normalize_speaker_name(name),
            "source_char_start": line_start,
            "source_char_end": source_end,
            "source_text": document_text[line_start:source_end],
            "source_offset_basis": "document_content_text_unicode_codepoints",
        }
    return None


def _speaker_identity_evidence_for_block(
    block: SpeechBlock,
    presidency_evidence: dict[str, Any] | None,
    role_identity_evidence: dict[str, dict[str, Any]] | None = None,
) -> str:
    evidence: list[dict[str, Any]] = []
    if (
        presidency_evidence
        and block.turn_kind == "formal_turn"
        and block.normalized_speaker == presidency_evidence.get("role")
    ):
        evidence.append(presidency_evidence)
    role_key = _speaker_role_key_from_heading(block.speaker_heading)
    role_item = (role_identity_evidence or {}).get(role_key)
    if role_item and not _last_heading_parenthetical(block.speaker_heading):
        evidence.append(role_item)
    return json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_ROLE_IDENTITY_TOKENS = {
    "alcalde",
    "alcaldesa",
    "candidato",
    "candidata",
    "comisionada",
    "comisionado",
    "consejera",
    "consejero",
    "defensor",
    "defensora",
    "delegada",
    "delegado",
    "director",
    "directora",
    "embajadora",
    "embajador",
    "fiscal",
    "gerente",
    "gobernadora",
    "gobernador",
    "jefa",
    "jefe",
    "letrada",
    "letrado",
    "magistrada",
    "magistrado",
    "manager",
    "ministra",
    "ministro",
    "oficial",
    "ponente",
    "profesor",
    "profesora",
    "presidenta",
    "presidente",
    "rector",
    "rectora",
    "secretaria",
    "secretario",
    "sindica",
    "sindico",
    "decana",
    "decano",
    "vicedecana",
    "vicedecano",
    "vicepresidenta",
    "vicepresidente",
    "vocal",
}


def _last_heading_parenthetical(heading: str) -> re.Match[str] | None:
    return re.search(r"\((?P<value>[^()]{2,160})\)\s*$", heading)


def _speaker_role_key_from_heading(heading: str) -> str:
    without_name = re.sub(r"\([^()]*\)\s*$", "", str(heading or "")).strip(" .:")
    without_honorific = re.sub(
        r"^(?:El|La)\s+(?:señor|señora|senor|senora|seÃƒÂ±or|seÃƒÂ±ora)\s+",
        "",
        without_name,
        flags=re.I,
    )
    return normalize_speaker_name(without_honorific)


def _document_unique_role_identity_evidence(
    document_text: str,
    blocks: list[SpeechBlock],
) -> dict[str, dict[str, Any]]:
    """Resolve a generic role only when the document names one unique holder.

    Official question time sometimes prints ``PRESIDENTE DEL GOBIERNO (Surname)``
    on the first replies and later shortens it to ``PRESIDENTE DEL GOBIERNO``.  The
    earlier literal heading is sufficient identity evidence only when the exact role
    has one unique parenthetical personal name throughout the selected document.
    """

    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for block in blocks:
        if block.turn_kind != "formal_turn":
            continue
        parenthetical = _last_heading_parenthetical(block.speaker_heading)
        role_key = _speaker_role_key_from_heading(block.speaker_heading)
        if parenthetical is None or not (set(role_key.split()) & _ROLE_IDENTITY_TOKENS):
            continue
        speaker_name = normalize_text(parenthetical.group("value")).strip(" ,.;")
        words = re.findall(r"[^\W\d_]+", speaker_name, flags=re.UNICODE)
        if not 1 <= len(words) <= 8:
            continue
        normalized_speaker = normalize_speaker_name(speaker_name)
        if not normalized_speaker:
            continue
        try:
            removals = json.loads(block.removed_spans_json)
        except (json.JSONDecodeError, TypeError):
            continue
        heading_span = next(
            (
                item
                for item in removals
                if isinstance(item, dict) and item.get("kind") == "speaker_heading_wrapper"
            ),
            None,
        )
        if heading_span is None:
            continue
        source_start = heading_span.get("source_start")
        source_end = heading_span.get("source_end")
        if not (
            isinstance(source_start, int)
            and isinstance(source_end, int)
            and 0 <= source_start < source_end <= len(document_text)
        ):
            continue
        candidates.setdefault(role_key, {})[normalized_speaker] = {
            "kind": "official_document_unique_role_identity",
            "role": role_key,
            "speaker_name": speaker_name,
            "normalized_speaker": normalized_speaker,
            "source_char_start": source_start,
            "source_char_end": source_end,
            "source_text": document_text[source_start:source_end],
            "source_offset_basis": "document_content_text_unicode_codepoints",
        }
    return {
        role: next(iter(identities.values()))
        for role, identities in candidates.items()
        if len(identities) == 1
    }


def _layout_removals_for_block(
    block: SpeechBlock,
    layout_removals: list[str] | None,
) -> str:
    if not layout_removals or block.page_hint is None:
        return "[]"
    page_end = block.page_end if block.page_end is not None else block.page_hint
    selected: list[dict[str, Any]] = []
    for raw in layout_removals:
        try:
            removal = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Invalid PDF layout-removal provenance JSON") from exc
        if not isinstance(removal, dict):
            raise ValueError("PDF layout-removal provenance must be a JSON object")
        page_number = removal.get("page_number")
        if not isinstance(page_number, int):
            raise ValueError("PDF layout-removal provenance lacks page_number")
        if block.page_hint <= page_number <= page_end:
            selected.append(removal)
    selected.sort(
        key=lambda item: (
            int(item.get("page_number") or 0),
            int(item.get("source_start") or 0),
            int(item.get("source_end") or 0),
            str(item.get("kind") or ""),
        )
    )
    return json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_page_geometry_json(
    block: SpeechBlock,
    page_diagnostics: list[str] | None,
) -> str | None:
    if not page_diagnostics or block.page_hint is None:
        return None
    last_page = block.page_end or block.page_hint
    selected: list[dict[str, Any]] = []
    for page_number in range(block.page_hint, last_page + 1):
        if page_number < 1 or page_number > len(page_diagnostics):
            continue
        try:
            diagnostic = json.loads(page_diagnostics[page_number - 1])
        except (json.JSONDecodeError, TypeError):
            continue
        geometry = diagnostic.get("pymupdf_geometry")
        if geometry:
            selected.append({"page_number": page_number, **geometry})
    return (
        json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if selected
        else None
    )


def match_intervention_text(
    *,
    speaker: str,
    blocks: Iterable[SpeechBlock],
    occurrence_index: int = 0,
) -> InterventionMatch:
    wanted = normalize_speaker_name(speaker)
    candidates = [block for block in blocks if _speaker_matches(wanted, block.normalized_speaker)]
    if not candidates:
        return InterventionMatch(text_fragment=None, confidence=0.0, reason="speaker_not_found")
    index = min(occurrence_index, len(candidates) - 1)
    candidate = candidates[index]
    confidence = 0.9 if wanted == candidate.normalized_speaker else 0.72
    return InterventionMatch(
        text_fragment=candidate.text,
        confidence=confidence,
        reason="speaker_heading",
    )


def intervention_id(row: dict[str, Any]) -> str:
    item = normalize_record_keys(row)
    return stable_id(
        _first_present(item, "legislatura"),
        initiative_reference_identity(_first_present(item, "numexpediente", "numeroexpediente")),
        _first_present(item, "sesion", "fecha"),
        _first_present(item, "organo", "nombresesion"),
        _first_present(item, "objetoiniciativa"),
        _first_present(item, "fase"),
        _first_present(item, "tipointervencion", "tipo"),
        _first_present(item, "orador"),
        _first_present(item, "cargoorador"),
        _first_present(item, "iniciointervencion", "horainicio"),
        _first_present(item, "finintervencion", "horafin"),
        item.get("enlacepdf"),
    )


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _speaker_matches(wanted: str, found: str) -> bool:
    if not wanted or not found:
        return False
    wanted_token_list = _speaker_match_tokens(wanted)
    found_token_list = _speaker_match_tokens(found)
    wanted_tokens = set(wanted_token_list)
    found_tokens = set(found_token_list)
    if not wanted_tokens or not found_tokens:
        return False
    overlap = wanted_tokens & found_tokens
    if len(overlap) >= min(2, len(wanted_tokens), len(found_tokens)):
        return True
    if len(overlap) == 1:
        token = next(iter(overlap))
        if len(token) >= 5 and (
            wanted_token_list.count(token) > 1 or found_token_list.count(token) > 1
        ):
            return True
    fuzzy_scores = sorted(
        (
            max(SequenceMatcher(None, wanted, found).ratio() for found in found_tokens)
            for wanted in wanted_tokens
        ),
        reverse=True,
    )
    if len(fuzzy_scores) >= 2 and fuzzy_scores[1] >= 0.72:
        return (fuzzy_scores[0] + fuzzy_scores[1]) / 2 >= 0.84
    return False


def _speaker_match_tokens(value: str) -> list[str]:
    stopwords = {"de", "del", "la", "las", "los", "y", "don", "dona", "doa"}
    return [token for token in value.split() if len(token) > 2 and token not in stopwords]

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pypdf import PdfReader

PDF_ROLL_CALL_EXTRACTION_METHOD = "pypdf_text_layer"
PDF_ROLL_CALL_EXTRACTION_VERSION = "1.0.0"
CONGRESS_MEMBER_COUNT = 350

_START_HEADER = "senoras y senores diputados que dijeron «si»:"
_RESULT_PATTERN = re.compile(
    r"resultado de la votacion.{0,240}?votos emitidos,?\s*(\d+)",
    re.DOTALL,
)
_PAGE_HEADER_PATTERN = re.compile(
    r"^num\.\s+\d+\s+.+\s+pag\.\s+\d+$",
)
_PAGE_HEADER_COMPONENT_PATTERNS = (
    re.compile(r"^num\.\s+\d+$"),
    re.compile(r"^\d{1,2}\s+de\s+\w+\s+de\s+\d{4}$"),
    re.compile(r"^pag\.\s+\d+$"),
)


@dataclass(frozen=True)
class PdfRollCallVote:
    raw_deputy_name: str
    deputy_name: str
    vote: str
    roll_call_section: str
    source_page: int


@dataclass(frozen=True)
class PdfRollCallResult:
    votes: tuple[PdfRollCallVote, ...]
    page_start: int
    page_end: int
    emitted_votes: int
    yes_votes: int
    no_votes: int
    abstentions: int
    null_votes: int
    not_voting: int
    footnote_markers: tuple[str, ...]


def parse_vote_summary_pdf_roll_call(
    *,
    pdf_path: Path,
    detail_pdf_url: str,
    expected_yes_votes: int,
    expected_no_votes: int,
    expected_abstentions: int,
    expected_member_count: int = CONGRESS_MEMBER_COUNT,
) -> PdfRollCallResult:
    """Parse and strictly reconcile an official Diario roll-call text layer."""

    target_page = _page_hint(detail_pdf_url)
    reader = PdfReader(pdf_path)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    return parse_vote_summary_roll_call_texts(
        page_texts=page_texts,
        target_page=target_page,
        expected_yes_votes=expected_yes_votes,
        expected_no_votes=expected_no_votes,
        expected_abstentions=expected_abstentions,
        expected_member_count=expected_member_count,
    )


def parse_vote_summary_roll_call_texts(
    *,
    page_texts: list[str],
    target_page: int,
    expected_yes_votes: int,
    expected_no_votes: int,
    expected_abstentions: int,
    expected_member_count: int = CONGRESS_MEMBER_COUNT,
) -> PdfRollCallResult:
    """Pure parser used by runtime and deterministic regression fixtures."""

    if target_page < 1 or target_page > len(page_texts):
        raise ValueError("Vote PDF page hint is outside the document")
    expected = (expected_yes_votes, expected_no_votes, expected_abstentions)
    if expected_member_count <= 0 or any(value < 0 for value in expected):
        raise ValueError("Expected vote totals are invalid")

    votes: list[PdfRollCallVote] = []
    current_vote: str | None = None
    current_section: str | None = None
    started = False
    stopped = False
    footnote_markers: set[str] = set()
    footnote_blocks: set[str] = set()

    for page_index in range(target_page - 1, len(page_texts)):
        in_footnote = False
        lines = [line.strip() for line in page_texts[page_index].splitlines() if line.strip()]
        for line in lines:
            normalized_line = _normalized(line)
            if not started:
                if normalized_line == _START_HEADER:
                    started = True
                    current_vote = "Sí"
                    current_section = "floor"
                continue
            if _is_result_speaker_line(normalized_line):
                stopped = True
                break

            header = _roll_call_header(line)
            if header is not None:
                current_vote, current_section = header
                in_footnote = False
                continue
            if _looks_like_unknown_roll_call_header(line):
                raise ValueError(f"Unsupported vote roll-call header: {line}")
            if _is_page_noise(normalized_line):
                continue

            footnote_start = re.match(r"^(\d+)\s+", line)
            if footnote_start and not _valid_name(line):
                footnote_blocks.add(footnote_start.group(1))
                in_footnote = True
                continue
            if in_footnote:
                continue
            if current_vote is None or current_section is None:
                raise ValueError("Vote roll-call name appeared before a category header")

            raw_name = line
            footnote_match = re.search(r"(?<=\D)(\d+)$", raw_name)
            deputy_name = (
                raw_name[: footnote_match.start()].rstrip()
                if footnote_match is not None
                else raw_name
            )
            if not _valid_name(deputy_name):
                raise ValueError(
                    f"Unexpected text inside official vote roll call on page "
                    f"{page_index + 1}: {line}"
                )
            if footnote_match is not None:
                footnote_markers.add(footnote_match.group(1))
            votes.append(
                PdfRollCallVote(
                    raw_deputy_name=raw_name,
                    deputy_name=_normalize_name(deputy_name),
                    vote=current_vote,
                    roll_call_section=current_section,
                    source_page=page_index + 1,
                )
            )
        if stopped:
            break

    if not started or not stopped or not votes:
        raise ValueError("Official PDF does not contain a bounded nominal roll call")
    if footnote_markers != footnote_blocks:
        raise ValueError(
            "Vote PDF footnote markers do not reconcile: "
            f"names={sorted(footnote_markers)} blocks={sorted(footnote_blocks)}"
        )

    identities = [_identity_key(vote.deputy_name) for vote in votes]
    if len(identities) != len(set(identities)):
        raise ValueError("Official PDF roll call contains duplicate deputy names")

    counts = {
        label: sum(vote.vote == label for vote in votes)
        for label in ("Sí", "No", "Abstención", "Nulo", "No vota")
    }
    if (
        counts["Sí"],
        counts["No"],
        counts["Abstención"],
    ) != expected:
        raise ValueError(
            "PDF roll-call decisions do not reconcile with the official summary: "
            f"expected={expected} actual="
            f"{(counts['Sí'], counts['No'], counts['Abstención'])}"
        )

    emitted_votes = _emitted_votes(page_texts[target_page - 1 :])
    observed_emitted = counts["Sí"] + counts["No"] + counts["Abstención"] + counts["Nulo"]
    if emitted_votes != observed_emitted:
        raise ValueError(
            "PDF emitted-vote total does not reconcile with the nominal roll call: "
            f"reported={emitted_votes} observed={observed_emitted}"
        )
    expected_not_voting = expected_member_count - emitted_votes
    if expected_not_voting < 0 or counts["No vota"] != expected_not_voting:
        raise ValueError(
            "PDF absence total does not reconcile with Congress membership: "
            f"expected={expected_not_voting} observed={counts['No vota']}"
        )
    if len(votes) != expected_member_count:
        raise ValueError(
            "PDF roll-call coverage is incomplete: "
            f"expected={expected_member_count} observed={len(votes)}"
        )

    return PdfRollCallResult(
        votes=tuple(votes),
        page_start=min(vote.source_page for vote in votes),
        page_end=max(vote.source_page for vote in votes),
        emitted_votes=emitted_votes,
        yes_votes=counts["Sí"],
        no_votes=counts["No"],
        abstentions=counts["Abstención"],
        null_votes=counts["Nulo"],
        not_voting=counts["No vota"],
        footnote_markers=tuple(sorted(footnote_markers)),
    )


def _roll_call_header(line: str) -> tuple[str, str] | None:
    clean = _normalized(line)
    if not clean.endswith(":"):
        return None
    if "votos nulos" in clean:
        vote = "Nulo"
    elif "ausentes" in clean:
        vote = "No vota"
    elif "abstuv" in clean:
        vote = "Abstención"
    elif "«si»" in clean:
        vote = "Sí"
    elif "«no»" in clean:
        vote = "No"
    else:
        return None

    if "telematicamente" in clean:
        section = "telematic"
    elif "miembros del gobierno" in clean:
        section = "government"
    elif "miembros de la mesa" in clean:
        section = "bureau"
    else:
        section = "floor"
    return vote, section


def _looks_like_unknown_roll_call_header(line: str) -> bool:
    clean = _normalized(line)
    return clean.endswith(":") and any(token in clean for token in ("diputad", "miembros", "votos"))


def _is_result_speaker_line(normalized_line: str) -> bool:
    return normalized_line.startswith(("la senora presidenta:", "el senor vicepresidente"))


def _is_page_noise(normalized_line: str) -> bool:
    return bool(
        normalized_line.startswith("cve:")
        or normalized_line.startswith("diario de sesiones")
        or normalized_line == "congreso de los diputados"
        or normalized_line == "pleno y diputacion permanente"
        or _PAGE_HEADER_PATTERN.match(normalized_line)
        or any(pattern.match(normalized_line) for pattern in _PAGE_HEADER_COMPONENT_PATTERNS)
    )


def _valid_name(value: str) -> bool:
    if len(value) > 90 or value.count(",") != 1:
        return False
    surname, given_name = (part.strip() for part in value.split(",", 1))
    return _valid_name_part(surname) and _valid_name_part(given_name)


def _valid_name_part(value: str) -> bool:
    first_letter = next((character for character in value if character.isalpha()), None)
    return bool(
        value
        and first_letter is not None
        and first_letter.isupper()
        and all(character.isalpha() or character in " -.'’" for character in value)
    )


def _normalize_name(value: str) -> str:
    surname, given_name = (" ".join(part.split()) for part in value.split(",", 1))
    return f"{surname}, {given_name}"


def _identity_key(value: str) -> str:
    return _normalized(" ".join(value.split()))


def _normalized(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _emitted_votes(page_texts: list[str]) -> int:
    normalized_text = _normalized("\n".join(page_texts))
    matches = _RESULT_PATTERN.findall(normalized_text)
    if not matches:
        raise ValueError("Vote PDF has no reported emitted-vote total after the roll call")
    return int(matches[-1])


def _page_hint(url: str) -> int:
    parsed = urlparse(url)
    fragment = parse_qs(parsed.fragment)
    raw_page = fragment.get("page", [None])[0]
    if raw_page is None or not str(raw_page).isdigit():
        raise ValueError("Vote detail PDF URL has no numeric page fragment")
    return int(raw_page)

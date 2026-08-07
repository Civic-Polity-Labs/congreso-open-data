from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

VOTE_DETAIL_PDF_EXTRACTION_METHOD = "pypdf_text_layer_strict"
VOTE_DETAIL_PDF_EXTRACTION_VERSION = "1.2.0"

_TOTALS_PATTERN = re.compile(
    r"\bpresentes\s+si\s+no\s+abstenciones\s+no\s+votan\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\b",
    re.DOTALL,
)
_METADATA_PATTERN = re.compile(
    r"(?m)^\s*(\d+)\s*votacion:\s*sesion:\s*(\d+)\s+"
    r"fecha:\s*(\d{1,2})-(\d{1,2})-(\d{4})\s*$"
)


@dataclass(frozen=True)
class VoteDetailPdfNominalRow:
    deputy_name: str
    seat: str
    vote: str
    parliamentary_group: str | None
    source_page: int


@dataclass(frozen=True)
class VoteDetailPdfResult:
    session_number: int
    vote_number: int
    vote_date: str
    present: int
    yes_votes: int
    no_votes: int
    abstentions: int
    not_voting: int
    nominal_rows: tuple[VoteDetailPdfNominalRow, ...]
    page_count: int
    first_page_text: str = ""


def parse_vote_detail_pdf(pdf_path: Path) -> VoteDetailPdfResult:
    reader = PdfReader(pdf_path)
    return parse_vote_detail_page_texts([page.extract_text() or "" for page in reader.pages])


def parse_vote_detail_page_texts(page_texts: list[str]) -> VoteDetailPdfResult:
    """Strictly parse a Congreso per-vote detail PDF without inventing fields."""

    if not page_texts or not page_texts[0].strip():
        raise ValueError("Vote detail PDF has no readable first-page text layer")
    first_page = _fold(page_texts[0])
    metadata = _METADATA_PATTERN.search(first_page)
    # Long joint-vote descriptions can fill page 1 and move the official totals
    # table to page 2. Keep metadata anchored to page 1, but inspect a tightly
    # bounded initial window for totals before nominal parsing begins.
    totals = _TOTALS_PATTERN.search("\n".join(_fold(text) for text in page_texts[:3]))
    if metadata is None or totals is None:
        raise ValueError("Vote detail PDF metadata or totals are missing")
    vote_number, session, day, month, year = (int(value) for value in metadata.groups())
    present, yes_votes, no_votes, abstentions, not_voting = (
        int(value) for value in totals.groups()
    )
    if present != yes_votes + no_votes + abstentions:
        raise ValueError("Vote detail PDF totals do not reconcile")

    rows: list[VoteDetailPdfNominalRow] = []
    current_vote: str | None = None
    current_group: str | None = None
    for page_number, text in enumerate(page_texts[1:], start=2):
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            folded = _fold(line)
            category = _vote_category(folded)
            if category is not None:
                current_vote = category
                current_group = None
                continue
            if folded.startswith("pagina ") or folded.startswith("total:"):
                continue
            if line == "-":
                # The official percentage table uses a lone dash for an
                # inapplicable value. A nominal row always has text after '-'.
                continue
            if line.startswith("-"):
                if current_vote is None:
                    raise ValueError("Vote detail PDF nominal row precedes a category")
                deputy_name, seat = _parse_nominal_line(line)
                rows.append(
                    VoteDetailPdfNominalRow(
                        deputy_name=deputy_name,
                        seat=seat,
                        vote=current_vote,
                        parliamentary_group=current_group,
                        source_page=page_number,
                    )
                )
                continue
            if current_vote is not None:
                current_group = line

    identities = [(_identity(row.deputy_name), row.vote) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Vote detail PDF contains duplicate nominal identities")
    observed = Counter(row.vote for row in rows)
    if rows and (
        observed["Sí"] != yes_votes
        or observed["No"] != no_votes
        or observed["Abstención"] != abstentions
        or observed["No vota"] != not_voting
    ):
        raise ValueError("Vote detail PDF nominal rows do not reconcile with totals")
    return VoteDetailPdfResult(
        session_number=session,
        vote_number=vote_number,
        vote_date=f"{year:04d}{month:02d}{day:02d}",
        present=present,
        yes_votes=yes_votes,
        no_votes=no_votes,
        abstentions=abstentions,
        not_voting=not_voting,
        nominal_rows=tuple(rows),
        page_count=len(page_texts),
        first_page_text=page_texts[0],
    )


def _vote_category(folded_line: str) -> str | None:
    labels = {
        "si": "Sí",
        "no": "No",
        "abstencion": "Abstención",
        "abstenciones": "Abstención",
        "no votan": "No vota",
    }
    return labels.get(folded_line)


def _parse_nominal_line(line: str) -> tuple[str, str]:
    value = line.removeprefix("-").strip()
    telematic = re.fullmatch(r"(.+?)(TELEM[AÁ]TICO)", value, flags=re.IGNORECASE)
    if telematic is not None:
        deputy_name = telematic.group(1).strip()
        seat = "TELEMÁTICO"
    else:
        match = re.fullmatch(r"(.+?)(\d{1,4})", value)
        if match is None:
            raise ValueError(f"Malformed vote detail PDF nominal row: {line}")
        deputy_name = match.group(1).strip()
        seat = match.group(2)
    if deputy_name.count(",") != 1 or not all(part.strip() for part in deputy_name.split(",", 1)):
        raise ValueError(f"Malformed deputy name in vote detail PDF: {line}")
    return " ".join(deputy_name.split()), seat


def _identity(value: str) -> str:
    return _fold(" ".join(value.split()))


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))

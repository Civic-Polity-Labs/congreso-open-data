from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_INITIATIVE_FILE_NUMBER = re.compile(
    r"(?P<kind>\d{1,3})\s*/\s*(?P<number>\d{1,6})(?:\s*/\s*(?P<suffix>\d{1,4}))?"
)
_QUALIFIED_INITIATIVE_REFERENCE = re.compile(
    r"(?P<file>\d{1,3}\s*/\s*\d{1,6}(?:\s*/\s*\d{1,4})?)\s+"
    r"(?P<qualifier>\d+)"
)


@dataclass(frozen=True)
class InitiativeReference:
    """Canonical parent file plus the exact official source reference.

    Some historical intervention exports append a numeric sub-item/internal
    qualifier to an otherwise valid expediente (for example
    ``121/000125/0000 43424``). The parent file is linkable to initiatives, while
    the raw value and qualifier must remain available to distinguish source rows.
    """

    file_number: str | None
    raw_value: str | None
    qualifier: str | None


def canonical_initiative_file_number(value: Any) -> str | None:
    """Return the Congreso Open Data form ``NNN/NNNNNN/NNNN`` when recognized."""

    if value in (None, ""):
        return None
    text = str(value).strip()
    match = _INITIATIVE_FILE_NUMBER.fullmatch(text)
    if match is None:
        return text
    kind = match.group("kind").zfill(3)
    number = match.group("number").zfill(6)
    suffix = (match.group("suffix") or "0").zfill(4)
    return f"{kind}/{number}/{suffix}"


def parse_initiative_reference(value: Any) -> InitiativeReference:
    """Split the observed historical qualified-reference form without guessing."""

    if value in (None, ""):
        return InitiativeReference(None, None, None)
    raw_value = str(value).strip()
    qualified = _QUALIFIED_INITIATIVE_REFERENCE.fullmatch(raw_value)
    if qualified is not None:
        return InitiativeReference(
            file_number=canonical_initiative_file_number(qualified.group("file")),
            raw_value=raw_value,
            qualifier=qualified.group("qualifier"),
        )
    return InitiativeReference(
        file_number=canonical_initiative_file_number(raw_value),
        raw_value=raw_value,
        qualifier=None,
    )


def initiative_reference_identity(value: Any) -> str | None:
    """Return a stable identity component that preserves a source qualifier."""

    reference = parse_initiative_reference(value)
    if reference.file_number is None:
        return None
    if reference.qualifier is None:
        return reference.file_number
    return f"{reference.file_number} {reference.qualifier}"


def initiative_file_numbers_in_text(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    matches = _INITIATIVE_FILE_NUMBER.finditer(str(value))
    return [
        canonical
        for match in matches
        if (canonical := canonical_initiative_file_number(match.group(0))) is not None
    ]

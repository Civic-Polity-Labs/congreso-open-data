"""Typed, serializable query contracts used by both users and orchestrators."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date as Date
from enum import StrEnum
from typing import Any, Literal, Self

from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from congreso_open_data.plugins import ExtractionTask


class RefreshPolicy(StrEnum):
    AUTO = "auto"
    ALWAYS = "always"


class SortOrder(StrEnum):
    ASCENDING = "asc"
    DESCENDING = "desc"


class TextPolicy(StrEnum):
    NONE = "none"
    NATIVE = "native"
    OCR = "ocr"


_ROMAN_VALUES = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def legislature_roman(value: str | int) -> str:
    """Normalize ``15``, ``XV`` and ``Leg.15`` to the public ``XV`` form."""

    raw = str(value).strip().upper()
    raw = re.sub(r"^LEG[.\s_-]*", "", raw)
    if raw == "0":
        return "0"
    if raw.isdigit():
        number = int(raw)
        if number < 1 or number > 99:
            raise ValueError("legislature must be between 0 and 99")
        result: list[str] = []
        remaining = number
        for unit, roman in _ROMAN_VALUES:
            while remaining >= unit:
                result.append(roman)
                remaining -= unit
        return "".join(result)
    if not re.fullmatch(r"[IVXLCDM]+", raw):
        raise ValueError(f"invalid legislature: {value!r}")
    number = legislature_number(raw)
    return legislature_roman(number)


def legislature_number(value: str | int) -> int:
    raw = str(value).strip().upper()
    raw = re.sub(r"^LEG[.\s_-]*", "", raw)
    if raw.isdigit():
        return int(raw)
    values = {roman: unit for unit, roman in _ROMAN_VALUES}
    total = 0
    index = 0
    while index < len(raw):
        if index + 1 < len(raw) and raw[index : index + 2] in values:
            total += values[raw[index : index + 2]]
            index += 2
        elif raw[index] in values:
            total += values[raw[index]]
            index += 1
        else:
            raise ValueError(f"invalid legislature: {value!r}")
    if legislature_roman(total) != raw:
        raise ValueError(f"invalid canonical Roman legislature: {value!r}")
    return total


class CongressQuery(BaseModel):
    """Common bounded filters; concrete domains add their own fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    domain: str
    legislatures: tuple[str, ...] = ("XV",)
    date_from: Date | None = None
    date_to: Date | None = None
    last_months: int | None = Field(default=None, ge=1, le=1200)
    refresh: RefreshPolicy = RefreshPolicy.AUTO
    sort: SortOrder = SortOrder.ASCENDING
    max_results: int = Field(default=50_000, ge=1, le=1_000_000)
    allow_partial: bool = False
    extractions: tuple[ExtractionTask, ...] = ()

    @field_validator("legislatures", mode="before")
    @classmethod
    def _legislatures(cls, value: Any) -> tuple[str, ...]:
        raw = value if isinstance(value, (list, tuple, set)) else (value,)
        normalized = tuple(dict.fromkeys(legislature_roman(item) for item in raw))
        if not normalized:
            raise ValueError("at least one legislature is required")
        return normalized

    @model_validator(mode="after")
    def _date_contract(self) -> Self:
        if self.last_months is not None and (
            self.date_from is not None or self.date_to is not None
        ):
            raise ValueError("last_months cannot be combined with date_from/date_to")
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_to < self.date_from
        ):
            raise ValueError("date_to must be on or after date_from")
        return self

    def resolved(self, *, today: Date) -> Self:
        if self.last_months is None:
            start = self.date_from
            end = self.date_to
        else:
            start = today - relativedelta(months=self.last_months)
            end = today
        return self.model_copy(update={"date_from": start, "date_to": end, "last_months": None})

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class InterventionQuery(CongressQuery):
    domain: Literal["interventions"] = "interventions"
    speaker: str | None = None
    speaker_id: str | None = None
    title: str | None = None
    text: str | None = None
    initiative_type: str | None = None
    initiative_file_number: str | None = None
    phase: str | None = None
    body: str | None = None
    author: str | None = None
    text_policy: TextPolicy = TextPolicy.NATIVE

    @model_validator(mode="after")
    def _speaker_contract(self) -> Self:
        if self.speaker is not None and self.speaker_id is not None:
            raise ValueError("speaker and speaker_id are mutually exclusive")
        return self


class DeputyQuery(CongressQuery):
    domain: Literal["deputies"] = "deputies"
    name: str | None = None
    deputy_id: str | None = None
    constituency: str | None = None
    parliamentary_group: str | None = None


class ProfileQuery(CongressQuery):
    domain: Literal["profiles"] = "profiles"
    name: str | None = None
    deputy_id: str | None = None


class InterestQuery(CongressQuery):
    domain: Literal["interests"] = "interests"
    deputy: str | None = None
    declaration_kind: str | None = None


class FinancialDocumentQuery(CongressQuery):
    domain: Literal["financial_documents"] = "financial_documents"
    deputy: str | None = None
    document_kind: str | None = None


class InitiativeQuery(CongressQuery):
    domain: Literal["initiatives"] = "initiatives"
    title: str | None = None
    text: str | None = None
    author: str | None = None
    file_number: str | None = None
    initiative_type: str | None = None


class VoteQuery(CongressQuery):
    domain: Literal["votes"] = "votes"
    session: str | None = None
    vote_number: str | None = None
    deputy: str | None = None


class OrganQuery(CongressQuery):
    domain: Literal["organs"] = "organs"
    name: str | None = None
    organ_type: str | None = None


class SalaryEntitlementQuery(CongressQuery):
    domain: Literal["salary_entitlements"] = "salary_entitlements"
    role: str | None = None


class DocumentQuery(CongressQuery):
    domain: Literal["documents"] = "documents"
    source_families: tuple[str, ...] = ()
    source_datasets: tuple[str, ...] = ()
    document_kind: str | None = None
    entity_id: str | None = None
    mime_type: str | None = None


AnyCongressQuery = (
    InterventionQuery
    | DeputyQuery
    | ProfileQuery
    | InterestQuery
    | FinancialDocumentQuery
    | InitiativeQuery
    | VoteQuery
    | OrganQuery
    | SalaryEntitlementQuery
    | DocumentQuery
)

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from congreso_open_data.html import HtmlLink
from congreso_open_data.identifiers import (
    canonical_initiative_file_number,
    initiative_file_numbers_in_text,
    parse_initiative_reference,
)
from congreso_open_data.initiative_ownership import INITIATIVE_OWNER_PREFIX_SET
from congreso_open_data.interventions import (
    canonical_intervention_pdf_url,
    canonical_intervention_text_url,
    canonical_official_resource_url,
    intervention_document_id_from_urls,
    intervention_id,
    page_hint_from_url,
)
from congreso_open_data.normalization import normalize_record_keys, parse_spanish_date, stable_id


def split_links(value: str | None) -> list[str]:
    if not value:
        return []
    output: list[str] = []
    for part in re.split(r"\s+", value):
        raw = part.strip()
        if not raw.startswith("http"):
            continue
        fragment = urlparse(raw).fragment
        canonical = canonical_official_resource_url(raw)
        if canonical:
            output.append(f"{canonical}#{fragment}" if fragment else canonical)
    return output


def split_file_numbers(value: str | None) -> list[str]:
    return initiative_file_numbers_in_text(value)


def parse_euro_amount(value: str) -> float | None:
    # Accept the correctly decoded symbol and the legacy mojibake still present
    # in some historical snapshots. New normalized output remains Unicode.
    match = re.search(
        r"(\d{1,3}(?:\.\d{3})*,\d{2})\s*(?:\u20ac|\u00e2.{0,3}\u00ac)",
        value,
    )
    if not match:
        return None
    return float(match.group(1).replace(".", "").replace(",", "."))


def deputy_row(
    row: dict[str, Any],
    *,
    source_sha256: str,
    snapshot_date: str,
    default_legislature: str | None = None,
) -> dict[str, Any]:
    item = normalize_record_keys(row)
    first_name, last_names, full_name = _deputy_names(item)
    legislature = item.get("legislatura") or default_legislature
    birth_date = _first_date(
        item,
        "fechanacimiento",
        "fecha_nacimiento",
        "nacimiento",
        "fecha de nacimiento",
    )
    return {
        "deputy_term_id": stable_id(
            full_name,
            legislature,
            item.get("fechalta"),
            item.get("circunscripcion"),
            item.get("grupoparlamentario"),
            item.get("fechaaltaengrupoparlamentario"),
            item.get("fechabajaengrupoparlamentario"),
        ),
        "person_id": stable_id(full_name),
        "full_name": full_name,
        "first_name": first_name,
        "last_names": last_names,
        "birth_date": birth_date,
        "gender": _gender_value(_first_present(item, "genero", "sexo", "gender")),
        "age_at_snapshot": _age_at(birth_date, snapshot_date),
        "legislature": legislature,
        "constituency": item.get("circunscripcion"),
        "electoral_party": item.get("formacionelectoral"),
        "parliamentary_group": item.get("grupoparlamentario"),
        "term_start_date": parse_spanish_date(item.get("fechaalta")),
        "term_end_date": parse_spanish_date(item.get("fechabaja")),
        "group_start_date": parse_spanish_date(item.get("fechaaltaengrupoparlamentario")),
        "group_end_date": parse_spanish_date(item.get("fechabajaengrupoparlamentario")),
        "biography": item.get("biografia"),
        "profile_url": _first_present(item, "urlficha", "urlfichadiputado", "ficha"),
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }


def interest_row(row: dict[str, Any], *, source_sha256: str, snapshot_date: str) -> dict[str, Any]:
    item = normalize_record_keys(row)
    full_name = item.get("nombre")
    return {
        "interest_id": stable_id(
            full_name,
            item.get("fecharegistro"),
            item.get("declaracion"),
            item.get("tipo"),
            item.get("periodo"),
            item.get("empleador"),
            item.get("sector"),
            item.get("destinatario"),
            item.get("benefactor"),
            item.get("descripcion"),
            item.get("observaciones"),
        ),
        "person_id": stable_id(full_name),
        "full_name": full_name,
        "registered_at": parse_spanish_date(item.get("fecharegistro")),
        "declaration_kind": item.get("declaracion"),
        "item_type": item.get("tipo"),
        "period": item.get("periodo"),
        "employer": item.get("empleador"),
        "sector": item.get("sector"),
        "recipient": item.get("destinatario"),
        "benefactor": item.get("benefactor"),
        "description": item.get("descripcion"),
        "observations": item.get("observaciones"),
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }


def deputy_profile_row(
    *,
    visible_text: str,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> dict[str, Any]:
    lines = _visible_lines(visible_text)
    params = parse_qs(urlparse(source_url).query)
    cod_parlamentario = _first_param(params, "codParlamentario")
    legislature = _first_param(params, "idLegislatura")
    full_name = _profile_full_name(lines)
    birth_index, birth_date = _profile_birth_date(lines)
    birth_place = _profile_birth_place(lines, birth_index)
    full_condition_date = _profile_full_condition_date(lines)
    return {
        "deputy_profile_id": stable_id(full_name, legislature, cod_parlamentario),
        "person_id": stable_id(full_name),
        "full_name": full_name,
        "cod_parlamentario": cod_parlamentario,
        "legislature": legislature,
        "birth_date": birth_date,
        "age_at_snapshot": _age_at(birth_date, snapshot_date),
        "birth_place": birth_place,
        "constituency": _profile_constituency(lines),
        "parliamentary_group": _first_line_matching(lines, r"^G\.P\."),
        "parliamentary_group_code": _profile_group_code(lines),
        "electoral_party": _profile_party(lines),
        "email": _first_line_matching(lines, r"@congreso\.es"),
        "full_condition_date": full_condition_date,
        "legislature_history": _profile_legislature_history(lines),
        "source_url": source_url,
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }


def deputy_financial_document_rows_from_profile(
    *,
    links: list[HtmlLink],
    profile: dict[str, Any],
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for link in links:
        document_kind = _financial_document_kind(link.text, link.url)
        if document_kind is None:
            continue
        rows.append(
            {
                "financial_document_id": stable_id(
                    profile["person_id"],
                    profile.get("legislature"),
                    document_kind,
                    link.url,
                ),
                "person_id": profile["person_id"],
                "full_name": profile["full_name"],
                "legislature": profile.get("legislature"),
                "parliamentary_group": profile.get("parliamentary_group"),
                "registered_at": _date_from_parenthesized_java_text(link.text),
                "declaration_kind": _declaration_label(link.text),
                "document_kind": document_kind,
                "source_url": link.url,
                "document_sha256": None,
                "extraction_status": "linked",
                "snapshot_date": snapshot_date,
            }
        )
    return rows


def initiative_row(
    row: dict[str, Any],
    *,
    source_sha256: str,
    snapshot_date: str,
    source_dataset: str | None = None,
) -> dict[str, Any]:
    item = normalize_record_keys(row)
    subject = item.get("objeto") or item.get("titulo_ley")
    file_number = canonical_initiative_file_number(
        item.get("numexpediente") or item.get("numero_ley")
    )
    law_date = parse_spanish_date(item.get("fecha_ley"))
    legislature = item.get("legislatura") or _approved_law_legislature(
        source_dataset=source_dataset,
        law_date=law_date,
    )
    if file_number and "/" in file_number:
        initiative_id = stable_id(
            "congreso",
            "initiative",
            legislature,
            file_number,
        )
    else:
        # Approved-law catalog rows expose a law number rather than a parliamentary
        # expediente. The same short number can recur within one legislature, so the
        # richer source identity remains necessary for this non-expedient fallback.
        initiative_id = stable_id(
            source_dataset,
            legislature,
            file_number,
            subject,
            item.get("fecha_ley"),
            item.get("numero_boletin"),
            item.get("pdf"),
        )
    return {
        "initiative_id": initiative_id,
        "source_dataset": source_dataset,
        "legislature": legislature,
        "super_type": item.get("supertipo"),
        "grouping": item.get("agrupacion"),
        "type": item.get("tipo"),
        "subject": subject,
        "file_number": file_number,
        "law_number": item.get("numero_ley"),
        "gazette_number": item.get("numero_boletin"),
        "gazette_date": parse_spanish_date(item.get("fecha_boletin")),
        "law_date": law_date,
        "presented_at": parse_spanish_date(item.get("fechapresentacion")),
        "qualified_at": parse_spanish_date(item.get("fechacalificacion")),
        "author": item.get("autor"),
        "procedure_type": item.get("tipotramitacion"),
        "current_status": item.get("situacionactual"),
        "processing_result": item.get("resultadotramitacion"),
        "competent_committee": item.get("comisioncompetente"),
        "deadlines": item.get("plazos"),
        "rapporteurs": item.get("ponentes"),
        "processing_history": item.get("tramitacionseguida"),
        "origin_file_numbers": split_file_numbers(item.get("iniciativasdeorigen")),
        "related_file_numbers": split_file_numbers(item.get("iniciativasrelacionadas")),
        "bocg_links": split_links(item.get("enlacesbocg") or item.get("pdf")),
        "ds_links": split_links(item.get("enlacesds")),
        "pdf_links": split_links(item.get("pdf")),
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }


def historical_initiative_rows_from_list_payload(
    payload: dict[str, Any],
    *,
    source_dataset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _historical_initiative_items(payload):
        file_number = canonical_initiative_file_number(item.get("id_iniciativa"))
        if (
            source_dataset == "GeneralInitiatives"
            and _initiative_prefix(file_number) in INITIATIVE_OWNER_PREFIX_SET
        ):
            continue
        detail_url = _historical_initiative_detail_url(item.get("enlace"))
        result = item.get("resultado_tram")
        general_group = item.get("atis") or item.get("tipo")
        rows.append(
            {
                "LEGISLATURA": _leg_label(item.get("legislatura")),
                "SUPERTIPO": (
                    "Iniciativas"
                    if source_dataset == "GeneralInitiatives"
                    else "Iniciativas legislativas"
                ),
                "AGRUPACION": (
                    general_group
                    if source_dataset == "GeneralInitiatives"
                    else _historical_initiative_group(source_dataset)
                ),
                "TIPO": (
                    item.get("tipo") or item.get("atis")
                    if source_dataset == "GeneralInitiatives"
                    else _historical_initiative_type(source_dataset)
                ),
                "OBJETO": item.get("titulo"),
                "NUMEXPEDIENTE": file_number,
                "FECHAPRESENTACION": item.get("fecha_presentado"),
                "FECHACALIFICACION": item.get("fecha_calificado"),
                "AUTOR": _historical_authors(item.get("autores") or item.get("autor")),
                "SITUACIONACTUAL": result,
                "RESULTADOTRAMITACION": result,
                "TRAMITACIONSEGUIDA": detail_url,
            }
        )
    return rows


def _initiative_prefix(file_number: str | None) -> str | None:
    if not file_number or "/" not in file_number:
        return None
    return file_number.split("/", 1)[0]


def approved_law_rows_from_payload(
    payload: dict[str, Any],
    *,
    default_year: str | None = None,
    include_current_legislature: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return rows
    for law_code, values in data.items():
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            tramitacion_url = _absolute_congreso_url(item.get("tramitacion"))
            legislature = _leg_label(_query_param(tramitacion_url, "_iniciativas_legislatura"))
            law_date = _law_date_from_title(item.get("titulo"), default_year=default_year)
            if legislature is None:
                legislature = _leg_label(_legislature_from_date(law_date))
            if not include_current_legislature and legislature == "Leg.15":
                continue
            pdf_url = _approved_law_pdf_url(item.get("pdf") or item.get("pdf2"))
            rows.append(
                {
                    "LEGISLATURA": legislature,
                    "SUPERTIPO": "Iniciativas legislativas aprobadas",
                    "AGRUPACION": _approved_law_group(law_code),
                    "TIPO": _approved_law_group(law_code),
                    "OBJETO": item.get("titulo"),
                    "TITULO_LEY": item.get("titulo"),
                    "NUMERO_LEY": item.get("numLey"),
                    "FECHA_LEY": _date_text(law_date),
                    "PDF": pdf_url,
                    "RESULTADOTRAMITACION": item.get("descrTipoTexto"),
                    "TRAMITACIONSEGUIDA": tramitacion_url,
                }
            )
    return rows


def _approved_law_legislature(*, source_dataset: str | None, law_date: date | None) -> str | None:
    if source_dataset != "IniciativasLegislativasAprobadas" or law_date is None:
        return None
    if law_date >= date(2023, 8, 17):
        return "Leg.15"
    return None


def intervention_row(
    row: dict[str, Any],
    *,
    source_sha256: str,
    snapshot_date: str,
    source_export_page: int | None = None,
    source_record_ordinal: int | None = None,
) -> dict[str, Any]:
    item = normalize_record_keys(row)
    full_text_url = canonical_intervention_text_url(item.get("enlacetextointegro"))
    raw_pdf_url = item.get("enlacepdf")
    pdf_url = canonical_intervention_pdf_url(raw_pdf_url)
    session_date = parse_spanish_date(_first_present(item, "sesion", "fecha"))
    initiative_reference = parse_initiative_reference(
        _first_present(item, "numexpediente", "numeroexpediente")
    )
    return {
        "intervention_id": intervention_id(row),
        "document_id": intervention_document_id_from_urls(
            full_text_url=full_text_url,
            pdf_url=pdf_url,
        ),
        "legislature": _leg_label(item.get("legislatura")),
        "initiative_file_number": initiative_reference.file_number,
        "initiative_reference_raw": initiative_reference.raw_value,
        "initiative_reference_qualifier": initiative_reference.qualifier,
        "initiative_subject": item.get("objetoiniciativa"),
        "session_date": session_date,
        "session_year": session_date.year if session_date else None,
        "body": _first_present(item, "organo", "nombresesion"),
        "phase": item.get("fase"),
        "intervention_type": _first_present(item, "tipointervencion", "tipo"),
        "speaker_name": item.get("orador"),
        "speaker_role": item.get("cargoorador"),
        "starts_at": _first_present(item, "iniciointervencion", "horainicio"),
        "ends_at": _first_present(item, "finintervencion", "horafin"),
        "video_url": canonical_official_resource_url(
            _first_present(item, "enlacediferido", "enlaceemision")
        ),
        "direct_video_url": canonical_official_resource_url(
            _first_present(item, "enlacedescargadirecta", "enlacedescarga")
        ),
        "full_text_url": full_text_url,
        "pdf_url": pdf_url,
        "page_hint": page_hint_from_url(full_text_url) or page_hint_from_url(raw_pdf_url),
        "source_export_page": source_export_page,
        "source_record_ordinal": source_record_ordinal,
        "source_index_sha256": source_sha256,
        "text_fragment": None,
        "fragment_confidence": None,
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }


def vote_rows(
    payload: dict[str, Any],
    *,
    source_sha256: str,
    snapshot_date: str,
    legislature: str | None = None,
    source_url: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    info = payload["informacion"]
    totals = payload["totales"]
    nominal_items = payload.get("votaciones", [])
    nominal_data_available = bool(nominal_items)
    decision_method = (
        "assent"
        if _fold(str(totals.get("asentimiento") or "").strip()) == "si"
        else "recorded_vote"
    )
    source_mode = (
        "official_json_nominal"
        if nominal_data_available
        else "official_json_assent"
        if decision_method == "assent"
        else "official_json_totals_only"
    )
    vote_date = parse_spanish_date(info.get("fecha"))
    event_id = stable_id(
        "Congreso",
        "vote_event",
        legislature,
        info.get("sesion"),
        info.get("numeroVotacion"),
        vote_date.isoformat() if vote_date is not None else info.get("fecha"),
    )
    event = {
        "vote_event_id": event_id,
        "legislature": legislature,
        "session_number": info.get("sesion"),
        "vote_number": info.get("numeroVotacion"),
        "vote_date": vote_date,
        "title": info.get("titulo"),
        "file_text": info.get("textoExpediente"),
        "subgroup_title": info.get("tituloSubGrupo"),
        "present": totals.get("presentes"),
        "yes_votes": totals.get("afavor"),
        "no_votes": totals.get("enContra"),
        "abstentions": totals.get("abstenciones"),
        "null_votes": 0,
        "not_voting": totals.get("noVotan"),
        "source_mode": source_mode,
        "decision_method": decision_method,
        "nominal_data_available": nominal_data_available,
        "nominal_group_data_available": nominal_data_available
        and all(bool(str(item.get("grupo") or "").strip()) for item in nominal_items),
        "nominal_seat_data_available": nominal_data_available
        and all(bool(str(item.get("asiento") or "").strip()) for item in nominal_items),
        "initiative_file_number": None,
        "initiative_reference_raw": None,
        "source_url": source_url,
        "detail_pdf_url": None,
        "detail_pdf_sha256": None,
        "image_url": None,
        "image_sha256": None,
        "archive_url": None,
        "archive_sha256": None,
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }
    nominal = [
        {
            "nominal_vote_id": stable_id(
                "Congreso",
                "nominal_vote",
                event_id,
                item.get("diputado"),
            ),
            "vote_event_id": event_id,
            "legislature": legislature,
            "vote_date": vote_date,
            "seat": item.get("asiento"),
            "deputy_name": item.get("diputado"),
            "raw_deputy_name": item.get("diputado"),
            "parliamentary_group_code": item.get("grupo"),
            "vote": item.get("voto"),
            "source_mode": "official_json_nominal",
            "roll_call_section": None,
            "source_url": source_url,
            "source_page_start": None,
            "source_page_end": None,
            "extraction_method": "official_json",
            "extraction_version": "1.0.0",
            "source_file_sha256": source_sha256,
            "snapshot_date": snapshot_date,
        }
        for item in nominal_items
    ]
    source_items = [
        ("primary", info.get("textoExpediente")),
        *(
            ("joint", item.get("textoExpediente"))
            for item in (info.get("votacionesConjuntas") or [])
        ),
    ]
    vote_items = [
        {
            "vote_item_id": stable_id(
                "Congreso",
                "vote_item",
                event_id,
                item_order,
                source_item_kind,
            ),
            "vote_event_id": event_id,
            "legislature": legislature,
            "vote_date": vote_date,
            "item_order": item_order,
            "source_item_kind": source_item_kind,
            "item_text": str(item_text).strip(),
            "initiative_file_number": None,
            "initiative_reference_raw": None,
            "initiative_link_status": "unresolved_no_structured_reference",
            "initiative_link_method": None,
            "source_url": source_url,
            "source_file_sha256": source_sha256,
            "snapshot_date": snapshot_date,
        }
        for item_order, (source_item_kind, item_text) in enumerate(source_items, start=1)
        if str(item_text or "").strip()
    ]
    return event, nominal, vote_items


def organ_membership_rows(
    payload: dict[str, Any],
    *,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    params = parse_qs(urlparse(source_url).query)
    legislature = _first_param(params, "_organos_selectedLegislatura")
    organ_code = _first_param(params, "_organos_selectedSuborgano")
    organ_id = stable_id(organ_code)
    organ_name = _known_organ_name(organ_code)
    organ_type = _known_organ_type(organ_code)
    rows: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        full_name = item.get("apellidosNombre")
        role = item.get("descCargo")
        started_at = parse_spanish_date(item.get("fechaAltaFormat"))
        ended_at = parse_spanish_date(item.get("fechaBajaFormat"))
        rows.append(
            {
                "membership_id": stable_id(
                    legislature,
                    organ_id,
                    full_name,
                    role,
                    item.get("fechaAltaFormat"),
                    item.get("fechaBajaFormat"),
                ),
                "organ_id": organ_id,
                "organ_code": organ_code,
                "organ_name": organ_name,
                "organ_type": organ_type,
                "legislature": legislature,
                "person_id": stable_id(full_name),
                "full_name": full_name,
                "role": role,
                "parliamentary_group": item.get("siglas"),
                "started_at": started_at,
                "ended_at": ended_at,
                "source_url": source_url,
                "source_file_sha256": source_sha256,
                "snapshot_date": snapshot_date,
            }
        )
    return rows


def organ_rows_from_links(
    links: list[HtmlLink],
    *,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    organs: dict[str, dict[str, Any]] = {}
    for link in links:
        organ_type = _organ_type_from_link(link.text, link.url)
        if organ_type is None:
            continue
        organ_code = _organ_code_from_url(link.url)
        organ_id = stable_id(organ_code or link.url)
        organs[organ_id] = {
            "organ_id": organ_id,
            "name": link.text,
            "organ_type": organ_type,
            "url": link.url,
            "snapshot_date": snapshot_date,
        }
    return list(organs.values())


def salary_rows_from_text(
    text: str,
    *,
    source_url: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    current_role: str | None = None
    current_concept: str | None = None
    valid_from = _updated_at_from_text(text)
    role_markers = {
        "Secretario General",
        "Secretarios Generales Adjuntos",
        "Directores",
    }
    for line in lines:
        if line in role_markers:
            current_role = line
            current_concept = None
            continue
        amount = parse_euro_amount(line)
        if amount is not None and current_role and current_concept:
            rows.append(
                {
                    "salary_entitlement_id": stable_id(
                        current_role,
                        current_concept,
                        amount,
                        valid_from,
                    ),
                    "role": current_role,
                    "body": "Administración parlamentaria",
                    "concept": current_concept.rstrip(":"),
                    "amount_eur": amount,
                    "periodicity": "monthly_gross",
                    "valid_from": valid_from,
                    "valid_to": None,
                    "source_url": source_url,
                    "snapshot_date": snapshot_date,
                }
            )
            current_concept = None
            continue
        if current_role and not line.endswith(".") and not parse_spanish_date(line):
            current_concept = line
    return rows


def document_asset_rows_from_records(
    rows: Iterable[dict[str, Any]],
    *,
    family: str,
    dataset: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_record_keys(row)
        entity_id = (
            normalized.get("numexpediente")
            or normalized.get("numeroexpediente")
            or normalized.get("nombre")
            or normalized.get("objetoiniciativa")
        )
        for key, value in normalized.items():
            if "enlace" not in key and key not in {"pdf"}:
                continue
            for url in split_links(value):
                document_kind = re.sub(r"[^a-z0-9]+", "", key.casefold())
                document_id = stable_id(family, dataset, entity_id, url)
                assets[document_id] = {
                    "document_id": document_id,
                    "family": family,
                    "dataset": dataset,
                    "entity_id": entity_id,
                    "document_kind": document_kind,
                    "title": normalized.get("objeto") or normalized.get("titulo_ley") or entity_id,
                    "url": url,
                    "mime_type": _mime_type_from_url(url),
                    "page_hint": _page_hint_from_url(url),
                    "sha256": None,
                    "snapshot_date": snapshot_date,
                }
    return list(assets.values())


def document_asset_rows_from_links(
    *,
    links: list[HtmlLink],
    family: str,
    dataset: str,
    entity_id: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for link in links:
        document_kind = _financial_document_kind(link.text, link.url)
        if _mime_type_from_url(link.url) is None or document_kind is None:
            continue
        document_id = stable_id(family, dataset, entity_id, document_kind, link.url)
        assets[document_id] = {
            "document_id": document_id,
            "family": family,
            "dataset": dataset,
            "entity_id": entity_id,
            "document_kind": document_kind,
            "title": link.text,
            "url": link.url,
            "mime_type": _mime_type_from_url(link.url),
            "page_hint": _page_hint_from_url(link.url),
            "sha256": None,
            "snapshot_date": snapshot_date,
        }
    return list(assets.values())


def _organ_type_from_link(text: str, url: str) -> str | None:
    lower = f"{text} {url}".lower()
    if "codcomision" in lower or text.lower().startswith(("comisión", "subcomisión", "ponencia")):
        return "commission"
    if text in {"Mesa", "Junta de Portavoces", "Diputación Permanente"}:
        return "governing_body"
    if "secretaría general" in lower:
        return "administration"
    return None


def _organ_code_from_url(url: str) -> str | None:
    params = parse_qs(urlparse(url).query)
    return _first_param(params, "_organos_codComision")


def _first_param(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def _known_organ_name(organ_code: str | None) -> str | None:
    known = {
        "1": "Mesa",
        "300": "Junta de Portavoces",
        "401": "Diputación Permanente - titulares",
        "402": "Diputación Permanente - suplentes",
    }
    return known.get(organ_code)


def _known_organ_type(organ_code: str | None) -> str | None:
    if organ_code in {"1", "300"}:
        return "governing_body"
    if organ_code in {"401", "402", "403", "404", "405", "406", "407"}:
        return "permanent_deputation"
    if organ_code and organ_code.isdigit():
        return "commission"
    return None


def _visible_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _profile_full_name(lines: list[str]) -> str:
    try:
        deputy_marker = lines.index("Diputados")
    except ValueError:
        deputy_marker = -1
    if deputy_marker >= 0:
        for line in lines[deputy_marker + 1 : deputy_marker + 12]:
            if re.fullmatch(r"[^,]{1,80}, [^,]{1,80}", line):
                return line
    for line in lines:
        if re.fullmatch(r"[^,]{1,80}, [^,]{1,80}", line):
            return line
    if lines:
        return lines[0].split(" - ", 1)[0]
    return ""


def _profile_birth_date(lines: list[str]) -> tuple[int | None, date | None]:
    for index, line in enumerate(lines):
        if line.startswith(("Nacido el ", "Nacida el ")):
            return index, _parse_java_date(line)
    return None, None


def _profile_birth_place(lines: list[str], birth_index: int | None) -> str | None:
    if birth_index is None or birth_index + 1 >= len(lines):
        return None
    next_line = lines[birth_index + 1]
    return next_line[3:] if next_line.startswith("en ") else None


def _profile_full_condition_date(lines: list[str]) -> date | None:
    for line in lines:
        if _fold(line).startswith("condicion plena:"):
            return _parse_java_date(line)
    return None


def _profile_constituency(lines: list[str]) -> str | None:
    for line in lines:
        match = re.fullmatch(r"Diputad[ao] por (.+)", line)
        if match:
            return match.group(1)
    return None


def _first_line_matching(lines: list[str], pattern: str) -> str | None:
    for line in lines:
        if re.search(pattern, line):
            return line
    return None


def _profile_group_code(lines: list[str]) -> str | None:
    for index, line in enumerate(lines[:-1]):
        if line == "(" and re.fullmatch(r"[A-Z][A-Z0-9-]{1,8}", lines[index + 1]):
            return lines[index + 1]
    for line in lines:
        match = re.search(r"\(([A-Z][A-Z0-9-]{1,8})\)", line)
        if match:
            return match.group(1)
    return None


def _profile_party(lines: list[str]) -> str | None:
    for index, line in enumerate(lines[:-1]):
        if "@congreso.es" in line:
            candidate = lines[index + 1]
            return candidate if candidate != "Ficha personal" else None
    return None


def _profile_legislature_history(lines: list[str]) -> str | None:
    for line in lines:
        if "Legislatura" in line and "(" in line and ")" in line:
            return line
    return None


def _financial_document_kind(text: str, url: str) -> str | None:
    lower = f"{text} {url}".lower()
    if "declaración de actividades" in lower or "registro_intereses" in lower:
        return "activities"
    if "bienes y rentas" in lower or "/docbienes/" in lower:
        return "assets_income"
    if "intereses económicos" in lower or "/docacteco/" in lower:
        return "economic_interests"
    return None


def _declaration_label(text: str) -> str:
    return re.sub(r"\s*\(.+\)\s*$", "", text).strip()


def _date_from_parenthesized_java_text(text: str) -> date | None:
    match = re.search(r"\(([^)]+)\)", text)
    return _parse_java_date(match.group(1)) if match else None


def _updated_at_from_text(text: str) -> Any:
    match = re.search(r"Información actualizada el (\d{1,2} de [a-záéíóú]+ de \d{4})", text, re.I)
    if not match:
        return None
    return _parse_long_spanish_date(match.group(1))


def _parse_long_spanish_date(value: str) -> Any:
    months = {
        "enero": "01",
        "febrero": "02",
        "marzo": "03",
        "abril": "04",
        "mayo": "05",
        "junio": "06",
        "julio": "07",
        "agosto": "08",
        "septiembre": "09",
        "setiembre": "09",
        "octubre": "10",
        "noviembre": "11",
        "diciembre": "12",
    }
    match = re.fullmatch(r"(\d{1,2}) de ([a-záéíóú]+) de (\d{4})", value.lower())
    if not match:
        return None
    day, month_name, year = match.groups()
    return parse_spanish_date(f"{int(day):02d}/{months[month_name]}/{year}")


def _historical_initiative_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("lista_iniciativas")
    if isinstance(raw, dict):
        return [item for item in raw.values() if isinstance(item, dict)]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _historical_initiative_group(source_dataset: str) -> str:
    return {
        "ProyectosDeLey": "Proyectos de ley",
        "ProposicionesDeLey": "Proposiciones de ley",
        "PropuestasDeReforma": "Propuestas de reforma de Estatutos de Autonomia",
    }.get(source_dataset, source_dataset)


def _historical_initiative_type(source_dataset: str) -> str:
    return {
        "ProyectosDeLey": "Proyecto de ley",
        "ProposicionesDeLey": "Proposicion de ley",
        "PropuestasDeReforma": "Propuesta de reforma de Estatuto de Autonomia",
    }.get(source_dataset, source_dataset)


def _historical_authors(value: Any) -> str | None:
    if isinstance(value, dict):
        names = [
            str(item.get("nombre")).strip()
            for item in value.values()
            if isinstance(item, dict) and item.get("nombre")
        ]
        return "; ".join(names) or None
    if isinstance(value, list):
        names = [
            str(item.get("nombre")).strip()
            for item in value
            if isinstance(item, dict) and item.get("nombre")
        ]
        return "; ".join(names) or None
    return str(value).strip() if value not in (None, "") else None


def _historical_initiative_detail_url(enlace: Any) -> str | None:
    if not isinstance(enlace, dict):
        return None
    path = enlace.get("url")
    if not path:
        return None
    params = {
        key: value for key, value in enlace.items() if key != "url" and value not in (None, "")
    }
    if params:
        from urllib.parse import urlencode

        return f"{urljoin('https://www.congreso.es', str(path))}?{urlencode(params)}"
    return urljoin("https://www.congreso.es", str(path))


def _approved_law_group(value: Any) -> str | None:
    return {
        "L": "Ley",
        "LO": "Ley Organica",
        "RD": "Real Decreto-ley",
        "RDL": "Real Decreto Legislativo",
    }.get(str(value or ""), str(value) if value not in (None, "") else None)


def _approved_law_pdf_url(filename: Any) -> str | None:
    if filename in (None, ""):
        return None
    return urljoin(
        "https://www.congreso.es/constitucion/ficheros/leyes_espa/",
        str(filename).strip(),
    )


def _absolute_congreso_url(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return urljoin("https://www.congreso.es", str(value))


def _query_param(url: str | None, key: str) -> str | None:
    if not url:
        return None
    values = parse_qs(urlparse(url).query).get(key)
    return values[0] if values else None


def _law_date_from_title(value: Any, *, default_year: str | None) -> date | None:
    if not isinstance(value, str):
        return None
    months = {
        "enero": "01",
        "febrero": "02",
        "marzo": "03",
        "abril": "04",
        "mayo": "05",
        "junio": "06",
        "julio": "07",
        "agosto": "08",
        "septiembre": "09",
        "setiembre": "09",
        "octubre": "10",
        "noviembre": "11",
        "diciembre": "12",
    }
    folded = _fold(value)
    matches = re.findall(
        r"\bde\s+(\d{1,2})\s+(?:de\s+)?([a-z]+)(?:\s+de\s+(\d{4}))?",
        folded,
    )
    for day, month_name, year in matches:
        month = months.get(month_name)
        effective_year = year or default_year
        if month and effective_year:
            return parse_spanish_date(f"{int(day):02d}/{month}/{effective_year}")
    return None


def _date_text(value: date | None) -> str | None:
    return value.strftime("%d/%m/%Y") if value else None


def _leg_label(value: Any) -> str | None:
    number = _leg_number(value)
    if number is None:
        return str(value) if value not in (None, "") else None
    return f"Leg.{number}"


def _leg_number(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper().replace("LEG.", "").replace("LEGISLATURA", "").strip()
    if text.startswith("LEG") and text[3:].isdigit():
        text = text[3:]
    if text.isdigit():
        return int(text)
    if text == "C":
        return 0
    return {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
        "XIII": 13,
        "XIV": 14,
        "XV": 15,
    }.get(text)


def _legislature_from_date(value: date | None) -> str | None:
    if value is None:
        return None
    boundaries = (
        (date(2023, 8, 17), "XV"),
        (date(2019, 12, 3), "XIV"),
        (date(2019, 5, 21), "XIII"),
        (date(2016, 7, 19), "XII"),
        (date(2016, 1, 13), "XI"),
        (date(2011, 12, 13), "X"),
        (date(2008, 4, 1), "IX"),
        (date(2004, 4, 2), "VIII"),
        (date(2000, 4, 5), "VII"),
        (date(1996, 3, 27), "VI"),
        (date(1993, 4, 13), "V"),
        (date(1989, 11, 21), "IV"),
        (date(1986, 7, 15), "III"),
        (date(1982, 11, 18), "II"),
        (date(1979, 3, 23), "I"),
        (date(1977, 7, 13), "0"),
    )
    for start, legislature in boundaries:
        if value >= start:
            return legislature
    return None


def _parse_java_date(value: str) -> date | None:
    months = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    match = re.search(r"\b[A-Z][a-z]{2}\s+([A-Z][a-z]{2})\s+(\d{1,2}).*(\d{4})\b", value)
    if not match:
        return parse_spanish_date(value)
    month_name, day, year = match.groups()
    return date(int(year), months[month_name], int(day))


def _mime_type_from_url(url: str) -> str | None:
    path = url.lower().split("#", 1)[0].split("?", 1)[0]
    if path.endswith(".pdf"):
        return "application/pdf"
    if path.endswith(".mp4"):
        return "video/mp4"
    if path.endswith(".xml"):
        return "application/xml"
    return None


def _page_hint_from_url(url: str) -> int | None:
    match = re.search(r"#page=(\d+)", url, re.I)
    return int(match.group(1)) if match else None


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return None


def _first_date(item: dict[str, Any], *keys: str) -> date | None:
    value = _first_present(item, *keys)
    if not isinstance(value, str):
        return None
    return parse_spanish_date(value) or _parse_long_spanish_date(value)


def _deputy_names(item: dict[str, Any]) -> tuple[str | None, str | None, str]:
    first_name = item.get("nombre") if item.get("apellidos") else None
    last_names = item.get("apellidos")
    if first_name and last_names:
        return first_name, last_names, f"{last_names}, {first_name}"

    full_name = (
        item.get("apellidosnombre") or item.get("nombrecompleto") or item.get("nombre") or ""
    )
    if not last_names and "," in full_name:
        last_names, first_name = [part.strip() or None for part in full_name.split(",", 1)]
    return first_name, last_names, full_name


def _gender_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "h", "hombre", "varon", "varón", "masculino", "v"}:
        return 1
    if text in {"2", "f", "mujer", "femenino"}:
        return 2
    return None


def _age_at(birth_date: date | None, snapshot_date: str) -> int | None:
    if birth_date is None:
        return None
    try:
        current = date.fromisoformat(snapshot_date[:10])
    except ValueError:
        return None
    age = current.year - birth_date.year
    if (current.month, current.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()

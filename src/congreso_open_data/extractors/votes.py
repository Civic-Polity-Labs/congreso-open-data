from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from lxml import html as lxml_html

from congreso_open_data.catalog import DatasetResource, snapshot_token_from_url
from congreso_open_data.durable_io import write_json_atomically
from congreso_open_data.http import CongresoHttpClient

VOTE_LEGISLATURES = ("10", "11", "12", "13", "14", "15")
VOTE_DISCOVERY_STATE_VERSION = 4
VoteDiscoveryProgress = Callable[[dict[str, Any]], None]
_ROMAN_BY_NUMBER = {
    "10": "X",
    "11": "XI",
    "12": "XII",
    "13": "XIII",
    "14": "XIV",
    "15": "XV",
}
_CURRENT_LEGISLATURE = "XV"


def discover_historical_vote_resources(
    *,
    client: CongresoHttpClient | None = None,
    legislatures: tuple[str, ...] = VOTE_LEGISLATURES,
    progress: VoteDiscoveryProgress | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    checkpoint_interval: int = 10,
    sample_dates_per_legislature: int | None = None,
) -> list[DatasetResource]:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if sample_dates_per_legislature is not None and sample_dates_per_legislature <= 0:
        raise ValueError("sample_dates_per_legislature must be positive")
    client = client or CongresoHttpClient()
    state = _load_discovery_state(
        checkpoint_path=checkpoint_path,
        legislatures=legislatures,
        resume=resume,
        sample_dates_per_legislature=sample_dates_per_legislature,
    )
    resources = {
        resource.url: resource
        for resource in (DatasetResource(**item) for item in state.get("resources", []))
    }
    unstructured_dates = {
        f"{item['legislature']}|{item['vote_date']}": item
        for item in state.get("unstructured_dates", [])
    }
    completed_dates = set(state.get("completed_dates", []))
    dates_by_legislature = state.setdefault("dates_by_legislature", {})
    calendar_dates_by_legislature = state.setdefault(
        "calendar_dates_by_legislature",
        {},
    )
    calendar_pages = state.setdefault("calendar_pages", {})
    date_pages = state.setdefault("date_pages", {})
    source_variants = state.setdefault("source_variants", {})
    dates_since_checkpoint = 0
    for legislature in legislatures:
        if progress:
            progress(
                {
                    "event": "legislature_started",
                    "legislature": legislature,
                    "dates_completed": 0,
                    "dates_planned": None,
                    "resources": len(resources),
                }
            )
        cached_dates = dates_by_legislature.get(legislature)
        if cached_dates is None:
            calendar_url = _vote_page_url(legislature=legislature)
            calendar_result = client.get(calendar_url)
            calendar_dates = _vote_dates_from_content(
                calendar_result.content,
                legislature=legislature,
            )
            calendar_dates_by_legislature[legislature] = list(calendar_dates)
            dates = (
                calendar_dates[-sample_dates_per_legislature:]
                if sample_dates_per_legislature is not None
                else calendar_dates
            )
            dates_by_legislature[legislature] = list(dates)
            calendar_pages[legislature] = _source_page_record(
                legislature=legislature,
                vote_date=None,
                page_url=calendar_result.url,
                content=calendar_result.content,
                resource_urls=[],
            )
            _write_discovery_state(
                checkpoint_path,
                state,
                resources,
                completed_dates,
                unstructured_dates,
            )
        else:
            dates = tuple(str(value) for value in cached_dates)
        if progress:
            progress(
                {
                    "event": "calendar_discovered",
                    "legislature": legislature,
                    "dates_completed": 0,
                    "dates_planned": len(dates),
                    "resources": len(resources),
                }
            )
        for index, vote_date in enumerate(dates, start=1):
            date_key = f"{legislature}|{vote_date}"
            if date_key in completed_dates:
                if progress:
                    progress(
                        {
                            "event": "date_reused",
                            "legislature": legislature,
                            "vote_date": vote_date,
                            "dates_completed": index,
                            "dates_planned": len(dates),
                            "resources": len(resources),
                        }
                    )
                continue
            html = client.get(_vote_page_url(legislature=legislature, target_date=vote_date))
            all_date_resources = vote_source_resources_from_html(
                html.content,
                legislature=legislature,
            )
            date_resources = []
            for resource in all_date_resources:
                context = vote_resource_context(resource.url)
                if resource.format == "json" and context is not None and context[2] == vote_date:
                    date_resources.append(resource)
            unstructured = None
            if not date_resources:
                unstructured = unstructured_vote_date_from_html(
                    html.content,
                    legislature=legislature,
                    vote_date=vote_date,
                    page_url=html.url,
                )
                if unstructured is None:
                    raise ValueError(
                        "Historical vote date page has neither matching official JSON "
                        "nor an auditable official summary: "
                        f"legislature={legislature} date={vote_date}"
                    )
                unstructured_dates[date_key] = unstructured
            for resource in date_resources:
                resources[resource.url] = resource
            for resource in all_date_resources:
                source_variants[resource.url] = asdict(resource)
            date_pages[date_key] = _source_page_record(
                legislature=legislature,
                vote_date=vote_date,
                page_url=html.url,
                content=html.content,
                resource_urls=[resource.url for resource in all_date_resources],
            )
            completed_dates.add(date_key)
            dates_since_checkpoint += 1
            if dates_since_checkpoint >= checkpoint_interval or index == len(dates):
                _write_discovery_state(
                    checkpoint_path,
                    state,
                    resources,
                    completed_dates,
                    unstructured_dates,
                )
                dates_since_checkpoint = 0
            if progress:
                progress(
                    {
                        "event": ("date_unstructured" if unstructured else "date_discovered"),
                        "legislature": legislature,
                        "vote_date": vote_date,
                        "dates_completed": index,
                        "dates_planned": len(dates),
                        "resources": len(resources),
                    }
                )
    state["status"] = "completed"
    _write_discovery_state(
        checkpoint_path,
        state,
        resources,
        completed_dates,
        unstructured_dates,
    )
    return list(resources.values())


def _load_discovery_state(
    *,
    checkpoint_path: Path | None,
    legislatures: tuple[str, ...],
    resume: bool,
    sample_dates_per_legislature: int | None,
) -> dict[str, Any]:
    expected = {
        "version": VOTE_DISCOVERY_STATE_VERSION,
        "legislatures": list(legislatures),
        "sample_dates_per_legislature": sample_dates_per_legislature,
    }
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Historical vote discovery checkpoint does not match the request")
        _validate_discovery_state(payload, legislatures=legislatures)
        return payload
    return expected | {
        "status": "running",
        "dates_by_legislature": {},
        "calendar_dates_by_legislature": {},
        "completed_dates": [],
        "resources": [],
        "unstructured_dates": [],
        "calendar_pages": {},
        "date_pages": {},
        "source_variants": {},
    }


def _write_discovery_state(
    checkpoint_path: Path | None,
    state: dict[str, Any],
    resources: dict[str, DatasetResource],
    completed_dates: set[str],
    unstructured_dates: dict[str, dict[str, Any]],
) -> None:
    if checkpoint_path is None:
        return
    payload = state | {
        "completed_dates": sorted(completed_dates),
        "resources": [asdict(resource) for resource in resources.values()],
        "unstructured_dates": [unstructured_dates[key] for key in sorted(unstructured_dates)],
    }
    _validate_discovery_state(
        payload,
        legislatures=tuple(str(value) for value in payload["legislatures"]),
    )
    write_json_atomically(checkpoint_path, payload)


def discover_vote_dates(
    *,
    client: CongresoHttpClient,
    legislature: str,
) -> tuple[str, ...]:
    content = client.get(_vote_page_url(legislature=legislature)).content
    return _vote_dates_from_content(content, legislature=legislature)


def _vote_dates_from_content(content: bytes, *, legislature: str) -> tuple[str, ...]:
    html = content.decode(
        "utf-8",
        errors="replace",
    )
    match = re.search(r"var\s+diasVotaciones\s*=\s*\[([^\]]*)\]", html)
    if not match:
        raise ValueError(
            f"Historical vote calendar is missing diasVotaciones: legislature={legislature}"
        )
    dates = tuple(dict.fromkeys(re.findall(r"\d{8}", match.group(1))))
    if not dates:
        raise ValueError(f"Historical vote calendar has no vote dates: legislature={legislature}")
    return dates


def _vote_resources_from_html(content: bytes, *, legislature: str) -> list[DatasetResource]:
    return [
        resource
        for resource in vote_source_resources_from_html(
            content,
            legislature=legislature,
        )
        if resource.dataset == "Votacion" and resource.format == "json"
    ]


def vote_source_resources_from_html(
    content: bytes,
    *,
    legislature: str,
) -> list[DatasetResource]:
    """Return every official per-vote and session artifact exposed by a date page."""

    try:
        document = lxml_html.fromstring(content.decode("utf-8", errors="replace"))
    except (ValueError, lxml_html.ParserError):
        return []
    raw_urls = [
        *document.xpath("//a[@href]/@href"),
        *document.xpath("//img[@src]/@src"),
    ]
    resources: dict[str, DatasetResource] = {}
    for raw_url in raw_urls:
        absolute = urljoin("https://www.congreso.es", str(raw_url))
        if f"/opendata/votaciones/Leg{legislature}/" not in absolute:
            continue
        suffix = absolute.split("#", 1)[0].rsplit(".", 1)[-1].casefold()
        if suffix not in {"json", "xml", "pdf", "png", "zip"}:
            continue
        vote_context = vote_resource_context(absolute)
        session_context = vote_session_resource_context(absolute)
        if vote_context is not None:
            context_legislature, session, _, vote_number = vote_context
            if context_legislature != legislature or suffix == "zip":
                continue
            dataset = "Votacion"
        elif session_context is not None and suffix == "zip":
            context_legislature, session, _ = session_context
            if context_legislature != legislature:
                continue
            dataset = "SesionVotaciones"
            vote_number = None
        else:
            continue
        resources[absolute] = DatasetResource(
            family="votaciones",
            dataset=dataset,
            format=suffix,
            url=absolute,
            snapshot_token=snapshot_token_from_url(absolute),
            legislature=f"Leg{legislature}",
            session=session,
            vote_number=vote_number,
        )
    return [resources[url] for url in sorted(resources)]


def vote_resource_context(url: str) -> tuple[str, str, str, str] | None:
    """Return legislature, session, date, and vote number from an official vote URL."""

    match = re.search(
        r"/votaciones/Leg(\d+)/Sesion(\d+)/(\d{8})/Votacion(\d+)(?:/|$)",
        url,
        flags=re.IGNORECASE,
    )
    return match.groups() if match else None


def vote_session_resource_context(url: str) -> tuple[str, str, str] | None:
    match = re.search(
        r"/votaciones/Leg(\d+)/Sesion(\d+)/(\d{8})/(?!Votacion\d+/)",
        url,
        flags=re.IGNORECASE,
    )
    return match.groups() if match else None


def vote_supporting_source_resources(
    discovery_checkpoint: dict[str, Any],
) -> list[DatasetResource]:
    """Build a deterministic plan for HTML and non-JSON vote source variants."""

    resources: dict[str, DatasetResource] = {}
    for legislature, page in discovery_checkpoint.get("calendar_pages", {}).items():
        url = str(page["page_url"])
        resources[url] = DatasetResource(
            family="votaciones",
            dataset="VoteCalendarPage",
            format="html",
            url=url,
            snapshot_token=f"Leg{legislature}-calendar",
            legislature=f"Leg{legislature}",
        )
    for date_key, page in discovery_checkpoint.get("date_pages", {}).items():
        legislature, vote_date = date_key.split("|", 1)
        url = str(page["page_url"])
        resources[url] = DatasetResource(
            family="votaciones",
            dataset="VoteDatePage",
            format="html",
            url=url,
            snapshot_token=f"Leg{legislature}-{vote_date}-date-page",
            legislature=f"Leg{legislature}",
        )
    for raw in discovery_checkpoint.get("source_variants", {}).values():
        resource = DatasetResource(**raw)
        if resource.format != "json":
            resources[resource.url] = resource
    return [resources[url] for url in sorted(resources)]


def _source_page_record(
    *,
    legislature: str,
    vote_date: str | None,
    page_url: str,
    content: bytes,
    resource_urls: list[str],
) -> dict[str, Any]:
    return {
        "legislature": legislature,
        "vote_date": vote_date,
        "page_url": page_url,
        "page_sha256": hashlib.sha256(content).hexdigest(),
        "page_bytes": len(content),
        "resource_urls": sorted(set(resource_urls)),
    }


def unstructured_vote_date_from_html(
    content: bytes,
    *,
    legislature: str,
    vote_date: str,
    page_url: str,
) -> dict[str, Any] | None:
    """Profile an official vote summary when no structured JSON was published."""

    try:
        document = lxml_html.fromstring(content.decode("utf-8", errors="replace"))
    except (ValueError, lxml_html.ParserError):
        return None
    body_text = " ".join(document.xpath("//body//text()[normalize-space()]"))
    session_match = re.search(
        r"Sesi[oó]n(?:\s+Plenaria)?\s+n[uú]mero\s+(\d+)",
        body_text,
        flags=re.IGNORECASE,
    )
    summaries: list[dict[str, Any]] = []
    for ordinal, result in enumerate(
        document.xpath(
            "//div[contains(concat(' ', normalize-space(@class), ' '), ' result_vot ')]"
        ),
        start=1,
    ):
        result_text = " ".join(result.xpath(".//text()[normalize-space()]"))
        counts = {
            "yes_votes": _summary_integer(result_text, ("si",)),
            "no_votes": _summary_integer(result_text, ("no",)),
            "abstentions": _summary_integer(result_text, ("abstenciones",)),
        }
        if any(value is None for value in counts.values()):
            continue
        container_items = result.xpath("ancestor::div[starts-with(@id, 'accordionEst')][1]")
        container = container_items[0] if container_items else result.getparent()
        title = _normalized_node_text(container, ".//h5[1]//text()")
        initiative = _normalized_node_text(
            container,
            ".//a[contains(concat(' ', normalize-space(@class), ' '), ' n_exp ')][1]//text()",
        )
        detail_urls = [
            urljoin(page_url, str(value))
            for value in container.xpath(".//a[contains(@href, '.PDF')]/@href")
        ]
        image_urls = [
            urljoin(page_url, str(value))
            for value in container.xpath(".//img[contains(@src, '/votaciones/')]/@src")
        ]
        summaries.append(
            {
                "ordinal": ordinal,
                "title": title,
                "initiative_reference_raw": initiative,
                **counts,
                "detail_pdf_url": detail_urls[0] if detail_urls else None,
                "image_url": image_urls[0] if image_urls else None,
            }
        )
    if not summaries:
        return None
    alternatives: list[dict[str, str]] = []
    for format_name, xpath in (
        ("zip", "//a[contains(@href, '.zip')]/@href"),
        ("pdf", "//a[contains(@href, '.PDF')]/@href"),
        ("png", "//img[contains(@src, '/votaciones/')]/@src"),
    ):
        for raw_url in document.xpath(xpath):
            absolute = urljoin(page_url, str(raw_url))
            item = {"format": format_name, "url": absolute}
            if item not in alternatives:
                alternatives.append(item)
    record = {
        "legislature": legislature,
        "vote_date": vote_date,
        "page_url": page_url,
        "page_sha256": hashlib.sha256(content).hexdigest(),
        "page_bytes": len(content),
        "classification": "official_summary_without_structured_json",
        "session_number": int(session_match.group(1)) if session_match else None,
        "summary_events": summaries,
        "alternative_resources": alternatives,
    }
    return record if validate_unstructured_vote_date(record) else None


def validate_unstructured_vote_date(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    legislature = str(item.get("legislature") or "")
    vote_date = str(item.get("vote_date") or "")
    summaries = item.get("summary_events")
    alternatives = item.get("alternative_resources")
    if (
        not legislature.isdigit()
        or not _valid_vote_date(vote_date)
        or item.get("classification") != "official_summary_without_structured_json"
        or not isinstance(item.get("page_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", item["page_sha256"])
        or not isinstance(item.get("page_bytes"), int)
        or item["page_bytes"] <= 0
        or not isinstance(summaries, list)
        or not summaries
        or not isinstance(alternatives, list)
        or not alternatives
    ):
        return False
    ordinals: set[int] = set()
    for summary in summaries:
        if not isinstance(summary, dict):
            return False
        ordinal = summary.get("ordinal")
        counts = [
            summary.get("yes_votes"),
            summary.get("no_votes"),
            summary.get("abstentions"),
        ]
        if (
            not isinstance(ordinal, int)
            or ordinal <= 0
            or ordinal in ordinals
            or not all(isinstance(value, int) and value >= 0 for value in counts)
            or not str(summary.get("title") or "").strip()
        ):
            return False
        ordinals.add(ordinal)
    return all(
        isinstance(artifact, dict)
        and artifact.get("format") in {"zip", "pdf", "png"}
        and str(artifact.get("url") or "").startswith("https://www.congreso.es/")
        for artifact in alternatives
    )


def _summary_integer(text: str, labels: tuple[str, ...]) -> int | None:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    for label in labels:
        match = re.search(rf"\b{re.escape(label)}\s*:\s*(\d+)\b", normalized, re.I)
        if match:
            return int(match.group(1))
    return None


def _normalized_node_text(node: Any, xpath: str) -> str | None:
    text = " ".join(str(value).strip() for value in node.xpath(xpath) if str(value).strip())
    return " ".join(text.split()) or None


def _validate_discovery_state(
    payload: dict[str, Any],
    *,
    legislatures: tuple[str, ...],
) -> None:
    status = payload.get("status")
    if status not in {"running", "completed"}:
        raise ValueError("Historical vote discovery checkpoint has an invalid status")
    dates_by_legislature = payload.get("dates_by_legislature")
    calendar_dates_by_legislature = payload.get("calendar_dates_by_legislature")
    completed_dates = payload.get("completed_dates")
    resource_items = payload.get("resources")
    unstructured_items = payload.get("unstructured_dates")
    calendar_pages = payload.get("calendar_pages")
    date_pages = payload.get("date_pages")
    source_variants = payload.get("source_variants")
    if not isinstance(dates_by_legislature, dict) or not isinstance(
        calendar_dates_by_legislature,
        dict,
    ):
        raise ValueError("Historical vote discovery checkpoint has invalid calendars")
    if not isinstance(completed_dates, list) or len(completed_dates) != len(set(completed_dates)):
        raise ValueError("Historical vote discovery checkpoint has invalid completed dates")
    if not isinstance(resource_items, list):
        raise ValueError("Historical vote discovery checkpoint has invalid resources")
    if not isinstance(unstructured_items, list):
        raise ValueError("Historical vote discovery checkpoint has invalid unstructured dates")
    if (
        not isinstance(calendar_pages, dict)
        or not isinstance(date_pages, dict)
        or not isinstance(source_variants, dict)
    ):
        raise ValueError("Historical vote discovery checkpoint has invalid source-page lineage")

    requested = set(legislatures)
    sample_size = payload.get("sample_dates_per_legislature")
    if sample_size is not None and (
        not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0
    ):
        raise ValueError("Historical vote discovery checkpoint has an invalid sample size")
    if not set(dates_by_legislature).issubset(requested):
        raise ValueError("Historical vote discovery checkpoint has unexpected legislatures")
    valid_date_keys: set[str] = set()
    for legislature, raw_dates in dates_by_legislature.items():
        if not isinstance(raw_dates, list) or not raw_dates:
            raise ValueError("Historical vote discovery checkpoint has an empty calendar")
        dates = [str(value) for value in raw_dates]
        if len(dates) != len(set(dates)) or any(not _valid_vote_date(value) for value in dates):
            raise ValueError("Historical vote discovery checkpoint has invalid calendar dates")
        valid_date_keys.update(f"{legislature}|{value}" for value in dates)
        raw_calendar_dates = calendar_dates_by_legislature.get(legislature)
        if not isinstance(raw_calendar_dates, list) or not raw_calendar_dates:
            raise ValueError("Historical vote discovery checkpoint has invalid full calendars")
        calendar_dates = [str(value) for value in raw_calendar_dates]
        if (
            len(calendar_dates) != len(set(calendar_dates))
            or any(not _valid_vote_date(value) for value in calendar_dates)
            or not set(dates).issubset(calendar_dates)
        ):
            raise ValueError("Historical vote discovery checkpoint has invalid full calendars")
        expected_dates = (
            calendar_dates[-int(sample_size) :] if sample_size is not None else calendar_dates
        )
        if dates != expected_dates:
            raise ValueError(
                "Historical vote discovery checkpoint sample calendar is not deterministic"
            )
    if set(calendar_dates_by_legislature) != set(dates_by_legislature):
        raise ValueError("Historical vote discovery checkpoint has incomplete full calendars")
    if set(calendar_pages) != set(dates_by_legislature):
        raise ValueError(
            "Historical vote discovery checkpoint has incomplete calendar-page lineage"
        )
    for legislature, page in calendar_pages.items():
        if not _valid_source_page_record(
            page,
            legislature=legislature,
            vote_date=None,
        ):
            raise ValueError(
                "Historical vote discovery checkpoint has invalid calendar-page lineage"
            )

    completed = {str(value) for value in completed_dates}
    if not completed.issubset(valid_date_keys):
        raise ValueError("Historical vote discovery checkpoint has unplanned completed dates")
    if set(date_pages) != completed:
        raise ValueError("Historical vote discovery checkpoint has incomplete date-page lineage")
    date_page_resource_urls: set[str] = set()
    for date_key, page in date_pages.items():
        legislature, vote_date = date_key.split("|", 1)
        if not _valid_source_page_record(
            page,
            legislature=legislature,
            vote_date=vote_date,
        ):
            raise ValueError("Historical vote discovery checkpoint has invalid date-page lineage")
        date_page_resource_urls.update(str(value) for value in page["resource_urls"])

    variant_urls: set[str] = set()
    for url, raw in source_variants.items():
        if not isinstance(raw, dict):
            raise ValueError("Historical vote discovery checkpoint has malformed source variants")
        try:
            resource = DatasetResource(**raw)
        except TypeError as exc:
            raise ValueError(
                "Historical vote discovery checkpoint has malformed source variants"
            ) from exc
        vote_context = vote_resource_context(resource.url)
        session_context = vote_session_resource_context(resource.url)
        if (
            url != resource.url
            or resource.family != "votaciones"
            or resource.format not in {"json", "xml", "pdf", "png", "zip"}
            or (
                resource.dataset == "Votacion"
                and (
                    vote_context is None
                    or resource.format == "zip"
                    or resource.legislature != f"Leg{vote_context[0]}"
                    or str(resource.session) != vote_context[1]
                    or int(str(resource.vote_number or "-1")) != int(vote_context[3])
                )
            )
            or (
                resource.dataset == "SesionVotaciones"
                and (
                    session_context is None
                    or resource.format != "zip"
                    or resource.legislature != f"Leg{session_context[0]}"
                    or str(resource.session) != session_context[1]
                    or resource.vote_number is not None
                )
            )
            or resource.dataset not in {"Votacion", "SesionVotaciones"}
        ):
            raise ValueError("Historical vote discovery checkpoint has invalid source variants")
        variant_urls.add(resource.url)
    if variant_urls != date_page_resource_urls:
        raise ValueError(
            "Historical vote discovery checkpoint source variants do not match date pages"
        )

    resource_dates: set[str] = set()
    resource_urls: set[str] = set()
    for item in resource_items:
        if not isinstance(item, dict):
            raise ValueError("Historical vote discovery checkpoint has malformed resources")
        try:
            resource = DatasetResource(**item)
        except TypeError as exc:
            raise ValueError(
                "Historical vote discovery checkpoint has malformed resources"
            ) from exc
        context = vote_resource_context(resource.url)
        if (
            context is None
            or context[0] not in requested
            or resource.family != "votaciones"
            or resource.dataset != "Votacion"
            or resource.format != "json"
            or resource.legislature != f"Leg{context[0]}"
            or resource.session != context[1]
            or int(str(resource.vote_number or "-1")) != int(context[3])
        ):
            raise ValueError("Historical vote discovery checkpoint has invalid resources")
        date_key = f"{context[0]}|{context[2]}"
        if date_key not in completed:
            raise ValueError("Historical vote discovery checkpoint has resources for open dates")
        if resource.url in resource_urls:
            raise ValueError("Historical vote discovery checkpoint has duplicate resources")
        resource_urls.add(resource.url)
        resource_dates.add(date_key)
        variant = source_variants.get(resource.url)
        if variant != item:
            raise ValueError(
                "Historical vote discovery checkpoint JSON resources lack variant lineage"
            )

    unstructured_date_keys: set[str] = set()
    for item in unstructured_items:
        if not validate_unstructured_vote_date(item):
            raise ValueError("Historical vote discovery checkpoint has invalid unstructured dates")
        date_key = f"{item['legislature']}|{item['vote_date']}"
        if date_key not in completed or date_key in unstructured_date_keys:
            raise ValueError("Historical vote discovery checkpoint has invalid unstructured dates")
        unstructured_date_keys.add(date_key)

    if not (resource_dates | unstructured_date_keys).issuperset(completed):
        raise ValueError(
            "Historical vote discovery checkpoint has completed dates without coverage"
        )
    if status == "completed" and (
        set(dates_by_legislature) != requested or completed != valid_date_keys
    ):
        raise ValueError("Historical vote discovery checkpoint is incompletely marked completed")


def _valid_vote_date(value: str) -> bool:
    try:
        return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        return False


def _valid_source_page_record(
    item: Any,
    *,
    legislature: str,
    vote_date: str | None,
) -> bool:
    return bool(
        isinstance(item, dict)
        and item.get("legislature") == legislature
        and item.get("vote_date") == vote_date
        and str(item.get("page_url") or "").startswith("https://www.congreso.es/")
        and isinstance(item.get("page_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", item["page_sha256"])
        and isinstance(item.get("page_bytes"), int)
        and item["page_bytes"] > 0
        and isinstance(item.get("resource_urls"), list)
        and len(item["resource_urls"]) == len(set(item["resource_urls"]))
        and all(
            isinstance(url, str) and url.startswith("https://www.congreso.es/")
            for url in item["resource_urls"]
        )
    )


def _vote_page_url(*, legislature: str, target_date: str | None = None) -> str:
    roman = _ROMAN_BY_NUMBER[legislature]
    url = (
        "https://www.congreso.es/es/opendata/votaciones"
        "?p_p_id=votaciones&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view"
        f"&targetLegislatura={roman}&currentLegislatura={_CURRENT_LEGISLATURE}"
    )
    if target_date:
        url += f"&targetDate={target_date[6:8]}/{target_date[4:6]}/{target_date[:4]}"
    return url

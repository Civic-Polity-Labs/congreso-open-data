from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from congreso_open_data.durable_io import write_json_atomically
from congreso_open_data.html import parse_visible_html
from congreso_open_data.http import CongresoHttpClient

BASE_URL = "https://www.congreso.es"
OPEN_DATA_PAGES = {
    "votaciones": "https://www.congreso.es/es/opendata/votaciones",
    "diputados": "https://www.congreso.es/es/opendata/diputados",
    "iniciativas": "https://www.congreso.es/es/opendata/iniciativas",
    "intervenciones": "https://www.congreso.es/es/opendata/intervenciones",
    "organos": "https://www.congreso.es/es/opendata/organos",
}


@dataclass(frozen=True)
class DatasetResource:
    family: str
    dataset: str
    format: str
    url: str
    snapshot_token: str | None = None
    legislature: str | None = None
    session: str | None = None
    vote_number: str | None = None
    post_data: dict[str, Any] | None = None


def dataset_name_from_url(url: str) -> str:
    filename = url.rsplit("/", 1)[-1]
    stem = re.sub(r"\.(json|csv|xml|zip|pdf|png)$", "", filename, flags=re.I)
    return stem.split("__", 1)[0]


def snapshot_token_from_url(url: str) -> str | None:
    match = re.search(r"__(\d{14})\.(?:json|csv|xml)$", url, flags=re.I)
    if match:
        return match.group(1)
    match = re.search(r"VOT_(\d{14})\.(?:json|xml|pdf|png|zip)$", url, flags=re.I)
    return match.group(1) if match else None


def _resource_from_link(family: str, text: str, href: str) -> DatasetResource | None:
    absolute = urljoin(BASE_URL, href)
    if "/webpublica/opendata/" not in absolute:
        return None
    suffix = absolute.rsplit(".", 1)[-1].lower()
    if suffix not in {"json", "csv", "xml", "zip", "pdf", "png"}:
        return None
    dataset = dataset_name_from_url(absolute)
    legislature = session = vote_number = None
    vote_match = re.search(r"/(Leg\d+)/Sesion(\d+)/(\d{8})/Votacion(\d+)/", absolute)
    if vote_match:
        legislature, session, _, vote_number = vote_match.groups()
        dataset = "Votacion"
    session_match = re.search(r"/(Leg\d+)/Sesion(\d+)/(\d{8})/", absolute)
    if session_match and suffix == "zip":
        legislature, session, _ = session_match.groups()
        dataset = "SesionVotaciones"
    return DatasetResource(
        family=family,
        dataset=dataset,
        format=(
            text.lower() if text.lower() in {"json", "csv", "xml", "zip", "pdf", "png"} else suffix
        ),
        url=absolute,
        snapshot_token=snapshot_token_from_url(absolute),
        legislature=legislature,
        session=session,
        vote_number=vote_number,
    )


def discover_catalog(client: CongresoHttpClient | None = None) -> list[DatasetResource]:
    client = client or CongresoHttpClient()
    resources: list[DatasetResource] = []
    seen: set[tuple[str, str, str]] = set()
    for family, page_url in OPEN_DATA_PAGES.items():
        parsed = parse_visible_html(client.get(page_url).content, base_url=page_url)
        for link in parsed.links:
            resource = _resource_from_link(family, link.text, link.url)
            if resource is None:
                continue
            identity = (resource.family, resource.url, resource.format)
            if identity not in seen:
                resources.append(resource)
                seen.add(identity)
    return resources


def write_catalog(resources: list[DatasetResource], output: Path) -> None:
    write_json_atomically(output, [asdict(resource) for resource in resources])


def read_catalog(path: Path) -> list[DatasetResource]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("resources")
    if not isinstance(raw, list):
        raise ValueError("Catalog must be a resource list or contain a resources list")
    if any(not isinstance(item, dict) for item in raw):
        raise ValueError("Every catalog resource must be a JSON object")
    return [DatasetResource(**item) for item in raw]

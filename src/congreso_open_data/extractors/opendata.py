from __future__ import annotations

import json
from pathlib import Path

from congreso_open_data.catalog import DatasetResource, discover_catalog
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.storage import (
    BronzeManifest,
    content_matches_format_contract,
    persist_bronze,
)


def select_resources(
    *,
    family: str,
    dataset: str | None = None,
    format_name: str = "json",
    catalog_path: Path | None = None,
) -> list[DatasetResource]:
    if catalog_path:
        from congreso_open_data.catalog import read_catalog

        resources = read_catalog(catalog_path)
    else:
        resources = discover_catalog()
    return [
        resource
        for resource in resources
        if resource.family == family
        and resource.format == format_name
        and (dataset is None or resource.dataset == dataset)
    ]


def extract_resource(
    *,
    resource: DatasetResource,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
) -> BronzeManifest:
    client = client or CongresoHttpClient()
    if resource.post_data is not None:
        result = client.post(
            resource.url,
            data={key: str(value) for key, value in resource.post_data.items()},
        )
    else:
        result = client.get(resource.url)
    if resource.format.casefold() == "json":
        try:
            payload = json.loads(result.content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Official resource returned invalid JSON: {resource.url}") from exc
        if not isinstance(payload, (dict, list)):
            raise ValueError(f"Official JSON resource has an unsupported root type: {resource.url}")
    elif not content_matches_format_contract(
        content=result.content,
        format_name=resource.format,
    ):
        raise ValueError(
            f"Official resource returned invalid {resource.format.upper()}: {resource.url}"
        )
    return persist_bronze(root=output_root, resource=resource, run_date=run_date, result=result)

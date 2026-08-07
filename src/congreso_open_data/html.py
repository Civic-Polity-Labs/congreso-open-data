from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import urljoin

from lxml import html as lxml_html

from congreso_open_data.normalization import normalize_text

BASE_URL = "https://www.congreso.es"
SKIP_XPATH = "//script|//style|//noscript|//template|//svg"
INTERVENTION_TEXT_XPATH = (
    "//*[contains(concat(' ', normalize-space(@class), ' '), ' textoIntegro ')]"
)


@dataclass(frozen=True)
class HtmlLink:
    text: str
    url: str


@dataclass(frozen=True)
class VisibleHtmlDocument:
    visible_text: str
    links: list[HtmlLink]
    content_selector: str = "document"
    quality_issues: tuple[str, ...] = field(default_factory=tuple)
    source_projection_json: str | None = None


def parse_visible_html(content: bytes | str, *, base_url: str = BASE_URL) -> VisibleHtmlDocument:
    document = lxml_html.fromstring(content)
    for node in document.xpath(SKIP_XPATH):
        node.drop_tree()
    links = [
        HtmlLink(
            text=normalize_text(" ".join(anchor.itertext())),
            url=urljoin(base_url, anchor.get("href")),
        )
        for anchor in document.xpath("//a[@href]")
        if normalize_text(" ".join(anchor.itertext())) and anchor.get("href")
    ]
    transcript_nodes = document.xpath(INTERVENTION_TEXT_XPATH)
    quality_issues: list[str] = []
    if len(transcript_nodes) > 1:
        quality_issues.append("multiple_intervention_text_containers")
    scope = transcript_nodes[0] if len(transcript_nodes) == 1 else document
    content_selector = ".textoIntegro" if len(transcript_nodes) == 1 else "document"
    text_nodes = scope.xpath(".//text()[normalize-space()]")
    if scope is document:
        text_nodes = document.xpath("//body//text()[normalize-space()]")
    if not text_nodes:
        text_nodes = document.xpath("//text()[normalize-space()]")
    lines: list[str] = []
    projection: list[dict[str, object]] = []
    cursor = 0
    tree = document.getroottree()
    for node in text_nodes:
        line = normalize_text(str(node))
        if not line:
            continue
        if lines:
            cursor += 1
        start = cursor
        lines.append(line)
        cursor += len(line)
        parent = node.getparent() if hasattr(node, "getparent") else None
        projection.append(
            {
                "content_start": start,
                "content_end": cursor,
                "xpath": tree.getpath(parent) if parent is not None else None,
                "node_slot": (
                    "text"
                    if bool(getattr(node, "is_text", False))
                    else "tail"
                    if bool(getattr(node, "is_tail", False))
                    else "unknown"
                ),
                "normalization": "nbsp_to_space_whitespace_collapse_v1",
            }
        )
    return VisibleHtmlDocument(
        visible_text="\n".join(lines),
        links=links,
        content_selector=content_selector,
        quality_issues=tuple(quality_issues),
        source_projection_json=json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

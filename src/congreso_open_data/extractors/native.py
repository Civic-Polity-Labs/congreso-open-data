"""Literal parsers. These backends never infer facts."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import ClassVar

from lxml import etree, html

from congreso_open_data.models import ExtractionEvidence, ExtractionSpec
from congreso_open_data.protocols import ExtractionContext, ExtractionResult


def _evidence(text: str, *, backend: str, model: str, version: str) -> ExtractionEvidence:
    return ExtractionEvidence(
        text=text,
        span_start=0,
        span_end=len(text),
        confidence=1.0,
        backend=backend,
        model=model,
        version=version,
        literal=True,
    )


@dataclass(frozen=True)
class JsonExtractor:
    model: str = "stdlib-json"
    name: ClassVar[str] = "json"
    engine: ClassVar[str] = "native"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> JsonExtractor:
        return cls(model=spec.model)

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        value = json.loads(content.decode("utf-8-sig"))
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return ExtractionResult(
            texts=(text,),
            evidence=(_evidence(text, backend=self.name, model=self.model, version=self.version),),
            diagnostics={"root_type": type(value).__name__},
        )


@dataclass(frozen=True)
class CsvExtractor:
    model: str = "stdlib-csv"
    encoding: str = "utf-8-sig"
    dialect: str = "excel"
    name: ClassVar[str] = "csv"
    engine: ClassVar[str] = "native"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> CsvExtractor:
        return cls(
            model=spec.model,
            encoding=str(spec.options.get("encoding", "utf-8-sig")),
            dialect=str(spec.options.get("dialect", "excel")),
        )

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        stream = io.StringIO(content.decode(self.encoding))
        rows = tuple(csv.DictReader(stream, dialect=self.dialect))
        texts = tuple(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        return ExtractionResult(
            texts=texts,
            evidence=tuple(
                _evidence(text, backend=self.name, model=self.model, version=self.version)
                for text in texts
            ),
            diagnostics={"row_count": len(rows)},
        )


@dataclass(frozen=True)
class XmlExtractor:
    model: str = "lxml"
    name: ClassVar[str] = "xml"
    engine: ClassVar[str] = "native"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> XmlExtractor:
        return cls(model=spec.model)

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
        root = etree.fromstring(content, parser=parser)
        texts = tuple(text.strip() for text in root.itertext() if text.strip())
        return ExtractionResult(
            texts=texts,
            evidence=tuple(
                _evidence(text, backend=self.name, model=self.model, version=self.version)
                for text in texts
            ),
            diagnostics={"root_tag": str(root.tag)},
        )


@dataclass(frozen=True)
class HtmlExtractor:
    model: str = "lxml-html"
    name: ClassVar[str] = "html"
    engine: ClassVar[str] = "native"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> HtmlExtractor:
        return cls(model=spec.model)

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        document = html.fromstring(content)
        for node in document.xpath("//script|//style|//noscript"):
            node.drop_tree()
        text = "\n".join(part.strip() for part in document.itertext() if part.strip())
        return ExtractionResult(
            texts=(text,),
            evidence=(_evidence(text, backend=self.name, model=self.model, version=self.version),),
        )


@dataclass(frozen=True)
class PyPdfExtractor:
    model: str = "pypdf"
    name: ClassVar[str] = "pypdf"
    engine: ClassVar[str] = "native"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> PyPdfExtractor:
        return cls(model=spec.model)

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install congreso-open-data[pdf] for the pypdf backend") from exc
        reader = PdfReader(io.BytesIO(content))
        texts: list[str] = []
        evidence: list[ExtractionEvidence] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            texts.append(text)
            evidence.append(
                ExtractionEvidence(
                    text=text,
                    span_start=0,
                    span_end=len(text),
                    page=page_number,
                    confidence=1.0 if text.strip() else 0.0,
                    backend=self.name,
                    model=self.model,
                    version=self.version,
                    literal=True,
                )
            )
        return ExtractionResult(texts=texts, evidence=evidence, diagnostics={"pages": len(texts)})


@dataclass(frozen=True)
class PyMuPDFExtractor:
    model: str = "pymupdf"
    name: ClassVar[str] = "pymupdf"
    engine: ClassVar[str] = "native"
    version: ClassVar[str] = "1.0.0"

    @classmethod
    def from_spec(cls, spec: ExtractionSpec) -> PyMuPDFExtractor:
        return cls(model=spec.model)

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("Install congreso-open-data[pdf] for the PyMuPDF backend") from exc
        texts: list[str] = []
        evidence: list[ExtractionEvidence] = []
        with fitz.open(stream=content, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text") or ""
                texts.append(text)
                evidence.append(
                    ExtractionEvidence(
                        text=text,
                        span_start=0,
                        span_end=len(text),
                        page=page_number,
                        confidence=1.0 if text.strip() else 0.0,
                        backend=self.name,
                        model=self.model,
                        version=self.version,
                        literal=True,
                    )
                )
        return ExtractionResult(texts=texts, evidence=evidence, diagnostics={"pages": len(texts)})


NativeExtractor = (
    JsonExtractor | CsvExtractor | XmlExtractor | HtmlExtractor | PyPdfExtractor | PyMuPDFExtractor
)

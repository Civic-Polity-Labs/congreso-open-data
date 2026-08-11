"""Use any synchronous model callable on official intervention text."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from congreso_open_data import (
    Congress,
    ExtractionLimits,
    ExtractionSpec,
    ExtractionTask,
    ModelRequest,
)


def _console_json(payload: Any) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        rendered.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return json.dumps(payload, ensure_ascii=True, indent=2)
    return rendered


def keyword_model(request: ModelRequest) -> dict[str, Any]:
    """Tiny deterministic stand-in for an SDK, local model, or remote LLM."""

    match = re.search(r"\b[\wáéíóúüñ]{5,}\b", request.text, flags=re.IGNORECASE)
    if match is None:
        return {"candidates": []}
    quote = match.group(0)
    return {
        "candidates": [
            {
                "kind": "keyword",
                "value": quote.casefold(),
                "quote": quote,
                "confidence": 1.0,
            }
        ]
    }


def run(*, speaker: str, data_dir: Path, max_results: int = 100) -> dict[str, Any]:
    with Congress(data_dir=data_dir) as congress:
        congress.models.register_callable(
            "my-model",
            model="example-keyword-model",
            version="1",
            provider="user-code",
            function=keyword_model,
        )
        task = ExtractionTask(
            name="keywords",
            instructions="Return a keyword with an exact quote from the text.",
            backend=ExtractionSpec(
                engine="llm",
                backend="my-model",
                model="example-keyword-model",
            ),
            limits=ExtractionLimits(max_input_characters=250_000),
        )
        result = congress.interventions.search(
            speaker=speaker,
            legislatures=("XV",),
            last_months=3,
            text_policy="native",
            extractions=(task,),
            max_results=max_results,
        )
        rows = result.collect(max_items=max_results)

    candidates = [candidate for row in rows for candidate in row.extractions]
    return {
        "records": len(rows),
        "candidates": len(candidates),
        "all_review_required": all(item.status == "review_required" for item in candidates),
        "all_literal": all(evidence.literal for item in candidates for evidence in item.evidence),
        "models": sorted({item.source.model for item in candidates}),
        "complete": result.run.complete,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", default="Pedro Sánchez")
    parser.add_argument("--data-dir", type=Path, default=Path(".congreso-data"))
    parser.add_argument("--max-results", type=int, default=100)
    args = parser.parse_args(argv)
    if args.max_results < 1:
        parser.error("--max-results must be positive")
    print(
        _console_json(
            run(
                speaker=args.speaker,
                data_dir=args.data_dir,
                max_results=args.max_results,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

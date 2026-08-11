"""Get the last three months of one speaker's interventions and native text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from congreso_open_data import Congress


def _console_json(payload: Any) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        rendered.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return json.dumps(payload, ensure_ascii=True, indent=2, default=str)
    return rendered


def run(
    *,
    speaker: str,
    data_dir: Path,
    max_results: int = 100,
    show: int = 5,
) -> dict[str, Any]:
    """Run the complete official query and return a JSON-serializable summary."""

    with Congress(data_dir=data_dir) as congress:
        result = congress.interventions.search(
            speaker=speaker,
            legislatures=("XV",),
            last_months=3,
            text_policy="native",
            max_results=max_results,
        )
        rows = result.collect(max_items=max_results)

    return {
        "records": len(rows),
        "complete": result.run.complete,
        "resolved_entities": result.run.resolved_entities,
        "reused_resources": result.run.reused_resources,
        "failures": list(result.run.failures),
        "sample": [
            {
                "date": row.session_date,
                "title": row.title,
                "speaker": row.speaker,
                "text": row.text,
                "text_method": row.text_method,
                "source_sha256": row.source.sha256,
            }
            for row in rows[:show]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", default="Pedro Sánchez")
    parser.add_argument("--data-dir", type=Path, default=Path(".congreso-data"))
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args(argv)
    if args.max_results < 1 or args.show < 0:
        parser.error("--max-results must be positive and --show cannot be negative")
    print(
        _console_json(
            run(
                speaker=args.speaker,
                data_dir=args.data_dir,
                max_results=args.max_results,
                show=args.show,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

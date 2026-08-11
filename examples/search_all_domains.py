"""Run the ergonomic ``Congress.*.search`` API against every public collection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from congreso_open_data import Congress, SearchResult

VERIFIED_AS_OF = date(2026, 8, 10)


def _consume(result: SearchResult[Any], *, max_items: int) -> dict[str, Any]:
    rows = result.collect(max_items=max_items)
    if not result.run.complete:
        raise RuntimeError(f"Incomplete {result.query.domain} search: {result.run.failures}")
    return {
        "records": len(rows),
        "types": dict.fromkeys(sorted(type(row).__name__ for row in rows), 0),
        "raw_records": result.run.raw_records,
        "duplicates_removed": result.run.duplicate_records,
        "planned_resources": result.run.planned_resources,
        "reused_resources": result.run.reused_resources,
        "rows": rows,
    }


def run(*, data_dir: Path, as_of: date = VERIFIED_AS_OF) -> dict[str, Any]:
    """Execute the ten collection searches used in the verified live example."""

    with Congress(data_dir=data_dir, today=as_of) as congress:
        cases = {
            "deputies": _consume(
                congress.deputies.search(
                    name="Pedro Sánchez", legislatures=("XV",), max_results=10
                ),
                max_items=10,
            ),
            "profiles": _consume(
                congress.profiles.search(name="Pedro Sánchez", legislatures=("XV",), max_results=1),
                max_items=1,
            ),
            "interests": _consume(
                congress.interests.search(
                    deputy="Pedro Sánchez", legislatures=("XV",), max_results=100
                ),
                max_items=100,
            ),
            "financial_documents": _consume(
                congress.financial_documents.search(
                    deputy="Pedro Sánchez", legislatures=("XV",), max_results=20
                ),
                max_items=20,
            ),
            "initiatives": _consume(
                congress.initiatives.search(
                    file_number="121/000001", legislatures=("XV",), max_results=20
                ),
                max_items=20,
            ),
            "interventions": _consume(
                congress.interventions.search(
                    speaker="Pedro Sánchez",
                    legislatures=("XV",),
                    last_months=3,
                    text_policy="native",
                    max_results=100,
                ),
                max_items=100,
            ),
            "votes": _consume(
                congress.votes.search(
                    session="193",
                    vote_number="1",
                    legislatures=("XV",),
                    max_results=500,
                ),
                max_items=500,
            ),
            "organs": _consume(
                congress.organs.search(name="Comisión", legislatures=("XV",), max_results=500),
                max_items=500,
            ),
            "salary_entitlements": _consume(
                congress.salary_entitlements.search(role="Secretario General", max_results=50),
                max_items=50,
            ),
            "documents": _consume(
                congress.documents.search(
                    source_families=("intervenciones",),
                    source_datasets=("IntervencionesCronologicamente",),
                    entity_id="180/001050/0000",
                    mime_type="application/pdf",
                    date_from=date(2026, 5, 20),
                    date_to=date(2026, 5, 20),
                    max_results=500,
                ),
                max_items=500,
            ),
        }

    for case in cases.values():
        for row_type in case["types"]:
            case["types"][row_type] = sum(type(row).__name__ == row_type for row in case["rows"])
        case.pop("rows")
    return {
        "status": "passed",
        "as_of": as_of.isoformat(),
        "data_dir": str(data_dir.resolve()),
        "cases": cases,
    }


def _console_json(payload: object) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return rendered.encode(encoding, errors="backslashreplace").decode(encoding)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".congreso-data"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=VERIFIED_AS_OF)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(data_dir=args.data_dir, as_of=args.as_of)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(_console_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

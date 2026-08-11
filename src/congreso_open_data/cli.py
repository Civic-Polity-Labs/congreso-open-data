"""Command line interface for the public extraction package."""

from __future__ import annotations

import json
import sys
from datetime import date as Date
from pathlib import Path
from typing import Annotated

import typer

from congreso_open_data.api import Congress, InterventionQuery
from congreso_open_data.client import CongressClient
from congreso_open_data.exporters import export_records
from congreso_open_data.models import ExtractionPlan
from congreso_open_data.registry import default_registry

app = typer.Typer(no_args_is_help=True, help="Official Congreso extraction and normalization")


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


@app.callback()
def configure_console() -> None:
    """Configure deterministic UTF-8 JSON output before running a command."""

    _configure_utf8_streams()


def _console_json(payload: object, *, stderr: bool = False) -> str:
    rendered = json.dumps(payload, ensure_ascii=False)
    stream = sys.stderr if stderr else sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        rendered.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return json.dumps(payload, ensure_ascii=True)
    return rendered


@app.command("catalog")
def catalog_command(
    output: Annotated[Path | None, typer.Option(help="Optional output directory")] = None,
    format_name: Annotated[str, typer.Option("--format")] = "jsonl",
) -> None:
    client = CongressClient()
    resources = client.catalog()
    if output is not None:
        paths = export_records(resources, output, format_name=format_name, batch_size=1000)
        for path in paths:
            typer.echo(path)
        return
    for resource in resources:
        typer.echo(_console_json(resource.model_dump(mode="json")))


@app.command("extract")
def extract_command(
    output_root: Annotated[Path, typer.Option("--output-root")],
    family: Annotated[list[str] | None, typer.Option("--family")] = None,
    dataset: Annotated[list[str] | None, typer.Option("--dataset")] = None,
    format_name: Annotated[list[str] | None, typer.Option("--format")] = None,
    run_date: Annotated[str | None, typer.Option("--run-date")] = None,
    max_workers: Annotated[int, typer.Option(min=1, max=16)] = 4,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    values: dict[str, object] = {
        "families": tuple(family or ()),
        "datasets": tuple(dataset or ()),
        "output_root": output_root,
        "max_workers": max_workers,
        "resume": resume,
    }
    if format_name:
        values["formats"] = tuple(format_name)
    if run_date:
        values["run_date"] = run_date
    client = CongressClient(output_root=output_root)
    for manifest in client.extract(ExtractionPlan.model_validate(values)):
        typer.echo(_console_json(manifest.model_dump(mode="json")))


@app.command("backends")
def backends_command(discover_plugins: bool = True) -> None:
    if discover_plugins:
        default_registry.discover_entry_points()
    for name in default_registry.names():
        typer.echo(name)


@app.command("interventions")
def interventions_command(
    speaker: Annotated[str | None, typer.Option(help="Speaker name or surname")] = None,
    speaker_id: Annotated[str | None, typer.Option(help="Official deputy code")] = None,
    legislature: Annotated[list[str] | None, typer.Option("--legislature", "-l")] = None,
    date_from: Annotated[str | None, typer.Option(help="ISO start date (YYYY-MM-DD)")] = None,
    date_to: Annotated[str | None, typer.Option(help="ISO end date (YYYY-MM-DD)")] = None,
    last_months: Annotated[int | None, typer.Option(min=1)] = None,
    text_policy: Annotated[str, typer.Option(help="none, native or ocr")] = "native",
    data_dir: Annotated[Path | None, typer.Option(help="Bronze/checkpoint directory")] = None,
    max_results: Annotated[int, typer.Option(min=1, max=1_000_000)] = 50_000,
    allow_partial: Annotated[bool, typer.Option("--allow-partial/--fail-fast")] = False,
) -> None:
    """Stream official interventions as JSON Lines using the high-level facade."""

    values: dict[str, object] = {
        "speaker": speaker,
        "speaker_id": speaker_id,
        "legislatures": tuple(legislature or ("XV",)),
        "last_months": last_months,
        "text_policy": text_policy,
        "max_results": max_results,
        "allow_partial": allow_partial,
    }
    if date_from is not None:
        values["date_from"] = _iso_date(date_from, option="--date-from")
    if date_to is not None:
        values["date_to"] = _iso_date(date_to, option="--date-to")
    with Congress(data_dir=data_dir) as congress:
        result = congress.interventions.execute(InterventionQuery.model_validate(values))
        for record in result:
            typer.echo(_console_json(record.model_dump(mode="json")))
        typer.echo(
            _console_json(
                {"query_run": result.run.model_dump(mode="json")},
                stderr=True,
            ),
            err=True,
        )


@app.command("models")
def models_command() -> None:
    """List installed provider-neutral model plugins."""

    with Congress() as congress:
        for name in congress.models.names():
            typer.echo(name)


def _iso_date(value: str, *, option: str) -> Date:
    try:
        return Date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter("expected YYYY-MM-DD", param_hint=option) from exc


if __name__ == "__main__":
    app()

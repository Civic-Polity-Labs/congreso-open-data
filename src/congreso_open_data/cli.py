"""Command line interface for the public extraction package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from congreso_open_data.client import CongressClient
from congreso_open_data.exporters import export_records
from congreso_open_data.models import ExtractionPlan
from congreso_open_data.registry import default_registry

app = typer.Typer(no_args_is_help=True, help="Official Congreso extraction and normalization")


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
        typer.echo(json.dumps(resource.model_dump(mode="json"), ensure_ascii=False))


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
        typer.echo(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False))


@app.command("backends")
def backends_command(discover_plugins: bool = True) -> None:
    if discover_plugins:
        default_registry.discover_entry_points()
    for name in default_registry.names():
        typer.echo(name)


if __name__ == "__main__":
    app()

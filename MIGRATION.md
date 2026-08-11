# Migration lineage

This repository starts with a clean Git history. Extraction code was separated from
`Civic-Polity-Labs/cpl-data-foundry` at commit
`1ddde1079604fe1a9088e1fbb29244288a2d4c22` using the working-tree snapshot recorded
on 2026-08-07 in the repository artifact `MIGRATION_SOURCE_INVENTORY.sha256`.

Exact-source modules preserve their original path and bytes where possible. Public
models, protocols, registries, factories, provider-neutral extraction contracts,
exporters and clients are new in the package split. Foundry materialization, quality,
review, Silver/Gold and serving code was deliberately excluded.

## Local package gate

On 2026-08-08 the package-owned regression suite contains more than 265 tests across
all extraction domains. Coverage is measured over the entire distribution (not a
curated public-API subset) and its migration floor is 69% with branches. Ruff, strict
mypy, wheel/sdist, clean-wheel import/CLI and the sibling-package consumer gate are
part of the release workflow. Exact run evidence is recorded in the workspace
`PROJECT_STATE.md` and `.codex_audit` artifacts.

## Additive 1.1 facade

Version 1.1 adds `Congress`, serializable domain queries, single-pass results and a
provider-neutral model registry. It does not remove or redirect the 1.0 low-level
surface. Existing orchestration may keep `CongressClient` and source-specific
discoverers while applications move to `Congress().<domain>.search(...)`.

The compatibility promise for 1.x is:

- existing public imports and normalized field meanings remain available;
- runtime clients and callables never enter `ExtractionSpec` or checkpoints;
- probabilistic output remains review-required and cannot be published here;
- any future consolidation of low-level paths gets a documented warning and at
  least one minor release of overlap.

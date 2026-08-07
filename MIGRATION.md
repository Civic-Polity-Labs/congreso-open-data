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

On 2026-08-07: 55 tests passed; Ruff and public-API mypy passed; measured public-API
coverage was 81.65%; wheel and sdist passed `twine check`; the wheel imported and its
CLI ran beside `official-data-connectors` 1.0.0 in a clean environment.

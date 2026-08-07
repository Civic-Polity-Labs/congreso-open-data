from __future__ import annotations

from collections.abc import Mapping

# Official current detailed datasets own initiative families, not single prefixes.
# ProposicionesDeLey contains Congreso expediente families 122 through 125 in both
# historical and current official sources.
INITIATIVE_OWNER_PREFIXES: Mapping[str, frozenset[str]] = {
    "ProyectosDeLey": frozenset({"121"}),
    "ProposicionesDeLey": frozenset({"122", "123", "124", "125"}),
    "PropuestasDeReforma": frozenset({"127"}),
}
INITIATIVE_OWNER_DATASETS = tuple(INITIATIVE_OWNER_PREFIXES)
INITIATIVE_OWNER_PREFIX_SET = frozenset().union(*INITIATIVE_OWNER_PREFIXES.values())


def initiative_prefix_is_owned(*, dataset: str, prefix: str | None) -> bool:
    return prefix in INITIATIVE_OWNER_PREFIXES.get(dataset, frozenset())


def canonical_initiative_owner_keys(values: list[str]) -> list[str]:
    """Order compound keys by their two fields, not by the serialized separator."""

    return sorted(set(values), key=lambda value: tuple(value.split("|", 1)))

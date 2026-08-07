from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from congreso_open_data.catalog import DatasetResource
from congreso_open_data.durable_io import write_json_atomically
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.storage import BronzeManifest

INITIATIVE_LEGISLATURES = (
    "0",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
)
GENERAL_INITIATIVE_LEGISLATURES = (*INITIATIVE_LEGISLATURES, "XV")
INITIATIVE_DISCOVERY_STATE_VERSION = 4
PAGE_SIZE = 25
LIST_PORTLET_PARAMS = {
    "p_p_id": "iniciativas",
    "p_p_lifecycle": "2",
    "p_p_state": "normal",
    "p_p_mode": "view",
    "p_p_resource_id": "filtrarListado",
    "p_p_cacheability": "cacheLevelPage",
}
APPROVED_LAWS_ENDPOINT = "https://www.congreso.es:443/es/iniciativas-legislativas-aprobadas"
APPROVED_LAWS_PARAMS = {
    "p_p_id": "iniciativasLegislativasAprobadas",
    "p_p_lifecycle": "2",
    "p_p_state": "normal",
    "p_p_mode": "view",
    "p_p_resource_id": "resourceIDsearch",
    "p_p_cacheability": "cacheLevelPage",
}


@dataclass(frozen=True)
class HistoricalInitiativeList:
    dataset: str
    path: str
    cini: str
    general_search: bool = False


@dataclass(frozen=True)
class HistoricalInitiativeScope:
    """A bounded Airflow/CLI unit whose large page plan stays on disk."""

    dataset: str
    legislature: str | None


HISTORICAL_INITIATIVE_LISTS = (
    HistoricalInitiativeList(
        dataset="ProyectosDeLey",
        path="proyectos-de-ley",
        cini="121.CINI.",
    ),
    HistoricalInitiativeList(
        dataset="ProposicionesDeLey",
        path="proposiciones-de-ley",
        cini="(proposicion+adj2+ley).tipo.",
    ),
    HistoricalInitiativeList(
        dataset="PropuestasDeReforma",
        path="propuestas-de-reforma-de-estatutos-de-autonomia",
        cini="127.CINI.",
    ),
)
GENERAL_INITIATIVE_LIST = HistoricalInitiativeList(
    dataset="GeneralInitiatives",
    path="busqueda-de-iniciativas",
    cini="",
    general_search=True,
)
GENERAL_INITIATIVE_FILTERS = (
    "titulo",
    "texto",
    "autor",
    "competencias",
    "tipo",
    "tramitacion",
    "expedientes",
    "hasta",
    "tipo_tramitacion",
    "comision_competente",
    "fase",
    "organo",
    "fechaDe",
    "fechaDesde",
    "fechaHasta",
    "materias",
    "iniciativas_relacionadas",
    "iniciativas_origen",
    "iscc",
)

# The official AJAX endpoint returns the literal JSON object ``{}``, rather than its
# usual zero-total schema, for these six historically empty list/legislature pairs.
# This exact allowlist was revalidated across all 61 first pages on 2026-07-15.
OFFICIAL_EMPTY_INITIATIVE_SCOPES = frozenset(
    {
        ("ProyectosDeLey", "Leg.11"),
        ("PropuestasDeReforma", "Leg.0"),
        ("PropuestasDeReforma", "Leg.1"),
        ("PropuestasDeReforma", "Leg.2"),
        ("PropuestasDeReforma", "Leg.3"),
        ("PropuestasDeReforma", "Leg.7"),
    }
)


def historical_initiative_scopes(
    *,
    legislatures: tuple[str, ...] = INITIATIVE_LEGISLATURES,
    general_legislatures: tuple[str, ...] = GENERAL_INITIATIVE_LEGISLATURES,
    include_approved_laws: bool = True,
    datasets: set[str] | None = None,
) -> tuple[HistoricalInitiativeScope, ...]:
    """Return stable, network-free scopes instead of one task per official page."""

    scopes = [
        HistoricalInitiativeScope(definition.dataset, legislature)
        for definition in HISTORICAL_INITIATIVE_LISTS
        if datasets is None or definition.dataset in datasets
        for legislature in legislatures
    ]
    if datasets is None or GENERAL_INITIATIVE_LIST.dataset in datasets:
        scopes.extend(
            HistoricalInitiativeScope(GENERAL_INITIATIVE_LIST.dataset, legislature)
            for legislature in general_legislatures
        )
    if include_approved_laws and (
        datasets is None or "IniciativasLegislativasAprobadas" in datasets
    ):
        scopes.append(HistoricalInitiativeScope("IniciativasLegislativasAprobadas", None))
    return tuple(scopes)


def discover_historical_initiative_scope_resources(
    *,
    scope: HistoricalInitiativeScope,
    client: CongresoHttpClient | None = None,
    approved_laws_max_year: int | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
) -> list[DatasetResource]:
    """Discover exactly one bounded scope with its own durable checkpoint."""

    known_datasets = {
        *(definition.dataset for definition in HISTORICAL_INITIATIVE_LISTS),
        GENERAL_INITIATIVE_LIST.dataset,
        "IniciativasLegislativasAprobadas",
    }
    if scope.dataset not in known_datasets:
        raise ValueError(f"Unsupported historical initiative dataset: {scope.dataset}")
    if scope.dataset == "IniciativasLegislativasAprobadas":
        if scope.legislature is not None:
            raise ValueError("Approved-law scope must not define a legislature")
        legislatures: tuple[str, ...] = ()
        general_legislatures: tuple[str, ...] = ()
        include_approved_laws = True
    elif scope.legislature is None:
        raise ValueError(f"Scope {scope.dataset} requires a legislature")
    elif scope.dataset == GENERAL_INITIATIVE_LIST.dataset:
        legislatures = ()
        general_legislatures = (scope.legislature,)
        include_approved_laws = False
    else:
        legislatures = (scope.legislature,)
        general_legislatures = ()
        include_approved_laws = False
    return discover_historical_initiative_resources(
        client=client,
        legislatures=legislatures,
        general_legislatures=general_legislatures,
        include_approved_laws=include_approved_laws,
        approved_laws_max_year=approved_laws_max_year,
        datasets={scope.dataset},
        checkpoint_path=checkpoint_path,
        resume=resume,
    )


def discover_historical_initiative_resources(
    *,
    client: CongresoHttpClient | None = None,
    legislatures: tuple[str, ...] = INITIATIVE_LEGISLATURES,
    general_legislatures: tuple[str, ...] = GENERAL_INITIATIVE_LEGISLATURES,
    include_approved_laws: bool = True,
    approved_laws_max_year: int | None = None,
    datasets: set[str] | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
) -> list[DatasetResource]:
    client = client or CongresoHttpClient()
    state = _load_discovery_state(
        checkpoint_path=checkpoint_path,
        legislatures=legislatures,
        general_legislatures=general_legislatures,
        include_approved_laws=include_approved_laws,
        approved_laws_max_year=approved_laws_max_year,
        datasets=datasets,
        resume=resume,
    )
    resources = {
        _resource_key(resource): resource
        for resource in (DatasetResource(**item) for item in state.get("resources", []))
    }
    completed_scopes = set(state.get("completed_scopes", []))
    definition_scopes = (
        *((definition, legislatures) for definition in HISTORICAL_INITIATIVE_LISTS),
        (GENERAL_INITIATIVE_LIST, general_legislatures),
    )
    for definition, definition_legislatures in definition_scopes:
        if datasets and definition.dataset not in datasets:
            continue
        for legislature in definition_legislatures:
            scope = f"{definition.dataset}|{legislature}"
            if scope in completed_scopes:
                continue
            first = _historical_list_resource(
                definition=definition,
                legislature=legislature,
                page=1,
            )
            payload = _json_payload(client.post(first.url, data=first.post_data))
            total = _initiative_total(
                payload,
                dataset=definition.dataset,
                legislature=legislature,
            )
            # Persist page 1 even for an official zero result. A durable empty
            # Bronze response is auditable evidence; an empty manifest index is
            # indistinguishable from a discovery or scheduler failure.
            pages = max(1, math.ceil(total / PAGE_SIZE))
            for page in range(1, pages + 1):
                resource = _historical_list_resource(
                    definition=definition,
                    legislature=legislature,
                    page=page,
                )
                resources[_resource_key(resource)] = resource
            completed_scopes.add(scope)
            _write_discovery_state(
                checkpoint_path,
                state,
                resources,
                completed_scopes,
            )
    if include_approved_laws and (
        datasets is None or "IniciativasLegislativasAprobadas" in datasets
    ):
        approved_resources = discover_approved_law_resources(
            client=client,
            max_year=approved_laws_max_year,
        )
        for resource in approved_resources:
            year = str(resource.post_data["_iniciativasLegislativasAprobadas_anyoSelec"])
            scope = f"IniciativasLegislativasAprobadas|{year}"
            if scope not in completed_scopes:
                resources[_resource_key(resource)] = resource
                completed_scopes.add(scope)
        _write_discovery_state(
            checkpoint_path,
            state,
            resources,
            completed_scopes,
        )
    state["status"] = "completed"
    _write_discovery_state(
        checkpoint_path,
        state,
        resources,
        completed_scopes,
    )
    return list(resources.values())


def _load_discovery_state(
    *,
    checkpoint_path: Path | None,
    legislatures: tuple[str, ...],
    general_legislatures: tuple[str, ...],
    include_approved_laws: bool,
    approved_laws_max_year: int | None,
    datasets: set[str] | None,
    resume: bool,
) -> dict[str, Any]:
    expected = _discovery_identity(
        legislatures=legislatures,
        general_legislatures=general_legislatures,
        include_approved_laws=include_approved_laws,
        approved_laws_max_year=approved_laws_max_year,
        datasets=datasets,
    )
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                "Historical initiative discovery checkpoint does not match the request"
            )
        actual = _discovery_identity(
            legislatures=tuple(payload.get("legislatures") or ()),
            general_legislatures=tuple(payload.get("general_legislatures") or ()),
            include_approved_laws=bool(payload.get("include_approved_laws")),
            approved_laws_max_year=payload.get("approved_laws_max_year"),
            datasets=(set(payload["datasets"]) if payload.get("datasets") is not None else None),
            version=payload.get("version"),
        )
        if actual != expected:
            raise ValueError(
                "Historical initiative discovery checkpoint does not match the request"
            )
        # Migrate legacy CLI checkpoints whose inactive legislature collections were
        # populated even though the selected dataset could never consume them.
        return payload | expected
    return expected | {
        "status": "running",
        "completed_scopes": [],
        "resources": [],
    }


def _discovery_identity(
    *,
    legislatures: tuple[str, ...],
    general_legislatures: tuple[str, ...],
    include_approved_laws: bool,
    approved_laws_max_year: int | None,
    datasets: set[str] | None,
    version: Any = INITIATIVE_DISCOVERY_STATE_VERSION,
) -> dict[str, Any]:
    detailed_selected = datasets is None or any(
        definition.dataset in datasets for definition in HISTORICAL_INITIATIVE_LISTS
    )
    general_selected = datasets is None or GENERAL_INITIATIVE_LIST.dataset in datasets
    approved_selected = include_approved_laws and (
        datasets is None or "IniciativasLegislativasAprobadas" in datasets
    )
    return {
        "version": version,
        "legislatures": list(legislatures) if detailed_selected else [],
        "general_legislatures": (list(general_legislatures) if general_selected else []),
        "include_approved_laws": approved_selected,
        "datasets": sorted(datasets) if datasets is not None else None,
        "approved_laws_max_year": approved_laws_max_year if approved_selected else None,
    }


def _write_discovery_state(
    checkpoint_path: Path | None,
    state: dict[str, Any],
    resources: dict[tuple[str, str | None], DatasetResource],
    completed_scopes: set[str],
) -> None:
    if checkpoint_path is None:
        return
    payload = state | {
        "completed_scopes": sorted(completed_scopes),
        "resources": [asdict(resource) for resource in resources.values()],
    }
    write_json_atomically(checkpoint_path, payload)


def _resource_key(resource: DatasetResource) -> tuple[str, str | None]:
    """Keep POST requests distinct even when their endpoint URL is identical."""

    parameters = (
        json.dumps(
            {str(key): str(value) for key, value in resource.post_data.items()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if resource.post_data is not None
        else None
    )
    return resource.url, parameters


def discover_approved_law_resources(
    *,
    client: CongresoHttpClient | None = None,
    years: tuple[str, ...] | None = None,
    max_year: int | None = None,
) -> list[DatasetResource]:
    client = client or CongresoHttpClient()
    if years is None:
        years = _approved_law_years(client)
    if max_year is not None:
        years = tuple(year for year in years if int(year) <= max_year)
    return [_approved_law_resource(year) for year in years]


def extract_historical_initiative_resources(
    *,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
    legislatures: tuple[str, ...] = INITIATIVE_LEGISLATURES,
    general_legislatures: tuple[str, ...] = GENERAL_INITIATIVE_LEGISLATURES,
    include_approved_laws: bool = True,
    approved_laws_max_year: int | None = None,
    datasets: set[str] | None = None,
) -> list[BronzeManifest]:
    client = client or CongresoHttpClient()
    manifests: list[BronzeManifest] = []
    for resource in discover_historical_initiative_resources(
        client=client,
        legislatures=legislatures,
        general_legislatures=general_legislatures,
        include_approved_laws=include_approved_laws,
        approved_laws_max_year=approved_laws_max_year,
        datasets=datasets,
    ):
        manifests.append(
            extract_resource(
                resource=resource,
                run_date=run_date,
                output_root=output_root,
                client=client,
            )
        )
    return manifests


def _historical_list_resource(
    *,
    definition: HistoricalInitiativeList,
    legislature: str,
    page: int,
) -> DatasetResource:
    if definition.general_search:
        post_data = {
            **{f"_iniciativas_{field}": "" for field in GENERAL_INITIATIVE_FILTERS},
            "_iniciativas_legislatura": legislature,
            "_iniciativas_paginaActual": str(page),
        }
        query = LIST_PORTLET_PARAMS | {
            "_iniciativas_legislatura": legislature,
            "_iniciativas_paginaActual": str(page),
        }
    else:
        post_data = {
            "_iniciativas_legislatura": legislature,
            "_iniciativas_estadoTramitacion": "",
            "_iniciativas_faseTramitacion": "",
            "_iniciativas_cini": definition.cini,
            "_iniciativas_tipoLlamada": "T",
            "_iniciativas_paginaActual": str(page),
            "_iniciativas_comision_competente": "",
        }
        query = LIST_PORTLET_PARAMS | {
            "_iniciativas_mode": "mostrarListado",
            "_iniciativas_legislatura": legislature,
            "_iniciativas_cini": definition.cini,
            "_iniciativas_paginaActual": str(page),
        }
    return DatasetResource(
        family="iniciativas",
        dataset=definition.dataset,
        format="json",
        url=_url_with_query(f"https://www.congreso.es:443/es/{definition.path}", query),
        snapshot_token=(f"historical-{definition.dataset}-{_token(legislature)}-{page:04d}"),
        legislature=_leg_label(legislature),
        post_data=post_data,
    )


def _approved_law_resource(year: str) -> DatasetResource:
    post_data = {
        "_iniciativasLegislativasAprobadas_anyoSelec": year,
        "_iniciativasLegislativasAprobadas_tipoLeySelec": "",
    }
    return DatasetResource(
        family="iniciativas",
        dataset="IniciativasLegislativasAprobadas",
        format="json",
        url=_url_with_query(APPROVED_LAWS_ENDPOINT, APPROVED_LAWS_PARAMS | post_data),
        snapshot_token=f"historical-IniciativasLegislativasAprobadas-{year}",
        post_data=post_data,
    )


def _approved_law_years(client: CongresoHttpClient) -> tuple[str, ...]:
    response = client.get("https://www.congreso.es/es/iniciativas-legislativas-aprobadas")
    html = response.content.decode("utf-8", errors="replace")
    marker = '_iniciativasLegislativasAprobadas_anyoSelec"'
    start = html.find(marker)
    if start < 0:
        raise ValueError("Approved-laws year selector is missing from the official page")
    end = html.find("</select>", start)
    select_html = html[start : end if end >= 0 else start + 5000]
    years = re.findall(r'<option[^>]+value="(\d{4})"', select_html, flags=re.I)
    unique_years = tuple(dict.fromkeys(years))
    if not unique_years:
        raise ValueError("Approved-laws year selector contains no years")
    return unique_years


def _json_payload(result: Any) -> dict[str, Any]:
    content = result.content.decode("utf-8-sig")
    if not content.strip():
        return {}
    payload = json.loads(content)
    return payload if isinstance(payload, dict) else {}


def _initiative_total(
    payload: dict[str, Any],
    *,
    dataset: str | None = None,
    legislature: str | None = None,
) -> int:
    if payload == {} and initiative_empty_scope_is_expected(dataset, legislature):
        return 0
    if "iniciativas_encontradas" not in payload:
        raise ValueError("Historical initiative response is missing iniciativas_encontradas")
    value = payload["iniciativas_encontradas"]
    digits = re.sub(r"\D+", "", str(value))
    if not digits:
        raise ValueError(
            f"Historical initiative response has a non-numeric iniciativas_encontradas: {value!r}"
        )
    return int(digits)


def initiative_empty_scope_is_expected(
    dataset: str | None,
    legislature: str | None,
) -> bool:
    if not dataset or legislature is None:
        return False
    text = str(legislature).strip()
    if text.casefold().startswith("leg."):
        text = text[4:]
    label = f"Leg.{int(text)}" if text.isdigit() else _leg_label(text.upper())
    return (dataset, label) in OFFICIAL_EMPTY_INITIATIVE_SCOPES


def _url_with_query(base_url: str, params: dict[str, Any]) -> str:
    return f"{base_url}?{urlencode(params)}"


def _token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-") or "0"


def _leg_label(value: str) -> str:
    numbers = {
        "0": 0,
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
        "XIII": 13,
        "XIV": 14,
        "XV": 15,
    }
    return f"Leg.{numbers.get(value, value)}"

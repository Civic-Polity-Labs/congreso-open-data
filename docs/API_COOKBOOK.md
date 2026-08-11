# API completa: ejemplos ejecutables y probados

La frontera estable de `congreso-open-data 1.1` son los 63 nombres de
`congreso_open_data.__all__`. Para una aplicación normal casi siempre basta con
`Congress`; el resto existe para consultas tipadas, orquestación, procedencia,
extensiones de modelos y manejo explícito de errores.

Hay dos verificaciones complementarias:

- [`public_api_contracts.py`](../examples/public_api_contracts.py) usa una fuente
  diminuta inyectada y ejecuta sin red los 63 símbolos públicos: adquisición real,
  manifiesto, checkpoint, normalización, fachada, streaming y modelo callable.
- [`verify_live_all_domains.py`](../examples/verify_live_all_domains.py) comprueba de
  forma acotada todos los dominios y formatos contra las fuentes oficiales actuales.
- [`search_all_domains.py`](../examples/search_all_domains.py) ejecuta las diez
  colecciones de la fachada mediante sus firmas explícitas `.search(...)`.

## 1. La fachada que usa una aplicación

```python
from congreso_open_data import Congress

with Congress(data_dir=".congreso-data") as congress:
    result = congress.interventions.search(
        speaker="Pedro Sánchez",
        legislatures=("XV",),
        last_months=3,
        text_policy="native",
        max_results=100,
    )

    for intervention in result:  # streaming y una sola pasada
        print(intervention.session_date, intervention.title)
        print(intervention.text)
        print(intervention.source.requested_url, intervention.source.sha256)

    if not result.run.complete:
        raise RuntimeError(result.run.failures)
```

`Congress` resuelve entidades, consulta y pagina la fuente oficial, conserva Bronze,
normaliza, enriquece texto cuando se solicita y reconcilia el resultado. No hace falta
construir URLs, POST, manifests ni parsers.

## 2. Todos los dominios de `Congress`

```python
from datetime import date

from congreso_open_data import Congress

with Congress(data_dir=".congreso-data") as congress:
    deputies = congress.deputies.search(
        name="Pedro Sánchez", legislatures=("XV",), max_results=10
    ).collect(max_items=10)

    profiles = congress.profiles.search(
        deputy_id="189", legislatures=("XV",), max_results=1
    ).collect(max_items=1)

    interests = congress.interests.search(
        deputy="Pedro Sánchez", declaration_kind="Declaración inicial", max_results=100
    ).collect(max_items=100)

    financial_documents = congress.financial_documents.search(
        deputy="Pedro Sánchez", document_kind="assets_income", max_results=20
    ).collect(max_items=20)

    initiatives = congress.initiatives.search(
        file_number="121/000001", title="representación paritaria", max_results=20
    ).collect(max_items=20)

    interventions = congress.interventions.search(
        speaker="Pedro Sánchez", last_months=3, max_results=100
    ).collect(max_items=100)

    votes = congress.votes.search(session="193", vote_number="1", max_results=500).collect(
        max_items=500
    )

    organs = congress.organs.search(name="Comisión", max_results=500).collect(max_items=500)

    salaries = congress.salary_entitlements.search(
        role="Secretario General", max_results=50
    ).collect(max_items=50)

    documents = congress.documents.search(
        source_families=("intervenciones",),
        source_datasets=("IntervencionesCronologicamente",),
        entity_id="180/001050/0000",
        mime_type="application/pdf",
        date_from=date(2026, 5, 20),
        date_to=date(2026, 5, 20),
        max_results=500,
    ).collect(max_items=500)
```

| Servicio | Query tipada | Tipos que devuelve |
| --- | --- | --- |
| `deputies` | `DeputyQuery` | `Deputy` |
| `profiles` | `ProfileQuery` | `DeputyProfile` |
| `interests` | `InterestQuery` | `InterestDeclaration` |
| `financial_documents` | `FinancialDocumentQuery` | `FinancialDocument` |
| `initiatives` | `InitiativeQuery` | `Initiative` |
| `interventions` | `InterventionQuery` | `InterventionRecord` |
| `votes` | `VoteQuery` | `VoteEvent`, `VoteItem`, `NominalVote` |
| `organs` | `OrganQuery` | `Organ` |
| `salary_entitlements` | `SalaryEntitlementQuery` | `SalaryEntitlement` |
| `documents` | `DocumentQuery` | `DocumentAsset` |

`collect()` solo debe usarse para resultados pequeños y exige un máximo explícito.
Para históricos, se itera el resultado o se usa la API física de la sección 8.

La lista de miembros de un órgano no comparte los mismos filtros ni el mismo ciclo de
descubrimiento que el catálogo de órganos. Por ello no se mezcla en
`organs.search()`: se obtiene con los recursos dinámicos y `CongressClient.organs()`
de la API física. Una futura fachada puede exponerla como colección propia, no como
un tipo que aparezca de forma sorpresiva en una búsqueda de órganos.

La fuente oficial de intervenciones acepta el rango de fechas, pero actualmente
ignora el expediente al calcular y paginar su export. Por eso una búsqueda de
documentos de intervención por `entity_id` debe incluir una ventana temporal acotada;
el paquete vuelve a filtrar el expediente localmente y falla si `max_results` no basta.

## 3. Queries tipadas, fechas, orden y fingerprint

Cada servicio admite `.execute(query)` además del atajo `.search(...)`:

```python
from datetime import date

from congreso_open_data import (
    Congress,
    InterventionQuery,
    RefreshPolicy,
    SortOrder,
    TextPolicy,
)

query = InterventionQuery(
    speaker_id="189",
    legislatures=(15,),  # se normaliza a ("XV",)
    date_from=date(2026, 5, 10),
    date_to=date(2026, 8, 10),
    text_policy=TextPolicy.NATIVE,
    refresh=RefreshPolicy.AUTO,
    sort=SortOrder.DESCENDING,
    max_results=100,
)

print(query.fingerprint())  # SHA-256 estable de la consulta

with Congress(data_dir=".congreso-data") as congress:
    result = congress.interventions.execute(query)
    rows = result.collect(max_items=100)
```

Los filtros comunes de `CongressQuery` son `legislatures`, `date_from`, `date_to`,
`last_months`, `refresh`, `sort`, `max_results`, `allow_partial` y `extractions`.
`last_months` no se puede mezclar con fechas explícitas; `speaker` y `speaker_id`
tampoco se pueden usar a la vez.

## 4. Streaming y reconciliación

```python
from congreso_open_data import ResultConsumedError

result = congress.deputies.search(legislatures=("XV",), max_results=500)

for deputy in result:
    consume(deputy)

print(result.run.model_dump(mode="json"))
assert result.run.normalized_records == 350
assert result.run.complete

try:
    list(result)
except ResultConsumedError:
    print("El resultado es de una sola pasada")
```

`SearchResult.run` es un `QueryRun` con recursos planificados, adquiridos, reutilizados
y fallidos; filas normalizadas antes de aplicar filtros (`raw_records`), filas
entregadas (`normalized_records`) y duplicados eliminados; textos no emparejados;
rutas del checkpoint/log y errores. Consultarlo después de consumir el iterador da el
estado final.

## 5. Procedencia y objetos normalizados

Todos los registros son modelos Pydantic inmutables y serializables:

```python
from congreso_open_data import Deputy, SourceRef

source = SourceRef(
    requested_url="https://example.test/diputados.json",
    sha256="a" * 64,
    parameters={"token": "no debe persistirse", "legislature": "XV"},
    adapter="mi-adapter",
    adapter_version="1",
    normalization_version="1",
    method="native-json",
)

deputy = Deputy(source=source, deputy_id="189", full_name="Sánchez, Pedro")
payload = deputy.model_dump(mode="json")

assert payload["source"]["parameters"]["token"] == "[REDACTED]"
assert len(payload["source"]["sha256"]) == 64
```

Además de los tipos de la tabla anterior, la API expone `Intervention` y
`InterventionOccurrence` para la normalización física, `DocumentText` para texto PDF
u OCR, `SpeechBlock` para bloques de diario, y `ExtractionEvidence` /
`ExtractionCandidate` para inferencias revisables.

## 6. Cualquier LLM, NLP, OCR o callable

El caso sencillo es registrar una función síncrona. Puede llamar a cualquier SDK o
ser un modelo local; el cliente, las credenciales y el callable nunca se serializan.

```python
from congreso_open_data import (
    Congress,
    ExtractionLimits,
    ExtractionSpec,
    ExtractionTask,
    ModelRequest,
    ModelResponse,
)


def my_model(request: ModelRequest) -> ModelResponse:
    # Aquí puede entrar OpenAI, Anthropic, Ollama, vLLM, LiteLLM, HF o código propio.
    return ModelResponse(
        payload={
            "candidates": [
                {
                    "kind": "topic",
                    "value": "vivienda",
                    "quote": "acceso a la vivienda",
                    "confidence": 0.91,
                }
            ]
        },
        request_id="provider-request-id",
    )


with Congress(data_dir=".congreso-data") as congress:
    congress.models.register_callable(
        "mine",
        model="my-model-id",
        version="2026-08-10",
        provider="self-hosted",
        function=my_model,
    )
    task = ExtractionTask(
        name="topics",
        instructions="Extrae temas respaldados por citas literales.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="my-model-id"),
        limits=ExtractionLimits(max_input_characters=250_000),
    )
    rows = congress.interventions.search(
        speaker="Pedro Sánchez",
        last_months=3,
        extractions=(task,),
        max_results=100,
    ).collect(max_items=100)

assert all(candidate.status == "review_required" for row in rows for candidate in row.extractions)
```

Un objeto con estado implementa estructuralmente `ModelBackend`:

```python
from congreso_open_data import ModelDescriptor, ModelRequest, ModelResponse


class MyBackend:
    descriptor = ModelDescriptor(
        name="mine", model="my-model-id", version="2026-08-10", provider="local"
    )

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(payload={"candidates": []})
```

`ModelRegistry.register`, `register_factory` y `register_callable` cubren instancias,
factorías y funciones. Un paquete externo puede usar el entry point
`congreso_open_data.models`. `StructuredModelExtractor` valida `CandidateEnvelope`,
trocea texto según `ExtractionLimits` y conserva evidencia literal, identidad,
versión, usage y diagnósticos.

## 7. Contratos de extracción y candidatos

```python
from congreso_open_data import (
    CandidateEnvelope,
    CandidateValue,
    ExtractionEvidence,
)

envelope = CandidateEnvelope(
    candidates=(
        CandidateValue(
            kind="topic",
            value="vivienda",
            quote="acceso a la vivienda",
            confidence=0.95,
        ),
    )
)

evidence = ExtractionEvidence(
    text="acceso a la vivienda",
    span_start=3,
    span_end=23,
    confidence=0.95,
    backend="mine",
    model="my-model-id",
    version="2026-08-10",
)
```

Una inferencia sigue siendo `review_required`; la API no la convierte en dato
canónico. `ExtractionSpec` es serializable y solo identifica engine/backend/model y
opciones no secretas. `ExtractionTask` añade instrucciones, esquema y límites. Los
objetos runtime se resuelven por nombre en `ModelRegistry`.

## 8. API física para DAGs y cargas históricas

```python
from pathlib import Path

from congreso_open_data import CongressClient, ExtractionPlan

root = Path("bronze-congreso").resolve()
client = CongressClient(output_root=root)

resource = next(
    item
    for item in client.catalog()
    if item.family == "votaciones" and item.dataset == "Votacion" and item.format == "json"
)
manifests = tuple(
    client.extract(
        ExtractionPlan(
            resources=(resource,),
            output_root=root,
            batch_size=1,
            max_resources=1,
            max_workers=1,
            resume=True,
            continue_on_error=False,
        )
    )
)

for vote in client.votes(manifests):
    print(type(vote).__name__, vote.source.sha256)

assert client.last_run is not None
assert client.last_run.failed == 0
```

Los normalizadores físicos públicos son:

```python
client.deputies(manifests)
client.profiles(manifests)
client.interests(manifests)
client.financial_documents(manifests)
client.initiatives(manifests)
client.interventions(manifests)
client.intervention_occurrences(manifests)
client.votes(manifests)
client.organs(manifests)
client.documents(manifests)
client.document_texts(manifests, use_ocr=False)
client.document_texts(manifests, use_ocr=True)
client.speech_blocks(manifests)
client.salary_entitlements(manifests)
```

`CatalogResource` describe lo planificado; `ArtifactManifest` lo adquirido;
`ExtractionPlan` limita y configura la ejecución; `ExtractionRun` y
`ExtractionFailure` registran la reconciliación. El `output_root` del plan debe ser el
mismo del cliente.

## 9. Exportar por lotes

El exportador es una API de módulo avanzada. Publica un directorio atómicamente y
crea varios shards cuando hay más filas que `batch_size`:

```python
from congreso_open_data.exporters import export_records

paths = export_records(
    congress.deputies.search(legislatures=("XV",), max_results=500),
    "exports/deputies-jsonl",
    format_name="jsonl",  # json, jsonl, csv o parquet
    batch_size=100,
)
print(paths)
```

Parquet necesita `pip install congreso-open-data[parquet]`. El exportador rechaza
sobrescribir un directorio existente.

## 10. Errores públicos

```python
from congreso_open_data import (
    AmbiguousEntityError,
    CongressError,
    EntityNotFoundError,
    IncompleteResultError,
    QueryValidationError,
    SourceContractError,
    SourceUnavailableError,
)

try:
    rows = congress.interventions.search(
        speaker="nombre ambiguo", last_months=3, max_results=100
    ).collect(max_items=100)
except (EntityNotFoundError, AmbiguousEntityError) as exc:
    correct_identity(exc)
except QueryValidationError as exc:
    correct_filters(exc)
except SourceUnavailableError as exc:
    retry_later(exc)
except (SourceContractError, IncompleteResultError) as exc:
    quarantine_and_alert(exc)
except CongressError as exc:
    handle_unexpected_package_error(exc)
```

El paquete falla cerrado ante una fuente irreconciliable salvo que se declare
`allow_partial=True`. Incluso entonces, `QueryRun` conserva fallos y `complete=False`.

## 11. CLI

```bash
congreso-open-data interventions \
  --speaker "Pedro Sánchez" \
  --last-months 3 \
  --data-dir .congreso-data \
  --max-results 100

congreso-open-data catalog --output catalog --format jsonl
congreso-open-data backends
congreso-open-data models
```

Las filas se escriben como JSON Lines UTF-8 por stdout y `QueryRun` por stderr.

## 12. Ejecutar exactamente los ejemplos probados

```powershell
# Los 63 símbolos públicos, sin red
uv run python examples/public_api_contracts.py `
  --output-dir D:\tmp\congreso-public-api

# Caso corto real
uv run python examples/interventions_last_three_months.py `
  --data-dir D:\tmp\congreso-example

# Callable agnóstico sobre intervenciones reales
uv run python examples/custom_model_callable.py `
  --data-dir D:\tmp\congreso-example

# Todos los dominios y formatos contra la fuente oficial
uv run python examples/verify_live_all_domains.py `
  --data-dir D:\tmp\congreso-e2e `
  --report D:\tmp\congreso-e2e-report.json
```

La suite exige que `PUBLIC_API_EXAMPLES == congreso_open_data.__all__`, ejecuta el
ejemplo offline y comprueba sus manifests, checkpoints, filas normalizadas,
streaming, redacción de secretos, errores y candidato de modelo. El runner vivo
validado produjo 10/10 casos correctos; el detalle y los gates están en
[`TESTING.md`](TESTING.md).

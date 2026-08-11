# Ejemplos end to end

La aplicación normal usa `Congress`: proporciona filtros y recibe objetos Pydantic
tipados. No construye URLs, POST, paginación, manifests, OCR ni parsers. El paquete
conserva el cuerpo oficial en Bronze, su SHA-256 y la procedencia antes de normalizar.

## Intervenciones de Pedro Sánchez en los últimos tres meses

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
    rows = result.collect(max_items=100)

for row in rows:
    print(row.session_date, row.title)
    print(row.text)
    print(row.text_method, row.source.sha256)

assert result.run.complete
```

Es exactamente el flujo de
[`examples/interventions_last_three_months.py`](../examples/interventions_last_three_months.py).
La consulta resuelve la persona en el índice oficial, envía los filtros al endpoint,
reconcilia el total, conserva Bronze y obtiene el texto nativo del diario. Una segunda
ejecución con el mismo `data_dir` reutiliza artefactos verificados.

## Todos los dominios de la fachada

Cada servicio acepta también su query tipada mediante `.execute(query)`. Estos son
los accesos cortos equivalentes:

El recorrido completo es ejecutable en
[`examples/search_all_domains.py`](../examples/search_all_domains.py). La auditoría
viva del 10 de agosto de 2026 ejecutó esas diez búsquedas y revisó manualmente sus
identidades, conteos, categorías, expediente, importes, votos, URLs y páginas.

```python
from datetime import date

from congreso_open_data import Congress

with Congress(data_dir=".congreso-data") as congress:
    deputies = congress.deputies.search(
        name="Sánchez Pérez-Castejón",
        legislatures=("XV",),
        max_results=100,
    ).collect(max_items=100)

    profiles = congress.profiles.search(
        name="Pedro Sánchez",
        legislatures=("XV",),
        max_results=1,
    ).collect(max_items=1)

    interests = congress.interests.search(
        deputy="Pedro Sánchez",
        legislatures=("XV",),
        max_results=100,
    ).collect(max_items=100)

    financial_documents = congress.financial_documents.search(
        deputy="Pedro Sánchez",
        legislatures=("XV",),
        max_results=20,
    ).collect(max_items=20)

    initiatives = congress.initiatives.search(
        file_number="121/000001",
        legislatures=("XV",),
        max_results=20,
    ).collect(max_items=20)

    votes = congress.votes.search(
        session="193",
        vote_number="1",
        legislatures=("XV",),
        max_results=500,
    ).collect(max_items=500)

    organs = congress.organs.search(
        name="Comisión",
        legislatures=("XV",),
        max_results=500,
    ).collect(max_items=500)

    salaries = congress.salary_entitlements.search(
        role="Secretario General",
        max_results=50,
    ).collect(max_items=50)

    document_assets = congress.documents.search(
        source_families=("intervenciones",),
        source_datasets=("IntervencionesCronologicamente",),
        entity_id="180/001050/0000",
        mime_type="application/pdf",
        date_from=date(2026, 5, 20),
        date_to=date(2026, 5, 20),
        max_results=500,
    ).collect(max_items=500)
```

Las fuentes que solo publican exports completos se filtran después de conservar y
normalizar el export oficial; `max_results` hace fallar cerrada una consulta demasiado
amplia. Para una carga histórica, un DAG debe seleccionar alcances físicos explícitos
con la API de bajo nivel en vez de esconder una descarga masiva detrás de una consulta.

Los tipos de salida son `Deputy`, `DeputyProfile`, `InterestDeclaration`,
`FinancialDocument`, `Initiative`, `InterventionRecord`, `VoteEvent`, `VoteItem`,
`NominalVote`, `Organ`, `SalaryEntitlement`, `DocumentAsset` y
`DocumentText`. Todos llevan `source` con URL, hash, adaptador, versión y método.

`organs.search()` devuelve exclusivamente el catálogo `Organ`. Las membresías se
descubren por órgano y se normalizan con `CongressClient.organs()` en la API física;
no se mezclan de forma implícita con el catálogo. En documentos de intervenciones hay
que aportar además una fecha acotada: la web oficial pagina por fecha pero, en su
comportamiento actual, no reduce el export por expediente aunque reciba ese filtro.

## Cualquier LLM, NLP o modelo propio

El caso sencillo es un callable. Puede llamar a Ollama, vLLM, LiteLLM, OpenAI,
Anthropic, Hugging Face o a una función local; el paquete no conoce ni serializa ese
cliente.

```python
from congreso_open_data import Congress, ExtractionLimits, ExtractionSpec, ExtractionTask


def my_model(request):
    response = my_sdk.generate(
        model="mi-modelo",
        prompt=request.instructions,
        text=request.text,
        schema=request.output_schema,
    )
    return response  # mapping, JSON string o ModelResponse


with Congress(data_dir=".congreso-data") as congress:
    congress.models.register_callable(
        "mine",
        model="mi-modelo",
        version="2026-08-09",
        provider="self-hosted",
        function=my_model,
    )
    task = ExtractionTask(
        name="topics",
        instructions="Extrae temas con una cita literal.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="mi-modelo"),
        # Límite total explícito; cada llamada sigue troceada y acotada.
        limits=ExtractionLimits(max_input_characters=250_000),
    )
    rows = congress.interventions.search(
        speaker="Pedro Sánchez",
        last_months=3,
        extractions=(task,),
        max_results=100,
    ).collect(max_items=100)
```

El ejemplo ejecutable
[`examples/custom_model_callable.py`](../examples/custom_model_callable.py) no usa red
para el modelo y valida que cada resultado sea un `ExtractionCandidate` con
`status="review_required"`, identidad de modelo y evidencia literal. Un backend con
estado implementa estructuralmente `ModelBackend`; un paquete externo puede publicarlo
con el entry point `congreso_open_data.models`.

## Pipeline físico para un DAG histórico

Cuando el orquestador ya conoce el recurso que quiere materializar, usa
`CongressClient`. Sigue siendo end to end: adquisición, hash/manifiesto, checkpoint y
normalización.

```python
from pathlib import Path

from congreso_open_data import CongressClient, ExtractionPlan

root = Path("bronze-congreso")
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
            max_resources=1,
            max_workers=1,
            resume=True,
        )
    )
)

for vote in client.votes(manifests):
    print(type(vote).__name__, vote.source.sha256)

assert client.last_run is not None
assert client.last_run.failed == 0
```

El runner
[`examples/verify_live_all_domains.py`](../examples/verify_live_all_domains.py)
aplica este patrón de forma acotada a cada dominio y escribe un informe JSON después
de cada caso. No toca Silver/Gold, Airflow ni el lake del consumidor.

## CLI

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

Las filas salen por stdout como JSON Lines UTF-8 y `QueryRun` por stderr, para poder
encadenar la CLI sin perder la reconciliación.

## Qué se ha probado

La suite determinista cubre HTTP GET/POST, reintentos y límites; JSON, CSV/TSV/gzip,
XML, HTML, PDF, PNG y ZIP; extracción PDF primaria y fallback OCR; NLP/Transformers
inyectados; callables, protocolos y entry points; todos los normalizadores; errores,
reanudación, hashes, checkpoints y resultados parciales. La auditoría viva acotada
complementa esos fixtures contra las fuentes oficiales actuales. Consulta
[`TESTING.md`](TESTING.md) para el gate exacto y sus límites.

# congreso-open-data

API Python tipada para adquirir, preservar y normalizar datos oficiales del Congreso
de los Diputados: diputados, perfiles, intereses y bienes, iniciativas,
intervenciones, órganos, votaciones, diarios y documentos históricos.

## Uso end to end

```bash
pip install congreso-open-data
```

```python
from congreso_open_data import Congress

with Congress() as congress:
    result = congress.interventions.search(
        speaker="Pedro Sánchez",
        last_months=3,
    )
    for intervention in result:
        print(intervention.session_date, intervention.title)
        print(intervention.text)

    print(result.run.complete, result.run.raw_records)
```

El usuario no construye URLs, POST, páginas, manifiestos ni parsers. `Congress`
resuelve el nombre contra el índice oficial, aplica filtros y paginación, guarda
Bronze con hash y procedencia, normaliza, reconcilia el total oficial y obtiene texto
nativo. OCR solo se usa si se solicita con `text_policy="ocr"`.

El directorio por defecto es el directorio persistente de datos de usuario de la
plataforma. En jobs y tests conviene declararlo con `Congress(data_dir=...)`.

```bash
congreso-open-data interventions --speaker "Pedro Sánchez" --last-months 3
```

## El modelo lo elige el usuario

No hay un proveedor obligatorio. Se puede registrar un callable, implementar el
protocolo `ModelBackend` o instalar un plugin con el entry point
`congreso_open_data.models`. La identidad del modelo es obligatoria para conservar
procedencia reproducible.

```python
from congreso_open_data import Congress, ExtractionSpec, ExtractionTask


def my_model(request):
    # Aquí puede llamarse a cualquier SDK, servidor local o modelo propio.
    return {
        "candidates": [
            {
                "kind": "topic",
                "value": "vivienda",
                "quote": "acceso a la vivienda",
                "confidence": 0.91,
            }
        ]
    }


with Congress() as congress:
    congress.models.register_callable(
        "mine",
        model="my-model-id",
        version="2026-08-09",
        function=my_model,
        provider="self-hosted",
    )
    task = ExtractionTask(
        name="topics",
        instructions="Extrae temas apoyados por citas literales.",
        backend=ExtractionSpec(engine="llm", backend="mine", model="my-model-id"),
    )
    rows = congress.interventions.search(
        speaker="Pedro Sánchez",
        last_months=3,
        extractions=(task,),
    )
```

La salida del modelo nunca se promueve a hecho: son `ExtractionCandidate` con
`status="review_required"`, fuente, modelo y evidencia literal o marcada como no
literal. Clientes SDK, callables y credenciales no se serializan en la consulta.

## API de bajo nivel compatible

La API 1.0 continúa disponible durante toda la serie 1.x:

```python
from congreso_open_data import CongressClient, ExtractionPlan

client = CongressClient(output_root="bronze")
plan = ExtractionPlan(families=("diputados",), output_root="bronze")
manifests = tuple(client.extract(plan))
for deputy in client.deputies(manifests):
    print(deputy.full_name, deputy.source.sha256)
```

Extras: `pdf` (fallback PyMuPDF), `ocr`, `paddle`, `transformers`, `nlp`,
`openai`, `anthropic`, `local`, `parquet`, `all`, `gpu` y `dev`. La extracción PDF
nativa con `pypdf` forma parte de la instalación base; OCR y GPU son opcionales.
`ocr` (RapidOCR) y `paddle` son motores alternativos y deben instalarse en entornos
separados porque sus distribuciones OpenCV se solapan; por ello `all` incluye el
backend RapidOCR mantenido, pero excluye Paddle.

Consulta [ejemplos end to end](docs/EXAMPLES.md),
[cookbook probado de toda la API](docs/API_COOKBOOK.md),
[arquitectura](docs/ARCHITECTURE.md), [pruebas](docs/TESTING.md),
[orquestación](docs/ORCHESTRATION_ROADMAP.md) y [migración](MIGRATION.md).

El paquete termina en evidencia original, normalización determinista y candidatos
revisables. Silver/Gold, decisiones editoriales, publicación y serving pertenecen a
`cpl-data-foundry`; la planificación recurrente pertenece a
`civic-factory-platform`.

Licensed under MIT. Release setup: [PUBLISHING.md](PUBLISHING.md).

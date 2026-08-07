# congreso-open-data

Installable, streaming-first Python API for acquiring, preserving and normalizing
official data from the Congreso de los Diputados. It covers the discoverable open-data
catalog and linked official resources: deputies, profiles, interests and financial
documents, initiatives, interventions, organs, votes, journals, documents and
transparency assets, including historical sources.

```bash
pip install congreso-open-data
congreso-open-data catalog --format jsonl
```

```python
from congreso_open_data import CongressClient, ExtractionPlan

client = CongressClient(output_root="bronze")
for resource in client.catalog():
    print(resource.family, resource.url)

plan = ExtractionPlan(families=("diputados",), output_root="bronze")
for manifest in client.extract(plan):
    print(manifest.sha256, manifest.payload_path)
```

Extractor selection is explicit. Cloud backends never run automatically and no
fallback exists unless the caller declares one. API keys are read at call time from
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, or passed directly in memory. They are redacted
from models, logs and manifests.

```python
from congreso_open_data import ExtractionSpec
from congreso_open_data.extractors import create_extractor

spec = ExtractionSpec(engine="llm", backend="openai", model="your-model-id")
backend = create_extractor(spec)
result = backend.extract(content, context)
```

Extras: `pdf`, `ocr`, `paddle`, `transformers`, `nlp`, `openai`, `anthropic`,
`local`, `parquet`, `all`, `gpu`, and `dev`. The `all` extra uses CPU-safe runtime
packages; GPU support is opt-in through `gpu`.

## Español

El paquete solo realiza extracción en bruto, preservación de evidencia, parsing y
normalización. No publica Silver/Gold, no escribe tablas de serving y nunca convierte
una inferencia probabilística en un hecho canónico. Los resultados NLP/LLM se devuelven
como candidatos revisables con evidencia y procedencia.

## English

This package is limited to raw extraction, evidence preservation, parsing and
normalization. It does not publish lakehouse tables or serving data. Probabilistic
NLP/LLM output is always represented as reviewable candidates with provenance.

Licensed under MIT.

Release setup: see [PUBLISHING.md](PUBLISHING.md).

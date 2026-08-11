# Ejemplos ejecutables

Todos los scripts usan únicamente la API pública instalada. Desde la raíz del
repositorio:

```powershell
uv run python examples/public_api_contracts.py `
  --output-dir D:\tmp\congreso-public-api

uv run python examples/interventions_last_three_months.py `
  --data-dir D:\tmp\congreso-example

uv run python examples/custom_model_callable.py `
  --data-dir D:\tmp\congreso-example
```

El segundo comando reutiliza el Bronze del primero y demuestra que el modelo puede
ser cualquier callable del usuario. No necesita OpenAI, Anthropic ni un SDK concreto.
El primer comando no usa red: prueba los 63 símbolos del import raíz con una fuente
inyectada diminuta y falla si una futura exportación pública queda sin ejemplo.

La verificación viva completa es intencionadamente un runner operativo, no código
que una aplicación deba copiar:

```powershell
uv run python examples/verify_live_all_domains.py `
  --data-dir D:\tmp\congreso-e2e `
  --report D:\tmp\congreso-e2e-report.json
```

Comprueba catálogo, diputados, intereses/bienes, iniciativas, intervenciones y texto,
callable de modelo, votos nominales, perfiles, documentos financieros, assets, texto
PDF con política OCR, órganos/membresías, retribuciones y adquisición de JSON, CSV,
XML, HTML, PDF y ZIP. Usa un trabajador, 32 MiB por artefacto, checkpoints, reanudación
y un informe JSON durable.

Los fragmentos cortos por dominio y la explicación de cuándo usar `Congress` o
`CongressClient` están en [la guía de ejemplos](../docs/EXAMPLES.md). El inventario
completo de consultas, registros, procedencia, modelos, streaming, errores,
exportación y CLI está en el [cookbook de la API](../docs/API_COOKBOOK.md).

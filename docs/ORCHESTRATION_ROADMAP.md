# Consultas y orquestación

## Disponible en 1.1

La fachada pública unifica la consulta ad hoc y el contrato que puede usar un job:

```python
from congreso_open_data import Congress, InterventionQuery

query = InterventionQuery(
    speaker="Pedro Sánchez",
    legislatures=("XV",),
    date_from="2026-05-09",
    date_to="2026-08-09",
)

with Congress(data_dir="/ruta/persistente/bronze") as congress:
    result = congress.interventions.execute(query)
    for row in result:
        consume(row)
```

El mismo contrato sirve para hoy, una ventana, una legislatura o una consulta
histórica. En intervenciones resuelve identidad, empuja filtros al servidor, pagina,
reanuda Bronze, extrae texto y reconcilia. Los demás dominios ofrecen servicios
tipados sobre el catálogo y normalizadores existentes.

Los DAGs actuales siguen siendo válidos: usan `CongressClient` y descubridores de
bajo nivel porque materializan planes físicos y checkpoints de backfill. Daily y
backfill pertenecen a `civic-factory-platform`; la publicación pertenece a
`cpl-data-foundry`.

## Siguiente evolución

La plataforma puede construir DAOs pequeños sobre estas consultas sin filtrar
detalles de endpoints a la aplicación:

- `historical_interventions(query)` para backfills por legislatura/fecha;
- `today_interventions()` como query preconfigurada;
- jobs incrementales que persistan fingerprint y `QueryRun`;
- endpoints web que traduzcan HTTP a queries públicas.

La migración operativa posterior debe sustituir construcción duplicada de planes en
DAGs por etapas y comparando manifiestos. No se reescribe de golpe un backfill
histórico validado ni se mueve Silver/Gold al paquete.

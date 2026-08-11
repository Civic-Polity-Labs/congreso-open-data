# Arquitectura final y límites

`congreso-open-data` aplica una fachada de dominio sobre puertos y adaptadores. El
contrato termina en artefactos Bronze inmutables, objetos normalizados y candidatos
revisables. No contiene lakehouse, scheduler ni política editorial.

## Árbol público

```text
src/congreso_open_data/
├── __init__.py                 # imports públicos estables
├── api/
│   ├── congress.py             # Congress y servicios por dominio
│   ├── queries.py              # filtros serializables y acotados
│   ├── records.py              # vistas enriquecidas de alto nivel
│   ├── results.py              # SearchResult + QueryRun
│   └── errors.py               # jerarquía pública de errores
├── plugins/
│   ├── models.py               # protocolo/callable agnóstico
│   └── registry.py             # registro runtime + entry points
├── models.py / protocols.py    # contratos de dominio y puertos
├── client.py / adapters.py     # API 1.0 y adaptador oficial
├── extractors/                 # descubrimiento/parsers por fuente
├── normalizers.py              # Bronze -> modelos públicos
├── transforms.py               # transformaciones puras
├── http.py / storage.py        # transporte acotado y Bronze
└── cli.py / exporters.py       # bordes de entrada/salida
```

Los módulos planos conservan compatibilidad 1.x. El camino recomendado para una
aplicación es `Congress`; para un DAG que ya posee un plan físico es
`CongressClient` y los contratos públicos.

## Dirección de dependencias

```text
usuario / CLI / DAG
        │
        ▼
api.Congress ───────► queries, results, errors
        │
        ▼
CongressClient ─────► protocols, models
        │
        ├───────────► adaptador HTTP ─► Bronze + manifiesto
        └───────────► normalizers ────► objetos de dominio
                                      │
                                      └► plugins de modelo
                                         (candidatos revisables)
```

Las dependencias apuntan hacia contratos. Los modelos no conocen HTTP, plugins ni
foundry. El foundry importa el paquete; el paquete nunca importa foundry, Airflow,
DuckDB, Gold, serving ni publicación.

## Responsabilidades

El paquete sí puede:

- descubrir páginas, catálogos, exportaciones y documentos oficiales;
- ejecutar GET/POST con timeout, reintentos, `Retry-After`, límites de bytes y ritmo;
- persistir cuerpo, URL efectiva, parámetros, fecha, tamaño y SHA-256;
- validar/parsear JSON, CSV/TSV, XML, HTML, PDF, PNG y ZIP de forma acotada;
- normalizar diputados, perfiles, intereses, patrimonio, iniciativas,
  intervenciones, votos, órganos, documentos y retribuciones;
- ejecutar OCR/NLP/LLM solicitado explícitamente y emitir evidencia/candidatos.

El paquete no puede:

- materializar Silver/Gold o tablas de serving;
- adjudicar candidatos, elegir verdad analítica o publicar;
- ocultar un fallback PDF/OCR como salida del parser primario;
- guardar clientes SDK, callables o credenciales en modelos serializables;
- seleccionar un backend solo por estar instalado.

## Consultas y resultados

Cada dominio tiene un `CongressQuery` inmutable. `last_months` se resuelve una vez
contra una fecha explícita y el fingerprint usa la consulta resuelta. Las consultas
incluyen límite, política parcial, orden y política de texto. Los filtros de
intervención se envían al endpoint oficial y el total se reconcilia al normalizar.

Las colecciones normalizadas usan un patrón uniforme: `.search(...)` es el atajo
ergonómico con firma explícita y `.execute(Query)` es el contrato tipado, serializable
y adecuado para un orquestador. No se fuerza ese verbo sobre operaciones distintas:
catálogo/adquisición, extracción de texto u OCR, exportación y registro de modelos
mantienen nombres que expresan su efecto. Una búsqueda solo planifica formatos que el
normalizador del dominio admite y selecciona una representación preferida por fuente.

`SearchResult` es de un solo uso; `collect(max_items=...)` falla si excede el límite.
`QueryRun` conserva recursos planificados/reutilizados/fallidos, filas
oficiales/normalizadas, duplicados, texto no emparejado, checkpoint y fallos. Un
resultado parcial no queda marcado como completo.

## Extensión de modelos

Hay tres mecanismos:

1. `ModelRegistry.register_callable(...)` para una función síncrona.
2. Implementar `ModelBackend.descriptor` y `.generate(request)`.
3. Publicar una factory en el entry point `congreso_open_data.models`.

`ExtractionTask` es serializable y solo contiene identidad, instrucciones, esquema y
límites. El runtime vive en `ModelRegistry`. Toda respuesta se valida como
`CandidateEnvelope`; cada candidato retiene todas las apariciones de su cita. Una
cita ausente se marca `literal=False`, nunca se presenta como evidencia.

## Compatibilidad

- La 1.1 es aditiva: `CongressClient`, modelos y módulos 1.0 siguen disponibles.
- Añadir campos compatibles es un cambio menor; eliminarlos o cambiar su semántica
  exige versión mayor.
- Los plugins se cargan de forma lazy y una colisión falla explícitamente.
- Una `TypeError` interna de una factory no se interpreta como otra firma.

Un módulo nuevo debe producir evidencia/normalización, funcionar sin lakehouse o
scheduler, conservar procedencia, tener límites explícitos y pruebas propias. Si no,
debe vivir en foundry o plataforma.

## Fundamento de las decisiones

- Se usa el [layout `src` recomendado por PyPA](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
  para que los tests y consumidores no importen accidentalmente archivos del árbol de
  trabajo que no estarán en el wheel.
- Los proveedores externos se descubren mediante
  [entry points de metadata](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/),
  el mecanismo interoperable de PyPA para plugins distribuidos por separado.
- `ModelBackend` es un
  [`Protocol` estructural](https://docs.python.org/3/library/typing.html#typing.Protocol):
  el usuario no hereda de una clase del paquete. `register_callable` conserva el caso
  sencillo para una función; el protocolo expresa correctamente backends con estado,
  descriptor y firma más rica.

El registro carga plugins de forma lazy, valida motor y modelo exactos y falla ante
colisiones. De este modo, instalar un SDK o un plugin no cambia silenciosamente qué
modelo se ejecuta ni la procedencia declarada de un resultado.

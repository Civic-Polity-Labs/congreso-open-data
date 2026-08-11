# Estrategia de pruebas

Las pruebas del comportamiento publicable viven junto al paquete. El foundry solo
prueba sus adaptadores, materialización, calidad y publicación; no debe ser el único
lugar donde se comprueba HTTP, OCR, parsing o normalización del paquete.

## Gate local reproducible

```powershell
uv sync --locked --extra dev --extra pdf --extra ocr
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest --cov=congreso_open_data --cov-branch `
  --cov-report=term-missing --cov-fail-under=69
uv build
uv run --no-sync twine check dist/*
```

La cobertura se calcula sobre `congreso_open_data` completo. El umbral 69 no es un
objetivo final: es el suelo honesto del gate de migración (72,02 % con ramas en la
validación inicial 1.1.0; 72,27 % en el gate actual de 313 pruebas), y solo puede
subir. No se permite volver a una lista reducida de módulos para aparentar 80 %.

La validación de compatibilidad 1.1.0 instaló el wheel desde cero en CPython 3.11,
3.12 y 3.13 y comprobó los 63 exports públicos, el TOML de contratos, los entry
points nativos, `pip check`, la CLI y una auditoría de vulnerabilidades. El gate
actual ejecuta 313 pruebas en CPython 3.13; además ejecuta desde un wheel limpio
`examples/public_api_contracts.py`, cuya tabla explícita debe coincidir exactamente
con `congreso_open_data.__all__`. Los backends opcionales se resolvieron también
juntos y por separado para detectar conflictos de importación.

## Matriz mínima

- catálogo y adquisición: GET/POST, URL efectiva, parámetros, reintentos, 403/429,
  errores 5xx, timeouts, límites por cabecera y por cuerpo;
- durabilidad: hash, byte count, manifiesto, reanudación, corrupción y publicación
  atómica;
- legislaturas: intervenciones 0–15, iniciativas históricas y actuales, perfiles y
  órganos; votos 10–15, que es el rango que ofrece la fuente estructurada;
- dominios: diputados, intereses, perfiles, patrimonio/bienes, iniciativas,
  intervenciones, diarios/documentos, votaciones, órganos y transparencia/retribución;
- formatos: JSON, CSV/TSV/gzip, XML, HTML, PDF, PNG y ZIP;
- backends: PDF nativo, Rapid/Paddle inyectados, Transformers y spaCy inyectados;
  cada backend real se instala en su job CI opcional;
- lotes: el fixture contiene más recursos que `submission_batch_size`, reconcilia el
  número lógico y comprueba múltiples objetos físicos;
- consumidores: una suite del foundry instala los dos paquetes hermanos y valida que
  no existe implementación duplicada.

La fachada 1.1 añade regresiones offline para resolución exacta y ambigua de
personas, fechas relativas, POST filtrado, paginación mayor que una página,
reconciliación oficial, iteración de un solo uso, varias intervenciones del mismo
orador y ejecución de callables. Las respuestas inválidas fallan cerradas y las citas
ausentes quedan `literal=False`.

Las pruebas offline usan fixtures representativos y nunca dependen de una web viva.
Una auditoría viva acotada complementa el gate para detectar cambios del proveedor,
pero guarda log/checkpoint y no sustituye a las regresiones deterministas.

## OCR y NLP

No se descarga ni carga un modelo en la suite unitaria. Se inyecta el motor para
verificar cajas, páginas, confianza, modelo, literalidad y estado de revisión. Un
smoke real de modelo es un gate operativo separado, con presupuesto de memoria,
artefactos y hardware explícitos. Paddle, Transformers y spaCy tienen jobs de CI que
al menos resuelven el extra y ejecutan el contrato inyectado.

El extra `ocr` usa la distribución mantenida `rapidocr` 3.x con ONNX Runtime. La
familia antigua `rapidocr-onnxruntime` no forma parte del contrato porque está en
mantenimiento decreciente y no declara compatibilidad con Python 3.13. El smoke de
release debe instalar `.[ocr]` desde el wheel en 3.13, ejecutar un PDF real y auditar
el grafo transitivo de dependencias.

RapidOCR y Paddle se validan en entornos separados: ambos proveedores instalan una
distribución distinta que expone `cv2`, por lo que combinarlos deja un import ambiguo
aunque `pip check` no lo detecte. El extra `all` usa RapidOCR; Paddle conserva su
extra `paddle` dedicado y su propio job de CI.

## Release

Una release requiere: suites y checks estáticos verdes, wheel/sdist válidos, import y
CLI desde el wheel en un entorno limpio, auditoría de dependencias y prueba conjunta
con `official-data-connectors` y `cpl-data-foundry`. El consumidor debe comprobarse
también con `uv build --no-sources` cuando las distribuciones 1.x ya estén publicadas.
La instalación limpia debe validar todos los exports de `__all__`, datos empaquetados,
entry points, `pip check` y los comandos `--help`, `backends` y `models`; importar solo
el módulo raíz no es un smoke suficiente.

En la validación integrada 1.1.0, `cpl-data-foundry` pasó 581 pruebas con su lock
reconciliado y `civic-factory-platform` pasó sus 51 pruebas actuales. La configuración
Docker Compose también debe resolver con `docker compose config --quiet`. Estos gates
comprueban tanto la API recomendada como la ausencia de implementaciones extractoras
duplicadas en consumidores.

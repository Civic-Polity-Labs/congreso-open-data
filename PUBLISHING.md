# Publishing 1.0.0

No GitHub or PyPI secret is required. This repository publishes with PyPI Trusted
Publishing (OIDC); do not create `PYPI_API_TOKEN`.

Before creating the first tag, register a pending publisher in PyPI with exactly:

- PyPI project: `congreso-open-data`
- Owner: `Civic-Polity-Labs`
- Repository: `congreso-open-data`
- Workflow: `release.yml`
- Environment: `pypi`

The GitHub `pypi` environment already accepts only `v*` tags and requires approval
from `alejandromorislara`. Once the publisher exists, create and push the annotated
tag `v1.0.0`. The workflow builds one wheel/sdist pair, checks it, retains it as an
artifact and publishes it with attestations.

Provider credentials are user runtime configuration, never repository or release
secrets. Users may set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, or inject credentials
directly into a backend instance. Ollama needs no token by default; custom
OpenAI-compatible endpoints can receive their own credential in memory. Credentials
must not be committed, logged or written to manifests.

## Español

No hay que añadir secretos a GitHub. Registra primero el publisher OIDC anterior en
PyPI y solo después publica el tag `v1.0.0`. Las claves de OpenAI, Anthropic o de un
endpoint compatible pertenecen siempre al usuario final y no forman parte del release.

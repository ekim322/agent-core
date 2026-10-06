# Development

Use Python 3.12 or later. From the repository root, install the development and
ASGI test dependencies with uv:

```sh
uv sync --extra dev --extra asgi
```

Run the tests and build distributions:

```sh
uv run --extra dev --extra asgi pytest -q
uv build
```

Run the module CLI help with `uv run python -m observability --help`.
Tests cover local telemetry and intercepted OTLP requests. They do not verify
live collector ingestion, backend retention, or retrieval.

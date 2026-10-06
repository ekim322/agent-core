# Contributor map

Read [README.md](README.md) for application instrumentation, then
[docs/architecture.md](docs/architecture.md) for module ownership and constraints.
Usage, configuration, ASGI integration, and log investigation are documented in
[docs/usage.md](docs/usage.md).

This is a Python library and finite-input query CLI. The consuming application
owns its server, tasks, collector, storage, and deployment.

Preserve public imports, CLI flags, telemetry names and fields, redaction, and
runtime lifecycle when making changes. Optional Starlette integration belongs under
`src/observability/integrations/`; core modules must remain framework independent.
Avoid live exporter calls for routine verification. Run the checks in
[docs/development.md](docs/development.md).

For substantial features, migrations, or refactors, read [docs/PLANS.md](docs/PLANS.md)
and create or continue a living plan listed in
[docs/exec-plans/README.md](docs/exec-plans/README.md). Record actual validation and
the next action at stopping points; archive only completed work.

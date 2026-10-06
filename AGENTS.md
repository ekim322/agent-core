# Agent Core

Read README.md for setup, extension points and lifecycle ownership. Before
implementation work, read docs/design-principles.md for compatibility and
reliability constraints. Implement changes substantively; existing source may
guide behavior and naming but must not be reused verbatim.
Keep the package application-independent: no consuming application imports,
business-specific agents, prompts, tools or concrete database adapters.
Keep public contracts explicit and preserve cancellation and retry ownership.
Generic observability belongs in the vendor/observability submodule. Import the
shared package directly; agent-specific execution tracing stays in
src/agent_core/utils/telemetry. Install the pinned shared package for tests.
Tests use fake models or local servers; no provider credentials are required.
Record actual checks and outstanding work in the relevant documentation or
change summary.

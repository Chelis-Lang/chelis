## Why

Authenticated Reef operations currently may invoke `gh auth token` implicitly when no explicit token is supplied. That ambient subprocess fallback obscures credential authority, behaves differently by host installation state, and grants a capability without an explicit orchestration input.

## What Changes

- Resolve configured credential sources only at the CLI/service boundary.
- Pass an opaque credential reference and authorized secret provider to the network executor; never place secret bytes in plans, diagnostics, notices, logs, or identities.
- Delete the implicit `gh auth token` and equivalent subprocess/environment fallback from authenticated Reef operations.
- Return a structured missing-credential/capability diagnostic when an operation requires authentication and no explicit credential reference is available.
- Preserve explicitly configured authenticated behavior and unauthenticated public operations.

## Capabilities

### New Capabilities

- `explicit-reef-credentials`: Authenticated Reef operations use explicitly resolved credential authority and never launch an ambient credential-discovery subprocess.

### Modified Capabilities

None.

## Impact

This is an approved security correction affecting Reef credential resolution, CLI configuration, integration tests, and user documentation. It does not implement the broader Reef workflow protocol, transaction plans, blob ownership, or remote publication state machine. `separate-reef-workflow-io` treats this correction as a prerequisite and retains its regression tests. Before implementation, `establish-fcis-contract-mechanics` registers stable coverage IDs, the typed credential-scope/secret boundary, and the fail-closed oracle so missing no-child or secret-exclusion evidence cannot be counted green.

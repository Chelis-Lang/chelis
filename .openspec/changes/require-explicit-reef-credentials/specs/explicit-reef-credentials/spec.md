## ADDED Requirements

### Requirement: Authenticated Reef operations use explicit credential authority
An authenticated Reef operation SHALL receive an opaque credential reference resolved by an outer caller and a separately authorized provider. Credential reference, endpoint/operation scope, and provider-handle fields SHALL be private and constructible only through validated authority paths. The executor SHALL revalidate scope before secret resolution. Adapter-provided references MUST NOT self-authorize or widen access to an endpoint or operation.

#### Scenario: Explicit credential enables authentication
- **WHEN** configuration resolves an authorized credential reference for the target endpoint
- **THEN** the network executor may resolve and use that credential for the operation

#### Scenario: Wrong-endpoint reference is denied
- **WHEN** a credential reference is not authorized for the requested endpoint or operation
- **THEN** execution returns a structured capability denial and performs no authenticated request

### Requirement: Ambient credential discovery is forbidden
Normal Reef operations MUST NOT invoke `gh auth token`, search for a credential helper, or inspect an undeclared fallback environment variable when explicit credentials are absent.

#### Scenario: Missing required credential launches no child
- **WHEN** an authenticated operation has no explicit credential reference
- **THEN** Reef returns the structured missing-credential/capability diagnostic and launches no `gh` or other credential-discovery process

#### Scenario: Hostile fake gh is inert
- **WHEN** `PATH` contains a fake `gh` executable and no explicit credential is supplied
- **THEN** the executable is not launched and cannot alter the operation result

### Requirement: Secret bytes remain executor-only
Secret bytes SHALL be supplied only to the authorized network executor through a redacted secret wrapper with no persistent serialization or revealing debug representation and MUST NOT enter plans, requests intended for persistence, notices, diagnostics, logs, machine output, identities, snapshots, or replay records.

#### Scenario: Provider error does not leak a token
- **WHEN** credential resolution or an authenticated request fails after secret access
- **THEN** returned and rendered errors contain only stable redacted provider/operation context

#### Scenario: Opaque reference is not a secret value
- **WHEN** policy or planning code receives a credential reference
- **THEN** it cannot inspect secret bytes or use the reference outside its authorized executor

### Requirement: Public unauthenticated operations remain available
Removing implicit credentials SHALL NOT force authentication for an endpoint and operation that explicitly permit anonymous access.

#### Scenario: Public read succeeds without credential
- **WHEN** a public package read requires no authentication and no credential reference is supplied
- **THEN** the operation proceeds anonymously without a discovery subprocess

#### Scenario: Required authentication does not silently downgrade
- **WHEN** operation policy requires authentication but no credential is supplied
- **THEN** Reef fails explicitly rather than attempting an anonymous fallback

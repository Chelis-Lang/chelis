## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, exact configuration/provider/executor boundary, stable requirement/scenario IDs, explicit polarity, and scenario-to-fixture-to-oracle traces in the FCIS contract manifest
- [ ] 0.2 Register typed credential scope/reference/secret constraints plus no-serialization/debug, wrong-scope, no-child, and secret-exclusion fixtures
- [ ] 0.3 Make `.venv/bin/python scripts/fcis_gate.py reef-explicit-credentials` exist and fail on the current `gh` fallback/no-child fixtures; missing evidence is never skipped
- [ ] 0.4 Pass the owning FCIS contract registration/traceability slice before credential behavior changes

## 1. Lock Credential Behavior With Tests

- [ ] 1.1 Add configured-credential positive tests for every affected authenticated operation
- [ ] 1.2 Add required-auth missing-credential tests that assert the structured diagnostic and zero child launches
- [ ] 1.3 Add hostile `PATH`/fake-`gh` and undeclared-environment negative fixtures
- [ ] 1.4 Add wrong-endpoint/operation authorization and provider-failure tests
- [ ] 1.5 Add secret-exclusion tests over diagnostics, notices, logs, machine output, and any persisted structures
- [ ] 1.6 Add public unauthenticated positive coverage and required-auth no-silent-downgrade negative coverage
- [ ] 1.7 Commit the stubs and verify no-child/source-absence cases fail before implementation

## 2. Require Explicit Credentials

- [ ] 2.1 Define opaque endpoint/operation-scoped credential references at the outer configuration boundary
- [ ] 2.2 Restrict secret resolution to the authorized network executor
- [ ] 2.3 Route configured authenticated operations through explicit references
- [ ] 2.4 Delete `gh auth token`, helper discovery, and undeclared environment fallback branches
- [ ] 2.5 Return stable missing-credential/capability diagnostics without leaking secret/provider internals
- [ ] 2.6 Preserve operations that explicitly permit anonymous access

## 3. Documentation And Acceptance

- [ ] 3.1 Document explicit credential configuration and migration from implicit GitHub CLI login
- [ ] 3.2 Implement `.venv/bin/python scripts/fcis_gate.py reef-explicit-credentials`
- [ ] 3.3 Run the oracle and require exit 0, an empty error list, zero discovery subprocesses, no secret leakage, and preserved configured authentication
- [ ] 3.4 Record this change as a prerequisite of `separate-reef-workflow-io` without counting the larger workflow oracle as completion evidence

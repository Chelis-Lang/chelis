## Context

Reef credential lookup currently checks configured environment and may run `gh auth token` as an implicit fallback. The fallback couples package/network behavior to executable discovery and hidden host login state.

## Goals / Non-Goals

**Goals:**

- Make credential authority an explicit outer-boundary input.
- Remove all implicit credential-discovery subprocesses.
- Keep secret bytes confined to the authorized network executor.
- Preserve configured authentication and public unauthenticated operations.

**Non-Goals:**

- Implementing the Reef request/observation workflow, local transactions, or remote publication plans.
- Defining persistent credential storage.
- Serializing secret bytes or ephemeral provider handles.

## Decisions

### 1. Outer callers resolve credential references

CLI/service configuration resolves whether an authenticated operation has an allowed credential source and supplies an opaque `CredentialRef`. The reference identifies a separately authorized secret provider; it is not itself secret and cannot self-assert authorization.

### 2. Secrets are executor-only

Only the network executor authorized for the endpoint may resolve a `CredentialRef` to secret bytes. Secret bytes do not enter package policy, plans, notices, diagnostics, logs, machine output, cache keys, or test snapshots. Errors expose only stable provider/operation classifications.

### 3. Missing credentials fail without discovery

When authentication is required and no explicit reference exists, Reef returns `missing_credential`/`missing_capability` according to the owning public error surface. It does not inspect unrelated environment variables, search `PATH`, invoke `gh`, or silently downgrade an operation whose policy requires authentication.

Public unauthenticated reads remain available when the operation and endpoint permit them; absence of a credential is not globally fatal.

### 4. Credential scope and secrecy are type-enforced

`CredentialRef`, `CredentialScope`, and the authorized provider handle have private fields and validated constructors. Scope contains canonical endpoint identity and a closed operation class; an adapter cannot manufacture or widen it. The executor accepts only a validated endpoint/operation-scoped reference and rechecks that scope before secret resolution. Secret material uses a redacted wrapper with no `Serialize` or revealing `Debug` implementation and cannot enter plan, diagnostic, notice, log, machine-output, identity, snapshot, or replay types.

Before behavior changes, this change registers stable requirement/scenario IDs, no-child and secret-exclusion fixtures, the exact credential resolver/provider/executor boundary, and `reef-explicit-credentials` in the FCIS contract manifest. The hostile fake-`gh` fixture must be registered while it still fails; absent evidence is blocked rather than skipped.

## Risks / Trade-offs

- Users relying on GitHub CLI login without explicit Chelis configuration will receive a migration diagnostic.
- Tests must prove both no child launch and no secret leakage; asserting only the returned error is insufficient.

## Migration Plan

1. Add explicit-config positive tests and missing-credential/no-child negative tests.
2. Add secret-exclusion fixtures across diagnostics, logs, notices, and machine output.
3. Move credential resolution to outer configuration and pass opaque references.
4. Delete `gh auth token` and equivalent fallback branches.
5. Update migration documentation and run the oracle.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py reef-explicit-credentials
```

The runner must cover configured authentication, permitted unauthenticated access, required-auth missing credentials, hostile `PATH`/fake `gh`, provider failure, and secret-exclusion fixtures. Success means exit status 0, an empty error list, zero credential-discovery subprocess launches, no secret bytes in returned or rendered structures, and preserved explicitly configured authenticated behavior.

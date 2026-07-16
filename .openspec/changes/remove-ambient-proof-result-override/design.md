## Context

`solve_property`/Tier B proof execution reads `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` to manufacture a raw SMT result for tests. Because the branch is compiled into production proof code, ambient process state can affect a trust-critical verdict path.

## Goals / Non-Goals

**Goals:**

- Remove every production read of the forced-result variable.
- Keep tests deterministic through explicit raw scripted outcomes.
- Prove hostile environment values cannot manufacture `Proved`.
- Preserve established mapping for explicitly obtained raw outcomes.

**Non-Goals:**

- Introducing `EngineSpec`, request/observation dispatch, replay, or proof caching.
- Changing mathematical encodings, soundness policy, tiers, or public artifact shape.

## Decisions

### 1. Tests inject raw outcomes explicitly

A compact test-support fixture accepts a scripted raw Tier B outcome through a Rust value constructed by the test. It does not read process environment and does not construct final composite proof status directly. Existing mapping code remains responsible for turning the raw outcome into the established fail-closed result.

The fixture is available only to unit/integration test support. Because Rust integration tests compile the library without `cfg(test)`, the implementation mechanism must be a dev-only helper or private test harness rather than a production Cargo feature that downstream builds can enable.

### 2. Production result substitution is deleted, not ignored

The named environment read, parser, and branch are removed from every production target. A hostile environment test sets the old variable to `proved`, `disproved`, `timeout`, `unknown`, invalid, and oversized values and verifies that production request/engine behavior is identical to the variable being absent.

### 3. Exact boundary

Outer engine discovery variables such as `CHELIS_BEACON_BIN` remain outside this change. The acceptance gate is specifically about variables that substitute an engine result or verdict, not variables that resolve an explicit adapter descriptor.

### 4. Test support is mechanically unreachable from production

The scripted raw-outcome implementation lives in a separate dev-only test-support target or harness crate, not behind a production Cargo feature. Normal `chelis-prove` dependencies and production feature combinations have no dependency edge to that target and expose no constructor for it. Compile/public-API fixtures prove the negative boundary; source-absence checks for the removed variable supplement that type/dependency isolation rather than replacing it.

Before result-substitution removal, this change registers stable requirement/scenario IDs, hostile-environment and production-absence fixtures, the test-support/production boundary, and `proof-forced-result-removal` in the FCIS contract manifest. The initial registered oracle must fail on the existing branch; a missing runner or fixture cannot be counted as red or green evidence.

## Risks / Trade-offs

- Existing integration tests may rely heavily on the environment shortcut. Migrate them to one reusable explicit fixture before deleting the branch.
- A test-only Cargo feature would be enableable in production and is therefore not sufficient isolation.

## Migration Plan

1. Add explicit raw scripted fixtures and hostile-environment test stubs.
2. Verify the `proved` hostile case fails before deletion.
3. Migrate affected tests to the explicit fixture.
4. Delete the production environment branch and parser.
5. Run the acceptance oracle.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py proof-forced-result-removal
```

The runner must execute default and SMT-enabled hostile-environment tests, scripted proved/disproved/timeout/unknown/error mapping, production-source absence checks, and public artifact compatibility. Success means exit status 0, an empty error list, no production reference to `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` or an equivalent result-substitution variable, and no hostile environment value capable of producing or changing a green verdict.

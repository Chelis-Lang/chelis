## Why

`chelis-prove` currently honors `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` in production proof-verdict code. Ambient process state can therefore substitute a trust-bearing solver result. This security correction is independently valuable and must not wait for the larger proof request/observation migration.

## What Changes

- Add hostile-environment positive and negative tests around the production proof entry points.
- Replace tests that force SMT outcomes through process environment with an explicit scripted solver fixture that supplies raw test observations.
- Delete `CHELIS_PROVE_TEST_FORCE_SMT_RESULT` handling and equivalent production result-substitution branches.
- Preserve the established timeout, unknown, error, disproved, and proved mapping when those raw outcomes come from an explicitly selected test fixture or real engine.
- Keep the fixture test-only and unable to grant production trust authorization.

## Capabilities

### New Capabilities

- `proof-result-override-hardening`: Production proof verdicts cannot be fabricated by ambient result-substitution variables.

### Modified Capabilities

None.

## Impact

This narrowly affects `chelis-prove` Tier B outcome injection and the tests that depend on it. It does not introduce the future engine request protocol, redesign engine registration, change proof tiers, or add caching. The larger `make-proof-verdict-inputs-explicit` change treats this completed correction as a prerequisite and retains hostile-environment regression coverage. After the standalone `establish-dylint-tooling` oracle passes, `establish-fcis-contract-mechanics` registers stable coverage IDs, the production/test-support dependency boundary, and the fail-closed oracle so missing or production-reachable scripted evidence cannot be counted green.

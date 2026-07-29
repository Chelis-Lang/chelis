## Why

`opaque-domain-construction` currently performs an unfiltered repository walk and reparses the entire Surf corpus for every checked `.ch` file. As reported in Chelis-Lang/chelis#603, this makes `chelis lint --check .` quadratic in the Surf corpus and allows long-lived `target/`, `.git/`, and `.venv/` trees to turn the local gate into a multi-hour operation.

## What Changes

- Discover the opaque-domain Surf corpus through the lint walker's canonical directory exclusions.
- Build the opaque-domain Surf catalog once per `chelis_lint::lint` invocation and reuse it for every Surf file checked during that invocation.
- Preserve the rule's existing cross-file diagnostics, file-scoped module-less shadowing, single-file style-gate behavior, and Deep-source behavior.
- Add positive and negative regression coverage for filtered discovery, one catalog build per lint invocation, and unchanged opaque-construction diagnostics.

## Capabilities

### New Capabilities

- `opaque-domain-catalog`: Defines filtered corpus discovery and lint-run-scoped catalog reuse for the `opaque-domain-construction` rule.

### Modified Capabilities

None. OpenSpec has no existing capability for lint-run corpus preparation.

## Impact

- Affected crate: `chelis-lint`.
- Primary code: `crates/chelis-lint/src/lib.rs`, `crates/chelis-lint/src/walker.rs`, and `crates/chelis-lint/src/rules/opaque_domain_construction.rs`.
- Tests will cover the crate API and rule behavior; CLI-visible diagnostics and exit semantics remain unchanged.
- No dependency, file-format, or public command-line compatibility changes are expected.

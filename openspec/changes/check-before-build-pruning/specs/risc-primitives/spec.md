## ADDED Requirements

### Requirement: Eval-only rejection uses the retained compile target

After complete semantic checks accept the selected program, `chelis build` SHALL apply the compiled-target rejection from [`spec/05-risc-primitives.md` `[05-HOST-2]`](../../../../../spec/05-risc-primitives.md) to the retained compile target.

Build SHALL remove well-typed unreachable definitions from a linked Reef target when they directly or transitively reference `chelis_ir::host::EVAL_ONLY_HOST_BUILTINS`. Removal SHALL occur before backend checks and code emission.

This package requirement SHALL NOT change the existing removal and public-surface behavior for a loose source.

A retained definition that reaches an eval-only builtin SHALL fail through the `Unsupported` channel. No backend SHALL emit a stub or default value for that call.

This scope SHALL NOT change the separate whole-program `tensor_scan` rule in `spec/05-risc-primitives.md` §3.6.

#### Scenario: Reachable eval-only builtin is rejected

- **WHEN** a retained definition reaches an eval-only builtin during a compiled build
- **THEN** build rejects it through `Unsupported` and directs the user to `chelis eval` or `chelis test`

#### Scenario: Well-typed unreachable eval-only definition is removed

- **WHEN** a well-typed unreachable definition in a linked Reef target directly references an eval-only builtin
- **THEN** build removes that definition before backend checks and emits no code for it

#### Scenario: Well-typed unreachable wrapper chain is removed

- **WHEN** well-typed unreachable wrappers transitively reference an eval-only builtin at arbitrary depth
- **THEN** build removes the complete tainted chain and does not report a fabricated unbound-variable error

#### Scenario: Invalid unreachable eval-only definition fails first

- **WHEN** an unreachable eval-only-tainted definition contains a type, effect, or linearity error
- **THEN** the complete semantic gate rejects the error before build applies eval-only removal

#### Scenario: Unreachable tensor scan remains rejected

- **WHEN** a well-typed entry-unreachable definition calls `tensor_scan` in a compiled build
- **THEN** the separate whole-program `tensor_scan` gate rejects the build

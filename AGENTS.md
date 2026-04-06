# Chelis Agent Contract

Canonical agent instructions for this repository.
`CLAUDE.md` should resolve to this file so Claude-style and Codex-style entry points do
not drift.

## Quality Standards

### Spec-First Development

- Before writing implementation, write test stubs derived from the owning spec.
- Every spec requirement should have a corresponding test before the code exists.
- If the spec says "X is a type error," write the failing test before implementing the checker.
- A phase is not done until every active spec requirement has both positive and negative
  coverage.

### Negative Test Parity

- For every test that checks something works, add the corresponding failure test.
- If you cannot name the failure case, the spec understanding is still weak.

### Do Not Trust Green

- Passing tests prove alignment with the tests, not necessarily with the spec.
- After green CI, check what active requirements still lack tests.
- Audit silent fallbacks, default values, empty error vectors, and `unwrap_or` paths.

### Phase Completion Criteria

- Do not claim a phase is done based on crate-local green or narrative progress.
- A phase is done when the executable acceptance surface is green, docs are honest, and
  the red team can only find minor residual issues.
- Budget for at least one adversarial validation pass that runs code, not just source review.

## Red Team Protocol

### Baseline

- Fresh review context is preferred when practical.
- Red team against the spec, the code, the tests, the examples, and the CLI behavior.
- Execute tests and commands; do not treat source inspection as sufficient proof.

### Required Red-Team Behaviors

- Write and run adversarial tests when coverage is missing.
- Verify that inputs which should fail do fail, and with the right reason.
- Verify that inputs which should pass do pass, with exact outputs where applicable.
- Check docs and phase claims against the shipped behavior, not just intent.

## Documentation And Spec Sync

### Documentation Hierarchy

1. `spec/design/chelis_canonical_reference.md`
2. `spec/00-12*.md`
3. `spec/design/chelis_project_plan.md`
4. `spec/design/archive/`

If active docs disagree, fix the disagreement instead of adding a third explanation.

### Public-Surface Change Rule

When behavior changes, update the owning code, tests, docs, and examples in the same
change set:

- parser / type system / IR / backend tests
- CLI integration tests
- executable examples in `examples/`
- active specs and current-state docs

## Example Corpus Policy

- `examples/` means executable Phase 0 examples that should survive `chelis fmt` and
  `chelis check` cleanly.
- `examples/illustrative/` is for syntax or design examples that are useful but not on
  the executable Phase 0 path.
- Do not mix those meanings in tests or docs.

## Build And Gate Commands

Minimum repo gate:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Phase-specific manual gates must be called out explicitly when they are not part of the
default workspace run.

## Chelis-Specific Rules

### Deep AST

- Every Deep node is a 3-tuple: `(tag {} children...)`
- Metadata map is always present at element 1
- 56-tag closed vocabulary; see `spec/03-deep-syntax.md`
- Function application is `app`, names are `var`, literals are `lit`
- RISC primitives are built-in functions, not tags

### Type System

- No implicit precision promotion
- Named tensor dimensions match by name
- No implicit broadcasting; explicit `expand` only
- Integer literals default to `int32`, float literals to `f32`

### Build Reality

- `chelis build` currently emits C, header, and runtime artifacts plus compile flags
- It does not itself invoke the native compiler yet

## Shared Local Skills

Project-local skills live in `agent-skills/`.
`.claude/skills` and `.codex/skills` should resolve to that same directory so both tool
surfaces load the same skill library.

Current shared skill set:

- `redteam-exec`
- `spec-sync`
- `phase-gate`
- `backend-numerics`
- `example-corpus`


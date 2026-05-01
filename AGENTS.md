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

### One Acceptance Oracle Per Phase

- Every phase must name one authoritative completion oracle.
- That oracle may be a single command, a named suite, or a documented manual runner, but
  it must be explicit.
- Supporting evidence may exist, but it does not replace the oracle.
- If the oracle is manual or long-running, document it in the owning phase plan and the
  current-state docs.

## Red Team Protocol

### Baseline

- Fresh review context is preferred when practical.
- Red team against the spec, the code, the tests, the examples, and the CLI behavior.
- Execute tests and commands; do not treat source inspection as sufficient proof.
- For this repository, a requested "red team agent" means a fresh local subagent in a
  new context. A main-thread validation pass does **not** satisfy that request.

### Required Red-Team Behaviors

- Write and run adversarial tests when coverage is missing.
- Verify that inputs which should fail do fail, and with the right reason.
- Verify that inputs which should pass do pass, with exact outputs where applicable.
- Check docs and phase claims against the shipped behavior, not just intent.

### Fresh-Context Enforcement

- When asked to run a red team or "spawn a red team agent", first close any known stale
  or failed subagents from the current session and then spawn a new local subagent with
  fresh context.
- If the first spawn attempt routes to remote infrastructure, errors, or comes back in a
  broken state, close that handle and retry until you have either:
  1. a working fresh local subagent, or
  2. an explicit statement that red-team validation is blocked because fresh local
     subagent execution is unavailable.
- Do not substitute main-thread validation and call it a red team.
- Do not mark a phase as red-teamed unless the fresh-context subagent actually ran the
  validation work.

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

## Contract Invariants

Machine-facing contracts should be expressed as invariants and locked with tests.

Examples:

- if a command reports perfect success, its error list must be empty
- formatter output must remain parseable on the supported corpus
- decompiler output must round-trip through the supported parser path
- executable examples must remain executable after canonical formatting
- status docs must not claim a stronger guarantee than the repo actually proves

When a public surface has an implicit invariant, make it explicit and test it.

## Example Corpus Policy

- `examples/` means executable Phase 0 examples that should survive `chelis fmt` and
  `chelis check` cleanly.
- `examples/illustrative/` is for syntax or design examples that are useful but not on
  the executable Phase 0 path.
- Do not mix those meanings in tests or docs.
- Decide the executable-vs-illustrative split early in a phase, not after examples have
  already been used as proof artifacts.

## Scripting Language Policy

- **Python** for all scripts, utilities, report generators, and automation helpers.
  Write tests for them.
- **Rust** where the task naturally fits a compiled workspace member.
- **Never shell.** Do not write `.sh` scripts. If a CI step needs a one-liner, invoke
  Python instead. Shell is fragile and untestable.
- Existing `scripts/` directory uses Python; follow that convention.

## Build And Gate Commands

Minimum repo gate:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Default-gate discipline:

- `cargo test --workspace` is the inner development loop and should stay under roughly 60
  seconds on a machine without GPU/PyTorch
- tests that exceed that budget or require heavyweight local prerequisites should be
  `#[ignore]` by default and invoked through a documented manual gate
- every ignored test must have a concrete manual command and expected success condition in
  the owning phase docs

Phase-specific manual gates must be called out explicitly when they are not part of the
default workspace run.

## Local HIP Environment

This repository is currently being worked on from a real AMD/ROCm machine, not a
CPU-only dev box.

- Host OS: Fedora 43 (`Linux fedora 6.18.16-200.fc43.x86_64`)
- CPU marketing name from `rocminfo`: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`
- GPU marketing name from `rocminfo`: `Radeon 8060S Graphics`
- ROCm ISA from `rocminfo`: `amdgcn-amd-amdhsa--gfx1100`
- `hipcc` on PATH: HIP `7.13.26162-1140233ffe`, installed via the
  `_rocm_sdk_core` Python wheel at
  `~/.local/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin/clang++`
- `rocminfo` is available and should be treated as the source of truth for local GPU
  probing
- `rocm-smi` may be absent; do not assume it exists before using it in instructions or
  validation scripts

**Toolchain drift — HIP gates currently broken (as of 2026-04-29).** The
Python-installed `hipcc` 7.x compiles against the system HIP headers under
`/usr/include/hip/`, which were installed by the older Fedora ROCm package and
do not define `__AMDGCN_WAVEFRONT_SIZE`. Even a trivial `__global__ void noop()
{}` kernel fails to build with:

```
/usr/include/hip/amd_detail/amd_warp_functions.h:96:37: error: use of
undeclared identifier '__AMDGCN_WAVEFRONT_SIZE'
```

Consequence: `cargo test --workspace` fails its HIP-related tests, and any
manual HIP gate that drives `hipcc` also fails until the toolchain is
reconciled (either by aligning system HIP headers with the wheel-installed
clang, or by routing around the Python wheel and using a single coherent ROCm
install). This is a workstation configuration issue, not a code defect.

Implication for agent work: HIP validation gates on this machine are
**currently NOT runnable** despite the GPU and `rocminfo` being healthy. Do
not claim a HIP gate as runnable unless `hipcc` itself actually compiles a
trivial kernel. The reconciliation work is out of scope for any individual
phase; flag it as a workstation prerequisite when a HIP gate is needed.

## Manual Gates

- Every manual acceptance gate must have a documented command, expected success condition,
  and owning phase.
- If default CI does not run the gate, the docs must say so directly.
- Ignored tests are allowed only when they clearly mirror a documented manual gate or an
  environment-dependent prerequisite.
- Phase summaries must not imply that a manual gate is part of the default workspace pass
  when it is not.

## CLI Surface Discipline

- CLI commands are part of the product surface, not wrappers around library tests.
- Formatter, decompiler, evaluator, checker, and build-command behavior should be tested
  against a corpus, not only single happy-path examples.
- For machine-facing CLI output, test both shape and semantic invariants.

## Surf Style Guide

When writing or rewriting Surf in this repository:

- prefer `def ... -> T = ...` over `def ... : T = ...`
- put types on function parameters, not on load-style top-level bindings
- use symbolic dimensions such as `batch` and `seq` for runtime-varying axes
- keep fixed architecture dimensions concrete
- do not annotate intermediate expressions when inference already determines the type
- keep meaningful intermediates like `h1`, `logits`, `probs`, and `loss`
- combine short tensor operations when the composed expression is clearer than over-decomposed single-op bindings
- treat decompiler-generated verbose load chains and checker-inserted ascriptions as debug output, not example style

## Chelis-Specific Rules

### Deep AST

- Every Deep node is a 3-tuple: `(tag {} children...)`
- Metadata map is always present at element 1
- 60-tag closed vocabulary; see `spec/03-deep-syntax.md`
- Function application is `app`, names are `var`, literals are `lit`
- RISC primitives are built-in functions, not tags

### Type System

- No implicit precision promotion
- Named tensor dimensions match by name
- No implicit broadcasting; explicit `expand` only
- Integer literals default to `int32`, float literals to `f32`

### Build Reality

- `chelis build` emits C, header, and runtime artifacts plus compile flags (default target)
- `chelis build --target hip` emits C/HIP host code with embedded GPU kernel strings
- Neither target invokes the native compiler — the user runs `gcc`/`hipcc` manually
- On this workstation specifically, the local HIP toolchain is currently
  broken (system-headers / wheel-clang mismatch — see "Local HIP
  Environment" above); HIP manual gates are NOT runnable until the
  toolchain is reconciled. `rocminfo` itself is healthy and remains the
  source of truth for environment confirmation.

## Shared Local Skills

Project-local skills live in `agent-skills/`.
`.claude/skills` and `.codex/skills` should resolve to that same directory so both tool
surfaces load the same skill library.
Command wrappers should stay mirrored too: `.claude/commands/` and `.codex/commands/`
should stay behaviorally aligned so slash-command access does not drift between tool
surfaces. Keep a `red-team` alias wired to `redteam-exec`, and make that wrapper enforce
stale-agent cleanup plus a fresh local subagent before any validation is counted as a
red team.

Current shared skill set:

- `redteam-exec`
- `spec-sync`
- `phase-gate`
- `backend-numerics`
- `example-corpus`
- `cli-surface`

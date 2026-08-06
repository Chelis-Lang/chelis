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

### Documentation Authority

Authority is subject-specific during the migration from numbered chapters to OpenSpec capabilities.

1. `spec/design/chelis_canonical_reference.md` controls cross-subject architecture and project boundaries.
2. For a transferred chapter, its named `openspec/specs/<capability>/spec.md` controls the subject.
3. An untransferred `spec/00-12*.md` chapter controls its subject.
4. `spec/design/chelis_project_plan.md` controls project sequence that a higher authority does not define.
5. `spec/design/archive/` is historical reference only.

A chapter transfers only through a reviewed change that records the transfer. The chapter must mark itself superseded and link the controlling capability.

If a chapter has no complete transfer record, the numbered chapter remains controlling. `openspec/specs/spec-authority-migration/spec.md` defines the complete transfer contract.

If active documents disagree, correct the document that controls the subject. Do not add a third explanation.

That ordering is for **project-level** questions: what Chelis is, what it is for, what
the roadmap says. It is not the ordering for language semantics, and the canonical
reference says so itself: "Language semantics still belong in the numbered spec
documents."

### Normative Specs Are Timeless Contracts

Normative specifications describe the decided Chelis architecture and language
semantics, irrespective of how completely any compiler version implements them. This
rule applies to an untransferred numbered chapter, a controlling
`openspec/specs/<capability>/spec.md`, and every normative `spec.md` delta under an
active OpenSpec change.

- Do not put project or implementation status in normative specs. Prohibited material
  includes status/version-draft banners, phase or milestone labels, completion claims,
  delivery histories, release or PR inventories, acceptance-oracle results, temporary
  workarounds, and descriptions of what the implementation currently happens to do.
- State the fully decided rule without weakening it to match a bug, an incomplete
  backend, or a temporary restriction. In particular, never narrow an operation to one
  dtype merely because that is the only dtype a lane implements today. The
  implementation must move toward the spec; the spec must not move toward bad behavior.
- When an implementation gap would materially mislead a reader, one short
  non-normative parenthetical may say that the requirement is not fully implemented and
  link its owning issue. It must not describe the current workaround, phase, release,
  partial inventory, or completion percentage, and it must not qualify the normative
  requirement.
- Put sequencing and status in `spec/design/`, `docs/`, GitHub trackers, or OpenSpec
  proposals/designs/tasks. A chapter-to-capability transfer records implementation
  divergence in the owning OpenSpec change, never by copying that status into the
  receiving capability spec.

### Numbered Specs Decide; Design Docs Implement

- **`spec/00-12*.md` is the authority on WHAT the language does and HOW it must
  behave.** Any decision about semantics, types, dtypes, syntax, effects, op behavior,
  diagnostics, or another user-visible contract belongs here. This is the only tier
  that outlives the work that produced it. (For a chapter with a recorded transfer,
  the Documentation Authority rules above hand that chapter's subject to its
  controlling `openspec/` capability spec - the tier boundary is unchanged: the
  normative tier decides, design docs implement.)
- **`spec/design/*.md` is the authority on how we IMPLEMENT and SEQUENCE those
  decisions**: phase plans, oracles, module layout, privacy contracts, migration
  order, consumer maps, evidence. A design doc may elaborate a numbered-spec rule and
  should record the reasoning behind it, but it does not get to decide one.
- **Where the two disagree, the numbered spec wins and the design doc has a bug.** Say
  so in the doc when you find it rather than reconciling silently in code.

**Why this matters, with a measured instance.** A design doc is a working artifact: it
is read constantly while its phases are in flight and stops being read the moment they
ship. A decision parked in one does not survive the work that made it. In 2026-07 that
produced a three-level drift. `spec/04-type-system.md` [04-NUM-2] PERMITTED one narrow
thing ("computing a single op in f64 and rounding once is a conforming
implementation"); `spec/design/dtype_semantics.md` cited that permission to MANDATE
f64 computation for every float op; the evaluator then extended the mandate to
multi-step reductions and to comparison operands. Each step was a reasonable reading
of the one above it, nobody re-checked against the numbered spec, and the result was a
language that computed f32 programs in f64.

Two rules follow, and both are cheap:

1. **When a design doc states a rule that is really a language decision, lift it into
   the numbered spec and leave a pointer behind.** `spec/05-risc-primitives.md` §8's
   [05-OBS-1..5] is the worked example: the observation contract moved out of
   `faithful_observation.md` and now survives independently of it. `spec/04` §9's
   [04-NUM-9..11] and §9.1 followed, for the trap contract, the exactness guarantee,
   and the per-dtype value table.
2. **Watch for permission-to-mandate escalation.** "X is a conforming implementation"
   in a spec does not license "therefore we do X" in a design doc, and neither
   licenses "therefore we do X everywhere" in code. If your implementation needs a
   stronger rule than the spec states, amend the spec first and say so in the PR;
   `spec/design/dtype_semantics.md` §B1 calls that "the protocol, not a failure."

### Numeric Surface Discipline

The numeric remediation's covered-family surface ratchet and typed entry
edges (`spec/design/dtype_semantics.md` §C6) bind every change that touches
numeric data, whether or not you have read that document:

- **No new numeric channel outside the tagged carrier.** A public ADT variant, wire
  field, exported C signature, exported C data declaration, or binding parameter that
  carries numeric values as
  bare `f64`/`double`, or that takes a raw integer dtype id, is a review-blocking
  finding. On every covered family the capacity census freezes both the canonical
  surface identity and its enforcement-relevant derived classification: an UNFLAGGED
  addition cites an OPEN issue; a FLAGGED capacity seam has NO citation path at all -
  redesign onto the tagged carrier, remove it, or obtain a maintainer override in
  review. Opening a fresh issue to cite is not authorization; new capacity debt does
  not land. This applies to the stdlib ADT family on the same terms as the C ones: a
  variant or field carrying a bare float primitive is a seam, while an INTEGER
  primitive is classified `numeric-op` rather than being a seam. On a new,
  post-ratchet row, that classification owes the semantic registration below;
  the frozen initial descriptors retain their capacity dispositions without
  retroactive registrations. This is why source-faithful ingestion variants
  (`JsonInt(int64)`) are the wanted shape and a float funnel is not. The
  grandfathered seam citation and the initial non-seam permanent disposition are
  frozen to exact `(kind, canonical id, flags)` complete-descriptor lists, so neither
  can be copied onto a new row to skip its own disposition. A rename, signature
  change, or reclassification is a removal of the old descriptor plus an addition of
  the successor descriptor; the successor follows the ordinary addition rules and
  never inherits a frozen disposition by resemblance. In particular, a flagged
  successor requires the named maintainer-override path until chelis#1160's separately
  reviewed, tamper-evident relocation mechanism lands. The pre-Phase-1 baseline is deliberately partial: wire-schema and PyO3
  coverage become mandatory only through their named executable entry gates in §C6,
  and Phase 1 may not start before both are green. Coverage state comes from the
  test's typed `coverage_manifest()` (artifact, enumerator, command, expected
  success, and mutations), never an editable field in the baseline JSON.
- **A new numeric op requires an exact semantic registration in the same change
  set.** The owning family registry binds the callable's exact canonical identity to
  one verbatim, existing `[05-OP-N]` authority in
  `spec/05-risc-primitives.md`. A new callable authors that atom and its mapping
  together. A chapter substring, `[05-OBS-1]`, an absent atom, or a Rust doc comment
  is not authority; existence means a normative definition line beginning
  `> **[05-OP-N]**`, not a cross-reference elsewhere. The atom states the signature,
  per-dtype semantics at [04-NUM-8]'s declared widths, adjoint or
  non-differentiability rule, and accumulator rule where applicable. Tooling validates
  the atom group and existence; reviewers validate that the selected atom's normative
  text actually governs the callable. Review does not confer semantic authority: if
  no existing atom governs it, amend the numbered spec first and register that new
  atom. Before allocating its number, re-check the highest existing `[05-OP-N]` on
  current `main`. Authoring the atom also requires running
  `.venv/bin/python scripts/generate_rejection_registries.py --write` and committing
  the resulting `crates/chelis-types/src/rejection_registry_generated.rs`; that
  generated membership artifact is required in addition to, and is not a substitute
  for, the exact `SemanticRegistration`.
- **Never silently narrow at ingress.** Choosing a lossy dtype for ingested data
  (int64 IDs into an f32 tensor) is a decision: use the named lossy form (the
  chelis#759 pattern) or the exact dtype, never a quiet convenience cast. Better
  still, type the boundary so the checker can defend it: ingestion APIs preserve
  the source format's numeric distinctions as ADT variants (`io/json`'s
  `JsonInt(int64)` beside `JsonFloat(f64)` is the precedent), never one float
  funnel.
- **A new surface KIND** (a new serialization format, IPC channel, or export
  mechanism that can carry numbers) extends the §C6 enumerators in the same change
  set, or does not land.
- **Published C ABI is configuration-invariant, and every declaration in it is
  attributable.** Public declarations may not vary
  by preprocessor feature/context. The §C6 header leg enforces that prohibition and
  compares toolchain-stable canonical declaration identities, never a
  preprocessor's whitespace or pretty-print spelling. Published headers may contain
  no `#line` directive or hand-written linemarker - those rewrite the file
  attribution the census reads back from `cc -E`, which can delete a real callable
  export from the inventory - and every `.h` in the published include directory must
  be reachable from a declared root.
- **C numeric-callable classification is conservative.** A non-boolean,
  non-character built-in arithmetic value type - including bare `int`, `short`,
  `long`, signed/unsigned forms, pointer-sized integers, and the exact-width integer
  types - makes a callable `numeric-op`. Names and parameter-name heuristics never
  turn a future callable into plumbing. The only exception is the closed set of
  exactly three reviewed canonical declarations in
  `NON_NUMERIC_INTEGER_PLUMBING_EXPORTS`; each must also remain in an exact reviewed
  seam-disposition set. Two remain in the shrink-only pre-ratchet set; PR #1149's
  `chelis_alloc` successor remains in its closed one-off maintainer-override set.
  Neither set is a route for a future declaration. Conditional macro definitions
  likewise taint their whole connected local-include component, by either include
  spelling: a public declaration consuming a tainted token is rejected even when the
  definition lives in another header. These two rules are locked by PR #956 commit
  `6ddf1a72d6dea6770a330d5c2ef3b8fa7d023c43`; widening either exception requires
  changing this contract and the negative controls together.
- **An arithmetic spelling the census does not recognize is a BUILD FAILURE, not an
  unflagged row.** The closed list is the non-numeric one
  (`NON_NUMERIC_C_TYPE_WORDS`): qualifier and aggregate keywords plus the two
  non-arithmetic value spellings. An allowlist of arithmetic spellings can never be
  complete - `_Float16`, `__fp16`, `__bf16`, `_Decimal64`, and `__int128` were all
  classifying as dtype-free - so the rule is inverted. If you add a type word to a
  published header and the census rejects it, decide which list it belongs in; do
  not route around it. The same lock exists on the language side: adding a variant
  to `chelis_types::Prim` stops the tripwire compiling until the new dtype is
  classified. Typedef aliases are resolved before classifying, including array
  aliases (`typedef double chelis_vec4[4];`), and a typedef shape the resolver
  cannot read is rejected rather than skipped.
- **An exported stdlib `def` declares its signature.** The census reads a public
  def's numeric capacity off its `defsig`, so an exported def that declares none is
  numeric surface nobody can see. Declare it, or stop exporting the binding.
- The census, tripwire, and oracle files are guard artifacts: editing one to make
  your change pass is never the fix. The failure message names the sanctioned
  actions; take one of those.

### Public-Surface Change Rule

When behavior changes, update the owning code, tests, docs, and examples in the same
change set:

- parser / type system / IR / backend tests
- CLI integration tests
- executable examples in `examples/`
- active specs and current-state docs

## OpenSpec (captured capabilities and advisory validation)

The `openspec/` tree is the canonical OpenSpec planning and capability root.
OpenSpec planning is optional. OpenSpec validation does not gate merges.
`spec/design/spec_provenance.md` describes the future governance regime.
This regime is not active.

- The `openspec-validate` workflow uses a pinned `Chelis-Lang/ci` action.
  The action supplies Node 24.18.0 and the locked OpenSpec 1.6.0 package.
  `scripts/check_openspec.py` runs structural validation only.
  Schema findings produce warnings because the action uses advisory mode.
  Operational failures stay nonzero. The workflow uses only `contents: read`.
- `openspec validate` enumerates active changes and specifications only.
  It does not enumerate artifacts under `openspec/changes/archive/`.
- Local validation requires OpenSpec 1.6.0:

```sh
openspec validate --all --strict --no-interactive
```

- The captured capabilities cite their source chapters. Capture alone does not
  transfer authority.
- No numbered chapter is transferred in this pull request. Thus, the numbered
  chapters remain controlling, and their captured capabilities are reference material.
- Use the documentation authority rules above for all chapter and capability
  disagreements.
- `spec/design/spec_provenance.md` § OpenSpec boundary blocks the first chapter
  transfer until a separate change amends that boundary.
- Do not run `openspec init`'s tool generation. `.claude/skills` and
  `.codex/skills` are symlinks to `agent-skills/`, so `--tools claude,codex`
  writes generated skill trees into the shared skill library through both
  paths. OpenSpec is not yet wired into the agent workflow — drive the CLI
  directly.

## Issue Tracking Conventions

### One Tracking Issue Per Class

A recurring defect class gets **one tracking issue**, which is also the GitHub
**sub-issue parent** for every instance. Its body carries the plan (phases,
oracles, freeze points, the class statement); its evidence lives in the owning
design doc under `spec/design/` or in `docs/investigations/`.

**Do not create a separate META issue alongside it.** The existing META/tracker
pairs ([#727]/[#729], [#703]/[#730], and siblings) are historical, not a pattern
to copy: the METAs were filed during the 2026-07 numeric audit as evidence
records, and the trackers were filed later, when the design docs were written,
as delivery contracts. Three reasons the split has stopped paying for itself:

1. **It has already broken down.** Two of the five "METAs" are closed (#709,
   #710) while their class continues under an open #731, and neither was written
   as a META - both are instance reports the class map promoted after the fact.
2. **Sub-issues do the job the pairing was improvising.** When the tracker is
   the parent, "what belongs to this class" is a structural fact. A second issue
   whose content is a list of instances duplicates the child list and drifts
   from it.
3. **Two bodies means two things to keep honest**, and the evidence half has a
   better home: `docs/investigations/` already holds the probe corpus and the
   audit record.

Rules:

- An issue has **one** parent. When a defect splits across classes (the #689
  shape: a silent half and a support half), parent it to whichever class's
  **oracle turns green when it is fixed**, and add an explicit `Also part of #N`
  line or comment for the other. Do not leave the second half implicit; that is
  how a half gets dropped when the first parent closes.
- A tracking issue carries the `tracking` label so it can be excluded from the
  work queue. **`-label:tracking` is the work queue.**
- A class without a design doc is legitimate. Say so in the tracker ("no design
  doc is planned; this is a parent, not a plan") rather than leaving a reader to
  wonder which doc they failed to find.

### Labels

`tracking` and the `area:*` family carry navigation; `bug` and `soundness`
carry severity. Prefer the specific one - an issue labelled only `soundness`
is not findable by anyone who does not already know it exists.

| label | means |
|---|---|
| `tracking` | a hub: a class or a plan, not a work item |
| `spec-gap` | normative text was never authored. Distinct from `design-discussion`, which means the decision exists and is contested |
| `soundness` | semantics divergence, type-safety, or a wrong answer. **Not** CI tooling |
| `area:eval` | the `chelis eval` interpreter lane |
| `area:runtime` | `chelis-runtime` and the C ABI surface |
| `area:backend` | backend codegen: C, HIP, Metal, and IR lowering |
| `area:prove` | `chelis-prove`, SMT, contract discharge |
| `area:bindings` | Python bindings and the compiler-api embedding surface |
| `area:ecosystem` | shell repos, `reef conform`, ecosystem drift |
| `area:perf` | performance and benchmarking |
| `area:ci` | CI workflows, coverage, mutation testing, dev infrastructure |

`type-system` and `cli` predate this table and keep their existing meanings.

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
- **Shell exceptions.** Shell is permitted only for these artifacts:
  - The published `chelisup.sh` bootstrap runs before Chelis, Cargo, or Python exists.
  - The generated Nix `chelisup` launcher runs with only its package closure.
  - The cargo-husky `commit-msg` hook locates a repository-managed Python interpreter.
  Each artifact MUST use minimal POSIX `sh`. Each artifact MUST pass `sh -n` and
  `shellcheck` when available. All other scripts remain Python.
- Existing `scripts/` directory uses Python; follow that convention.
- **Use a uv-managed Python**, not the system Python. Install uv from
  <https://docs.astral.sh/uv/getting-started/installation/>, verify it with
  `uv --version`, and provision the project version with
  `uv python install 3.11`. Inside Devenv, use the activated environment at
  `.devenv/state/venv`. Outside Devenv, use `.venv/bin/python` or
  `uv run --managed-python --python 3.11 --no-project python`.
  `scripts/gate.py` is the one allowed `python3` bootstrap: it automatically
  re-executes through that uv command before running gate logic.
- Create a primary checkout's manual environment once with
  `uv venv --python 3.11`. A dedicated git worktree does not need to copy or
  symlink another checkout's `.venv`; for direct Cargo commands there, export
  `PYO3_PYTHON="$(uv python find 3.11)"`. An explicit `PYO3_PYTHON` is
  authoritative and an invalid path must fail rather than fall back.
  `py/pyproject.toml` pins `requires-python = ">=3.11"`. See
  [`README.md`](README.md) for the full setup.

## Build Toolchain

`chelis-python` links against `libpython`. Outside Devenv, `.cargo/config.toml`
sets `PYO3_PYTHON` to `.venv/bin/python`. Devenv overrides that variable with
`.devenv/state/venv/bin/python` and activates the same environment.

A managed Python is a hard prerequisite on every platform. Outside Devenv,
direct `cargo build` for a crate that pulls pyo3 uses `.venv/` by default.
Create it with `uv venv --python 3.11`, or set
`PYO3_PYTHON="$(uv python find 3.11)"` in a dedicated worktree. On macOS,
Apple's bundled Python reports a stale `sysconfig.LIBDIR` path; do not route
PyO3 to it. `scripts/gate.py` sets `PYO3_PYTHON` to its uv-selected interpreter
for every child command.

## Local Git Hook

Devenv installs the `no-ai-authorship` hook at the `commit-msg` stage. The hook
runs `scripts/check_commit_message.py`.

Cargo-husky remains the fallback for the manual setup. `cargo test` installs its
POSIX wrapper, which runs the same Python checker through Devenv, `.venv`, or
`uv run --managed-python --python 3.11 --no-project` in that precedence order.
All formatting and lint hooks remain disabled. CI remains the remote
enforcement boundary.

## Build And Gate Commands

Minimum repo gate:

```sh
python3 scripts/gate.py
```

`scripts/gate.py` is the single source of truth for the per-PR
developer-runnable gate. CI calls `python3 scripts/gate.py <stage>` for
each split job, and `scripts/test_gate.py` asserts the CI workflow
hand-inlines no gate command the script does not produce. To see the
canonical list:

```sh
python3 scripts/gate.py --list
# cargo build --workspace --all-targets  # ci-owned
# cargo clippy --workspace --all-targets -- -D warnings  # local + ci
# cargo fmt --all -- --check  # local + ci
# cargo run -p chelis-cli --bin chelis --quiet -- lint --check .  # local + ci
# cargo test -p chelis-types --doc  # local + ci
# cargo test -p chelis-compiler-api --doc  # local + ci
# cargo test -p chelis-pipeline-core --doc  # local + ci
# <managed-python> scripts/check_checkpoint_compile_fail.py  # local + ci
# <managed-python> scripts/pipeline_core_dependency_guard.py  # local + ci
# <managed-python> scripts/pipeline_core_documentation_guard.py  # local + ci
# <managed-python> scripts/check_pipeline_core_compile_fail.py  # ci-owned
# cargo nextest run --workspace --no-fail-fast  # full gate; CI coverage split
# # --local also runs: cargo nextest run -p <crate> --no-fail-fast for each crate changed vs origin/main
```

`python3` is only the gate bootstrap. An unmanaged invocation re-executes via
`uv run --managed-python --python 3.11 --no-project`; an active Devenv or
uv-created environment is preserved. If uv is missing, the gate exits with
installation and Python-provisioning commands.

The gate runs `cargo nextest run --no-fail-fast` (CI's actual runner), not
`cargo test --workspace`, and includes `chelis lint --check .` (the §8.6 /
§12 naming gate). All nextest profiles also set `fail-fast = false`, so direct
and non-gate CI nextest runs expose every failure instead of cancelling the
remainder. The sanitizer, macOS-smoke, LOC-report, no-AI-authorship, docs, and
smt-build CI jobs are out of scope for this script by design.

The three explicit rustdoc stages exist because `cargo nextest` does not
execute doctests. The `chelis-types` command runs the chelis#731
`ErrorWitness` contracts. The other commands run the compiler pipeline
artifact contracts.

The gate does not use `--workspace --doc`. That command makes every
workspace doc example part of the gate without a reviewed scope change.

**Doctests only run where something invokes them.** The canonical gate
invokes doctests for `chelis-types`, `chelis-compiler-api`, and
`chelis-pipeline-core`. The C-backend job also runs `cargo test -p chelis-backend-c` without a
filter. A `compile_fail` oracle in another crate runs nowhere until that
crate gains an equivalent invocation in the same change set.

The checkpoint script checks the raw-offset fixture against exact Rust
diagnostics. Its Python unit tests use fake runners and do not execute
the fixture.

Local pre-push gate (chelis#360):

```sh
python3 scripts/gate.py --local
```

`--local` runs the developer pre-push subset: workspace clippy
(`-D warnings`, compile-only), `cargo fmt --check`, `chelis lint
--check .`, all three explicit rustdoc commands, the checkpoint fixture, and
`cargo nextest run -p <crate> --no-fail-fast` for each
crate changed vs `origin/main` (committed diff plus uncommitted work;
owning packages are resolved from each member's `Cargo.toml`, not the
directory name). The derived crate list is always printed; "no crate
changes detected" means the per-crate stage was skipped, not silently
empty. The workspace nextest stage is CI-owned. Run `--local` before
pushing. Open a draft PR early. The required `Integration Tests (Linux)`
check is the authoritative CI workspace oracle. See
[`docs/local_macos_environment.md`](docs/local_macos_environment.md)
for why the workspace suite does not belong in the local loop on
macOS.

Every command's combined stdout and stderr streams live. On failure the gate
retains the complete transcript under `target/gate-failures/`, replays the
final 200 lines, and prints the stage/index, duration, exit code or signal,
relevant environment, exact rerun command, and transcript path. Successful
command transcripts are removed.

The gate normalizes `CARGO_TARGET_DIR` to an absolute path inside the current
worktree and rejects paths outside it. It also sets
`CARGO_HUSKY_DONT_INSTALL_HOOKS=1` for child builds, preventing cargo-husky
from mutating the clone's shared `.git/hooks` while sibling worktrees run.
These controls isolate writable state; concurrent agents may still contend
for CPU and make each other slower.

Documentation-only changes (Markdown/prose with no code, fixture, or
example edits) are exempt from `--local`: skip the local gate, push,
and require green CI instead. The gate's clippy/build/test stages
cannot be affected by prose, and CI still runs the lint stage plus the
Docs job (mdBook build and the `skill_suite` example validator), which
cover everything a docs-only diff can break.

Default-gate discipline:

- `cargo test --workspace` is the inner development loop and should stay under roughly 60
  seconds on a machine without GPU/PyTorch
- tests that exceed that budget or require heavyweight local prerequisites should be
  `#[ignore]` by default and invoked through a documented manual gate
- every ignored test must have a concrete manual command and expected success condition in
  the owning phase docs

Phase-specific manual gates must be called out explicitly when they are not part of the
default workspace run.

## Build Concurrency And Process Hygiene

- Concurrent agents or subagents that build or test MUST use an isolated target
  directory: set `CARGO_TARGET_DIR=target/agents/<name>`, or use a separate git
  worktree with its own `target/`. A relative `CARGO_TARGET_DIR` resolves
  against the invocation cwd, not the workspace root, so run cargo from the
  repo root or use an absolute path. Never share the primary `target/` with a
  session that may be building concurrently; cargo's target-dir lock serializes
  the builds and feature/profile differences invalidate each other's caches.
- `scripts/gate.py` enforces this for gate runs: its target directory must
  resolve inside the current worktree. An inherited absolute target in another
  checkout fails before any command runs.
- Before building, list orphaned cargo/rustc/cargo-nextest/chelis processes with
  `python3 scripts/reap_orphans.py` and reap them with
  `python3 scripts/reap_orphans.py --kill`. Inside Devenv, use
  `chelis-reap-orphans` and `chelis-reap-orphans --kill`. Review the dry-run
  listing first:
  ppid==1 cannot distinguish an abandoned build from a deliberately detached
  one (`nohup cargo build` you are still tailing). Run it from the checkout
  whose `target/` you are about to use; scoping is per-checkout. Orphaned runs
  keep burning CPU and hold the cargo lock across sessions.
- Contention diagnostic: several unrelated tests FAILing at near-identical
  wall-clock times (for example all ~217s, nextest's slow-kill) means CPU
  starvation, not code breakage. Measured 2026-06-10: the 25-test
  `rank_poly_tier3` suite took 2,434s under contention vs 24s on a quiet
  machine. Re-run on a quiet machine before treating those as real failures.
- Recommended inner loop: `cargo nextest run -p <crate> --test <file>` compiles
  only that test target.
- macOS workstation only: first-exec assessment can degrade under mass
  fresh-binary bursts and stall multi-binary test runs at ~0 CPU (chelis#356).
  Probe with `python3 scripts/preflight_exec_probe.py` (exit 1 wedged, exit 3
  slow). Inside Devenv, use `chelis-exec-preflight`. Run the probe before the
  local workspace nextest stage. If the probe reports degradation, dispatch the
  macOS Smoke CI stage manually per
  [`docs/local_macos_environment.md`](docs/local_macos_environment.md).

## Local HIP Environment

This workstation has a reconciled AMD/ROCm HIP setup, so HIP manual gates are locally
runnable. The authoritative runbook is [`docs/local_hip_environment.md`](docs/local_hip_environment.md).

**For any HIP manual gate (especially hipBLAS-linked tests), run via**
`scripts/hip_test.py` — e.g.
`scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1`.
Inside Devenv, use the equivalent `chelis-hip-test` command.
The wrapper sets `HSA_OVERRIDE_GFX_VERSION=11.5.1`, the full `LD_LIBRARY_PATH`, and
the full `HIPCC_COMPILE_FLAGS_APPEND` (including `-L` to the gfx1151 wheel lib that
hipBLAS link resolution needs). Plain `cargo test --ignored` inherits only the
`environment.d/hip.conf` defaults, which segfault hipBLAS-linked binaries at process
exit with empty output — looks like a code regression but is purely environmental.

Key durable invariant: the wheel ROCm stack is authoritative, and
`~/.config/environment.d/hip.conf` provides the `HIPCC_COMPILE_FLAGS_APPEND` include
override so non-interactive shells and `cargo test --workspace` resolve HIP headers from
the `_rocm_sdk_core` wheel. Use `rocminfo` as the source of truth for local GPU probing;
do not assume `rocm-smi` is installed.

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

## Style Gate

`chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` invoke `chelis fmt --check` and the full
blocking `chelis lint` rule set on the input file before the
front-end pipeline runs. Style failures block the build by default and
emit a one-issue-per-line diagnostic to stderr. Advisory lint warnings
report valid-but-non-preferred source and do not fail `lint --check` or
the built-in gate.

- Authoritative source of truth: `spec/01-nomenclature.md` (the rule
  spec) and `crates/chelis-lint/src/rules/` (the executable
  enforcement). The canonical formatter is `chelis_surf::format` for
  `.ch` and `chelis_deep::printer` for `.dp`.
- Override flag: `--allow-style-violations` bypasses the gate with a
  stderr warning. Use only for emergency local builds and one-off
  migrations. CI must not pass it. The flag bypasses only the style
  gate, not parse, type, effect, validation, evaluation, or backend
  errors.
- Test override env var: `CHELIS_STYLE_GATE_DISABLE=1` disables the
  gate process-wide. Reserved for the integration-test corpus that
  synthesizes ad-hoc Surf to exercise type/effect/linearity behavior;
  do not set it in production CI or in user-facing scripts.

When writing new code or fixtures, run `chelis fmt --inplace <file>`
and `chelis lint --check` before pushing. The gate replaces the older
manual checklist of "remember to run fmt"; if the gate is green and
`cargo test --workspace` passes, the change is ready.

## Surf Style Guide

When writing or rewriting Surf in this repository:

- prefer `def ... -> T = ...` over `def ... : T = ...` (enforced by the
  `surf-def-arrow-form` lint rule, §3.5)
- put types on function parameters, not on load-style top-level bindings
- use symbolic dimensions such as `batch` and `seq` for runtime-varying axes
- keep fixed architecture dimensions concrete
- do not annotate intermediate expressions when inference already determines the type
- keep meaningful intermediates like `h1`, `logits`, `probs`, and `loss`
- combine short tensor operations when the composed expression is clearer than over-decomposed single-op bindings
- pipe stages use first-argument insertion: `x |> f(y)` means
  `f(x, y)`. Use `x |> fn (v) -> f(y, v)` when the piped value belongs
  in a later argument position.
- treat decompiler-generated verbose load chains and checker-inserted ascriptions as debug output, not example style
- user code typically writes neither explicit `&` nor explicit `copy()` for fan-out
  into read-only primitives; auto-borrow handles it. Write `&x` when an exported
  API or dense signature benefits from clarity. Write `copy(x)` only when forking
  ownership for downstream consumption.
- existing fixtures and migration baselines may keep explicit `copy()`
  or `drop()` calls when they prove compatibility or preserve baseline
  evidence. `redundant-linearity-call` is advisory and should not be
  papered over with blocking-rule exceptions.
- lowered IR now carries compiler-inserted `Copy` and `Drop` nodes for implicit
  linearity. If auto-copy/auto-drop produces unexpected IR, treat it as a
  structural blocker and escalate against `spec/design/implicit_linearity.md`
  rather than papering over it as a routine fixture bug.
- type identifiers are PascalCase (`surf-type-pascal-case`, §3.1)
- function/value identifiers are snake_case (`surf-value-snake-case`, §3.2)
- functions carrying the `Test` effect are named `test_*` or `example_*`
  (`surf-test-name-prefix`, §10.1)

### Canonical Surf v0.19 (chelis#1031, shipped 0.18.4)

**The grammar changed, and the style gate enforces it.** `chelis fmt
--check` runs ahead of `build`, `check`, `validate`, and `eval --file`,
so Surf that was canonical under 0.18.3 can now fail before the
front-end pipeline runs. The authority is `spec/02-surf-syntax.md` §0.1
(canonical forms and the bidirectional contract); §P10-P12 define the
wider set of input spellings the parser still *accepts* but the
formatter rewrites.

Read that section before authoring Surf or debugging a parse error that
"should" work. The forms that most often bite:

- **A nullary definition needs `()`: `def name() -> T`, not `def name ->
  T`.** This is the highest-frequency breakage — it turned every fixture
  in chelis#1176 into a hard parse error (`expected function parameter
  list `()`, found Arrow`) on rebase. Expect it on any branch predating
  #1031.
- Applying a returned value needs explicit grouping: `(f(x))(y)`.
  Ungrouped `f(x)(y)` is rejected — Chelis has flat multi-argument
  application and no implicit currying. Juxtaposition stays rejected.
- Zero-arity decoration is dropped in *types*, kept in *expressions*:
  `Ctor()` is a Deep `app`, bare `Ctor` is a Deep `var`, and a zero-field
  record keeps `{}` to distinguish `record`/`pat-record`. `! {}` is a
  declared-pure upper bound, semantically distinct from an omitted
  effect clause.
- The unit value is `()`, the unit type is `unit`, and a singleton tuple
  is `(x,)` — that comma is semantic, not cosmetic.
- Effects carry exact casing: `Diff`, `Random`, `Accum`, `IO`, `Test`,
  `Resource(...)`.
- Non-primary transform arguments are named: `grad(f, wrt=x)`,
  `vmap(f, axis=n)`; axis zero is bare `vmap(f)`.
- Canonical output omits trailing separators and prints the canonical
  literal spelling (shortest round-trippable float, no digit separators
  or redundant zeroes).

**Migrating an existing tree:** `chelis migrate surf --from 0.18 --check
<paths>...` reports, `--inplace` rewrites as a preflighted batch
transaction (whole batch validated before any write; atomic replacement
with rollback; symlinks and multiply-hard-linked files rejected).
`--from 0.18` is the only accepted value. Identifiers that became
reserved words are **not** guessed — the migrator stops and names the
byte offset, and the rename is yours to author. Semantic boundaries are
not migrated for you: suffix adoption, overflow, non-finite values,
literal patterns, raw controls, invalid escapes, and structurally
ambiguous legacy forms still reject.

Do not reach for `--allow-style-violations` or
`CHELIS_STYLE_GATE_DISABLE=1` to get past a v0.19 failure. Those exist
for emergency local builds and the ad-hoc integration corpus
respectively; migrate the source instead.

## Chelis-Specific Rules

### Deep AST

- Every Deep node is a 3-tuple: `(tag {} children...)`
- Metadata map is always present at element 1
- 62-tag closed vocabulary; see `spec/03-deep-syntax.md`
- Function application is `app`, names are `var`, literals are `lit`
- RISC primitives are built-in functions, not tags
- **Decompilation routes through a typed Deep-to-Surf resugaring
  boundary** (chelis#1031) with a total disposition for every public
  Deep tag, printed by the shared Surf printer. It fails closed on
  malformed Deep, invalid surface identifiers, unknown `surf_*`
  metadata, incompatible literal metadata, and non-finite constructed
  values — a resugaring failure is a real defect, not output to work
  around. Canonical Deep and canonical Surf are two representations of
  one public language, bound by the three executable laws in
  `spec/02-surf-syntax.md` §0.1 (`desugar(resugar(·))`, formatter
  idempotence, and the semantic retraction). If you change either
  printer or the desugarer, those laws are the oracle.

### Type System

- No implicit precision promotion
- Named tensor dimensions match by name
- No implicit broadcasting; explicit `expand` only
- Integer literals default to `int32`, float literals to `f32`

### Build Reality

- `chelis build` emits C, header, and runtime artifacts plus compile flags (default target)
- `chelis build --target hip` emits C/HIP host code with embedded GPU kernel strings
- Neither target invokes the native compiler — the user runs `gcc`/`hipcc` manually
- On this workstation specifically, the local HIP toolchain was
  reconciled 2026-05-01 and HIP manual gates are runnable. See
  [`docs/local_hip_environment.md`](docs/local_hip_environment.md) for the authoritative
  runbook.

## Toolchain And Packaging Orchestration

Chelis has a rustup-style install/version layer and a one-command project
orchestrator. The authoritative design is
[`spec/design/chelis_packaging_and_install.md`](spec/design/chelis_packaging_and_install.md);
the user-facing guide is [`docs/book/src/install.md`](docs/book/src/install.md)
and [`docs/book/src/reef.md`](docs/book/src/reef.md).

- **`chelisup`** (`crates/chelisup`) is the toolchain installer and the
  pin-resolving `chelis` **shim**. It owns the toolchain lifecycle: bootstrap
  (`crates/chelisup/bootstrap/chelisup.sh`, the one permitted shell script),
  `install` / `default` / `list-installed` / `which` / `show` / `uninstall` /
  `self uninstall`. The store is `$CHELIS_HOME` (default `~/.chelis`):
  `toolchains/<ver>`, `bin/{chelis,chelisup}`, `reef/`, `src/`.
- **Nix `chelisup` closure:** the Nix package uses `bin/chelisup` as a wrapper
  around `libexec/chelisup`. Before `install`, the wrapper creates the staging
  root `$CHELIS_HOME/nix-gcroots/chelisup.next`. After success, it promotes
  `$CHELIS_HOME/nix-gcroots/chelisup`. A failed install preserves the prior root.
  If the install copied a new binary, the wrapper promotes
  `$CHELIS_HOME/nix-gcroots/chelisup.partial`. A new attempt recovers a stale
  staging root before it changes that root. After success, the Nix wrapper
  restores itself at `$CHELIS_HOME/bin/chelisup`. The generic Rust installer
  contains no Nix root path or cleanup logic. The installed Nix wrapper removes
  all three roots after the real `self uninstall` command succeeds. The
  `chelisupLauncherLint` flake check gates the generated launcher with `bash -n`
  plus `shellcheck`.
- **Shim resolution order** (first match wins): `+<ver>` arg → `CHELIS_TOOLCHAIN`
  → nearest `chelis-toolchain` file → nearest `reef.toml` `compiler =` pin →
  recorded default. A resolved-but-not-installed version is a loud error naming
  `chelisup install <ver>`; the shim never auto-installs on `cd` and never
  silently falls back. `+latest` is not a supported reference (concrete pins
  only).
- **`chelis reef setup [--path]`** (`cmd_reef_setup` in `crates/chelis-cli`) is
  the orchestrator: ensure the pinned toolchain (auto-install via the chelisup
  binary), `reef install --from-lockfile`, `reef src sync` when `[chelis-src]`
  is present, then a `reef doctor` summary. `reef doctor` is its read-only
  counterpart across all classes (toolchain, source crates, binary artifacts).
- **Shim-corruption trap (do not regress):** `reef setup`'s toolchain step MUST
  subprocess the real `chelisup` binary. Never call `chelisup::install::install`
  in-process from `chelis-cli`: that helper copies `current_exe()` into
  `<home>/bin/{chelis,chelisup}`, which from the `chelis` binary overwrites the
  shim with the compiler. **This is enforced at compile time:** chelisup's
  `install` and `ensure_shim_installed` are `pub(crate)`, so a call from
  `chelis-cli` is an `E0603` build error caught by the normal
  clippy/build/test stages. Keep that visibility (and the call-site comment);
  do not widen it to `pub`.
- **§5.4 invariant:** `setup`'s auto-install is *explicit* provisioning and is
  therefore exempt from the "no auto-install" rule, which governs only the
  *implicit* shim. The unknown-subcommand hint augments only clap's
  `InvalidSubcommand`.

Use the `packaging-install` skill when changing or validating any of this.

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
- `packaging-install`
- `issue-resolution`

## Downstream Shell Contract

Downstream shell repos (every repo in the canonical-reference §Shell
Ecosystem table) inherit this `AGENTS.md` through a **stamped pointer
managed block** (not a verbatim copy; machine-local environment sections
excepted — see the contract §1) AND must satisfy
[`spec/design/shell_repo_contract.md`](spec/design/shell_repo_contract.md):
pin hygiene with a mechanical multi-location consistency guard, a
per-shell `docs/CHELIS_SURFACE.md` capability inventory, the
narrowing-citation rule (every workaround cites `chelis#NNN` at the site —
file upstream, never silently work around), expected-to-fail blocker
probes under `tests_blocked/` re-run at every pin bump, negative-test
sidecars, the shared skill set materialized as an upstream pointer, and
uv-managed Python.

The contract ships **in the toolchain** as `chelis reef conform`
(audit / init / sync / bump / bump-check), backed by the
`chelis-conformance` crate — the machine-readable form of the contract's
§11 table (`MANIFEST`) and the §Shell-Ecosystem registry (`REGISTRY`),
tripwire-locked to this doc and `.github/workflows/ecosystem-drift.yml`.
Shells wire `conform audit` + `conform bump-check` into CI; pin bumps land
as `conform bump` PRs (never direct to `main`). `Chelis-Lang/school` is the
reference implementation. Changes to the contract land here first (edit the
doc **and** `MANIFEST`/`REGISTRY` in lockstep, or the tripwire fails) and
propagate to every shell via `conform sync` per the scaffolding drift rule.

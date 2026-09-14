# Cross-Lane Agreement Gate

**Status:** Prerequisite design for [#763], the exact-only, Nix-hermetic first
slice of [#754]. No `chelis lane-check` implementation exists. This document
records the four decisions that must be made before the command is written, the
slice order that follows from them, and the acceptance boundary the gate is not
allowed to soften. It does not claim any executed acceptance evidence.

**Owning specs:** `spec/05-risc-primitives.md` [05-OBS] owns the rendered
observation grammar both lanes must produce.
[`faithful_observation.md`](faithful_observation.md) §C2/§C4.3 owns the
agreement contract and the one-comparator rule. This document owns only how the
shell-invokable gate is built and sequenced; it decides no language semantics.

**Class fixed when:** one shell-invokable command runs eval and the compiled C
lane over a declared corpus, compares complete stdout through the single
production comparator, and emits a versioned report whose `proof_scope`
identifies the closure that produced the verdict — with every non-vacuity,
timeout, and toolchain-contradiction case failing loudly rather than passing.

## Summary

[#763] is not blocked on semantics. For its exact-safe corpus, byte identity
already holds: [#732] Phase 2 made both lanes render through one frozen [05-OBS]
grammar, and [#729] Phase 3 returned the former [#751]/[#761] curated gaps to
ordinary regression rows. The comparator exists and is already shared
(`chelis_types::agreement::compare_exact_observations`, consumed by
`crates/chelis-cli/tests/parity.rs`).

This draft proposes two implementation contracts for review:

1. how the command learns its Nix identity, and
2. which compile/link profile it uses, given that the one shared toolchain
   resolver contradicts the issue's strict profile.

PD1 and PD2 describe those contracts. PD3 and PD4 describe maintenance and report stability. This draft delivers no command implementation.

## What already exists, and must be reused rather than duplicated

| Need | Existing artifact | Note |
|---|---|---|
| Exact comparator | `chelis_types::agreement::compare_exact_observations` | Already the parity suite's comparator. No `f64` reparse, no tolerant fallback. Reuse verbatim. |
| Eval lane | `chelis eval --file <program>` | Existing CLI surface. |
| C emit lane | `chelis build <program> --target c --output <dir>` | Existing CLI surface. |
| Build/link/run pattern | `crates/chelis-cli/tests/parity.rs` | Working reference for emit, link against `-L. -lchelis_runtime`, run, compare. Test-local today. |
| Locked flake | `flake.nix`, `flake.lock` | Already present and locked. |
| Check scaffolding | `nix/checks.nix`, `nix/contracts.nix` | `supportedSystems` already includes `x86_64-linux`; `runtimeConsumers.x86_64-linux = "OpenBLAS"`. |
| Nix CI entry | `.github/workflows/nix-packages.yml` | `nix flake check --print-build-logs`, on `workflow_dispatch` and published release only. |
| Flake contract test | `scripts/test_nix_flake_contract.py` | Asserts against `nix/checks.nix` text; a new check enrolls here. |

The issue body says it "introduces a locked flake." That is stale: the flake,
the lock, the check set, and the `x86_64-linux` system entry all landed before
this slice. The remaining Nix work is one new check derivation plus its
enrollment, not a flake bootstrap. Record this when the issue is next updated so
the slice is not re-scoped upward by its own prose.

## Prerequisite decisions

### PD1. How `chelis lane-check` ingests Nix identity (blocking)

**Question.** The report's `proof_scope` must carry the `flake.lock`/nixpkgs
revision, Nix derivation and store identities, the resolved compiler store path,
the pinned libc/libm identity, and the math-provider identity. A process cannot
derive any of that about itself. Nothing in the repository supplies it today:
`proof_scope` appears in no source file, and `chelis-image-id` is a single
`lib.rs`.

**Why it blocks.** Every unresolved answer here is a fabrication risk. An
implementer who guesses will either shell out to `nix` from inside the CLI
(forbidden: the sandbox has no network and no `nix`), read ambient environment
variables of the implementer's choosing (unversioned, and silently absent
outside the derivation), or emit placeholder strings (inventing provenance). The
ingestion surface is a versioned public contract and belongs to the issue owner.

**Options.**

- **A — typed environment file.** The derivation writes one JSON document
  describing its own closure; the command takes `--proof-scope-input <FILE>`,
  parses it into a typed validated value at the boundary, and refuses to emit a
  verdict if a required field is missing or unparsable. Outside Nix the flag is
  absent and the command reports `proof_scope.kind = "unpinned-host"`, which is
  a diagnosis mode and never acceptance.
- **B — environment variables.** The derivation exports `CHELIS_LANE_CHECK_*`.
  Cheaper, but unversioned, silently partial, and collides with N1's rule that
  ambient variables must not steer the run.
- **C — build-time constants.** Baked into the binary. Makes the executable
  non-reusable across closures and defeats N4's distinct-`proof_scope` rule.

**Recommendation: A.** It is explicit, versionable, absent-by-default off Nix,
and it keeps the "pinned" and "recorded" halves of the proof separable, which is
what the issue's proof-scope section actually asks for. It also makes N4
mechanical: a different closure produces a different input document, therefore a
different `proof_scope`, therefore no silent verdict sharing.

**Decision owner:** brittonr (issue owner). Needs a schema version number and
the required-field list before implementation.

### PD2. Which compile/link profile the gate uses (blocking)

**Question.** The brief requires reusing real build/link code rather than
duplicating it. The one shared resolver is
`chelis_backend_c::toolchain::runtime_toolchain`. It conflicts with the issue's
strict reference profile in two concrete ways:

- `crates/chelis-backend-c/src/toolchain.rs:52` unconditionally pushes
  `-march=native`. The portable reference derivation forbids `-march=native`.
- `crates/chelis-backend-c/src/toolchain.rs:96` honors an ambient compiler
  override variable. N1 requires that hostile ambient `CC`/`CFLAGS`/`LDFLAGS`
  and OpenMP settings never enter argv or the report.

**Why it blocks.** Reusing the resolver as-is produces a verdict under a profile
the issue explicitly rules out, and quietly fails N1. The two ways out have
different blast radii and neither is the implementer's call.

**Options.**

- **A — add a strict reference profile beside the existing resolver.** A new
  public constructor in `chelis-backend-c` returns a `NativeToolchain` built
  from an explicit profile: no ambient inheritance, no `-march=native`, pinned
  optimization level, explicit `-ffp-contract`, explicit absence of fast-math,
  fixed thread count. `runtime_toolchain` keeps its current behavior for every
  existing consumer. Shares the `NativeToolchain` type and the link-flag
  vocabulary, so it is reuse rather than duplication.
- **B — change `runtime_toolchain` itself.** Correct in the long run, but it
  moves the build profile for `chelis build`, `parity.rs`, and the e2e crates in
  the same change. That is a separable slice with its own regression surface and
  its own red-team pass.

**Recommendation: A now, B tracked separately.** A keeps [#763] to one coherent
slice. B — asking whether `-march=native` should ever be the default for a
product build command — deserves its own issue and is not a lane-check question.

**Decision owner:** brittonr, with backend review.

Jeff's measured `uniform_like` case is the standing argument for why the
`-ffp-contract` value must be in the recorded argv and not merely in the
profile: same source, different bytes, entirely from the flag. The fix in
PR [#779] removed that one sensitivity at the source; it did not remove the
class.

### PD3. `flake.lock` bump owner and cadence (blocking for acceptance, not for code)

rlronan asked for this in review and it was never answered. The gate's entire
claim rests on the lock. An unowned lock rots, and a rotted lock produces a
`proof_scope` that names a closure nobody maintains. Name an owner and a bump
cadence, and state what re-validation a bump requires. Recommended: same owner
as the gate, bump on a fixed cadence plus on demand for a security advisory,
with a dispatched `nix-packages.yml` run required before any bump merges.

**Decision owner:** brittonr and rlronan.

### PD4. Report stability rules for N2 (small, but must be written down first)

N2 requires a byte-identical report from the same locked source on the same
hardware tuple. That forces explicit exclusions before the writer exists: no
wall-clock timestamps, no durations, no temp directory paths in any field, no
hash-map iteration order, records emitted in deterministic relative-path order,
and a fixed key order in every NDJSON object. Decide whether elapsed time is
excluded entirely or emitted only under a flag that the acceptance derivation
never passes. Recommended: excluded entirely from the report; timing belongs in
human-readable output.

**Decision owner:** brittonr.

## Already decided — do not reopen

These come from the issue and need no further decision: the command spelling
`chelis lane-check <FILE|DIRECTORY> [--json]`; statuses `pass`, `divergence`,
`error`; stages `discovery`, `eval`, `build`, `link`, `run`, `compare`; exit
codes 0/1/2; one program record per program plus one summary record in NDJSON;
a C-unsupported program is an error and never an inferred skip; an empty corpus,
a library-only input, and zero observable output all exit 2; integer and bool
payloads are never compared through `f64`.

## Slice order once PD1 and PD2 land

- **S1 — strict profile.** PD2's chosen constructor plus unit coverage that the
  emitted argv contains no ambient value and no `-march=native`. Independently
  reviewable.
- **S2 — the runner.** A library module owning discovery, the four stages, the
  comparator call, and the typed verdict. Timeouts kill and reap the child, and
  argv is passed as an array so program arguments keep their boundaries. Owns
  E1–E5 and D1–D3 as integration tests with focused fixtures.
- **S3 — the CLI surface and report writer.** `lane-check` wired into
  `crates/chelis-cli/src/main.rs`, PD1's typed input parsed at the boundary,
  PD4's stability rules enforced by a test that renders the same verdict twice.
  Owns P1–P4, E6, N3.
- **S4 — the Nix check.** `checks.x86_64-linux.lane-check` in `nix/checks.nix`,
  the exact-safe corpus in the derivation input, the report as a derivation
  output, and the assertion that the queried compiler target matches the flake's
  intended target platform. Enroll the new check name in
  `scripts/test_nix_flake_contract.py`. Owns N1, N2, N4.

S2 and S3 add integration targets. Any of them that routine PR CI must execute
needs enrollment in `.config/ci-test-targets.toml`.

## Corpus contract

The initial corpus is exact-safe by construction: integers including values
above 2^53, bools as scalar and tensor, and dyadic float controls whose bytes
are already identical across lanes. It lives in the derivation input, is walked
in deterministic relative-path order, and every member must produce at least one
stdout byte. Negative rows are fixtures owned by the runner's integration tests;
they are never members of the all-green reference corpus. Corpus expansion to
non-dyadic floats is follow-up work, not a ship blocker.

Per [#732]'s exit census, P3's signed-zero row must state which route it
exercises: the f32 direct-literal route preserves the sign and the f64
cast-element route does not. Pin the route in the fixture rather than the value
alone.

## Acceptance oracle and the CI boundary

The authoritative oracle is `nix build .#checks.x86_64-linux.lane-check`,
executed in a native `x86_64-linux` Nix sandbox with no network and no ambient
host toolchain. A local macOS run does not satisfy this requirement.
A configured Linux builder or the hosted Linux workflow can supply the required evidence.

Routine PR CI does not run it. `.github/workflows/nix-packages.yml` triggers
only on `workflow_dispatch` and published releases, so claiming [#763] complete
requires an explicit dispatch on the candidate and a link to that run.
A future required PR job can run the same sandboxed command.
Neither a skipped job nor an unexecuted command counts as a pass.

## Boundaries

- [#1351]'s independent semantic leg is a separate branch. This gate compares
  two lanes; it does not stand in for an independent oracle.
- [#1286] may reuse this machinery but owns its own non-vacuous ownership
  oracle. Lane agreement cannot prove ownership balance.
- [#750] is a required negative canary, not a prerequisite. The gate must report
  its missing compiled root line as a divergence.
- [#738] checks that shells wire the released gate. It does not compile
  anything.
- Float tolerance, capability-aware skips, GPU lanes, and cross-platform verdict
  comparison stay out. Each has its own owner.

## Issue map

[#754] parent gate proposal; [#763] this slice; [#732] the one-comparator rule
and the Phase 2 rendering that makes exact comparison viable; [#729] Phase 3,
which returned the [#751]/[#761] curated gaps to ordinary regression rows;
[#750] negative canary; [#738] downstream shell wiring; [#1286] and [#1351]
adjacent oracles with their own scope.

[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#738]: https://github.com/Chelis-Lang/chelis/issues/738
[#750]: https://github.com/Chelis-Lang/chelis/issues/750
[#751]: https://github.com/Chelis-Lang/chelis/issues/751
[#754]: https://github.com/Chelis-Lang/chelis/issues/754
[#761]: https://github.com/Chelis-Lang/chelis/issues/761
[#763]: https://github.com/Chelis-Lang/chelis/issues/763
[#779]: https://github.com/Chelis-Lang/chelis/pull/779
[#1286]: https://github.com/Chelis-Lang/chelis/issues/1286
[#1351]: https://github.com/Chelis-Lang/chelis/issues/1351

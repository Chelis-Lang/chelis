# Cross-Lane Agreement Gate

**Status:** Design decisions for [#763], the exact-only, Nix-hermetic first
slice of [#754]. No `chelis lane-check` implementation or sandboxed acceptance
result exists yet. This document chooses the provenance and compile/link
boundaries, orders their implementation, and leaves language semantics to the
owning specs.

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
ordinary regression rows. The production exact comparator is
`chelis_types::agreement::compare_exact_observations`.

The missing decisions are how the command receives its pinned Nix identity and
how it compiles C without inheriting the product build's native/ambient
toolchain profile. PD1 and PD2 choose those boundaries; PD3 assigns lock
maintenance, and PD4 fixes report stability. None of these choices is evidence
that the unimplemented command or Nix acceptance check passes.

## What already exists, and must be reused rather than duplicated

| Need | Existing artifact | Note |
|---|---|---|
| Exact comparator | `chelis_types::agreement::compare_exact_observations` | Accepts UTF-8 text; compare complete stdout without the lossy decoding and line splitting in `parity.rs`. |
| Eval lane | `chelis eval --file <program> --target c` | `--target c` selects the same target-aware root manifest as the C build. |
| C emit lane | `chelis build <program> --target c --emit-c --output <dir>` | Emits C, headers, and a staged runtime; its printed `Compile:` recipe uses the non-reference product profile. |
| Build/link/run patterns | `crates/chelis-cli/tests/parity.rs`; `scripts/core_fragment_parity_receipt.py` | The former uses the shared resolver; the latter runs the emitted recipe through a shell. Neither is a hermetic runner to copy unchanged. |
| Locked flake | `flake.nix`, `flake.lock` | Already present and locked. |
| Check scaffolding | `nix/checks.nix`, `nix/contracts.nix` | `supportedSystems` includes `x86_64-linux`; `runtimeConsumers.x86_64-linux = "OpenBLAS"`. |
| Nix CI entry | `.github/workflows/nix-packages.yml` | `nix flake check --print-build-logs`, on `workflow_dispatch` and published release only. |
| Flake contract test | `scripts/test_nix_flake_contract.py` | Enroll a new check here. |

The issue body says it "introduces a locked flake." That is stale: the flake,
the lock, the check set, and the `x86_64-linux` system entry all landed before
this slice. The remaining Nix work is one new check derivation plus its
enrollment, not a flake bootstrap. Record this when the issue is next updated so
the slice is not re-scoped upward by its own prose.

## Prerequisite decisions

### PD1. Nix identity is an explicit, versioned input

The CLI cannot discover `flake.lock`, store paths, libc/libm, or the derivation
that invoked it merely by examining its executable. It must not invoke `nix`
inside the sandbox, guess provenance, or silently accept a partial set of
ambient environment variables.

**Decision:** the Nix derivation supplies a version-1 JSON document through
`--proof-scope-input <FILE>`. Parsing rejects missing, malformed, or contradictory
pre-run fields. Its required categories are the source and corpus hashes; lock
hash and nixpkgs revision; relevant input derivation and store paths (including
Chelis and the compiler); executable/compiler digests, versions, and host/target
triples; libc/libm and any linked math-provider identities; the required
compile/link profile; Nix and execution platform/CPU identity; and
floating-point/thread-affecting environment. The implementation owns concrete
field names, but it may not omit an issue-required category from the final report.

The derivation constructs this input from its locked closure. The runner queries
the compiler target and, when an input is supplied, verifies the selected
compiler, libraries, and profile against it before claiming pinned provenance.
An absent or contradictory target is an error. The runner then records
the **executed** compile/link argv arrays in `proof_scope`. Those arrays depend on
staged filenames and cannot be known to the derivation before the runner stages
each program; they are report fields, not claims supplied in the input. A
different toolchain, flags, library closure, or corpus changes the recorded scope;
comparing reports from distinct scopes is not an agreement claim.

An input file is data, **not an attestation**. Anyone can pass a forged file to
the CLI outside Nix; even a complete typed document cannot establish that the
caller was sandboxed. Without the flag, the CLI may reuse `runtime_toolchain`
for **host compiler and dependency discovery only**, passing those identities
to the strict-profile constructor rather than copying its ambient flags or
printed recipe; reports say `proof_scope.kind = "unpinned-host"`. With the flag,
the CLI reports supplied and queried identities but never claims acceptance by
itself. Only the report produced and checked by
`nix build .#checks.x86_64-linux.lane-check` is the authoritative pinned verdict.
The derivation, not a user-controlled report field, binds the input document
to its closure. This also avoids baking closure-specific constants into a
reusable CLI.

### PD2. The gate needs a separate strict compile/link profile

`chelis_backend_c::toolchain::runtime_toolchain` currently inserts
`-march=native` and honors `CHELIS_CC`; `chelis build` prints that resolver's
recipe. The reference gate cannot execute or copy this printed shell command:
the portable profile forbids `-march=native`, and a hostile `CHELIS_CC` changes
the printed compiler without changing the generated C. The narrower
`scripts/core_fragment_parity_receipt.py` intentionally executes a printed
recipe with `shell=True`; its receipt is not the strict Nix acceptance gate.

**Decision:** add an explicit strict-reference constructor beside
`runtime_toolchain` in `chelis-backend-c`. Reuse `NativeToolchain` and its
requirements/link-flag vocabulary, not the ambient resolver or the printed
recipe. Supply the pinned compiler store path and native dependencies from
PD1's Nix closure. For this first reference profile, use `-O2`,
`-ffp-contract=off`, `-fno-fast-math`, no `-march=native`, and one thread.
The runner assembles C compile/link argv arrays and removes caller-supplied
compiler, linker, and OpenMP overrides from both child processes. Nix cc wrappers
also consume `NIX_CFLAGS_COMPILE` and `NIX_LDFLAGS`: replace inherited values
with derivation-authored closure flags, preserving the pinned sysroot and library
paths. Record executed argv, effective profile, and resolved library identities
in `proof_scope`.

This choice leaves `chelis build`, `parity.rs`, and the e2e crates unchanged.
Changing the product build's default profile is a separate change, not a
precondition for [#763]. A constructor-only unit test is insufficient for N1:
the runner and Nix check must exercise the spawned compiler under hostile
`CHELIS_CC`, `CC`, `CFLAGS`, `LDFLAGS`, `NIX_CFLAGS_COMPILE`, `NIX_LDFLAGS`,
and OpenMP settings. A wrapper can inject fast-math and linker flags after the
caller assembles argv; inspect the effective compiler/linker invocation, not
merely the outer vector, to prove the executed reference profile.

Jeff's measured `uniform_like` case explains why the contraction flag belongs
in recorded argv: identical source produced different bytes under different
compiler contraction settings. PR [#779] repaired one source sensitivity, not
the class.

### PD3. `flake.lock` has a maintenance owner

**Decision:** brittonr, the [#763] assignee, owns a four-week review of the
lock and an on-demand review for security advisories. A bump to shared inputs
also keeps `flake.lock` and `devenv.lock` in parity. Before merging a bump,
dispatch `.github/workflows/nix-packages.yml` on its candidate and inspect the
native Linux checks; once the lane check exists, rerun its sandboxed acceptance
command too. This answers rlronan's lock-ownership question without treating a
fresh lock as proof that any program agreed.

### PD4. Stable reports exclude run-local fields

**Decision:** versioned NDJSON contains no wall-clock time, duration, or temp
directory path. Run each case from its own scratch working directory with stable
relative staged filenames in the compile/link argv. The report records the
complete executed argv without leaking the random working directory. Program
records use deterministic relative-path order, and fields have fixed key order.
Timing, if useful, belongs only in human output. Two uncached runs of the same
locked source and declared hardware tuple must produce byte-identical report
files; a cached Nix result does not test N2.

## Already decided — do not reopen

These come from the issue and need no further decision: the command spelling
`chelis lane-check <FILE|DIRECTORY> [--json]`; statuses `pass`, `divergence`,
`error`; stages `discovery`, `eval`, `build`, `link`, `run`, `compare`; exit
codes 0/1/2; one program record per program plus one summary record in NDJSON;
a C-unsupported program is an error and never an inferred skip; an empty corpus,
a library-only input, and zero observable output all exit 2; integer and bool
payloads are never compared through `f64`.

## Slice order

- **S1 — strict profile.** Implement PD2's constructor and verify its arguments
  exclude `-march=native` and ambient overrides. This does not by itself prove
  the compiler that the runner executes.
- **S2 — the runner.** Own discovery, eval, C emission, native link, execution,
  complete-output comparison, and typed verdicts. Call eval with `--target c`;
  ignore `build`'s printed compile recipe. Pass argv as arrays, stage relative
  filenames under each scratch cwd, and kill/reap timed-out children. Record the
  complete executed argv, not the random scratch path. Validate complete stdout
  as UTF-8 without replacement, then use the single production comparator on
  the **whole** strings, including trailing newlines. Invalid rendering is an
  output-contract error, never a lossy-equal pass. Test differing final newlines
  and distinct invalid bytes as well as E1–E5 and D1–D3.
- **S3 — CLI and report.** Wire `lane-check` into
  `crates/chelis-cli/src/main.rs`; parse PD1's typed input, and enforce PD4 by
  rendering the same verdict twice. Own P1–P4, E6, and N3.
- **S4 — Nix acceptance.** Add `checks.x86_64-linux.lane-check` to
  `nix/checks.nix`; bind PD1's file to the evaluated closure, supply the
  exact-safe corpus as a derivation input, and retain the report as an output.
  Assert the queried compiler target against the declared platform and exercise
  the **effective** compile/link profile under hostile ambient and Nix-wrapper
  variables. Enroll the check in `scripts/test_nix_flake_contract.py`; own N1,
  N2, and N4.

S2 and S3 add integration targets. Enroll any that routine PR CI must execute
in `.config/ci-test-targets.toml`.

## Corpus contract

The initial corpus is exact-safe by construction: integers including values
above 2^53, bools as scalar and tensor, and dyadic float controls whose bytes
are already identical across lanes. It lives in the derivation input, is walked
in deterministic relative-path order, and every member must produce at least one
stdout byte. Negative rows are fixtures owned by the runner's integration tests;
they are never members of the all-green reference corpus. Corpus expansion to
non-dyadic floats is follow-up work, not a ship blocker.

Pin P3's signed-zero control to the f32 direct-literal route in the initial
exact-safe corpus. Do not infer that other routes discard the sign from [#732]'s
older exit census: a Darwin compile with `-O2`, `-ffp-contract=off`, and
`-fno-fast-math` preserved `-0.0` through an f64 cast-to-tensor element in
eval and C. Native Linux needs its own observation before that route joins the
reference corpus.

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

- [#1351]'s independent [05-OP-33] structural oracle has merged. This gate
  compares two lanes; it cannot replace independent semantic evidence.
- [#1286] may reuse this machinery but owns its own non-vacuous ownership
  oracle. Lane agreement cannot prove ownership balance.
- [#750] is a required negative canary, not a prerequisite. The gate must report
  its missing compiled root line as a divergence.
- [#738] checks that shells wire the released gate. It does not compile
  anything.
- Float tolerance, capability-aware skips, GPU lanes, and cross-platform verdict
  comparison stay out. Each has its own owner.
- [#2102]'s core-fragment parity receipt is a narrower, executable
  observation/trap corpus. It supplies fixtures and a build/run example, but
  neither the public command nor a sandboxed proof scope. Its shell-executed
  product recipe is not the strict profile specified here.
- Exact agreement is not an independent oracle: if both lanes omit the same
  owed root, nonempty identical output can still pass. [#1351] and the
  core-fragment receipt's declared-root witnesses remain separate evidence.

## Issue map

[#754] parent gate proposal; [#763] this slice; [#732] the one-comparator rule
and Phase 2 rendering; [#729] Phase 3's ordinary regression rows for the former
[#751]/[#761] gaps; [#2102] the narrower executable parity receipt; [#750]
negative canary; [#738] downstream shell wiring; [#1286] and [#1351]
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
[#2102]: https://github.com/Chelis-Lang/chelis/issues/2102

## Context

The source inventory is based on `d1c5cb2f`. Issue #1354's original incident is a
real stale-runtime execution; its later equal-timestamp probe demonstrates selection
order using small C archives, not a complete Chelis execution. Preserve that distinction.
Draft PR #2036 and the local WIP listed in `proposal.md` supply design and fixture
ideas, not runtime acceptance. In particular, do not reuse their first-found "good"
archive discovery, empty-archive substitutes for real stale builds, or assertions
that allow either successful selection or arbitrary linker failure.

## Goals / Non-Goals

Select the intended runtime independently of incidental filesystem ordering, retain
byte-bound evidence through native consumption, and prevent stale-correct masking of
source mutations. Cover production CLI/Python, the enumerated harness families, and
shipping producers. Arithmetic correctness beyond the chosen witnesses, malicious
build-tool compromise, package signatures, and general reproducible-build certification
are outside this change. An honest build toolchain and collision-resistant digests
remain assumptions; an identity record is not an authenticity signature.

## Decisions

### D1: Separate expected build identity, archive bytes, and compiler image

Use one versioned descriptor and canonical, domain-separated SHA-256 encoding.
Compatibility is exact equality of these runtime-specific dimensions:

| Dimension | Required input |
|---|---|
| Source closure | Logical paths and bytes of runtime sources, relevant untracked inputs, transitive local dependencies, manifests/lock resolution, build scripts, and their declared generated/configuration inputs; resolved external dependency identities and checksums |
| Public interface | Complete shipped runtime-header closure and its ABI identity |
| Target | Rust target specification/triple and effective target CPU/features |
| Features | Sorted effective runtime/dependency feature closure, including `ownership-ledger` |
| Compile configuration | Exactly the settings the runtime compilation observes directly: optimization level, debug info, `debug_assertions`, `panic` strategy, rustflags reaching the compile, and Cargo's debug/release profile class. Identity records those effects; it never infers them from the Cargo config hierarchy (`.cargo/config.toml`, `$CARGO_HOME/config.toml`, `--config` or `CARGO_PROFILE_*` overrides) or from a profile name, which build scripts see only at debug/release granularity. Settings no compilation can observe (overflow checks, LTO, codegen-units, a custom profile's name) are not dimensions; a runtime whose defined behavior depended on one would be a runtime defect, not an identity gap |
| Toolchain | Rust compiler identity and code-affecting native/build tool inputs |

Absolute checkout paths, mtimes, archive filename hashes, and release version alone
are not compatibility inputs. Feature and configuration equality applies to the runtime recipe,
not unrelated CLI features such as `smt`. There is no implicit compatibility exemption
for debug/release or ABI-preserving instrumentation: a harness using
`ownership-ledger` requests that runtime recipe explicitly.

A single owned build-input producer derives this descriptor for both the runtime
build and the consuming compiler build. It runs inside each producing crate's own
build step and reads only inputs present there: the workspace lockfile and manifests
resolve the transitive local-dependency closure and external dependency
identities/checksums without invoking Cargo, local sources are hashed from the
workspace tree, and compile configuration comes from the build environment and cfgs.
Plain `cargo build -p chelis-runtime` therefore remains a complete producer; no
wrapper is required. The CLI and Python extension embed the expected descriptor
independently of runtime discovery. Separate Nix derivations (crate2nix
`buildRustCrate`, one rustc per crate with no Cargo available) must supply the same
lockfile, manifests, and local sources to each producing crate; they need not produce
byte-identical compiler images or share derivation paths. A build that lacks a
required input, or meets an unrecordable one, fails identity production rather than
dropping it from the hash.

Keep three identities distinct: expected runtime build descriptor, SHA-256 of the
actual archive bytes, and the loaded compiler image used for provenance/cache identity.
`chelis-image-id` already identifies the loaded CLI or Python extension, not merely
`current_exe()`. Reuse it for that purpose, never as an archive checksum or as a
substitute for the expected runtime descriptor. Python does not borrow an adjacent
CLI's expectation. An unavailable image observation is reported as unavailable;
it cannot manufacture a verified runtime expectation from a degraded cache key.

### D2: Carry identity inside the built archive

Emit one versioned private identity record from the runtime compilation into a
retained object section that survives staticlib construction. Read the record from
the archive/object formats supported by the shipped targets, without executing the
candidate. The same producer supplies the compiler's expected record. Reject missing,
duplicate, malformed, or unsupported records. The record is build metadata, not a
new public C callable or a change to the callable metadata's `abi_version: 2`.

This chooses an embedded record over a mutable adjacent sidecar. A post-build script
must not stamp an old archive with whatever source is currently checked out. Build
invalidation covers additions/deletions as well as changed bytes in D1's closure.
The record participates in compilation; it is not selected from another archive.

Compute the complete archive digest at consumption, separately from its embedded
record, avoiding a self-referential digest. Packaging copies/preserves the record and
verifies it after any strip/repack operation. A distribution without a readable record
fails packaging. No sidecar inferred from a filename can upgrade it to verified.

### D3: One fail-closed resolver, with explicit discovery inputs

Put descriptor decoding, matching, selection errors, and verified staging in one
low-level shared owner usable by CLI, Python, and harness adapters without depending
on a compiler backend or making the runtime depend on the compiler. Callers supply
an expected descriptor and explicit candidate roots; they do not implement their own
winner policy. `tests/support/runtime_archive.rs` becomes an adapter to this owner,
not a second implementation.

Without an override, preserve each caller's documented discovery locations but evaluate
the complete candidate set in those locations. A nonexistent optional root contributes
no candidates; an I/O error that prevents establishing the set is a failure. Canonical
and Cargo-hashed filenames have identical standing. Unstamped/incompatible candidates
are rejected with reasons; they do not displace a valid matching candidate.

Group valid matches by build identity and complete archive digest. Exactly one group
succeeds. Byte-identical copies/hardlinks are one artifact; a stable path ordering may
choose its representative only after proving equivalence. Multiple distinct byte
groups with the same expected build identity are ambiguous and fail. Zero matches
fails. Directory order and timestamps cannot change any of these outcomes.

`CHELIS_RUNTIME_DIR`, when present, is the sole discovery root. An invalid, empty,
unreadable, or incompatible override fails without searching defaults. Retain the
existing harness-only `CHELIS_RUNTIME_LIB` exact-file pin: validate that file under
the same contract, require an absolute regular-file target, and never fall back.
When a harness receives both variables, the pin must be a regular file inside that
directory, the pair `scripts/runtime_representation_phase1.py::runtime_pin` already
sets; any other pair is conflicting and rejected.
Do not introduce an unverified exact-file escape hatch on any surface.

Errors distinguish missing, incompatible, ambiguous, malformed/unverifiable, and
changed-during-staging cases; list expected identity and candidate paths/reasons in
deterministic order. A selection failure occurs before native linking or publication
of a successful usable result. Python raises the existing `ChelisError` category.

### D4: Preserve identity through staging, linking, and reuse

Selection returns a verified artifact, not an unqualified path. Stage from the verified
open file into task-owned temporary output while hashing; revalidate the completed
copy against the selection digest and record before atomic publication. A replacement
or in-place mutation that changes copied bytes fails. The compiler's embedded public
runtime headers (the shipped runtime-header closure, not the HIP/Metal backend
extras) must match the record's public-header digest. Remove partial current-attempt
outputs or withhold the success marker; never leave a new success receipt beside old
usable artifacts after a failed attempt.

Write `chelis_runtime.identity.json` beside the staged archive. Its versioned receipt
records the consumer's expected and observed runtime descriptors, archive SHA-256,
source and staged paths, compiler-image observation, and discovery/override mode.
The CLI names the staged archive and its identity in its existing stdout artifact
report; a successful build's stderr stays empty, as the CLI's silent-success contract
requires.
A CLI receipt says **staged**, not **linked** or **executed**.

Python and native harnesses link the verified staged path explicitly, not an ambient
`-L`/`-l` search that can find another runtime. Extend their receipt with the linker
input and resulting native artifact digest before publishing a loaded/usable result.
Python compiled-artifact cache reuse, and loading artifacts produced by this path,
require a matching persisted receipt and native artifact digest; a missing or stale
receipt cannot turn a cache hit into an identity bypass. This receipt has its own
schema, separate from callable tensor metadata and compiler-context cache identities.

A user can manually link CLI output against something else; the staging receipt does
not certify that external action. The acceptance harness records the actual linker
invocation and resulting executable, then executes that product.

### D5: Make source freshness an explicit mode, not a guess about cwd

Exact compatibility with an old compiler is not proof of the current working source.
Build provenance explicitly distinguishes a source-worktree consumer from a sealed
distribution consumer. Ordinary development builds carry a declared source root and
runtime recipe; before selecting, recompute the relevant live source closure and
compare it with their embedded expectation. Dirty/untracked relevant inputs are
content, not a boolean Git flag. A changed or unavailable required root fails with a
rebuild diagnostic; it never silently downgrades to distribution mode.

Installed CLI/wheel builds deliberately use sealed distribution provenance and their
embedded expected descriptor, without reading the user's cwd or requiring the build
checkout. Package production chooses this mode explicitly; deleting a source directory
or setting a discovery override does not choose it.

Mutation/source-certification harnesses always supply and verify the checkout's live
expectation, even when invoking a sealed consumer. Thus an unchanged old compiler and
correct archive cannot certify mutated source. If source changes but the runtime is
not rebuilt, acceptance fails for freshness. If the broken runtime and its consumer
are rebuilt consistently, they are compatible and must actually execute the broken
behavior; identity checking must not reject a compatible mutant merely to get a
convenient nonzero exit.

### D6: Migrate the complete known boundary

This is the initial inventory, not a claim that lexical search proves class closure.
Implementation must enumerate remaining occurrences and give every archive selector a
disposition: shared resolver, verified exact Cargo artifact, or demonstrated nonconsumer.
No unverified selector remains on a claimed path.

| Boundary | Existing owners and required cutover |
|---|---|
| Production staging | `crates/chelis-cli/src/main.rs` C/HIP/Metal staging; `crates/chelis-python/src/lib.rs` compile/stage/link/load and reuse |
| C harnesses | `crates/chelis-backend-c/src/lib.rs`; tests `dtype_matrix_bf16_f16`, `dtype_matrix_bf16_f16_extended`, `emit_uniform_like_bit_pattern`, `exec_compile`, `fused_in_place_exec`, `issue_1281_exact_reductions`, `issue_189_const_bit_pattern`, `rt_cleanup_redteam`, `span_comments` |
| CLI/E2E harnesses | `crates/chelis-cli/tests/cli.rs`, `cbackend_{cast_arithmetic_composition,cast_memcpy,empty_tensor_numel,print_tensor_f64,reshape_memcpy}.rs`, `issue_300_grad_codegen_compiles.rs`, `issue_352_captured_global_c_emit.rs`, `ws2b_numeric_identifier_divergence.rs` (`issue_1314_json_bigint.rs` already selects the exact Cargo `compiler-artifact` and is the pattern to keep); `crates/chelis-e2e/src/bench.rs`, `tests/eval_agreement.rs`, `tests/spec_suite.rs` |
| Accelerator harnesses | HIP `bf16_f16_matmul`, `codegen_structure`, `gpu_correctness`, `span_comments`, `logical_comparison_where_gpu`, `device_entry_execution`, `device_owner_contract`; Metal `gpu_correctness`; shared `tests/support/runtime_archive.rs` |
| Build/package producers | Runtime Cargo build; `nix/packages.nix`, `nix/contracts.nix`, `nix/checks.nix`; `.github/workflows/release.yml` and `ecosystem-drift.yml`; Python maturin/wheel assembly |
| Installed evidence | `crates/chelisup/src/install.rs` package validation, `scripts/installed_artifact_canary.py`, runtime-representation runners, and fresh-environment wheel loading |

Nix currently selects a first hashed archive before renaming it. Replace that choice
with the producer's exact artifact output and validate the embedded descriptor when
assembling compiler/runtime outputs. Release/container copies get the same validation.
A Python wheel must contain its matching runtime/header bundle and expected identity;
it must not depend on a neighboring CLI or a developer's target directory. Include
relocation and source-free installation in package acceptance.

Reuse `scripts/runtime_representation_phase1.py`'s exact Cargo `compiler-artifact`
selection and before/after digest pattern for test setup. Obtain fixture artifacts
from the invocation that built them, filtered by package/target/kind/features, never
from a first/newest archive scan. The shared runtime-input producer remains the owner
of descriptor semantics, including for Python automation.

## Acceptance oracle and evidence

The implementation introduces one authoritative suite and command:

```text
.venv/bin/python scripts/runtime_artifact_identity_oracle.py
```

This is a planned command, not an existing executable or a result of this proposal.
It emits a receipt under `target/runtime-artifact-identity/` binding the tested head,
source closure, consumer/configuration matrix, commands, staged/archive/native digests,
results, and explicit unrun accelerator rows. A green aggregate requires all mandatory
CPU, packaging, and mutation rows; missing prerequisites are failures, not green skips.

| Evidence row | Positive control and discriminating negative |
|---|---|
| CLI | Actual `chelis build --target c`, link the staged archive explicitly, run a runtime-dependent program and assert its exact result; incompatible/missing identity fails at build, not only at the later linker |
| Python | Exercise `compile_and_load` through the extension's own compile/load/call job with exact expected values (hosted through the `chelis-python` test binary) and from a Python interpreter locally; an incompatible runtime raises `ChelisError` before a usable model is returned |
| Candidate permutations | Intended compatible archive remains usable beside real incompatible archives under equal/reversed mtimes and reversed enumeration; zero matches and distinct-byte compatible ambiguity reject |
| Identity dimensions | Matching target/features/observed compile configuration/headers/dependencies succeed; one-dimension mismatches and missing/malformed/unsupported records reject |
| Overrides | Valid directory and existing harness exact-file modes work; conflicting, missing, wrong-identity, or unreadable overrides cannot fall back |
| Mutation masking | Build good A; mutate runtime behavior in an isolated checkout to B while retaining A; no-rebuild attempt fails freshness; rebuild B and its consumers, prove B is staged/linked and its wrong result kills the behavioral oracle for CLI and Python; restore/rebuild and recover the positive result |
| Dependency mutation | Change a behavior-affecting transitive local runtime input while runtime-crate source is unchanged; old artifact cannot certify that closure |
| Staging/reuse | Byte-identical duplicates, relocation, and valid reuse succeed; swapped bytes/headers, copy races, mismatched native output, or absent/stale reuse receipts reject |
| Distribution | Fresh source-free release/Nix install executes intended runtime behavior, and a locally built Python wheel does the same; crossed compiler/runtime packages and missing embedded records fail |
| Harness closure | Every inventoried consumer uses validated selection or exact producer output; production-resolver mutation and stale-correct controls must make the corresponding executed rows fail |

Hosted evidence is bounded by the lanes that exist: the workspace nextest lanes build
the CLI and the `chelis-python` crate, so the CLI witness and the extension's
compile/load/call path run hosted there, and the Nix package checks cover the Nix
outputs. No hosted lane builds a Python wheel or imports the extension from a Python
interpreter today, and this change adds none: the wheel witness and the
interpreter-level Python witness are local and manual-gate evidence, recorded as such
in the receipt. C/HIP/Metal archive staging is mandatory without requiring GPU
hardware. GPU execution remains in the existing HIP/Metal manual or hardware-owned
lanes; record their commands and actual results separately. CPU green never claims
accelerator execution or all arithmetic correctness. Record every remaining selector
disposition before asserting #1354 closure.

## Migration Plan

1. Author the public rules in `spec/08-backends.md` and `spec/11-ffi.md`; synchronize
   the existing captured/package contracts. Land fixtures before selector implementation.
2. Implement common identity production/decoding, proving ordinary Cargo and separate
   package derivations can supply matching descriptors, including Python's extension.
3. Migrate producers, production consumers, and harness adapters as one coordinated
   cutover. Rebuild/reinstall paired compiler, extension, runtime, and headers; do not
   temporarily accept unstamped archives to accommodate old installations.
4. Update runtime rewrite, packaging, manual gates, phase-oracle documentation, and
   changelog; reconcile PR #2036 rather than leave contradictory draft decisions.
5. Strictly validate the proposal, execute the oracle and adversarial controls, then
   inspect exact-head hosted/package receipts. Only those receipts can support issue
   closure; merging this plan cannot.

Rollback is a coordinated package/code rollback, not a resolver fallback. During an
incomplete deployment, fail runtime-consuming operations until a matching set is
installed. Reverting enforcement withdraws the identity guarantee and reopens the
issue; it is not a passing acceptance state. No new governance policy is activated.

## Risks / Trade-offs

- Full input closure is harder than hashing `chelis-runtime/src`: centralize derivation,
  include transitive/build inputs, and require source/dependency mutation controls.
- Exact configuration/toolchain/feature matching rejects some ABI-compatible combinations:
  intentional conservative policy, with explicit recipes for instrumentation.
- Archive metadata retention varies across object formats and packaging: prove it on
  supported Cargo/Nix/release/wheel outputs before consumer cutover.
- Whole-archive hashing costs I/O: hash during verification/copy, avoid duplicate reads
  within one verified operation, and never use path/mtime-only memoization for mutable
  candidate directories.
- WIP tests can go green for the wrong reason: keep actual positive executions mandatory
  and classify identity failures separately from intended mutant numerical failures.

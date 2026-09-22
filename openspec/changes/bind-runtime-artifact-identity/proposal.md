## Why

Part of [chelis#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
Runtime archives are selected by modification time, directory order, or canonical
filename existence rather than their relationship to the compiler and source under
test. The issue records a real stale-runtime wrong answer and explains the inverse:
a stale correct archive can hide a broken runtime during mutation testing.

`chelis build` stages generated code, headers, and the runtime; it does not invoke
the native compiler. Python's `compile_and_load` does link and load native code.
Both production paths, and the harnesses bypassing them, need the same identity
contract. Logging or a lexicographic tie-break alone does not fix the defect.

### Existing work and handoff

Discovery on 2026-09-22 found:

- [Draft PR #2036](https://github.com/Chelis-Lang/chelis/pull/2036), head
  `1a8dab61d6af9eb34120d6e3ca585335630470ab`, adds a proposed identity section to
  `spec/design/phase3m_rust_runtime_rewrite.md`. Its discussion records passing
  required documentation checks, not runtime acceptance. It remains a design draft.
- Local branch `agent/1354-runtime-artifact-identity`, head
  `c4ec84e82b598ac1b8df729f2a3ef43a74377cdb`, has uncommitted design/Python unit-test
  additions and untracked CLI/Python issue-1354 tests. These are regression sketches,
  not an implemented selector or evidence of passing execution. They remain untouched.
- No matching active or archived OpenSpec change was found on the proposal base,
  `d1c5cb2f`. `canonicalize-library-proof-identity` concerns checked-library proofs,
  not native runtime archives.

This proposal carries forward the useful draft decisions and resolves its open
identity-carrier, compatibility, package, and override choices. Reconcile #2036 and
reuse useful fixtures during implementation; do not silently treat either draft or
its local test sketches as completed work. This document change does not close #1354.

## What Changes

- Define a versioned runtime build identity over the runtime's actual source and
  dependency inputs, public headers, target, effective runtime features, observed compile configuration,
  and toolchain/build configuration. Bind the running CLI or Python extension to an
  independently produced expected identity; do not equate it with a compiler image ID.
- Embed identity evidence in the runtime archive at build time and verify the exact
  bytes staged or linked. Use one shared resolver for CLI, Python, and harnesses.
- **BREAKING:** reject unstamped, incompatible, ambiguous, or unverifiable selections
  before claiming a usable compiled result. `CHELIS_RUNTIME_DIR` restricts discovery;
  it does not waive validation. Existing harness exact-file pins are validated too.
- Record archive path, expected/observed identity, digest, and selection mode in
  diagnostics and persisted receipts. Native execution evidence binds the actual
  staged archive and linked product, not merely a discovered filename.
- Cover source-worktree freshness separately from installed-package compatibility,
  so an old compiler/archive pair cannot certify newly mutated source.
- Migrate Cargo, Nix, release/container, and Python-wheel production and all identified
  selector families together. Installed packages work without a source checkout.
- Add one authoritative acceptance suite with actual CLI staging/link/run, Python
  compile/load/call, positive controls, stale-correct masking mutations, and explicit
  hosted-versus-manual execution scope.

### Non-Goals

- No language, arithmetic, ownership, callable ABI, or GPU kernel semantic changes.
- No native compiler invocation added to `chelis build`.
- No runtime ABI compatibility ranges, unstamped legacy fallback, or identity bypass.
- No package signing service, general build cache, or new provenance framework.
- No implementation, test execution claim, or issue closure in this OpenSpec-only change.

## Capabilities

### New Capabilities

- `runtime-artifact-identity`: identity-bound runtime production, discovery, staging,
  linking, receipts, and source-sensitive execution evidence across native consumers.

### Modified Capabilities

None. The new cross-cutting capability adds an identity obligation without replacing
the existing `backends`, `ffi`, or `nix-package-outputs` contracts; implementation
adds cross-references to them, not a MODIFIED delta.

## Impact

The eventual implementation changes CLI staging and diagnostics, Python compiled
execution, harness selection, and distributed runtime contents. It does not alter
the meaning of generated programs. The producer/consumer inventory and acceptance
boundary are in `design.md`; every implementation task remains open in `tasks.md`.

Owning contracts are `spec/08-backends.md` §2 (generated host artifacts),
`spec/11-ffi.md` §§1.4 and 2 (compiled loading and C interop), and the existing
`openspec/specs/nix-package-outputs/spec.md` package requirements. The implementation
must first author the new public identity/failure rules in the numbered chapters,
then synchronize captured capabilities and package documentation. The runtime rewrite
and packaging design documents own implementation detail, not new language rules.

OpenSpec remains planning/review evidence under `spec/design/spec_provenance.md`:
schema validation, provider review, and executable acceptance establish different
facts. This proposal neither transfers numbered-spec authority nor activates a
repository governance policy.

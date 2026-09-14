# Phase 3m: Rust Runtime Rewrite

## Goal

Replace the growing C runtime implementation with a Rust static library while cleaning
up the compiled host-value ABI before any more runtime-heavy Phase 3 work lands.

This is not a language-semantics phase. Chelis programs keep the same observable
meaning. The change is to the compiled runtime contract, ownership model, and build
surface used by generated C and HIP host code.

## Why 3m Exists

Phase `3c` and `3d` introduced a real host-value lane: strings, lists, dicts, tuples,
`Option`, and print/debug all compile through the generated host program path. The
original C runtime was acceptable when it mostly handled tensors plus a few helpers.
It is now the wrong substrate for the next Phase 3 work:

- `3g` adds file I/O, CSV/JSON, tokenizer loading, and batching
- `3i` adds time/date and exact-decimal runtime support
- later shell work (`nautilus`, `coral`, `shoals`) depends on a stable host runtime

If those features are added to `chelis_runtime.c`, Chelis accumulates more manual memory
management, more ad hoc container code, and more C-side ownership bugs exactly where the
language is growing fastest.

`3m` is the emergency correction: move runtime implementation into Rust now, while the
host-value surface is still small enough to migrate cleanly.

## Scope

### In Scope

- new crate: `crates/chelis-runtime/`
- Rust implementation of the runtime as `libchelis_runtime.a`
- ownership of `chelis_runtime.h` moves to the runtime crate
- ABI cleanup for host values: opaque handles + explicit retain/release
- generated host C updated to use accessors and retain/release instead of field peeking
- `chelis build` updated to copy/link the Rust runtime library
- test and doc migration away from `chelis_runtime.c`

### Out Of Scope

- changing Chelis language semantics
- replacing generated tensor kernels
- changing the evaluator contract
- changing the HIP kernel ABI
- adding new language/runtime features beyond what current codegen already needs

## ABI Decision

This phase took the **ABI cleanup now** path described below. The tensor half is a
historical delivery record, not the current target contract: chelis#1286 and
[`compiled_value_ownership.md`](compiled_value_ownership.md) supersede it with one
opaque, refcounted heap model and a verified ownership boundary.

### Superseded tensor decision

`chelis_tensor` remains a source-visible `#[repr(C)]` struct with the current stable
field order. Generated numeric C and HIP code dereference tensor fields directly in hot
loops, which was why this delivery kept the layout visible. The successor cut removes
that reachability atomically: generated code uses tagged read/unique-write access and
the shared reuse proof, while the public handle becomes opaque. No compatibility ABI
retains the visible layout.

### Host values become opaque

`chelis_string`, `chelis_list`, `chelis_tuple`, and `chelis_dict` stop exposing concrete
field layouts. They become opaque runtime-managed handles declared in
`chelis_runtime.h`.

Generated host code must not read `.data`, `->len`, `->items`, or `->entries` directly.
It must call runtime accessors.

### Explicit ownership

The current compiled host path has no first-class retain/release contract for host
values. `3m` adds one:

- per-handle `retain` / `release`
- `chelis_value_retain`
- `chelis_value_release`

Generated host code becomes responsible for releasing owned temporaries and balancing
container insertion/return paths. The runtime owns internal storage; codegen owns local
handle lifetime discipline.

## Runtime Structure

Create a new workspace crate:

```text
crates/chelis-runtime/
  Cargo.toml
  include/chelis_runtime.h
  src/
    lib.rs
    tensor.rs
    blas.rs
    string.rs
    list.rs
    tuple.rs
    dict.rs
    value.rs
    print.rs
    io.rs
```

Implementation rules:

- crate type: `staticlib`
- non-tensor host values use Rust-owned storage behind opaque handles
- strings preserve current character-count semantics for `len` and slicing
- dict iteration preserves insertion order
- print/debug output remains byte-for-byte compatible with the current surface unless
  an active spec says otherwise

## Header Contract

`chelis_runtime.h` remains the C ABI contract. The historical `3m` cut made these
changes:

- `chelis_tensor` stays source-visible
- host-value structs become opaque forward declarations
- add accessor APIs for every current generated-code field peek
- add retain/release APIs for handles and `chelis_value`

The successor chelis#1286 cut replaces the first bullet with an opaque tensor declaration,
adds tensor/storage lifetime and guarded tagged-data operations, and removes
`chelis_free`, `chelis_alloc_view`, and ambiguous value-conversion aliases in one
change set.

The compiler and tests must treat the header shipped by `chelis-runtime` as the only
source of truth.

## Build Surface

### Output Artifacts

`chelis build` for C or HIP now emits:

- generated program source (`.c` / `_hip.cpp`)
- generated function header when applicable
- `chelis_runtime.h`
- `libchelis_runtime.a`
- `chelis_hip_runtime.h` for HIP builds

It no longer emits `chelis_runtime.c`.
When a generated object-mode header would otherwise export a source-level
`main`, the emitted C symbol is renamed to `<program>__main` so downstream
drivers can link their own `main(void)` without collision.

### Runtime Discovery

Runtime library discovery order is fixed and explicit:

1. `CHELIS_RUNTIME_DIR`
2. path relative to `std::env::current_exe()`
3. hard error with a clear message

Do not search arbitrary directories.

Development expectation:

- `cargo run -p chelis-cli` users set `CHELIS_RUNTIME_DIR` to the Cargo build output
  directory containing `libchelis_runtime.a`

Installed-binary expectation:

- the runtime library is found relative to the installed `chelis` executable

Required failure text shape:

- it must clearly mention `libchelis_runtime.a`
- it must mention `CHELIS_RUNTIME_DIR`
- it must tell the user to install Chelis correctly or set the environment variable

### Proposed Runtime Artifact Identity (chelis#1354)

This draft describes the compatibility choices for review. It changes no selector
and does not resolve chelis#1354. The implementation must use an explicit rule
rather than infer compatibility from file modification time.

#### What the discovery order above does not say

The order above answers *which directories to search*. It does not answer
*which archive to take when a directory holds several*, and it assumes the
exact name `libchelis_runtime.a`. Both shipped selectors added an undocumented
rule for the multiple-candidate case:

- `find_runtime_library` / `find_in_dir` in `crates/chelis-cli/src/main.rs`
- `find_runtime_library_inner` / `find_in_dir` in `crates/chelis-python/src/lib.rs`

Both collect every `libchelis_runtime*.a` other than the exact name, take the
maximum modification time, and fall back to the exact name only when no hashed
candidate exists. `newest_runtime_archive` in `crates/chelis-backend-c/src/lib.rs`
(`cfg(test)`) applies the same policy with the opposite tie-break.

Modification time is not identity. It names the archive written most recently,
not the archive built from the source the compiler was built from. Cargo keeps
hashed archives in `target/<profile>/deps`, where archives from earlier builds
can remain beside the current archive.

chelis#1354 records wrong output from a stale archive. The same selection rule
can also hide a broken runtime behind a stale correct archive. That second
case requires its own mutation test. Preferring any hashed
candidate over the exact name makes the uplifted current-configuration archive
the last choice rather than the first.

A lexicographic tie-break does not fix this. It makes the wrong winner stable.
Reporting the selected path does not fix it either. It makes the guess visible
without validating it. Both remain useful, and neither closes the defect.

#### What identity machinery already exists

Three mechanisms exist. None can be lifted into the shipped selectors as it
stands, and the reasons constrain the decision.

1. `scripts/runtime_representation_phase1.py` (`runtime_artifact`, `runtime_pin`)
   is the only authored select-by-identity rule in the repository. It takes the
   artifact path from `cargo build --message-format=json`, requires the record's
   `manifest_path`, `src_path`, crate kind, and features to match, requires
   exactly one archive inside the current target directory, then copies it to an
   exclusive directory and checks its digest after execution. Identity comes
   from Cargo, which knows the configuration it just built.

   This cannot move into `chelis build`. An installed toolchain has no Cargo,
   no workspace, and no `Cargo.toml` to compare against, and `chelis build` must
   not start a build. `spec/design/runtime_representation.md` states this
   directly: the pin "does not resolve the broader production archive discovery
   work in #1354; other consumers retain that issue's obligations."

2. `crates/chelis-image-id` derives a build identity for a *loaded object image*
   from the linker's `LC_UUID` or `NT_GNU_BUILD_ID`, with a SHA-256 fallback. It
   answers "which build is this running code from" for the compiler process and
   is deliberately content-derived rather than mtime-derived.

   It does not apply to a static archive. A `.a` is an `ar` container of object
   files with no image-level content id, and the crate excludes `object`'s
   archive support on purpose. Reusing it for chelis#1354 means extending it,
   not calling it.

3. `CHELIS_RUNTIME_LIB` names one exact archive file. It is honored only by
   `scripts/runtime_representation_phase1.py` and two `chelis-backend-hip`
   tests. No shipped selector reads it and no document defines it. It is the
   most precise existing convention and it is still an assertion by the caller,
   not a verified identity.

The runtime crate itself carries no identity. `crates/chelis-runtime` has no
`build.rs` and exports no version, ABI, or build-stamp symbol, so an archive on
disk cannot currently be asked what it was built from.

#### Decisions needed

**D1. Compatibility dimensions.** Which dimensions make an archive usable by
this compiler? Candidates: runtime source revision, target triple, profile,
and Cargo features. `ownership-ledger` is the live case — `crates/chelis-runtime/Cargo.toml`
states it is test-only and "retains the published ABI", so it may be an identity
difference that is not an incompatibility. Decide each dimension explicitly.
Do not infer any of them from file recency.

**D2. Identity carrier.** Where does an archive's identity live? Options: a
stamp symbol or section emitted by a new `chelis-runtime` `build.rs`; a sidecar
receipt written beside the archive by whatever produces it; or a digest the
compiler is built knowing. An embedded stamp travels with the archive.
A sidecar must bind the archive digest and travel with the package.
The selected carrier must support both development and installed toolchains.

**D3. Installed-package obligation.** `openspec/specs/nix-package-outputs/spec.md`
requires `packages.chelis` to contain `bin/chelis` and `lib/libchelis_runtime.a`,
and `packages.chelis-runtime` to ship the archive and headers for external C
consumers. Those archives come from separate derivations. Any rule the CLI
enforces must be satisfiable there, or the shipped toolchain stops working. If
the answer is that packaged archives must carry the D2 carrier, that is a
packaging obligation and belongs in that spec as a coordinated change. If the
answer is that a single unhashed archive in an install layout is trusted
without proof, say so and bound it.

**D4. Override semantics.** `CHELIS_RUNTIME_DIR` is the documented escape. Does
it waive the identity check or only redirect the search? chelis#1354 states that
a directory override is not proof that a unique compatible archive is present.
Decide whether an exact-file pin (`CHELIS_RUNTIME_LIB`, today undocumented)
becomes the supported way to bypass the check, and whether bypass is permitted
at all.

**D5. Enforcement.** The issue already requires a hard failure before a usable
compiled result is reported when identity cannot be resolved. The implementation
must name the candidates and record the selected identity in execution receipts.
A warning alone does not meet this requirement.

**D6. Scope.** Which consumers the rule binds. One owned selector shared by the
CLI and the Python bindings is preferred over the current duplicates. The
harness selectors in `crates/chelis-backend-c/src/lib.rs`, `crates/chelis-e2e/src/bench.rs`,
`crates/chelis-e2e/tests/eval_agreement.rs`, `crates/chelis-e2e/tests/spec_suite.rs`,
and the HIP and Metal helpers must be dispositioned explicitly. Fifty Rust files
reference `libchelis_runtime`; a converted subset is not class closure.

#### Consequences to weigh

- Requiring proof everywhere is the safest rule and the one most likely to
  break installed toolchains and external C consumers. D3 decides this.
- Rejecting on ambiguity without a carrier turns every ordinary
  `target/debug/deps` into a hard error, because several valid candidates
  coexist there by design.
- Preferring the uplifted `target/<profile>/libchelis_runtime.a` over hashed
  candidates would fix the common development case, because Cargo uplifts the
  configuration it just built. It is still an unverified guess about which
  build wrote that file, and it changes selection silently. It is not a
  substitute for D1 and D2, and it must not land as one.

#### Blocked until decided

Once D1 through D6 are answered, this section becomes the rule, the acceptance
list below gains the chelis#1354 cases, and the selectors are replaced with one
shared typed selector. Until then the mtime policy stays in place and is
documented as a known defect rather than a contract.

## Execution Plan

1. land this spec and sync active planning docs
2. add `3m` test stubs from this spec before runtime implementation
3. create `chelis-runtime` and move the runtime header into it
4. implement the Rust runtime modules
5. update host codegen to use accessors and retain/release
6. update CLI build output and runtime discovery
7. migrate compile/run tests away from `chelis_runtime.c`
8. remove `crates/chelis-backend-c/runtime/chelis_runtime.c`

## Acceptance Oracle

Authoritative oracle:

```sh
cargo test -p chelis-cli phase3m_rust_runtime_acceptance_oracle -- --nocapture
```

That oracle must prove all of the following in one named suite:

- a mixed tensor + scalar/string/list/dict program builds through `chelis build --target c`
- the emitted output contains `chelis_runtime.h` and `libchelis_runtime.a`
- the emitted output does not contain `chelis_runtime.c`
- the printed compile line links `-lchelis_runtime`
- the generated host C no longer reads host-value fields directly
- the compiled binary matches `chelis eval`

## Supporting Gates

Default repo gate:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Runtime-focused suite:

```sh
cargo test -p chelis-runtime
```

## Manual HIP Gate

This gate is not part of the default workspace run and must stay documented as manual:

```sh
CHELIS_RUNTIME_DIR=<runtime-dir> cargo test -p chelis-cli phase3m_rust_runtime_hip_manual_gate -- --ignored --nocapture
```

Expected success condition:

- `chelis build --target hip` emits `_hip.cpp`, `chelis_runtime.h`,
  `libchelis_runtime.a`, and `chelis_hip_runtime.h`
- the emitted program links and runs on the local ROCm machine
- output matches `chelis eval` for the same mixed host+tenso​r program

## Required Tests

### Runtime crate

- string length/slice use character counts, not byte counts
- string concat/trim/contains/starts_with/ends_with match the current surface
- list and dict accessors match codegen expectations
- dict preserves insertion order
- handle retain/release is safe under clone/drop patterns
- `chelis_value_retain` / `chelis_value_release` dispatch correctly by tag
- tensor ABI layout assertions remain stable

### CLI / build

- runtime discovery prefers `CHELIS_RUNTIME_DIR`
- fallback to executable-relative runtime works
- missing runtime produces the documented hard error
- `chelis build` copies `libchelis_runtime.a`, not `chelis_runtime.c`
- printed compile lines mention `-lchelis_runtime`

### Codegen

- generated host programs no longer emit `.data` on strings or `->len` on host
  collections
- generated host programs balance retain/release across branches, loops, and returns
- evaluator-vs-compiled agreement still holds for mixed programs

## Red-Team Checklist

Before calling `3m` healthy enough to unblock `3g`, red-team these concrete surfaces:

- no active docs still claim `chelis build` emits `chelis_runtime.c`
- no generated host code still peeks into non-tensor runtime struct fields
- no active tests compile `chelis_runtime.c`
- runtime discovery failures are explicit and actionable
- mixed-program compiled execution agrees with `chelis eval` on both C and HIP paths

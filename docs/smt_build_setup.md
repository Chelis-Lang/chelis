# SMT Build Setup (cvc5)

The `chelis-prove` crate uses cvc5 as its SMT solver via the `cvc5-rs` Rust
binding. This document covers build prerequisites and configuration.

## §0 Discovery Results

- **Binding:** `cvc5-rs = "0.3"` (safe high-level Rust API)
- **Underlying:** cvc5 1.3.1 (built from bundled source by `cvc5-sys`)
- **License:** BSD 3-clause (commercial-friendly, compatible with MIT workspace)
- **Decision tree outcome:** cvc5-rs binding surface adequate — proceed with
  library binding (Option A).

## Build Prerequisites

The `cvc5-sys` crate builds cvc5 from source on first compile. Required:

| Tool        | Minimum Version | Install (Debian/Ubuntu)          | Install (macOS)       |
|-------------|-----------------|----------------------------------|-----------------------|
| C++ compiler| GCC 9+ / Clang 10+ | `apt install g++`            | Xcode CLT             |
| CMake       | 3.16            | `apt install cmake`              | `brew install cmake`  |
| libclang    | —               | `apt install libclang-dev`       | Xcode CLT             |
| Git         | —               | `apt install git`                | `brew install git`    |

## Feature Gate

cvc5 is only pulled when the `smt` feature is enabled:

```toml
# In chelis-cli/Cargo.toml
[features]
smt = ["chelis-prove/smt"]

# In chelis-prove/Cargo.toml
[features]
smt = ["cvc5-rs"]

[dependencies]
cvc5-rs = { version = "0.3", optional = true }
```

Default builds (`cargo build`) do NOT pull cvc5. Only `cargo build --features smt`
triggers the cvc5 source build (~2-5 minutes on first compile, cached thereafter).

### Carcara audit gate

The optional `carcara` feature re-checks cvc5 Alethe proofs. The pinned
Carcara dependency enables only Rug's integer and rational support, so this
gate needs GMP but not MPFR or MPC. Install `libgmp-dev` on Debian/Ubuntu or
`brew install gmp` on macOS.

Run the complete suite serially. A nightly parallel process exited with
SIGSEGV after tests, while the same unit, integration, and doctest set passed
with one test thread. Serialization contains that nondeterministic failure
without narrowing the corpus; it does not establish the upstream root cause:

```bash
cargo test -p chelis-prove --features carcara -- --test-threads=1
```

Inside the Devenv shell no extra setup is needed: `devenv/toolchains.nix`
provides GMP with the `CPATH`/`LIBRARY_PATH` wiring, and `CVC5_DIR` can
point at the flake's pinned tree
(`nix build .#legacyPackages.<system>.cvc5-dir`).

Outside Devenv with Homebrew, GMP is keg-only, so its headers and library
may not be on Clang's default paths. `gmp-mpfr-sys` invokes the compiler
directly for its system-library probe; use `CPATH` and `LIBRARY_PATH` so
that probe sees the Homebrew installation, and point `CVC5_DIR` at any
previously built cvc5 tree to skip the source build:

```bash
GMP_PREFIX="$(brew --prefix gmp)"
CPATH="${GMP_PREFIX}/include${CPATH:+:${CPATH}}" \
LIBRARY_PATH="${GMP_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
cargo test -p chelis-prove --features carcara -- --test-threads=1
```

## CI Configuration

The required `smt-build` job in `.github/workflows/ci.yml` keeps the branch
protection context name `SMT Feature Build (Linux)`, but it is now the fast
SMT smoke lane. It builds on the Devenv toolchain and activates the `smt`
profile. That profile sets `CVC5_DIR` from `outputs.cvc5-dir`. The lane runs
`cargo build -p chelis-cli --features smt` inside that profile and verifies
the built binary discharges a real obligation through cvc5 with
`.github/scripts/verify_release_smt.py`, and runs a narrow cvc5 engine smoke
(`cargo test -p chelis-prove --features smt --lib cvc5_engine_`). It is a
non-gate job (rule-id GATE-SCOPE-SMT in `scripts/test_gate.py`): out of
`scripts/gate.py` scope by design, like the sanitizer job, because the smt
feature is not a per-PR developer-loop prerequisite.

The full solver/proof corpus moved to `.github/workflows/smt-full-prove.yml`.
That workflow runs on a nightly schedule and on manual dispatch only -- NOT on
PRs. The ~46m corpus is too heavy for the per-PR path, and `SMT Full Prove
(Linux)` is not a required status check, so dropping the PR trigger cannot
deadlock branch protection; the per-PR cvc5 signal is ci.yml's fast
`SMT Feature Build (Linux)` smoke, and prove-stack regressions are caught by
the nightly run (which opens/closes a tracking issue) or on demand via
`workflow_dispatch`. It carries the expensive suites that used to sit in
required CI: `--features smt`, `carcara`, `z3`, the cvc5+Z3 cross-engine
oracle, `clarabel`, the production `smt clarabel` config, Gappa
`--check-only`, the Arb certifier, and `--features arb`.

The full workflow restores the same `shared-key: smt-smt-build` cargo cache as
the fast smoke lane. The key intentionally matches the old required
`smt-build` job cache namespace (`key: smt` plus job id `smt-build`) so the
split can reuse the existing cvc5 build cache while making that cache stable
across the smoke and full-prove jobs. The split removes the full proof corpus
from the required context; it must not make the optional lane cold-build cvc5
before reaching its proof steps.

### The pinned cvc5 supply (openspec `converge-smt-lanes-on-nix-cvc5` + `converge-full-prove-on-devenv`)

Building cvc5 from source is ~22 minutes of CMake/make. Every SMT lane
(`smt-build`, the manual `smt-build-darwin-arm64`, and `full-smt-prove`)
uses the same supply path. Each lane builds Devenv `outputs.cvc5-dir`, which
comes from `nix/cvc5.nix`. The `smt` profile sets that output as `CVC5_DIR`.
`cvc5-sys` sees `build/src/libcvc5.a` and links it
instead of running CMake/make. The closure is cached in the GitHub Actions
cache through the shared `.github/actions/cvc5-cache-restore` and
`.github/actions/cvc5-cache-save` composite pair (one key derivation for
every lane, including `nix-packages.yml`, `release.yml`, and the drift
canary), so all jobs share one cached closure. A cvc5 or nixpkgs pin bump changes the derivation
key and pays one ~30m Nix build; there is no separate publish step and no
runbook. Every lane builds inside `devenv --profile smt shell`, so rustc, cc, and
libclang match the repository pin. The full-prove solver stack uses OpenBLAS,
Z3, Gappa, GMP, m4, and make from the Devenv shell (`devenv/toolchains.nix`).
The Clarabel Linux feature selects `openblas-src/system`. Cargo links the
Devenv package instead of building OpenBLAS source.

The former harvested-prebuilt machinery retired in two steps: the durable
Release-asset producer (`build-cvc5.yml`) and its publish script left with
the smoke-lane conversion, and the Actions-cache harvest cycle
(`scripts/ci_cvc5_cache.py`) left when the full-prove lane converged. No CI
lane compiles cvc5 through `cvc5-sys` any more; the cargo-from-source recipe
in "Build Prerequisites" above remains the documented path for contributors
outside Devenv, exercised locally rather than in CI.

### Cache-pool pruning

The Actions-cache pool contains Cargo target snapshots and caches for pull requests.
The weekly and manual `cache-prune.yml` workflow uses the reviewed shared prune action.
Chelis owns the schedule, apply mode, retention count, and protected prefixes.

The workflow keeps one primary generation for each prefix.
It protects every `cvc5-prebuilt-*` cache.
Manual dispatch uses dry-run mode unless the caller selects `apply`.
The shared action fails closed if it cannot read a complete cache snapshot.

One companion CI lane, the manual `smt-build-darwin-arm64`
(`macos-latest`) job, builds `chelis-cli --features smt` on Darwin with
the Devenv toolchain and the flake's pinned cvc5 tree, and runs the
post-build verifier (chelis#422). The former `smt-build-glibc231` lane
(a `debian:11` container) retired with the Cargo Linux release jobs
when the Linux toolchain moved to the Nix release output (openspec
`switch-linux-release-to-nix`); the Nix package CI now proves the cvc5
build on both systems.

## Release builds (chelis#422)

A manual dispatch starts `release.yml`. A dispatch at a `v*` tag publishes the release assets.

`release.yml` builds two release artifacts. The Linux toolchain
(`chelis-v<ver>-linux-x86_64.tar.gz`) comes from the Nix release
output: `devenv build --no-tui outputs.release-chelis` builds the
compiler with the `smt` feature from the committed `Cargo.nix` graph
and the pinned cvc5 1.3.1 tree (`nix/cvc5.nix`), stages the tarball,
and runs the cvc5 discharge verifier inside the derivation. The
recorded glibc floor of the portable binary lives in
`nix/contracts.nix` (`linuxReleaseGlibcFloor`). A separate
`consume-chelis-release` job then unpacks the exact staged tarball in a
digest-pinned Ubuntu container without Nix, re-runs the cvc5 discharge
verifier in `--tarball` mode, and compiles emitted C against the
shipped runtime archive with the system `gcc` and OpenBLAS.

The Darwin artifact (darwin-arm64) ships from the same Devenv release
output: the derivation builds the compiler with the `smt` feature from
the committed graph and the pinned cvc5 tree, statically links
`libzstd`, rewrites the `libiconv` load path to `/usr/lib`, asserts
every load command is an Apple system path, and runs the behavior
probes (including the cvc5 discharge) on the REWRITTEN binary -- the
Darwin sandbox can execute it, so Darwin gets the strongest
in-derivation proof. A `consume-chelis-release-darwin` job then unpacks
the exact staged tarball on a stock macOS runner (no Nix), re-runs the
cvc5 discharge verifier in `--tarball` mode, and compiles emitted C
against the shipped runtime archive with the system clang and
Accelerate.

WI-11 (license): the cvc5 build forces GMP and disables the GPL CLN
path (`nix/cvc5.nix` passes `ENABLE_GPL=OFF` and `USE_CLN=OFF`; only
`libgmp.a` is linked, never `libcln.a`), so the shipped artifact is
distributable.

Both release legs build cvc5 from the pinned source tree inside the Nix
sandbox with content-addressed inputs, so there is no release-time
network fetch to retry and no trust in a harvested asset. The former
`scripts/ci_cvc5_build.py` fetch-retry wrapper (chelis#1004) retired
with the last host-Cargo release job. The standing cold
cargo-from-source cvc5 proof is `smt-full-prove.yml`'s cold-cache path.

`publish-release` depends on every build job and on both off-Nix
consumption jobs. One failure stops publication for the manually
dispatched tag.

## Downstream Impact

Shell repos consuming the chelis workspace (Shoals, Coral, Nautilus, Hull) do
NOT transitively pull cvc5 unless they explicitly enable the `smt` feature.
Default workspace builds are unaffected.

## Prebuilt cvc5 (Alternative)

To skip building from source, set `CVC5_LIB_DIR` to a directory containing
prebuilt `libcvc5.a`:

```bash
CVC5_LIB_DIR=/path/to/cvc5/lib cargo build --features smt
```

Headers are expected at `$CVC5_LIB_DIR/../include` by default. Override with
`CVC5_INCLUDE_DIR` if needed.

## Solver Capabilities

cvc5 supports the following logics relevant to property verification:

- `QF_NRA` — quantifier-free nonlinear real arithmetic
- `QF_LRA` — quantifier-free linear real arithmetic
- `NRA` — nonlinear real arithmetic with quantifiers
- `ALL` — all theories (fallback)

Transcendental support (exp, log) is available via cvc5's nonlinear extension
(enabled by default in NRA logics). Performance varies by formula shape.

## Timeout Configuration

Per-query timeout is set via:
```rust
solver.set_option("tlimit-per", "5000"); // 5 seconds per check-sat call
```

CLI flag: `chelis prove --smt-timeout 5000`

On smt-only prove paths, cvc5 timeout/unknown is reported as
`unsupported` with a reason, never `failed` without a counterexample. In
auto mode the runner may fall back to Tier C fuzz validation.

## Reverting to Subprocess (Fallback)

If the library binding proves problematic in future, the `Solver` trait in
`chelis-prove/src/solver.rs` allows swapping to an `easy-smt` subprocess
implementation without architectural change. The trait surface is:
`assert`, `check_sat`, `get_model`, `push`, `pop`, `set_timeout`.

## Known Build Issues

### Static libc not available (link failure on cvc5 binary)

The cvc5-sys build script builds both `libcvc5.a` (the library we need) and
the `cvc5` binary. On systems without static libc/libstdc++, the binary link
fails and the build script panics. Workaround: set `CVC5_LIB_DIR` to point
at the already-built library:

```bash
CVC5_LIB_DIR=target/debug/build/cvc5-sys-*/out/cvc5/build/src \
CVC5_INCLUDE_DIR=target/debug/build/cvc5-sys-*/out/cvc5/include \
LIBCLANG_PATH=/usr/lib/llvm-*/lib \
cargo build --features smt
```

### Missing cvc5_export.h

The generated `cvc5_export.h` lives in `build/include/` but the source headers
reference it from `include/`. Copy it:

```bash
cp target/debug/build/cvc5-sys-*/out/cvc5/build/include/cvc5/cvc5_export.h \
   target/debug/build/cvc5-sys-*/out/cvc5/include/cvc5/
```

### bindgen can't find stddef.h

Set `BINDGEN_EXTRA_CLANG_ARGS` to include system headers:

```bash
BINDGEN_EXTRA_CLANG_ARGS="-I/usr/include" cargo build --features smt
```

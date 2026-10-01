# SMT Build Setup (cvc5)

The `chelis-prove` crate uses cvc5 as its SMT solver via the `cvc5-rs` Rust
binding. This document covers build prerequisites and configuration.

## Solver Binding

- **Binding:** `cvc5-rs = "0.3"` (safe high-level Rust API)
- **Underlying:** cvc5 1.3.1 (built from bundled source by `cvc5-sys`)
- **License:** BSD 3-clause (commercial-friendly, compatible with MIT workspace)

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
smt = ["chelis-prove/smt", "chelis-tide/smt"]

# In chelis-prove/Cargo.toml
[features]
smt = ["cvc5-rs", "num-rational"]

[dependencies]
cvc5-rs = { version = "0.3", optional = true }
```

Default builds (`cargo build`) do NOT pull cvc5. Only `cargo build --features smt`
triggers the cvc5 source build (~2-5 minutes on first compile, cached thereafter).

### Carcara audit gate

The optional `carcara` feature re-checks cvc5 Alethe proofs. The pinned
Carcara dependency enables only Rug's integer and rational support, so this
gate needs GMP but not MPFR or MPC. Cargo builds the GMP version locked by
`gmp-mpfr-sys`; do not install or configure a distribution GMP for this gate.
The source build needs a C toolchain, `m4`, and `make`.

Run the complete suite serially. A nightly parallel process exited with
SIGSEGV after tests, while the same unit, integration, and doctest set passed
with one test thread. Serialization contains that nondeterministic failure
without narrowing the corpus; it does not establish the upstream root cause:

```bash
cargo test -p chelis-prove --features carcara -- --test-threads=1
```

`CVC5_DIR` reuses the durable local cvc5 artifacts when that cache has already
been populated:

```bash
CVC5_DIR="${HOME}/.cache/chelis-cvc5/darwin-arm64" \
cargo test -p chelis-prove --features carcara -- --test-threads=1
```

## CI Configuration

The required `smt-build` job in `.github/workflows/ci.yml` keeps the branch
protection context name `SMT Feature Build (Linux)` and is the fast SMT smoke
lane. It installs the cvc5 build prerequisites, runs
`cargo build -p chelis-cli --features smt`, verifies the built binary discharges
a real obligation through cvc5 with `.github/scripts/verify_release_smt.py`, and
runs a narrow cvc5 engine smoke (`cargo test -p chelis-prove --features smt
--lib cvc5_engine_`) and the integration worker controls (`cargo test -p
chelis-prove --features smt --test integration_solver_isolation`). It is a
non-gate job (rule-id GATE-SCOPE-SMT in
`scripts/test_gate.py`): out of `scripts/gate.py` scope by design, like the
sanitizer job, because cvc5 builds from source and is not a per-PR
developer-loop prerequisite.

The full solver/proof corpus lives in `.github/workflows/smt-full-prove.yml`.
That workflow runs on a nightly schedule and on manual dispatch only -- NOT on
PRs. The ~46m corpus is too heavy for the per-PR path, and `SMT Full Prove
(Linux)` is not a required status check, so the absence of a PR trigger cannot
deadlock branch protection; the per-PR cvc5 signal is ci.yml's fast
`SMT Feature Build (Linux)` smoke, and prove-stack regressions are caught by
the nightly run (which opens/closes a tracking issue) or on demand via
`workflow_dispatch`. It carries the expensive suites: `--features smt`, `carcara`, `z3`, the cvc5+Z3 cross-engine
oracle, `clarabel`, the production `smt clarabel` config, Gappa
`--check-only`, the Arb certifier, and `--features arb`.

The full workflow restores the same `shared-key: smt-smt-build` cargo cache as
the fast smoke lane. The key matches the required `smt-build` job's cache
namespace (`key: smt` plus job id `smt-build`), so the optional lane reuses the
cvc5 build cache and does not cold-build cvc5 before reaching its proof
steps.

The required `smt-build` job realizes the `ci-smt` Devenv profile only on the
self-hosted runner, which substitutes it from the private Nix cache:
`nix/ci-cvc5.nix` builds the locked CVC5 1.3.1, GMP, CaDiCaL and LibPoly
archive/header tree, and Cargo consumes its `CVC5_DIR` without rebuilding it.
The job's Devenv-prefixed Cargo namespace keeps those outputs apart from the
native full-prove lane. GitHub-hosted runs of the same job keep main's toolchain and link the
prebuilt cvc5 stores described next.

### Durable prebuilt cvc5

Building cvc5 from source is ~22 minutes of CMake/make. The SMT lanes must
never pay that on the per-PR path, so every SMT lane (the required `smt-build`,
the Linux `smt-build-glibc231` and nightly `smt-build-darwin-arm64` lanes, and
`smt-full-prove.yml`) LINKS a prebuilt cvc5 instead of rebuilding it. The
prebuilt tree is held in TWO stores, tried in order, driven by
`scripts/ci_cvc5_cache.py`:

1. **Durable Release asset (primary).** `.github/workflows/build-cvc5.yml`
   builds cvc5 from source once per (cvc5-sys version, namespace) and publishes
   `<store-key>.tar.gz` + a `.sha256` sidecar to a
   `cvc5-prebuilt-cvc5sys<version>` **prerelease** tag (deliberately not `v*`,
   so it never triggers `release.yml`). Each SMT lane's **`fetch`** step
   downloads the asset, verifies the sha256 BEFORE extraction, rejects unsafe
   tar members, and re-checks every required path. A Release asset has **no
   10GB Actions-cache LRU budget, no 7-day idle TTL, and no branch scope**, so
   it survives the three conditions that would otherwise force a cold rebuild
   onto the per-PR path: a rustc-stable bump, a quiet/docs-only stretch, and cache-pool
   pressure.
2. **Actions cache (fallback).** The stable-key `actions/cache` restore runs
   only when `fetch` reported `warm=false` (`if: steps.cvc5fetch.outputs.warm
   != 'true'`). It covers the window between a cvc5-sys bump and
   `build-cvc5.yml` republishing, and any run where the Release lookup fails.
   The dir lives OUTSIDE `target/` (`~/.cache/chelis-cvc5/<namespace>`), so it
   never contends with rust-cache's `target/` domain.

The store key — `cvc5-prebuilt-<namespace>-cvc5sys<version>-<schema>` — depends
on the pinned `cvc5-sys` crate version (which moves with the bundled cvc5
release), the os/arch namespace (`linux-x86_64`, `linux-glibc231`,
`darwin-arm64`), and the cache schema, but **NOT** on `Cargo.lock` and **NOT**
on the rustc version. The harvested payload is 100% cvc5 C++/CMake output
(`bindings.rs` is regenerated per build in `OUT_DIR` and is not harvested), so
rustc is provably irrelevant to the cached bytes. A rustc component in the key
would only force needless cold rebuilds on every ~6-week stable bump.

After a genuine from-source build (the fallback/bootstrap path), **`harvest`**
copies the cvc5 link/bindgen inputs into the store dir and writes a
completeness sentinel; **`activate`** exports `CVC5_DIR` only when the store is
present AND sentinel-complete, so `cvc5-sys`'s build script sees
`build/src/libcvc5.a` and LINKS it instead of running CMake/make (the same
prebuilt-link path the `z3` feature uses; `docs/local_z3_environment.md`). This
is an optimization, never a correctness risk: a missing asset, a network error,
a sha256 mismatch, an unsafe/incomplete archive, or a partial cache all leave
no sentinel, so the build falls back to from-source (worst case is "no
speedup", never a broken or mislinked build). cvc5-sys's own
`check_cvc5_version` (reads the harvested `cmake/version-base.cmake`)
additionally hard-fails a wrong-version link. `scripts/test_ci_cvc5_cache.py`
covers the key, asset naming, harvest/pack/fetch round-trip, and activate logic
that cannot be exercised in CI without a real ~22m cvc5 build.

Bump `CACHE_SCHEMA` in `scripts/ci_cvc5_cache.py` if the harvested artifact set
ever changes shape (it rotates the store key AND the asset filename, and the
`scripts/ci_cvc5_cache.py` path trigger republishes `build-cvc5.yml`).

#### cvc5-sys version-bump runbook

A cvc5-sys bump is the ONLY event that legitimately puts a cold from-source
cvc5 build back on the per-PR path, because the new version's Release asset
does not exist until `build-cvc5.yml` republishes it (Linux on relevant main
pushes; Darwin nightly or on manual dispatch). To keep the bump PR itself warm, publish the new
assets first:

1. On the bump branch, run `build-cvc5.yml` via **workflow_dispatch** (it must
   already exist on `main`). `plan` finds the new-version assets missing
   and builds + publishes all three namespaces.
2. Re-run the bump PR's CI; each SMT lane's `fetch` now links the freshly
   published asset and stays warm.

If step 1 is skipped, the bump PR's `smt-build` builds cvc5 cold and will
exceed its 25-minute timeout — a loud, deliberate failure that points here,
not a silent per-PR tax.

### Cache-pool pruning

The Actions-cache pool is dominated by `Swatinem/rust-cache` `target/`
snapshots — one per (job, `Cargo.lock`/rustc generation) — plus per-PR caches.
Left alone it creeps over GitHub's 10GB per-repo limit and LRU-evicts whatever
is least-recently-used. `.github/workflows/cache-prune.yml` (weekly + manual)
runs `scripts/ci_cache_prune.py`, which deletes closed-PR-ref caches and stale
duplicate `main` generations while PROTECTING the `cvc5-prebuilt-*` fallback
caches. A failed open-PR lookup fail-safes to no PR pruning (never mass-delete
on error); manual dispatch is dry-run unless `apply` is set.
`scripts/test_ci_cache_prune.py` covers the deletion policy.

Two companion prove-in-CI lanes, `smt-build-glibc231` (a digest-pinned
Python 3.11 Bullseye container) and `smt-build-darwin-arm64` (`macos-latest`), build
`chelis-cli --features smt` on the other two release targets and run
the post-build verifier. They prove cvc5 builds on those toolchains
before `release.yml` ships the feature there. The
Bullseye lane uses the same pinned container digest in `ci.yml`,
`build-cvc5.yml`, and `release.yml`; the image supplies Python, git, curl,
and CA certificates before checkout. After checkout, `ci_apt_get.py
--debian-bullseye-snapshot` replaces every moving apt source with the
immutable `20260901T000000Z` Debian and Debian Security snapshots before
installing build dependencies. Python 3.11 also lets cvc5 use `tomllib`
without a separate moving PyPI bootstrap.

### Darwin validation cadence

`macos-nightly.yml` owns `smt-build-darwin-arm64`: daily at 04:17 UTC and
manual dispatch on a chosen branch, with a 60-minute timeout for cold cvc5 builds.
Ordinary PRs and main pushes run no Darwin builds. The prebuilt producer's daily
01:17 UTC run fills missing Darwin assets before validation; manual dispatch can
force a rebuild. Relevant main pushes can produce Linux assets only. Release
validation still builds and checks its own Darwin shipping artifact in `release.yml`.

## Release builds

`release.yml` builds all three release artifacts (linux-x86_64,
linux-x86_64-glibc2.31, darwin-arm64) with `cargo build --release -p
chelis-cli --features smt,sealed-runtime`, so the shipped `chelis` binary
discharges property obligations through cvc5 instead of degrading to the
solver-free fuzz path, and carries a sealed runtime whose export the tarball
ships (`spec/08-backends.md` §2.1). Each release job:

- installs the cvc5 build prerequisites for its platform (the glibc 2.31 job
  uses the pinned Python 3.11 Bullseye container and immutable Debian snapshot
  described above, with no separate PyPI bootstrap; macOS relies on the
  image's CMake/Python/Xcode CLT plus an idempotent `brew install cmake`), and
- runs `.github/scripts/verify_release_smt.py` against the freshly
  built binary, which proves a known producer obligation discharges
  via cvc5 (`proof_tier=smt`, `discharge_tier.engine=cvc5`). A binary
  accidentally built without `--features smt` fails this step, and
- after staging the `.tar.gz`, runs the verifier once more in
  `--tarball` mode against the EXTRACTED (stripped) `bin/chelis` inside
  the packaged artifact. This is the most faithful guard: it checks the
  exact binary users download, not a pre-staging proxy, and would also
  catch a staging step that packaged the wrong binary.

License: the `cvc5-sys` build forces GMP and disables the GPL
CLN path (the cvc5 CMake cache records `ENABLE_GPL=OFF` and
`USE_CLN=OFF`; only `libgmp.a` is linked, never `libcln.a`), so the
shipped artifact is distributable.

`release.yml` deliberately does NOT link the durable prebuilt asset: each
release job builds cvc5 cold from source, so the shipped binary is an
INDEPENDENT, license-safe proof of the exact recipe rather than trusting an
asset produced by `build-cvc5.yml`. A `build-cvc5.yml` bug therefore can never
silently reach a shipped artifact. Treat a `release.yml` cvc5 failure as real
even when the per-PR SMT lanes are green off the asset.

### The cold build's fetch is retried; its compile is not

Building cvc5 cold begins by pulling the cvc5 source and its dependencies
over the network, so the release jobs are the only lanes exposed to a
transient GitHub refusal there. Because `publish-release` has `needs:` on all
three build jobs, one such failure skips the publish and leaves a pushed tag
with no GitHub Release. Such failures occur at `cvc5-sys` `build.rs` dependency
downloads (HTTP 403) and at the source clone, with no change to the tree.

Each `Build chelis-cli (release, smt)` step therefore runs through
`scripts/ci_cvc5_build.py`, which retries **only** when the failing attempt's
output carried a transient-fetch signature. This does not weaken anything
above: no prebuilt is linked, no compile is skipped, and a compile, CMake
configure, or link failure is **not** retried at all -- it fails on the first
attempt. "Treat a `release.yml` cvc5 failure as real" still holds, because the only failures the wrapper absorbs are ones that never
reached the compiler. A sustained outage still fails the job once the bounded
attempts are exhausted.

The signature list is deliberately narrow, and
`scripts/test_ci_cvc5_build.py` pins both directions: the observed 403 and
clone-failure lines classify as transient, while an `error[E0308]`, a
`CMake Error`, a linker failure, and a bare `build.rs` panic with no fetch
diagnostic above it do not. Widening that list is a decision to retry
something new; make it deliberately.

## Downstream Impact

Shell repos consuming the chelis workspace (Shoals, Coral, Nautilus) do
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

## Solver Trait

The `Solver` trait in `chelis-prove/src/solver.rs` isolates the binding: an
`easy-smt` subprocess implementation could replace the library binding without
architectural change. The trait surface is: `assert`, `check_sat`, `get_model`,
`push`, `pop`, `set_timeout`.

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

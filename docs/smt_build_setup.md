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

## CI Configuration

The required `smt-build` job in `.github/workflows/ci.yml` keeps the branch
protection context name `SMT Feature Build (Linux)`, but it is now the fast
SMT smoke lane. It installs the cvc5 build prerequisites, runs
`cargo build -p chelis-cli --features smt`, verifies the built binary discharges
a real obligation through cvc5 with `.github/scripts/verify_release_smt.py`, and
runs a narrow cvc5 engine smoke (`cargo test -p chelis-prove --features smt
--lib cvc5_engine_`). It is a non-gate job (rule-id GATE-SCOPE-SMT in
`scripts/test_gate.py`): out of `scripts/gate.py` scope by design, like the
sanitizer job, because cvc5 builds from source and is not a per-PR
developer-loop prerequisite.

The full solver/proof corpus moved to `.github/workflows/smt-full-prove.yml`.
That workflow runs on relevant prove/solver/proof/CI PR paths, on a nightly
schedule, and on manual dispatch. It carries the expensive suites that used to
sit in required CI: `--features smt`, `carcara`, `z3`, the cvc5+Z3
cross-engine oracle, `clarabel`, the production `smt clarabel` config, Gappa
`--check-only`, the Arb certifier, and `--features arb`. The PR path trigger
includes `chelis-prove`, its proof-facing local dependencies
(`chelis-surf`, `chelis-deep`, `chelis-pred`, `chelis-types`, and
`chelis-compiler-api`), proof scripts/data, and the CI workflow/action files
that define the lane.

The full workflow restores the same `shared-key: smt-smt-build` cargo cache as
the fast smoke lane. The key intentionally matches the old required
`smt-build` job cache namespace (`key: smt` plus job id `smt-build`) so the
split can reuse the existing cvc5 build cache while making that cache stable
across the smoke and full-prove jobs. The split removes the full proof corpus
from the required context; it must not make the optional lane cold-build cvc5
before reaching its proof steps.

### Stable-key cvc5 cache (chelis#583)

`Swatinem/rust-cache` keys its `target/` cache on `Cargo.lock`, so a PR that
bumps the lockfile for an UNRELATED reason blows the cache and forces a fresh
~30-minute from-source cvc5 build. To decouple cvc5 from lockfile churn, every
SMT lane (the required `smt-build`, the `smt-build-glibc231` and
`smt-build-darwin-arm64` prove-in-CI lanes, and `smt-full-prove.yml`) layers a
dedicated cvc5 cache, driven by `scripts/ci_cvc5_cache.py`, in front of the
rust-cache restore:

1. **`key`** computes a STABLE cache key — `cvc5-prebuilt-<namespace>-cvc5sys
   <version>-rustc<version>-<schema>` — that depends on the pinned `cvc5-sys`
   crate version (which moves with the bundled cvc5 release), the os/arch
   namespace (`linux-x86_64`, `linux-glibc231`, `darwin-arm64`), and the rustc
   version, but NOT on `Cargo.lock`.
2. **`actions/cache/restore`** restores the prebuilt artifacts to a dir OUTSIDE
   `target/` (`~/.cache/chelis-cvc5/<namespace>`), so the cvc5 cache never
   contends with rust-cache's `target/` domain.
3. **`activate`** exports `CVC5_DIR` only when the restored cache is complete.
   `cvc5-sys`'s build script then sees `build/src/libcvc5.a` already present and
   LINKS the prebuilt static lib instead of running CMake/make — the same
   prebuilt-link path the `z3` feature uses (`docs/local_z3_environment.md`).
4. After a cold (from-source) build, **`harvest`** copies the cvc5 link/bindgen
   inputs into the cache dir and writes a completeness sentinel; a dedicated
   **`actions/cache/save`** persists them only when the harvest is complete.

This is an optimization, never a correctness risk: `activate` links the cache
only when it is present AND sentinel-complete, so a cold, partial, or
incomplete cache simply falls back to the normal from-source build (worst case
is "no speedup", never a broken build). The lanes also set
`cache-on-failure: true` on the rust-cache step so a slow cold run still warms
the workspace cache. Bump `CACHE_SCHEMA` in `scripts/ci_cvc5_cache.py` if the
harvested artifact set ever changes shape (it invalidates every namespace's
cvc5 cache). `scripts/test_ci_cvc5_cache.py` covers the key, harvest, and
activate logic that cannot be exercised in CI without a real ~30m cvc5 build.

Two companion prove-in-CI lanes, `smt-build-glibc231` (a `debian:11`
container) and `smt-build-darwin-arm64` (`macos-latest`), build
`chelis-cli --features smt` on the other two release targets and run
the post-build verifier. They prove cvc5 builds on those toolchains
before `release.yml` ships the feature there (chelis#422). The
`debian:11` lane additionally installs `python3-pip` and `pip install
tomli`, because cvc5's build-time TOML codegen imports `tomli` on
Python < 3.11 and `python3-tomli` is not in the main bullseye suite.

## Release builds (chelis#422)

`release.yml` builds all three release artifacts (linux-x86_64,
linux-x86_64-glibc2.31, darwin-arm64) with `cargo build --release -p
chelis-cli --features smt`, so the shipped `chelis` binary discharges
property obligations through cvc5 instead of degrading to the
solver-free fuzz path. Each release job:

- installs the cvc5 build prerequisites for its platform (the
  `debian:11` job adds `tomli` as above; macOS relies on the image's
  CMake/Python/Xcode CLT plus an idempotent `brew install cmake`), and
- runs `.github/scripts/verify_release_smt.py` against the freshly
  built binary, which proves a known producer obligation discharges
  via cvc5 (`proof_tier=smt`, `discharge_tier.engine=cvc5`). A binary
  accidentally built without `--features smt` fails this step (the
  feature-inert regression that motivated chelis#422), and
- after staging the `.tar.gz`, runs the verifier once more in
  `--tarball` mode against the EXTRACTED (stripped) `bin/chelis` inside
  the packaged artifact. This is the most faithful guard: it checks the
  exact binary users download, not a pre-staging proxy, and would also
  catch a staging step that packaged the wrong binary.

WI-11 (license): the `cvc5-sys` build forces GMP and disables the GPL
CLN path (the cvc5 CMake cache records `ENABLE_GPL=OFF` and
`USE_CLN=OFF`; only `libgmp.a` is linked, never `libcln.a`), so the
shipped artifact is distributable.

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

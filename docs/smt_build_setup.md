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

CI images need the build prerequisites above. Add to the CI Dockerfile or
workflow:

```yaml
- name: Install cvc5 build deps
  run: apt-get install -y cmake g++ libclang-dev git
```

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

# Install

## Rust Toolchain

Chelis is built with stable Rust:

```sh
rustup default stable
rustup component add rustfmt clippy
```

## C Toolchain

The C backend expects a native compiler and a BLAS provider for matmul fast paths.

Fedora / RHEL:

```sh
sudo dnf install gcc openblas-devel valgrind
```

Ubuntu / Debian:

```sh
sudo apt-get install gcc libopenblas-dev valgrind
```

macOS:

```sh
xcode-select --install
```

Supported macOS paths:

- Default: Apple clang + Accelerate. This is the supported correctness path on Apple Silicon and does not require Homebrew OpenBLAS.
- Optional: Homebrew GCC for OpenMP-enabled CPU loops.

```sh
brew install gcc
```

## Build and Test the Repo

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Build the Book

```sh
mdbook build docs/book
```

## Validate Docs Examples

```sh
cargo test -p chelis-e2e --test skill_suite
```

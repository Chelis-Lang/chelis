# Phase 3j-pre — Compiler Release Infrastructure

Status: shipped through `v0.1.10`; Linux x86_64 and macOS darwin-arm64 release packaging are live.

This document describes how the Chelis compiler binary is released and how
downstream shells (Nautilus, Coral, and any future domain packages) should
pin a toolchain version.

As of April 19, 2026, `v0.1.10` is the current Chelis compiler release and the
exact compiler pin used by the first downstream shell release,
`Nautilus v0.1.0`.

## Scope of the `v0.1.x` line

The `v0.1.x` line is the shipped Phase 3j-pre compiler surface:

- `chelis` CLI (`chelis-cli`) with `fmt`, `check`, `eval`, `build`, `deep`,
  `surf`, and `tide` subcommands as documented in the Phase 0–3j-pre specs.
- Deep AST + Surf surface through Phase 3j-pre Batch 5b.
- Expanded `Std.Nn` surface: RMSNorm, GELU, SiLU, Linear, Conv1d/2d wrappers,
  and `Std.Nn.Attention` (SDPA / grouped-query attention importable surface).
- Expanded `Std.Loss` and `Std.Init` surface (Kaiming init, etc.).
- Reductions: `sum`, `mean`, `max`, `min`, `prod`, `argmax`, `argmin` in IR
  and C backend.
- C backend host-lane pure-tensor wrapper lowering (Batch 5b fix).

Known acknowledged limitations are tracked in
`spec/design/chelis_phase3_plan.md` §3j-pre Acceptance Oracle (Batch 5
limitations section).

## Release Mechanism

Release is driven by `.github/workflows/release.yml`:

- **Trigger:** pushing a tag matching `v*` to `origin/main`.
- **Platforms:** `ubuntu-latest` for `linux-x86_64` and `macos-latest` for `darwin-arm64`.
- **Build:** `cargo build --release -p chelis-cli`.
- **Packaging:** each release stages the stripped `chelis` binary, `libchelis_runtime.a`,
  `chelis_runtime.h`, `chelis_blas.h`, `README.md`, and `LICENSE` into a platform tarball.
- **Artifacts:** releases attach
  `chelis-<version>-linux-x86_64.tar.gz` and
  `chelis-<version>-darwin-arm64.tar.gz`, each with a sibling `.sha256` file.
- **Publish:** build jobs upload artifacts and a final `softprops/action-gh-release@v2`
  step attaches them to the tag.

The workflow also exposes a `workflow_dispatch` trigger so the build path
can be exercised on a branch without publishing. Dispatch runs upload the
tarball as a workflow artifact and do NOT create a Release.

This is a hand-rolled workflow. Chelis does not use `cargo-dist` in
3j-pre; if multi-platform matrix support becomes a priority in a later
phase, that decision can be revisited. No `[workspace.metadata.dist]`
section is needed in the root `Cargo.toml`.

## Platform Scope

Historical note: the `v0.1.x` line shipped as **Linux x86_64 only**.

The platform-expansion release line supports:

- `linux-x86_64`
- `darwin-arm64` (Apple Silicon)

Support contract:

- macOS support is CPU-first only.
- The default macOS path is Apple clang + Accelerate.
- Homebrew GCC is optional for OpenMP-enabled CPU loops.
- Intel macOS and Linux aarch64 remain out of scope for this release line.

## Pinning the Toolchain in `reef.toml`

Downstream reef packages (Nautilus, Coral, and user-facing apps) should
pin an exact compiler version in their `reef.toml`:

```toml
[package]
compiler = "=0.1.10"
```

Meaning:

- `=0.1.10` is an exact match. Until Chelis stabilizes its surface,
  downstream shells should prefer exact pins over caret or tilde ranges.
- `reef` resolves the pin by downloading the matching platform tarball from the
  GitHub Releases page of `Chelis-Lang/chelis`.
- The `.sha256` sibling file is the expected checksum.

When a shell needs a newer compiler surface, it should bump its pin in a
dedicated change set that also runs its own acceptance oracle against the
new compiler.

Current downstream state:

- `Nautilus v0.1.0` should bump to `chelis = "=0.1.10"` to pick up the
  published darwin-arm64 artifact, the `eval --file` Reef-import fix for ad
  hoc snippet files, and the streamlined release pipeline that avoids a second
  full workspace validation pass during packaging.
- `v0.1.10` is a patch release on top of the `v0.1.7` line: it keeps the same
  documented Phase 3j-pre compiler surface while shipping the validated
  multi-platform release packaging, the file-backed Reef eval fix needed by
  downstream startup probes, and a faster release workflow.

## Versioning Policy

- The tag name and `workspace.package.version` in the root `Cargo.toml`
  must agree. `v0.1.10` corresponds to `version = "0.1.10"` in
  `Cargo.toml`.
- Until `v1.0.0`, minor bumps may include breaking surface changes.
  Downstream shells should respond with exact-pin bumps, not range
  widening.
- Patch bumps (`v0.1.1`, `v0.1.2`, ...) are reserved for bug fixes that
  do not change documented behavior.

## Release Checklist

For each new tag:

1. Workspace gates pass locally: `cargo build --workspace --all-targets`,
   `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D
   warnings`, `cargo fmt --all -- --check`.
2. Workspace version in root `Cargo.toml` matches the intended tag.
3. The owning phase plan lists a concrete acceptance oracle that is
   green.
4. `git tag -a vX.Y.Z -m "<phase> release"` and `git push origin
   vX.Y.Z`.
5. Confirm `gh release view vX.Y.Z` shows the tarball and checksum
   attached.

## Context

The release workflow builds `chelisup` with host Cargo in two full toolchain jobs. Devenv already owns the reviewed Rust toolchain and the committed crate2nix graph.

The Nix-oriented `outputs.chelisup` package includes a wrapper and Nix closure behavior. Its real payload can also retain Nix library paths, so it is not a portable release asset.

The release bootstrap contract uses a bare executable named `chelisup-<platform>`. The current supported release platforms are `linux-x86_64` and `darwin-arm64`.

This change affects package behavior and CI process only. It does not change language semantics, compiler behavior, runtime behavior, backends, or generated code.

OpenSpec remains advisory evidence under `spec/design/spec_provenance.md`. Current specifications and executable tests retain their authority.

## Goals / Non-Goals

**Goals:**

- Add `devenv build outputs.release-chelisup` as the portable build interface.
- Use the committed `Cargo.nix` graph and the configured Devenv Rust toolchain.
- Produce exactly one executable and one SHA-256 sidecar.
- Produce a static Linux x86-64 executable.
- Produce an Apple Silicon executable with Apple runtime dependencies only.
- Fail during the derivation when an artifact contract is false.
- Make the release workflow upload the Devenv output without another `chelisup` Cargo build.

**Non-Goals:**

- The change does not make the complete Chelis toolchain portable.
- The change does not replace the cold cvc5 release builds.
- The change does not add Intel macOS support.
- The change does not change `outputs.chelisup` or its Nix wrapper.
- The change does not change `chelisup install` or bootstrap checksum verification.
- The change does not change release triggers, publication policy, or required checks.

## Decisions

### Add a separate release output module

Add `devenv/release-outputs.nix` and import it after the toolchain module. The module defines `outputs.release-chelisup` for the two supported native systems.

Keep this output outside `devenv/package-outputs.nix`. That module continues to expose Nix installation packages, while the new module exposes publication files.

The output uses the repository worktree as its source. It does not evaluate the root flake or reuse `outputs.chelisup`.

### Build from the committed crate2nix graph

Add a focused helper under `nix/` that imports the filtered source and committed `Cargo.nix`. The helper selects only `workspaceMembers."chelisup"` and sets an empty feature list.

The helper receives `config.languages.rust.toolchainPackage`. It does not create a second Rust version or fetch a crate graph during evaluation.

The Linux path extends that toolchain with `x86_64-unknown-linux-musl`. It uses `pkgs.pkgsCross.musl64` as the crate build package set.

Do not use `pkgs.pkgsStatic`. The hosted Ubuntu oracle ran the static `crc32fast` build script, which exited with signal 11.

The Darwin path uses the native package set and the configured toolchain. Both paths use the lock data already captured in `Cargo.nix` and `Cargo.lock`.

### Keep release files separate from the Nix package

The helper copies the real crate binary into a new output root. It does not copy the Nix launcher from `outputs.chelisup`.

The output root contains only these files:

- `chelisup-linux-x86_64` and its `.sha256` sidecar on Linux
- `chelisup-darwin-arm64` and its `.sha256` sidecar on macOS

The sidecar uses the standard two-space `sha256sum` format. The file name in the sidecar is relative to the output root.

The Rust host detector and the bootstrap script expose only these two platform slugs. Intel macOS fails at detection and makes no asset request.

### Verify portability inside the derivation

The derivation runs the built executable with `--version` and `--help`. The version output must equal the workspace version.

The Linux check uses ELF metadata. It requires x86-64, forbids an interpreter segment, and forbids dynamic `NEEDED` entries.

The Darwin check uses Mach-O metadata. It requires arm64 and permits only `/usr/lib` or `/System/Library/Frameworks` dependencies.

The Darwin build rewrites the Nix `libiconv` load path to `/usr/lib/libiconv.2.dylib` before verification. The executable test proves that the system library remains compatible.

Both checks verify the exact file inventory and the SHA-256 sidecar. Any failed check stops the build before GitHub Actions can upload files.

### Use a dedicated release workflow matrix

Add one `build-chelisup-release` matrix job for `ubuntu-latest` and `macos-latest`. The job keeps read-only repository permissions and runs only through the current manual workflow.

The job checks out the repository, sets up the pinned shared Devenv action, and authenticates the private ci input. It then runs `devenv build --no-tui outputs.release-chelisup`.

The upload step selects the exact executable and sidecar for the matrix platform. The current full toolchain jobs remove direct Cargo build and stage steps.

The glibc 2.31 job continues to publish `chelisup.sh`. The publish job adds the release-output matrix to its dependencies and retains the current asset pattern.

### Lock the contract with positive and negative tests

Add a Python contract suite before implementation. It checks the Devenv module, committed graph use, platform rules, output inventory, dependencies, and workflow structure.

Mutation tests remove each required check and restore a direct Cargo build. Each mutation must make the contract inspector report an error.

The authoritative completion oracle is the `Build portable chelisup` release workflow matrix. A local native build provides support evidence.

## Risks / Trade-offs

- **The musl cross crate graph fails** → Keep the target explicit and fail before publication.
- **A Darwin dependency enters the Nix store** → Reject every non-Apple load path during the derivation.
- **The system `libiconv` ABI differs** → Run the rewritten executable before the output succeeds.
- **A checksum describes the pre-fixup binary** → Generate it only after all binary edits.
- **The matrix publishes the wrong platform file** → Use an explicit platform slug and exact upload paths.
- **The private ci input cannot resolve** → Fail before the Devenv build and expose no partial release artifact.
- **A full toolchain job still builds `chelisup`** → Reject any release workflow Cargo command for that package.

## Migration Plan

1. Add contract tests that fail and add negative mutations.
2. Add the release helper and Devenv output module.
3. Build and verify the output on the current native host.
4. Add the release workflow matrix and remove duplicate Cargo builds.
5. Run the Python workflow suites, Devenv tests, and Nix format checks.
6. Run the manual release workflow matrix on a reviewed branch or tag.

For rollback, remove the matrix and the release output module. Restore the prior platform-specific Cargo build steps only if publication must continue.

## Open Questions

None.

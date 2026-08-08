## Context

The release workflow builds the Linux toolchain twice with Cargo: once on `ubuntu-latest` and once inside a `debian:11` container that caps the glibc floor at 2.31 (#330). The Nix flake builds the same artifacts from the committed `Cargo.nix` graph with a fully pinned cvc5 1.3.1 stack, in a sandbox, with a contract check suite that executes the artifacts.

Chelis has zero users. Distribution reach is not a constraint. The remaining hard constraints are technical: the runtime static library must link under plain glibc `gcc` and under `hipcc`/ROCm, the build must stay reproducible, and our own lanes must be able to consume the artifacts.

The 2026-08 investigation established the facts this design relies on:

- A `.a` archive has no dynamic section. glibc symbol-version binding happens at the consumer's final link against the consumer's libc. The archive does not inherit the build host's glibc floor.
- `chelis-runtime` is pure Rust (`staticlib` + `rlib`; deps `chelis-vocab`, `libc`, `memmap2`, `half`). `chelis-vocab` carries a crate-purity test that forbids `cc` build dependencies. No C objects compiled against Nix glibc headers enter the archive.
- A musl-target staticlib bundles musl libc objects (`+crt-static` is the musl default), which conflict with a glibc consumer. musl is correct for the standalone `chelisup` binary and wrong for this archive.
- The Rust `x86_64-unknown-linux-gnu` target supports glibc >= 2.17, but a binary built on Nix glibc records the highest symbol version it references. The measured expectation is 2.38-2.39; the exact value comes from the migration spike.
- `nix/release-chelisup.nix` is the in-repo template for a portable release derivation with executable postconditions.

## Goals / Non-Goals

**Goals:**

- One pinned, sandboxed build path produces the shipped Linux toolchain.
- The shipped tarball works on a modern glibc system without Nix.
- The runtime archive links under off-Nix glibc `gcc` and under `hipcc`.
- The glibc floor of the shipped binary is a recorded, reviewed value.
- The Cargo container lane and its duplicate cvc5 recipe leave the repository.

**Non-Goals:**

- The change does not set a public glibc-floor support policy. That decision waits for users.
- The change does not cap the floor at any value. The recorded value ratchets visibly; it does not gate on age.
- The change does not alter the Darwin release lane or the macOS artifacts.
- The change does not patch the flake packages. `packages.chelis` stays store-linked; portability belongs to the release output only.
- The change does not build the compiler binary against musl.
- The change does not alter compiler, runtime, CLI, backend, or generated-code behavior.

## Decisions

### The release output mirrors `release-chelisup.nix`

A new `nix/release-chelis.nix` builds `chelis` and `libchelis_runtime.a` from the committed `Cargo.nix` graph with the pinned cvc5 tree, on `x86_64-linux` only. `devenv/release-outputs.nix` exposes it as `outputs.release-chelis`. The derivation stages the same tarball layout the Cargo jobs stage today: `bin/chelis`, `lib/libchelis_runtime.a`, the five public headers, `README.md`, and `LICENSE`, then creates the `.tar.gz` and its SHA-256 sidecar.

The output must not evaluate the root flake, generate a crate graph at evaluation time, use import-from-derivation, or run a host Cargo build. These are the same prohibitions the chelisup release output already carries.

### Portability uses static-archive link resolution plus an interpreter rewrite

The compiler binary links cvc5 statically but inherits dynamic `libstdc++.so.6`, `libzstd.so.1`, `libgcc_s.so.1`, and store rpaths from the Nix build. Two facts shape the mechanism (spike, 2026-08-07):

- `cvc5-sys` emits `cargo:rustc-link-lib=stdc++` on Linux. A driver flag such as `-static-libstdc++` cannot override an explicit `-lstdc++`.
- `chelis-cli` depends on the `zstd` crate, and the Nix build resolves `libzstd` dynamically.

GNU ld searches command-line `-L` directories before the driver's default directories for every `-l` option. A directory that contains only the static archives therefore forces static resolution - the manylinux pattern. The derivation:

1. builds a link directory containing only `libstdc++.a` (from the gcc output) and `libzstd.a` (from `zstd.override { static = true; }`),
2. injects `extraRustcOpts = [ "-Lnative=<that directory>" ]` on the `chelis-cli` crate through the graph's `crateOverrides`,
3. strips the binary,
4. runs `patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath`,
5. asserts with `readelf` that no `/nix/store` string remains and that every remaining `NEEDED` soname is a glibc member or `libgcc_s.so.1`,
6. executes the rewritten binary through an explicit glibc dynamic loader as the in-sandbox run proof (the sandbox has no `/lib64`).

`libgcc_s.so.1` stays dynamic: the Rust standard library references the shared unwinder explicitly, and manylinux whitelists it because every mainstream distro carries it in the base system.

Rejected alternatives: `autoPatchelfHook` patches toward the store; shipping the closure is the wrong shape for a tarball whose consumers run `gcc`; a musl-static compiler binary requires an unproven cvc5-under-musl stack and stays the documented fallback if this leg proves fragile.

### Probe order works around the sandbox

The rewritten binary cannot run inside the Nix sandbox: `/lib64/ld-linux-x86-64.so.2` does not exist there. The derivation therefore probes behavior (`--version`, `--help`, the release fixture, the cvc5 discharge check) on the pre-rewrite binary, then rewrites, then runs static ELF checks only. The post-rewrite behavior proof lives in the release workflow's off-Nix consumption job: a clean Linux container without Nix unpacks the tarball, runs the binary, compiles emitted C against the shipped archive with `gcc`, and executes the program. The container image must carry a glibc at or above the recorded floor.

### The glibc floor is a recorded contract, not a cap

`nix/contracts.nix` gains the recorded floor value. The derivation computes the maximum `GLIBC_*` version among the portable binary's undefined dynamic symbols and fails on any difference from the recorded value, in either direction. A nixpkgs bump that moves the floor is then a one-line reviewed diff beside the lock change. The spike (task 1) measures the initial value; the design does not guess it.

The recorded floor must not exceed the glibc of the off-Nix consumption image. That inequality keeps Constraint C (our own lanes can consume the artifacts) executable.

### The runtime archive keeps its glibc identity with a negative control

The archive stays on `x86_64-unknown-linux-gnu`. A new check fails if the archive defines libc allocator symbols (`malloc`, `free`, `calloc`, `realloc`) - the signature of a bundled-libc (musl) regression. The off-Nix consumption job is the positive control: an off-store `gcc` links the shipped archive and the program runs. HIP consumption keeps its existing documented manual gate on the HIP workstation; this change adds no hosted GPU lane.

### Release workflow topology

- A `build-chelis-release` job (Linux) replaces `build-linux-x86_64` and `build-linux-x86_64-glibc231`. It uses the pinned shared Devenv setup, private-ci authentication, `devenv build --no-tui outputs.release-chelis`, a tag-parity verification (at a `v*` tag, the tag must equal `v<workspace-version>`), and the off-Nix consumption job described above.
- The `chelisup.sh` bootstrap asset staging moves from the deleted glibc231 job into the new Linux job.
- The published Linux asset keeps the exact name `chelisup` requests: `chelis-v<version>-linux-x86_64.tar.gz`. The `-glibc2.31` asset disappears. Branch dispatches upload the workspace-version-named artifact; the `dev-<sha>` naming scheme retires with the Cargo jobs.
- The `smt-build-glibc231` CI lane loses its purpose (it proves the debian:11 recipe) and leaves with the recipe. The plain `smt-build` lane stays: it still proves the cold Cargo cvc5 build that contributors use.
- `publish-release` depends on the new job set. Manual `v*`-tag dispatch and publication policy are unchanged.

### Deferred low-floor retrofit paths

If a low public floor is ever needed: (1) a digest-pinned manylinux-class container running the Cargo recipe, (2) `cargo-zigbuild --target x86_64-unknown-linux-gnu.2.XX` with zig as the CMake toolchain for cvc5, or (3) a musl-static compiler binary. The archive needs no retrofit; the floor question is compiler-binary-only. This paragraph is the record; no work lands now.

## Risks / Trade-offs

- **The floor exceeds a consumer machine's glibc** → The recorded-floor check plus the off-Nix consumption image inequality make the mismatch a build failure, not a field failure. The HIP workstation's glibc is checked in the spike.
- **The static-archive link trick regresses under a toolchain bump** → The `NEEDED` postcondition and the explicit-loader run fail the derivation the moment `libstdc++.so.6` or `libzstd.so.1` reappears; the fallback is shipping those libraries beside the binary with an `$ORIGIN` rpath, and the second fallback is the musl binary.
- **The rewritten binary breaks in a way the pre-rewrite probes cannot see** → The off-Nix consumption job runs the full behavior probe set on the exact shipped tarball.
- **cvc5 discharge regresses in the Nix artifact** → The existing `verify_release_smt.py` check runs in-derivation on the pre-rewrite binary and again in the off-Nix job on the shipped binary.
- **Losing the debian:11 lane loses the only low-floor proof** → Accepted deliberately: zero users, and the retrofit paths are recorded. The `assert no glibc requirement above 2.31` check retires with the lane.
- **A future nixpkgs bump silently raises the floor** → The recorded contract fails the build until the diff is reviewed.

## Migration Plan

1. Spike on `x86_64-linux`: build the flake packages, measure `NEEDED` and the maximum `GLIBC_*` verneed, prove the `-static-libstdc++` link and the patchelf rewrite, and run the off-Nix consumption sequence in a clean container. Record the measured floor.
2. Add the release-output contract tests and workflow contract tests first, failing.
3. Land `nix/release-chelis.nix`, the Devenv output, the recorded floor in `nix/contracts.nix`, and the archive negative control.
4. Rewire `release.yml`: add the new Linux job and off-Nix consumption job, move `chelisup.sh` staging, delete the two Cargo Linux jobs, update `publish-release`.
5. Remove the `smt-build-glibc231` lane from `ci.yml`.
6. Update `docs/smt_build_setup.md`, `docs/book/src/install.md`, and the changelog.
7. Run the local gate and the Nix package CI. Dispatch the release workflow at a branch; require every job green, including off-Nix consumption.
8. Red-team pass per the repository protocol before the phase closes.

Rollback restores the two Cargo release jobs and the `smt-build-glibc231` lane from git history; the flake packages are untouched throughout.

## Spike Results (2026-08-07)

Measured on the local x86_64-linux Nix builder from the pinned flake (glibc 2.42, gcc 15.2.0, zstd 1.5.7):

- Baseline flake binary `NEEDED`: `libstdc++.so.6`, `libzstd.so.1`, `libgcc_s.so.1`, `libm.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2`, with store rpaths and a store interpreter.
- With the static link directory: `NEEDED` reduces to `libgcc_s.so.1`, `libm.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2`.
- After `patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath`: zero `/nix/store` references remain, and the binary prints `chelis 0.18.4` when run through the glibc dynamic loader with only the glibc library directory.
- Maximum `GLIBC_*` verneed: **2.39**. The initial recorded floor is therefore 2.39, below Ubuntu 24.04's glibc.
- `libchelis_runtime.a` defines no libc allocator symbol.
- `extraRustcOpts` reaches the crate build through `crateOverrides` on the committed graph; the crate2nix member wrapper does not accept it directly.

## Open Questions

- The true off-Nix container run (task 1.4) still needs a container runtime; the in-sandbox explicit-loader run is the current approximation. The release workflow's consumption job provides the full proof.
- The HIP workstation's glibc version has not been checked against the 2.39 floor (task 1.6).
- The off-Nix consumption image is a digest-pinned Ubuntu 24.04 (glibc 2.39) or newer Debian; final selection happens with task 4.2.

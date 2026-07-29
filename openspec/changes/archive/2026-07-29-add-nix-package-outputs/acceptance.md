# Acceptance evidence

Date: 2026-07-28

## Inputs and versions

| Item | Revision or version |
| --- | --- |
| Nixpkgs | `f205b5574fd0cb7da5b702a2da51507b7f4fdd1b` |
| Rust overlay | `19a19f3921ae195f2fbd85f5dc57e6d1df63aa0b` |
| crate2nix | `0.15.0` (`7c33e664668faecf7655fa53861d7a80c9e464a2`) |
| `chelis` | `0.17.1` |
| `chelis-runtime` | `0.17.1` |
| `chelisup` | `0.17.1` |
| cvc5 source | `1.3.1` |
| `cvc5-sys` | `0.3.1` |

The lock parity check confirmed that the Nixpkgs and Rust overlay revisions match `devenv.lock`.

## Package output paths

| System | Package | Output path |
| --- | --- | --- |
| `x86_64-linux` | `chelis` | `/nix/store/2qgcbip5s8x2j173ln5xzpr83hr1x94x-chelis-0.17.1` |
| `x86_64-linux` | `chelis-runtime` | `/nix/store/m5vpj20w19hcq0cqk0hkvr4dmd7vms04-chelis-runtime-0.17.1` |
| `x86_64-linux` | `chelisup` | `/nix/store/9r154b9g6nap5x17mcic00f1769qk3wr-chelisup-0.17.1` |
| `aarch64-darwin` | `chelis` | `/nix/store/3q322qf962g1np1r98vf841s0bf2fjya-chelis-0.17.1` |
| `aarch64-darwin` | `chelis-runtime` | `/nix/store/2dxnpzc5g7v2mvlc1gqcvcmv5b99pmcx-chelis-runtime-0.17.1` |
| `aarch64-darwin` | `chelisup` | `/nix/store/svazbsn2m8yjn6iga5bmk25c3249vjfa-chelisup-0.17.1` |

## Exact package contents

Both systems produced the same relative package contents.

### `chelis`

```text
bin/chelis
include/chelis_blas.h
include/chelis_math.h
include/chelis_runtime.h
include/chelis_runtime_dtype.h
include/chelis_simd.h
lib/libchelis_runtime.a
```

### `chelis-runtime`

```text
include/chelis_blas.h
include/chelis_math.h
include/chelis_runtime.h
include/chelis_runtime_dtype.h
include/chelis_simd.h
lib/libchelis_runtime.a
```

### `chelisup`

```text
bin/chelisup
```

## Native validation

| System | Command | Result | Duration |
| --- | --- | --- | --- |
| `x86_64-linux` | `nix build --print-build-logs .#checks.x86_64-linux.native` | Pass | 21 minutes 43 seconds |
| `aarch64-darwin` | `nix flake check --print-build-logs` | Pass | 6 minutes 13 seconds |

The final correction checks used cached package outputs. They passed in 22 seconds on Linux and 8 seconds on macOS.

## Native CI jobs

| Job | Result | Duration |
| --- | --- | --- |
| `Nix Packages (x86_64-linux)` | Pass | 42 minutes 26 seconds |
| `Nix Packages (aarch64-darwin)` | Pass | 23 minutes 25 seconds |

Both jobs passed against commit `d9e067cb` after the crate2nix replacement.

## Fresh red-team result

A fresh local Pi subagent ran the six required mutations in an isolated worktree.

All six mutations failed for the required reason:

- A Nixpkgs revision mismatch failed the lock parity check.
- A Rust overlay revision mismatch failed the lock parity check.
- A missing runtime header failed the exact package inventory check.
- An extra `chelisup` executable failed the exact package inventory check.
- A host command path failed the application contract check.
- A compiler without the SMT feature failed the cvc5 activation check.

The restored macOS flake check passed. The subagent reported a clean worktree and `RED TEAM COMPLETE`.

## Accepted corrections

The adversarial reviews produced three accepted corrections:

1. Package checks now compare each complete output inventory.
2. The compiler check now compares the complete version output.
3. Each native CI job now verifies its exact Nix runner system.

The final native checks passed on both supported systems after these corrections.

## crate2nix replacement

The Rust package layer now uses the checked-in graph from crate2nix 0.15.0. The graph input digest is `c9d143cb620dc6e8ba6128e641f02a2e5d614bcecee31de7a8e5f41abfbe7214`.

The complete `aarch64-darwin` flake check passed after the replacement. The check proved the exact package inventories, runtime consumer, applications, and active cvc5 discharge.

The graph regeneration check reproduced the complete `Cargo.nix` file with crate2nix 0.15.0 in offline Cargo mode.

The final replacement checks passed on both supported systems:

| System | Command | Result | Duration |
| --- | --- | --- | --- |
| `x86_64-linux` | `nix build --print-build-logs .#checks.x86_64-linux.native` | Pass | 1 minute 3 seconds |
| `aarch64-darwin` | `nix flake check --print-build-logs` | Pass | 9 seconds |

Both final checks reused crate outputs from the first replacement builds. The local repository gate passed inside Devenv.

## crate2nix replacement red-team result

A fresh local Pi subagent tested the staged replacement in an isolated worktree.

All five required mutations failed for the required reason:

- A changed `Cargo.lock` failed the graph digest check.
- A changed `Cargo.nix` body failed the exact regeneration check.
- An import-from-derivation marker failed the native crate2nix contract.
- A missing `CVC5_DIR` override failed the fixed cvc5 input contract.
- A missing workspace source override failed the compile asset contract.

The subagent restored all mutations. The focused suites and the exact regeneration check then passed.

The review found one medium test harness defect. The lock parity tests used a `.venv` path in the active worktree.

The test runner now uses `sys.executable`. A detached worktree without `.venv` passed all four lock parity tests.

No product defect remained after the correction. Both native CI jobs passed after the replacement.

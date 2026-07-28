# Acceptance evidence

Date: 2026-07-28

## Inputs and versions

| Item | Revision or version |
| --- | --- |
| Nixpkgs | `f205b5574fd0cb7da5b702a2da51507b7f4fdd1b` |
| Rust overlay | `19a19f3921ae195f2fbd85f5dc57e6d1df63aa0b` |
| `chelis` | `0.17.1` |
| `chelis-runtime` | `0.17.1` |
| `chelisup` | `0.17.1` |
| cvc5 source | `1.3.1` |
| `cvc5-sys` | `0.3.1` |

The lock parity check confirmed that the Nixpkgs and Rust overlay revisions match `devenv.lock`.

## Package output paths

| System | Package | Output path |
| --- | --- | --- |
| `x86_64-linux` | `chelis` | `/nix/store/h9x9gj2jkghg13hy63hglys06n380rnb-chelis-0.17.1` |
| `x86_64-linux` | `chelis-runtime` | `/nix/store/5hmwf0i2cd29rw4c08hkih3zn4rzvrld-chelis-runtime-0.17.1` |
| `x86_64-linux` | `chelisup` | `/nix/store/71r9rmc525yygw6901qbz4x7f6v6qpdg-chelisup-0.17.1` |
| `aarch64-darwin` | `chelis` | `/nix/store/33baipsydiral8xdhhbiz07dp8hplmpc-chelis-0.17.1` |
| `aarch64-darwin` | `chelis-runtime` | `/nix/store/9madhn0ain26ib1kcg13nxzy17lkv0db-chelis-runtime-0.17.1` |
| `aarch64-darwin` | `chelisup` | `/nix/store/dyk7hxrx7v7w4ga8ygq840yxr7j3pyb7-chelisup-0.17.1` |

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
| `Nix Packages (x86_64-linux)` | Pending | Pending |
| `Nix Packages (aarch64-darwin)` | Pending | Pending |

This section remains pending until both GitHub Actions jobs run after the accepted corrections.

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

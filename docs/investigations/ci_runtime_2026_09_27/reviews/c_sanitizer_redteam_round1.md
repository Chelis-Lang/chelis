# PR #2681 red-team round 1

Reviewed head: `4bd1f2d3ba2559f28e17702974bfdcc6c55aa00d`.
Scope: `crates/chelis-backend-c/tests/exec_compile.rs`; sanitizer fixture batching and preservation of existing positive/negative coverage.

## Findings

No in-scope P0, P1, P2, or P3 finding. I am satisfied; round 1 is closed. No out-of-scope defect was established.

## Executed validation

Used the supplied clean worktree and warm executable; no Rust rebuild or tracked-source mutation. Inspected the complete diff, existing negative controls, toolchain selection, temporary-directory lifecycle, and the owning movement/sparse rules in `spec/05-risc-primitives.md`.

Ran each of these independently with the command:

`CHELIS_TEST_CC=<temporary Python clang logging wrapper> target/debug/deps/exec_compile-f8111bacb5046d13 --exact <test-name> --nocapture`

| Test name | Result | Compiler invocations |
| --- | --- | --- |
| checked_c_sparse_mappings_preserve_stored_bits_under_sanitizers | PASS, 20.86 s | 9 batches, 6 kernels each |
| checked_c_movement_permute_and_expand_preserve_bits_under_sanitizers | PASS, 34.11 s | 8 batches, 9 kernels each; 17 separate Int64 positive/mutation compilations |
| checked_c_movement_affine_maps_preserve_bits_under_sanitizers | PASS, 26.17 s | 9 batches, 8 kernels each |
| checked_c_sparse_empty_and_invalid_domains_execute_under_sanitizers | PASS, 10.47 s | 20 separate compilations |
| checked_c_movement_runtime_affine_bounds_reject_before_allocation | PASS, 5.31 s | 11 separate compilations |

The wrapper copied compiler arguments and C fixtures without changing baseline inputs, then delegated to clang. Checked all retained arguments: the 26 batch invocations contain exactly 198 separate kernel `.c` inputs plus one harness `.c` each, with `-fsanitize=address,undefined` and `-fno-sanitize-recover=all`. Source checks confirm each cell is launched by its own `Command::new` process with its index, and results are checked for exit status and exact case-specific stdout. Existing independent bit/map/shape assertions remain in those harnesses.

Adversarial execution on copied sparse batch fixtures, using their recorded compiler arguments:

- Injected a volatile heap out-of-bounds write into case 1: ASan reported it and the process aborted. Cases 0, 2, 3, 4, and 5 each exited 0 with their correct identity output.
- Injected a volatile negative shift into case 1: UBSan reported it and the process aborted. The other five cases again passed individually.
- Corrupted the first expected gather map: case 0 exited 5, proving the exact-value oracle remains active.
- Corrupted the expected output shape: case 0 exited 3.
- Changed only a generated batch dispatcher from `cases[index]()` to `cases[0]()` through the compiler wrapper, then ran the actual Rust sparse test: it failed with exit 101 at the exact stdout identity assertion. Repeated success of the wrong case cannot satisfy the test.

The normal movement run also executed its original inverse-permutation, coordinate-bypass, malformed-target, and missing/late-validation controls. Sparse empty/invalid-domain and dynamic affine before-allocation controls passed on their original per-cell route.

## Limits

This review establishes the stated finite matrix and sanitizer-routing claims on local macOS/clang. It does not establish broader dtype completeness, Linux/compiler portability, hosted CI completion, or hosted speedup. The author's earlier 34.49 s to 28.14 s affine comparison was not independently reconstructed; this review's instrumented affine run was 26.17 s and is not a controlled benchmark. No production, CLI, example, or semantic behavior changed, so those surfaces were not rerun. Per brief, the full fast gate and GitHub state were not rederived.

## Restoration and handoff

All temporary wrapper, retained fixtures, generated binaries, and adversarial mutations under `target/redteam-batch` were removed. No tracked source was edited. Final command:

`.venv/bin/python scripts/worktree_status.py --path /Users/robertronan/chelis-worktrees/c-sanitizer-batch-20260927`

At 2026-09-27 18:26:44 UTC: **FREE**, head `4bd1f2d3ba2559f28e17702974bfdcc6c55aa00d`, clean (0 modified/staged/untracked/unmerged), lease free, no scoped processes, no Git operation. CPU handed back to the orchestrator. Reviewer remains available for verification if needed.

# Concurrent source-bound binding census proof — implementation report

**Verdict: implemented and locally unit-validated; live Rust acceptance and speedup are unmeasured.**

## Delivery

- Worktree: `/Users/robertronan/chelis-worktrees/binding-parallel-proof-20260929`
- Branch: `agent/binding-parallel-proof-20260929`
- Base, freshly fetched `origin/main`: `465a3623be19bd68ada6c39aa7552b63e0ccf8c3`
- Local commit: `520bea2b3ef14dd3653dd55a5e16dac7851a5157` (`perf(ci): overlap binding and wire census proofs`)
- Clean worktree after commit. No push, PR, subagent or numbered-spec change.
- Changed: `scripts/capacity_census_typed.py`, `scripts/capacity_census_compiler_json.py`, their two focused test files, and `.config/ci-test-targets.toml` for the previously unrouted paths. No wire implementation, wire test, baseline or design document changed. The design doc defines authority and coverage, not an execution-order rule.

## Proof boundary

`bindings-discovery` still checks the supplied registration provenance for missing declared functions before doing proof work. It now starts one binding worker for compiler-JSON collection followed by native verification. The calling thread runs `verify_wire_census` concurrently. Compiler-JSON collection executes its supervised Python controls, exact Rust libtest and registration probe, compiler driver/library collection, 22 construction controls and MIR controls without reading wire classifications or constructing `VerifiedCompilerJsonBindings`. Native verification follows serially on the same binding target.

The join always waits for the binding worker, including when wire fails. A failure on either side prevents compiler-JSON finalization, discovery and writing `target/capacity-census-compiler-json-execution.json`; two failures are both reported. Success requires exact `VerifiedWireCensus`, `VerifiedNativeBindings` and private current compiler-JSON collection types; equal source identity at start, from wire, from compiler-JSON collection, from native and at join; then live witness validation. Only then does finalization build `SchemaWireGraph` from the live wire schema, validate the final `VerifiedCompilerJsonBindings`, and pass it to unchanged `discover_bindings`. A saved wire report or dictionary is rejected. The collection and final witness constructors reject normal caller construction. Existing selected/executed, artifact/process hash, registration, native, baseline and mutation validation remains in its owning classes and Rust gate.

## Target and process evidence

The tested default resolver chooses these sibling worktree-local paths:

```
wire:    /Users/robertronan/chelis-worktrees/binding-parallel-proof-20260929/target/agents/729-capacity-rustdoc
binding: /Users/robertronan/chelis-worktrees/binding-parallel-proof-20260929/target/agents/729-capacity-binding-proof
```

The join rejects equal, nested or foreign targets before starting work. The event-controlled unit test observed binding collection and wire verification both started before either completed; releasing wire alone did not return success. It also checked that native verification follows collection and receives the binding target. The compiler-JSON collection unit test checked target forwarding to its libtest, driver, library, construction and MIR helpers, and that no wire verifier runs in collection. Source inspection found their Cargo/rustdoc/probe builds set `CARGO_TARGET_DIR` to the passed target, with compiled binding invocation Cargo under `binding-invocations/cargo`; direct rustc fixture outputs and dependency paths also stay under that target. No actual Cargo subprocess overlap or Cargo target directory was observed in this worktree: the live census was not run.

## Exact local checks

- Before implementation, the new stubs failed against the old interface (missing split functions, mode-aware target resolver and join). After implementation:
  `PYTHONPATH=scripts .venv/bin/python -m unittest test_capacity_census_typed test_capacity_census_compiler_json test_capacity_census_bindings test_capacity_census_wire_verifier test_ci_test_targets test_ci_script_tests -q`
  **PASS, 78 tests**. This includes overlap, no early success, both individual failures, both failures together, source change, target isolation, saved receipt rejection and the nearby ownership/CI routing controls. The argparse usage text in output is the expected rejected `--t` negative test.
- `.venv/bin/python scripts/ci_change_owned.py classify-paths --from-git` **PASS, 5 paths**. It initially identified three unrouted files; the committed path rules close that gap.
- `.venv/bin/python scripts/ci_script_tests.py pr --list` **PASS**, with 3,078 PR tests selected. The fresh venv initially lacked PyYAML, so I installed CI's pinned `PyYAML==6.0.3`; I also installed the four declared Python binding dependencies. These installs changed only this worktree's venv.
- `git diff --check` **PASS** before commit. `python3 scripts/reap_orphans.py` reported no processes for this worktree before build assessment; a final process scan found no owned Cargo, rustc, nextest or census process.

## Missing acceptance and risk

I did **not** run `cargo nextest run -p chelis-python --test capacity_census_bindings` or `cargo nextest run -p chelis-compiler-api --test capacity_census_wire`. Other worktrees had active Cargo/rustc/nextest work; at the decision point one compiler-API test used about 100% CPU, macOS scanning used about 69%, and the host load average was 12.14. Starting two nested target rebuilds then would add significant contention. Consequently this report does not certify the live 17 binding rows, eight final numeric authorities, baseline equality, standalone wire census, actual Cargo overlap, or any speedup.

On a quiet machine, run the binding test first with this worktree's `.venv/bin/python` as `PYO3_PYTHON` and a separate outer `CARGO_TARGET_DIR` under `target/agents/`, then inspect the two nested target paths and timed child-process traces. Run the standalone wire Rust test as a separate acceptance control. Recompiling shared dependencies in sibling targets, CPU contention and the independent wire-probe change may erase the expected wall-time gain; the trial needs measured cold and warm runs before claiming a performance improvement.

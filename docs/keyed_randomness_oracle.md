# Keyed randomness acceptance

The compiler completion command for the finite chelis#2413 corpus is:

```sh
.venv/bin/python scripts/keyed_randomness_oracle.py
```

Run from a clean, committed task worktree with its own uv-managed Python 3.11,
Cargo/nextest, and the native C toolchain used by the integration tests. Check
for orphan builds first. The runner builds the CLI and carried runtime, then
executes 91 exact test identities across 18 integration targets. Its reviewed
catalog is `SUITES` in the runner; additions or replacements are reviewed
changes to the acceptance corpus, not automatic discovery.

The command ends `KEYED RANDOMNESS ORACLE: PASS` only if every required test
is present, selected, non-ignored, executed once and passing, and the retirement
check passes. It bypasses nextest's default test filter, disables retries, and
compares the fresh JUnit identities with the exact nextest listing. An empty
selection, renamed test, skipped case, failed command, absent report or
changed/dirty candidate fails. Tests outside the catalog are not this oracle's
coverage. The required `chelis-compiler-api::key_tensor_forms` target is never
optional: a tree without the tensor-key implementation cannot pass.

Build, list and run commands use `--locked`. The compiler-api list and run
commands enable `chelis-compiler-api/ownership-ledger`, required by the native
execution targets; the other selected packages need no additional feature.

Each invocation uses `target/agents/2413-keyed-oracle`, locks that target, and
writes a new `run-*` directory containing `receipt.json`, command arguments,
stdout/stderr, and each completed group's JUnit. The receipt names the exact
candidate SHA, required and passed/failed/skipped/unrun identities, and the
lanes exercised by each target. The lane labels describe the reviewed tests;
they are not automatic proof of every possible dispatch route. Keep this
directory with the acceptance record. A nonzero child may leave partial
receipts, which never produce PASS.

The runner copies the checked-in nextest config into that run directory and
sets its store to the isolated target. It refuses a changed store or JUnit
layout until the receipt routing is reviewed.

Each build/list/run command has a 1,800-second wall-clock cap; `--timeout N`
changes that per-command bound. The runner kills the command's process group
on timeout/interruption. Test execution uses two nextest workers. It does not
share another worktree's target or accept cached execution reports.

## Corpus and authority

| Requirement | Selected target families |
|---|---|
| [05-RNG-1/2] root, split/fold, nested key derivations and stored bits | `random_key_reference`, `key_surface_lanes`, `key_operations_ir`, `key_operations_c` |
| Helpers, recursion, runtime branches and unrelated control flow | `key_surface_lanes`, `dropout_fixed_stream_api`, `dropout_fixed_stream_cli` |
| Unused results: independent live keys, retained invalid-call traps | `key_operations_ir`, `dropout_fixed_stream_api`, `issue_2463_key_dead_draw_traps` |
| Runtime rates/bounds, empty inputs and [05-OP-8/37] adjoints | `key_operand_random_ir`, `key_operand_random_c`, `dropout_fixed_stream_api`, `dropout_fixed_stream_cli` |
| Key rows, nested `vmap`, activation, branch joins and `grad` replay | `key_operations_ir`, `key_operations_c`, `key_operation_activations`, `key_operations_in_branch_arms`, `key_surface_lanes` |
| [05-OP-69..72] tensor surface forms, runtime count/shape failures | `key_tensor_forms` |
| [04-LIN-9/10] affinity, tensor-key ranks and transported contracts, captures, generics and builtin policies | `key_linearity`, `key_builtin_cases` |
| Symbolic derivations, one consumer and replay on the wire | `key_operations_ir`, `key_operand_random_ir`, `wire_random_domains` |
| Stdlib initialization and the executable dropout example | `key_std_initialisers_cli`, `dropout_fixed_stream_cli` |
| `par` disposition: typed #2503 refusal, before Eval/C execution | `jit_par_passthrough`, `jit_par_runtime_gap` |

Expected words come from checked-in independent Rust transcriptions of the
numbered specs and pinned worked bit patterns, not an Eval-versus-C comparison
alone. Direct IR tests invoke the DAG evaluator; the native graph tests compile
and run generated C; public surface and CLI tests cover their host/tensor
entry paths. Some uniform tests use f32 bounds at multiple result widths.
Their selected keyed results do not prove [05-OP-8]'s entire same-dtype bound
signature domain, which remains separately owned by chelis#1295.

The retired counter-era nested-seed/ordinal conditions are replaced by explicit
key derivations: no handler or ordinal contributes to a draw. `par` currently
has the documented checker rejection in spec/07, so its positive execution is
outside this corpus and remains chelis#2503. This disposition does not weaken
its normative semantics. HIP/Metal execution, every possible key program,
LaCaDiLE certification completeness, and shell release are not certified here.

The retirement invariant scans production Rust under `chelis-ir/src` and
`chelis-compiler-api/src` for the retired exact identifiers `static_controls`,
`StaticControls`, `execution_exclusion`, and `FixedControlPlan`. Missing source
roots also fail. This narrow reintroduction tripwire complements the selected
recursion/match/branch witnesses; it does not prove the absence of a renamed
or differently represented classifier.

## Completion record

This manual oracle is not dispatched by default PR CI. Script-unit tests its
failure handling; ordinary CI continues to select Rust tests under its own
rules. A #2413 compiler completion claim needs this command's terminal receipt
at the final candidate, applicable required CI and package-expansion receipts,
and the extended validation required for a completion claim. General green CI
or `gate.py --fast` alone is not the named acceptance decision.

The hub also requires affected-shell migrations/releases and coordinated
Hull/compiler release. Record those independently, including the explicit
`sample(k)` / `f_given(noise, x)` layering and release receipts. A compiler PASS
does not complete Hull's aggregate acceptance or close those release tasks.

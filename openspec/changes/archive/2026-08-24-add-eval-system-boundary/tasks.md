## 1. Red Tests and Fixtures

- [x] 1.1 Add fake-adapter tests for all eight covered builtins before the port implementation.
- [x] 1.2 Verify that each fake-adapter test records exactly one operation with the expected path or argument vector.
- [x] 1.3 Add refusal tests for `Filesystem` and `Process` that verify zero adapter calls.
- [x] 1.4 Add a construction test that proves the invariant evaluator context receives a deny-all policy.
- [x] 1.5 Add default-adapter parity tests for success values, directory entry order, process output, and exit-code conversion.
- [x] 1.6 Add parity tests for all seven exact error templates from the capability specification.
- [x] 1.7 Add a directory-open adapter test and a typed template test for directory-entry errors.
- [x] 1.8 Add source-guard fixtures for qualified calls, imports, aliases, process construction, and every path-existence form.

## 2. Typed Evaluator Boundary

- [x] 2.1 Add the closed `EvalSystemCapability` type with only `Filesystem` and `Process`.
- [x] 2.2 Add typed system and refusal errors with one conversion to the current evaluator error text.
- [x] 2.3 Add the eight-method `EvalSystem` port and the typed process output value.
- [x] 2.4 Add the mandatory policy wrapper with a permissive default and fail-closed refusal behavior.
- [x] 2.5 Add the standard library adapter as the only evaluator module that imports filesystem and process APIs.
- [x] 2.6 Add one crate-private program evaluator entry that accepts a mutable policy wrapper for fake-adapter tests.
- [x] 2.7 Give the `runtime/invariant.rs` evaluator context a deny-all boundary without a decode API parameter.
- [x] 2.8 Keep `eval`, `eval_selected`, `eval_many`, `eval_in_context`, `eval_in_context_with_bindings`, and `eval_many_in_context` unchanged.
- [x] 2.9 Keep `prepare_eval`, `prepare_eval_in_context`, and both prepared `eval_root` methods unchanged.
- [x] 2.10 Route every public program evaluation path through a fresh permissive boundary.

## 3. Operation Migration

- [x] 3.1 Route `read_file`, `read_lines`, `read_bytes`, and `mmap_file` through the filesystem port methods.
- [x] 3.2 Route `write_file`, `file_exists`, and `list_dir` through the filesystem port methods.
- [x] 3.3 Route `process_run` through the process port method without a shell or fallback.
- [x] 3.4 Keep line splitting, output decoding, exit conversion, and runtime-value construction in pure evaluator code.
- [x] 3.5 Preserve `std::fs::read_dir` order without a sort.
- [x] 3.6 Abort `list_dir` on the first directory-entry error.
- [x] 3.7 Keep `print` and `debug` on the evaluator transcript path without a system adapter call.
- [x] 3.8 Remove direct filesystem, path-existence, and process imports from evaluator dispatch.
- [x] 3.9 Run the new compiler API tests.
- [x] 3.10 Run `cargo test -p chelis-cli --test process_run_builtin`.

## 4. Architecture Guard and Oracle

- [x] 4.1 Implement `scripts/eval_system_guard.py` with fail-closed source classification.
- [x] 4.2 Make the guard allow direct system access only in the selected adapter module.
- [x] 4.3 Run every positive and negative guard fixture, including imported aliases and method-form `.exists()`.
- [x] 4.4 Implement `<managed-python> scripts/eval_system_oracle.py` as the authoritative completion oracle.
- [x] 4.5 Make the oracle run compiler API tests, the live CLI parity suite, guard tests, and the source guard.
- [x] 4.6 Add the focused oracle to the `lint-and-unit` stage and lock that command with `scripts/test_gate.py`.
- [x] 4.7 Keep the restrictive policy private, program evaluation permissive, and invariant evaluation deny-all.

## 5. Documentation and Review Routing

- [x] 5.1 Correct stale file-access and `process_run` statements in `spec/design/effect_taxonomy_expansion.md`.
- [x] 5.2 Correct the corresponding stale assurance statement in `spec/design/chelis_trust_stack.md`.
- [x] 5.3 Correct stale `process_run` status entries in `spec/design/chelis_hull_design_spec.md`.
- [x] 5.4 State that the boundary does not mediate generated artifacts or the runtime C ABI.
- [x] 5.5 Verify that no numbered language specification or public effect vocabulary changed.
- [x] 5.6 State that #1170 owns compiled-lane inventory and #729 owns the separate numeric conformance matrix.
- [x] 5.7 State that #267 owns compiled `process_run` support and runtime sandbox work.

## 6. Strict and Adversarial Validation

- [x] 6.1 Run `<managed-python> scripts/eval_system_oracle.py` and require complete success.
- [x] 6.2 Run `openspec validate add-eval-system-boundary --strict --no-interactive`.
- [x] 6.3 Run `cargo fmt --all -- --check` after all Rust changes.
- [x] 6.4 Run a fresh-context adversarial review against the capability specification and implementation.
- [x] 6.5 Plant direct filesystem, process, and path-existence bypasses in guard fixtures and verify rejection.
- [x] 6.6 Compile and execute a supported filesystem program to verify that the compiled lane remains outside this claim.
- [x] 6.7 Verify exact public error text and result parity for all eight operations after extraction.

## 7. Local Acceptance

- [x] 7.1 Run `python3 scripts/gate.py --local` and resolve every applicable failure.
- [x] 7.2 Record `<managed-python> scripts/eval_system_oracle.py` as the authoritative completion evidence.

Pull-request creation and hosted checks remain separate remote gates. They require separate approval after this local change is complete.

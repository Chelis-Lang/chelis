# Issue #207 diagnosis: `chelis check` exit-code path

## Symptom

`chelis check src/runtime.ch` prints JSON with a non-empty `errors`
array (e.g. a `TypeMismatch`) but the process exits `0`. The same
program then fails `chelis test` with exit `2` because the
"compile test context" stage rejects it. Downstream CI gates that
treat `chelis check` exit `0` as success silently accept programs
that later fail.

## Root cause

`cmd_check` in `crates/chelis-cli/src/main.rs` (single-file branch
near the end of the function) computes the JSON report through
`cmd_check_one`, prints it to stdout, and returns `Ok(())`. The
caller in `main` only treats `Result::Err` as a non-zero exit, so a
populated `errors` array in the JSON is invisible to the exit-code
path. The doc comment above `cmd_check` pins this as the RT-205 F7
contract — exit 0 even with type errors, so JSON-parsing consumers
do not have to special-case the success-with-errors path.

The directory-walk branch has the same shape: it accumulates per-file
JSON entries, prints the aggregate, and only sets `had_error` when a
per-file processing failure produced `Err(_)`. Per-file `errors[]`
contents from `cmd_check_one` Ok-paths are never inspected.

## Decision (per issue #207 and the user's pinned design)

Invert the RT-205 F7 contract. The new invariant is
`exit_code != 0 iff json.errors.len() > 0`. Empty errors imply exit 0;
any errors imply exit non-zero. The machine-facing JSON shape is
unchanged.

Exit code on errors: `2`, matching `chelis test`'s exit code for
"compile test context" failures (the convention for type errors at
that surface, per `cmd_test`'s contract docstring and the `main` arm
that maps `chelis test`'s `Err(_)` to `std::process::exit(2)`).

## Fix surface

1. `cmd_check` / `cmd_check_one` in `crates/chelis-cli/src/main.rs`
   propagate an exit code based on the JSON `errors` array, including
   the directory-walk aggregation branch.
2. The doc comment above `cmd_check` is rewritten to describe the
   new invariant and reference issue #207.
3. Tests that previously asserted `.success()` on programs producing
   errors are updated to drop the exit-code assertion (the JSON content
   assertions stay as-is). Test helpers (`run_check`, `run_json_check`)
   that are shared between clean and error programs drop their
   blanket `.success()` so both sides keep working.
4. The RT-205 F7 contract test (`check_exits_zero_with_errors_contract.rs`)
   is inverted to assert exit `2` on the validator-rejection fixture.

## Downstream caller impact

`chelis-tide`'s `chelis_check` MCP tool and `/check` HTTP endpoint
read stdout JSON directly; they do not inspect the process exit code
of an internal call, so the contract flip is transparent to them.
The CLI integration test in `crates/chelis-tide/tests/api.rs` runs
`chelis check` on the clean `hello_tensor.ch` fixture and continues
to assert `.success()`, which still holds for clean programs.

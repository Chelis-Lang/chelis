# Testing

`chelis test` runs Chelis tests from a [Reef package](reef.md). Start in a package with a
`reef.toml`, a `src/` module, and a `tests/` directory. A standalone file without a Reef
package does not provide the package graph this command needs.

## Write and run a test

Put a `.ch` file under `tests/`, such as `tests/core.ch`. Test functions have a `test_` name,
no parameters, and a `unit` result. For a package whose module prefix is `Demo`:

```chelis-surf-fragment
module Demo.Tests.Core
def test_value() -> unit = test_assert(true, "value is valid")
```

Replace `true` with the condition your program must satisfy. From the package root, run:

```sh
chelis test
chelis test tests/core.ch
chelis test tests/ --filter test_value
```

With no path, `chelis test` searches `tests/` beneath the current directory. It discovers `.ch`
files recursively. `--filter` matches a substring of the displayed `<file>::<test_name>`; it
is useful for rerunning one test or group.

A passing suite exits `0`. A selected test that fails makes the suite exit `1` and reports the
failed test and diagnostic. If no `.ch` files or runnable tests are found, or a filter selects
none, the command exits `2` with an error rather than reporting a passing empty suite. Check
the test path and filter, then run again. A missing or invalid Reef package is also a runner
error.

## Time limits and output

Each test has a 30-second limit by default; the whole command has a 600-second limit, including
package preparation. Set either limit in seconds:

```sh
chelis test tests/ --timeout 10 --suite-timeout 120
```

A whole-suite timeout exits `1` and marks the run incomplete. Completed test results remain
available, but they do not mean the suite finished.

On a completed run, `--json` writes newline-delimited JSON test records and a summary. If an
ordinary run selects no tests, JSON output contains one `errors` record with
`empty_test_selection` and no passing summary. A timed-out JSON run marks the suite as timed
out and its summary as incomplete.

For a simpler debugging run, `chelis test tests/ --batch-mode file --jobs 1` runs files
separately, one at a time. The default `--batch-mode auto` may group eligible files; both modes
report test results in discovery order.

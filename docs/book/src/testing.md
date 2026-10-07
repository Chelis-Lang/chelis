# Testing

`chelis test` runs Chelis tests from a [Reef package](reef.md). Start in a package with a
`reef.toml`, a `src/` module, and a `tests/` directory. A standalone file without a Reef
package does not provide the package graph this command needs.

## Write and run a test

Put a `.ch` file under `tests/`, such as `tests/core.ch`. A test is a
definition whose name starts with `test_`, with no parameters and a `unit`
result. Its body calls assertions. For a package whose module prefix is
`Demo`:

```chelis-surf-fragment
module Demo.Tests.Core
def test_value() -> unit = test_assert(gt(3i64, 2i64), "3 is greater than 2")
def test_sum() -> unit = test_assert_eq(add(2i64, 2i64), 4i64, "2 + 2")
def test_softmax() -> unit = test_assert_close_tensor(softmax(to_tensor([0.0, 0.0], f32), 0), to_tensor([0.5, 0.5], f32), 1e-6f32, "uniform softmax")
def test_wrong() -> unit = test_assert_eq(mul(2i64, 3i64), 5i64, "2 * 3")
```

Running `chelis test` from the package root prints:

```text
tests/core.ch
  test_value .................... PASS
  test_sum ...................... PASS
  test_softmax .................. PASS
  test_wrong .................... FAIL (assert_eq (2 * 3): expected 5, got 6)

3 passed, 1 failed
```

and exits `1` because `test_wrong` failed. The four assertions are built in,
need no import, and carry the `Test` effect. Each takes a label last; a
failure reports the label and stops that test.

| Assertion | Arguments | Passes when |
|---|---|---|
| `test_assert` | `(cond: bool, label: string)` | `cond` is true. |
| `test_assert_eq` | `(actual: Q, expected: Q, label: string)` | the values are equal. `Q` is any scalar, or a `List`, tuple, `Option`, or data type of comparable values. Float NaN is never equal; `0.0` equals `-0.0`. |
| `test_assert_eq_tensor` | `(actual: &tensor[..r, p], expected: &tensor[..r, p], label: string)` | the shapes are equal and every element is equal. Integers and `bool` compare exactly, floats as in `test_assert_eq`. A failure names the first unequal element in row-major order. |
| `test_assert_close_tensor` | `(actual: &tensor[..r, p], expected: &tensor[..r, p], tol: p, label: string)` | `p` is a float dtype, the shapes are equal, and every element satisfies `abs(actual - expected) <= tol`. |

For `test_assert_close_tensor`, the tolerance has the tensors' dtype and must
be finite and nonnegative; zero means exact equality. The difference is
computed in `f32` for `f16`, `bf16`, and `f32` tensors and in `f64` for `f64`
tensors. NaN is never close to anything, and an infinity is close only to the
same signed infinity. The comparison is absolute, so choose `tol` for the
magnitude of your values. `Std.Test` wraps these builtins with a named
`assert_*` family, including a scalar `assert_close` and `assert_shape`; see
[Runtime and standard library](stdlib.md).

To run all tests, one file, or the tests whose name contains a string:

```sh
chelis test
chelis test tests/core.ch
chelis test tests/ --filter test_value
```

With no path, `chelis test` searches `tests/` beneath the current directory. It discovers `.ch`
files recursively and runs them in the Reef package that contains the path, found by walking
up from the path itself, wherever the command runs. A discovered file that belongs to another
package, such as one nested under the path with its own `reef.toml`, is an error; run that
package's tests separately. `--filter` matches a substring of the displayed
`<file>::<test_name>`; it is useful for rerunning one test or group.

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

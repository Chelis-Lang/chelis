# Checking Properties

`chelis prove` checks `@property` declarations in Surf or Deep inputs. With obligation support
available, it also checks obligations derived from declared invariants; see
[Opaque Types With Declared Invariants](opaque-invariants.md) for an example. Read each
result's method and qualifications before treating a pass as a guarantee.

## Check what this compiler can run

```sh
chelis prove --capabilities
```

This prints JSON without loading a source file. Check `smt_available` before requesting solver
checks and `obligation_engine_available` when relying on invariant obligations.
`beacon_scalar_available` indicates whether the optional `--tier beacon-only` scalar range
path can be attempted. The available methods depend on how the compiler is built.
`supported_tiers` lists possible methods, including some this installation cannot use. Confirm
the method in each result.

## Run properties

A Surf property can live in `properties/nonnegative.ch`:

```chelis-surf-fragment
@property nonnegative forall(x: f32):
  (x * x) >= 0.0
```

From the directory containing `properties/` or `src/`, `chelis prove` discovers `.ch` and
`.dp` files under `properties/` and `src/`. It does not search `tests/`. If neither directory
contains source files, the command exits `3`. Pass a directory or file to narrow the input:

```sh
chelis prove
chelis prove properties/
chelis prove properties/nonnegative.ch --json
chelis prove properties/ --only 'nonnegative*' --samples 1000 --seed 42
```

`--only` filters property names; a trailing `*` matches a name prefix. `--samples` sets the
requested sample count, and `--seed` makes sampling repeatable. The default `--tier auto` may
use sampling. Use `--tier fuzz-only` to request sampling or `--tier smt-only` to request the
solver. Check `status` and `proof_tier` together to see which method produced a passing result;
a command-line request alone does not establish the method used. An unsupported result does not
establish the property. An explicit `.ch` file discovers properties in that file; imports can
resolve names but are not additional discovery targets.

## Read the result

In a standard build, `--json` writes newline-delimited JSON property and obligation records,
then a summary record when the run reaches its summary. Check that `total` in the summary
record is greater than zero and inspect each record:

| Field | What to check |
|---|---|
| `status` | `passed`, `failed`, `unsupported`, or `error`. |
| `proof_tier` | The method associated with the result, such as `fuzz` or `smt`. On an unsupported result, it can name an attempted path; `none` means no method was selected. |
| `composite_verdict` and `qualifiers` | The strength and limits of a passing or failing result. |
| `samples`, `sampling_method`, and sample counts | Evidence for a sampled result; accepted, attempted, and rejected counts appear when sampling details apply. |
| `arith_model` | `real` means the solver checked real arithmetic. |

A custom build that omits `proof_tier` cannot establish a formal method from its JSON; use
`--capabilities`, `samples`, and the verdict to understand its limits.

A sampled `passed` result is empirical (`composite_verdict: "fuzz_validated"`); it does not
establish the property for every input. A solver result over `real` arithmetic does not
establish identical behavior for IEEE floating-point execution. For example,
`proven_modulo_real_arithmetic` records that limitation. A proof involving a sampled assumption
may have further qualifications. Read the complete verdict and `qualifiers`, even when the
exit code is `0`.

The command exits `0` when all reported checks pass, `1` when a property or obligation fails,
`2` when one is unsupported, and `3` for an error. When results differ, an error takes
precedence, then a failure, then an unsupported result. An input or setup error can stop the
run before a JSON summary is written; check the exit code and stderr as well as stdout.

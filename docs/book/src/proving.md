# Checking properties

`chelis prove` checks `@property` declarations in Surf or Deep inputs. With obligation support
available, it also checks obligations derived from declared invariants; see
[opaque types with declared invariants](opaque-invariants.md) for an example. Read each
result's method and qualifications before treating a pass as a guarantee.

## Check what this compiler can run

```sh
chelis prove --capabilities
```

This prints JSON without loading a source file. Check `smt_available` before requesting solver
checks and `obligation_engine_available` when relying on invariant obligations.
The available methods depend on how the compiler is built.
`supported_tiers` lists possible methods, including some this installation cannot use. Confirm
the method in each result.

## Run properties

A Surf property can live in `properties/nonnegative.ch`:

```chelis-surf-fragment
@property nonnegative forall(x: f32):
  ((x * x) >= 0.0)
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

The `--tier` values select how a property is checked:

| Tier | Method | What a pass establishes |
|---|---|---|
| `auto` (default) | Type checking first. Then induction when the property calls a recursive function, otherwise the SMT solver. Sampling when neither applies. | Whatever the method named in `proof_tier` establishes. |
| `smt-only` | The SMT solver, over real numbers. | The property for every input satisfying the `where` preconditions, in real arithmetic. |
| `induction-only` | Induction on one integer argument of a recursive function: the solver proves a concrete base case and a symbolic step case. | As `smt-only`: every input, in real arithmetic. Both cases must be proved; either failing or being vacuous leaves the property unproved. |
| `fuzz-only` | Seeded random sampling of inputs that satisfy the preconditions. | The property held on every sampled input, and nothing about the others. |

An unknown tier is a usage error before discovery or verification. With
`auto`, an SMT proof reports zero samples even when `--samples` was supplied;
the sample count and seed apply when sampling runs.

Induction applies to a recursion on an integer such as this bond valuation,
which recurses on `n` down to a base case at `n <= 0`:

```chelis-surf
module Examples.InductionBond
def bond_value(n: i32, coupon: f64, discount: f64) -> f64 = if (n <= 0) then 1.0f64 else (coupon + (discount * bond_value((n - 1), coupon, discount)))
@property bond_value_nonnegative forall(n: i32, coupon: f64, discount: f64) where n >= 0, coupon >= 0.0f64, discount >= 0.0f64:
  (bond_value(n, coupon, discount) >= 0.0f64)
```

`chelis prove induction_bond.ch --json` passes it with `"proof_tier":"induction"`,
`"composite_verdict":"proven_modulo_real_arithmetic"`, and `"samples":0`.
Its `induction` field holds the base and step goals the solver discharged,
each with `"status":"proved"`, and names the induction variable
(`"variable":"n"`).

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

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

`--only` keeps the properties and invariant obligations (named like
`invariant:Probability:scale`) whose unqualified name contains the given text, or, with a trailing `*`, starts with the text before it. A `*` elsewhere is not a wildcard, and a module-qualified name such as `Stats.Opaque.prob_value_in_unit_interval` matches nothing. A filter that matches nothing is not an error: the run reports `0 passed, 0 failed, 0 unsupported, 0 errors` and exits `0`, so check the counts. `--samples` sets the
requested sample count, and `--seed` makes sampling repeatable. The default `--tier auto` may
use sampling. Use `--tier fuzz-only` to request sampling or `--tier smt-only` to request the
solver. Check `status` and `proof_tier` together to see which method produced a passing result;
a command-line request alone does not establish the method used. An unsupported result does not
establish the property. An explicit `.ch` file discovers properties in that file; imports can
resolve names but are not additional discovery targets.

The `--tier` values select how a property is checked:

| Tier | Method | What a pass establishes |
|---|---|---|
| `auto` (default) | Type checking first. Then induction when the property calls a recursive function, otherwise the SMT solver. Sampling when neither applies, and again after a solver proof, to look for a counterexample the real-arithmetic model cannot see. | Whatever the method named in `proof_tier` establishes, and, for a solver proof, whatever `machine_arithmetic` reports about machine arithmetic. |
| `smt-only` | The SMT solver, over real numbers. | The property for every input satisfying the `where` preconditions, in real arithmetic. |
| `induction-only` | Induction on one integer argument of a recursive function: the solver proves a concrete base case and a symbolic step case. | As `smt-only`: every input, in real arithmetic. Both cases must be proved; either failing or being vacuous leaves the property unproved. |
| `fuzz-only` | A fixed set of edge inputs, then seeded random sampling. The edge inputs are each binder's guard endpoints and their adjacent values, the neighbours of a `!=` guard's excluded value, both zeros, the smallest positive subnormals, the largest finite values of the type, small integers, and a ladder of decade magnitudes. Every input, edge or random, must satisfy the preconditions. | The property held on every sampled input, and nothing about the others. |

An unknown tier is a usage error before discovery or verification. With
`auto`, an SMT proof reports zero samples even when `--samples` was supplied;
the sample count and seed apply when sampling runs, including the
machine-arithmetic search described below, whose own count is reported
separately in `machine_arithmetic`.

## A solver proof is checked again in machine arithmetic

The solver decides a property over the real numbers, so it rules out a real
counterexample and nothing else. A property can be true over the reals and
false in binary64: `(x + 1.0f64) - x > 0.5f64` holds over the reals and fails
at `x = 1e17f64`, where `x + 1.0f64` rounds back to `x`.

`--tier auto` therefore does not stop at a solver proof. It samples the
property — or the derived producer obligation — afterwards, looking only for a
counterexample, and reports the result in the additive `machine_arithmetic`
field:

- `"status":"refuted"` — sampling found a concrete machine counterexample. The
  record is `"status":"failed"` with `"composite_verdict":"failed"` and carries
  the counterexample. A machine value is a definite refutation, so it replaces
  the proof.
- `"status":"corroborated"` — every sampled input satisfied the property. The
  proof stands exactly as the solver produced it: the verdict is still
  `proven_modulo_real_arithmetic` and `real_arithmetic` is still in
  `qualifiers`, because sampling is not exhaustive.
- `"status":"unchecked"` — sampling established nothing (an input the generator
  cannot produce, or preconditions it could not satisfy), with a `reason`. The
  proof is unchanged; a sampler limit is not evidence about the property.

The field is absent when no such check ran: under `--tier smt-only`,
`--tier induction-only`, and `--tier beacon-only`, an explicitly requested tier
is never widened into a second one. An induction proof carries no check either,
and neither does a result conditional on an asserted axiom such as
`clarabel.qp.ideal_optimality`, which idealizes the float solver on purpose —
sampling there would contradict the assumption you declared, not the proof.

In plain text an `auto` run prints the disclosure rather than a sample ratio a
proof does not have:

```text
property: sq_nonneg -- proven_modulo_real_arithmetic (proved over the reals; machine arithmetic corroborated on 100 samples)
```

What this check can find is bounded by what the sampler generates; it does not
promise that every machine counterexample is reachable. What it does establish
is that `auto` is never weaker than `--tier fuzz-only` on the same file.

The edge inputs do not depend on the seed, so a property refuted at one is
refuted under every seed. They are proposed first, so a budget smaller than
the edge set spends all of it on edge inputs and draws nothing at random; raise
`--samples` to add random draws on top.

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

## Prove a property about a Clarabel QP

The optional `chelis-clarabel` package exposes `Clarabel.Qp.solve` to ordinary
Chelis code. Build `chelis` with `clarabel-provider,smt`, declare the package as
a Reef dependency, and match the result of `solve` or a typed helper that
returns one direct `solve` call. A property may attach
`with contract = "clarabel.qp.ideal_optimality"` to use the `Solved` primal as
an ideal exact-real optimizer for that call's `P`, `q`, `A`, `b`, and cones. The
non-`Solved` arm must establish its claim without that assumption. See the
executable files in `examples/clarabel_qp/tests/` and the commands
in `packages/chelis-clarabel/README.md`.

The SMT lowering accepts fixed-size `f64` QP data with symbolic entries,
pure helpers with named bindings and conditionals, and fixed zero and
nonnegative cone blocks.
It proves the PSD premise for exact literal `P` or a `B^T B` construction.
For another symbolic `P`, add
`with contract = "clarabel.qp.assume_psd"` to the same property. That explicit
premise applies to the resolved call's `P` and is listed as a separate axiom.
For example, `tests/ideal_symbolic_constrained_quality.ch` proves that a
feasible baseline cannot score better than the optimizer after the algorithm
applies its step, for every input satisfying the property's `where` condition.
Other cone families can still be used at runtime; properties that require
their ideal contract report `unsupported`. A successful dependent result says
`proven_modulo_asserted_axiom` and lists `real_arithmetic`. It is a conditional
mathematical result, not a bound on the `f64` values returned by the solver.

## Trace a generated property to its requirement

A translator can supply a `.spans.json` manifest beside generated Surf source.
The manifest connects each property name to a requirement in its original file.
For example, a generated `properties.ch` file uses `properties.spans.json`:

```sh
chelis prove properties.ch --json
chelis prove properties.ch --spans properties.spans.json --json
```

The first command discovers the sibling manifest. The second selects it
explicitly. An explicit manifest accompanies a single `.ch` or `.dp` input.
Directory discovery uses each source file's sibling manifest.

Before reporting property results, Chelis checks the manifest's generated-file
identity, source hashes, byte ranges, and property mappings. An edited
requirement or generated file produces an input error until the translator
regenerates the artifacts. The displayed requirement text comes from the
verified original file. JSON records retain `source.file` for the generated
file and add `source.requirement` with its ID, original file, line, column,
text, source hash, and byte span.

The provenance check leaves proof results intact. A counterexample remains a
counterexample, and a sampled result retains its sampling verdict. A valid
source mapping establishes where the proposition came from; the proposition
and the named verification method establish what was checked.

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
| `machine_arithmetic` | For a solver proof under `auto`: whether sampling in machine arithmetic `refuted`, `corroborated`, or left the proof `unchecked`. Absent when no such check ran. |

A custom build that omits `proof_tier` cannot establish a formal method from its JSON; use
`--capabilities`, `samples`, and the verdict to understand its limits.

A sampled `passed` result is empirical (`composite_verdict: "fuzz_validated"`); it does not
establish the property for every input. A solver result over `real` arithmetic does not
establish identical behavior for IEEE floating-point execution. For example,
`proven_modulo_real_arithmetic` records that limitation, and under `auto` the
`machine_arithmetic` field records what sampling in machine arithmetic found
about it. A proof involving a sampled assumption
may have further qualifications. Read the complete verdict and `qualifiers`, even when the
exit code is `0`.

For a checked scalar source call that takes a constructed record, SMT substitutes the
record's fields into the function body before lowering arithmetic. A square root enters
SMT only after a separate proof establishes that its argument is nonnegative from
independent, total preconditions. The solver then uses the unique nonnegative root of
`q*q = argument`. A missing domain proof or an inconclusive solver result stays
`unsupported`. When the solver instead shows the domain obligation **false**, it has a witness:
an input, admitted by the conjuncts the domain proof was allowed to use, at which the square
root takes a negative argument. Those conjuncts are only part of the property's preconditions
and the solver reasons over real arithmetic, so the witness is a candidate, not a result: it is
rounded to each parameter's declared width and evaluated concretely. If every
precondition holds there and the property body is false, the result is `failed` with that
machine value as the counterexample, and the verdict is a definite `failed` rather than
`disproved_modulo_real_arithmetic`, because a concrete evaluation decided it. If the witness
violates a precondition the domain proof did not use, or the body holds there, the result stays
`unsupported`. [The executable record example](../../../examples/record_sqrt_property.ch) shows
this route.

The command exits `0` when all reported checks pass, `1` when a property or obligation fails,
`2` when one is unsupported, and `3` for an error. When results differ, an error takes
precedence, then a failure, then an unsupported result. An input or setup error can stop the
run before a JSON summary is written; check the exit code and stderr as well as stdout.

For a contract-bound property, a subprocess failure in the configured external
contract checker reports a branded `unsupported` result and exits nonzero. A
failed checker cannot turn into a sampled pass.

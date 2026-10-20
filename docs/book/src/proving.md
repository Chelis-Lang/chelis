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

| Option | Method | What a pass establishes |
|---|---|---|
| `auto` (default) | Type checking first. Induction handles recursive properties, otherwise SMT. Sampling runs when neither applies and after solver proofs to find counterexamples the real-arithmetic model cannot see. | Whatever the method named in `proof_tier` establishes. For a solver proof, read `machine_arithmetic` for the machine-arithmetic result. |
| `smt-only` | The SMT solver, over real numbers. | The property for every input satisfying the `where` preconditions, in real arithmetic. |
| `induction-only` | Induction on one integer argument of a recursive function: the solver proves a concrete base case and a symbolic step case. | As `smt-only`: every input, in real arithmetic. Both cases must be proved; either failing or being vacuous leaves the property unproved. |
| `fuzz-only` | A fixed set of edge inputs, then seeded random sampling. Edge inputs include each binder's guard endpoints and adjacent values, plus neighbours of a `!=` guard's excluded value. They also include both zeros, the smallest positive subnormals, the largest finite values, small integers, and a ladder of decade magnitudes. Every input must satisfy the preconditions. | The property held on every sampled input, and nothing about the others. |

An unknown `--tier` value is a usage error before discovery or verification. With
`auto`, an SMT proof reports zero samples even when `--samples` was supplied.
The sample count and seed apply when sampling runs, including the
machine-arithmetic search below. That search reports its own count in
`machine_arithmetic`.

The edge inputs do not depend on the seed, so a property refuted at one is
refuted under every seed. The sampler proposes edge inputs first.
A budget smaller than the edge set spends all of it on edge inputs and draws
nothing at random. Raise `--samples` to add random draws on top.

## A solver proof is checked again in machine arithmetic

The solver decides a property over the real numbers, so it rules out a real
counterexample and nothing else. A property can be true over the reals and
false in binary64. For example, `(x + 1.0f64) - x > 0.5f64` holds over
the reals and fails at `x = 1e17f64`. There, `x + 1.0f64` rounds back to `x`.

`--tier auto` therefore does not stop at a solver proof. It samples the property,
or the derived producer obligation, afterwards to look only for a counterexample.
It reports the result in the additive `machine_arithmetic` field:

- `"status":"refuted"`: sampling found a concrete machine counterexample. The
  record is `"status":"failed"` with `"composite_verdict":"failed"` and carries
  the counterexample. A machine value is a definite refutation, so it replaces
  the proof.
- `"status":"corroborated"`: every sampled input satisfied the property. The
  proof stands exactly as the solver produced it: the verdict is still
  `proven_modulo_real_arithmetic` and `real_arithmetic` is still in
  `qualifiers`, because sampling is not exhaustive.
- `"status":"unchecked"`: sampling established nothing, with a `reason`.
  The generator could not produce an input, or could not satisfy its preconditions.
  The proof is unchanged. A sampler limit is not evidence about the property.

The field is absent when no such check ran. An explicitly selected method
runs on its own. Thus `--tier smt-only` and `--tier induction-only` run no
machine-arithmetic check. An induction proof carries none either.

A result conditional on an asserted axiom such as `clarabel.qp.ideal_optimality`
also carries no check. This axiom idealizes the float solver on purpose.
Sampling there would contradict the assumption you declared, not the proof.

In plain text an `auto` run prints the disclosure rather than a sample ratio a
proof does not have:

```text
property: sq_nonneg -- proven_modulo_real_arithmetic (proved over the reals; machine arithmetic corroborated on 100 samples)
```

The check can find only counterexamples that the sampler generates.
It does not promise that every machine counterexample is reachable.
A counterexample that sampling would find cannot hide behind a solver proof.
Thus `auto` does not report such a pass where `--tier fuzz-only` reports a
failure. An induction proof is outside this check.

The two modes also differ when sampling is inconclusive. When sampling errors or
establishes nothing, `auto` reports its proof and `--tier fuzz-only` reports
that failure.

The check costs the sample budget. When all properties pass, a run takes
roughly as long as the solver and the sampler together. Use `--tier smt-only`
to pay for the solver alone. That run leaves the machine-arithmetic question
unchecked rather than answering it.

## A solver disproof is checked again at its own witness

The same real-arithmetic model applies when the solver *disproves* a property.
It exhibits a point in the real numbers where the property is false. Reporting
that as a failure would claim a defect at a machine input, which the solver has
not shown. No binary64 value equals an irrational, and an exact rational need
not be representable either.

For example, `1.0f64 / ((x * x) - 2.0f64)` is infinite over the reals at
`x = sqrt(2)`, so a property that asserts it is finite on `[1, 2]` is disproved
there. No binary64 value squares to exactly 2. The two doubles nearest
`sqrt(2)` give `x * x` of 2.0000000000000004 and 1.9999999999999996, so the
quotient is finite at every double in that range.

Chelis therefore rounds the witness to each parameter's declared width and
evaluates the property concretely, the same route a square-root domain witness
takes. It reports the outcome in the same `machine_arithmetic` field:

- `"status":"confirmed"`: every precondition holds at the rounded witness and
  the body is false there. The result is `failed` with that machine value as
  the counterexample. The verdict is a definite `failed` rather than
  `disproved_modulo_real_arithmetic`, because a concrete evaluation decided it.
- `"status":"unconfirmed"`: the witness is not a machine counterexample, with a
  `reason`. No machine value equals it, a precondition rejects it once rounded,
  or the body holds there. The result is then `unsupported`: there is no basis
  for a failure, because the only point the solver exhibited is not one; and no
  basis for a pass either, because the solver decided the property false over
  the reals and the check says nothing about any other input. Under
  `--tier auto` the run first samples the property, so a sampled counterexample
  is still reported as a definite failure; only a clean run is `unsupported`.
- `"status":"unchecked"`: the witness could not be read or evaluated, with a
  `reason`. The disproof is unchanged. An evaluation limit is not evidence
  about the property.

Do not read `unconfirmed` as "the property holds". `(x * x) != 11.0f64` on
`[3, 4]` is `unconfirmed`, because `sqrt(11)` is irrational and no `f64` equals
the witness. The property is nevertheless false at `3.3166247903554`, whose
square is exactly `11.0`. Sampling does not find it. `unsupported` is the
honest answer: the machine-arithmetic question is open, and `chelis prove` exits
`2` rather than `0` or `1`.

This check is one evaluation of the solver's own witness rather than a second
verification method, so `--tier smt-only` runs it too. A disproof conditional on an asserted
axiom carries no check, for the reason above.

In plain text an undecided record prints its reason, and no sample ratio:

```text
property unsupported: g_finite: disproved over the reals, and no machine counterexample established: <witness reason>. ...
```

A `failed` record prints neither a counterexample nor a reason, so a `confirmed`
or `unchecked` disproof discloses its check only under `--json`.

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
Chelis code. Build `chelis` with `clarabel-provider,smt` and declare the package
as a Reef dependency. Match the result of `solve` or a typed helper that returns
one direct `solve` call.

A property may attach `with contract = "clarabel.qp.ideal_optimality"`.
This uses the `Solved` primal as an ideal exact-real optimizer for that call's
`P`, `q`, `A`, `b`, and cones. The non-`Solved` arm must establish its claim
without that assumption. See the executable files in `examples/clarabel_qp/tests/`
and the commands in `packages/chelis-clarabel/README.md`.

The SMT lowering accepts fixed-size `f64` QP data with symbolic entries.
It also accepts pure helpers with named bindings and conditionals, and fixed
zero and nonnegative cone blocks. It proves the PSD premise for exact literal
`P` or a `B^T B` construction.

For another symbolic `P`, add
`with contract = "clarabel.qp.assume_psd"` to the same property. That explicit
premise applies to the resolved call's `P` and is listed as a separate axiom.
For example, `tests/ideal_symbolic_constrained_quality.ch` compares a feasible
baseline with the optimizer after the algorithm applies its step. It proves
that the baseline cannot score better, for every input that satisfies the
property's `where` condition.

Other cone families can still be used at runtime. Properties that need their
ideal contract report `unsupported`. A successful dependent result says
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

Before it reports property results, Chelis checks the manifest's generated-file
identity, source hashes, byte ranges, and property mappings. An edited
requirement or generated file produces an input error until the translator
regenerates the artifacts. The displayed requirement text comes from the
verified original file.

JSON records retain `source.file` for the generated file.
They add `source.requirement` with the requirement's ID, original file, line,
column, text, source hash, and byte span.

The provenance check leaves proof results intact. A counterexample remains a
counterexample, and a sampled result retains its sampling verdict. A valid
source mapping establishes where the proposition came from. The proposition
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
| `machine_arithmetic` | What machine arithmetic established about a solver result. For a proof under `auto`: sampling `refuted` it, `corroborated` it, or left it `unchecked`. For a disproof: its witness was `confirmed` at a machine input, `unconfirmed` as one, or `unchecked`. Absent when no such check ran. |

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

For a checked scalar source call that takes a constructed record, SMT substitutes
the record's fields into the function body before it lowers arithmetic.
A square root enters SMT only after a separate proof establishes that its
argument is nonnegative from independent, total preconditions. The solver then
uses the unique nonnegative root of `q*q = argument`. A missing domain proof
or an inconclusive solver result stays `unsupported`.

When the solver shows the domain obligation **false**, it has a witness.
At that input, the square root takes a negative argument.
The input satisfies the conjuncts that the domain proof was allowed to use.
Those conjuncts form only part of the property's preconditions, and the solver
reasons over real arithmetic. Thus the witness is a candidate, not a result.

Chelis rounds the candidate to each parameter's declared width and evaluates it
concretely. If every precondition holds and the property body is false, the
result is `failed` with that machine value as the counterexample.
The verdict is a definite `failed` rather than `disproved_modulo_real_arithmetic`,
because a concrete evaluation decided it.

If the witness violates a precondition that the domain proof did not use, or
the body holds there, the result stays `unsupported`.
[The executable record example](https://github.com/Chelis-Lang/chelis/blob/v0.21.0/examples/record_sqrt_property.ch)
shows this route.

The command exits `0` when all reported checks pass, `1` when a property or obligation fails,
`2` when one is unsupported, and `3` for an error. When results differ, an error takes
precedence, then a failure, then an unsupported result. An input or setup error can stop the
run before a JSON summary is written; check the exit code and stderr as well as stdout.

For a contract-bound property, a subprocess failure in the configured external
contract checker reports a branded `unsupported` result and exits nonzero. A
failed checker cannot turn into a sampled pass.

# Chelis Property Specification

## Purpose

`chelis prove` is the Level 2 executable-property verifier. It discovers
first-class `@property` declarations, generates deterministic samples from
their binder types, filters samples through `where` preconditions, evaluates
the property predicate, and reports pass/fail/unsupported/error records.

Properties desugar to ordinary callable `bool` definitions. Deep does not gain
a new tag.

## Surf Syntax

Canonical v1 form:

```chelis
@property call_price_non_negative forall(
    s: f32, k: f32, r: f32, sigma: f32, t: f32
) where s >= 0.0, k > 0.0, sigma > 0.0, t >= 0.0:
  call_price(s, k, r, sigma, t) >= 0.0
  with tolerance = 1e-6
  with seed = 0
  with samples = 100
  with contract = "std.normal_cdf.reflection"
```

Rules:

- Property names are required Surf value identifiers and are unique in the
  containing module's value namespace.
- Binder types are explicit-only in v1. Omitted binder types are parse errors.
- `where` clauses are harness filters. False-precondition samples are not
  counted and the predicate is not called for them.
- `with tolerance`, `with seed`, and `with samples` are optional.
- `with contract = "..."` is a repeatable string-literal dependency on a
  standard contract invariant. Unknown contract IDs make the property
  unsupported.
- Numeric suffixes in property names have no harness semantics.

Contract options are part of Tier B lowering, not only report metadata. A
contract-bound call to the trusted implementation is replaced by a fresh SMT
symbol and the contract assumptions needed for that symbol. The prover must
check that the call resolves to the implementation named by the discharge
record before applying the abstraction. For `std.normal_cdf.reflection`, the
lowering recognizes syntactic `normal_cdf(x)` / `normal_cdf(-x)` pairs and
asserts the reflection coupling between their fresh symbols.

For `std.quantile.monotonicity`, the trusted implementation is the resolved
Reef dependency declaration for `Nautilus.Stats.quantile_vec`, whose linker
symbol is `pkg__nautilus__Nautilus__Stats__quantile_vec`. Trust comes from the
linker's dependency-owned declaration partition, not from parsing source names
or accepting a root-package lookalike. Lowering intercepts the call before the
generic scalar-argument pass: the tensor operand stays as compiler AST
identity, while the scalar quantile level lowers to SMT. Only two calls over
the same dataset identity receive the relational monotonicity assumption.
Missing trusted calls, different-dataset pairs, and the not-yet-bridged range
or boundary contracts return `unsupported` under `smt-only`.

The source-bound Clarabel QP contract and the design for extending it to
symbolic fixed-shape inputs are in [Clarabel as an external optimizer](clarabel_external_optimizer.md).

The chelis#979 acceptance oracle is:

```sh
cargo test -p chelis-cli --features smt --test issue_979_nautilus_quantile
```

### General-n structural induction

`--tier induction-only` is a Surf-only, fail-closed deductive lane. The prover
selects an `int*` induction binder from the checked compiler AST, never from a
caller classification. The v1 accepted shape has an explicit `n >= 0` domain,
one scalar model call in the proposition, and one directly recursive model:
`if n <= 0 then base else step`, where `step` contains exactly one
`f(n - 1, unchanged_args...)` call. Contract abstractions, non-transparent
mutual or non-structural recursion, multiple model calls, uninterpreted
residual calls, and other shapes are `unsupported`; this lane never falls
through to sampling.

“Direct” is measured after the compiler's ordinary bounded helper inlining, so
a type-checked transparent alias may expose the same exact recurrence. This is
intentional: argument substitution and symbol ownership come from the compiler
AST, and the resulting base/step goals are identical to the unaliased form.
A syntactically exact decreasing call under a literal-dead branch is also
accepted. The classifier does not erase that branch: the full `if` remains in
the dispatched goal, so the solver proves its unreachability and no dead call
can manufacture an induction hypothesis or a green case.

The prover constructs a concrete `P(0)` obligation from a full one-step model
unfolding and a symbolic `P(k) => P(k + 1)` obligation whose induction
hypothesis replaces only the exact `f(k, unchanged_args...)` subproblem. Both
goals, including their branch conditions and non-vacuity checks, are dispatched
separately to the existing SMT engine. A green result requires both discharges.
Machine records use `proof_tier:"induction"`, `arith_model:"real"`, and
`induction:{variable,base:{status,arith_model},step:{status,arith_model}}`.
`ASSUMED`, missing, sampled, unknown, timed-out, or vacuous cases cannot produce
a pass. The executable acceptance oracle is:

```sh
cargo test -p chelis-cli --features smt --test issue_978_induction
```

The default `--tier auto` ordering is induction, then the existing Tier B SMT
and Tier C fuzz lanes. Auto enters induction only when the checked Surf AST
shows that the property reaches a recursive model. An accepted induction plan
is terminal whether its base/step proves, disproves, times out, or errors; an
unsupported recursive structure is likewise terminal and reports
`proof_tier:"induction"` with zero samples. This prevents recursive general-n
claims from reaching finite sampling or overflowing the evaluator stack.
Properties that do not reach a recursive model retain the existing Tier B then
Tier C behavior. No auto-induction result may contain `ASSUMED` evidence or a
sampling record.

### Scalar gradient goals in Tier B

Tier B lowers an applied scalar gradient into the same real-arithmetic
obligation language as an ordinary scalar property. The supported v1 shape is
`grad(f, wrt=x)(args...)`, with exactly one explicit `wrt`, where `f` is an
inline lambda or pure top-level function whose parameters and result are
`f32`/`f64`. The differentiated body may contain scalar literals and variables,
negation, `+`, `-`, `*`, `/`, named scalar block bindings, and recursively
inlined pure scalar helpers with `f32`/`f64` results within the normal Tier B
inlining-depth bound. Argument substitutions are resolved in the caller's
scope before differentiation.

This is a prover-owned symbolic dual lowering; it must agree with Chelis scalar
AD semantics but does not replace the compiler's `grad` transform. Its SMT
verdict retains the `real_arithmetic` qualifier. Multi-target or implicit
`wrt`, tensor/ADT gradients, non-floating results, conditionals, casts in
differentiated bodies, effects, recursion, nested transforms, helper-inlining
depth overflow, unsupported intrinsics, and malformed calls do not silently sample under
`smt-only`: they return
`status:"unsupported"` with a reason naming the scalar-gradient capability
boundary. Under `auto`, the same boundary may continue to Tier C fuzz
validation. In particular, conditionals remain outside this prover-owned
subset until the compiler's scalar AD transform can build the same programs.
Float casts in differentiated bodies follow the same executable-parity rule.

## Deep Representation

Desugaring emits a `defsig` plus an ordinary property-tagged `def`:

```deep
(defsig {} call_price_non_negative
  (t-fn {} (t-prim {} f32) (t-prim {} bool)))

(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (s {type: (t-prim {} f32)})),
      property_preconditions: (tuple {} ...)}
  call_price_non_negative
  (fn {} (params {} (s {type: (t-prim {} f32)})) ...))
```

Required metadata:

- `chelis_role: "property"`
- `property_source_kind: "user"` or `"bridge:c-earchin"`
- `property_quantifiers: (params {} ...)`
- `property_preconditions: (tuple {} ...)`

Optional metadata:

- `property_source_id`
- `property_tolerance`
- `property_seed`
- `property_samples`
- `property_contracts: (tuple {} "contract.id" ...)`

The Deep `def` name is the property name. There is no `property_name` or
`property_predicate_body` metadata. Def parameters are the canonical binder
source; `property_quantifiers` is required as a serialization aid and must match
the parameter list.

Legacy c-earchin Deep compatibility:

- Property discovery requires canonical `chelis_role: "property"` and its
  required schema. `c_earchin_role` is opaque producer data, not a discovery
  synonym (`spec/03-deep-syntax.md` [03-META-3]).
- c-earchin v0.2 and v0.3 emitted canonical metadata alongside
  `c_earchin_*` producer metadata. Those artifacts retain Deep discovery.
- c-earchin v0.4 emits typed Surf properties with versioned source spans.
  The structured property JSON adapter retains the historical `PropertySpec`
  input schema; the Surf pipeline uses native property discovery.
- The witness `def` name is the property name. `property_source_id` carries the
  EARS ID.

### Canonical numeric values and typed input domains

Surf Tier B prepares one owned semantic declaration view per property run.
The complete declaration set first desugars to canonical Deep, resolving all
§5.6 declaration sites and standalone signatures. Semantic resugaring retains
every literal dtype instead of stripping suffixes for source presentation.
Direct goals, induction numerical plans, guard plans and helper substitutions
use that same immutable view; authored declarations still decide induction
admission and recursion routing;
leaf-by-leaf desugaring cannot reconstruct a declaration's expected dtype.
Original declarations still own property discovery paths, source attribution,
displayed goals, concrete evaluation and trusted dependency membership.
Semantic-view preparation does not add inference or change source formatting.
Canonical reconstruction retains callable parameter identity through `grad`
and validates every inner and outer selector. Valid nested transforms reach
their existing property-local capability boundaries without changing supported
numerical siblings or their guard plans.

Tier B interprets numeric literals through the canonical Surf-to-Deep literal
boundary (`spec/03` §6.4, `spec/04` [04-LIT-2], §5.6). A checked cast that
states an unsuffixed literal's numeric dtype denotes that literal, including
its signed value and one target-width finalization. Direct and substituted
arithmetic lowering and scalar-gradient literal coefficients share this
interpretation. Integer payloads remain exact
integers; float literals round at their stated dtype before entering the
ideal-arithmetic goal. The parser's signed-i64-minimum sentinel contributes
one minus sign. This rule does not erase a conversion of an already-bound
value, a named cast, or a cast to a dtype binder.

A property's typed integer inputs range over their representable value sets
(`spec/04` §4.6, [04-NUM-8], §9.1). Both Surf and Deep Tier B goals constrain
each signed integer input to its exact minimum and maximum, using the type
system's `Prim::integer_range`. The SMT sort remains unbounded `Int`;
intermediate arithmetic retains the ideal-arithmetic model and its existing
qualifier. Inlining inherits the property inputs' domains; it does not add
range assumptions for computed helper arguments or results. Intrinsic input
domains are type semantics, not authored `where` assumptions. They participate
in satisfiability/non-vacuity checks without creating a user-assumption record;
non-vacuity evidence distinguishes their count from authored assumptions.

## Generation And Evaluation

V1 generators support:

- `bool`
- `i32`, `i64`
- `f32`, `f64`
- `string`
- fixed-shape numeric tensors with literal dimensions and `f32`/`f64` elements,
  e.g. `tensor[3, f32]` and `tensor[2, 3, f64]`

Unsupported in v1:

- symbolic tensor dimensions
- tensor rank greater than 2
- ADTs and records
- higher-order binders
- state, capability, or effect assertions

Unsupported selected properties exit `2`. Generator exhaustion exits `3`.
Default max attempts are `samples * 100`; `--max-attempts` overrides this.
Default samples are `100`; `with samples` and `--samples` override it. Default
seed is `0`; `with seed` and `--seed` override it. Runs with the same seed and
inputs must produce the same sample sequence.

For a guarded property whose binders are all `f32`/`f64`, Tier C derives a
sampling domain from conjunctions of scalar interval and binder-order
comparisons (`<`, `<=`, `>`, `>=`), plus `binder != literal` exclusions.
Constant bounds propagate through binder
orders before sampling, so narrow guards such as
`0.99 < alpha1 < alpha2 < 1.0` are generated in-domain rather than discovered
by rejection from `[-10, 10]`. Explicit bounds are not clipped to that legacy
uniform range; negative literals and reversed comparison spellings are
equivalent interval bounds. Strict spacing is computed with the binders'
actual IEEE `f32`/`f64` successor and predecessor values. Non-strict order
edges permit equality and reserve no strict spacing, while strict chains
reserve enough representable values for their remaining successors. The
construction and its random choices are seed-deterministic. A one-sided finite
interval chooses its missing endpoint within the binder dtype's finite range;
the synthesis clamps at `f32::MAX`/`f64::MAX` rather than overflowing near an
IEEE extremum. A `binder != literal` guard keeps the binder's interval and
records the excluded value; a draw that lands on an excluded value moves to
the nearest representable neighbour still inside the interval. An empty interval,
an interval with too few representable values for its strict order chain,
an exclusion that empties a singleton interval,
cyclic ordering, disjunction, equality, `!=` between two binders,
arithmetic operand other than unary
literal negation, function predicate, or other unsupported guard shape is
`status:"unsupported"` (exit `2`) under `fuzz-only`; it cannot fall through to
a green empirical verdict. Non-scalar guarded properties retain their existing
typed generator and rejection behavior.

### The edge schedule (chelis#3636)

Which points a tier proposes decides what it can refute, and a draw taken
uniformly over an interval visits no exact representable singularity. Such a
draw alone cannot refute `1/(x+1) - 1/x` being infinite at `x = -1.0` strictly
inside `-2 < x < -0.5`, nor `x * x` overflowing at a magnitude outside the
`[-10, 10]` window, so it yields `fuzz_validated` for properties that fail at
ordinary inputs inside their own guards.

Tier C therefore proposes a **deterministic edge set before any random
draw**, in both the constraint-directed and the typed-generator lanes. The
run's zero-based attempt ordinal indexes the schedule: attempt `i` proposes
the binder's edge candidate `i`, and attempts past the end of the list fall
back to a random draw over the proposal window. The schedule governs only
which points inside the guards are chosen; the window's width is chelis#644.

Each binder's candidates are generated in this order, deduplicated by bit
pattern, and kept only where the dtype represents them inside the
**admissible range** — the guards clamped to the dtype's finite range, which
for an unguarded binder is `±MAX` rather than the proposal window:

1. the interval's lower and upper endpoints. For a binder the guards leave
   unbounded on a side, that endpoint is the dtype's finite extremum, so this
   class is what reaches `±MAX`;
2. their adjacent representable neighbours. Under a strict guard the endpoint
   already *is* the neighbour, so the two classes coincide there and only a
   non-strict guard distinguishes them;
3. for each `!=` exclusion, the excluded value's two neighbours;
4. `0.0` and `-0.0` (distinct: `1/0.0` and `1/-0.0` differ in sign);
5. `±` the smallest positive subnormal (not `MIN_POSITIVE`: `1.0/MIN_POSITIVE`
   is finite at both widths, so only the subnormal reaches overflow);
6. the small integers `±1`, `±2`, `±3`, `±10`;
7. a fixed ladder of decade magnitudes, geometric in both directions from
   `1e1` and ending at the `f64` extremes: `1e1, 1e3, 1e9, 1e27, 1e81, 1e243,
   1e308` and `1e-1, 1e-3, 1e-9, 1e-27, 1e-81, 1e-243, 1e-308, 1e-323`, each
   taken with both signs. The rungs are correctly rounded literals, not
   `10f64.powi(e)`, which drifts from the decade at 7 of those 15 rungs and
   returns exactly `0.0` for `powi(-323)`. `1e-323` is the floor because
   `1e-324` rounds to zero even as a literal. A rung outside a narrower
   dtype's range clamps onto that range's extremum and is then dropped as a
   duplicate of the endpoint, so no rung contributes an infinity.

An integer binder in the typed-generator lane has its own schedule: the ends
of the width's fuzz range, `0`, `±1`, `±2`, and each end's inward neighbour.
Widening that range is chelis#644.

The ladder is a fixed list and not a log-uniform random draw, so "this
magnitude class is covered" stays a claim a test can pin rather than a
probability. Nothing in the schedule depends on the seed: a property refuted
at an edge point is refuted under every seed, and a budget smaller than the
schedule spends what it has on the earliest classes rather than on uniform
draws. The schedule is part of the sample sequence, so a run remains
reproducible for a given seed and inputs.

Edge candidates are proposals, filtered exactly as random ones are: the
precondition still runs, so a rejected edge point is a rejected sample and
consumes its attempt. `rejected_samples` counts every attempt that produced no
accepted sample, which includes an attempt whose draw could not be resolved as
well as one whose draw the precondition refused; `accepted_samples` plus
`rejected_samples` equals `attempted_samples` in every record.

Because the schedule reads the admissible range while the random draw prefers
the narrower proposal window, an order predecessor can be drawn from outside
its successor's window: `forall(lo, hi) where lo < hi` can put `lo` at a
magnitude the window's width cannot reach, after which `hi`'s window admits
nothing. **The window is a preference, not a constraint.** A binder whose
window admits no value, given the predecessors already drawn, draws from its
admissible range instead.

An empty *admissible* range is a different matter. Whether it is a fact about
the attempt or about the property decides the verdict, and **that is observed
rather than predicted.** Nothing in the plan's shape decides it: a binder
pinned onto a `!=`-excluded value by its order predecessor recovers on the
next attempt only if the predecessor can itself take another value, which
depends on *its* guards and predecessors transitively. A rule reading one
level of that graph gets the pinned-predecessor case wrong.

So a draw that yields no value reports only that, carrying its reason, and the
attempt is rejected. When the budget ends with fewer accepted samples than
requested, the run reports:

- `status:"unsupported"`, with the recorded reason, when **no attempt in the
  whole budget ever placed every binder and at least one reported that it
  could not**. The guards admit nothing. Both halves are required: a budget
  that permitted no attempt at all yields no evidence either way, and claiming
  unsupportability from zero attempts would be a stronger statement than the
  run can support.
- `status:"error"` with the exhausted-generator reason when some draw did
  succeed. The domain is sampleable and the budget ran out -- a guard the
  proposal window starves (chelis#644) reaches this, and calling it
  unsupported would claim its guards admit nothing when they do.

A binder the typed generator cannot sample at all, such as a tensor outside
rank 1 or 2, is the one terminal case that is *not* an observation: the
sampler knows it from the binder's type, reports it immediately, and so
records `attempted_samples: 0`.

Derivation-time emptiness checks remain, and reporting
`attempted_samples: 0` is their entire purpose: the sampling loop reaches the
same verdict for the shapes they miss, only after spending the budget.

Where the window does admit a value it is always inside the admissible range,
because the synthesis narrows the same guards and both ranges take the same
predecessor and reserve adjustments. The fallback therefore needs no
intersection: an unresolvable window is an empty one, which is exactly the
case the admissible range covers. That invariant is asserted in debug builds
rather than assumed, since a later change to the window synthesis could break
it with no visible symptom.

One known limitation: because every binder indexes its schedule by the same
attempt ordinal, a guard the typed-generator lane can only enforce by
rejection discards the other binders' candidates at the same ordinals. A
property with an `i32` binder guarded `n > 0` loses the float binder's
candidates at the attempts whose integer candidate is non-positive. The lane
still proposes the rest of each schedule.

No remedy is available for that example today: the constraint-directed lane
admits a property only when every binder is `f32`/`f64`, so a property with an
`i32` binder cannot enter it. Widening that lane to mixed binders would be the
fix.

### Pairing the extremal classes (chelis#3697)

The shared attempt ordinal indexes every binder's schedule, so binders with
equal ranges take the same candidate at every scheduled attempt. On its own
that evaluates a multi-binder property only on the **diagonal** of the edge
grid: `forall(x: f64, y: f64) where x > 0.0f64, y > 0.0f64` never paired a
large `x` with a tiny `y`, and `x / y` being finite — false at exactly such a
pair — reported `fuzz_validated` at every sample count measured, up to 50000
with `accepted == attempted` throughout.

A **cell phase** now runs between the diagonal and the uniform draw. With
`S = 44` the span, `L_i` each scheduled binder's own schedule length, and
`W = max(S, L_0, ..., L_n)` the boundary the whole property shares:

1. **Diagonal**, for `attempt < W`: candidate `attempt` of this binder's own
   schedule, exactly as before and at the same one-for-one rate. A binder
   whose own schedule is shorter than `W` runs past the end of its list and
   spends the remainder on uniform draws, as it did before the phase
   existed.
2. **Cell**, for as many of the following attempts as the share below allows
   — a mixed-radix odometer over **two** classes per binder. For a float
   binder those are its largest and smallest admissible candidates by
   `total_cmp`, and digit 0 is the largest; for an integer binder they are
   the first two entries of its own schedule, which `int_edge_candidates`
   orders `lower` then `upper`, so digit 0 is the *smallest*. Either way the
   odometer enumerates the same `2^n` corners. Cell `c` gives binder `j`
   digit `(c / 2^j) % 2`.

   **The share is stated once, below**, under "the attempts after the
   schedule is spent are not spare": a published formula that lives in two
   places has now been wrong in one of them twice.
3. **Uniform**, thereafter: unchanged.

The phase is skipped entirely for a property with fewer than two scheduled
binders. A binder that reads
no schedule — `bool`, `string`, a tensor — contributes no digit and consumes
no position, so it neither enables the phase nor shifts another binder's
digit.

At two binders that gives four cells at attempts 44–47, and cell 2 pairs the
first binder's largest candidate with the second's smallest. Under `> 0.0`
those are `f64::MAX` and `5e-324`, so the issue's own property is refuted at
the forty-seventh attempt under every seed.

Each figure below was taken with `--tier fuzz-only` at the stated budget. They
are recorded because each of the policy's choices — two classes, the
placement after the diagonal rather than before it, the share that bounds the
cell phase, and one boundary shared by the property rather than per binder —
was chosen by a measurement that a plausible alternative failed. A fifth
choice, switching the phase off for an order-guarded property, was made the
same way and later withdrawn when the shared boundary removed the figure that
justified it.

#### Reindexing alone cannot pay for it

An index derived from `attempt` is a pure function of `attempt`, so a
mixed-radix odometer over the full schedule would be exactly as reproducible
and as seed-independent as the diagonal. The unguarded `f64` schedule is 44
candidates long, so at the default 100 samples the diagonal gives every binder
all 44 of its own classes while covering 44 of 1936 pairs, and a full-schedule
odometer gives the slowest binder 3 of 44 while covering 100 of 1936. Full
grid coverage needs 1936 attempts, and `44^n` for `n` binders, so **no index
policy covers the grid within a budget linear in the schedule length.** That
is why the phase pairs two classes rather than reindexing the schedule: the
choice is which coverage to keep, not whether a pair is reachable.

The weaker claim — that no index policy reaches *any* off-diagonal pair
cheaply — is false, and these figures refute it: a full-schedule odometer
reaches the pair (binder 1's candidate `a`, binder 2's candidate `b`) at cell
`a + 44b`, so the `x / y` counterexample above is its cell 1.

#### The attempts after the schedule is spent are not spare

The budget comes from the attempts past the schedule, which draw uniformly
over the narrow proposal window (chelis#644 owns its width). chelis#3636
exists because such a draw rarely finds an IEEE edge case, and it is tempting
to read that as those attempts being nearly worthless. **They are not.** They
are the only thing that finds a counterexample in the *interior* of the
window, where no edge candidate and no extremal representative lands. That
sentence was written as a rationale before it was measured, and the
measurement refuted it.

Measured with a property false on `(-7.5, -7.0)` — 2.5% of the window,
containing no candidate of any schedule — at the default 100 samples over
seeds 0 to 29:

| scheduled binders | no cell phase | two classes (shipped) | six classes |
|---|---|---|---|
| 1 | 20/30 | 20/30 | 20/30 |
| 2 | 24/30 | 23/30 | 11/30 |
| 3 | 26/30 | 24/30 | **0/30** |
| 4 | 23/30 | 21/30 | **0/30** |

Arity 1 is the control: a single binder takes no cell, so an unchanged rate
there is what isolates the effect to the phase. The first two columns were
re-measured against the shipped build and current `main`; the third cannot be,
because the code that defined six classes is not in the tree, and it is
retained because the `6` in the arithmetic below is otherwise unverifiable.

**This is the measurement that sets the class count at two.** A six-class set
— largest and smallest, smallest strictly positive and largest strictly
negative, a zero where the guards admit one, and the candidate nearest `1.0`
in magnitude — needs `6^n >= 216` attempts at three binders, which exceeds the
56 that follow the 44-attempt diagonal at the default budget. Every
post-diagonal attempt becomes a cell and the uniform generator never runs; the
loss is total and seed-independent rather than a reduced chance. Two classes
need 4, 8 and 16 cells at two, three and four binders, which fit inside those
56 with the generator still reachable.

**Two classes alone do not bound this, and a fixed cap cannot either.** `2^n`
outgrows any constant, so for every allowance there is an arity at which the
cells take every post-diagonal attempt — a cap of 256 reproduced the
six-class failure exactly, relocated from three binders to six, where
`2^6 = 64 > 56`. The quantity the allowance has to stay below is the budget,
which a constant does not know, so the allowance **is** a share of the
budget.

The quantity that gets divided is **the attempts this run may still make
once the property's shared window opens** — bounded below by the attempt
ceiling as well as the sample count, and measured from `W` rather than from
the span. `CELL_BUDGET_DIVISOR` in
`crates/chelis-prove/src/property_runner.rs` states the arithmetic, and is
the only place that does: this document names the quantity and points there,
because a formula kept in two places was wrong in one of them twice.

Each of those two refinements was a false pass of chelis#3697's own class.
Measuring from the span rather than from the binder's own diagonal end
starved the generator completely for a 72-candidate schedule at the default
budget — fourteen `!=` exclusions on each binder reach that, and four reach
it at `--samples 60`, so it is not a corner case. Ignoring the attempt
ceiling did the same whenever `--max-attempts` fell below the sample count.
With the right quantity divided, the uniform draw keeps at least half of the
attempts the diagonal left, at every arity, every budget and every schedule
length, and the grid is covered to whatever depth the other half allows.

Extending the same measurement past the published four rows, seeds 0–29 at
the default budget:

| scheduled binders | no cell phase | budget share (shipped) | fixed 256-cell cap |
|---|---|---|---|
| 5 | 21/30 | 11/30 | 9/30 |
| 6 | 23/30 | 15/30 | **0/30** |
| 7 | 23/30 | 11/30 | **0/30** |
| 8 | 23/30 | 19/30 | **0/30** |

The share does not restore the no-phase rate above four binders — half the
post-diagonal attempts are spent on cells by design — but the loss is a
reduced chance rather than the total, seed-independent loss a fixed cap
produced. The same holds in the constraint-directed lane — the same band property under
range guards `vi > -10.0f64, vi < 10.0f64` on every binder, which is what
routes it through `ConstraintSamplingPlan` — seeds 0–19 at arities 5/6/7:
15/18/16 with no phase, 9/17/14 with the share, 9/9/5 with the fixed cap.

**The share costs little pairing**, because the corners worth proposing sit
at low cell indices: a pair of opposite extremes on the first two binders is
cell 2 whatever the arity, so it is reached at attempt 47 at every arity from
two to eight while a run without the phase passes its whole budget. What the
share gives up is the corners that need several binders at once on their low
digit, since `2^5 = 32` exceeds the 28 cells the default budget allows. Each
binder's *own* extremal digit is still reached at arity five — cell 16 is
under 28 — and it is from arity six that a binder's own digit (cell 32) falls
outside the prefix.

A policy that widens the class set, or the share, owes these tables a
re-measurement at the arity it targets, and a negative control that a
window-interior refutation still happens there.

#### The constraint-directed lane has no stable per-binder diagonal

`ConstraintSamplingPlan::sample` recomputes each binder's
`float_edge_candidates` **inside the attempt**, from the order predecessors
already drawn in that attempt. So an index into that list does not name a
fixed value: it names entry `i` of a list whose contents depend on what the
predecessor got. Changing a predecessor's value re-indexes every successor,
and entries the schedule proposed at attempt `i` can end up proposed at no
attempt at all — a predecessor taking its largest candidate pushes `y > x`
into a non-negative range and deletes every negative candidate from `y`'s
list.

**That is a reason to keep the cells clear of every diagonal, not a reason to
withhold them from the property.** The phase was switched off for an
order-guarded plan at first, and the measurement that justified it was taken
under a *per-binder* window: on `forall(x: f64, y: f64) where y > x, <twenty
`!=` exclusions on y>: (y != -1.0e-323f64)` at the default budget over 30
seeds, 18/30 refutations with the phase off against 0/30 with cells proposed
everywhere — that 0/30 was taken against a six-class build and is not
re-measurable, and a partial exemption withholding cells from predecessors
only measured 11/30 under the same build.

The cause was mechanical rather than deep, and the re-measurable part of it
is this: twenty exclusions make `y`'s schedule 84 candidates long and the
refuting candidate sits near index 75, but a per-binder window opened `x`'s
cells at attempt 44, displacing `y`'s draw before its diagonal reached it.
Reproducing just that — two classes, the current share, a per-binder window
and no exemption — measures **10/30**, so the window accounts for eight of
the eighteen lost refutations and the rest belonged to the six-class radix.

One shared boundary removes that, because it clears the *longest* schedule in
the property. A successor's exact length is not derivable at plan-derive time
— it depends on a value drawn inside the attempt — but its bound is: a
schedule is the exclusion-free list, at most `S` long, plus at most two
representable neighbours per `!=` guard, and a boundary larger than a
binder's real length costs nothing, because the binder runs past the end of
its list and draws uniformly as a short schedule already does. With the
boundary clear of `y`'s diagonal the same chain measures **18/30 with the
phase on, identical to the phase being off, seed for seed and attempt for
attempt.**

So an order-guarded property now pairs. What it pays is displacement, which
is the trade the phase makes everywhere and is bounded the same way: the
cells take at most half the attempts the diagonal left, and the attempts
before the boundary are untouched, so the only refutations at risk are the
ones a run without the phase found in its final `cells` attempts.

**That bound, not a comparison of figures, is why the share is not reduced
further for an order-guarded property.** The loss is budget-dependent and
recovers: two disjoint edges at arity 4 measures 24/30 against 16/30 at the
default 100 samples, and 29/30 against 29/30 at 200 and 30/30 against 30/30
at 500.

The figures themselves are worth stating carefully, because the obvious
summary of them is wrong. Over seeds 0–29 at the default budget, ten order
shapes at arity two to four lose between 0 and 8 refutations of 30, mean
−1.9. But that sweep stops at arity four, and it does not extend: an arity-5
single-edge shape measures 24/30 against 11/30, and an arity-6 two-edge shape
21/30 against 9/30. Nor is a 30-seed delta a rate — the arity-4 worst row is
77/100 against 61/100 over a hundred seeds, which is −4.8 per 30 rather than
−8, while an arity-6 chain is −10.5 per 30 and is the worst stable loss
found. A policy that widens the cell phase owes these figures a
re-measurement at the arity it targets, over enough seeds to separate a rate
from an excursion.

#### What the phase costs

Two classes are the two extremes, so a counterexample needing an extreme
paired with a **moderate** value is not reached off the diagonal.
`x >= 1e300` with `y == 1.0` is such a pair: `1.0` is a candidate of `y`'s
schedule but not one of its two classes, and no uniform draw over the window
proposes `1e300`. The six-class set did reach it — "nearest `1.0`" was one of
the six — and the trade was made against the table above. An order-guarded
property pays displacement for its pairing rather than going without, which
the subsection above measures.

A third cost existed and is closed. **The cell window opens at one boundary
shared by the whole property** — `max(44, L_0, ..., L_n)` over the binders'
own schedule lengths — rather than at each binder's own `max(44, L)`. A
per-binder window was as wide as the cell count and therefore disjoint
between binders whose lengths differed by that much: at two binders, four
cells and two candidates per `!=` exclusion made **two exclusions on one
binder** enough to lose every pairing, at 500, 2000 and 20000 samples alike.
Sharing the boundary makes that unreachable rather than unlikely. The corner
is then cell 2 of the shared window, reached at attempt `47 + 2E` for `E`
exclusions on either binder, and the deliverable, the long-schedule shape,
the order-guarded chain and both arity tables are unchanged attempt for
attempt at the default budget.

It is not free at every budget, and the exception is worth stating because it
looks like a regression and is not. At exactly `--samples 50` and `51`, with
an offset of one exclusion, a run reaches the corner *without* the shared
boundary and not with it: the boundary rises from 44 to 46, so the share
allows two cells and cell 2 falls outside the prefix. What the unshared
window hit there was not an odometer corner at all — it was one binder in its
cell 0 while the other was in its cell 2, which the windows only made
possible by being misaligned. With the lengths equal, an unshared window
passes at those budgets too. So the aligned behaviour is the designed one and
the loss is the accident going away; it is confined to those two budgets and
to an offset of one, since reaching it needs `2E` under the cell count.

One cost remains, and it is asserted as a test that a property is *not*
refuted, so a future policy that reaches it has to change that test
deliberately and re-measure the displaced-draw rates. The two closed ones are
asserted the other way round: a pairing that must happen at every offset, and
a pairing that must happen under an order guard.

A binder whose schedule is longer than the span keeps the diagonal until that
schedule is spent, which is what the `L_i` terms of `W` are for: a boundary
fixed at the span displaced the tail of a long schedule past the budget, and
one taken per binder displaced it for every binder but the longest. An `!=` exclusion adds
**at most** two candidates — its two representable neighbours — so
`L <= 44 + 2k` for `k` exclusions. It is only a bound because `push` drops a
value the exclusion list contains, so an exclusion that is itself a schedule
member removes itself first: `x != 1.0f64` measures `L = 45`, not 46, and
`x != 0.0f64` measures **42**, because `0.0 == -0.0` in Rust deletes both
zeros while the neighbours it would add are already the subnormal candidates.

The shared Tier-C runner is enabled in every normal CLI build, including a
build without the `smt` feature, and is also the Tide implementation. Solver
availability may change `auto` dispatch into Tier B, but it never changes
`fuzz-only` generation or its machine record.

The authoritative chelis#977 acceptance oracle is:

```sh
cargo test -p chelis-cli --test issue_977_constraint_fuzz
cargo test -p chelis-tide --test mcp issue_977_tide_and_cli_match
cargo test -p chelis-cli --features smt --test issue_977_constraint_fuzz
```

The chelis#3636 edge-schedule oracle is:

```sh
cargo nextest run -p chelis-cli --test issue_3636_edge_sampling
```

The chelis#3697 extremal-pairing oracle is:

```sh
cargo nextest run -p chelis-cli --test issue_3697_extremal_pairing
```

It pins the pairing itself, the one cost above, the retirement of the other
two, and the measurements that
chose the policy: the post-diagonal draws' yield at arities one to eight, the
same yield at a 72-candidate schedule, a long schedule's tail, the low-order
corner's reach at every arity, and the rejection lane's accepted count. The chelis#3636
oracle three lines above no longer asserts the diagonal limitation —
`issue_3636_multi_binder_schedule_visits_only_the_diagonal` asked to be
deleted when the pairing improved, and was — but it keeps
`issue_3636_the_diagonal_witness_is_real_when_the_guards_align_it`, which
separates a pairing regression from a magnitude-reach one.

V1 reports the first deterministic counterexample. Shrinking is deferred.

## CLI Contract

```sh
chelis prove
chelis prove properties/
chelis prove src/pricing.ch
chelis prove references/pricing_rules.dp --spans references/pricing_rules.spans.json
chelis prove --only call_price_*
chelis prove --samples 1000 --seed 42 --max-attempts 100000
chelis prove --json
```

Discovery:

- No path: scan current Reef package `properties/**/*.ch` and `src/**/*.ch`.
- Exclude `tests/`, hidden directories, `target/`, `dist/`, and dependencies.
- Explicit `.ch` files scan only those files for properties.
- Explicit `.dp` files validate Deep first, then scan property metadata.
- Imports are followed for name resolution, not property discovery.
- Dependency properties do not run unless explicitly targeted.

Span lookup:

- For each `.ch` or `.dp` input, discover sibling `<stem>.spans.json`.
- `--spans <path>` overrides discovery for one explicitly selected source file.
- A present or explicitly selected manifest is verified before any selected
  file emits property verdicts. Invalid manifests are input errors (exit `3`).
- Surf mappings use the property name in source. Reef linker names are restored
  to that name before attribution. `--only` filters execution, not validation
  of the complete manifest against the generated file's discovered properties.

### Verified property spans

The versioned manifest uses `schema_version: 1`, with these required fields:

| Field | Contract |
|---|---|
| `artifact` | `{path, sha256}` identifying the selected generated source file |
| `sources` | Array of distinct `{path, sha256}` source identities |
| `requirements` | Array of `{id, source, span, clauses}` records with distinct nonempty IDs |
| `properties` | Array of `{name, requirement_id}` mappings covering the discovered properties |

Each `span` has `start_byte` and `end_byte`, a nonempty half-open byte range
within UTF-8 source. Each clause has a distinct nonempty `kind` and a `span`
contained in its requirement's span. A requirement's `source` names an entry
in `sources`. Recorded requirements may have no property mapping. Each mapped
property names an existing requirement. Duplicate property names, unknown
names or IDs, unknown fields, and unsupported schema versions are errors.

Paths resolve relative to the manifest directory; absolute paths are also
accepted. Source identities must resolve to distinct files. A checksum has
the form `sha256:` followed by the lowercase hexadecimal SHA-256 digest of
the exact file bytes, including line endings. All source checksums and the
artifact identity and checksum must match. The selected artifact is resolved
once, so a producer may publish immutable bundles through an atomic symlink.

The consumer derives requirement text, line, and byte column from verified
source bytes. It does not accept independently supplied display text. Lines
and columns are one-based. Text and NDJSON diagnostics retain generated-file
identity and add original requirement identity and location. Attribution does
not change proof engines, assumptions, evidence, or verdict precedence.

Deep also accepts the legacy c-earchin manifest with `source`, `source_hash`,
and `spans`. Its paths resolve from the invocation directory, matching that
producer's path convention. Source hashes and byte spans are verified; stored
text and line/column fields do not override the verified source. The legacy
format has no generated-artifact checksum; the consumer validates its mapping
against the discovered Deep properties. Surf requires the versioned format.

Exit codes:

- `0`: all selected runnable properties passed.
- `1`: at least one runnable property failed.
- `2`: at least one selected property is unsupported in v1.
- `3`: setup/config/input error.

Precedence is `3 > 2 > 1 > 0`.

A module that does not type-check is an `Error` (exit `3`): `chelis prove`
surfaces the check diagnostics and never reports success on a type-broken
module. Derived producer obligations cannot be meaningfully verified
without the checker-inferred signatures, so a rejectable producer must not
be hidden behind an unrelated type error. Strict downstream prove-compat
admission relies on this: a `prove` that exits `0` warrants
that the module type-checked and every obligation was discharged.

## JSON Output

`--json` emits NDJSON. Each selected property emits one record, followed by one
summary record. Property and obligation `name`, `goal`, `reason` and check
`diagnostics` fields use source spellings under `spec/04-type-system.md` §2.5.
The linker-owned source map renders package declarations at the report
boundary: the selected module uses bare names and imported modules use
qualified names. Proof identities and the dependency graph retain their
existing representation. Tide accepts source text and uses the same authored
spellings for that source; its interface does not load a package by path.

```json
{"kind":"property","name":"confidence_tail_order","status":"passed","composite_verdict":"fuzz_validated","qualifiers":["fuzz","fuzz_base"],"assumptions":[{"name":"preconditions:confidence_tail_order","discharge":{"method":"fuzz","evidence":{"status":"validated","sampling_method":"constraint_directed","accepted_samples":100,"attempted_samples":100,"rejected_samples":0}},"non_vacuity":{"status":"established","evidence":{"method":"fuzz","sampling_method":"constraint_directed","accepted_samples":100,"attempted_samples":100,"rejected_samples":0}}}],"proof_tier":"fuzz","sampling_method":"constraint_directed","accepted_samples":100,"attempted_samples":100,"rejected_samples":0,"samples":100,"seed":0}
{"kind":"property","name":"req_PRC_001","status":"failed","composite_verdict":"failed","assumptions":[],"samples":1,"seed":0,"source":{"kind":"bridge:c-earchin","spans":"references/pricing_rules.spans.json"}}
{"kind":"property","name":"tensor_symbolic_shape","status":"unsupported","composite_verdict":"unsupported","assumptions":[],"proof_tier":"none","reason":"symbolic tensor dimensions are not supported in L2 v1"}
{"kind":"summary","total":3,"passed":1,"failed":1,"unsupported":1,"errors":0,"dependency_graph":{"status":"complete","declarations":[{"id":"decl:…","name":"call_price","kind":"function","package":"pricing","module":"Pricing.BlackScholes","source":{"file":"src/black_scholes.ch","span":{"offset":42,"len":180}}}],"edges":[{"from":"decl:…property","to":"decl:…"}]}}
```

The summary's `dependency_graph` is the compiler-owned declaration ownership
wire (chelis#922):

- `status:"complete"` means linker analysis ran. Empty `declarations` and
  `edges` arrays are a complete empty result, not missing analysis.
- `status:"unavailable"` carries a `reason` and never carries a guessed partial
  graph. Bare Surf files have no stable Reef package/module identity, and Deep
  inputs do not carry compiler-owned source-file ownership, so both are
  unavailable.
- A declaration `id` is the deterministic
  `(package,module,kind,author-facing-name)` identity. Body and span edits keep
  the ID stable; a rename changes it. Every node also carries the package,
  module, declaration kind, and package-relative source file plus byte span.
- Edges are stable-ID `from`/`to` pairs derived from Reef's linker-resolved Surf
  AST. Consumers must not reconstruct ownership by parsing source. The graph
  covers functions, values, properties, types, constructors, aliases, macros,
  and each module-level dimension declaration; type/invariant/macro references
  participate alongside value references.
- Every root-package declaration is present, including unused declarations.
  Referenced dependency-package declarations are included transitively. Linker
  identity preserves same-name declarations, lexical shadowing, imports, and
  cycles without name guessing.
- A multi-input summary is `unavailable` if any selected input lacks complete
  attribution; Chelis does not present a partial union as complete.

Every Tier-C property record additively reports `sampling_method`,
`accepted_samples`, `attempted_samples`, and `rejected_samples`. Guarded
properties repeat those counts and the method in their precondition discharge
and non-vacuity evidence, so a consumer can distinguish empirical evidence
from an exhausted or unsupported generator without reconstructing it from
source. CLI and Tide user-property records rendered by the shared runner always
carry `proof_tier`; terminal outcomes report the explicit value `none` rather
than encoding it as field absence.

The legacy name-only `dependency_edges` array remains additive and deprecated
for at least one published release after `dependency_graph` is introduced.
New consumers use only `dependency_graph`.

Every `{kind:"property"}` and `{kind:"obligation"}` result record carries:

- `composite_verdict`: the single WEAKEST badge token, one of `proven`,
  `proven_modulo_real_arithmetic`, `proven_modulo_fuzz_validated_contract`,
  `proven_modulo_asserted_axiom`, `sound_approximate`, `fuzz_validated`,
  `invalid`, `unsupported`, or `failed`.
  `status:"passed"` remains the compatibility bucket; consumers that need
  proof strength must read `composite_verdict`. A fuzz-validated result must
  not render as `composite_verdict:"proven"` or any `proven_*` badge: a
  property whose BASE was established by fuzz sampling only (no SMT proof
  underneath -- a non-smt build, or a `--tier auto` fuzz fall-through, or
  `--tier fuzz-only`) renders `fuzz_validated`, a green-exit empirical pass
  that is not proven. A base discharged by a sound over-approximation (e.g. an
  interval engine) renders `sound_approximate`. An SMT proof is over the reals
  (see the Tier B caveat below), so under the current real-sorted lowering an
  SMT green renders `proven_modulo_real_arithmetic`, disclosing the
  machine-arithmetic gap; plain `proven` is reserved for a future
  exact-machine-arithmetic lowering. The
  `proven_modulo_fuzz_validated_contract` badge is reserved for an exact SMT
  base discharged modulo a fuzz-validated CONTRACT assumption -- the base
  itself is proven there, only a contract is fuzz-validated.
- `qualifiers`: an array of the FULL disclosed caveat set as snake_case
  strings (a green's union of every contributing qualifier), e.g.
  `["fuzz","real_arithmetic"]` for an over-reals proof modulo a fuzz contract.
  The `composite_verdict` token is the weakest single badge; `qualifiers`
  carries every caveat so a consumer sees them all. A non-green outcome carries
  an empty array.
- `assumptions`: an array of assumption records. Each record has `name`,
  optional `source_type` / `producer`, optional
  `discharge:{method:"smt"|"fuzz"|"axiom", evidence:{...}}`, and optional
  `non_vacuity`. A missing or failed discharge degrades the composite verdict.

### Derived obligation records (additive — `opaque_invariants_rfc.md` D-OBLIG)

In addition to `{kind:"property"}` records, `chelis prove` emits one
`{kind:"obligation"}` record per derived producer obligation of an
invariant-carrying opaque type, and the `{kind:"summary"}` record gains an
`obligations` field. These are **additive**: existing `kind:"property"` and
`kind:"summary"` records and fields are unchanged in meaning, and a strict
downstream admission parser must add `kind:"obligation"` to its accepted
record set deliberately at pin time — it is a new record kind, not a change
to an existing one.

```json
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"probability","name":"invariant:Probability:probability","status":"passed","composite_verdict":"proven_modulo_real_arithmetic","qualifiers":["real_arithmetic"],"assumptions":[{"name":"invariant:Probability:probability","source_type":"Probability","producer":"probability","discharge":{"method":"smt","evidence":{"status":"proved","obligation":"invariant:Probability:probability","arith_model":"real"}},"non_vacuity":{"status":"established","evidence":{"solver":"cvc5","result":"sat","assumption_count":0,"trivial":true}}}],"proof_tier":"smt","samples":0,"seed":0,"arith_model":"real"}
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"bad_prob","name":"invariant:Probability:bad_prob","status":"failed","composite_verdict":"failed","assumptions":[{"name":"invariant:Probability:bad_prob","source_type":"Probability","producer":"bad_prob","discharge":{"method":"smt","evidence":{"status":"failed","obligation":"invariant:Probability:bad_prob","counterexample":{"__arg0":"2.0"}}}}],"proof_tier":"smt","samples":0,"seed":0,"arith_model":"real","counterexample":{"__arg0":"2.0"}}
{"kind":"obligation","obligation_kind":"invariant_producer","status":"error","composite_verdict":"unsupported","assumptions":[],"reason":"opaque type `Probability`: exported producer `many` returns the type through an unsupported container (generic `List`); decompose-or-reject (RFC D-PRODUCER)"}
{"kind":"summary","total":1,"passed":1,"failed":0,"unsupported":0,"errors":0,"obligations":1}
```

Fields: `obligation_kind` is `"invariant_producer"` in V1; `source_type`
is the opaque type; `producer` is the exported def (or constant) under
obligation; `name` is `invariant:<Type>:<producer>`; `status` is one of
`passed`/`failed`/`unsupported`/`error`; `proof_tier` is `"smt"` (Tier B)
or `"fuzz"` (Tier C); `samples`/`seed` mirror the property records;
`counterexample` and `reason` are optional. A `counterexample` keys the
producer's inputs by **positional placeholder** (`__arg0`, `__arg1`, ...),
not by source parameter name. A placeholder the module itself spells is
suffixed (`__arg0_1`, ...) so it never aliases that name; every solver symbol
the prover generates is fresh against the names in its goal in the same way.
`arith_model:"real"` is
present on `proof_tier:"smt"` records (the SMT-over-reals caveat — see
the Tier B note below). A `status:"error"` record is a declaration-time
covered-or-rejected / signature-rejection failure and carries only
`obligation_kind`, `status`, and `reason`. Exit codes are unchanged in
meaning: a failed or errored obligation participates in the same
worst-status exit code as user properties (`Passed=0`, `Failed=1`,
`Unsupported=2`, `Error=3`).

### Composition and non-vacuity

COMPOSE folds each result with the assumption discharges it depends on. A
fuzz-only BASE (the property/obligation itself established by fuzz, with no
SMT proof underneath) composes to `composite_verdict:"fuzz_validated"` and can
never compose to any `proven_*` badge: a base that was not proved is not
proven-modulo-anything. A base discharged by a sound over-approximation
composes to `"sound_approximate"`. For an SMT-proved base: all-SMT discharges
compose to `"proven_modulo_real_arithmetic"` (the proof is over the reals --
see the Tier B caveat); any fuzz CONTRACT discharge composes to
`"proven_modulo_fuzz_validated_contract"` unless a weaker discharge is present;
any asserted axiom composes to `"proven_modulo_asserted_axiom"`.
Missing/unsupported discharges compose to `"unsupported"`;
counterexample-backed discharges compose to `"failed"`. The single
`composite_verdict` token is the weakest badge; the `qualifiers` array carries
the full disclosed set (e.g. an over-reals proof modulo a fuzz contract is
token `proven_modulo_fuzz_validated_contract` with
`qualifiers:["fuzz","real_arithmetic"]`).

For every claimed green result that has assumptions, `chelis prove` checks
the assumptions alone for satisfiability. SAT establishes non-vacuity. UNSAT
sets `composite_verdict:"invalid"` and the record does not render as a pure
pass. cvc5 unknown/timeout sets `composite_verdict:"unsupported"` with a
reason such as `non_vacuity_unestablished: smt unknown`; it is never reported
as `failed`. A `failed` verdict requires a counterexample.

### Invariant-binder generation and starvation (`--invariant-min-rate`)

A binder whose type is an invariant-carrying opaque type — in a derived
obligation or in a user `@property` — is generated only over
invariant-satisfying values: tiered rejection sampling then
constructor-based generation, with every accepted sample
predicate-validated (full design in `opaque_invariants_rfc.md` D-STARVE /
D-INJECT). When both tiers fall below the floor the binder starves and its
record is `status:"unsupported"` (exit `2`) with a `reason` naming the
type, per-method accepted/attempted counts, the rate, the floor, the
predicate-shape classification, and the recommended route. This is
distinct from user-precondition generator exhaustion, which stays an
`Error` (exit `3`): the two failure modes remain separable. The
`--invariant-min-rate <f64>` flag (default `0.01`) sets the floor for the
rejection tier; `--invariant-min-rate 0.0` disables the starvation
classification and preserves the legacy exhaustion-as-`Error` path.
Generation is deterministic under a fixed seed. A constructor probe program
the prover assembles that is rejected before it runs (it does not parse,
check, or lower) is an internal failure: when generation then finds no
sample, the record is `status:"error"` with that diagnostic, never a
starvation verdict.

Every root the prover declares in the user's program to evaluate a sample, a
producer value, a generated value, or a constant is named fresh against every
name that program spells, so a module that happens to define a probe's plain
spelling keeps its definition and its verdict (chelis#3267).

### Tier B SMT proofs are over the reals (caveat)

A `proof_tier:"smt"` obligation (or property) is discharged by the SMT
solver over the **reals**, while runtime arithmetic is IEEE
floating-point (and integer widths lower to an unbounded integer sort, so
overflow is not modelled either); typed integer inputs still have their exact
representable domains. The gap is now disclosed ON THE VERDICT:
such artifacts carry the `real_arithmetic` qualifier and render the
`proven_modulo_real_arithmetic` token (chelis#422), so a consumer reading
`composite_verdict` alone sees the caveat rather than only the legacy
`arith_model:"real"` side field (which is retained as a mirror). No
float-level soundness is claimed from a Tier B proof; admission policies
that quote the composed opaque-invariant guarantee inherit the disclosure
from the verdict. Tier C (`proof_tier:"fuzz"`) validates concrete float
samples and renders `fuzz_validated` (or, for a fuzz contract under an SMT
base, contributes the `fuzz` qualifier); it carries no `arith_model` field.

The real-arithmetic model does not license cvc5's partial operations outside
their mathematical domains. In particular, Tier B lowers `sqrt(a)` only after
an auxiliary obligation proves every exact argument `a >= 0` from independent,
sqrt-free total-algebraic conjuncts in the user's preconditions. The derived
domain facts are redundant assertions in the main query, never added user
assumptions. If an argument is not proved non-negative because the proof times
out, is unknown, or lies outside that conservative fragment, Tier B returns a
loud unsupported result and `auto` routes to Tier C (chelis#1475).

An obligation the solver decides **false** is the one case that carries
evidence: the model that falsifies it gives some `sqrt` argument a negative
value under the user's total, sqrt-free conjuncts. That model is not a
counterexample on its own -- the evidence set is a weakening of the stated
preconditions, and the decision is over the reals -- so its only authority is a
concrete evaluation. Tier B hands the witness to the property's host lane, which
rounds it once to each binder's declared width and runs exactly the evaluation a
Tier C sample gets. A witness every precondition admits and at which the body is
false is reported as `failed` with that machine value as the counterexample; its
verdict is a definite `failed` and carries no `real_arithmetic` qualifier,
because the concrete evaluation, not the solver, decided it. A witness the host
lane rejects, or one the lane cannot build (a binder outside the scalar
fragment, a value the model parser does not claim), keeps the unsupported result
above, so `unsupported` stays reserved for an obligation nothing decided
(chelis#3635). The derived-obligation lane does not run this confirmation: its
Tier C is a constructor-probe lane over a producer's arguments rather than an
evaluation of typed property binders, so a false domain obligation there is
still reported unsupported.

### `auto` corroborates an over-the-reals green in machine arithmetic

A Tier B decision is made over the reals, so it rules out a real
counterexample and nothing else. A property can be true over the reals and
false in binary64: `(x + 1.0f64) - x > 0.5f64` holds over the reals and fails
at `x = 1e17f64`, where `x + 1.0` rounds back to `x`. A tier that stops at such
a green reports `passed` on a property `--tier fuzz-only` refutes on the same
file, which would make the default tier weaker than sampling alone.

Under `--tier auto` an over-the-reals green is therefore **not terminal**. The
runner additionally runs Tier C over the same property or derived obligation as
a **refutation search** and reports what it established in the additive
`machine_arithmetic` object:

- **A Tier C counterexample replaces the green.** The record is
  `status:"failed"` with `composite_verdict:"failed"` — definite, not
  reals-hedged, because the witness is a concrete machine value, not a
  real-arithmetic model — at `proof_tier:"fuzz"`, carrying the counterexample,
  a `reason` naming why a proof did not prevent it, and
  `machine_arithmetic:{"status":"refuted", ...}` naming the over-the-reals
  verdict the refutation displaces.
- **A clean Tier C run corroborates the green without strengthening it.** The
  Tier B result stands unchanged — same `status`, `proof_tier:"smt"`,
  `samples:0`, same `composite_verdict`, same `qualifiers` — and carries
  `machine_arithmetic:{"status":"corroborated","samples":N}`. Sampling is not
  exhaustive, so corroboration never drops the `real_arithmetic` qualifier or
  promotes the badge.
- **A Tier C run that cannot sample leaves the green exactly as it was**, with
  `machine_arithmetic:{"status":"unchecked","reason":...}` (a binder Tier C
  cannot generate, invariant-binder starvation, precondition exhaustion). A
  sampler limitation never degrades a proof.

The rule applies to `{kind:"property"}` and `{kind:"obligation"}` records
alike. A refutation search is only as good as Tier C's own verdict, which is
why an invariant predicate is always lowered from its defining program, on the
concrete-eval validation path as well as the solver path: a module constant
lowered without its declaration would compare against an integer-sorted field
as a real, and the false counterexample that produces would displace a correct
proof.

The lane is keyed on the record's own published `qualifiers` array, not on a
verdict token, so a future badge that discloses the same gap inherits the
corroboration without a second list to keep in step, and the lane can never
disagree with the disclosure it acts on. A green is corroborated when
`qualifiers` contains `real_arithmetic` and does **not** contain `axiom`: an
asserted axiom is a user-declared assumption the concrete evaluator does not
honor — `clarabel.qp.ideal_optimality` idealizes the float solver away on
purpose — so Tier C there would refute the axiom rather than the derivation,
and the conditional green stands. A `fuzz`-qualified CONTRACT is not an
idealization and does not exempt its base.

`machine_arithmetic` is **absent** when no check was attempted: under
`--tier smt-only`, `--tier induction-only` and `--tier beacon-only` (an
explicitly selected tier is never widened), and for a green whose base is not
over the reals. `auto`'s induction lane is terminal by its own contract
(chelis#978: an accepted induction plan does not trail into a finite fuzz
sample), so an induction green carries no check even though it discloses
`real_arithmetic`. Extending the check to that lane is residual work, not a
property of this rule.

The guarantee this rule establishes is exactly one direction of parity: a
counterexample Tier C would find can no longer be suppressed by an
over-the-reals proof, so an `auto` solver proof never reports a pass where
`fuzz-only` reports a refutation. An induction green is outside it, for the
reason above. It is not a claim that `auto` and `fuzz-only` agree on every
input or exit code — a Tier C run that errors or establishes nothing leaves
`auto` reporting its proof while `fuzz-only` reports that failure, which is the
intended asymmetry. The check also does not change what Tier C can find:
whether a given machine counterexample is reachable is a property of the
sampler (chelis#3636).

`auto` therefore spends the sample budget after every over-the-reals proof, so
proving an all-green module costs roughly the sum of the deductive and sampling
tiers rather than the deductive tier alone. `--tier smt-only` is the way to pay
only for the proof, at the cost of reporting the machine-arithmetic question as
unchecked.

Plain text renders the disclosure in place of a sample ratio an over-the-reals
proof does not have:

```text
property: sq_nonneg -- proven_modulo_real_arithmetic (proved over the reals; machine arithmetic corroborated on 100 samples)
property: guarded -- proven_modulo_real_arithmetic (proved over the reals; machine arithmetic unchecked: <reason>)
obligation: invariant:Probability:scale -- proved (smt; proved over the reals; machine arithmetic corroborated on 100 samples)
```

A sample ratio is reserved for a result that took samples: `0/0 passed` reads
as a count of zero out of zero rather than as a proof that takes none.

### `auto` confirms an over-the-reals disproof in machine arithmetic

A Tier B **disproof** is decided over the same real-arithmetic model, so it is
the symmetric half of the rule above. The solver exhibits a point in the reals
at which the goal is false; reporting that as a property failure claims a defect
at some MACHINE input, which the solver has not established. An exact rational
need not have a binary64 value at which the property is false, and an irrational
algebraic witness has none at all: `1/((x*x) - 2.0)` is disproved finite over
the reals at `x = sqrt(2)`, while no binary64 value squares to exactly 2 -- the
two doubles nearest `sqrt(2)` give `x*x` of 2.0000000000000004 and
1.9999999999999996 -- so the quotient is finite at every double in `[1, 2]`.

The witness is therefore handed to the property's host lane, which projects the
model at the binders' sorts, rounds it once to each binder's declared width, and
runs exactly the evaluation a Tier C sample gets. Three answers, reported in the
same additive `machine_arithmetic` object:

- **Confirmed.** Every precondition admits the rounded witness and the body is
  false there. The record is `status:"failed"` with `composite_verdict:"failed"`
  — definite, not reals-hedged, and carrying no `real_arithmetic` qualifier,
  because the concrete evaluation decided it — at `proof_tier:"smt"`, with that
  machine value as the `counterexample`, a `reason` stating what was checked,
  and `machine_arithmetic:{"status":"confirmed"}`. This is the chelis#3635
  domain-witness rule applied to the main goal's own model.
- **Unconfirmed.** The witness is **not** a machine counterexample: no machine
  value equals it (an irrational algebraic witness), the property's own
  preconditions reject it once rounded, or the body holds there. Under
  `--tier auto` the run routes to Tier C, which is the tier that can actually
  search machine arithmetic; whatever Tier C reports carries
  `machine_arithmetic:{"status":"unconfirmed","reason":...}` naming the
  over-the-reals disproof it reports in place of. Under the terminal
  `--tier smt-only` there is nothing to route to, so the over-the-reals disproof
  stays the reported verdict with the same object recorded beside it.
- **Unchecked.** Nothing was established: a model value the projection cannot
  read, or a witness the concrete evaluator could not run. The Tier B disproof
  stands exactly as it was, with
  `machine_arithmetic:{"status":"unchecked","reason":...}`. An evaluation
  limitation never displaces a decision, which is this side's form of "a sampler
  limitation never degrades a proof".

Only the first answer may report a failure that the solver alone motivated, and
only the second may withdraw one: the lane needs a positive finding about the
witness in either direction. An algebraic witness's isolating interval is
deliberately not read as a candidate point — its endpoints bracket the root, so
a point taken from them is a different real from the one the solver exhibited.

Unlike the green side's refutation search, the confirmation is a **single
concrete evaluation** of the solver's own witness rather than a second tier, so
it runs under `--tier smt-only` as well and does not widen an explicitly
selected tier. `machine_arithmetic` is absent from a disproof when no check was
attempted: under `--tier induction-only` and `--tier beacon-only`, for a
disproof whose base is not over the reals (already a definite `failed`), and for
one conditional on an asserted `axiom`, which the concrete evaluator does not
honor. The derived-obligation lane does not run the confirmation, for the
chelis#3635 reason: its Tier C is a constructor-probe lane over a producer's
arguments rather than an evaluation of typed property binders.

Plain text keeps the sample ratio a Tier C pass actually has and adds the
disclosure, rather than substituting one for the other:

```text
property: g_finite -- 100/100 passed (disproved over the reals; not confirmed in machine arithmetic: <reason>)
```

## Relationship To `chelis test`

`chelis test` remains the deterministic assertion runner for `Std.Test`.
`chelis prove` is the property runner. V1 does not run properties
automatically from `chelis test`.

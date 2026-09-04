# Prove-Obligation Unification (review 3 architectural rework)

Owning RFC: [`opaque_invariants_rfc.md`](opaque_invariants_rfc.md) (v7,
frozen). This document records a structural rework of the
SMT-obligation / prove layer, not a contract change to the RFC. The
RFC decisions (D-PARITY, D-INJECT, D-OBLIG, D-TIERB, D-SOUND, D-WF,
D-STARVE) are unchanged; this rework removes the duplicated parallel
code paths that implemented them, so a spot-fix to one path can no
longer miss its sibling.

## Why a rework, not a fix

Three xhigh code reviews of opaque types with declared invariants found
that the checker-side opacity, invariant declarations, decode, and docs
are solid, but every review's residual bugs clustered in ONE place: the
SMT-obligation / prove layer (`chelis-prove`
obligation/injection/lowering plus `chelis-tide`). The root is
architectural: this layer carried DUPLICATED PARALLEL CODE PATHS for the
same logical operation, and a spot-fix to one path never propagated to
its sibling. The four unifications below each collapse a divergence
class to a single chokepoint.

## U1 -- one produced-value validation chokepoint

Contract: there is exactly ONE function that validates a produced opaque
value (or a generated sample) against its invariant, and it walks EVERY
representation leaf (scalar fields, EVERY tensor element, nested-record
fields) and rejects fail-closed if ANY leaf is non-finite (NaN OR Inf)
BEFORE evaluating the invariant predicate.

Parallel paths merged:

- `eval_obligation_body_values` branched on `all_scalar`. The all-scalar
  branch called `eval_obligation_predicate` (host-runtime eval), which
  NEVER reached a finiteness guard, so a scalar NaN representation field
  shipped as a PASSING obligation under a `!=`/`not(==)` invariant (the
  strict evaluator gives `NaN != C == true`). The non-scalar branch went
  through `eval_producer_value -> validate_value -> validate_with_predicate`,
  which HAD the NaN/Inf guard. The generator's `validate_env` was a third
  copy of the same finiteness-then-predicate check.

Single chokepoint: `obligation_engine::validate_produced_env` is the one
function that (a) walks an opaque record `ExecutionValue` into a
flattened dotted-path env covering every leaf, (b) rejects any
non-finite leaf, then (c) evaluates the flattened invariant predicate
with the strict concrete evaluator. The produced-value path
(scalar AND tensor/record) routes its produced record through this one
function; the generator's `validate_env` shares the SAME finiteness
helper (`any_non_finite`) and the SAME strict-predicate evaluation, so
proposal and acceptance agree by construction. The host-runtime
`eval_obligation_predicate` branch is removed: a produced value is always
evaluated structurally (the runtime cannot lower a `sum`/constant-bearing
tensor invariant anyway, and routing scalars through it was the only
reason the NaN guard could be bypassed).

## U2 -- one type-aware constant lowering

Contract: there is exactly ONE function that lowers an in-module constant
reference to an `SmtExpr`, used by BOTH the reduce/producer-body path AND
the invariant-application path. It preserves the declared numeric type
for ALL integer widths (int8/int16/int32/int64 -> `SmtSort::Int` /
`IntLit`; f32/f64 -> `Real`), resolves a constant whose body references
ANOTHER constant transitively, and never produces a sort mismatch.

Parallel paths merged:

- The producer-body path (`reduce` -> `const_lit_node` ->
  `const_declared_int_type`) was made type-aware for int32/int64 only.
- The invariant-application path (`lower_pred_arith`) hardcoded
  `consts.get(name).copied().map(SmtExpr::RealLit)` -- so within ONE
  property the SAME constant lowered as `IntLit` on the producer body and
  `RealLit` in the invariant; compared against an Int-sorted var, cvc5
  ABORTED THE PROCESS ("Subexpressions must have the same type:
  Int/Real").
- `const_declared_int_type` handled only int32/int64 (not int8/int16) and
  read the declared type only from a DIRECT `lit` body, so a constant
  whose body referenced another constant lost its type.

Single chokepoint: `chelis_prove::opaque::lower_const_ref(exprs, name,
value)` returns the typed `SmtExpr` for a constant. It reads the declared
numeric type via `const_declared_int_type` / `const_declared_numeric_type`,
which (a) recognizes every integer width int8/int16/int32/int64 as
`SmtSort::Int`, f32/f64 as `Real`, reading the AUTHORITATIVE declared
return type from a sibling `defsig` (a typed `def a() -> int8 = 1` carries
int8 there, not on the default-int32 body literal), and (b) follows a
`(var other_const)` body transitively to the literal that carries the type
tag. The resolver lives in `opaque` (the shared module both surfaces use)
so the producer-body path (`tier_b_lower::reduce` ->
`const_lit_node`), the invariant-application path
(`tier_b_lower::lower_pred_arith`), and the opaque flattened-predicate
precondition path (`opaque::lower_arith`) all consult the SAME resolver.
The hardcoded `RealLit`-for-constant arms are gone.

## U3 -- one cvc5-lowerable intrinsic source of truth + sort-mismatch pre-check

> Superseded by **W5** below. The U3 sort pre-check was a blacklist that
> kept missing aborting shapes; W5 replaces it with the whitelist gate
> (`is_cvc5_lowerable`). The `CVC5_LOWERABLE` / `CVC5_TRANSCENDENTAL`
> single-source-of-truth for intrinsics from U3 is retained and consumed by
> the whitelist. This section is kept for history.

Contract: `validate_smt_arity`, `lower_to_cvc5`, and
`contains_transcendental` all derive from the single `CVC5_LOWERABLE` /
`CVC5_TRANSCENDENTAL` source (no fourth list that can drift). A
whitelisted-but-not-cvc5-lowerable function (`log`) routes to a clean
Tier-B-unsupported -> Tier C. `lower_to_cvc5` never panics AND never
hands cvc5 a term that aborts it -- including a U2-style operand-sort
mismatch: before solving, a sort pre-check rejects any
comparison/arithmetic with mismatched operand sorts (Int vs Real) as a
clean `TierBResult::Error`, NOT a cvc5 abort.

Parallel paths merged: this is mostly hardening of CR2-1 (the three lists
already derive from one source). The new single chokepoint is
`infer_sort` + `check_operand_sorts`, run in `solve_property_cvc5` after
`validate_smt_arity`: it infers each leaf's sort (from the declared
variable sorts and the literal kinds) and rejects an Int-vs-Real
comparison or arithmetic operand mismatch before any cvc5 term is built.
This is defense in depth behind U2 (U2 makes the mismatch
unconstructible in the obligation path; the pre-check guarantees that any
mismatch from any future caller is a clean Error, never a process abort).

## U4 -- proper tide chelis_prove property dispatch

Contract: the tide `chelis_prove` tool discovers user `@property`
declarations the SAME way the CLI does (no hardcoded `property` name),
for `.ch` AND `.dp` modules, runs them through the shared engine, runs
the producer obligations, and reports `ok=true` ONLY if every property
AND obligation is a genuine pass (Proved, or StatisticallyValidated with
`samples>0`); `ok=false` for Disproved, Rejected, NotAmenable, error,
unsupported, AND StatisticallyValidated with `samples==0` (the
Tier-B-timeout sentinel under smt-only). tide's ok/failed matches the
CLI's pass/fail and exit code for the same module (test-locked parity).

Parallel paths merged:

- The tide handler gated user-property dispatch on a binding literally
  named `property`, so a property with any other name was silently never
  proved (response reported ok:true / total:0).
- `source_defines_binding` always parsed with the SURF parser even for
  `source_kind="deep"`.
- The fold treated `StatisticallyValidated{samples:0}` (the smt-only
  timeout sentinel) as a pass.

Single chokepoint: a shared property runner lives in
`chelis_prove::property_runner` (with `smt_lower` + `injection`
submodules relocated from the CLI). It discovers user `@property`
declarations from `.ch` (parse + flatten modules) and `.dp` (Deep
metadata scan) -- the SAME discovery the CLI used -- runs each through the
shared Tier B (SMT) -> Tier C (fuzz) engine with assumption injection for
invariant-carrying opaque binders, and returns a `PropertyOutcome` whose
`is_pass()` is true ONLY for `Proved` or fuzz-validated-with-`samples>0`
(the zero-sample sentinel is NOT a pass). The tide tool calls the runner
directly; the CLI delegates to it under the `chelis-prove` capability
(`prove::property_run`) and renders the outcomes as its NDJSON property
records. The CLI's own `smt_lower` and `injection` modules were retired.
The no-`chelis-prove` CLI build (the degraded default with no SMT/cvc5
and no injection) keeps a local Tier-C-only fuzz path; it does not reach
the shared runner because the runner's Tier B / injection require the
capability. Because tide and the capability-enabled CLI compute their
verdict from the SAME runner, they agree by construction (test-locked
parity).

## W5 -- the Tier B whitelist invariant (supersedes the U3 blacklist; itself superseded by RT6 below)

> SUPERSEDED by RT6 (total lowering + process isolation). The separate
> `is_cvc5_lowerable` gate described here was DELETED: a sixth review proved
> a separate sort gate and the term builder are two enumerations of cvc5's
> rules that diverge (the gate checked sorts but not arity, so it ADMITTED an
> empty/single-child `and`/`or`, a non-binary `implies`, an integer `/`
> feeding a comparison, and a non-finite literal, all of which the builder
> then aborted cvc5 on). RT6 folds the check into the builder. This section
> is kept for history.

Invariant: **Tier B lowers only provably-safe terms; anything else routes
to Tier C; lowering never aborts cvc5.**

The U3 sort pre-check BLACKLISTED known aborting shapes (Int-vs-Real
comparisons, mixed-sort `min`/`max`, `ITE` branches). A blacklist keeps
missing siblings: three more abort shapes survived three reviews -- a
transcendental over an `Int` argument, a `Bool`-vs-`Int` comparison, and a
quantifier whose bound-var sort mismatched a literal. The invariant is
inverted to a WHITELIST.

`is_cvc5_lowerable(prop)` (the SOLE pre-lowering gate in
`solve_property_cvc5`) walks the whole property -- every precondition and
the postcondition -- carrying a sort environment seeded from
`prop.variables`, and returns lowerable ONLY IF every node is PROVABLY
cvc5-safe:

- `Var`: its sort is KNOWN in the env (else NOT lowerable).
- literal: sort known (`IntLit` -> Int, `RealLit` -> Real, `BoolLit` ->
  Bool).
- `Cmp`: both operands lower to KNOWN, EQUAL sorts; a numeric comparison
  (`<`/`<=`/`>`/`>=`) additionally rejects a `Bool` operand. `Eq`/`Ne`
  admit any equal sort (including `Bool == Bool`).
- `Arith`: operands known + matching numeric sorts; unary `Neg` checks
  ONLY its real operand (skip the placeholder).
- `Apply`: name in `CVC5_LOWERABLE` at correct arity AND each arg meets
  that function's cvc5 sort requirement -- `sqrt`/`exp`/`sin`/`cos`
  require a Real arg (an Int arg -> NOT lowerable -> Tier C); `min`/`max`
  require equal-sort args; `abs` preserves a numeric sort. Any other name
  (e.g. `log`, with no cvc5 kind) -> NOT lowerable.
- `Ite`: `c` is Bool; `t` and `e` have known EQUAL sorts.
- `Forall`/`Exists`: ADD the bound vars (with their declared sorts) to the
  env, THEN recurse into the body; restore shadowed bindings afterward.
- ANY node where a sort is Unknown, a requirement is unproven, or an op is
  unrecognized -> NOT lowerable. The DEFAULT is NOT-lowerable; only
  explicitly-proven-safe terms pass.

Because the default is reject, a shape no review enumerated
(transcendental-over-Int, Bool-vs-Int, quantifier-bound-var, or any future
shape) routes to Tier C automatically. `lower_to_cvc5` is therefore only
ever called on a provably-safe property; its `Result`-not-panic arms are
defense in depth. Quantifier lowering uses cvc5 bound variables (`mk_var`)
and the logic selection drops the `QF_` prefix when a quantifier is
present, so an admitted quantifier lowers under a quantified logic rather
than aborting. The old blacklist (`check_operand_sorts`, `definitely_mixed`,
`infer_sort`, `unify_sort`) and the separate arity guard
(`validate_smt_arity`) are removed -- the whitelist subsumes them.

The abort-proof guarantee is asserted at the subprocess level
(`whitelist_corpus_never_aborts_cvc5`): a cvc5 process-abort cannot be
caught in-process, so a diverse predicate corpus is run through `chelis
prove` and EACH run must exit cleanly with no cvc5 sort/type abort marker.

W5 also single-sources the signed-integer-width decision across the whole
workspace: `chelis_types::Prim::is_integer` / `integer_range` /
`integer_fuzz_bounds` own the int-width set, range, and fuzz-sampling
window; the prove layer's `is_int_width` / `int_sample_bounds` are thin
wrappers, and every recognition / sort / sampling / literal site routes
through them, so an `int8`/`int16` field, param, or constant is handled
identically to `int32`/`int64` everywhere.

## RT6 -- total lowering + process isolation (supersedes the W5 whitelist)

A sixth review found the W5 whitelist gate (`is_cvc5_lowerable`) and the
term builder (`lower_to_cvc5`) were TWO enumerations of cvc5's requirements
that DIVERGED: the gate checked operand SORTS but not cvc5's ARITY /
kind-domain rules, so it ADMITTED terms the builder then aborted cvc5 on.
The fix has two layers.

### Layer 1 -- the lowering is TOTAL with respect to aborts

The separate gate is DELETED. `lower_to_cvc5` now returns `(Term, SmtSort)`
and is the SOLE authority: it is one bottom-up pass where every `mk_term`
call site verifies cvc5's requirement for that kind FIRST -- operand sorts
AND arity -- and returns `Err` (routed to a clean Tier C result) rather
than handing cvc5 an aborting term. Because the sort it returns is the sort
cvc5 actually builds (e.g. integer `/` reports `Real`, since cvc5 promotes
it), a parent node's check sees the truth and can no longer diverge from
the construction. Degenerate connective arities normalize to their logical
identity (empty `and` -> true, empty `or` -> false, single-child -> the
child) so a single-conjunct invariant still proves at the SMT tier rather
than aborting or regressing to Tier C; `implies` requires exactly 2.
Additional guards at their call sites: a non-finite `RealLit` (cvc5
`mk_real_from_str` aborts on `inf`/`NaN`); an empty quantifier binder list
(`mk_term(VARIABLE_LIST, &[])` aborts -- normalized to the body, since
`forall (). P == P`); a non-Bool precondition / postcondition (cvc5
`assert_formula` / `NOT` abort on a non-Bool); an interior-NUL
variable/binder name (cvc5-rs `CString::new(...).unwrap()` PANICS); and an
`SmtExpr` deeper than `MAX_SMT_EXPR_DEPTH` (the recursive walks overflow the
stack -> SIGABRT), bounded by an ITERATIVE check at the entry that cannot
itself overflow.

Partial solver kinds also require a semantic domain gate, not only a
sort/arity gate. Before Tier B constructs any cvc5 `SQRT`, it collects every
distinct exact argument and proves their conjunction non-negative from the
user's other sqrt-free, top-level conjunctive preconditions. The proof fragment
is deliberately total algebraic arithmetic (`+`, `-`, `*`, unary negation and
comparisons): division, applications, conditionals, nested `sqrt`, and
quantified evidence cannot authorize the partial kind. Only a proved auxiliary
obligation authorizes the exact arguments in the private builder context;
SAT, timeout, unknown, or lowering error routes the original property to Tier
C. The proved domain facts are then asserted in the main query as redundant
facts already implied by the user's assumptions. The public unguarded builder
has no authorization context and therefore rejects `sqrt`. This is the
chelis#1475 soundness boundary: Tier B may lose reach, but it never adds
`arg >= 0` as an unproved assumption.

### Layer 2 -- process isolation (the residual)

cvc5 fails by PROCESS ABORT, which cannot be caught in-process, so Layer 1
closes every KNOWN cause but cannot PROVE no unknown cvc5-internal abort
remains. `chelis_prove::worker` makes it moot: the `chelis` binary's `main`
calls `enable_isolation()`, and `solve_property` then runs every cvc5 solve
in a short-lived CHILD process (a re-exec of `current_exe` carrying the
`CHELIS_PROVE_WORKER` marker, fed a bincode-encoded `SmtProperty` over
stdin, returning a result over stdout). ANY way the child can fail -- a
cvc5 C++ abort, a cvc5-internal assertion on a well-formed formula, a stack
overflow, an OOM kill, a panic, a hang past the deadline -- becomes a clean
`TierBResult::Error`/`Unknown` in the parent (routed to Tier C); the
`chelis` process is never taken down by a solve. Isolation is opt-in: only
a host that calls `enable_isolation` spawns workers, so tests solve
in-process (no spawn) and exercise the Layer-1 lowering directly, while the
end-to-end isolated path -- including recovery from a worker that
aborts/panics/overflows on every solve -- is locked by the
`prove_isolation` integration test running the real `chelis` binary.

## Acceptance oracle

Authoritative completion oracle for this rework:

```
cargo nextest run -p chelis-prove --features smt
cargo nextest run -p chelis-cli   --features smt
cargo nextest run -p chelis-tide  --features smt
```

all green, with the new red-tests below passing. The full repo gate
(`scripts/gate.py`) plus `cargo build -p chelis-cli --features smt` must
also be green. The RT6 process-isolation acceptance is the
`prove_isolation` integration test (under the `chelis-cli --features smt`
run above): the baseline proves obligations at the SMT tier through the
worker, and a worker forced to abort / panic / overflow on every solve must
leave the parent alive with obligations fallen to Tier C.

The focused chelis#1475 regression oracle is:

```sh
cargo nextest run -p chelis-prove --features smt --test issue_1475_sqrt_domain
```

Success is exit zero with all cases passing: both reported unsafe shapes and
unguarded/nested/quantified/partial-evidence forms fail closed, while guarded
and algebraically non-negative arguments remain in Tier B, a false in-domain
property still disproves, and the `exp`/`abs` controls retain their results.

## Test matrix (union of all three reviews)

Non-finiteness (U1), every axis that previously hid a bug:

- NaN AND Inf, in: a scalar field, EACH tensor element, a nested-record
  field;
- under an inequality invariant AND a `!=` / `not(==)` invariant;
- at `--tier fuzz-only` AND `--tier auto` AND (where buildable) the
  no-smt path;
- on the obligation path AND injection sampling.
- All fail-closed (obligation FAILED / sample REJECTED).

Constants (U2):

- a constant of EACH numeric type int8 / int16 / int32 / int64 / f32 /
  f64, used in a producer guard AND in an invariant predicate (same
  property);
- a constant whose body references another constant.
- No cvc5 sort-mismatch abort; correct lowering.

Intrinsics (U3):

- EVERY whitelisted intrinsic (`chelis_pred::INTRINSIC_WHITELIST`) at
  correct arity => lowers+proves OR clean Tier-C (`log`); wrong arity =>
  clean error; never panic; never cvc5 abort. The test loops over the
  whitelist so none is skipped.
- an Int-vs-Real comparison constructed directly yields a clean
  `TierBResult::Error`, never a process abort.

tide dispatch (U4):

- a property named `property`, a DIFFERENTLY-named property, multiple
  properties, no property;
- a `.ch` module AND a `.dp` module;
- a failing / Rejected / timeout property => ok:false; a clean module =>
  ok:true;
- CLI-vs-tide parity asserted for each (same pass/fail, same exit-code
  sign).

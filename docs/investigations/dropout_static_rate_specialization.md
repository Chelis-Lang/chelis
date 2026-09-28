# Bounded typed dropout rates

## Generic evaluator dispatch follow-up (2026-09-12)

The evaluator now routes checked concrete calls to precision-polymorphic,
fixed-rank tensor helpers through its existing fixed-control execution planner.
The original `keep[p: Float](x) = dropout(x, cast(0.5, p))` source evaluates
at f16/bf16/f32/f64. The callee's unresolved generic declaration is not itself
an executable plan: admission retains the checked call's actual types and
source control expressions until existing lowering specializes that call.

Isolated call classification attaches missing argument type annotations from
the evaluator's existing checked frame types. It does not replace argument
expressions with their evaluated values. Missing or invalid type evidence
stays absent. Data arguments are evaluated once and staged by the existing
typed-placeholder route. Admission and scalar-staging trials use the same
annotated source call, so an effecting scalar data argument is not mistaken
for a control and executed again. Source-static scalar controls retain their
original expressions. Runtime-rate variables and computed scalar rates remain excluded,
as do rank-polymorphic or already excluded control profiles. Local callable
bindings still shadow top-level helpers. No builtin dispatcher, RNG key cache,
ambient seed, public ABI, numeric carrier or runtime-rate C support is added.

`dropout_fixed_stream_api` checks the exact original source, all four float
dtypes, concrete loss wrappers differentiating generic helpers, complete
forward/gradient/next-draw values, shape/dtype, input preservation and repeated
prepared evaluation. Mixed f32/f64/f32 calls keep independent precision, and
effecting tensor and scalar operands consume their draws exactly once. The
scalar regression compares complete stored words for both bare and copied
tensor actuals through direct API and repeated prepared evaluation. Paired runtime-rate and
lexical-shadow controls retain their prior boundaries. `fixed_control_host_c`
executes generic forward/AD/next-draw source at all four dtypes and balances
the native ownership ledger. The existing CLI example now includes a generic
following draw, with unchanged results.

Owning command: `cargo nextest run -p chelis-compiler-api --features
ownership-ledger --test dropout_fixed_stream_api --test fixed_control_host_c
--locked --offline --build-jobs 1 --test-threads 1`, after the package's
`cargo check --tests`. This is bounded evaluator/native parity for the named
source profile, not exhaustive generic AD, arbitrary-rate native execution,
E1/E2 completion or a derivative theorem over IEEE arithmetic. In particular,
applying `grad` directly to an unresolved generic scalar-return loss still
reaches a separate checker limitation; the tested loss wrappers have concrete
scalar results. Earlier boundaries below are historical.

## Concrete call forwarding follow-up (2026-09-12)

The follow-up repairs concrete tensor-returning helper calls whose source-static
rate was lost when the evaluator entered the helper frame. Unbound declarations
remain checked library templates; actual calls reuse the existing typed scalar
grammar and execution planner. Scalar data operands are staged once, while only
the source expressions required to prove fixed controls survive staging. A local
binding cannot be mistaken for a same-named top-level template. Legacy helper
summary probes no longer eagerly lower an unresolved-rate template, and ordinary
dropout lowering shares the typed rate recognizer. No new random interpreter or
rate cotangent is introduced.

The executable `examples/dropout_static_rate.ch` exercises a local helper,
gradient replay, input preservation, and the following draw through actual CLI
eval and native C. API coverage adds distinct literal callsites and repeated
prepared evaluation through encoded/decoded Reef contexts. Runtime-computed
rates, dynamic control, unsupported generic evaluator calls, and lexical capture
exclusions remain outside this repair.

Export policy matters: without an explicit export list, every top-level def is
public (spec/02-surf-syntax.md). A whole-program build containing an unbound
public `keep(x, rate)` still rejects its standalone C entry; it must not silently
discard that entry to make a literal wrapper compile. A Reef library context
retains an exported helper as source for consumer specialization, not as a
standalone runtime-rate C ABI. In-context compilation selects new consumer roots,
not library functions (the `compile_for_execution_in_context` API contract).

Compiled-in-context C remains a separate known gap: its resolved-entry branch
attaches `None` instead of the monolithic selected-source execution plan. Thus a
consumer `main(x) = with seed(42i64) { keep(x, 0.5f32) }` still reaches the old
compiled-dropout rejection. This intended-success case is not a permanent
negative contract. The repair should reuse existing selected-plan construction
and transport in `compiler.rs`, with owning API/native tests; it must not add a
second C random implementation. Symbolic rank-four extent emission yielding
bare `*` is likewise separate. Neither issue is fixed or certified here.

The remainder records the earlier IR-only slice and its historical boundaries.

## Original IR specialization slice

Part of #1764. This slice specializes statically known dropout rates before
fixed-control admission and lowering. The source grammar is literals, checked
or truncating casts, negation, and lexical aliases/first-order helper parameters.
Every cast retains its source dtype and finalizes at its resolved target dtype.
Generic precision bindings come from the checked call's formal/actual types.
Gradient child contexts retain stable static lexical captures. Source arguments still
execute once in the ordinary lowering order; recognition does not execute them.

The controlling contracts are spec/05-risc-primitives.md [05-OP-37], [05-RNG-1],
and spec/04-type-system.md's numeric cast and precision-specialization rules.
Runtime rates, runtime seed/control extensions, rate cotangents, native C
emission, host ABI changes, Resource evidence, and certification are excluded.
The public evaluator, compiled-C, and public ABI acceptance obligations in #1764
remain open. The ordinary host path does not yet transport specialized rate
provenance. Its legacy static extractor deliberately retains the prior
pre-execution lowering rejection; letting it pass would expose an unsupported
host Dropout dispatch. A non-generic host wrapper containing a captured-rate
gradient also reaches that boundary through its helper-summary probe. Direct
checked execution-plan entries are the supported surface of this slice.
Late shadowing of a captured scalar remains outside the bounded profile: the
existing lowerer has no general lexical closure carrier. Admission forgets a
captured scalar's static fact when its caller binding differs. Only a rate
read depending on that fact causes rejection; nested helper calls and aliases
propagate the missing fact. Unrelated rebound scalars (including discarded
reads) do not reject a literal-rate closure. Function-local binders and actual
arguments establish their own static facts.
Host entry admission reads the original checked function body,
before signature preparation's local-callable substitution could erase that
distinction. Ordinary non-plan lowering and its legacy extractor are unchanged.

## Baseline

At `6dbbbf2bcafbe9c4f2a3feff7719f5b615fa0f94`, new helper/gradient and
source-width tests in `dropout_fixed_stream_ir` failed because profile admission
returned no tensor kernel. The exact #1764 `cast(0.5, p)` source failed through
`eval_selected` with the reported static-cast lowering error. Runtime-rate
admission declined, as expected.

Commands use the worktree's uv Python, one Cargo build job,
`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`,
and `CARGO_HUSKY_DONT_INSTALL_HOOKS=1`:

- `cargo check -p chelis-ir --tests`: passed, 24.29 seconds.
- `cargo nextest run -p chelis-ir --test dropout_fixed_stream_ir -E 'test(typed_static_rate)' --no-fail-fast`: positive failures reproduced.
- `cargo check -p chelis-compiler-api --tests`: passed, 29.90 seconds.
- `cargo nextest run -p chelis-compiler-api --test dropout_fixed_stream_api -E 'test(generic_static_rate_cast_issue_1764)' --no-fail-fast`: exact issue reproduced.

The negative controls distinguish runtime formals and lexical shadowing from
literal specialization; dtype-width and differing-call tests distinguish a
typed binding from a shared or f64-default constant. Execution tests compare
independently calculated masks and actual stream advancement.

## Author intent and verification

The invariant is a typed source value, not an f64 replacement expression:
admission and execution lowering call the same scalar recognizer. Argument
values are recognized in the caller before any formal shadows a name; ordinary
lowering still evaluates each argument once. Generic substitutions reuse the
existing formal/actual precision matchers. The checked cast-result identity
outranks an unrelated same-spelled caller binder, and unresolved targets do not
receive a default dtype. A shadowed `neg` is not treated as the primitive.

Positive tests cover helper parameters, scalar aliases, stable gradient
captures, all four active float dtypes, distinct callsite rates, source f32 to
f64 width, argument draws exactly once, and a reused plan's next stream ordinal.
Negative tests cover runtime parameters, alias/formal shadowing, late closure
and gradient-capture shadowing, unresolved dtype targets, and shadowed `neg`.
The R1 public regression compares complete values and the following draw after
rebinding an unrelated scalar. Paired IR probes preserve stable indirect
captures and local binding shadowing while rejecting changed transitive
captures, including gradient and inherited shadowed-primitive cases. Profile
precision comes only from checked metadata and scoped dtype facts, not from
inspecting a staged literal's payload. The structural inventory registers the
private module without adding or reclassifying representation debt.
The exact public #1764 repro remains a lower-stage rejection with no transcript.
Existing fixed-stream API acceptance tests remain unchanged.

Initial focused oracle, using the environment above:

`cargo nextest run -p chelis-ir -p chelis-compiler-api --lib --test dropout_fixed_stream_ir --test dropout_fixed_stream_api -E 'test(typed_static_control) | binary(~dropout_fixed_stream)' --no-fail-fast`

Passed 48/48 tests (1,151 unrelated unit tests filtered), build 42.03 seconds,
execution 1.041 seconds, run `608a9651-d39a-4a6c-9162-94c30785db45`.
This is an
IR specialization slice, not completion of #1764 or any native/public transport
phase. Fresh review and CI are required before merge.

Round-one author repairs reproduced the unrelated-capture API failure and the
matching direct-plan admission failure before changing the guard. An additional
discarded-capture probe failed an intermediate eager dependency check; forgetting
only unavailable static facts repairs both witnesses without evaluating closures.
The expanded owning command above, with `--test-threads 2`, passes 50/50 tests
(1,151 unrelated unit tests filtered), build 43.42 seconds, execution 1.792 seconds,
run `0aa128d6-428d-449d-b275-7d2d6ab53798`.
`python -m unittest scripts.test_runtime_representation_oracle scripts.test_runtime_representation_phase1`
passes 66 tests in 12.683 seconds using the worktree's managed environment.
The live structural scan validates 290 rows and zero rows in the added module;
all 358 foundation and 290 active rows equal the pre-repair inventory. These are
author repair checks, not a new review round or Phase 1 certification. The standing
reviewer must verify the repaired pushed head before either finding is closed.

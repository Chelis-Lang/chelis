# Bounded typed dropout rates

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
existing lowerer has no general lexical closure carrier. Admission compares
captured controls with the caller before entering a closure and rejects a
changed binding. Host entry admission reads the original checked function body,
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
The exact public #1764 repro remains a lower-stage rejection with no transcript.
Existing fixed-stream API acceptance tests remain unchanged.

Final focused oracle, using the environment above:

`cargo nextest run -p chelis-ir -p chelis-compiler-api --lib --test dropout_fixed_stream_ir --test dropout_fixed_stream_api -E 'test(typed_static_control) | binary(~dropout_fixed_stream)' --no-fail-fast`

Passed 48/48 tests (1,151 unrelated unit tests filtered), build 42.03 seconds,
execution 1.041 seconds, run `608a9651-d39a-4a6c-9162-94c30785db45`.
This is an
IR specialization slice, not completion of #1764 or any native/public transport
phase. Fresh review and CI are required before merge.

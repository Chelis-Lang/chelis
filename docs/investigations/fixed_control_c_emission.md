# Closed fixed-control C emission

Part of #1192 and LaCaDiLE's R3 compiler boundary. The sealed backend entry is
also carried through ordinary C source/API/CLI admission, including concrete
specializations of generic host helpers such as #1764. HIP/Metal remain
excluded and retain their typed rejection.

`EvaluationPlan::verify_ownership` consumes the lowering-owned graph and source
execution metadata. It validates the plan, requires the actual node schedule to
equal the graph order used by ownership lowering, and checks logical ownership.
The resulting `VerifiedEvaluationPlan` has no public fields or constructor.
`codegen_evaluation_with_options` consumes that sealed pair. A separately
verified graph or a borrowed graph inspection cannot stand in for the plan.

The C entry remains the four-argument tensor ABI. It accepts closed seed scopes
and rejects inherited Random rather than baking a lowering-context seed into a
public invocation. The unfused schedule emits seed entry/restore and captures
each forward seed/ordinal once. Replay reads its saved key without advancing the
ambient stream. Empty and rate-zero calls still enter their forward site, even
when their values are discarded. Existing ownership/storage planning controls
allocation, borrowed inputs, result transfer and final cleanup.

The numerical authority is [05-OP-37]/[05-RNG-1] in
`spec/05-risc-primitives.md`. `DropoutParameters` shares the evaluator's stored
rate validation and finalized denominator; generated code reuses unary emission
and declared-dtype loads/stores. Kept values divide by that stored denominator,
not a widened subtraction or reciprocal multiplication. Dropped values are +0;
kept values retain ordinary signed-zero/nonfinite behavior. Static invalid rates
are rejected during emission; that is not a native runtime-trap/stream theorem.
Uniform's existing sampler is unchanged; saved keys avoid extra consumption but
do not establish its agreement with the canonical RNG formula.

## Executable acceptance for this entry

Use the worktree's managed Python/target, one Cargo job and the normal pre-push
`python3 scripts/gate.py --fast`. The owning focused commands are:

- `cargo check -p chelis-backend-c -p chelis-compiler-api -p chelis-ir -p chelis-types --tests`
- `cargo nextest run -p chelis-compiler-api --test fixed_control_c --test-threads 2 --no-fail-fast`
- `cargo nextest run -p chelis-ir -p chelis-types --lib -E 'test(ownership_seal) | test(dropout_parameters) | test(prepared_dropout)' --no-fail-fast`
- `cargo test -p chelis-types -p chelis-backend-c --doc`

All tests must pass. The integration is registered in normal PR CI; doctests
run in the existing types/backend CI owners. It compiles and executes real C
against the ownership-ledger runtime. Cases include all four active float
dtypes with 0/1/32 elements, varied finite inputs and four public invocations;
nested equal/different seeds, saved-mask gradients and the following draw;
f32 signed zero/nonfinite classes; independent next-draw masks after discarded
empty/zero-rate calls, including negative/minimum signed seeds; and unchanged
entry borrows. Executed wrong-ordinal and reciprocal mutations fail value
assertions, and removed cleanup leaves a live owner. Type-level negative tests
reject forged ownership/metadata pairing and runtime tests reject inherited
Random, invalid graph order and duplicate terminal actions.

The source path collects execution metadata in the same lowering that produces
each concrete helper graph and keeps it in an opaque `HostExecutionPlan` until
ownership verification. Private host helpers inherit the invocation-local RNG
frame; inactive state is distinct from active seed zero, and forward/replay
associations cannot be split into independent helper plans. Ordinary public
Rust host structures and the public four-argument C entry ABI are unchanged.
Runtime rates/seeds, unrepresentable staged controls, raw/missing plans and
device targets remain loud. Resource target validation remains independent.
No wire protocol, adapter pin or shell changes are part of this slice.
Same-compilation AD observation collection, native-state/formal refinement,
full numerical campaigns and release/School adoption remain later obligations.
Exact reviewed heads, commands and outcomes belong to this PR's receipts, not
this scope statement.

## Feature-only native Random observation

The non-default `chelis-compiler-api/native-random-observer` feature forwards to
the C backend and emits one translation-unit-private observed wrapper beside
each authored host entry. The published header and ordinary wrapper are
unchanged. An observed invocation supplies its own synchronous sink, context,
and exact u64 identity; the wrapper owns both the observer and RNG state on its
stack. Private host calls and fixed-control tensor helpers receive those two
pointers together, so reentrant and concurrent invocations cannot share an
observer or Random frame unless their test driver deliberately supplies the
same sink context.

Events are taken at the statements that own the runtime values: invocation
initialization, host seed install/restore, fixed-helper enter/leave, forward
post-increment, and replay without increment. Every event carries the current
active flag and exact seed/counter, the complete linked stack of live saved
frames, and an observer-only continuation made by incrementing a copy. Numeric
fields use the private `chelis_random_observer_u64` carrier; the test driver
serializes them as tagged decimal strings and never routes them through a C or
JSON float. The native successor still wraps. A consumer must reject a
`UINT64_MAX` counter through its separate nonwrapping check rather than treating
instrumentation as a change to runtime arithmetic.

Fixed events identify the lowering-selected helper producer and retain its
producer-owned occurrence, draw, and scope IDs where applicable. Replay has no
second source occurrence. Host `WithSeed` currently has only span provenance,
which is not an occurrence authority, so its events say
`HOST_IDENTITY_UNSUPPORTED` and carry no manufactured source ID. Native-to-source
matching across that boundary therefore remains unsupported until lowering
owns an unambiguous mapping.

The sink is called synchronously and may copy or stream the bounded event; the
generated observer retains no history and performs no allocation. A null sink
or nonzero sink result aborts the private harness path, so missing output and I/O
failure cannot become success. The ordinary entry installs no observer and does
not execute the callback. This is runtime observation of an instrumented
artifact, not a certificate, native-to-source proof, mask/derivative proof, or
failure-prefix correspondence result. Evaluator-only invalid rates and
compile-time Resource admission remain in their owning lanes.

The focused acceptance command is:

```text
cargo nextest run --locked -p chelis-compiler-api --test native_random_observer --features native-random-observer --test-threads 1 --no-fail-fast
```

It compiles, links, and executes a nested host/fixed seed program with forward,
replay, restoration, a following draw, repeated invocations, exact active/saved
states, complete helper identities, independent missing/pre-increment/replay
mutants, a failing sink, and a driver-only `UINT64_MAX` rejection control. The
default-feature backend test separately locks the absence of observer vocabulary
from emitted C; the feature-on suite also executes the ordinary public wrapper
and requires that it emit no observation.

Standalone sealed tensor kernels retain their existing ABI and emit no observer
operations, even with the feature enabled: they have no host invocation observer.
The feature-enabled integration-support command runs both
`native_random_observer` and the existing `fixed_control_c` numerical suite;
the latter also runs in the default-feature PR selection. Feature execution is
owned by the nightly/manual `heavy-e2e.yml` frontend support slice, not ordinary
PR success. The solver-free Clippy configurations compile the observer feature.
This is one delivery slice: the private callback carrier, state-owner hooks,
and compile/run tests are useful only together; the feature adds no release ABI.

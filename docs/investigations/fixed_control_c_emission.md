# Closed fixed-control C emission

Part of #1192 and LaCaDiLE's R3 compiler boundary. This is the backend entry,
not ordinary source/API/CLI admission or completion of the multi-backend issue.
HIP/Metal, host helper transport and #1764's public program remain excluded here.

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

The tests lower checked function bodies or typed subexpressions directly. They
do not claim that ordinary `chelis build` now admits dropout. The next transport
slice must preserve the actual source plan through host helper cuts, invoke the
private RNG context, retain Resource admission and cover the public API/CLI.
No normal source guard, public ABI, wire protocol, adapter pin or shell changes
here. Same-compilation collection, native-state/formal refinement, full numerical
campaigns and release/School adoption remain later obligations. Exact reviewed
heads, commands and outcomes belong to this PR's receipts, not this scope statement.

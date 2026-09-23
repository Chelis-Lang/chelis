# Randomness assessment (2026-09-22)

This is the evidence for chelis#2413. It records what three independent investigations measured at chelis `7f9744c6c`. It is a dated record, not a contract:
- `spec/05-risc-primitives.md` [05-RNG-1], [05-OP-8] and [05-OP-37] own the semantics;
- `spec/design/randomness_counter_stream.md` owns the implementation plan;
- `spec/design/randomness_explicit_keys.md` owns the alternative language design.

## Method

The probes were Surf programs run through a release build of `chelis` at `7f9744c6c`:
- through `chelis eval --file`;
- through `chelis build`, then the printed `clang` command, then the binary;
- and, for `uniform_like`, through Metal.

Each result was compared with a reference written from the spec text alone, which calls no Chelis code. That reference implements:
- [05-RNG-1]'s word `splitmix64(seed_bits ^ rotl64(splitmix64(c),17) ^ rotl64(splitmix64(i),41))` and its 53-bit unit value;
- [05-OP-37]'s arithmetic-width mask comparison and finalized `div(x, sub(1, rate))`;
- [05-OP-8]'s fused `fma(high - low, u, low)`.

The same reference also implements the older mixing that the lanes turned out to use for `uniform_like`, so that each output could be attributed to one formula or the other.

The notation `(s, c)` means the reference draw for seed `s` and ordinal `c`.

## The specified model

[05-RNG-1] is a counter-based stream: each value is a pure function of three things:
- the seed;
- the dynamic call ordinal (the count of random primitives entered so far under the active `with seed` handler);
- the element index.

A helper call shares its caller's counter. Unused results, empty tensors and `rate = 0` each consume one ordinal. Dead-code elimination, CSE and fusion keep random nodes live, unmerged and in order (spec/06 §5.2, §5.3, §5.5). The model is consistent and implementable.

Before the 2026-09-23 amendments, the text had five problems:
- no ordinal rule for `vmap` (spec/06 §3.2 defines it as a stack of per-row applications, and §3.3 says it is not a loop);
- no ordinal rule for `par`;
- a nested-handler restore rule stated only in spec/08 §2;
- status and mechanism text in spec/08 §2;
- a pathwise rate cotangent in [05-OP-37].

That last one is a biased estimate of the gradient of the expected result. For inverted dropout, `E[sum(dropout(x, r))] = sum(x)` does not depend on `r`, but the pathwise rate derivative on kept elements, `g·x/(1-r)²`, averages to `x/(1-r)`.

## The implementation

| Lane | Random state | What conforms | What does not |
|---|---|---|---|
| eval, fixed-control plans (`chelis-ir/src/evaluation.rs`) | `RandomExecutionContext {seed, counter}` and a per-site key table | `dropout` at every float dtype, including gradient replay | Only programs the static-controls classifier admits. Every draw site must be entered exactly once. |
| eval, host interpreter (`chelis-compiler-api/src/runtime/eval.rs`) | `random_seed`, `random_counter` | none | Has no `dropout`, so it reports "unknown runtime name". Its `uniform_like` uses the older mixing. |
| plain DAG evaluator (`chelis-ir/src/eval.rs`) | the node's baked `seed` | none | Uses the older `uniform_like` mixing, and an older plan-less dropout formula reachable through eval `vmap` |
| compiled C (`chelis-backend-c/src/host_emit.rs`) | a threaded `chelis_rng_state {seed, counter, active}` | `dropout` in straight-line code, under recursion, and under a host `if` | Rejects `dropout` inside tensor kernels. Uses the older `uniform_like` mixing. Consumes an ordinal for an unselected arm. |
| HIP and Metal | none | none | `dropout` rejected (#1192); older `uniform_like` mixing |

The older `uniform_like` mixing computes `splitmix(seed ^ c·G ^ i·G)`, with G = `0x9E3779B97F4A7C15`. That word is symmetric in `c` and `i`: element `i` of draw `c` equals element `c` of draw `i`, and every diagonal element equals the same seed-only value. The symmetry was checked exhaustively for `c, i < 40`.

`RiscOp::Dropout { rate: f64, seed: u64 }` and `RiscOp::UniformLike { low: f64, high: f64, seed: u64 }` bake the rate, bounds and key into the node. The legacy lowering pre-mixes a lowering-time counter into `seed`.

## Probe matrix

| Probe | eval | C | Class |
|---|---|---|---|
| dropout through helpers; unused result, rate 0, discarded helper result | matches the reference | matches | conforming |
| dropout under a runtime `if` or `match` | "unknown runtime name `dropout`" | rejected (#1192) | misleading rejection |
| dropout under recursion (#2405) | "unknown runtime name `dropout`" | `8·mask(7,0..2)`, then `(7,3)`: correct | lanes disagree |
| dropout with a `match` elsewhere in the program | "unknown runtime name `dropout`" | correct | lanes disagree |
| `uniform_like` | older mixing | older mixing; Metal the same | silently wrong in every lane (#2408) |
| runtime `uniform_like` bounds; `grad` with respect to a rate | rejected | rejected | spec-legal, rejected |
| runtime dropout rate | "unknown runtime name `dropout`" | "requires a statically-resolvable rate (Chelis-Lang/chelis#776)" | misleading rejection (#2411) |
| nested `with seed` | inner restarts at 0; outer resumes at `(7,1)` | same | conforming |
| `grad` through dropout | `2·mask(7,0)`, next draw `(7,1)` | same | conforming |
| `vmap` over `uniform_like`, seeds 7 and 123 | identical output for both seeds: seed 0 | handler seed, one batched ordinal | silently wrong in eval; C is also wrong under the sequential reading decided on 2026-09-23 (#2409) |
| a draw after `vmap` | caller's counter not advanced | advanced by one | lanes disagree (#2409) |
| `uniform_like` under a scalar `if` whose arm is not selected | no ordinal consumed | one ordinal consumed | silently wrong in C (#2410) |
| `par`, `fold` of runtime length, identical calls, f64 with seed -5 | matches the reference | matches | conforming |

The recursion probe (#2405), with a runtime depth argument:

```
def thin_out(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = dropout(x, 0.5f32)
def walk(n: i64, x: tensor[4, f32]) -> tensor[4, f32] ! { Random } =
  if eq(n, 0i64) then x else walk(sub(n, 1i64), thin_out(double(x)))
```

The unselected-arm probe (#2410):

```
def pick(x: tensor[4, f32], flag: bool) -> tensor[4, f32] ! { Random } =
  if flag then uniform_like(x, 0.0f32, 1.0f32) else x
```

It calls `pick` once with `flag` and once with `not(flag)`, then draws twice more. The final draw is `(7,2)` in eval and `(7,3)` in C.

## Why the implementation has two dispatch paths

#1733 introduced conforming dropout only inside fixed-control plans. Those are straight-line schedules in which every draw site runs exactly once, and they are the shape of LaCaDiLE's certified fragment. #1733 left every program outside that shape on the dispatch that already existed. `static_controls.rs` (#1807) and the evaluator's inherited `execution_exclusion` apply that boundary to any program with a runtime `if`, a `match` or recursion, whether or not it draws random numbers.

LaCaDiLE's `design/canonical-revision.md` says of its fragment that it "restricts certification, not Chelis", and that "unsupported certification is not a language rejection". The boundary nevertheless became the evaluator's routing rule. It caused:
- the #2020, #2059 and #2391-#2393 slowdowns;
- eval rejecting programs that C runs.

## LaCaDiLE

LaCaDiLE's `origin/main` at `52a2099` was read and built:
- `lake build LaCaDiLECore LaCaDiLECore.Audit checkCore` takes 110 s;
- every Core test file passes its standard-axioms audit.

The current Core (`lean/LaCaDiLECore/`) implements [05-RNG-1] exactly:
- `Protocol.lean` builds the draw key from the seed and the dynamic ordinal;
- leaving a nested handler restores the parent's seed and ordinal.

It proves the pathwise derivative through dropout along the fixed forward random path (`EffectAD.lean`, `ownedGrad_correct`).

The `DiffCompat = {Resource, Accum}` rule, which forbids `grad` over `Random`, belongs to the legacy calculus that `origin/main` declares historical. The Core's other exclusions are certification scope:
- straight-line code;
- literal rates and seeds;
- no `vmap` and no `uniform_like`;
- no rate cotangent.

Of these exclusions, a stated theorem needs only one, in its unguarded form: the rate and the control flow must not depend on the inputs being differentiated, because the derivative does not exist where the loss jumps.

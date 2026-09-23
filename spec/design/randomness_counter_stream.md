# Randomness as a runtime primitive: the counter stream (option A)

Tracker: chelis#2413. Evidence: `docs/investigations/randomness_assessment_2026_09_22.md`.

This document plans how the implementation reaches the randomness semantics the numbered spec already decides:
- `spec/04-type-system.md` §7.1 (the `Random` effect and `with seed`);
- `spec/05-risc-primitives.md` [05-RNG-1], [05-OP-8] and [05-OP-37];
- spec/06 §2.11 and §3.2;
- spec/07 §2.

It decides no language semantics. Where it restates a rule, the numbered spec wins. The one normative change it needs, to spec/10 §3.2's wire layout, lands with phase 3.

The alternative language design, explicit single-use keys, is `randomness_explicit_keys.md`. Phases 1 to 3, 5 and 6 below are needed under either design. Section 4 says exactly which parts are specific to the counter stream.

## 1. Where the implementation stands

The assessment found that only `dropout` follows [05-RNG-1], and only inside fixed-control execution plans:
- `uniform_like` uses an older mixing whose draws mirror each other in every lane (#2408);
- `vmap` over a random function violates the sequential reading in both lanes: eval draws at seed 0, and C draws the whole batch at one ordinal (#2409);
- compiled C consumes an ordinal for an unselected arm (#2410).

These spec-legal programs are rejected:
- runtime rates and bounds, in every lane (#2411);
- dropout under runtime control or recursion, in eval (#2405), and under runtime control on some compiled C entry paths (#1192, #1872);
- dropout under `vmap`, in C.

Four implementation choices cause this:

1. **Conforming dropout exists only in a static schedule.** A fixed-control plan (`chelis-ir/src/evaluation.rs`) is a straight-line schedule in which every draw site is entered exactly once. The host interpreter has no dropout of its own. The evaluator therefore has to classify a program before it knows which executor can run it (`lower/static_controls.rs`), and it passes the exclusion down to every callee.
2. **Controls are baked into IR nodes.** `RiscOp::Dropout { rate, seed }` and `RiscOp::UniformLike { low, high, seed }` hold compile-time `f64` controls, and in the legacy lowering a seed with the ordinal already mixed in. That forces the static-rate grammar and makes the spec's rate and bound adjoints unreachable.
3. **A scalar `if` lowered into a kernel becomes a `where` that computes both arms.** Only some `UniformLike` nodes carry the activation input that stops an unselected draw from consuming an ordinal.
4. **Gradient replay uses a per-site key table.** Replay keys are indexed by a static draw site, so a site entered zero or several times (recursion, dynamic control) cannot be replayed.

## 2. Target design

**One kernel boundary.** Every random primitive, in every lane, derives its element words through one function:

```
word(key, i) = splitmix64(key XOR rotl64(splitmix64(i), 41))
```

The draw key of the counter stream is `key = seed_bits XOR rotl64(splitmix64(c), 17)`. By associativity of XOR, this is exactly [05-RNG-1]'s word.
- The numerical kernels in `chelis-types/src/dtype_semantics.rs` take the draw key as an argument, as `PreparedDropout::apply` already does with `(seed, ordinal)`. The same applies to a prepared `uniform_like`.
- C calls the same Rust runtime functions.
- HIP and Metal carry bit-tested ports.
- No kernel ever sees an ambient stream.

This boundary is also the one the explicit-key design needs; only the supplier of `key` differs.

**Random state is runtime state.** Each executor holds a random frame: a stack of `(seed, next ordinal)` entries, one per active `with seed` handler.
- **Entering a draw** takes the next ordinal and returns its key.
- **Leaving a handler**, normally or by a trap, restores the enclosing entry, as [05-RNG-1] states.

The frame is held:
- by the host interpreter, which already has `random_seed` and `random_counter`;
- by the DAG evaluator, as an explicit frame argument, since plan-less graph execution goes away;
- by generated C, whose `chelis_rng_state` already exists;
- by GPU launches, which receive the ordinal base and return the count consumed.

**Controls are operands.**
- `Dropout(x, rate)` and `UniformLike(template, low, high)` take their rate and bounds as ordinary scalar inputs of dtype `p`. The spec says so ([05-OP-8], [05-OP-37]).
- A random node carries a static site identity and no key. The executing frame supplies the key when the node is entered.
- Wire and serialized-library layouts change accordingly, under a versioned schema bump. spec/10 §3.2 currently makes the rate and bounds node fields, gives both random nodes a `seed` field, and limits `Dropout.inputs` to the data input. It is amended in the same change.

**Every draw is entered or not, by execution.**
- A draw inside a selected arm is entered.
- A draw inside an unselected arm is not, including when a kernel computes both arms of a lowered scalar `if`. Such a draw carries an activation input, and the kernel's reported count is the number of activated draws.
- A draw inside an argument of `where` is always entered.

**`vmap` and `par` follow the sequential reading ([05-RNG-1]).**
- When the vmapped body enters a static number `k` of draws per row, row `b`'s `j`-th draw takes ordinal `c0 + b·k + j`, which the batched kernel computes from the base `c0`.
- When activation makes per-row counts data-dependent, the row bases are an exclusive prefix sum of the per-row counts. Rows whose counts depend on their own earlier draws are evaluated in row order.
- `par` branch bases are the sequential prefix of branch counts, in source order.

**Gradient replay is an occurrence tape.** Each entered draw in the forward pass appends its site and key to the invocation's tape. The backward pass, and checkpoint recomputation, read keys back from the tape without touching the frame. This admits recursion, dynamic control and nested `grad` with no per-site restriction (spec/06 §2.10.1). The tape replaces the per-site key table and the static spine requirement.

**The rate is not differentiable.** Following the amended [05-OP-37], adjoint construction rejects a rate that depends on a differentiated input, with `AdRejectionReason::RandomSelectionParameter`. `stop_gradient(rate)` is the explicit way to treat such a rate as constant. `uniform_like`'s bounds keep their exact reparameterisation adjoints.

**What is deleted:**
- the static-controls classifier and `EvaluationProfile` (`FixedControl`/`Legacy(reason)`);
- `execution_exclusion` and its per-reason routing;
- the legacy random lowering that pre-mixes a lowering-time counter into `seed`;
- the static-rate grammar for `dropout`/`uniform_like` (`static_rate`, `resolve_static_scalar_arg` for random controls) and its #776-citing diagnostics;
- the plan-less dropout formula (`chelis-ir/src/eval.rs` ~373-403);
- the older uniform mixing (`uniform_sample`'s `seed ^ i·G`, the interpreter's `seed ^ c·G`, and the emitted C equivalents).

Fixed-control plans survive only as what the general frame and tape subsume. LaCaDiLE certification export remains opt-in and keeps producing its fragment's graphs from the same frame and tape.

## 3. Phases

Each phase is one pull request with its own red-team rounds. The oracle for every phase includes a spec-derived bit reference: an independent transcription of [05-RNG-1], [05-OP-8] and [05-OP-37] in Rust test code, never calling an evaluator helper, as `dropout_fixed_stream_api.rs` already does. It is run over a stream corpus that grows with each phase: helpers, unused results, nested seeds, runtime control, recursion, `vmap`, `par`, runtime controls and `grad`. Eval and compiled C must both match it bit for bit wherever both run.

1. **Per-definition kernel decisions (#2405).** Classify only where `dropout` is reachable. Each definition's kernel is decided from its own body, as the C lane already does. The inherited exclusion is gone.
   - Oracle: dropout under recursion, and in a program with an unrelated `match`, runs in eval and equals C's bits.
   - Dropout-free programs are unchanged.
2. **One generator (#2408).** `uniform_like` joins the kernel boundary above in every lane. The older mixing and the plan-less dropout formula are deleted.
   - This changes every `uniform_like`-derived value: `normal_like`, the Kaiming and Xavier initialisers, and shell distributions. Shells' pinned random outputs are regenerated in the release that ships it, with a migration note.
3. **Runtime primitive (#2411).**
   - Controls become operands and keys come from the frame.
   - The interpreter gains conforming `dropout` and `uniform_like` builtins.
   - Plans become the frame plus the tape; the classifier, the profile and the legacy random lowering are deleted.
   - spec/10 §3.2's wire layout is amended to carry the operands and drop the baked key.
   - Oracle: runtime rates and bounds run in eval and C and match the reference; dropout under a runtime `if` or `match` runs in eval; nothing selects an executor by static classification.
4. **Lane defects.**
   - `vmap` under the sequential reading (#2409);
   - activation on every lowered unselected draw (#2410);
   - the direct-return result-claim failure (#2407).
   - Oracle: the vmap and unselected-arm probes match the reference in both lanes, and #2407's program runs.
5. **Rate rejection.** `AdRejectionReason::RandomSelectionParameter` and its registry row, per the amended [05-OP-37] (#2421).
   - Oracle: a rate reached through adjoint-contract operations rejects with the typed reason; the same program with `stop_gradient(rate)`, or with a rate independent of the parameters, differentiates.
6. **Compiled dropout everywhere (#1192, #1872).** Tensor kernels, HIP and Metal use the kernel boundary. The remaining compiled-lane dropout rejections are removed.
   - Oracle: the stream corpus builds and runs on C, HIP and Metal and matches the reference.

## 4. What is specific to the counter stream

Only these parts would change under the explicit-key design:
- the frame's `(seed, ordinal)` stack and the key formula `seed_bits XOR rotl64(splitmix64(c), 17)`;
- the ordinal bookkeeping for activation, `vmap` row bases and `par` branch bases;
- the `with seed` handler plumbing in each executor.

Everything else carries over unchanged: the kernel boundary, operands instead of baked controls, the occurrence tape, rate rejection, and the deletions.

The counter stream's lasting costs are these:
- a draw's value depends on how many draws preceded it, so adding a draw to a library helper shifts every later value in its callers;
- regions whose draw counts depend on data serialise through the shared counter.

## 5. LaCaDiLE

LaCaDiLE's Core (`lean/LaCaDiLECore`) formalises exactly this stream and proves the pathwise gradient through dropout along the fixed forward path. Certification stays an opt-in check over its fragment. A program outside the fragment gets no certificate and runs normally. The Core does not model a rate cotangent, so the amended [05-OP-37] agrees with it.

`design/canonical-revision.md` lines 36-38 direct that compiled rates be statically resolvable and that unresolved rates be rejected. That directive contradicts [05-OP-37]'s runtime rate, and it should be amended in LaCaDiLE. The certified fragment can keep literal rates.

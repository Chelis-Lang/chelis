# Randomness as a runtime primitive: shared phases and the counter bridge

Tracker: chelis#2413. Evidence: `docs/investigations/randomness_assessment_2026_09_22.md`.

This document plans how the implementation reaches the randomness semantics the numbered spec already decides:
- `spec/04-type-system.md` §7.1 (the `Random` effect and `with seed`);
- `spec/05-risc-primitives.md` [05-RNG-1], [05-OP-8] and [05-OP-37];
- spec/06 §2.11 and §3.2;
- spec/07 §2.

It decides no language semantics. Where it restates a rule, the numbered spec wins. The one normative change it needs, to spec/10 §3.2's wire layout, lands with phase 3.

On 2026-09-23 Chelis decided to move to explicit single-use keys (`randomness_explicit_keys.md`). Phases 1 to 3, 5 and 6 below are shared by both designs. Phase 3's key-operand IR is the explicit-key IR, with a bridge that computes today's counter keys until the switch. No new counter-only ordinal machinery is built (new activation counting, `vmap` and `par` bases), because keys remove ordinals.

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

**Random nodes take a key operand.** In the IR the data input stays first, because `shape_source_for_axis` reads `inputs[0]` as the tensor:
- `Dropout[x, rate, key(, active)]`;
- `UniformLike[template, low, high, key(, active)]`.

Rate and bounds are ordinary scalar operands of dtype `p`, as [05-OP-8] and [05-OP-37] say. `key` is a new non-numeric dtype (`Prim::Key`) with an opaque carrier (`RandomKey` in `dtype_semantics.rs`), and kernels are pure once they have their key. The optional Bool `active` input keeps an unselected `where`-lowered arm from validating, and so trapping on, a runtime control.

The verifier adds three rules:
- each key value has at most one consuming use;
- AD replay nodes read their forward node's key without consuming it;
- a key given to any other operation is rejected.

This is the explicit-key IR. Only the key source changes at the switch.

**The bridge key source.** Until the switch, `DrawKey{handler: Inherited | Scoped{instance}}` produces the key of its handler's next ordinal, `ofDrawKey(seed, c) = seed_bits XOR rotl64(splitmix64(c), 17)`, from a runtime frame:
- in the host interpreter, today's `random_seed` and `random_counter`, with no seed-0 fallback;
- in the DAG evaluator, a `RandomFrame` argument;
- in generated C, today's `chelis_rng_state`.

`DrawKey` is effectful: a DCE root, never merged, never folded or recomputed, and ordered as today. It advances the frame exactly when today's lowering consumes an ordinal. Where lowering attaches today's `random_path_condition` (`lower.rs`, the AD transform subcontexts), `DrawKey` carries it as an activation and advances only when it is true; nowhere else is an activation added. It takes its kernel's controls as ordering inputs and, when active, advances the frame only after they validate, preserving [05-OP-37]'s "validation consumes no ordinal". When inactive, it neither validates nor advances. A scoped `with seed` inside a kernel carries its literal seed.

`vmap` over a function that draws is refused with a typed #2409 rejection in both lanes until the switch defines it. Today eval draws at seed 0, and C draws one batched ordinal. Both are silently non-conforming, so they become loud.

**Gradient replay reads the key edge.** `DropoutReplay` and `UniformBoundAdjoint` read the forward node's key, and recomputation gives the same bits. That edge replaces the per-site key table, its "entered twice" check and the static-spine requirement. Recursion, dynamic control and nested `grad` need nothing more (spec/06 §2.10.1).

**Wire.** WireDag moves to the next schema version. The cached lowered library bumps its format version, and spec/10 §3.2 is amended in the same change: random nodes carry no fields; their inputs are the data or template, the controls, the key and an optional activation; and `key` is a structural precision with no literal carrier, so every key is produced by a node.

**The rate is not differentiable.** Following the amended [05-OP-37], adjoint construction rejects a rate that depends on a differentiated input, with `AdRejectionReason::RandomSelectionParameter`. `stop_gradient(rate)` is the explicit way to treat such a rate as constant. `uniform_like`'s bounds keep their exact reparameterisation adjoints.

**What is deleted:**
- the static-controls classifier and `EvaluationProfile` (`FixedControl`/`Legacy(reason)`);
- `execution_exclusion` and its per-reason routing;
- the legacy random lowering that pre-mixes a lowering-time counter into `seed`;
- the static-rate grammar for `dropout`/`uniform_like` (`static_rate`, `resolve_static_scalar_arg` for random controls) and its #776-citing diagnostics;
- the plan-less dropout formula (`chelis-ir/src/eval.rs` ~373-403);
- the older uniform mixing (`uniform_sample`'s `seed ^ i·G`, the interpreter's `seed ^ c·G`, and the emitted C equivalents).

Fixed-control plans and the random parts of `evaluation.rs` and `execution_spine.rs` are deleted: the key edge subsumes them. Three consumers of those parts are kept:
- the spine's Resource-requirement capture moves to `lowering_trace.rs`, because the `compilation-trace` feature reads it;
- the `compilation-trace` feature's fixed-entry selection (`compiler.rs`, around lines 2188-2205) selects the lowered kernel with its `DrawKey` nodes;
- `random_observer.rs` is ported to observe `DrawKey` nodes until the switch deletes it.

LaCaDiLE certification export remains opt-in and reads the symbolic key derivations.

## 3. Phases

Each phase is one pull request with its own red-team rounds. The oracle for every phase includes a spec-derived bit reference: an independent transcription of [05-RNG-1], [05-OP-8] and [05-OP-37] in Rust test code, never calling an evaluator helper, as `dropout_fixed_stream_api.rs` already does. It is run over a stream corpus that grows with each phase: helpers, unused results, nested seeds, runtime control, recursion, `vmap`, `par`, runtime controls and `grad`. Eval and compiled C must both match it bit for bit wherever both run.

1. **Per-definition kernel decisions (#2405).** Classify only where `dropout` is reachable. Each definition's kernel is decided from its own body, as the C lane already does. The inherited exclusion is gone.
   - Oracle: dropout under recursion, and in a program with an unrelated `match`, runs in eval and equals C's bits.
   - Dropout-free programs are unchanged.
2. **One generator (#2408).** `uniform_like` joins the kernel boundary above in every lane. The older mixing and the plan-less dropout formula are deleted.
   - This changes every `uniform_like`-derived value: `normal_like`, the Kaiming and Xavier initialisers, and shell distributions. Shells' pinned random outputs are regenerated in the release that ships it, with a migration note.
3. **Key-operand IR with the bridge (#2411, #2421).** One PR, in six commits:
   1. the key carrier and the kernel signatures `apply(key)`;
   2. a mechanical rename of the old nodes;
   3. the new nodes, `Prim::Key`, the verifier rules, the `RandomFrame` evaluator, `grad` (including the `RandomSelectionParameter` rejection, which phase 3 needs because it is the first phase in which a rate can depend on a parameter) and C emission;
   4. conforming `dropout` and `uniform_like` interpreter builtins;
   5. the switch of lowering to the new nodes, deleting plans, the spine, the classifier, the profile, the static-rate grammar and the old nodes, with the wire and cache version bumps;
   6. spec/10 §3.2 and the changelog.

   Oracles:
   - every program that ran at the base produces identical bits, except programs that `vmap` a random function, which are now refused (#2409);
   - the reference-match oracle excludes C programs that draw in an unselected arm, which stay shifted by one ordinal until the switch (#2410);
   - runtime rates and bounds, and dropout under runtime `if`/`match`/recursion, run in eval and C and match the reference;
   - a rate reached through adjoint-contract slots rejects with `RandomSelectionParameter`, with the rejection registry regenerated, while `stop_gradient(rate)` and a rate independent of the parameters both differentiate;
   - no `EvaluationProfile` remains.
   - Before the fence is added, the shells are searched for `vmap` over random functions, and any use is reported.
4. **Lane defects.** The direct-return result-claim failure (#2407).
   - Oracle: #2407's program runs.
   - #2409's `vmap` ordinals are not built: phase 3 fences them, and the switch defines `vmap` over keys.
5. **Rate rejection.** Folded into phase 3.
6. **Compiled dropout everywhere (#1192, #1872).** Tensor kernels, HIP and Metal use the kernel boundary. The remaining compiled-lane dropout rejections are removed.
   - Oracle: the stream corpus builds and runs on C, HIP and Metal and matches the reference.

## 4. What the switch to explicit keys deletes

These parts are the bridge. The switch deletes them:
- `DrawKey`;
- the `RandomFrame` and the interpreter's counter;
- `chelis_rng_state` and its threading;
- the `with seed` plumbing in each executor.

Everything else carries over unchanged: the kernel boundary, operand controls, the key edge for replay, rate rejection, and the deletions above.

The counter stream's lasting costs are these:
- a draw's value depends on how many draws preceded it, so adding a draw to a library helper shifts every later value in its callers;
- regions whose draw counts depend on data serialise through the shared counter.

## 5. LaCaDiLE

LaCaDiLE's Core (`lean/LaCaDiLECore`) formalises exactly this stream and proves the pathwise gradient through dropout along the fixed forward path. Certification stays an opt-in check over its fragment. A program outside the fragment gets no certificate and runs normally. The Core does not model a rate cotangent, so the amended [05-OP-37] agrees with it.

`design/canonical-revision.md` lines 36-38 direct that compiled rates be statically resolvable and that unresolved rates be rejected. That directive contradicts [05-OP-37]'s runtime rate, and it should be amended in LaCaDiLE. The certified fragment can keep literal rates.

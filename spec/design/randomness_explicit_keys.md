# Explicit single-use random keys (option C)

Tracker: chelis#2413. Status: **decided 2026-09-23 (Robert).** Chelis moves from the counter stream of [05-RNG-1] to explicit keys, after the phases shared by both designs (`randomness_counter_stream.md` phases 1 to 3).

Until the language change lands, the numbered spec still specifies the counter stream, and every lane must keep meeting it. The numbered chapters are amended in the same change set as the switch (§5, step 3). This document plans that change; it does not decide semantics ahead of those amendments.

## 1. Why

Under the counter stream, a draw's value depends on how many random primitives were entered before it, under the same handler, anywhere in the dynamic call tree. That has three costs:
- **Reproducibility is non-local.** Adding one draw to a library helper shifts every later value in every caller, as with a global generator (PyTorch).
- **Transformations need their own ordinal rules.** `vmap` and `par` each need one, and a batched kernel must compute per-row ordinal bases, with a prefix sum when per-row draw counts depend on data. functorch had to add `vmap(randomness='different'|'same'|'error')` for this reason.
- **Random operations are effects.** Dead-code elimination, CSE and reordering must keep every random primitive and its ordinal.

With explicit keys, as in JAX, a random primitive is a pure function of a key value it is given. The dependence is visible in the program text, and transformations compose with randomness through their ordinary rules. The project's tenets favour this:
- explicit over implicit: no hidden effects;
- unambiguity over ergonomics: the author is an agent, for whom threading a key costs nothing;
- composition over special cases: no ordinal rule per transformation.

The counter stream is the special case of this design in which the compiler picks one key schedule. LaCaDiLE proves it: `Key.ofDrawKey(seed, c) = seed XOR rotl64(splitmix64(c), 17)`, and `keyMask_ofDrawKey` shows the counter mask equals the keyed mask at that key (LaCaDiLE PR #80).

## 2. Surface

| Form | Meaning |
|---|---|
| `key` | A non-numeric element dtype: no arithmetic, cast or comparison. A scalar `key` is a value; `tensor[n, key]` holds a batch of keys. |
| `key(seed: i64) -> key` | The root key of a seed: the seed's two's-complement bits, with no mixing. |
| `split(k: key) -> (key, key)` | `(derive(k, 0), derive(k, 1))`. |
| `split(k: key, n) -> tensor[n, key]` | Row `j` is `fold_in(k, j)`. |
| `fold_in(k: key, n: i64) -> key` | `derive(derive(k, 2), n)`. |
| `dropout(k: key, x: &tensor[D,p], rate: p) -> tensor[D,p]` | [05-OP-37], with `k` in place of the handler's seed and ordinal. |
| `uniform_like(k: key, t: &tensor[D,p], low: p, high: p) -> tensor[D,p]` | [05-OP-8], likewise. |

Here `derive(k, j) = splitmix64(k XOR rotl64(splitmix64(j), 29))`. These are LaCaDiLE PR #80's definitions, so the language and the model agree bit for bit. The derivation is a finalised mix, never a bare XOR of per-index terms: a chained `fold_in(fold_in(k, a), b)` would otherwise be symmetric in `a` and `b`, the defect #2408 found in the older `uniform_like` mixing.

**Keys are affine: each key is used at most once.** Passing a key to a random primitive, `split` or `fold_in` consumes it. The compiler never inserts a `copy` of a key, so reuse is a type error, and dropping an unused key is allowed. Affinity propagates to tuples, lists and data types that contain a key. The JAX idiom of repeated `fold_in(k, step)` on one retained key is written `split(k, n)` instead: consuming `k` once yields `n` keys.

**The `Random` effect and `with seed` are deleted.** A function that draws takes a `key` parameter, and its signature is the marker. The unhandled-`Random` check error goes, and so does the escape class #2318 guards against. `chelis manifest` reports key provenance instead of `Random`-annotated operations.

## 3. Semantics

**Words.** The element word is `word(k, i) = splitmix64(k XOR rotl64(splitmix64(i), 41))`, with [05-RNG-1]'s unit value. The mask and affine rules of [05-OP-37] and [05-OP-8], their dtype rules and their pre-draw validation are unchanged.

**Transformations:**
- `vmap` over a `tensor[n, key]` axis gives each row its own key. A captured key used in every row is a linearity error, not an implicit "same randomness" mode.
- `par` needs no rule.
- CSE, reordering and dead-code elimination treat a random primitive as they treat any pure operation.
- `grad` treats a key as a discrete input with `unit` cotangent. Recursion, control flow and nested `grad` need nothing more.
- Reading a consumed key's bits again in the backward pass, or in checkpoint recomputation, is not a second use.

**The rate.** It keeps the amended [05-OP-37] contract. The rejection `AdRejectionReason::RandomSelectionParameter` applies when a differentiated parameter reaches the rate through adjoint-contract slots; any other rate gets the exact zero cotangent.

**Carriers.** A key is a structurally non-numeric tagged carrier:
- `Prim::Key` in the IR;
- a `key` precision in the wire format;
- `{"type":"key","bits":"<16 lowercase hex>"}` in execution values;
- an opaque published `chelis_key` in the C ABI.

It is never a bare integer (`dtype_semantics.md` §C6).

## 4. The shared IR this builds on

Phase 3 of `randomness_counter_stream.md` gives random nodes a key operand. That IR is already C's final shape; only the key source changes when C lands:
- in the interim, the bridge op `DrawKey` computes `ofDrawKey` from today's counter frame;
- under C, keys come from `KeyFromSeed`, `Split`, `SplitN` and `FoldIn` nodes.

The following are unchanged by the switch: the kernels, the random nodes, their adjoints, their wire shape, the key-consume-once verifier rule and the replay reads.

The bridge exists to give the IR rewrite a bit-identical oracle. Every program that ran before phase 3 must produce the same bits after it. Only the switch changes random outputs.

## 5. What changes

**Spec.**
- **spec/02:** remove the `with seed` grammar and `Random` from the built-in effect names. Add the `key` dtype and the key constructors. Random calls take the key first.
- **spec/03:** `handle-effect` loses its `random` kind, leaving `resource`. Add `(t-prim {} key)`.
- **spec/04:**
  - §1.1 gains `key`;
  - §7.1 and [04-EFF-1] remove `Random`, `with seed` and the unhandled-`Random` error;
  - the linearity rules make keys affine;
  - the seed-literal note goes.
- **spec/05:**
  - [05-RNG-1] is recast over keys;
  - new atoms are added for `key`, `split`, `fold_in` and `derive`, each non-differentiable and each consuming its parent;
  - [05-OP-8] and [05-OP-37] take a key operand, and the §2.6 table follows;
  - the Box-Muller helper splits its key.
- **spec/06:**
  - §2.11 gets the key cotangent and replay rules;
  - §3.2 gets the `vmap` rule over key rows;
  - §5.3 drops "Random nodes never merge".
- **spec/07:** drop the `par` ordinal sentence.
- **spec/08:** replace host Random state with key carriers for public entries.
- **spec/10:** WireDag gains the key ops and loses `DrawKey`; execution values gain the key carrier.

**Implementation, after phase 3:**
1. Key operations in the IR, verifier, evaluators and C emission, with the next wire version.
2. The checker: the `key` dtype, affine linearity, and key-typed signatures.
3. **The switch, in one change set.** The surface, host lowering of key values, C host key locals, public-entry key parameters and the numbered-spec amendments land together. Deleted in the same change set: `DrawKey`, the counter frame, `chelis_rng_state` and its threading, the `with seed` host form, the random observer, and the `Random` effect rows.
4. Carriers: execution values, a published `chelis_key` with its census row, and Python bindings.
5. chelis-std (`normal_like` splits its key for its two uniforms; the initialisers take a key), then the examples, with `reef.lock` and the dist binaries committed.
6. A breaking changelog fragment.

**Shells** (a minor release; every shell bumps):
- **nautilus:** its samplers take a key; two-draw samplers split it.
- **shoals:** path generators use `split(k, n_steps)`; tests pass `key(7i64)`; pinned outputs and manual gates are regenerated.
- **school:** the dropout layer's private counter becomes a key argument. `random_unit()`, stochastic depth and model constructors take keys, and the training loop splits a key per step.
- **hull:** its `with seed` handler and `(seed, ordinal)` state are replaced by key paths mirroring LaCaDiLE's `KeyPath`, in lockstep with the compiler.
- **calcify and hydronnx:** they emit key threading.

**Superseded issues.** #2409 (`vmap` ordinals) and #2410 (unselected-arm ordinals) close when the switch lands, because keys remove ordinals. Until then, the phase 3 bridge keeps today's behaviour for both, which is out of spec and tracked. Anything silently wrong in a new way must be fenced.

## 6. LaCaDiLE

LaCaDiLE PR #80 models keys beside its counter protocol. The keyed modules prove three results with no `sorry` and only the standard axioms:
- the pathwise derivative at every key environment;
- key affinity, both in sources and in accepted schedules;
- validator soundness for keyed schedules.

For certification to extend to keyed Chelis programs, the compiler must meet three conditions:
- key derivations stay symbolic in exported graphs: key values are always produced by nodes, never literal bits;
- `split` and `fold_in` consume their parent;
- replay reads are not uses.

Certification completeness is not a Chelis priority. Certificates valid for every key, and keys inside data types or across calls, remain LaCaDiLE follow-up work.

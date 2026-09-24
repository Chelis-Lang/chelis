# Explicit single-use random keys (option C)

Tracker: chelis#2413. Status: **decided 2026-09-23 (Robert); names, key tensors, closures and branch arms decided 2026-09-24.** Chelis moves from the counter stream of [05-RNG-1] to explicit keys, after the phases shared by both designs (`randomness_counter_stream.md` phases 1 to 3).

Until the language change lands, the numbered spec still specifies the counter stream, and every lane must keep meeting it, apart from the gaps tracked under #2413. The numbered chapters are amended together with the implementation steps in §5, each step with the text it implements. This document plans that change; it does not decide semantics ahead of those amendments.

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
| `key` | A non-numeric element dtype: no arithmetic, cast, comparison, literal or default. A scalar `key` is a value; `tensor[n, key]` is an ordinary tensor of keys. |
| `key_from_seed(seed: i64) -> key` | The root key of a seed: the seed's two's-complement bits, with no mixing. |
| `split_key(k: key) -> (key, key)` | `(derive(k, 0), derive(k, 1))`. |
| `split_keys(k: key, n: i64) -> tensor[n, key]` | Row `j` is `fold_in(k, j)`. `n` is a runtime value, `n >= 0`. |
| `fold_in(k: key, n: i64) -> key` | `derive(derive(k, 2), n)`, with `n` read as its two's-complement 64 bits. |
| `dropout(k: key, x: &tensor[D,p], rate: p) -> tensor[D,p]` | [05-OP-37], with `k` in place of the handler's seed and ordinal. |
| `uniform_like(k: key, t: &tensor[D,p], low: p, high: p) -> tensor[D,p]` | [05-OP-8], likewise. |

The names avoid `split(x, axis, sizes)` ([05-OP-53]) and the `key` dtype name, so nothing is overloaded. Here `derive(k, j) = splitmix64(k XOR rotl64(splitmix64(j), 29))`; `derive` is spec-only ([05-RNG-2]), not a builtin, since exposing it would give `split_key`'s halves a second spelling. These are LaCaDiLE PR #80's definitions, so the language and the model agree bit for bit. The derivation is a finalised mix, never a bare XOR of per-index terms: a chained `fold_in(fold_in(k, a), b)` would otherwise be symmetric in `a` and `b`, the defect #2408 found in the older `uniform_like` mixing.

**Keys are affine: each key is used at most once.** Passing a key to a random primitive, `split_key`, `split_keys` or `fold_in` consumes it, and so does passing a key holder to a `grad(f)(...)` or `vmap(f)(...)` call, whose arguments are otherwise observed. The rule is stated on the category, not per container: the checker has one uniqueness analysis keyed on tensor evidence and no affine mode to reuse, so keys get a parallel `key_carrying` evidence predicate beside `tensor_carrying` (tuples, records, lists and data types recursively, and `tensor[n, key]`; function types carry none) and a per-binding affine mark. A value with key evidence cannot be borrowed, cannot be copied (whether by an explicit or a compiler-inserted `copy`), and has no read that leaves it live. A key is reached only by consuming its holder:
- a destructuring `match` or `let` pattern;
- `vmap` over a key axis, which gives each row to one application;
- a key-consuming operation.

Backward-pass replay reads are the only exception. Reuse is therefore a type error, and dropping an unused key is allowed.

**A closure may not capture a key or a key holder.** Function types record no captures, and a closure that consumed a captured key would use it once per call. Capturing one is a type error; keys are passed as parameters. Affine closures would be a separate feature.

**Keys under a where-lowered `if` or `match` (rule V3).** Inside a kernel both arms of a runtime branch are lowered, each draw carrying its arm's activation. Consumption is counted per selected arm: two consumers may share a key only when their activations are structurally exclusive, `And(P, X)` against `And(P, Not X)` or any conjunct chain containing such a pair. Refusing keys inside arms instead would fence ordinary programs such as `if c then dropout(k, x, r) else x`.

The JAX idiom of repeated `fold_in(k, step)` on one retained key is written `split_keys(k, n)` instead: consuming `k` once yields `n` keys.

**The `Random` effect and `with seed` are deleted.** A function that draws takes a `key` parameter, and its signature is the marker. The unhandled-`Random` check error goes, and so does the escape class #2318 guards against.

## 3. Semantics

**Words.** The element word is `word(k, i) = splitmix64(k XOR rotl64(splitmix64(i), 41))`, with [05-RNG-1]'s unit value. The mask and affine rules of [05-OP-37] and [05-OP-8], their dtype rules and their pre-draw validation are unchanged.

**Transformations:**
- `vmap` over a `tensor[n, key]` axis gives each row its own key. `vmap` maps no other scalar formal, so a vmapped key formal requires a mapped `tensor[n, key]` actual; a scalar key actual, or a captured key used in every row, is a type error, not an implicit "same randomness" mode.
- `par` needs no rule.
- CSE, reordering and dead-code elimination treat a random primitive as they treat any pure operation.
- `grad` treats a key as a discrete input with `unit` cotangent. Recursion, control flow and nested `grad` need nothing more.
- Reading a consumed key's bits again in the backward pass, or in checkpoint recomputation, is not a second use.

**The rate.** It keeps the amended [05-OP-37] contract. The rejection `AdRejectionReason::RandomSelectionParameter` applies when a differentiated parameter reaches the rate through adjoint-contract slots; any other rate gets the exact zero cotangent.

**Carriers.** A key is a structurally non-numeric tagged carrier:
- `Prim::Key` in the IR, a tensor element dtype like `bool`;
- runtime dtype `key` with id 9 (one 64-bit word, no arithmetic representation), so a `tensor[n, key]` is a `chelis_tensor` of dtype key; DLPack refuses key tensors with a typed rejection;
- a `key` precision at any rank in the wire format;
- `{"type":"key","bits":"<16 lowercase hex>"}` in execution values;
- an opaque published `chelis_key` for scalar keys at public entries.

It is never a bare integer (`dtype_semantics.md` §C6). Adding the runtime dtype changes the #893 runtime vocabulary (`chelis-vocab` `RuntimeDType`, the sealed `TensorElement` set, `chelis_runtime_dtype.h`), so it is a Phase 0 inventory freeze move (`runtime_representation.md` §B1), coordinated with that plan: the new element spellings route through the existing dtype authorities.

## 4. The shared IR this builds on

Phase 3 of `randomness_counter_stream.md` gives random nodes a key operand. That IR is already C's final shape; only the key source changes when C lands:
- in the interim, the bridge op `DrawKey` computes `ofDrawKey` from today's counter frame;
- under C, keys come from `KeyFromSeed`, `Split{branch}`, `SplitN{count}` and `FoldIn` nodes, or enter as key-typed `Load`s. `split_key` is two nodes, `Split{Left}` and `Split{Right}` (LaCaDiLE's `KeyPath.left/right`), because an IR node has one output. Key derivations are never constant-folded, so exported graphs keep them symbolic.

The following are unchanged by the switch: the scalar-key kernels, the random nodes, their adjoints, their wire shape and the replay reads. The key-consume-once verifier rule widens: a key is produced by a key operation, a `DrawKey` or a key-typed `Load`, and may be a root (V1); its one use is one draw, one `FoldIn`, one `SplitN` or one root, or at most one `Split` per branch, every `Load` of one parameter is one key, and a `DrawKey`'s key is used only by a draw (V2); rule V3 above admits exclusive activations; a key reaching any other operation, `Where` or a shape dependency is rejected (V4); and a random node may take a key tensor of any rank whose shape is its data's leading axes, with row `b`'s words `word(key[b], i)` and each control, activation or bound adjoint shaped like a leading part of the key's shape, so `vmap` composes over draws (V5, a new shape rule and a per-row kernel).

The bridge exists to give the IR rewrite a bit-identical oracle. Phase 3 states which programs keep identical bits.

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
  - new atoms are added for `derive`, `key_from_seed`, `split_key`, `split_keys` and `fold_in`, each non-differentiable and each derivation consuming its parent;
  - [05-OP-8] and [05-OP-37] take a key operand, and the §2.6 table follows;
  - the Box-Muller helper splits its key.
- **spec/06:**
  - §2.11 gets the key cotangent and replay rules;
  - §3.2 gets the `vmap` rule over key rows;
  - §5.3 drops "Random nodes never merge".
- **spec/07:** drop the `par` ordinal sentence.
- **spec/08:** replace host Random state with key carriers for public entries.
- **spec/10:** WireDag gains the key ops and loses `DrawKey`; execution values gain the key carrier.

**Implementation, after phase 3.** Each step lands with the numbered-spec text it implements.
1. **Additive key operations.** The `key` dtype (spec/04 §1.1) and new atoms for `derive`, `key_from_seed`, `split_key`, `split_keys` and `fold_in` (spec/05). The runtime dtype (a freeze move, §3). The IR operations, verifier, evaluators and C and HIP emission for them, and the `vmap` lifting over key rows. The next wire version, with spec/10 amended in the same step, including relaxing its "every key is the output of a `DrawKey`" sentence, since key `Load`s and roots become legal.
2. **The checker.** Affine linearity for keys and key tensors (spec/04's linearity rules) through the `key_carrying` predicate, the closure refusal, consumed transform-call arguments, the `vmap` key-formal rule, and key-typed signatures.
3. **The switch, in one change set.** It lands together:
   - removal of phase 3's `vmap` fence (#2409), now that `vmap` over key rows is defined;
   - the surface;
   - host lowering of key values, and C host key locals;
   - public-entry key parameters, with the published `chelis_key` carrier, its census row, execution values and Python bindings (key tensors cross as `chelis_tensor`s of dtype key);
   - chelis-std: `normal_like` splits its key for its two uniforms, the initialisers take a key, and `stdlib_numeric_manifest.md`'s signatures follow;
   - the examples, with `reef.lock` and the dist binaries committed;
   - the remaining numbered-spec amendments, with the guard artifacts that pin their text: `scripts/dtype_phase4b_oracle.py`'s [05-OP-37] phrases and their tests.

   Deleted in the same change set: `DrawKey`, the counter frame, `chelis_rng_state` and its threading, the `with seed` host form, the random observer, the `Random` effect rows (the effects-derived root manifest loses them), and the `random` handler kind in `chelis-vocab` `EffectKind`, `chelis-deep` metadata and `annotations_codec`.
4. A breaking changelog fragment, and the shell releases.

**Shells** (a minor release; every shell bumps). The affected set is derived at switch time, not listed here. Search every Chelis-Lang repository and spec document for `with seed`, `Random`, and each random builtin and helper. That search already includes hello-chelis, the spec registry, and the canonical reference's randomness paragraphs. Hull moves in lockstep with the compiler, because it differential-tests it.

**Superseded issues.** #2409 (`vmap` ordinals) closes when the switch lands, because keys remove ordinals. #2410 (unselected-arm ordinals) closes in phase 3, where every draw under a runtime branch carries its path condition as its activation. Until the switch:
- phase 3 refuses `vmap` over a function that draws, because eval's bits would otherwise change to a different non-conforming value;
- anything silently wrong in a new way must be fenced.

## 6. LaCaDiLE

LaCaDiLE PR #80 models keys beside its counter protocol. The keyed modules prove three results with no `sorry` and only the standard axioms:
- the pathwise derivative at every key environment;
- key affinity, both in sources and in accepted schedules;
- validator soundness for keyed schedules.

For certification to extend to keyed Chelis programs, the compiler must meet three conditions:
- key derivations stay symbolic in exported graphs: key values are always produced by nodes, never literal bits;
- `split_key`, `split_keys` and `fold_in` consume their parent;
- replay reads are not uses.

Certification completeness is not a Chelis priority. Certificates valid for every key, and keys inside data types or across calls, remain LaCaDiLE follow-up work.

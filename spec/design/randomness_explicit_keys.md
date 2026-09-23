# Explicit single-use random keys (option C): a proposed language change

Tracker: chelis#2413. Status: **proposal, not decided.** The numbered spec still specifies the counter stream of [05-RNG-1], and `randomness_counter_stream.md` plans the implementation of that stream. This document describes the alternative language design, what it would change, and how to move to it if it is chosen. Nothing here is normative until the decision is recorded and the numbered chapters are amended.

## 1. Why consider it

Under the counter stream, a draw's value depends on how many random primitives were entered before it, under the same handler, anywhere in the dynamic call tree. That has three costs:
- **Reproducibility is non-local.** Adding one draw to a library helper shifts every later value in every caller, the reproducibility failure mode of a global generator (PyTorch).
- **Transformations need extra ordinal rules.** `vmap` and `par` need their own ordinal rules, and a batched kernel must compute per-row ordinal bases, with a prefix sum when per-row draw counts depend on data. functorch had to add `vmap(randomness='different'|'same'|'error')` for the same reason.
- **Random operations cannot be treated as ordinary operations.** Dead-code elimination, CSE and reordering must treat every random primitive as an effect and keep its ordinal.

With explicit keys, as in JAX, a random primitive is a pure function of a key value it is given. The dependence is visible in the program text, and transformations compose with randomness through the ordinary rules. The project's tenets favour this design:
- explicit over implicit: no hidden effects;
- unambiguity over ergonomics, since the author is an agent for whom threading a key costs nothing;
- composition over special cases: no per-transformation ordinal rules.

## 2. Proposed surface

The names below are placeholders for the decision.

| Form | Meaning |
|---|---|
| `Key` | An opaque, non-numeric value type. `key` is also a tensor element dtype, so a tensor of keys can be mapped over. |
| `key(seed: i64) -> Key` | The root key of a seed. It is pure. |
| `split(k: Key) -> (Key, Key)` | Two independent keys derived from `k`. |
| `split(k: Key, n) -> tensor[n, key]` | `n` independent keys, for `vmap` and batching. |
| `fold_in(k: Key, n: i64) -> Key` | The key for index `n` (loop steps, ensemble members). |
| `dropout(k: Key, x: &tensor[D,p], rate: p) -> tensor[D,p]` | [05-OP-37] with `k` in place of the handler's seed and ordinal. |
| `uniform_like(k: Key, t: &tensor[D,p], low: p, high: p) -> tensor[D,p]` | [05-OP-8] likewise. |

**Keys are affine: used at most once.** Passing a key to a random primitive, `split` or `fold_in` consumes it. The linearity checker's compiler-inserted `copy` is forbidden for keys, so reuse is a type error, not a silent duplicate. Dropping an unused key is allowed.

**`with seed` and the `Random` effect go away.** A function that draws takes a `Key` parameter, and its signature is the marker. The unhandled-`Random` check error disappears, and so does the escape class #2318 guards against.

## 3. Semantics

**Words.** The element word is `word(key, i) = splitmix64(key XOR rotl64(splitmix64(i), 41))`: [05-RNG-1]'s formula with the handler's `seed XOR rotl64(splitmix64(c), 17)` replaced by the key. The unit value, the mask and affine rules of [05-OP-37] and [05-OP-8], their dtype rules and their pre-draw validation are unchanged.

**Derivation.** `key(seed)`, `split` and `fold_in` must be finalised mixes, for example `derive(k, j) = splitmix64(k XOR rotl64(splitmix64(j), 29))`, with distinct domain constants for `split` and `fold_in`. The derivation must not be a bare XOR of per-index terms, because a chained `fold_in(fold_in(k, a), b)` would then be symmetric in `a` and `b`. That is the defect #2408 found in the older `uniform_like` mixing. The exact constants are part of the decision. Each derivation is tested for independence against its siblings and ancestors.

**Transformations:**
- `vmap` over a `tensor[n, key]` axis gives each row its own key. A key captured by a vmapped body is still a single-use value, so using it in every row is a linearity error, not an implicit "same randomness" mode.
- `par` needs no rule.
- A random primitive is pure, so CSE, reordering and dead-code elimination apply to it as to any operation.
- `grad` treats a key as a discrete input with `unit` cotangent. Dropout's data adjoint is then the ordinary derivative of a pure function whose mask depends only on the key. Recursion, control flow and nested `grad` need nothing extra.
- Checkpointing recomputes with the same key and gets the same values.

**The rate.** It is still not differentiable: the output jumps where the rate crosses a sampled unit value. The rejection with `AdRejectionReason::RandomSelectionParameter` stays as in the amended [05-OP-37].

**Carriers.** A key is a structurally non-numeric tagged carrier in the IR, on the wire and in the C ABI (an opaque `chelis_key`). It is never a bare integer, per `dtype_semantics.md` §C6.

## 4. What changes

**Spec.**
- spec/04 §7: `Random` is removed from the effect vocabulary, and `with seed` is removed. The §1.1 dtype list gains `key`.
- spec/02: the grammar for `with seed` is removed, and `Key`, `split` and `fold_in` are added.
- spec/03: `handle-effect` loses its `random` kind, leaving `resource`.
- spec/05: [05-RNG-1] is recast over keys; [05-OP-8] and [05-OP-37] take a key operand; new atoms are added for the key operations.
- spec/06: §2.11 and §3.2 get key rules.
- spec/07: §2 needs no ordinal rule.
- spec/08: random state is removed.
- spec/10: key carriers are added.

**Implementation.** The phases of `randomness_counter_stream.md` carry over, with one difference. Phase 3's executor frame becomes a key operand: the kernel boundary, the operand-form IR nodes, the occurrence tape and the deletions stay as they are. What is added:
- the `Key` type and the key dtype;
- the linearity rule;
- the three key operations;
- key carriers in serialisation and the C ABI.

What is removed: the frame's `(seed, ordinal)` stack, `vmap` and `par` ordinal bookkeeping, activation counting for ordinal purposes, and the `with seed` handler.

**Standard library and shells.** Every random API takes a key. The affected APIs are:
- `chelis-std`: `normal_like`, and the Kaiming and Xavier initialisers;
- school: dropout layers and the training loop, which splits a key per step;
- nautilus: distribution sampling;
- shoals: SDE and path generation, which folds in the step index;
- calcify's and hydronnx's translations of seeded source programs;
- Hull's reference evaluator.

Pinned random outputs are regenerated. `chelis manifest` (`chelis_reproducibility_manifests.md`) reports key provenance instead of `Random`-annotated operations and their handlers.

## 5. LaCaDiLE

Chelis should remain a language that LaCaDiLE can certify. The Core (`lean/LaCaDiLECore`) currently formalises the counter protocol:
- `Protocol.lean`'s `DrawKey`, `RandomState` and `checkProtocol`;
- `EffectAD.lean`'s `ownedGrad_correct`.

An assessment of what moving the Core to explicit keys requires, what it costs and what it gains is running on LaCaDiLE branch `explicit-keys-feasibility`. Its result is recorded in chelis#2413 and summarised here before any decision.

## 6. Decision criteria

The counter stream (option A) is kept unless all of these hold:
- LaCaDiLE's Core can certify keyed programs at no less coverage than the counter protocol;
- the migration of the standard library and shells has an owner;
- the key carrier passes the §C6 numeric-surface rules.

Option B, keys derived from program position under an implicit handler, was considered and not pursued. It keeps the implicit effect and makes a stream depend on call-site paths, which is as refactor-fragile as the counter and less explicit than option C.

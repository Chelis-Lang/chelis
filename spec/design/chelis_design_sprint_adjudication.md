# Design Sprint Results: Adjudication and Next Steps

## Summary

Two agents ran the design sprint independently. The **Shallow agent** (Doc 3) produced a high-level pass across all 8 tasks. The **Deep agent** (Doc 4) went much deeper on Tasks 1 and 2 (Deep syntax and tag vocabulary) with a complete 46-tag vocabulary, PEG grammar, canonical form rules, detailed examples, and honest open questions. The Deep agent's work is the primary base for the spec; the Shallow agent contributes useful ideas for fitness scoring, named dimensions, and Phase 2 sketches.

---

## The Big Decision: Explicit `app`/`var`/`lit` vs Bare Lisp-Style

Both agents independently chose **explicit `app` for function application**, contradicting the earlier steering to the coding agent ("No apply tag — function calls are just lists with function name in head position"). This needs to be resolved because it affects the entire Deep syntax and the Phase 0c desugaring table.

**The case for explicit `app`/`var`/`lit` (both sprint agents):**
- `(f x)` is ambiguous for a generator: is `f` a structural tag or a function name? The generator must know the full tag vocabulary to avoid collisions.
- `(app {} (var {} f) (var {} x))` is structurally unambiguous. Every node's role is clear from its tag.
- No lookahead needed during generation — the tag position always determines how to interpret children.
- The tag vocabulary stays orthogonal to user-defined names.

**The case for bare Lisp-style (earlier steering):**
- Dramatically less verbose: `(add a b)` vs `(app {} (var {} add) (var {} a) (var {} b))`.
- The tag set is closed — the parser CAN distinguish tags from function names without `app`.
- More natural for humans inspecting Deep output.

**Ruling: Accept explicit `app`/`var`/`lit`.** The thesis is "written by AIs, for AIs." The verbosity cost is real (~3-5x more tokens per expression) but acceptable because: (a) Deep is not meant for human authoring, (b) the regularity radically simplifies structural manipulation (mutation, crossover, code analysis), (c) AI context windows are large and getting larger, (d) the `.chb` binary format compresses the verbosity away for storage/transfer. **This reverses the earlier steering correction to the coding agent. The Phase 0c desugaring table must be updated.**

---

## What the Deep Agent Got Right (Accept As-Is)

### 1. The Universal 3-Tuple: `(tag {} children...)`
Every node has identical shape. Metadata is always the second element, always a `{}` map. Empty is `{}`. This is the right model — uniform node structure is critical for AI generation. The Deep agent's rationale for rejecting alternatives (Clojure reader macros, explicit meta wrapper, property-list-in-tag-position) is sound.

### 2. The 46-Tag Vocabulary (Organized, Closed)
The tag set is well-partitioned: module structure (4), declarations (6), expressions (13), patterns (6), type expressions (7), dimension expressions (3), transforms (6), metaprogramming (3), helpers (3). The closed set means "any unknown tag is a parse error" which is exactly right for validation and fitness scoring.

### 3. RISC Primitives Are Not Tags
`add`, `mul`, `reshape`, etc. are built-in functions accessed via `(var {} add)` and called via `(app {} ...)`. If tinygrad adds a primitive, the syntax doesn't change. The tag vocabulary is for AST structure; the RISC primitive set is for computation. Orthogonal concerns stay orthogonal.

### 4. Transforms (`grad`/`vmap`/`jit`/`realize`/`cast`/`copy`) Are Tags
These have special compiler semantics — they're DAG-to-DAG rewrites, not function calls. Making them tags (not `app`) means the compiler can recognize them structurally during compilation passes without inspecting function names.

### 5. Typed Subtag Namespaces
Type expressions get `t-` prefix (`t-prim`, `t-fn`, `t-tensor`, `t-var`). Dimensions get `d-` prefix (`d-name`, `d-var`, `d-lit`). Patterns get `pat-` prefix (`pat-var`, `pat-ctor`, `pat-wild`). This prevents namespace collisions and makes the tag vocabulary self-documenting.

### 6. `defsig` Separate from `def`
Type signatures are separate declarations, not inline annotations. `(defsig {} f (t-fn {} ...))` then `(def {} f (fn {} ...))`. This is simpler than embedding the signature inside the def, and it matches ML-family convention.

### 7. `defdim` for Dimension Declarations
Named dimensions are explicitly declared: `(defdim {} batch)`. This is cleaner than implicit-on-first-use and gives the compiler a clear declaration site for scope management.

### 8. Canonical Form Rules
2-space indent, 80-char line width, declaration-order preservation (not sorted), alphabetized imports and record fields only, strict literal normalization. Comments discarded. All correct.

### 9. PEG Grammar
Simple, parseable, complete. An AI agent can generate valid Deep from this grammar with minimal context.

### 10. The Open Questions
All five are real and well-identified: tuples, strings, recursive types, module depth, `const` as tensor constructor. These need answers.

---

## What the Shallow Agent Got Right (Cherry-Pick)

### 1. Fitness Scoring Breakdown
The Shallow agent's concrete scoring: 0.2 parse success + 0.2 name resolution + 0.6 type check (fraction of unifying sub-expressions). The Deep agent didn't specify this. Adopt the Shallow agent's scoring formula as the starting point for the fitness algorithm in `spec/04-type-system.md`.

### 2. Named Dimensions as Function-Level Generics
```
def linear[batch, in_dim, out_dim](x: tensor[batch, in_dim, f32], w: tensor[in_dim, out_dim, f32]): tensor[batch, out_dim, f32]
```
This is cleaner than the Deep agent's `defdim` at module level for polymorphic dimensions. **Reconciliation:** Use `defdim` for concrete module-level dimensions (e.g., `(defdim {} max_seq_len)`) and function-level dimension parameters for polymorphic dimensions (which map to `d-var` in Deep). Both mechanisms are needed.

### 3. Dimension Arithmetic: `*` for Unknown
Concatenating `tensor[batch, seq1, f32]` and `tensor[batch, seq2, f32]` yields `tensor[batch, *, f32]`. The `*` wildcard is a practical v1 escape hatch that avoids dependent types. Adopt this.

### 4. Row Polymorphism for Effects (Phase 2)
The right formal model for effect inference. A function calling `g` inherits `g`'s effects unless handled. Track for Phase 2 design.

### 5. Linear Closures
A closure capturing a linear value becomes a linear closure (can only be called once). This is a clean interaction rule. Track for Phase 2.

---

## Where Both Agents Were Wrong or Incomplete

### 1. The Shallow Agent's `prim` Wrapper
The Shallow agent uses `(prim add)` as a special form to reference RISC primitives. But the Deep agent is right: RISC primitives are just built-in functions. `(var {} add)` is sufficient. No `prim` tag needed.

### 2. The Shallow Agent's Tensor Literal in Surf
The Shallow agent includes `TensorLit` in the Surf grammar. Our settled decision says no tensor literals in expression position. Reject.

### 3. The Deep Agent's `a - b` Lowering
The Deep agent says `a - b → (app {} (var {} add) a' (app {} (var {} neg) b'))`. This is technically correct per RISC primitive philosophy (no `sub` primitive) but means every subtraction generates a nested expression. More practically, `sub` should be a derived built-in function that the compiler lowers to `add(a, neg(b))` during IR construction. The desugarer should produce `(app {} (var {} sub) a' b')` and let the compiler lower it. Same for `div` — it's multiply by reciprocal in RISC, but the desugarer shouldn't decompose it.

### 4. The Shallow Agent's Grammar is Incomplete
The Surf grammar in Doc 3 is too skeletal — missing pattern syntax, record variant syntax, many expression forms. Use the existing Phase 0c plan's grammar decisions (which are more detailed) plus the Deep agent's examples for guidance.

### 5. Neither Agent Addressed the Surf → Deep Desugaring Table Completely
The Deep agent provides examples but not a systematic desugaring table. The Shallow agent's table is too terse. The Phase 0c coding agent's table (already reviewed and corrected) is the most complete. It needs updating for the `app`/`var`/`lit` change.

---

## Open Questions: Rulings

### Q1: Tuples
**Ruling: Add `tuple` and `tuple-get` tags.** The Deep agent is right — tuples are pervasive in multi-return (grad returns a tuple of gradients). They need first-class syntax: `(tuple {} expr₁ expr₂)` and `(tuple-get {} expr (lit {} index))`.

### Q2: String Type
**Ruling: Add `string` as a primitive type.** Needed for IO and error messages. Not for tensor computation. `(t-prim {} string)` is valid.

### Q3: Recursive Types
**Ruling: Implicit via named references.** `(deftype {} List (a) (variant {} Nil) (variant {} Cons (field {} head (t-var {} a)) (field {} tail (t-adt {} List (t-var {} a)))))` — the `List` reference in `Cons.tail` is resolved by name. No `mu` types needed for v1.

### Q4: Module System Depth
**Ruling: Flat modules for v1.** No nesting. Dimension names are module-scoped. Module signatures are Phase 2. Import/export as specified.

### Q5: `const` as Tensor Constructor
**Ruling: `const` and `load` are the only primitive tensor constructors.** Random init (`randn`, `uniform`) are library functions in `std.init` that combine `const` with the `Random` effect. `from_data` is a `load` variant. This is consistent with tinygrad's model.

---

## Impact on the Coding Agent

The biggest impact is the `app`/`var`/`lit` change. Here's what needs to happen:

### Must Update Before Phase 0c Continues

1. **Reverse the earlier steering correction.** The Phase 0c plan currently says "No apply tag." This is now wrong. The desugaring table must use `(app {} (var {} name) args...)` form.

2. **Update `spec/03-deep-syntax.md`** with the Deep agent's tag vocabulary (adjusted per rulings above). This becomes the authoritative reference the coding agent works from.

3. **Update the Phase 0c desugaring table.** Every row changes. Example:
   - Old: `f(x, y) → (f x y)`
   - New: `f(x, y) → (app {} (var {} f) (var {} x) (var {} y))`
   - Old: `a + b → (add a b)`
   - New: `a + b → (app {} (var {} add) (var {} a) (var {} b))`
   - New: `a - b → (app {} (var {} sub) (var {} a) (var {} b))` [sub is a derived built-in]

4. **Update the Deep parser (Phase 0b)** to expect the 3-tuple format `(tag {} children...)`. The parser currently doesn't require metadata in every node. It needs to at minimum accept `{}` as a valid second element and preserve it.

5. **Add the `var`, `lit`, `app` and other new tags** to whatever tag validation exists in the Deep parser.

### Does Not Require Phase 0b Rework

The Phase 0b Deep parser already parses general s-expressions. The 3-tuple structure is a semantic convention, not a syntactic one — `(app {} (var {} f) (var {} x))` is already parseable as nested s-expressions. The tag validation layer can be added incrementally.

The main Phase 0b change: if the parser currently strips or ignores `{}` in list position, it needs to preserve it as a metadata node. This should be a small change.

### Phase 1+ Items to Track (Not for Coding Agent Now)

| Item | Source | Phase |
|---|---|---|
| Row polymorphism for effects | Shallow agent | 2a |
| Linear closures (called-once semantics) | Shallow agent | 2b |
| `(borrow {} x)` syntax for borrowing | Deep agent §2.7, Shallow agent | 2b |
| `(copy {} expr)` for explicit tensor dup | Deep agent §2.7 | 2b |
| Effect metadata key `eff` | Deep agent §1 | 2a |
| Linearity metadata key `lin` | Deep agent §1 | 2b |
| Macro pre-pass before type checking | Both agents | 2c |
| Hygiene via Racket-style scope sets | Both agents | 2c |
| `splice` tag (unquote-splicing) | Deep agent §2.8 | 2c |
| Module signatures / nesting | Both agents (deferred) | 3 |
| `*` wildcard dimension for concat results | Shallow agent | 1 or 2 |
| Dimension arithmetic / dependent dims | Both agents (rejected for v1) | 3+ |

---

## Plan: Updating Specs and Guiding the Coding Agent

### Step 1: Produce Updated `spec/03-deep-syntax.md`

Base: Deep agent's document (Doc 4).
Modifications:
- Remove the `prim` concept (Shallow agent's invention; not needed with `var`)
- Add `tuple` and `tuple-get` tags (per Q1 ruling)
- Add `string` to `t-prim` (per Q2 ruling)
- Add `sub`, `div`, `eq`, `neg` etc. as derived built-in functions (not RISC primitives, not tags — just names in the compiler's built-in scope that lower to RISC primitive compositions during IR construction)
- Add a "Built-in Scope" section listing all names available without import: RISC primitives + derived ops + transforms
- Reconcile dimension declarations: `defdim` for module-level concrete dims + function-level `d-var` for polymorphic dims
- Include the complete desugaring table (all Surf→Deep transforms)
- Note that comments in `.dp` files use `;` and are discarded in canonical form

### Step 2: Produce Updated `spec/04-type-system.md`

Cherry-pick from both agents:
- Deep agent's type representation (`t-prim`, `t-fn`, `t-tensor`, etc.)
- Deep agent's inference rules (App, Let with generalization, Tensor dim match)
- Shallow agent's fitness scoring formula (0.2 + 0.2 + 0.6)
- Shallow agent's named dimension design (function-level generics + `*` wildcard)
- Deep agent's precision rules (strict, no implicit widening)
- Add partial inference annotation format: error metadata on failed nodes

### Step 3: Produce Updated `spec/05-risc-primitives.md`

Merge both agents:
- Deep agent's principle: RISC ops are built-in functions, not tags
- Shallow agent's rejection of broadcasting (unanimous — good)
- Deep agent's adjoint table (more complete)
- Shallow agent's standard lowerings (matmul, relu, softmax)
- Add the full lowering list from the design sprint prompt Task 3e
- Clarify: the desugarer emits derived ops (sub, div, eq, etc.); the IR lowering pass decomposes them to RISC primitives

### Step 4: Write the Coding Agent Steering Memo

A concise document telling the coding agent:
1. The `app`/`var`/`lit` decision and why it changed
2. The updated desugaring table (every Surf→Deep transform)
3. The 46-tag vocabulary (link to spec)
4. The metadata format (always-present `{}` map as second element)
5. The built-in scope (RISC primitives + derived ops + transforms)
6. The dimension design (defdim + function-level d-var)
7. What NOT to build yet (effects, linearity, macros — leave extension points)

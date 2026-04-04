# Steering Memo: Design Sprint Results for Phase 0c

**To:** Coding agent (Phase 0c)
**From:** Language designer
**Status:** Authoritative — overrides previous steering corrections

---

## Critical Change: Deep Uses Explicit `app`/`var`/`lit`

The earlier steering that said "No apply tag — function calls are just lists" is **reversed**. Two independent design agents concluded that explicit structural tags are necessary for AI-generation-first design. The new rule:

**Every Deep node is a 3-tuple: `(tag {} children...)`**

- `{}` is a metadata map. Always present. Empty when no metadata.
- Function application: `(app {} func arg1 arg2 ...)`
- Variable reference: `(var {} name)`
- Literal value: `(lit {type: t} value)`

This means `a + b` in Surf becomes:
```
(app {} (var {} add) (var {} a) (var {} b))
```

NOT:
```
(add a b)
```

**Why the reversal:** `(f x)` is ambiguous — is `f` a structural tag or a function name? With `app`/`var`/`lit`, every node's role is determined by its tag. An AI generator never needs to know the tag vocabulary to avoid collisions. The verbosity cost is acceptable because Deep is for machines; Surf handles human readability.

---

## Impact on Phase 0b (Deep Parser)

The existing Deep parser already handles nested s-expressions. The 3-tuple convention is semantic, not syntactic — `(app {} (var {} f) (var {} x))` parses fine as nested lists.

**Required changes:**
1. The parser must accept `{}` as a valid element (the metadata map). If it currently treats `{` as an error, add map parsing: `{}` or `{key: value, ...}`.
2. The printer must emit `{}` as the second element of every list node in canonical form.
3. Tag validation (if any) should accept the 46-tag vocabulary listed below. Unknown tags are warnings, not errors (per spec §8.3).

**NOT required:** Restructuring the AST types. The `List { elements: Vec<Expr> }` model (from the Bug 1 fix) already works. The convention that `elements[0]` is a tag, `elements[1]` is metadata, and `elements[2..]` are children is enforced semantically, not syntactically.

---

## The 46-Tag Vocabulary

Organized by category. This is the **closed** tag set — only these tags are valid in canonical Deep.

**Module Structure (4):** `module`, `import`, `import-all`, `export`

**Declarations (6):** `def`, `defsig`, `deftype`, `variant`, `field`, `defdim`

**Expressions (15):** `fn`, `app`, `let`, `match`, `arm`, `if`, `var`, `lit`, `record`, `access`, `pipe`, `block`, `par`, `tuple`, `tuple-get`

**Patterns (6):** `pat-var`, `pat-lit`, `pat-ctor`, `pat-record`, `pat-wild`, `pat-as`

**Type Expressions (7):** `t-prim`, `t-fn`, `t-tensor`, `t-adt`, `t-var`, `t-unit`, `t-tuple`

**Dimension Expressions (3):** `d-name`, `d-var`, `d-lit`

**Transforms (6):** `grad`, `vmap`, `jit`, `realize`, `cast`, `copy`

**Metaprogramming (3):** `quote`, `unquote`, `splice`

**Helpers (3):** `params`, `bind`, `kv`

**Total: 53** (46 from Deep agent + `tuple`, `tuple-get`, `import-all`, `defdim`, `par`, `block`, `tuple-get`)

### RISC Primitives Are NOT Tags

The ~12 RISC ops (`add`, `mul`, `exp`, `log`, `reshape`, `permute`, etc.) are **built-in functions**, not tags. They live in the compiler's built-in scope and are referenced as `(var {} add)`, applied via `(app {} ...)`.

```
;; Correct:
(app {} (var {} add) (var {} a) (var {} b))

;; WRONG:
(add {} (var {} a) (var {} b))    ;; add is NOT a tag
```

### Transforms ARE Tags (Not `app`)

`grad`, `vmap`, `jit`, `realize`, `cast`, `copy` have special compiler semantics (DAG rewrites, not function calls). They get their own tags:

```
(grad {} (var {} loss_fn))           ;; NOT (app {} (var {} grad) ...)
(cast {} (var {} x) (t-prim {} bf16))
(jit {} (var {} forward))
```

### Derived Built-In Functions

These are NOT RISC primitives and NOT tags. They are names in the built-in scope that the compiler lowers to RISC primitive compositions during IR construction:

`sub`, `div`, `neg`, `eq`, `neq`, `gt`, `gte`, `lte`, `sigmoid`, `relu`, `softmax`, `matmul`, `linear`, `mean`, `max_elem`, `min_elem`

The desugarer emits these as `(app {} (var {} sub) ...)`. The IR lowering pass decomposes them. Example: `sub(a, b)` → `add(a, neg(b))` at IR level.

---

## Updated Desugaring Table (Phase 0c)

Every Surf construct and its Deep form. `a'`, `b'` etc. mean "recursively desugared."

### Declarations

| Surf | Deep |
|---|---|
| `module Foo` | `(module {} foo ...)` |
| `import Foo.Bar (baz, qux)` | `(import {} foo.bar (baz qux))` |
| `def f(x: T, y: U): R = body` | `(defsig {} f (t-fn {} T' U' R'))` then `(def {} f (fn {} (params x y) body'))` |
| `def f(x, y) = body` (no sig) | `(def {} f (fn {} (params x y) body'))` |
| `let x = e` (top-level) | `(def {} x e')` |
| `type Opt a = \| Some a \| None` | `(deftype {} Opt (a) (variant {} Some (t-var {} a)) (variant {} None))` |
| `type Pt = \| Pt { x: f32, y: f32 }` | `(deftype {} Pt () (variant {} Pt (field {} x (t-prim {} f32)) (field {} y (t-prim {} f32))))` |

### Expressions

| Surf | Deep |
|---|---|
| `f(x, y)` | `(app {} (var {} f) x' y')` |
| `f x` (juxtaposition) | `(app {} (var {} f) x')` |
| `f x y` (juxtaposition chain) | `(app {} (var {} f) x' y')` — flatten left-associative Apply nesting |
| `x \|> f \|> g` | `(pipe {} x' (var {} f) (var {} g))` |
| `x \|> fn v -> f(v, a)` | `(pipe {} x' (fn {} (params v) (app {} (var {} f) (var {} v) (var {} a))))` |
| `a + b` | `(app {} (var {} add) a' b')` |
| `a - b` | `(app {} (var {} sub) a' b')` |
| `a * b` | `(app {} (var {} mul) a' b')` |
| `a / b` | `(app {} (var {} div) a' b')` |
| `-a` | `(app {} (var {} neg) a')` |
| `a == b` | `(app {} (var {} eq) a' b')` |
| `a != b` | `(app {} (var {} neq) a' b')` |
| `a < b` | `(app {} (var {} cmplt) a' b')` |
| `a > b` | `(app {} (var {} cmplt) b' a')` — note swap |
| `a <= b` | `(app {} (var {} lte) a' b')` |
| `a >= b` | `(app {} (var {} gte) a' b')` |
| `a && b` | `(app {} (var {} and) a' b')` |
| `a \|\| b` | `(app {} (var {} or) a' b')` |
| `!a` | `(app {} (var {} not) a')` |
| `if c then a else b` | `(if {} c' a' b')` |
| `match e with { \| P1 -> b1 \| P2 -> b2 }` | `(match {} e' (arm {} P1' () b1') (arm {} P2' () b2'))` |
| `let x = e in body` | `(let {} (bind x e') body')` |
| `let x = e1; let y = e2; body` (block) | `(let {} (bind x e1' y e2') body')` |
| `fn (x, y) -> body` | `(fn {} (params x y) body')` |
| `e : T` | The annotation is pushed into metadata: the desugared node for `e` gets `{type: T'}` |
| `cast(e, bf16)` | `(cast {} e' (t-prim {} bf16))` |
| `grad(f)` | `(grad {} (var {} f))` or `(grad {} f')` if f is an expression |
| `vmap(f)` | `(vmap {} f' (d-var {} 0))` (default axis) |
| `jit(f)` | `(jit {} f')` |
| `42` | `(lit {type: (t-prim {} int32)} 42)` |
| `3.14` | `(lit {type: (t-prim {} f64)} 3.14)` — or f32, TBD: default float precision |
| `true` | `(lit {type: (t-prim {} bool)} true)` |
| `"hello"` | `(lit {type: (t-prim {} string)} "hello")` |
| `(a, b, c)` | `(tuple {} a' b' c')` |

### Type Expressions

| Surf | Deep |
|---|---|
| `f32` | `(t-prim {} f32)` |
| `bool` | `(t-prim {} bool)` |
| `string` | `(t-prim {} string)` |
| `tensor[batch, seq, f32]` | `(t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))` — last child is precision |
| `tensor[a, b, f32]` (polymorphic) | `(t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))` |
| `A -> B -> C` | `(t-fn {} A' B' C')` — flat, last is return type |
| `Option f32` | `(t-adt {} Option (t-prim {} f32))` |
| `(f32, f32)` | `(t-tuple {} (t-prim {} f32) (t-prim {} f32))` |
| `_` (infer) | `(t-var {} _)` |

### Patterns

| Surf | Deep |
|---|---|
| `x` (variable) | `(pat-var {} x)` |
| `42` (literal) | `(pat-lit {} 42)` |
| `Some x` | `(pat-ctor {} Some (pat-var {} x))` |
| `Pt { x, y }` | `(pat-record {} Pt (kv {} x (pat-var {} x)) (kv {} y (pat-var {} y)))` |
| `_` | `(pat-wild {})` |
| `x @ Some _` | `(pat-as {} x (pat-ctor {} Some (pat-wild {})))` |

---

## Dimension Design

Two mechanisms, both needed:

1. **Module-level concrete dimensions:** `(defdim {} batch)` — declares a named dimension for the module. Used for fixed, known dimension names.

2. **Function-level polymorphic dimensions:** In Surf, `def f[a, b](x: tensor[a, b, f32])`. In Deep, dimension variables appear as `(d-var {} a)` in the type signature. The `[a, b]` syntax in Surf is desugared to dimension parameters on the function.

3. **Wildcard `*`:** `(d-name {} *)` represents an unknown/dynamic dimension. Used as the result of operations like concatenation where the compiler can't statically determine the dimension name.

---

## What NOT to Build in Phase 0

Leave extension points but don't implement:

- **Effects:** The `eff` meta key is reserved. Don't parse or validate it. Functions with effects just have `{}` metadata for now.
- **Linearity:** The `lin` meta key is reserved. All tensors are treated as unrestricted in Phase 0.
- **Macros:** `quote`/`unquote`/`splice` tags are in the vocabulary but the Phase 0 compiler can reject them with "macros not yet supported."
- **`copy` and `borrow`:** In the tag vocabulary but not implemented. Phase 0 treats all values as freely copyable.
- **`par`:** In the tag vocabulary but the Phase 0 compiler can evaluate it sequentially.

The tags exist so that AI-generated Deep programs that include Phase 2 features still parse — they just fail at a later compiler stage with a clear error.

---

## Metadata Format

Always a `{}` map as the second element of every node.

```
(def {} f ...)                          ;; empty metadata — let compiler infer
(def {type: (t-fn {} ...)} f ...)       ;; with type annotation
(lit {type: (t-prim {} f32)} 3.14)      ;; literal with type
(app {} (var {} add) ...)               ;; application, no metadata needed
```

**Phase 0 meta keys:** `type` and `loc` only.
**Phase 0 parser behavior:** Parse `{}` maps. Preserve all key-value pairs. Ignore unknown keys with a warning. The Phase 0 type checker reads `type` metadata as constraints.

---

## Canonical Form Quick Reference

- 2-space indent per nesting level
- Line break if node exceeds 80 chars
- `{}` always present (never omitted)
- `1.` → `1.0`, `.5` → `0.5`, `1e3` → `1.0E+3`
- No comments in canonical `.dp` (comments are Surf-only)
- Record `kv` pairs alphabetized by key
- Import names alphabetized
- Everything else in declaration order

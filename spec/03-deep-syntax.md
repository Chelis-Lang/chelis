# spec/03-deep-syntax.md — Chelis Deep Syntax Specification

**Status:** v0.2 (post design sprint)
**Scope:** The primary machine interface. Everything an AI agent or compiler needs to construct, parse, validate, and transform Deep programs.

---

## 1. Node Structure

Every Deep AST node is a 3-tuple:

```
(tag {key: value, ...} child₁ child₂ ... childₙ)
```

- **tag** — an identifier from the closed tag vocabulary (§2).
- **meta** — a property map enclosed in `{}`. Always present in canonical form. Empty map is `{}`.
- **children** — zero or more child nodes, or bare identifiers/literals inside helper tags like `params` and `bind`.

### 1.1 Metadata

The meta map carries compiler-relevant annotations. An agent MAY include metadata to constrain inference, or MAY write `{}` everywhere and let the compiler fill it in.

**Phase 0 keys:**

| Key | Value | Semantics |
|---|---|---|
| `type` | type-expr node | Type annotation (checked, not trusted) |
| `loc` | `(loc file line col)` | Source location for error reporting |

**Reserved for Phase 2 (parser accepts, compiler ignores with warning):**

| Key | Value | Semantics |
|---|---|---|
| `eff` | effect-set | Algebraic effects |
| `lin` | `once` / `borrow` / `unrestricted` | Linearity |
| `doc` | string | Documentation |

### 1.2 Rationale

The universal 3-tuple means every node has identical shape. An agent constructing Deep never decides where metadata goes — it's always element two. Compared to alternatives:

- Clojure reader metadata (`^{} expr`): prefix position requires lookahead during generation.
- Explicit meta wrapper (`(meta {} expr)`): redundant nesting.
- No metadata in `.dp` text: forces annotation into `.chb` only, making Deep text useless for constrained generation.

---

## 2. Tag Vocabulary

The tag set is **closed**. Only these tags produce valid Deep nodes. Unknown tags are parse errors.

### 2.1 Module Structure

| Tag | Form | Semantics |
|---|---|---|
| `module` | `(module {} name decl...)` | Top-level module |
| `import` | `(import {} path (name...))` | Selective import |
| `import-all` | `(import-all {} path)` | Wildcard import |
| `export` | `(export {} name...)` | Public API |

### 2.2 Declarations

| Tag | Form | Semantics |
|---|---|---|
| `def` | `(def {} name expr)` | Value/function binding |
| `defsig` | `(defsig {} name type-expr)` | Type signature (precedes `def`) |
| `deftype` | `(deftype {} name (type-params...) variant...)` | ADT declaration |
| `typealias` | `(typealias {} name (type-params...) type-expr)` | Transparent type alias |
| `variant` | `(variant {} Name field...)` | Sum type constructor (fields optional) |
| `field` | `(field {} name type-expr)` | Named field in variant |
| `defdim` | `(defdim {} name)` | Dimension name declaration |

### 2.3 Expressions

| Tag | Form | Semantics |
|---|---|---|
| `fn` | `(fn {} (params ...) body)` | Anonymous function |
| `app` | `(app {} func arg...)` | Function application |
| `let` | `(let {} (bind name₁ expr₁ ...) body)` | Sequential let binding |
| `match` | `(match {} scrutinee arm...)` | Pattern match |
| `arm` | `(arm {} pattern guard body)` | Match arm; guard is `()` if absent |
| `if` | `(if {} cond then else)` | Conditional |
| `var` | `(var {} name)` | Variable reference |
| `lit` | `(lit {type: prim-type} value)` | Literal value |
| `record` | `(record {} TypeName (kv {} k₁ v₁) ...)` | Record construction |
| `access` | `(access {} expr field-name)` | Field access |
| `pipe` | `(pipe {} expr₁ expr₂ ... exprₙ)` | Pipeline composition |
| `block` | `(block {} expr₁ ... exprₙ)` | Sequenced expressions; value is last |
| `tuple` | `(tuple {} expr₁ expr₂ ...)` | Tuple construction |
| `tuple-get` | `(tuple-get {} expr index)` | Tuple element access |
| `record-update` | `(record-update {} expr (kv {} k v) ...)` | Functional record update (reserved; Phase 1) |
| `par` | `(par {} expr₁ expr₂ ...)` | Parallel evaluation (v1: sequential) |

### 2.4 Patterns

| Tag | Form | Semantics |
|---|---|---|
| `pat-var` | `(pat-var {} name)` | Bind to name |
| `pat-lit` | `(pat-lit {} value)` | Match literal |
| `pat-ctor` | `(pat-ctor {} CtorName pat...)` | Deconstruct variant |
| `pat-tuple` | `(pat-tuple {} pat₁ pat₂ ...)` | Tuple destructuring pattern |
| `pat-record` | `(pat-record {} TypeName (kv {} k₁ pat₁) ...)` | Deconstruct record |
| `pat-wild` | `(pat-wild {})` | Wildcard |
| `pat-as` | `(pat-as {} name pattern)` | Bind name, then match |

### 2.5 Type Expressions

| Tag | Form | Semantics |
|---|---|---|
| `t-prim` | `(t-prim {} f32)` | Primitive type (f32, bf16, int32, bool, string) |
| `t-fn` | `(t-fn {} arg₁ arg₂ ... ret)` | Function type; last child is return |
| `t-tensor` | `(t-tensor {} dim₁ dim₂ ... precision)` | Tensor type; last child is precision |
| `t-adt` | `(t-adt {} Name type-arg...)` | ADT type application |
| `t-var` | `(t-var {} name)` | Type variable |
| `t-unit` | `(t-unit {})` | Unit type |
| `t-tuple` | `(t-tuple {} type₁ type₂ ...)` | Tuple type |

### 2.6 Dimension Expressions

| Tag | Form | Semantics |
|---|---|---|
| `d-name` | `(d-name {} batch)` | Named dimension (concrete) |
| `d-var` | `(d-var {} a)` | Dimension variable (polymorphic) |
| `d-lit` | `(d-lit {} 512)` | Literal dimension size |

### 2.7 Transforms

| Tag | Form | Semantics |
|---|---|---|
| `grad` | `(grad {} expr)` | Reverse-mode AD |
| `vmap` | `(vmap {} expr dim)` | Vectorization |
| `jit` | `(jit {} expr)` | Compilation trigger |
| `realize` | `(realize {} expr)` | Force DAG evaluation |
| `cast` | `(cast {} expr target-type)` | Precision cast |
| `copy` | `(copy {} expr)` | Explicit tensor duplication (Phase 2: linearity) |

### 2.8 Metaprogramming

| Tag | Form | Semantics |
|---|---|---|
| `quote` | `(quote {} expr)` | Reify as AST data |
| `unquote` | `(unquote {} expr)` | Splice into quoted AST |
| `splice` | `(splice {} expr)` | Splice list into quoted AST |

### 2.9 Helpers

| Tag | Form | Semantics |
|---|---|---|
| `params` | `(params {} p₁ p₂ ...)` | Parameter list; each pᵢ is a bare name or `(name {type: t})` |
| `bind` | `(bind {} name₁ expr₁ name₂ expr₂ ...)` | Binding pairs for `let` |
| `kv` | `(kv {} key value)` | Key-value pair for records |

**Note:** `params` and `bind` follow the universal 3-tuple rule: `(params {} x y)`, `(bind {} name₁ expr₁ ...)`. Names inside them are bare identifiers, not `(var ...)` wrapped.

### 2.10 Tag Count Summary

| Category | Count | Tags |
|---|---|---|
| Module | 4 | module, import, import-all, export |
| Declarations | 7 | def, defsig, deftype, typealias, variant, field, defdim |
| Expressions | 16 | fn, app, let, match, arm, if, var, lit, record, access, pipe, block, tuple, tuple-get, record-update, par |
| Patterns | 7 | pat-var, pat-lit, pat-ctor, pat-tuple, pat-record, pat-wild, pat-as |
| Types | 7 | t-prim, t-fn, t-tensor, t-adt, t-var, t-unit, t-tuple |
| Dimensions | 3 | d-name, d-var, d-lit |
| Transforms | 6 | grad, vmap, jit, realize, cast, copy |
| Meta | 3 | quote, unquote, splice |
| Helpers | 3 | params, bind, kv |
| **Total** | **56** | |

---

## 3. Built-In Scope

These names are available without import. They are NOT tags — they are functions/values referenced via `(var {} name)` and called via `(app {} ...)`.

### 3.1 RISC Primitives (~12 ops)

The irreducible computational basis. All tensor computation decomposes to these during IR lowering.

**Elementwise:** `add`, `mul`, `exp`, `log`, `sin`, `sqrt`, `cmplt`, `max_elem`
**Reduce:** `sum`, `max_reduce` (over axis)
**Movement:** `reshape`, `permute`, `expand`, `pad`, `shrink`, `stride`
**Memory:** `const`, `load`

### 3.2 Derived Functions

Convenience functions that the compiler lowers to RISC primitive compositions during IR construction. The desugarer emits these; the IR pass decomposes them.

`sub`, `div`, `neg`, `eq`, `neq`, `gt`, `gte`, `lte`, `and`, `or`, `not`, `relu`, `sigmoid`, `softmax`, `matmul`, `linear`, `mean`

### 3.3 Standard Library (imported)

Not built-in — require `(import {} std.x ...)`:

- `std.io`: `println`, `read_tensor`, `write_tensor`
- `std.init`: `randn`, `uniform`, `zeros`, `ones`, `arange`
- `std.nn`: `layer_norm`, `conv2d`, `embedding`, `multi_head_attention`, `cross_entropy`

---

## 4. Function Application

### 4.1 Multi-Argument

Deep uses explicit multi-argument application, not currying:

```scheme
(app {} (var {} f) (var {} x) (var {} y) (var {} z))
```

If `f` expects 3 arguments and receives 2, this is a **type error**, not partial application.

### 4.2 Partial Application

Explicit closure construction:

```scheme
;; f(x, _, z) where _ is the partial hole
(fn {} (params y) (app {} (var {} f) (var {} x) (var {} y) (var {} z)))
```

### 4.3 Operators

All operators desugar to `(app {} (var {} op) ...)`. No infix operators in Deep.

```scheme
;; a + b
(app {} (var {} add) (var {} a) (var {} b))

;; a - b (sub is a derived built-in, lowered to add(a, neg(b)) at IR level)
(app {} (var {} sub) (var {} a) (var {} b))
```

---

## 5. Pipe Semantics

`pipe` is a first-class node, not sugar for nested application:

```scheme
(pipe {} expr₁ expr₂ expr₃)
```

Evaluation: `(pipe {} e₁ e₂ e₃)` ≡ `(app {} e₃ (app {} e₂ e₁))`.

`pipe` is preserved in Deep (not desugared to nested `app`) because: (a) it's the primary composition idiom, (b) preserving it enables better Deep → Surf round-tripping, (c) the compiler can reason about dataflow directly.

Each element after the first must be a function (or lambda). Pipes with multi-arg functions use lambdas:

```scheme
(pipe {} (var {} x)
  (fn {} (params v) (app {} (var {} f) (var {} v) (var {} a)))
  (var {} g))
```

---

## 6. Canonical Form

Deep has exactly one textual representation per program.

### 6.1 Whitespace
- 2-space indent per nesting level.
- Node fits on one line if ≤ 80 characters (including indentation). Otherwise, each child starts on its own line.
- No trailing whitespace. Single newline at EOF.

### 6.2 Ordering
- Module declarations: declaration order (not sorted).
- Import names within an import: alphabetized.
- Record `kv` pairs: alphabetized by key.
- Match arms: declaration order (semantically meaningful).
- Bind pairs in `let`: declaration order (sequential semantics).

### 6.3 Comments
None in canonical Deep. Comments are Surf-only. Stripped during desugaring. Use the `doc` meta key for structured documentation.

### 6.4 Literal Normalization

| Type | Canonical | Normalizations |
|---|---|---|
| Integer | Decimal, no leading zeros | `07` → `7` |
| Float | `d.d` minimum | `1.` → `1.0`, `.5` → `0.5` |
| Float (sci) | `d.dE±d` (uppercase E, explicit sign) | `1e3` → `1.0E+3` |
| String | Double-quoted, standard escapes | |
| Boolean | `true` / `false` | |

### 6.5 Identifier Rules
- Variables/functions: `[a-z_][a-z0-9_]*` (snake_case)
- Types/variants: `[A-Z][a-zA-Z0-9]*` (PascalCase)
- Module paths: dot-separated identifiers

---

## 7. Grammar (PEG)

```peg
Program     ← Spacing Node+ EOF
Node        ← '(' Spacing Tag Spacing Meta Spacing Children ')' Spacing
Tag         ← [a-z] [a-z0-9-]*                    # lowercase, hyphens allowed (pat-var, t-fn, etc.)
Meta        ← '{' Spacing (MetaPair (',' Spacing MetaPair)*)? '}'
MetaPair    ← MetaKey ':' Spacing MetaValue
MetaKey     ← [a-z]+
MetaValue   ← Node / Literal / Identifier / TypeName
Children    ← (Child Spacing)*
Child       ← Node / BareName / Literal
BareName    ← Identifier / TypeName                # bare names only in params, bind, field contexts
Identifier  ← [a-z_] [a-zA-Z0-9_]*
TypeName    ← [A-Z] [a-zA-Z0-9]*
Literal     ← FloatLit / IntLit / BoolLit / StringLit
FloatLit    ← '-'? [0-9]+ '.' [0-9]+ ([Ee] [+-]? [0-9]+)?
           /  '-'? [0-9]+ [Ee] [+-]? [0-9]+       # exponent without decimal
IntLit      ← '-'? [0-9]+
BoolLit     ← 'true' / 'false'
StringLit   ← '"' (!'"' .)* '"'
Spacing     ← ([ \t\n\r] / Comment)*
Comment     ← ';' (![\n] .)*
EOF         ← !.
```

---

## 8. Validation Rules

### 8.1 Structural Validation (Parser)
- Every node is a 3-tuple: `(tag meta children...)`.
- Tag is from the closed vocabulary (§2).
- Meta is a valid `{}` map (may be empty).

### 8.2 Arity Validation (Post-Parse)
- `(if {} cond then else)` — exactly 3 children.
- `(arm {} pattern guard body)` — exactly 3 children.
- `(fn {} params body)` — exactly 2 children; first must be `(params ...)`.
- `(let {} bindings body)` — exactly 2 children; first must be `(bind ...)`.
- `(app {} func arg...)` — at least 1 child (the function).

### 8.3 Unknown Tags
Unknown tags are parse errors in strict mode (canonical validation). In fitness-scoring mode, unknown tags are parsed as generic nodes and penalized in the fitness score.

---

## 9. Examples

### 9.1 Hello Tensor

```scheme
(module {} hello_tensor
  (import {} std.io (println))

  (def {} twos
    (app {} (var {} add)
      (app {} (var {} const) (lit {type: (t-prim {} f32)} 1.0) (d-lit {} 2) (d-lit {} 3))
      (app {} (var {} const) (lit {type: (t-prim {} f32)} 1.0) (d-lit {} 2) (d-lit {} 3))))

  (def {} main
    (fn {} (params)
      (app {} (var {} println) (realize {} (var {} twos))))))
```

### 9.2 Linear Regression

```scheme
(module {} linear_regression
  (defdim {} features)
  (defdim {} samples)

  (defsig {} predict
    (t-fn {}
      (t-tensor {} (d-name {} features) (t-prim {} f32))
      (t-tensor {} (d-name {} features) (t-prim {} f32))
      (t-tensor {} (d-name {} samples) (d-name {} features) (t-prim {} f32))
      (t-tensor {} (d-name {} samples) (t-prim {} f32))))

  (def {} predict
    (fn {} (params w b x)
      (app {} (var {} add)
        (app {} (var {} matmul) (var {} x) (var {} w))
        (var {} b))))

  (def {} mse_loss
    (fn {} (params y_pred y_true)
      (let {} (bind
        diff (app {} (var {} sub) (var {} y_pred) (var {} y_true))
        sq   (app {} (var {} mul) (var {} diff) (var {} diff)))
        (app {} (var {} mean) (var {} sq))))))
```

### 9.3 MLP with Pattern Matching

```scheme
(module {} mlp
  (defdim {} batch)
  (defdim {} input_dim)
  (defdim {} hidden_dim)
  (defdim {} output_dim)

  (deftype {} Activation ()
    (variant {} ReLU)
    (variant {} Sigmoid))

  (def {} activate
    (fn {} (params act x)
      (match {} (var {} act)
        (arm {} (pat-ctor {} ReLU) ()
          (app {} (var {} relu) (var {} x)))
        (arm {} (pat-ctor {} Sigmoid) ()
          (app {} (var {} sigmoid) (var {} x))))))

  (def {} forward
    (fn {} (params w1 b1 w2 b2 act x)
      (pipe {} (var {} x)
        (fn {} (params v) (app {} (var {} add) (app {} (var {} matmul) (var {} v) (var {} w1)) (var {} b1)))
        (fn {} (params v) (app {} (var {} activate) (var {} act) (var {} v)))
        (fn {} (params v) (app {} (var {} add) (app {} (var {} matmul) (var {} v) (var {} w2)) (var {} b2)))))))
```

### 9.4 ADT with Record Variants

```scheme
(module {} shapes

  (deftype {} Shape ()
    (variant {} Circle
      (field {} radius (t-prim {} f32)))
    (variant {} Rectangle
      (field {} width (t-prim {} f32))
      (field {} height (t-prim {} f32))))

  (def {} area
    (fn {} (params s)
      (match {} (var {} s)
        (arm {} (pat-ctor {} Circle (pat-var {} r)) ()
          (app {} (var {} mul)
            (lit {type: (t-prim {} f32)} 3.14159)
            (app {} (var {} mul) (var {} r) (var {} r))))
        (arm {} (pat-ctor {} Rectangle (pat-var {} w) (pat-var {} h)) ()
          (app {} (var {} mul) (var {} w) (var {} h)))))))
```

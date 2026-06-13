# spec/03-deep-syntax.md — Chelis Deep Syntax Specification

**Status:** v0.2 (post design sprint)
**Scope:** The primary machine interface. Everything an AI agent or compiler needs to construct, parse, validate, and transform Deep programs.

**Executable surface note:** the 61-tag vocabulary documented here remains the
authoritative shipped Deep grammar. Future Phase `3c` / `3d` / `3g` language-
completeness work may add new Deep forms or keep some functionality as built-in helper
calls, but that future surface is not yet part of the active closed vocabulary unless
this document is explicitly revised to say so.

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

Metadata keys use the same identifier character set as user-defined Deep
symbols: `[A-Za-z_][A-Za-z0-9_]*`. This admits producer-specific keys such as
`c_earchin_role` while preserving the no-hyphen rule that keeps Deep symbols
portable across Surf and Reef boundaries.

**Active keys:**

| Key | Value | Semantics |
|---|---|---|
| `type` | type-expr node | Type annotation (checked, not trusted) |
| `loc` | `(loc file line col)` | Source location for error reporting |
| `eff` | effect-set | Declared effect annotation on `t-fn` type expressions |
| `effects` | effect-set | Inferred effect annotation on checked `fn` nodes |
| `source` | macro invocation | Provenance: the macro call this node expanded from |
| `span` | string | External-source span identifier (see §1.1.1) |
| `chelis_role` | string | Declaration role marker; `"property"` marks a `def` as a `chelis prove` property |
| `property_source_kind` | string | Property producer: `"user"` or `"bridge:c-earchin"` |
| `property_quantifiers` | `(params {} ...)` | Serialized property binder list; must match the `fn` parameter list |
| `property_preconditions` | `(tuple {} ...)` | Serialized `where` filters evaluated before the predicate |
| `property_source_id` | string | Optional producer-local source ID, e.g. an EARS requirement ID |
| `property_tolerance` | expr | Optional property runner tolerance metadata |
| `property_seed` | expr | Optional property runner seed metadata |
| `property_samples` | expr | Optional property runner sample-count metadata |

**Reserved for later phases:**

| Key | Value | Semantics |
|---|---|---|
| `lin` | `once` / `borrow` / `unrestricted` | Linearity |
| `doc` | string | Documentation |
| `span_*` | reserved | Future richer span fields (see §1.1.1) |

**Metadata propagation through transformations.** Metadata fields are
preserved by all spec-defined transformations and round-trip through the
canonical form (§6). Cross-tool provenance (the `span` field in
particular) is intended to survive the full compile pipeline once the
in-flight span survival work lands; see
`spec/design/chelis_span_survival.md` for the phased S0–S5 plan.

#### 1.1.1 External-source spans (`span`, `span_*` namespace)

The `span` metadata key carries a string identifier issued by an external
producer (today: Octant's LaTeX-to-Deep translator, which writes
`{span: "n_001"}` and ships a sidecar `<input>.spans.json` mapping each ID to
the original LaTeX byte range). Chelis treats `span` values as opaque strings
and preserves them end-to-end through parsing, IR lowering, optimization
passes, and backend codegen so a generated C/HIP/Metal source line can be
traced back to the original external source. Producer-side interpretation
(LaTeX byte range, Surf line/col, other DSL anchor) lives in the producer's
sidecar and is none of chelis's concern.

When Chelis itself desugars parsed Surf, it acts as the span producer for
ordinary Surf expression bodies and emits opaque byte-range IDs of the form
`surf:<start>..<end>`. These IDs follow the same preservation and validation
rules as producer-supplied Deep spans. If a programmatic Surf AST has no real
source byte range, the desugarer omits `span` and downstream synthesized-node
fallbacks remain available.

The `span_*` prefix is reserved for future richer span data. If a need arises
to embed byte offsets or file identifiers directly inside Deep metadata
(rather than indirecting through a sidecar), they MUST be added under the
`span_*` namespace (`span_start`, `span_end`, `span_file`, …). Competing keys
that carry span-related data outside the `span_*` namespace are forbidden so
tooling has a stable contract.

**Reserved synthesized-marker shape.** When a chelis transformation pass
introduces a node with no external-source region (gradient adjoints, lowered
sub-nodes from a parentless intrinsic, etc.), the canonical `span` value uses
the form `__synthesized_<pass>__` (double-underscore wrap, lowercase pass
name). This shape is reserved — external producers MUST NOT emit span IDs
matching `__synthesized_*__`; chelis MUST NOT mint a synthesized marker that
omits the wrap. Currently defined markers:

| Marker | Issued by |
|---|---|
| `__synthesized_tier2__` | Tier 2 RISC decomposition pass when the parent op had no span |
| `__synthesized_grad__` | Automatic differentiation pass for backward (adjoint) nodes |

Synthesized-marker nodes are required to also carry forward-node spans on the
companion `merged_spans` IR field (defined in `spec/design/chelis_span_survival.md`),
so the audit chain always resolves through at least one external-source span
even when the canonical `span` is a synthesized marker.

**Span ID character set (forbidden code points).** Span ID strings MUST NOT
contain ASCII control characters in the range U+0000 through U+001F EXCEPT
space (U+0020). U+007F (DELETE) is also forbidden. Concretely, the
following are forbidden: `\0` (U+0000 NUL), `\t` (U+0009 TAB), `\n` (U+000A
LF), `\r` (U+000D CR), and every other code point in U+0001..=U+001F (the
remaining C0 control characters), plus U+007F (DEL). All other code points,
including any printable Unicode at U+0020 or above (except U+007F), are
permitted.

*Rationale.* Span IDs are interpolated as comment-safe identifier strings
into generated C / HIP / Metal source (`// span: <id>`). Forbidden
characters can terminate `//` line comments (notably `\n` and `\r`),
embed in C string contexts, or otherwise corrupt the generated source
and produce compiling-but-semantically-altered output (a real injection
class, not a theoretical one). The constraint exists so emit code can
interpolate spans into comments without per-emit escaping. Defense in
depth: emitters apply backslash-escape sanitization (`\\n`, `\\r`, `\\0`,
`\\xNN`) as a backstop for forbidden code points that reach emit through
programmatic IR construction (which bypasses the parser); the spec-level
contract is still that producers MUST NOT emit forbidden characters in
the first place.

*Producer obligation.* Producers (e.g., Octant) MUST emit span ID values
containing only allowed characters. Producers MAY include any printable
Unicode (U+0020 and above, except U+007F) — Greek letters, mathematical
symbols, dot-paths, and similar identifier conventions are all permitted.

*Parser obligation.* The Deep parser MUST reject `span` metadata values
containing forbidden characters. The diagnostic MUST identify the offending
byte position and code point, and MUST point at this section
(`spec/03-deep-syntax.md` §1.1.1) so authors can find the rule.

*Empty span ID.* The empty string (`{span: ""}`) is a **valid** span ID.
Producers MAY emit `{span: ""}` as a "no provenance" sentinel without
needing to elide the metadata key entirely. Tooling MUST NOT silently
coerce `Some("")` to `None`.

*Leading/trailing whitespace.* Span IDs MAY contain leading or trailing
ASCII space (U+0020), e.g. `" eq1.body "`. Such IDs are valid but
discouraged; emit code does NOT trim them. Producers SHOULD avoid
incidental whitespace, but the spec treats span IDs as opaque strings and
does not normalize them.

#### 1.1.2 Trust-boundary pattern for producer-supplied strings

Span values are validated at the **parse boundary** (§1.1.1: parser
rejection plus emit-side `chelis_ir::span_sanitize` defense in depth).
The parallel construction-side rule applies to producer-supplied strings
that reach the IR through programmatic DAG construction without going
through the parser:

- `RiscOp::Load { name }` and `RiscOp::Store { name }` use the
  validating newtype `chelis_ir::LoadStoreName`. Its constructor enforces
  the Deep parser's identifier grammar (`is_ident_start` /
  `is_ident_continue` from `crates/chelis-deep/src/lexer.rs`) extended
  with `.` to admit the synthesized tuple-flatten names (`foo.0`,
  `grads.1`, `lib_double.0`) that lowering legitimately produces. The
  accepted alphabet is `[A-Za-z_][A-Za-z0-9_.-]*`; control bytes,
  whitespace, `%`, `/`, and non-ASCII characters are rejected at
  construction time. Empty strings are rejected.

The architectural rule, applied uniformly: **every producer-supplied
string that flows into generated source must be validated at its trust
boundary** — at parse time when the value enters via Deep text, or at
construction time when the value enters via direct IR construction.
Future IR fields that admit producer-supplied strings (module names,
type names, effect names, etc.) must follow the same pattern. The
deferred per-emission-context defense-in-depth work (comment-context
shared sanitizer, format-string-context sanitizer, comprehensive backend
audit) is tracked at `spec/upstream-bugs/producer-string-sanitization.md`.

**Provenance metadata (Phase 2c).** After macro expansion, each node in the expanded
form may carry a `source` key in its metadata map indicating the macro invocation it
originated from.
Example: `(app {source: (relu input)} (var {} max_elem) (var {} input) (lit {type: (t-prim {} f32)} 0))`.
Provenance is informational — it does not affect parsing, type checking, or evaluation.
The node is a standard `app` node; the `source` key is ignored by all compiler passes
except error reporting.
The shipped provenance format is `{source: (macro-name original-arg...)}` where the
value is a plain Deep list recording the macro name and original invocation arguments.

**Macro boundary rule (Phase 2c).** LLM-facing Deep is always expanded Deep. Macro
definition and invocation forms may exist as compiler-internal or pre-expansion syntax,
but the AST surfaced to AI generation, repair, fitness scoring, decompilation
workflows, or downstream transforms contains only ordinary Deep nodes plus optional
`source` metadata. Macro syntax is a human-authoring layer that compiles away before
those workflows begin.

Compiler-internal macro tags such as `defmacro` and `macro-invoke` are intentionally
outside the public Deep grammar. `chelis validate --deep` remains strict about the
public vocabulary and rejects those internal pre-expansion forms.

### 1.2 Rationale

The universal 3-tuple means every node has identical shape. An agent constructing Deep never decides where metadata goes — it's always element two. Compared to alternatives:

- Clojure reader metadata (`^{} expr`): prefix position requires lookahead during generation.
- Explicit meta wrapper (`(meta {} expr)`): redundant nesting.
- No metadata in `.dp` text: forces annotation into `.chb` only, making Deep text useless for constrained generation.

### 1.3 Deep as an Editing Target

The same properties that make Deep a stable generation target make it a
stable target for *structural edits* by tooling. A small closed
vocabulary makes pattern-match operations (find every `app` whose head
is `var foo`) unambiguous. The 3-tuple uniformity means an editor never
has to special-case where to attach metadata after a rewrite. Canonical
form (§6) means two structurally identical programs serialize identically,
so diffs reflect semantic change rather than incidental formatting drift.

These properties are what the exploratory Agent Editing Surface direction
(`spec/design/chelis_agent_editing_surface.md`) builds on. This spec
documents the substrate; that doc covers the tooling layer.

---

## 2. Tag Vocabulary

The tag set is **closed**. Only these tags produce valid Deep nodes. Unknown tags are parse errors.
The compiler may use extra internal tags during pre-expansion phases, but they are not
part of this public vocabulary.

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

`deftype` may carry `opaque: true` metadata:

```lisp
(module {} stats.prob
  (deftype {opaque: true} Probability ()
    (variant {} Probability (field {} value (t-prim {} f32)))))
```

The metadata is language semantics: the type checker hides the
constructors, fields, casts, and literal ascriptions of a marked type
outside its defining module (`spec/04-type-system.md` section 2.5),
and a marked `deftype` outside a named module is a declaration error.
The `opaque-domain-construction` lint rule remains as defense-in-depth
fast feedback (`spec/01-nomenclature.md` section 12.1).

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
| `handle-effect` | `(handle-effect {effect: name} arg body)` | Phase 2a effect handler block |

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
| `t-prim` | `(t-prim {} f32)` | Primitive type (active set: f32, f64, bf16, f16, int8, int16, int32, int64, bool, string — see `spec/04-type-system.md` §1.1; `f8e4m3` is reserved/deferred per §1.1.1; unsigned types are out of scope per §1.1.2) |
| `t-fn` | `(t-fn {} arg₁ arg₂ ... ret)` | Function type; last child is return |
| `t-tensor` | `(t-tensor {} dim₁ dim₂ ... precision)` | Tensor type; last child is precision |
| `t-ref` | `(t-ref {} type)` | Read-only borrow type |
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
| `grad` | `(grad {} expr)` or `(grad {wrt: ...} expr idx-or-idx-tuple)` | Reverse-mode AD; `wrt` child selects parameter indices |
| `vmap` | `(vmap {} expr dim)` | Vectorization |
| `jit` | `(jit {} expr)` | Compilation trigger |
| `realize` | `(realize {} expr)` | Force DAG evaluation |
| `cast` | `(cast {} expr target-type)` | Precision cast |
| `copy` | `(copy {} expr)` | Explicit tensor duplication (Phase 2: linearity) |
| `borrow` | `(borrow {} expr)` | Temporary read-only tensor view for a single call site |

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
| `effects` | `(effects {} eff₁ eff₂ ...)` | Effect-set literal used in metadata |
| `resource` | `(resource {} "device")` | Resource-effect payload inside an effect set |

**Note:** `params` and `bind` follow the universal 3-tuple rule: `(params {} x y)`, `(bind {} name₁ expr₁ ...)`. Names inside them are bare identifiers, not `(var ...)` wrapped.

### 2.10 Tag Count Summary

| Category | Count | Tags |
|---|---|---|
| Module | 4 | module, import, import-all, export |
| Declarations | 7 | def, defsig, deftype, typealias, variant, field, defdim |
| Expressions | 18 | fn, app, let, match, arm, if, var, lit, record, access, pipe, block, tuple, tuple-get, record-update, par, handle-effect, borrow |
| Patterns | 7 | pat-var, pat-lit, pat-ctor, pat-tuple, pat-record, pat-wild, pat-as |
| Types | 8 | t-prim, t-fn, t-tensor, t-ref, t-adt, t-var, t-unit, t-tuple |
| Dimensions | 3 | d-name, d-var, d-lit |
| Transforms | 6 | grad, vmap, jit, realize, cast, copy |
| Meta | 3 | quote, unquote, splice |
| Helpers | 5 | params, bind, kv, effects, resource |
| **Total** | **61** | |

### 2.11 Planned Phase 3+ Expansion Note

The remaining practical Phase 3 work is expected to stress Deep in new directions:

- first-class host-language scalars and strings
- collection values such as lists and dictionaries
- file/data/tokenizer helper surfaces

Those additions are not active Deep tags today.
If Chelis later needs dedicated Deep tags for those features, this closed-vocabulary
section and the tag-count summary must be revised at the same time. Until then, the
current 61-tag count remains the authoritative shipped grammar.

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

`sub`, `div`, `neg`, `eq`, `neq`, `gt`, `gte`, `lte`, `and`, `or`, `not`, `relu`, `sigmoid`, `softmax`, `matmul`, `linear`, `mean`, `dropout`

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
- Node fits on one line if ≤ 80 characters (including indentation). Otherwise, keep the
  opening paren, tag, and metadata together on the first line and start each child on
  its own indented line.
- Canonical helper-heavy forms such as `fn`, `let`, and `bind` are printed in structured
  multi-line form even when a short flat rendering would fit.
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

**Literal default rule.** An unsuffixed integer literal binds at type
`int32` (i.e. its `lit` node carries `{type: (t-prim {} int32)}`); an
unsuffixed float literal binds at type `f32`. The lexer accepts i64/f64
ranges so that out-of-range literals produce a useful diagnostic before
defaulting; the desugarer/type-checker narrows the value to `int32` /
`f32` before Deep is materialized. See `spec/04-type-system.md` §5.3 for
the type-system statement. The narrowing is overridable only by an
explicit literal suffix (§6.4.1), the contextual tensor-literal inference
rule (`spec/02-surf-syntax.md` §P10b), or an explicit `cast`.

#### 6.4.1 Literal Suffixes

Numeric literal tokens may carry an explicit precision suffix that binds
the literal at exactly that precision. The suffix is part of the literal
token only if it immediately follows the digit sequence with no
intervening whitespace or comment. The closed suffix set, identical to
the Surf suffix set (`spec/02-surf-syntax.md` §P10a):

| Suffix | Bound type | Example |
|---|---|---|
| `f32` | `(t-prim {} f32)` | `1.0f32` |
| `f64` | `(t-prim {} f64)` | `1.0f64` |
| `bf16` | `(t-prim {} bf16)` | `1.0bf16` |
| `f16` | `(t-prim {} f16)` | `1.0f16` |
| `i8` | `(t-prim {} int8)` | `42i8` |
| `i16` | `(t-prim {} int16)` | `42i16` |
| `i32` | `(t-prim {} int32)` | `42i32` |
| `i64` | `(t-prim {} int64)` | `42i64` |

In canonical Deep, a suffixed literal MAY be written either with the
suffix on the literal token (the producer-friendly shape) or as a `lit`
node carrying an explicit `{type: ...}` metadata key (the
canonical-form shape). The two shapes are interchangeable; the canonical
serialization printed by `chelis fmt` for `.dp` is the metadata-key
shape. Example:

```
;; producer-friendly shape (lexer accepts both)
(lit {} 1.0f64)

;; canonical Deep shape after fmt
(lit {type: (t-prim {} f64)} 1.0)
```

Float-typed suffixes (`f32`, `f64`, `bf16`, `f16`) attach to either an
integer or float literal token. Integer-typed suffixes (`i8`, `i16`,
`i32`, `i64`) attach to integer literal tokens only.

The suffix `f8e4m3` is reserved/deferred per `spec/04-type-system.md`
§1.1.1 and is rejected at lex time. Unsigned suffixes (`u8`, `u16`,
`u32`, `u64`) are out of scope per §1.1.2 and are rejected at lex time.
Hex integer literals interact with float-typed suffixes per the
hex-suffix rule in `spec/02-surf-syntax.md` §P10a; the same rule applies
to Deep.

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
FloatLit    ← ('-'? [0-9]+ '.' [0-9]+ ([Ee] [+-]? [0-9]+)?
            / '-'? [0-9]+ [Ee] [+-]? [0-9]+)       # exponent without decimal
              FloatSuffix?
IntLit      ← '-'? [0-9]+ (FloatSuffix / IntSuffix)?
FloatSuffix ← 'f32' / 'f64' / 'bf16' / 'f16'
IntSuffix   ← 'i8' / 'i16' / 'i32' / 'i64'
# Suffix must immediately follow the digit sequence (no whitespace, no comment).
# `f8e4m3`, `u8`, `u16`, `u32`, `u64` are reserved/deferred or out-of-scope per
# spec/04-type-system.md §1.1.1 / §1.1.2 and are rejected at lex time.
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

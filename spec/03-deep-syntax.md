# spec/03-deep-syntax.md — Chelis Deep Syntax Specification

**Scope:** The primary machine interface. Everything an AI agent or compiler needs to construct, parse, validate, and transform Deep programs.

The 62-tag vocabulary documented here is closed. A form outside that vocabulary is not
Deep unless the controlling specification adds it and updates the vocabulary census in
the same change.

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

**Defined keys:**

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
| `property_contracts` | `(tuple {} string...)` | Ordered repeatable standard-contract dependencies authored with `with contract = "..."` |
| `opaque` | `true` | On a `deftype`: the type is opaque (see §2.2) |
| `invariant` | `(fn {} (params {} <binder>) <expr>)` | On an opaque `deftype`: the declared invariant predicate (see §2.2) |
| `invariant_amenability` | string | On an invariant-carrying `deftype`: `"linear"`/`"polynomial"`/`"transcendental"`/`"opaque"`; derived data, recomputed on desugar (see §2.2) |
| `surf_path` | string | Exact canonical Surf module path spelling; permitted only on `module`, `import`, and `import-all`; its ASCII-lowercased value must equal the node's lowered module-path child |
| `surf_dim_group_size` | positive integer | Number of adjacent `defdim` declarations authored in one Surf `dim` group; permitted only on the first member |
| `surf_pipe_stage` | `"call-first"` | First-argument call-stage origin; permitted only on an `fn` child used as a non-initial `pipe` stage |
| `surf_literal_style` | `"unsuffixed"` / `"explicit"` | Numeric literal origin; permitted only on `lit` |
| `surf_binding_type` | `"inferred"` / `"explicit"` | Block-binding type origin; permitted only on the expression child of a `bind` name/value pair |

The `surf_*` namespace is closed. A public Deep parser or programmatic
validator MUST reject an unknown `surf_*` key. Resugaring MUST also reject a
known key with any value or placement outside the table above; a standalone
metadata map or metadata-expression wrapper is not a permitted
placement. These five keys preserve only surface distinctions that canonical
Deep otherwise erases; they do not change evaluation. Producers MUST NOT use
the namespace for arbitrary provenance.

**Reserved metadata:**

| Key | Value | Semantics |
|---|---|---|
| `lin` | `once` / `borrow` / `unrestricted` | Linearity |
| `doc` | string | Documentation |
| `span_*` | reserved | Span-metadata extension namespace (see §1.1.1) |

**Metadata propagation through transformations.** Semantic metadata and the
validated surface-fidelity keys are preserved by all spec-defined
transformations. Derived metadata may be recomputed according to §6.3.2.
Cross-tool provenance (the `span` field in particular) remains governed by
the span-survival contract below.

#### 1.1.1 External-source spans (`span`, `span_*` namespace)

The `span` metadata key carries a string identifier issued by an external
producer (for example, Octant's LaTeX-to-Deep translator, which writes
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

The `span_*` prefix is the extension namespace for richer span data. Embedded
byte offsets or file identifiers (rather than sidecar references) MUST use the
`span_*` namespace (`span_start`, `span_end`, `span_file`, …). Competing keys
that carry span-related data outside the `span_*` namespace are forbidden so
tooling has a stable contract.

**Reserved synthesized-marker shape.** When a chelis transformation pass
introduces a node with no external-source region (gradient adjoints, lowered
sub-nodes from a parentless intrinsic, etc.), the canonical `span` value uses
the form `__synthesized_<pass>__` (double-underscore wrap, lowercase pass
name). This shape is reserved — external producers MUST NOT emit span IDs
matching `__synthesized_*__`; chelis MUST NOT mint a synthesized marker that
omits the wrap. Defined markers:

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
Every IR field that admits a producer-supplied string (module names, type names,
effect names, and similar values) follows the same pattern. Each emission context must
apply a context-appropriate sanitizer, including separate comment and format-string
handling.

**Provenance metadata.** After macro expansion, each node in the expanded
form may carry a `source` key in its metadata map indicating the macro invocation it
originated from.
Example: `(app {source: (relu input)} (var {} max_elem) (var {} input) (lit {type: (t-prim {} f32)} 0.0))`.
Provenance is informational — it does not affect parsing, type checking, or evaluation.
The node is a standard `app` node; the `source` key is ignored by all compiler passes
except error reporting.
The provenance format is `{source: (macro-name original-arg...)}` where the
value is a plain Deep list recording the macro name and original invocation arguments.

**Macro boundary rule.** LLM-facing Deep is always expanded Deep. Macro
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

A named module may be opened by at most one `(module ...)` wrapper per
program: re-opening a module name forges module identity and is
rejected by both `chelis validate --deep` and the type checker
(`DuplicateModule`; `spec/04-type-system.md` §2.5). Hand-authored Deep
must not use the reef linker's reserved internal-name format
(`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) for declaration
names: that format is the linker's private output, and a raw program
using it forges module identity through the name stem
(`ReservedLinkerName`; `spec/04-type-system.md` §2.5).

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

A `defsig` annotates a Chelis `def`; it does not declare a runtime symbol or
an externally supplied implementation. Every `defsig` therefore has exactly
one same-name `def` in the same module and check unit. A `def` may omit its
`defsig` and use inference. Imports and an existing library context do not
satisfy this pairing rule: a new unit cannot redeclare an imported name with
an unbacked signature. A capability supplied by a host or another lane uses
that capability's explicit typed declaration form, never an orphan `defsig`.
After every authored source module has passed this rule, a trusted package
linker may materialize a dependency interface as signature-only internal
records whose bodies remain in the supplying artifact. Those records are not
an authored check unit, use the linker's reserved-name/provenance channel, and
cannot be produced by source-level `defsig` syntax.

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

An opaque `deftype` may additionally carry a **declared invariant**
(`spec/design/opaque_invariants_rfc.md` D-META):

```lisp
(deftype {opaque: true,
          invariant: (fn {} (params {} p)
            (app {} (var {} and)
              (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0))
              (app {} (var {} lte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 1.0)))),
          invariant_amenability: "linear"}
  Probability ()
  (variant {} Probability (field {} value (t-prim {} f32))))
```

- `invariant` is the predicate encoded as a Deep `fn` inside the
  metadata map (metadata values are full Deep expressions; the strict
  validator recurses into them). The fn has the canonical shape
  `(fn {} (params {} <binder>) <body>)` with exactly one binder. A
  `deftype` *child* node would break positional variant parsing and
  grow the closed tag vocabulary, so the predicate lives in metadata.
- `invariant_amenability` is one of `"linear"`, `"polynomial"`,
  `"transcendental"`, `"opaque"` (the `SmtAmenability` vocabulary).
- **Asymmetry — `invariant_amenability` is derived data.** The
  decompiler reconstructs the `@invariant(<binder>) <expr>` Surf line
  from the `invariant` fn, but does NOT decompile
  `invariant_amenability`: it is recomputed by
  `chelis_pred::classify_predicate` on the next desugar, so
  reconstructing it would be a redundant, drift-prone copy. The checker
  re-verifies the recorded value against a recomputation, which
  protects hand-written `.dp` (`spec/04-type-system.md` section 2.5.1).

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
| `record-update` | `(record-update {} expr (kv {} k v) ...)` | Reserved functional record update |
| `par` | `(par {} expr₁ expr₂ ...)` | Scheduler-independent parallel evaluation |
| `handle-effect` | `(handle-effect {effect: name} arg body)` | Effect handler block |

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
| `t-prim` | `(t-prim {} f32)` | Primitive type (language set: f32, f64, bf16, f16, int8, int16, int32, int64, bool, string — see `spec/04-type-system.md` §1.1; the reserved primitive names of §1.1.1 — `f8e4m3`, `f8e5m2`, `uint8`/`uint16`/`uint32`/`uint64`, `int4`/`uint4`, `complex64`/`complex128`, `decimal128`/`decimal256` — are rejected at check time) |
| `t-fn` | `(t-fn {} arg₁ arg₂ ... ret)` | Function type; last child is return |
| `t-tensor` | `(t-tensor {} dim₁ dim₂ ... precision)` | Tensor type; last child is precision |
| `t-ref` | `(t-ref {} type)` | Read-only borrow type |
| `t-adt` | `(t-adt {} Name type-arg...)` | ADT type application |
| `t-var` | `(t-var {} name)` | Type variable |
| `t-unit` | `(t-unit {})` | Unit type |
| `t-tuple` | `(t-tuple {} type₁ type₂ ...)` | Tuple type |

#### 2.5.1 Type-expression resolution and binders

Type expressions are recursively resolved before they may enter the checked
environment or a cached compiler context. Resolution is fail-closed:

- `t-prim` has exactly one symbol child and that symbol is in the language or
  explicitly-reserved primitive vocabulary owned by `spec/04-type-system.md`
  §1.1. An unknown primitive name is a type error, not an inference hole.
- `t-adt` has a symbol head naming a precollected `deftype` or `typealias`
  header and exactly that header's declared number of type arguments. Headers
  are collected before bodies are resolved, so self-recursive and forward
  nominal references are legal; unknown names and wrong arities are errors.
  The precollected header environment remains in scope for the entire check unit,
  including annotations in declaration bodies. A rejected declaration body is
  not installed in the reusable ADT/alias registry, but its already-declared
  header remains visible until the failing check ends so downstream references
  do not add a spurious `unknown nominal` cascade.
- `t-var`, `d-var`, and `d-rank` introduce no binding by themselves. A name is
  legal only when the surrounding resolution context supplies it: the
  explicit parameter list of a `deftype`/`typealias`, the implicit-generic
  binder set of one `defsig`, or trusted compiler-generated metadata. The
  special `(t-var {} _)` form is an inference hole only at a use site that
  explicitly admits holes; it is not a way to leave a declaration field or
  alias body unresolved.
- A `defsig` implicitly binds each well-formed `t-var`/`d-var`/`d-rank` name on
  first occurrence and reuses that binding throughout the signature. A
  `deftype` or `typealias` binds only names in its explicit parameter list;
  an undeclared variable name is an error. Surf declaration desugaring is
  scope-aware: a declared parameter becomes the corresponding variable form
  at a type/dimension/rank use site, while an unlisted symbolic tensor axis is
  emitted as `d-name` rather than inventing an implicit declaration binder.
- `t-fn` has at least one child (the last is its return type), `t-ref` has
  exactly one child, `t-tensor` has at least one child (the last is a
  primitive or bound type-variable precision), `t-unit` has no children, and
  every nested type and dimension child must resolve. Resolution never drops
  an invalid child, substitutes a wildcard/fresh variable for malformed
  input, or admits an unchecked nominal name.
- A bare atom or an expression tag in a type position is malformed. The sole
  alternate spelling is a language primitive symbol in a `cast` target (for
  example `(cast {} x f16)`); it resolves with
  the same meaning as `(t-prim {} f16)`. Bare forms remain non-canonical and
  are not accepted in declaration fields, aliases, signatures, or metadata.
  Both cast spellings cross this same resolver before cast semantics are
  classified: canonical `t-prim` still requires exactly one symbol child, so
  `(cast {} x (t-prim {} f16 extra))` is malformed rather than a cast to
  `f16` with an ignored child.

Each invalid type expression produces one owning checker diagnostic. Parents
propagate that witnessed failure without re-reporting it, so a malformed
nested type cannot be silently accepted or produce diagnostic spray.

### 2.6 Dimension Expressions

| Tag | Form | Semantics |
|---|---|---|
| `d-name` | `(d-name {} batch)` | Named dimension (concrete) |
| `d-var` | `(d-var {} a)` | Dimension variable (polymorphic) |
| `d-lit` | `(d-lit {} 512)` | Literal dimension size |
| `d-rank` | `(d-rank {} r)` | Rank variable — a name-preserving spread standing for a run of dims (rank polymorphism). Tier-2 uses it as the sole dim child; Tier-3 (§4.5.3) allows it interleaved with concrete anchors (`(t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32))`). A given rank name appears at most once per `t-tensor`. |

Every dimension tag has exactly one child: a symbol for `d-name`, `d-var`,
and `d-rank`, or an integer for `d-lit`. Unknown tags, missing/extra children,
wrong child kinds, and unbound `d-var`/`d-rank` names are type-resolution
errors. `d-name` is a concrete symbolic axis label (with `*` the explicit
wildcard spelling); it does not allocate an inference variable.

### 2.7 Transforms

| Tag | Form | Semantics |
|---|---|---|
| `grad` | `(grad {} expr)` or `(grad {wrt: ...} expr idx-or-idx-tuple)` | Reverse-mode AD; `wrt` child selects parameter indices |
| `vmap` | `(vmap {} expr dim)` | Vectorization |
| `jit` | `(jit {} expr)` | Compilation trigger |
| `realize` | `(realize {} expr)` | Force DAG evaluation |
| `cast` | `(cast {} expr target-type)` or `(cast {} expr target-type mode)` | Precision cast; the optional `trunc` mode selects [05-OP-6] |
| `copy` | `(copy {} expr)` | Explicit tensor duplication |
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
| Dimensions | 4 | d-name, d-var, d-lit, d-rank |
| Transforms | 6 | grad, vmap, jit, realize, cast, copy |
| Meta | 3 | quote, unquote, splice |
| Helpers | 5 | params, bind, kv, effects, resource |
| **Total** | **62** | |

### 2.11 Vocabulary Extension

Any feature that requires a dedicated Deep form must revise this closed-vocabulary
section, structural validation, canonical printing, and the tag-count summary in the
same semantic change. Functionality expressed through ordinary calls does not allocate
a new tag.

---

## 3. Built-In Scope

These names are available without import. They are NOT tags — they are functions/values referenced via `(var {} name)` and called via `(app {} ...)`.

### 3.1 RISC Primitives (~12 ops)

The irreducible computational basis. All tensor computation decomposes to these during IR lowering.

**Elementwise:** `add`, `mul`, `exp`, `log`, `sin`, `sqrt`, `cmplt`, `max_elem`
**Reduce:** `sum`, `count`, `max_reduce` (over one-or-more positional or
one-or-more named axes, never a mixture)
**Movement:** `reshape`, `permute`, `expand`, `pad`, `shrink`, `stride`
**Memory:** `const`, `load`

### 3.2 Derived Functions

Convenience functions with ordinary call syntax and operation-specific lowering
points. The desugarer emits their typed identities. An identity remains intact
through every semantic transform its governing atom names, including AD, and
only then may the IR passes decompose it to RISC primitives.

`sub`, `div`, `neg`, `lt`, `eq`, `neq`, `gt`, `gte`, `lte`, `and`, `or`, `not`, `relu`, `sigmoid`, `softmax`, `matmul`, `linear`, `mean`, `dropout`, `stop_gradient`

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

A nested application is nevertheless a distinct valid shape when the inner
application returns a function value. `(app {} (app {} f x) y)` is not
flattened to `(app {} f x y)`; canonical Surf writes it `(f(x))(y)`. The same
grouped-callee rule preserves `if`, `fn`, unary, and other expression-valued
callees.

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

### 4.4 Evaluation Order

In `(app {} f a₁ ... aₙ)`, the argument expressions `a₁ ... aₙ` evaluate in
written order, left to right, each to completion before the next begins, and
all before the application itself. Observable effects occur in that order,
and the first argument whose evaluation traps determines the trap the
application raises; later arguments are not evaluated after a trap.

This order is a semantic contract in every executable lane, not an
implementation convenience. Value-level rewrites — a derived built-in's
lowering to RISC primitives (`spec/05-risc-primitives.md` §3), constant
folding, or backend scheduling of the already-evaluated dataflow — operate
on argument *values* and never reorder or skip the evaluation of argument
*expressions* whose effects or traps are observable. Surf operator
expressions inherit this order through their desugaring, which preserves
the authored operand order for every operator (`spec/02-surf-syntax.md`
§2). Multi-value constructors follow the same written-order rule: tuple,
list, record, and record-update children evaluate left to right (§6.2's
`kv` ordering restates this for records).

Within a single primitive, elementwise and reduction evaluation order is
owned by `spec/04-type-system.md` [04-NUM-12] and [04-NUM-15]; this section
orders the argument expressions that produce a primitive's operands.

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
- Record and record-update `kv` pairs: written order, which is left-to-right
  evaluation order.
- Match arms: declaration order (semantically meaningful).
- Bind pairs in `let`: declaration order (sequential semantics).

### 6.3 Comments
None in canonical Deep. Comments are Surf-only. Stripped during desugaring. Use the `doc` meta key for structured documentation.

### 6.3.1 Canonical Surf resugaring

Every structurally valid public Deep tag has a canonical Surf representation.
Deep `pipe` resugars as a pipeline. A Deep `app` chain resugars as that same
pipeline only when the typed proof from `spec/01-nomenclature.md` §3.6
establishes a linear first-argument chain; otherwise it remains a flat
parenthesized call. Later-position insertion retains an explicit lambda.
Resolved ordinary calls to the fixed operator builtins use Surf infix/prefix
notation, while the same builtin name remains a value or pipe stage. A finite
`Cons`/`Nil` chain uses bracket-list syntax; an open-tail `Cons` remains an
explicit call. Explicit `borrow` and `copy` nodes remain explicit.

Deep `block` uses `do { e1; e2; ... }`, `record-update` uses
`base with { field: value, ... }`, and `quote`, `unquote`, and `splice` use
same-named call-like forms. A matching `defsig` and `def` resugar as one inline
typed Surf definition; a standalone `defsig` remains `sig`. A checked standalone
`def` carrying semantic `type` metadata resugars as a typed Surf declaration;
normalization materializes the equivalent `defsig` rather than erasing the type.
All ordered pairs in one `bind` become ordered Surf block bindings. Empty
`tuple` and `t-tuple` nodes normalize to the language's unit value and type;
empty `pat-tuple` is written `()` directly. Every zero-argument `app`, including
an application whose callee is an uppercase constructor, remains an explicit
call such as `Ctor()` or `f()`. A bare uppercase `var` remains `Ctor`, and a
zero-field `record` remains `Ctor {}`.

The normal and debug emitters share this AST-backed resugarer and Surf printer.
Debug output may append stable `-- deep-debug: ...` comments; it is not a
second Surf dialect.

Surf property declarations represent user-authored properties only. They have
no syntax for the non-forgeable `bridge:c-earchin` producer identity or for a
producer-local `property_source_id`. Resugaring a property with either form of
provenance therefore fails explicitly; it must never emit an ordinary
`@property` that would redesugar with `property_source_kind: "user"`.
`property_quantifiers` must also be present and exactly match the property
`fn` parameter list before resugaring; a mismatch fails rather than changing
the bound names.

### 6.3.2 Round-trip normalization

`normalize_deep` may erase `span`, `loc`, macro `source` provenance after
expansion, inferred `effects`, and `invariant_amenability` because those values
are informational or deterministically recomputed. It may also erase
matching `type` entries on a `def`, its `fn` value, and its function parameters
when an adjacent matching `defsig` already carries the exact same types; a
disagreement is never erased. For a standalone checked `def`, normalization may
materialize that metadata as an adjacent `defsig` and then apply the same exact
redundancy rule. Empty `tuple`/`t-tuple` normalize to `lit`/`t-unit`. No `app`
normalizes to a `var`: `Ctor`, `Ctor()`, and `Ctor {}` retain their distinct
`var`, `app`, and `record` structures. Because Surf negative
numerals are unary minus rather than signed tokens, a negative Deep `lit`
normalizes to the equivalent `neg` application. The full `int64` minimum uses
Surf's directly representable signed-minimum literal; a narrower signed minimum
uses `sub(neg(max), 1)` so its positive magnitude never overflows that literal
width.
An integer atom carrying a float primitive type normalizes to the equivalent
float atom before that sign rule. A valid, correctly placed non-semantic
`surf_literal_style` or `surf_binding_type` origin marker may be erased after it
has selected the canonical Surf reconstruction; desugaring that Surf recreates
the applicable marker. Normalization may likewise erase a `surf_path` equal to
the deterministic default spelling of its lowered path child and a
`surf_dim_group_size: 1` marker on the first member of its one-member group,
because canonical desugaring deterministically recreates those defaults. A
non-default, malformed, or misplaced marker is retained so the round-trip
oracle cannot hide a resugaring error. Normalization may not erase or rewrite
any other declared `type` or `eff` data, handler effects, `wrt`, `opaque`,
`invariant`, property semantics, or validated `surf_*` values. Implementations
compare macro-authored Surf after expansion. Any other metadata loss is a
round-trip failure.

A negative `pat-lit` is not normalized to an application because patterns do
not contain expression nodes. It resugars as minus followed by the one
unsuffixed canonical numeric pattern token, including `-0.0` and the full
`int64` minimum.

### 6.4 Literal Normalization

| Type | Canonical | Normalizations |
|---|---|---|
| Integer | Decimal, no leading zeros | `07` → `7` |
| Float | Finite, shortest round-trippable value spelling, with `.0` when otherwise integer-like | `1.` → `1.0`, `.5` → `0.5` |
| Float (sci) | Lowercase `e`, only when selected by the shortest printer | equivalent longer spellings normalize to the printer result |
| String | Double-quoted; named Surf escapes where available, otherwise minimal lowercase `\u{h}` for control scalars | Printable-character and named-escape Unicode aliases are not canonical |
| Boolean | `true` / `false` | |

Canonical Deep contains no non-finite float literal. Producers that construct
Deep programmatically must reject NaN and infinity before serialization;
Deep-to-Surf resugaring reports either as unrepresentable rather than emitting
an invalid Surf token.

Every valid Deep string atom has a Surf representation. Resugaring uses the
single P11 spelling: printable Unicode remains literal, the six named escapes
are preferred, and other C0/C1 controls use minimal lowercase `\u{h}`.

When a `lit` node's `type` metadata resolves to a primitive, the value atom
and primitive family have one closed canonical pairing:

| Value atom | Permitted primitive family |
|---|---|
| integer | `int8`, `int16`, `int32`, `int64` |
| float | `f16`, `bf16`, `f32`, `f64` |
| boolean | `bool` |
| string | `string` |

Type metadata is not a cast. Every cross-family pairing is a type error, not
a conversion or a request to reinterpret the atom. There is one explicit
source-preserving form: an integer-spelled token bound directly at a float
dtype carries its exact Int atom together with `literal_source: integer` in
the `lit` metadata. This covers a suffix such as `7f32` and an integer element
in a contextually `f32` tensor literal. The marker is valid exactly once, only
with an Int atom and a float primitive; every other use is a type error. It
keeps the exact integer available for the one target-width rounding required
by [04-NUM-1] and [04-NUM-14], instead of first rounding through f64. An
unmarked Int atom under a float primitive remains contradictory. An explicit
`cast` is the only form that converts an already-typed literal value between
primitive families. `literal_source` is producer-asserted provenance, not a
lexer authenticity proof: hand-written Deep MAY author the canonical marker,
and the checker validates its closed atom/primitive/uniqueness contract before
any consumer may rely on it.

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

No suffix exists for any reserved name of `spec/04-type-system.md` §1.1.1
(`f8e4m3`, `f8e5m2`, the `uint*` family, `int4`/`uint4`,
`complex64`/`complex128`, `decimal128`/`decimal256`); each is rejected at
lex time with a diagnostic citing §1.1.1; none is a literal suffix in this
grammar. The short unsigned spellings (`u8`, `u16`,
`u32`, `u64`) are not reserved in any form - `uint8`/`uint16`/`uint32`/
`uint64` are canonical per §1.1.2 - and are likewise rejected at lex time.
Deep's producer-friendly lexer accepts hexadecimal integer input and
canonical Deep printing rewrites the decoded value in decimal. Canonical Surf
also accepts value-preserving hexadecimal and binary integer spellings as
input; the shared Surf printer owns the same decimal rewrite.

### 6.5 Identifier Rules
- Variables/functions: `[a-z_][a-z0-9_]*` (snake_case)
- Types/variants: `[A-Z][a-zA-Z0-9]*` (PascalCase)
- Module paths: dot-separated identifiers

Deep's lexer admits a broader symbol alphabet for tag and producer use, but a
public declaration, expression, pattern, type, field, or module name that is to
be resugared MUST satisfy its corresponding Surf identifier rule and MUST NOT
be a reserved Surf word. Deep-to-Surf resugaring rejects an invalid name rather
than quoting it, rewriting it, or emitting text with changed meaning.

---

## 7. Grammar (PEG)

```peg
Program     ← Spacing TopLevelForm+ EOF
TopLevelForm ← &('(' Spacing TopLevelTag) Node        # §7.1
TopLevelTag ← ('module' / 'import-all' / 'import' / 'export'
            / 'defsig' / 'deftype' / 'defdim' / 'def'
            / 'typealias') ![a-z0-9-]                 # longest-first; tag must end here
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
# No suffix exists for any reserved name of spec/04-type-system.md §1.1.1
# (`f8e4m3`, `f8e5m2`, `uint*`, `int4`/`uint4`, `complex*`, `decimal*`); the
# short unsigned spellings `u8`/`u16`/`u32`/`u64` are not reserved at all.
# Every such sequence is rejected at lex time with a diagnostic citing §1.1.1.
BoolLit     ← 'true' / 'false'
StringLit   ← '"' (!'"' .)* '"'
Spacing     ← ([ \t\n\r] / Comment)*
Comment     ← ';' (![\n] .)*
EOF         ← !.
```

### 7.1 Top-Level Form

A Deep program is a namespace, not an expression. Deep has no top-level
evaluation position: a form at top level either introduces a name into the
program's namespace or declares module structure, and there is no other
position for it to occupy. A value produced at top level could not be named,
given a signature, exported, selected as a root, lowered, or observed, so a
program whose top level is an expression carries no content a consumer can
act on.

> **[03-PROG-1]** A Deep program SHALL consist of one or more top-level
> forms. Each top-level form SHALL be a `module` node (§2.1) or a declaration
> node whose tag is one of `def`, `defsig`, `deftype`, `typealias`, `defdim`,
> `import`, `import-all`, or `export`. Every other top-level form SHALL be
> rejected: an expression node, a pattern node, a type-expression or
> dimension-expression node, a transform node, a helper node, a bare
> identifier, a bare literal, an untagged list, and a node whose head is
> outside the closed vocabulary (§2). `variant` and `field` are structural
> children of `deftype` and `variant`; they bind nothing on their own and are
> not top-level forms.

The declarations a program contains are the children of its `module`
wrappers together with its bare top-level declarations; a program MAY mix
both spellings and MAY contain more than one `module` wrapper, subject to the
one-wrapper-per-module-name rule in §2.1.

Most of what [03-PROG-1] rejects is a list headed by a tag symbol, which the
rejection can name. Some of it has no head at all: a bare identifier, a bare
literal, an empty list, and a list whose first element is not a symbol are all
[03-PROG-1] rejections with nothing to quote. Those forms are identified by
syntactic class instead, from a closed set, so that a reader of the diagnostic
always learns which form was rejected.

> **[03-PROG-2]** A rejection under [03-PROG-1] SHALL identify the offending
> form and SHALL carry that form's source location. A form headed by a symbol
> SHALL be identified by that symbol. A form with no head SHALL be identified
> by its syntactic class, which SHALL be exactly one of: a bare identifier, a
> bare integer literal, a bare float literal, a bare string literal, a bare
> boolean literal, an empty list, a list without a tag symbol, a metadata map,
> or a metadata-annotated form. An implementation SHALL NOT substitute a
> placeholder for either identification. The rejection SHALL be reported at
> the ingress boundary that reads the program text, before name resolution,
> type checking, evaluation, lowering, or resugaring observes the program. An
> implementation SHALL NOT skip, ignore, or silently reinterpret a top-level
> form that [03-PROG-1] rejects.

[03-PROG-1] requires at least one top-level form, so text that yields none is
rejected too. That rejection is the one case with no offending form to
identify and no form location to carry, so the contract states its own shape
rather than leaving an implementation to invent a placeholder.

> **[03-PROG-3]** Program text that yields no top-level form SHALL be rejected
> under [03-PROG-1]. Text yields no top-level form when it is empty, when it
> is entirely whitespace, when it is entirely comments, or when it is any
> combination of those. That rejection SHALL identify itself as an empty
> program and SHALL carry the source position at which a top-level form was
> required, which is the end of the input. It is otherwise subject to
> [03-PROG-2]'s reporting rules.

The class set is closed because it partitions what the grammar can produce in
top-level position: `Child`'s three alternatives (`Node`, `BareName`,
`Literal`) plus the metadata forms a producer may emit. A bare identifier is
the `BareName` production, admissible inside `params`, `bind`, and `field`
contexts (§7) and never at top level; the four literal classes are `Literal`'s
alternatives, with `IntLit` and `FloatLit` distinguished because a producer's
mistake is usually specific to one.

---

## 8. Validation Rules

### 8.1 Structural Validation (Parser)
- Every node is a 3-tuple: `(tag meta children...)`.
- Tag is from the closed vocabulary (§2).
- Every top-level form satisfies [03-PROG-1]; a violation is rejected per
  [03-PROG-2].
- Meta is a valid `{}` map (may be empty).
- A colon-prefixed keyword token such as `:type` is metadata-key syntax, not
  an expression atom or node child. Outside metadata-key position, the parser
  MUST reject it before checker or evaluator processing.

### 8.2 Arity Validation (Post-Parse)
- `(if {} cond then else)` — exactly 3 children.
- `(arm {} pattern guard body)` — exactly 3 children.
- `(fn {} params body)` — exactly 2 children; first must be `(params ...)`.
- `(let {} bindings body)` — exactly 2 children; first must be `(bind ...)`.
- `(app {} func arg...)` — at least 1 child (the function).
- Type/dimension nodes obey the recursive shapes and binder rules in
  §2.5.1/§2.6. The type checker owns these semantic checks because nominal
  headers and binder context are not available to the syntax parser.

### 8.3 Unknown Tags
Unknown tags are parse errors in strict mode (canonical validation). In fitness-scoring mode, unknown tags are parsed as generic nodes and penalized in the fitness score. This distinction applies below the top level; an unknown tag in top-level position is a [03-PROG-1] rejection in every mode, because no unknown head is one of the top-level forms that rule enumerates.

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

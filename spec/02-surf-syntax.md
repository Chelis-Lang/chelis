# spec/02-surf-syntax.md — Chelis Surf Syntax Specification

## 0. Notation

PEG notation. `/` is ordered choice. `*` is zero-or-more. `+` is one-or-more. `?` is optional. `!` is negative lookahead. `&` is positive lookahead. Literal strings in single quotes.

Deep desugaring shown as **⟹** with the target Deep s-expression.

### 0.1 Canonical Surf and the bidirectional contract

Canonical Surf defines the formatting of each grammatical construct, rather
than one spelling per program. Authored operator sugar, including `|>`, is
preserved by `fmt`. Deep decompilation prints calls for normalized pipes and
canonical operator sugar for recognized operator builtins. The
normal parser also accepts the explicitly value-preserving
input families in P10-P12: numeric radix/digit-separator/exponent spellings,
float bodies carrying decimal digits beyond the shortest round-trippable
spelling, equivalent valid string escapes, whitespace before a parenthesized
call argument list, and trailing separators in delimited forms.
`chelis fmt` prints those inputs in canonical form and is idempotent on its own
output; `chelis fmt --check` enforces that output form in a repository. The
versioned `chelis migrate surf --from 0.18` path remains for genuinely legacy,
semantically incompatible, or otherwise non-normal syntax.
That migration rewrites the retired integer type spellings `int8`, `int16`,
`int32`, and `int64` to `i8`, `i16`, `i32`, and `i64` only in type positions
and cast targets. Normal parsing does not accept the retired spellings as
primitive types.

With `--inplace`, migration is a batch transaction over ordinary source files.
Before replacing any input, the command MUST preflight every path and reject a
symbolic link or a file with multiple hard links. Each individual replacement
MUST be atomic. If a later replacement fails, every earlier replacement MUST be
restored byte-for-byte before the command reports failure.
For `migrate pipes`, explicit `--keep-going` with `--check` or `--inplace`
selects independent per-file processing instead: every file MUST receive its
own complete expanded-Deep preservation proof; a rejected file MUST remain
unchanged, later files MUST still be attempted, and any failure MUST produce
a nonzero exit with a per-file failure summary. The default remains a batch
transaction.
Migration MUST NOT guess a replacement for an identifier that v0.19 reserves.
It MUST reject that input with a diagnostic directing the author to rename the
identifier manually before rerunning migration.

The canonical forms are:

- every function definition has a parameter list, including `()` for a
  nullary function, and an annotated result uses `->`;
- direct calls and constructor applications are flat and parenthesized
  (`f(x, y)`, `Some(x)`, `None()`). A bare constructor such as `None` is a
  constructor reference/value rather than an application. Applying a value
  returned by another expression uses explicit grouping (`(f(x))(y)`), while
  ungrouped `f(x)(y)` is rejected. The parser accepts whitespace before the
  argument list (`f (x)`, `Some (x)`), and the formatter removes it;
- consecutive numeric tuple projections group the receiver
  (`(nested.0).1`), because the ungrouped spelling `nested.0.1` would lex its
  adjacent numeric suffixes as a floating-point token. Ordinary single
  projections and mixed field/projection chains need no grouping;
- ordinary binding blocks use newlines as separators, contain at least one
  binding and one tail expression, and contain no semicolons; `par` and direct
  Deep sequencing use their distinct semicolon-delimited forms;
- the unit value is `()` and the unit type is `unit`; a singleton tuple is
  `(x,)`;
- canonical output omits trailing commas and semicolons, but the parser accepts
  one trailing separator in a delimited nonempty family. The comma in a
  singleton tuple or singleton tuple pattern is semantic rather than cosmetic;
- built-in effects are spelled `Diff`, `Accum`, `IO`, `Test`, and
  `Resource(...)` with exactly that casing;
- record fields preserve source order and evaluate left-to-right; a field
  whose value is the same-named variable is written as a pun;
- decorative zero-arity syntax has no empty delimiters: empty type or dimension
  arguments are omitted, a zero-field type variant is bare, and a
  qualified-only import omits `()`; exports and record updates must be
  nonempty. This omission rule does not apply to expressions: `Ctor()` is the
  explicit zero-argument Deep `app`, while bare `Ctor` is a Deep `var`. A
  zero-field record expression or record pattern retains `{}` to distinguish
  Deep `record`/`pat-record` from either form. An explicit empty effect row
  `! {}` is likewise semantic: it declares a pure upper bound and is distinct
  from an omitted, inferred effect clause;
- non-primary transform arguments are named: `grad(f, wrt=x)` and
  `vmap(f, axis=n)`. Axis zero is written `vmap(f)`;
- literal output is the canonical literal printer's decimal spelling for the
  decoded value and suffix: no digit separators or redundant leading/trailing
  zeroes; a float body uses the shortest round-trippable form with a decimal
  point or canonical exponent when required, and an integer body under a float
  suffix keeps its integer spelling (P10a). P10-P11 define the equivalent
  spellings accepted as input and the semantic forms that remain rejected.

Canonical Deep and canonical Surf are two representations of the same public
language. The required executable laws are:

```text
desugar(resugar(deep)) = normalize_deep(deep)
format(format(surf)) = format(surf)
normalize_deep(desugar(resugar(desugar(surf))))
  = normalize_deep(desugar(surf))
```

The formatting law is syntactic: it chooses a stable spelling for the authored
Surf constructs. The third law is the semantic retraction after
Deep has erased comments, whitespace, and distinctions among equivalent Surf
sugars; it does not require `resugar(desugar(surf))` to reproduce authored
bytes. It compares macro programs after expansion, because public Deep is
expanded Deep. `normalize_deep` may erase only the derived metadata enumerated by
`spec/03-deep-syntax.md` section 6.3.2. A well-formed public Deep node has a Surf
representation unless it carries non-forgeable producer provenance that Surf
deliberately cannot author; §6.3.1 defines that fail-closed exception. An
unmapped tag is an implementation or specification bug. A `grad` carrying
`wrt` is in the resugaring domain only when its operative selector is a
well-formed integer selector for a statically resolved callable origin and
agrees exactly with the ordered metadata names. Contradictory, malformed,
absent, or dynamically unresolved selectors fail resugaring; the round-trip
laws do not authorize dropping or reconstructing the operative child.

### 0.2 Pipe sugar and explicit grouping

> **[02-PIPE-1]** `|>` exists only in Surf. Desugaring SHALL normalize
> `x |> f(y)` to `f(x, y)` before literal dtype selection, with bare `f`
> meaning `f(x)`. `cast(T)`, each named cast rung (`cast_trunc(T)`,
> `cast_saturate(T)`, `cast_wrap(T)`), bare `copy`, and bare `realize`
> stages normalize to their corresponding operand forms. General authored
> lambdas remain function values applied to the carried expression; no
> arbitrary beta reduction or capture-prone substitution is permitted.

> **[02-PIPE-2]** Pipes SHALL NOT mix at the same ungrouped expression boundary
> with non-pipe binary or unary operators, postfix field access or tuple
> projection (including qualified callees), type ascription, record update,
> or open-ended `if`, `match`, or lambda forms. This rule is symmetric:
> a pipe in an ungrouped condition, branch, scrutinee, or lambda body also
> requires grouping. Parentheses,
> argument lists, and delimited blocks establish independent expression
> boundaries. Thus `(a * b) |> f`, `a * (b |> f)`,
> `(if c then a else b) |> f`, and `x |> (fn (v) -> v + y)` are valid;
> `a * b |> f`, `-x |> f`, `if c then a else b |> f`, and
> `x |> fn (v) -> v + y`, `x |> f.1`, and `r.field |> f` are rejected with
> a grouping diagnostic. `x |> (f.1)`, `(x |> f).1`, `(r.field) |> f`,
> `if c then (x |> f) else y`, and `fn (v) -> (v |> f)` are valid.

> **[02-PIPE-3]** Formatting SHALL preserve authored pipe sugar and comments,
> be type-independent and idempotent. Deep SHALL contain only the normalized
> operation and no pipe spelling-restoration metadata. Diagnostics SHALL
> refer to authored stage spans and source, not synthesized call text.


---

## 1. Keywords

Lexically reserved. These words cannot be used as identifiers.

```
def  sig  type  dim  macro  match  with  fn  module  import
export  if  then  else  grad  vmap  jit  realize  copy  tensor
cast  cast_trunc  cast_saturate  cast_wrap  par  do  quote  unquote
splice  true  false
```

**Total: 31.**

`property`, `forall`, `where`, `opaque`, `invariant`, `wrt`, `axis`,
`device`, the dtype-family names `Float`, `Int`, and `Numeric`, and
the property-option names are contextual words only in the productions that
name them. A dtype-family name is a family only in a type binder's bound
position; everywhere else it is an ordinary type name. `effect`, `handler`, `perform`, `resume`, and
`borrow` are reserved words and cannot be used as identifiers.
The read-only borrow expression is spelled `&expr`.

---

## 2. Operator Precedence

Final. Binding power from lowest to highest:

| BP | Operators | Assoc | Deep form |
|----|-----------|-------|-----------|
| 1 | `\|>` | left | nested `app` (first-argument insertion) |
| 2 | `\|\|` | left | `(app {} (var {} or) ...)` |
| 3 | `&&` | left | `(app {} (var {} and) ...)` |
| 4 | `==` `!=` | none | `eq` / `neq` |
| 5 | `<` `>` `<=` `>=` | none | `cmplt` / `gt` / `lte` / `gte` |
| 6 | `+` `-` | left | `add` / `sub` |
| 7 | `*` `/` `%` | left | `mul` / `div` / `mod` |
| 8 | unary `-` `!` | prefix | `neg` / `not` |
| 9 | function application | left | `(app {} ...)` |
| 10 | `.` field access | left | `(access {} ...)` / `(tuple-get {} ...)` |

Non-associative operators (BP 4, 5) produce a parse error on chaining: `a == b == c` is rejected.

Every operator row desugars with the authored operand order preserved: an
application row `a OP b` becomes `(app {} (var {} op) a' b')` with `a'`
first, no row swaps its operands, and `|>` keeps its stage order in the
application chain. `a > b` therefore desugars to the `gt` built-in, whose result
is defined as `cmplt(b, a)` over the already-evaluated operand values
(`spec/05-risc-primitives.md` §3.2), not to an operand-swapped `cmplt`
application. Combined with Deep's left-to-right application-argument
evaluation (`spec/03-deep-syntax.md` §4.4), the effects and traps of
operand expressions are observed in authored order. (`&&` and `||` are
eager like every other application; there is no short-circuit special
case.)

In a pipe stage, `x |> cast(p)`, `x |> cast_trunc(p)`,
`x |> cast_saturate(p)` and `x |> cast_wrap(p)` insert `x` as the value
argument of the corresponding two-argument cast. The one-argument spelling
is valid only after `|>`; `cast(p)` and the named forms `cast_trunc(p)`,
`cast_saturate(p)` and `cast_wrap(p)` are not standalone expressions.

No operator overloading. No infix bitwise operators. Host-side integer bitwise work uses
named built-ins such as `bitand`, `bitor`, `bitxor`, `shl`, and `shr`. No exponentiation
operator — use `pow(x, n)` from `Std.Math`.

The `/` operator desugars to `div`, which is **float-only**:
applying `/` (or `div`) to integer operands is a type error. Integer division
uses the named built-ins `floor_div(a, b)` (round toward −∞, matching Python `//`
/ torch / JAX / numpy `floor_divide`) and `trunc_div(a, b)` (round toward zero,
the C `/` quotient; integer-only). There is intentionally no infix operator for
either — integer division is explicit at the call site. See
`spec/05-risc-primitives.md` §2.1 for the full semantics. `%` continues to map
to `mod` (integer remainder).

---

## 3. Punchlist Decisions

### P1: Module System

One module per file. `module Name` is the first non-comment line. Declarations follow until EOF — no body braces. Module names are PascalCase. No nesting within a file.

File path mapping: `module Foo.Bar` lives in `foo/bar.ch` relative to the project root.

```
module School.Nn.Linear

import Std.Tensor (..)

def forward(x, w, b) = add(matmul(x, w), b)
```

**⟹** `(module {} school.nn.linear ...)` — module name lowercased and dot-joined in Deep.

### P2: Import / Export

| Surf form | Meaning | Deep form |
|-----------|---------|-----------|
| `import Foo.Bar (baz, qux)` | Import specific names | `(import {} foo.bar (baz qux))` |
| `import Foo.Bar (..)` | Import all exported | `(import-all {} foo.bar)` |
| `import Foo.Bar` | Qualified access only | `(import {} foo.bar ())` |

Qualified access (`Foo.Bar.baz`) is always available after any import form. Selective import additionally brings names into unqualified scope.

A source file belongs to the reef package whose manifest is found by walking up from the file's own directory. Every command classifies a file this way, and the directory a command runs in plays no part in which package a file belongs to or in what its imports resolve to. An explicit package option is a declared input and the one way to name another package: `chelis prove --package DIR` links a file with no `module` declaration as an entry of the package at `DIR`, and rejects a file that declares a `module` when the file's own location finds a different package. A file in a package that is not one of its modules (`spec/01-nomenclature.md` §6.5 decides which files are) is an entry of the package whose manifest the walk finds, and it imports that package's modules and dependencies as one of its modules does. A file that belongs to no package resolves its imports against the compiler-bundled `chelis-std` runtime under the same binding, visibility, and collision rules as a package module, and an import of a module it cannot reach that way, including every module outside `chelis-std`, is an error that names the module.

**Import and local declaration collision.** A module never both imports a name into unqualified scope and declares it. When a selective import names a name, or an `import M (..)` brings it in because `M` exports it, and the module also declares the same name at top level (a definition, signature, top-level binding, type, constructor, macro, dimension, or property), every command rejects the module with a diagnostic that names the name, the import, and the local declaration. No precedence rule chooses either binding. A qualified-only `import M` brings no name into unqualified scope and so never collides: `M.name` stays available beside a local `name`. Parameters and block-local bindings are not top-level declarations; they shadow an imported name under ordinary lexical scoping. A package entry and a file that belongs to no package follow the same rule.

A qualified reference names the module path followed by the exported name. The
trailing name may be a value or a **constructor**, and the whole reference may
be applied:

```
Foo.Bar.baz                 -- qualified value
Demo.Dropout.Eval           -- qualified nullary constructor
Demo.Dropout.use(mode)      -- qualified value applied to an argument
Demo.List.Cons(x, xs)       -- qualified constructor applied to arguments
```

A constructor may also be qualified in **pattern** position, so a `match` can
destructure one module's variant when same-named constructors are in scope:

```
match m with {
  | Demo.Dropout.Train => 1
  | Demo.Dropout.Eval  => 0
}
```

A **type** name may be qualified the same way in any type position, so a
consumer that imports two modules exporting the same type name can annotate
against one:

```
def relay(m: Demo.Dropout.Mode) -> i64 = Demo.Dropout.use(m)
```

This is the disambiguation escape hatch when two imported modules export the
same constructor or type name (e.g. each defines `type Mode = | Train | Eval`):
write `Demo.Dropout.Eval` and `Demo.Sd.Eval` to select each module's own
constructor. Importing both names unqualified is rejected as an ambiguous
reference; qualifying resolves it. In every position — value, constructor,
pattern, and type — a qualified reference whose head names an imported module
but whose trailing name that module does not export is rejected with a
`module \`M\` does not export \`N\`` error, not silently accepted.

**Value scope (unqualified references).** A bare value reference is in scope
only when it names a lexical binding, a function declared in the enclosing
module, a non-function value declared earlier in the enclosing module
(`spec/04-type-system.md` [04-INF-4]), a builtin, or a value brought into
unqualified scope by an `import`.
An exported value in another linked module does not enter scope merely because
its terminal name is a unique match. A bare name with no in-scope binding is an
`unbound variable: X` error at `chelis check`; adding, removing, or renaming an
unimported module cannot change that verdict. The qualified forms above remain
available after importing the declaring module, and naming the value in a
selective import brings it into unqualified scope.

Function calls do not give a named declaration access to its caller's locals.
Its free references keep their declaration scope; an anonymous `fn` instead
captures the lexical bindings at its creation site, including local shadows of
declarations. Arguments are evaluated left-to-right in the caller's scope before
the callee's parameters are installed ([04-LIN-1], [04-LIN-2]).

**Constructor scope (unqualified references).** A bare (unqualified)
constructor reference — at a construction site (`Alpha`, `Alpha(x)`,
`Alpha { ... }`) or in a `match` **pattern** (`| Alpha => ...`) — is in scope
only when the constructor is declared in the enclosing module **or** named in an
`import` that brings it into unqualified scope. Importing only the enclosing
**type** is not sufficient: the constructor itself must be named in the import
list (e.g. `import Pkg.Adt (Mode, Alpha, Beta, Gamma)`). A constructor that is
not in scope is an `unknown constructor \`X\`` error at `chelis check` that
names the constructor — the same way an unbound value is an `unbound variable`
(type-checker diagnostics name the offending identifier) — and must never
silently bind to a same-named constructor
declared in another module (which would defer the failure to a runtime
non-exhaustive match). This applies to record-shaped constructors
(`Alpha { ... }`) at both construction and match-pattern sites, and to the
case where two other modules export a same-named constructor (the reference is
rejected as unknown, not bound to either). The
module-qualified forms above (`Pkg.Adt.Alpha`, `| Pkg.Adt.Alpha =>`) remain in
scope without naming the constructor in the import list, because they name the
declaring module explicitly.

Naming a constructor in an `import` brings it into scope **even when the
declaring module's `export` list names only the type**: exporting a type
auto-exports its constructors (see Export below), so they are importable by
name regardless of whether they appear in the `export` list.

**Export:** Explicit. If no `export` declaration appears, all top-level `def` and `type` are public. Once any `export` appears, only listed names are public. Exporting a `type` also exports its constructors, so a downstream module can import them by name.

```
export (forward, Linear)
```

**⟹** `(export {} forward Linear)`

Surf has no re-export form.

### P3: Dimension Declaration

Module-level `dim` for concrete dimensions. Function-level `[...]` brackets for polymorphic dimension parameters.

```
dim batch, vocab_size

def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] =
  permute(x, [1, 0])
```

**⟹**
```
(defdim {} batch)
(defdim {} vocab_size)
(defsig {} transpose (a b) (t-fn {} (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))
                              (t-tensor {} (d-var {} b) (d-var {} a) (t-prim {} f32))))
(def {} transpose (fn {} (params x) ...))
```

**Disambiguation:** A name in a `dim` declaration or imported → concrete
`d-name`. A name in a function's `[...]` → variable `d-var`. A lowercase
single-letter name in a tensor type that is neither declared nor in brackets
remains a `d-var` use and is rejected as undeclared by the checker; a
multi-letter name remains a concrete `d-name`.

### P3b: Rank Variables (`..r`)

A **rank variable** `..r` is a *name-preserving spread* standing for a run of
dimensions, letting one `def` be generic over tensor *rank*. It binds (by
unification) to the actual named dims it covers, so per-axis names are
preserved, not erased. See
[`spec/design/rank_polymorphism.md`](design/rank_polymorphism.md) for the full
design and soundness boundary, including §4.5.3 (named-axis reduction).

**Tier-2 (identity):** `..r` as the sole shape element — one def, every rank:

```
def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)
```

**Tier-3 (name-preserving rank arithmetic, §4.5.3):** `..r` may be interleaved
with concrete **named anchors** (`tensor[..pre, seq, ..post, f32]`). A
named-axis reduction drops the named anchor and carries the surrounding spreads
through — one def reduces a named axis at any rank:

```
def reduce_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)
```

**⟹**
```
(defsig {} reduce_seq (pre post)
  (t-fn {} (t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))
           (t-tensor {} (d-rank {} pre) (d-rank {} post) (t-prim {} f32))))
```

`..r` is a variable use whose name must appear in the declaration's complete
`[...]` binder list. A spread name may not repeat within one tensor shape (a
**parse error**). A reduction names the axis it removes by the anchor's name
(`sum(x, seq)`). Multiple named axes use the variadic form
(`sum(x, head, seq)` or `count(mask, head, seq)`). Value reductions
lower to the canonical single-axis composition in spec/04 §4.5.3; `count`
lowers once with its complete named-axis vector because its result is `i64`,
not `bool`. The reduced axis must be a **named** anchor present
exactly once in the operand — a fully-literal or differently-named operand is
rejected (the Name↔Lit boundary, §4.5.3). A `def` whose signature mentions `..r`
is restricted by the §4.2 Body-Discipline check to *name-trackable* operations:
shape-identity (elementwise) ops and named-axis reductions only — never a
positional shape-rewriter like `permute`/`reshape` (meaningless at symbolic
rank).

### P4: Type Signatures

Both inline and standalone forms. All types are optional — inference fills them in.

**Inline:**
```
def add_vecs[d](x: tensor[d, f32], y: tensor[d, f32]) -> tensor[d, f32] = add(x, y)
```

When a `def` has inline type annotations and no standalone `sig`, its generated `defsig` owns each parameter type. The corresponding `fn` parameter carries `type: (t-var {} _)`, not a second copy of that type. This inference hole adds no constraint and receives the declared slot's type before the body is checked. A parameter without an annotation remains bare. The presence of the hole preserves the source annotation's presence for signature reports.

Each parameter type retains its inline binding context under §P4b, even in the generated signature. The result type retains its signature context. A standalone `sig` and any separately authored parameter annotations remain independent constraints. Anonymous lambdas and property quantifiers retain their own parameter annotations.

**Standalone (for long signatures):**
```
sig multi_head_attn:
  tensor[batch, heads, seq, d_k, f32]
  -> tensor[batch, heads, seq, d_k, f32]
  -> tensor[batch, heads, seq, d_k, f32]
  -> tensor[batch, 1, seq, seq, f32]
  -> tensor[batch, heads, seq, d_k, f32]

def multi_head_attn(q, k, v, mask) = ...
```

**⟹** `(defsig {} multi_head_attn (t-fn {} ...arg_types... ret_type))`

A `sig` may carry the same bracketed binder list a `def` carries, in the
same position — immediately after the declared name:

```text
sig arange[n, p: Int]: p -> p -> tensor[n, p]
def arange(start, stop) = ...
```

`sig` must precede its corresponding definition: either a function `def` or a
bare top-level value binding. Only a function `def` has an inline binder list.
A binder-bearing signature for a non-function value therefore stays standalone
in canonical Surf and is followed by an untyped value binding:

```text
sig empty[p]: List[p]
empty = Nil
```

Writing `empty: List[p] = Nil` instead would lose the quantifier because a typed
value binding has no binder list or declaration-binder scope.

Arrow chain reads as: arg₁ -> arg₂ -> ... -> return. Always flat in Deep (`t-fn` with last child as return type). The arrow is right-associative, so `a -> b -> c` is the curried 3-ary `a -> (b -> c)`. A function-typed argument must be parenthesized: `(a -> b) -> c` is a distinct, 1-ary type whose single argument is itself a function, and the formatter and decompiler preserve those grouping parentheses (a bare arrow in return position keeps no redundant parens).

Effect annotations are optional suffixes on either `sig` or `def`:

```text
sig predict[n]: tensor[n, f32] -> tensor[n, f32] ! { IO }
def train[n](x: tensor[n, f32]) -> tensor[n, f32] ! { IO, Resource("gpu:0") } = ...
```

Surf accepts the built-in names `Diff`, `Accum`, `IO`, `Test`, and
`Resource("device")`. `Resource("...")` is the user-handler boundary.
Randomness is not an effect: a function that draws takes a `key` parameter
(spec/05 §2.7). `IO` covers host interaction and may remain unhandled at the
program boundary. `Test` is handled by `chelis test`. `Diff` denotes a
compiler capability rather than a user-handled effect, and `Accum` is
internal-only.

Omitting all types is valid: `def f(x, y) = add(x, y)`. The compiler emits a note recommending a `sig` for module-level definitions.

#### P4b: Explicit Declaration Binders

Every type, dimension, and rank variable in a `sig`, annotated `def`, or
`@property` declaration is declared in that declaration's `[..]` binder list.
For a property the list follows the property name:

```text
@property accepts[p] forall(x: p): true
```

Names appearing
in the precision slot of a `tensor[...]` type that match the binder
list become `(t-var {} <name>)`, not `(t-prim {} <name>)`.
Names matching a primitive (`f32`, `f64`, `bf16`, `f16`, `i8`, `i16`,
`i32`, `i64`, `bool`) stay as `(t-prim {} <name>)`. Outside a sig
or def quantifier scope (e.g., in a top-level typed value binding with no
standalone `sig`), no quantifier exists, so the existing rule applies.
Within a quantified def or property body, the declaration's type binders
remain in scope for every annotation type position, including property
quantifier types, lambda parameters, expression ascriptions, block bindings,
nested ADT arguments, and tensor precision slots. One explicit declaration
binder scope applies throughout the declaration body; a listed name therefore
lowers according to its position even inside a body-local tensor type.
The retired v0.18 integer spellings `int8`, `int16`, `int32`, and `int64`
never become type variables. They are rejected with the
versioned-migration diagnostic even when listed.

The binder list is complete, not a hint. An unlisted lowercase name in
a scalar type or tensor precision position stays `(t-prim {} <name>)`
and is rejected as an unknown dtype with a nearest-active-dtype
suggestion. A single-letter symbolic dimension still lowers to `d-var`,
and `..r` still lowers to `d-rank`, but either is undeclared and rejected
unless its name appears in the list. An unlisted multi-letter tensor axis
keeps its existing concrete `d-name` meaning. There is no typo-shape or
alias heuristic and no implicit collection fallback.

An unbounded listed name may be unused; it denotes a vacuous universal
quantifier and is preserved by canonical Surf/Deep round-tripping. A listed
name with a dtype bound must occur in the declared type as §P4c
requires.

The `spec/04-type-system.md` §1.1.2 unsigned aliases (`u8`, `u16`,
`u32`, `u64`, `uint8`, `uint16`, `uint32`, `uint64`) are explicitly
excluded from binding so they reach the type-checker's §1.1.2 rejection
path with a precise diagnostic. A dtype spelling that
`spec/04-type-system.md` [04-DTYPE-1] rejects names no type variable in
any type position, so a `[..]` clause does not rebind one. The clause
overrides the case-split of §3.1, not rejected or primitive spellings.

The same identifier in a def's `[..]` clause may act as either a
dim-var or a precision tvar depending on its position inside a
`tensor[..]` type: the dim slots resolve to `d-var` and the
precision slot resolves to `t-var`. Position determines kind; the
quantifier list is unkinded.

A name in a def's `[..]` clause is also a **general type variable**
wherever it appears as a type by itself — as a bare parameter type, a
return type, or an argument/return position of a function-typed
(`->`) parameter. Such a name lowers to `(t-var {} <name>)` and is
generalized into the def's scheme, so each call site instantiates a
fresh variable that unifies against the concrete argument. The `[..]`
clause is the authoritative quantifier source and **overrides the
PascalCase-vs-snake_case case-split** (§3.1): a quantifier name that
happens to be PascalCase (e.g. `P`) is a type variable, not a rigid
ADT, because the user explicitly bound it. Threading such a name
through a function-typed parameter — e.g.

```text
def apply_resid[n, P](
  x: tensor[n, f32],
  inner_p: P,
  f: tensor[n, f32] -> P -> tensor[n, f32],
) -> tensor[n, f32] = add(x, f(x, inner_p))
```

must therefore type-check both in isolation and at every call site.
Outside an `[..]` clause the case-split still applies:
an unquantified PascalCase name is an ADT.

The case-split has a value-position mirror of this type-position
override. A **single-letter** uppercase name (`S`, `K`, `T`, `N`, `P`)
in a value-binding position — a top-level value-binding LHS, a function
or lambda parameter, or a block binder — is a value identifier, not a
constructor, because the explicit value binding makes it one
(`spec/01-nomenclature.md` §1.1, §3.2). The two overrides
are symmetric: a quantified single-letter uppercase name in a `[..]`
clause is a type variable, and a bound single-letter uppercase name in
a value position is a value. The override is single-letter only;
multi-letter PascalCase remains a type or constructor everywhere.

Example:

```text
sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]
def poly_id(x) = x
def use_f32(x: tensor[3, f32]) -> tensor[3, f32] = poly_id(x)
def use_int32(x: tensor[3, i32]) -> tensor[3, i32] = poly_id(x)
```

The sig explicitly declares `d` (a `DimVar`) and `p` (a precision
`TypeVar`). Each call site instantiates `p` with a fresh precision
variable that unifies with the call's actual precision; calling
`poly_id` with mismatched precisions across a single call (e.g.,
input `tensor[3, i32]` declared output `tensor[3, f32]`) is a type
error.

Def-level explicit quantifier:

```text
def take[batch, hidden, p](x: &tensor[batch, hidden, p]) -> &tensor[batch, hidden, p] = x
```

`batch` and `hidden` resolve to `d-var`; `p` resolves to `t-var`.
Writing `def f[a, b](x: &tensor[3, p])` (where `p` is not in
`[a, b]`) is rejected.

See `spec/04-type-system.md` §5.8 for the type-system semantics and
the `TensorPrec` representation that backs this surface rule.

#### P4c: Dtype Bounds

A binder in a `[..]` clause may declare a **dtype bound**,
written after the binder name:

```text
sig linspace[n, p: Float]: p -> p -> i64 -> tensor[n, p]
def linspace(start, stop, count) = ...

def arange_values[p: Int](current: p, stop: p, out: List[p]) -> List[p] = ...

def widen[p: {f32, f64}](x: p) -> p = ...
```

The bound is either a family name -- one of `Float`, `Int`, or
`Numeric` -- or an **explicit dtype set** written `{d1, d2, ...}`, a
brace-delimited comma-separated list of active dtype names. Both forms
restrict the binder per `spec/04-type-system.md` §5.9 [04-DTYPE-2]: a
family name to the active dtypes of that family, and a set to exactly
the dtypes it lists. Any other name in the
bound position is a syntax error, so an ADT name never becomes a
silent bound and a user type named `Float` is unaffected outside this
position. A binder with no bound keeps its existing meaning: an
unconstrained type variable, not a dtype.

A one-member set is written `{f64}`; there is no bare-dtype bound
spelling, so a bound is always a family name or a braced set. The
braces are required and carry the meaning: `p: f64` is a syntax error
rather than a bound, because a binder restricted to exactly one dtype
is a set of one and not a type ascription.

A `sig`'s `[..]` clause is complete: every `t-var`, `d-var`, and
`d-rank` name in the signature appears exactly once. Listing a
multi-letter dimension name makes it a dimension *variable* where an
unlisted one is a concrete symbolic axis. A listed name **that declares
a bound** must occur in the declared type. One binder list owns each
declaration: a standalone `sig` carries it, and a matching `def` must
not carry a second list.

A bounded binder is a type binder only. Using one in a dimension slot
or as a rank spread is an error, since a dtype bound cannot name an
extent.

The formatter prints a family bound as `name: Family` with one space
after the colon, and a set bound as `name: {d1, d2}` with one space
after the colon, no space inside the braces, one space after each
comma, and its members in `spec/04-type-system.md` §1.1 declaration
order rather than the authored order. It preserves the authored binder
order. The two differ because a binder list is ordered -- its sequence
is the declaration's, and an explicit type application reads it -- while
§5.9 makes a set bound unordered, so its members have no authored
sequence worth keeping.

### P4a: Canonical Surf Style

The parser accepts only `def f(x: T) -> U = ...`. A colon result annotation is
legacy v0.18 input and is available only through the migration command.

```text
def f(x: tensor[batch, 784, f32]) -> tensor[batch, 128, f32] = ...
```

Style rules for human-facing Surf:

- put types on parameters rather than top-level load bindings
- use symbolic dimensions for runtime-varying axes such as `batch` and `seq`
- keep fixed architecture dimensions concrete
- omit intermediate type ascriptions when inference already determines the type
- combine short tensor operations when that improves readability

Additional canonical style rules:

- use block bindings exclusively: `x = expr` inside `{ ... }` and bare top-level
  bindings such as `result = expr`
- use pipes or calls according to which makes the dataflow clearer
- break long or many-stage pipes after `=` and before every `|>` using the same
  flat-first, width-threshold approach as the Deep pretty printer

### P5: Blocks and Sequencing

Braces define binding blocks. Inside them, bindings are sequential and
newlines are the only separators. The final expression is the block's value.
That tail expression, like a binding value, is newline-bounded unless the next
line begins with one of the exact continuation tokens `|>`, `then`, or `else`,
or the break is inside `()`/`[]`/`{}`. A binding block has at
least one binding followed by exactly one tail expression. A bare non-tail
expression statement and a one-expression binding block are rejected.

```
{
  x = f(a)
  y = g(x)
  z = h(y)
  combine(x, y, z)
}
```

**⟹** All sequential Surf bindings collapse into a single Deep `let` node:
```
(let {} (bind x (app {} (var {} f) (var {} a))
              y (app {} (var {} g) (var {} x))
              z (app {} (var {} h) (var {} y)))
  (app {} (var {} combine) (var {} x) (var {} y) (var {} z)))
```

Blocks are expressions; the canonical form is:

```text
result = {
  temp = f(x)
  g(temp)
}
```

A newline is required between a binding and the next statement. Multiple
newlines are fine. Semicolons, including a trailing semicolon, are rejected.

The public Deep `block` node is represented directly, rather than confused
with a binding block, by `do { e1; e2; ... }`. It preserves child order and
evaluates left-to-right, returning the last value. `do` accepts one or more
expressions and requires semicolons between them. In parallel, `par { e1; e2;
... }` also requires semicolons but retains the concurrency semantics of
`par`; the two forms are not aliases.

There is no Surf `let ... in` expression form. Sequential bindings use blocks, and
top-level script-style bindings use bare `name = expr`.

No `where` clauses. Use blocks.

### P5a: Effect Handlers

Surf defines one `with` block form:

```text
with device("gpu:0") {
  body
}
```

A `with` handler is an expression. It takes exactly one argument in parentheses and a
brace-delimited block body.

The handler argument rules are:

- `with device(...)` requires an explicit string literal device name
- `device` is the only valid handler name

Randomness has no handler. A random primitive takes an explicit key
(`dropout(key_from_seed(42i64), x, 0.5)`; spec/05 §2.7).

### P5b: Macros

Surf has top-level macro definitions:

```text
macro linear_layer(x, w, b) = add(matmul(x, w), insert(b, 0, batch))
macro relu_ref(x) = max_elem(x, 0.0)
```

Macro invocations use the ordinary call surface: `linear_layer(x, w, b)`.

> **[02-MACRO-1]** The parameter names of a macro definition SHALL be
> distinct. A definition that names a parameter more than once SHALL be
> rejected during macro expansion, whether or not the program calls the
> macro, with a diagnostic that names the macro and the repeated parameter.

> **[02-MACRO-2]** Each call that macro expansion expands SHALL supply
> exactly one positional argument for each parameter of the macro it resolves
> to under the resolution order below. This applies to user-defined and
> standard prelude macros alike, and to a call that an earlier expansion
> produced. A call with fewer or more positional arguments SHALL be rejected
> during macro expansion, before substitution, with a diagnostic that names
> the macro, its parameter count, and the supplied argument count. Expanding a
> call SHALL NOT leave a parameter to resolve in the caller's scope, and SHALL
> NOT discard a surplus argument.

> **[02-MACRO-3]** Macro parameters are positional, so a call that macro
> expansion expands SHALL NOT carry a named argument such as `accumulator=`
> (§6.1). Such a call SHALL be rejected during macro expansion, before
> substitution, with a diagnostic that names the macro and the named
> argument. A named argument that a macro body writes on a call that does not
> resolve to a macro belongs to that call, and expansion SHALL preserve it.

> **[02-MACRO-4]** Each parameter of a macro definition SHALL occur in the
> definition's body as a reference that substitution replaces with the call's
> argument. A reference inside the scope of a body binder of the same name, such
> as a `fn` parameter, a block binding, or a pattern binder, is not such an
> occurrence. A definition with a parameter that its body never references SHALL
> be rejected during macro expansion, whether or not the program calls the
> macro, with a diagnostic that names the macro and the unused parameter.
> Expansion therefore never discards an argument: every argument of a call that
> macro expansion expands occurs in the expansion at each position where the
> body references its parameter. Where that position is an expression, macro
> expansion, name resolution, type checking, effect inference, and linearity
> checking apply to the argument as to any other expression there. Where it is
> not an expression position, such as a label in metadata, the argument is
> subject only to the rules of that position.

Macro rules:

- resolution order is lexical blockers first, then user-defined top-level macros, then
  the standard macro prelude, then ordinary function call resolution
- an ordinary top-level `def` or `sig` may not use the name of a loaded standard
  prelude macro; the declaration is rejected no later than macro expansion because its
  calls would otherwise expand as the standard macro before ordinary function resolution.
  The rule applies to the name as written, in every module: a reef package module is
  subject to it although package linking gives its declarations internal names, and
  the diagnostic names the declaration as written
- a local binding named `linear_layer` or `cross_entropy` blocks macro expansion for
  that identifier
- hygiene renames only binders introduced by the macro expansion (block-binding names, `fn`
  params, pattern binders), including typed parameters whose names require prefix
  metadata spelling in Deep; free references in the macro body remain free and resolve
  in the caller's scope
- expansion preserves an invocation's authored expression type ascription and
  block-binding type origin on the expanded value; a type ascription in the macro
  body remains a separate obligation when both apply; substituting a macro
  parameter preserves a type ascription on that parameter reference
- macro expansion runs before type checking, effect inference, linearity checking, and
  lowering

Standard prelude macros:

- `linear_layer(x, w, b)` -> `add(matmul(x, w), insert(b, 0, batch))`
- `residual(x, f)` -> `add(x, f(x))`
- `cross_entropy(logits, labels)` -> the standard `softmax` / `log` / `sum` / `mean`
  composition

### P6: Records

Braces construct records. A same-named field/variable pair must use a pun;
other values use `field: expression`. Fields remain in written order and their
value expressions evaluate left-to-right. Dot-chaining accesses fields.
Functional update is `base with { field: expression, ... }`; update fields use
the same order and pun rules.

```
lr = 0.01
opt = Adam { lr, eps: 1e-8 }         -- punning: lr: lr
rate = opt.lr                         -- field access
chain = model.layer1.weight           -- chained access
```

**⟹**
```
Adam { lr, eps: 1e-8 }        ⟹  (record {} Adam (kv {} lr (var {} lr)) (kv {} eps ...))
opt.lr                        ⟹  (access {} (var {} opt) lr)
model.layer1.weight           ⟹  (access {} (access {} (var {} model) layer1) weight)
```

Record and record-update `kv` pairs preserve written order in canonical Deep.

### P7: Pattern Matching

Match uses `=>` for arms (distinct from `->` in function types and lambdas).

```
match expr with {
  | Pattern => body
  | Pattern if guard => body
}
```

**Supported pattern forms:**

| Pattern | Example | Deep form |
|---------|---------|-----------|
| Variable | `x` | `(pat-var {} x)` |
| Wildcard | `_` | `(pat-wild {})` |
| Literal | `0`, `true`, `"hi"` | `(pat-lit {} value)` |
| Constructor | `Some(x)` | `(pat-ctor {} Some (pat-var {} x))` |
| Nested | `Some(Some(x))` | `(pat-ctor {} Some (pat-ctor {} Some (pat-var {} x)))` |
| Record | `Adam { lr, eps }` | `(pat-record {} Adam (kv {} lr (pat-var {} lr)) ...)` |
| Tuple | `(a, b, c)` | `(pat-tuple {} (pat-var {} a) ...)` |
| Unit/empty tuple | `()` | `(pat-tuple {})` |
| As-pattern | `x @ Some(_)` | `(pat-as {} x (pat-ctor {} Some (pat-wild {})))` |

**Guards:** `if` after pattern, before `=>`. Guard fills the guard slot in `(arm {} pattern guard body)`:

```
match n with {
  | x if x > 0 => positive(x)
  | x if x < 0 => negative(x)
  | _           => zero_case
}
```

A guard runs after its pattern matches, and a `false` guard passes control to
the next arm (`spec/04-type-system.md` [04-PAT-2]).

Record patterns allow punning and ignore unmentioned fields. Field order doesn't matter.

Surf has no or-patterns; write separate arms.

Exhaustiveness required. Every variant of the scrutinee's ADT must be covered. Non-exhaustive match is a compile error.

### P8: Tuples

Construction: `(a, b, c)` with commas. `(a)` is grouping, not a tuple. A
single-element tuple requires its distinguishing comma: `(a,)`.

Unit values use `()`. The unit type uses `unit`.

Access: dot-integer syntax.

```
pair = (w_new, b_new)
w = pair.0
b = pair.1
```

**⟹**
```
(a, b)   ⟹  (tuple {} a' b')
pair.0   ⟹  (tuple-get {} (var {} pair) (lit {type: (t-prim {} i32)} 0))
()       ⟹  (lit {type: (t-unit {})} ())
```

Destructuring via bindings:
```
(w, b) = train_step(w, b, x, y, lr)
```

### P9: Transforms

Transforms use call syntax in Surf but desugar to dedicated Deep tags. The parser recognizes these keywords and emits transform nodes, not `app` nodes.

| Surf | Deep | Notes |
|------|------|-------|
| `grad(f)` | `(grad {} f')` | Returns gradients only |
| `grad(f, wrt=w)` | `(grad {wrt: ...} f' idx)` | `wrt` names one parameter of `f` |
| `grad(f, wrt=(w, b))` | `(grad {wrt: ...} f' (tuple {} idx₁ idx₂))` | Multi-parameter `wrt` preserves the written order |
| `jit(f)` | `(jit {} f')` | |
| `vmap(f, axis=n)` | `(vmap {} f' n')` | Named nonzero axis |
| `vmap(f)` | `(vmap {} f' (lit {type: (t-prim {} i32)} 0))` | |
| `cast(e, bf16)` | `(cast {} e' (t-prim {} bf16))` | Second arg is a type literal (special form) |
| `cast_trunc(e, i32)` | `(cast {} e' (t-prim {} i32) trunc)` | Named truncating float-to-integer cast ([05-OP-6]) |
| `cast_saturate(e, i8)` | `(cast {} e' (t-prim {} i8) saturate)` | Named saturating cast to a signed integer ([05-OP-23]) |
| `cast_wrap(e, i8)` | `(cast {} e' (t-prim {} i8) wrap)` | Named wrapping signed-integer cast ([05-OP-24]) |
| `to_tensor(e, f64)` | `(app {} (var {} to_tensor) e' (t-prim {} f64))` | Optional second argument is a dtype; `to_tensor` is reserved |
| `realize(e)` | `(realize {} e')` | |
| `copy(e)` | `(copy {} e')` | |
| `&x` | `(borrow {} (var {} x))` | Explicit read-only borrow; usually inferred at call sites |

For a named `wrt`, desugaring resolves the target value to its immutable
callable origin and maps each written selector name to that origin's ordered
formal parameters. Direct definitions, inline lambdas, and lexical aliases
retain that identity through alias chains; rebinding or shadowing affects only
subsequent bindings. Tuple, ADT constructor payload, and record-field patterns
project the corresponding callable origin from their scrutinee, recursively
through nested patterns. A branch or aggregate join retains a callable origin
only when the exact lexical origin identity and ordered formal list agree on
every contributing path; equal display labels or equal function signatures do
not make distinct lambdas or declarations the same callable. Selector order
and duplicates are preserved. An unknown formal name, a dynamic or non-callable
target, or a target whose callable origin cannot be established is a desugaring
error; no positional fallback is permitted.

Transforms compose naturally: `jit(grad(loss_fn))` **⟹** `(jit {} (grad {} (var {} loss_fn)))`.
When `grad` targets one differentiable parameter, the result is that gradient value.
When it targets multiple parameters, the result is a flat tuple of gradients rather than
`(value, grad)` or nested tuples.

Each direct application is flat: `f(x, y)` is one Deep `app`, and the
ungrouped chained spelling `f(x)(y)` is rejected. Public Deep may nevertheless
apply a value produced by another expression. That distinct operation uses an
explicitly grouped callee: `(f(x))(y)` represents `(app {} (app {} f x) y)`,
and `(if c then f else g)(x)` applies the selected function. A transform is
self-delimiting, so `vmap(process)(xs)` likewise represents application whose
callee is the transform value rather than an ordinary chained-call alias.

`grad`, `vmap`, `jit`, `cast`, `cast_trunc`, `cast_saturate`, and `cast_wrap` require
their call-like special form;
`g = grad` is a parse error. Unary `realize` and `copy` additionally have a
bare callable form, used canonically by stages such as `x |> realize`.

The second argument of `cast`, and the optional second argument of a call to the reserved name `to_tensor` (§P10b), is a dtype: a primitive (`f32`, `bf16`, etc.) or a dtype-bounded type binder in scope, written in expression position. The second argument of a named cast, and the value of a final `accumulator=<dtype>` argument (spec/04 §5.7), are likewise dtypes. These are the only argument positions that hold a dtype, and the identifier there always names a dtype, never a value, even where a value of the same name is in scope.

### P10: Numeric Literals

Default float precision: **f32**. Default integer type: **i32**.

Canonical Surf output uses the exact spelling emitted by the literal printer.
Integers are decimal with no separators or redundant leading zeroes. Floats
are finite and use the shortest round-trippable decimal spelling for their
decoded value, with `.0` added when the shortest spelling would otherwise look
like an integer; a lowercase `e` form is used only when the shortest printer
emits it. A float-suffixed literal whose body is an integer (§P10a) is a
distinct source form rather than a spelling of the decimal-bodied literal, and
its canonical output keeps the integer body.

The normal parser additionally accepts value-preserving hexadecimal and binary
integers, underscores placed strictly between digits, and, for a float-bodied
literal, any finite decimal body that decodes to the literal's value. That
family covers equivalent exponent spellings such as `1e3` and `1.0E+3`, padded
decimals such as `1.00` and `1.10f64`, and bodies carrying digits beyond the
shortest round-trippable spelling such as `0.319381530` and
`0.99999999999980993f64`. The formatter decodes these forms and emits the
canonical token, so a transcribed reference constant is repaired by
`chelis fmt` rather than refused by the parser.

This allowance does not admit malformed separators, a non-canonical decimal
body on an integer-bodied literal, a suffix with a different type
meaning, or a token whose decoded value is non-finite.
Surf has no infinity or NaN literal. A token that decodes to a finite value
is still not a literal of a dtype at which that value is non-finite
(`spec/04-type-system.md` [04-LIT-2]): `1e40` at the `f32` default and
`70000.0f16` are rejected.

**Literal default rule (authoritative):** an unsuffixed integer literal binds
at type `i32`; an unsuffixed float literal binds at type `f32`, each under
`spec/04-type-system.md` [04-LIT-2]. The default is the **user-facing
contract** and is non-overridable except by:

1. an explicit literal suffix (P10a);
2. a dtype-stating construct that directly contains the literal (P10b): the
   declared type of the binding or function result it initializes, the
   `cast` whose operand it is, or the dtype argument of the `to_tensor` call
   it is an element of.

A literal element of a `to_tensor` argument has no default: an unsuffixed one
in a call without a dtype argument is rejected (P10b). There is no implicit
precision promotion. A bare `42` in any unannotated position binds at `i32`,
not `i64`. A bare `1.0` binds at `f32`, not `f64`. See
`spec/04-type-system.md` §5.3 for the type-system statement of this rule.

**Negative literals:** `-42` is always parsed as unary minus applied to `42`,
not as a signed literal token. A negative argument is written `f(-42)`.
Literal patterns are the exception because patterns contain no unary
expression node: `-42`, `-1.5`, and `-0.0` decode directly to a negative
`pat-lit`, including the full `i64` minimum.

### P10a: Literal Suffixes

Numeric literal tokens may carry an explicit precision suffix that binds the
literal at exactly that precision, with no inference, no widening, and no
narrowing. The closed suffix set is:

| Suffix | Bound type | Example | Notes |
|---|---|---|---|
| `f32` | `f32` | `1.0f32`, `42.0f32`, `42f32` | Float-typed |
| `f64` | `f64` | `1.0f64` | Float-typed |
| `bf16` | `bf16` | `1.0bf16` | Float-typed |
| `f16` | `f16` | `1.0f16` | Float-typed |
| `i8` | `i8` | `42i8` | Integer-typed |
| `i16` | `i16` | `42i16` | Integer-typed |
| `i32` | `i32` | `42i32` | Integer-typed |
| `i64` | `i64` | `42i64` | Integer-typed |

Default-type suffixes are semantic commitments, not syntax-safe aliases. In
particular, `cast(1.1f32, f64)` widens a value first bound at `f32`, whereas
`cast(1.1, f64)` binds the literal at `f64` under P10b and is the same value
as `1.1f64`.

A float suffix accepts either a float body or an integer body, and the two are
semantically distinct rather than spellings of one another. `42.0f32` decodes
its decimal body and binds the decoded float. `42f32` binds the exact integer
directly at the suffix width: it is
`spec/04-type-system.md` [04-LIT-1]'s suffix-bound cross-family form, an exact
Int atom marked `literal_source: integer`, finalized once at the declared width
rather than through `f64`. Both are canonical, and `chelis fmt` preserves
whichever body the author wrote rather than converting between them. Deep-to-Surf
output is governed separately: `spec/03-deep-syntax.md` §6.3.2 normalizes an
integer atom at a float primitive to the equivalent float atom, so a resugared
program prints the decimal body. §0.1's retraction law does not require
`resugar(desugar(surf))` to reproduce authored bytes.

An integer body binds at the suffix width, so its admissible range is that
width's, not the body's: a body whose exact value rounds to infinity at the
declared width is rejected as non-finite, exactly as `1e400` is. `65504f16`
binds; `65520f16` does not. A decimal body is held to the same width
(`spec/04-type-system.md` [04-LIT-2]): `65504.0f16` binds; `65520.0f16` does
not.

No radix body carries a float suffix (`spec/04-type-system.md` §5.5). That
holds for a hexadecimal body even though every float suffix is spelled in hex
digits: such a token is rejected, not read as a longer hexadecimal integer.

Integer-typed suffixes attach to integer literal tokens only; `1.0i8` is a
parse error.

**Adjacency rule.** A suffix is part of the literal token only if it
**immediately** follows the digit sequence with no intervening whitespace,
comment, or other character. `1.0 f32` (with whitespace) is two tokens (a
float literal followed by an identifier-position token); the literal is
unsuffixed, and its dtype follows §P10.

Hexadecimal and binary integers may carry an integer suffix; the decoded value
is printed as a canonical decimal token. A radix integer may not carry a float
suffix. Canonical decimal float literals carry float suffixes without ambiguity
(`1.0f32`).

**Rejected suffixes.**

- `f8e4m3` is reserved and rejected per `spec/04-type-system.md` §1.1.1; the suffix
  `f8e4m3` is rejected at lex time with a diagnostic pointing at §1.1.1.
- Unsigned suffixes (`u8`, `u16`, `u32`, `u64`) are rejected per
  `spec/04-type-system.md` §1.1.2; they are rejected at lex time with a
  diagnostic pointing at §1.1.2.
- An unrecognized identifier sequence directly adjacent to a numeric literal
  (e.g. `1.0xyz`) is a parse error rather than a silently-split
  literal-then-identifier pair. The diagnostic suggests adding whitespace
  if the adjacency was unintentional.

The suffix grammar is identical in Deep canonical form (`spec/03-deep-syntax.md`
§6.4); the Surf and Deep lexers parse the same token shape.

Suffixes are expression-literal syntax. A Deep `pat-lit` contains only its raw
value and has no precision slot, so Surf literal patterns are unsuffixed and a
suffixed literal pattern is rejected.

The four integer names `i8`, `i16`, `i32`, and `i64` are the only integer
primitive spellings in type positions and canonical output. Canonical Deep
uses the same names. The v0.18 spellings `int8`, `int16`, `int32`, and `int64`
are migration input only: `chelis migrate surf --from 0.18` rewrites them
without making them valid at normal compiler ingress. Neither the canonical
names nor the retired names may be rebound as type variables under
`spec/04-type-system.md` §5.8.1.

### P10b: Tensor Literals and Dtype-Stating Constructs

A bracket literal `[e1, e2, ...]` is a `List`: it desugars to the `Cons`/`Nil`
chain of its elements (`spec/03-deep-syntax.md` §6). Neither the spelling of
its elements nor the position it stands in selects another kind. A tensor is
written in one of two ways:

- as a `to_tensor` call, such as `to_tensor([1.0, 2.0, 3.0], f32)`, in any
  position. Its first argument is an ordinary `List`, and its optional second
  argument is a dtype (§P9);
- as a **tensor literal**: a bare bracket literal whose own declaration states
  a tensor type. That is the right-hand side of a binding whose declared type,
  by inline annotation or standalone `sig`, is a tensor type, or the body of a
  `def` whose declared result type is a tensor type. A `def`'s inline result
  type decides; a `def` without one takes the result type of its standalone
  `sig`. The desugarer emits the `to_tensor` call.

A numeric literal takes its dtype from its suffix (§P10a) or else from a
**dtype-stating construct** that directly contains it, when its kind admits
the stated dtype, and never from a callee's signature or from anything
further away. The closed set is exactly
(`spec/04-type-system.md` §5.6 states the full rule):

- **Declaration**: the declared type of the binding or function result whose
  initializer is the literal (`x: f64 = 1.1`, `def f() -> i8 = -128`), or,
  for a tensor literal, the declared element dtype, which every literal
  element takes (`xs: tensor[2, f64] = [1.1, 2.2]`);
- **Cast**: the target of the `cast` whose first argument is the literal
  (`cast(1.1, f64)`); the named casts state no dtype;
- **Dtype argument**: the second argument of the `to_tensor` call whose
  bracket-literal first argument has the literal as an element
  (`to_tensor([1.1, 2.2], f64)`).

The stated dtype may be a primitive or a dtype-bounded type binder; for a
binder the literal binds at each admissible instantiation
(`spec/04-type-system.md` §5.6). An element is an item of the bracket literal,
recursively through nested bracket literals, and grouping parentheses are
transparent. A unary negation folds into the literal it negates, so
`cast(-1.1, f64)` binds `-1.1` at `f64` and `x: i8 = -128` binds the `i8`
minimum. An explicit `neg(1.1)` call, like any other expression between the
construct and the literal, leaves the literal an ordinary operand that keeps
its suffix or default. The rule is judged on the source text after pipe
normalization (§0.2); a literal written as a macro argument is governed by the
constructs around the macro call.

`cast(1.1, f64)` with a primitive target desugars to exactly the Deep of
`1.1f64`, `(lit {type: (t-prim {} f64)} 1.1)`, with no `cast` node; it does not
narrow to the `f32` default and then widen. A cast to a binder keeps its
`cast` node, because at an integer member the cast converts a float literal.
Suffixed literals keep their suffix binding (§P10a; `cast(1.1f32, f64)` widens
the `f32` value), and a float literal under an integer target, or any
numeric literal under `bool`, keeps its default and then uses the checked
target-finalization rule: an integral value casts exactly, a fractional value
under an integer target traps `domain`, and `cast(1, bool)` is `true`. See
`spec/04-type-system.md` §5.2 and [04-NUM-14] for the full statement.

**Tensor elements state their dtype.** A literal element of a `to_tensor`
argument has no default: `to_tensor([1.1, 2.2])` is rejected, and the
diagnostic names the dtype-argument spelling. A tensor literal's elements
always take the declaration's dtype. A callee's declared parameter type and a
`cast` never make a bracket literal a tensor and state no dtype for its
elements: a bare bracket literal passed as an argument or cast stays a `List`
whose literals keep their own suffix or default.

**Mixed suffixes.** A suffixed element of a tensor literal, or under a dtype
argument, is well formed only if its suffix matches the stated dtype.
`[1.0, 2.0f64, 3.0]` declared `tensor[3, f32]` is a type error: the
f64-suffixed literal at index 1 has an explicit dtype that disagrees with the
stated `f32` element type. The diagnostic identifies the offending index and
suggests removing the suffix.

**Reserved names.** `to_tensor` may not be bound by any declaration,
parameter, binding, pattern, or import (`spec/04-type-system.md` §8.6), so a
`to_tensor` call always denotes the intrinsic conversion. `cast` and the named
casts are keywords (§1).

### P11: Strings

Double-quoted: `"hello world"`. Canonical output uses the named escapes `\"`,
`\\`, `\n`, `\t`, `\r`, and `\0`; a control character without a named escape
uses `\u{h}` with its minimal lowercase hexadecimal scalar value (for example,
U+0008 is `\u{8}` and U+007F is `\u{7f}`). The parser accepts any nonempty
one-to-six-digit hexadecimal `\u{...}` spelling of a valid Unicode scalar,
including padded/uppercase forms and aliases for printable or named-escape
characters. The formatter emits the canonical named escape or printable
character. Raw control characters, invalid scalars, multiline strings, and
interpolation are rejected.

**⟹** `(lit {type: (t-prim {} string)} "hello world")`

### P12: Whitespace and Line Continuation

Within a single expression, newlines are NOT significant — they are whitespace, and an expression can break across lines freely:

```
x
  |> transform_a
  |> transform_b
```

The one qualification is at separator boundaries in a **block sequencing context**: block bindings, block tails, `do` and `par` items, and property option values. There a top-level newline acts as a `Sep` and ends the expression being parsed unless the break is inside `()`/`[]`/`{}` or the next line begins with one of the exact continuation tokens below. That is what makes a bare non-tail statement a rejected juxtaposition rather than a silent application.

In those contexts the continuation set is exactly `|>`, `then`, and `else`. Each selected token is safe because it cannot head an expression: `|>` is an infix pipeline stage, and an `if` is not a legal expression without both `then` and `else`, so a newline before one is unambiguous. That safety property is necessary but does not itself define membership. Other infix tokens such as `+`, `*`, `==`, `&&`, and `||` are not selected and remain separators when they lead the next physical line. A token that CAN head an expression -- `with`, `match`, an identifier -- is likewise not a continuation there: after a newline it would be genuinely ambiguous, and admitting it would reintroduce the silent juxtaposition this boundary exists to reject. The leading-`|>` continuation above is one instance of the exact rule, not a special case.

**A declaration body and a property predicate bound differently, and the closed set above does not govern them.** A declaration body runs to the next token that begins a declaration (or to end of input); a property predicate runs to the next `with`, the next declaration start, or end of input. Every other token continues the expression across a top-level newline, so a newline-led `with` continues a declaration body:

```chelis
def update(p: Point) -> Point = p
  with { x: 1.0 }
```

Those two contexts are therefore permissive where a block sequencing context is closed. They never exhibited the missing-continuation defect the closed set fixes, because a leading `then` or `else` does not begin a declaration and so already continued.

The parser accepts one trailing comma or semicolon in a nonempty delimited
family, and the formatter removes it. This includes arguments, parameters,
type/dimension arguments, tuple/list/record fields, imports/exports, transform
options, effects, variant payloads, and `par`/`do` sequences. The comma in a
singleton tuple or tuple pattern, `(x,)`, is grammar-significant and remains in
canonical output. Ordinary binding blocks still reject semicolon separators.

### P13: Modulo

`%` confirmed at BP 7 (with `*`, `/`). Works on integer types only. **⟹** `(app {} (var {} mod) a' b')`

### P14: Where Clauses

Surf has no general `where` clause. The `where` keyword appears only in the
property-declaration production in §5.1.

### P15: Type Aliases

Distinguished from ADTs by absence of `|`.

```
type Weights = tensor[hidden, hidden, f32]
type Pair[a, b] = (a, b)
type Matrix[p, rows] = tensor[rows, p]
```

**⟹** `(typealias {} Weights () (t-tensor {} ...))` / `(typealias {} Pair (a b) (t-tuple {} ...))`

The optional parameter list is the exact binder scope for an alias body (and
the same rule applies to ADT field types). A listed name is emitted according
to its position: `p` in the precision slot becomes `(t-var {} p)`, while
`rows` in a dimension slot becomes `(d-var {} rows)`. An unlisted dimension
identifier is a concrete symbolic dimension and becomes `d-name`, even when it
is a single lowercase letter: `type Weights = tensor[n, f32]` therefore emits
`(d-name {} n)`, not an implicit dimension binder. Signatures retain their
separate explicit binder-list rule from P4/§5.8.

Parser disambiguation: after `type Name =`, if next non-whitespace is `|`, it's an ADT. Otherwise alias.

Aliases are transparent and expand during desugaring; aliases cannot be opaque.

### P16: Opaque Types

`@opaque` immediately before an ADT `type` declaration marks the type
opaque: constructible and inspectable only inside its defining module,
enforced by the type checker (`spec/04-type-system.md` §2.5).

```
module Stats.Prob
@opaque
type Probability = | Probability { value: f32 }
```

**⟹** `(deftype {opaque: true} Probability () (variant {} Probability (field {} value (t-prim {} f32))))`
inside `(module {} stats.prob ...)`.

Constraints: the annotated declaration must be an ADT (`@opaque` on a
type alias is a parse error), and the declaration must sit inside a
named module (`@opaque` at top level is a check-time declaration
error — top-level code has no module identity to enforce against).

#### P16a: Declared Invariants

An `@opaque` type may carry one **declared invariant**: a boolean
predicate over a single binder of the representation, written in Surf
between `@opaque` and `type` (`spec/design/opaque_invariants_rfc.md`
D-SYNTAX). The invariant is recorded as Deep metadata; the everyday
type checker never evaluates it (`spec/04-type-system.md` §2.5.1).

```
module Stats.Prob
@opaque
@invariant(p) (p.value >= 0.0) && (p.value <= 1.0)
type Probability = | Probability { value: f32 }
```

**⟹** inside `(module {} stats.prob ...)`:

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

Constraints (parse-time): `@invariant(<binder>) <expr>` must appear
after `@opaque` and before `type`; exactly one binder; exactly one
invariant; `@invariant` without `@opaque` is an error (assumption
injection is unsound for a forgeable type). Surf boolean conjunction is
`&&` (§2), which desugars to `(app {} (var {} and) ...)`. The predicate
grammar, value class, free-variable scoping, and amenability recording
are well-formedness checks (`spec/04-type-system.md` §2.5.1).

---

## 4. Formal Grammar (PEG)

```peg
# ═══════════════════════════════════════════════════
#  PROGRAM STRUCTURE
# ═══════════════════════════════════════════════════

Program       <- S ModuleDecl? S Decl* EOF
ModuleDecl    <- 'module' S ModulePath

Decl          <- ImportDecl / ExportDecl / DimDecl
               / OpaqueTypeDecl / TypeDecl / TypeAlias / SigDecl
               / PropertyDecl / FunDecl / MacroDecl / BindingDecl

# ═══════════════════════════════════════════════════
#  MODULE, IMPORT, EXPORT
# ═══════════════════════════════════════════════════

ModulePath    <- TypeIdent ('.' TypeIdent)*

ImportDecl    <- 'import' S ModulePath S ImportSpec?
ImportSpec    <- '(' S ImportNames S ')'
ImportNames   <- '..'
               / IdentOrType (S ',' S IdentOrType)* (S ',')?
IdentOrType   <- TypeIdent / Ident

ExportDecl    <- 'export' S '(' S IdentOrType
                  (S ',' S IdentOrType)* (S ',')? S ')'

# ═══════════════════════════════════════════════════
#  DIMENSIONS
# ═══════════════════════════════════════════════════

DimDecl       <- 'dim' S Ident (S ',' S Ident)*

# ═══════════════════════════════════════════════════
#  TYPE DECLARATIONS
# ═══════════════════════════════════════════════════

OpaqueTypeDecl <- '@opaque' S InvariantDecl? S TypeDecl
InvariantDecl  <- '@invariant' S '(' S Ident (S ',')? S ')' S Expr
TypeDecl      <- 'type' S TypeIdent TypeParams? S '='
                  S '|' S Variant (S '|' S Variant)*
TypeAlias     <- 'type' S TypeIdent TypeParams? S '='
                  S !('|') TypeExpr
TypeParams    <- '[' S Ident (S ',' S Ident)* (S ',')? S ']'

Variant       <- TypeIdent RecordFields?
               / TypeIdent TupleFields?
RecordFields  <- '{' S FieldDecl (S ',' S FieldDecl)* (S ',')? S '}'
TupleFields   <- '(' S TypeExpr (S ',' S TypeExpr)* (S ',')? S ')'
FieldDecl     <- Ident S ':' S TypeExpr

# ═══════════════════════════════════════════════════
#  TYPE SIGNATURES
# ═══════════════════════════════════════════════════

SigDecl       <- 'sig' S Ident TypeBinders? S ':' S TypeExpr EffectClause?

# ═══════════════════════════════════════════════════
#  PROPERTY DECLARATIONS
# ═══════════════════════════════════════════════════

PropertyDecl  <- '@property' S Ident TypeBinders? S 'forall' S Params
                 (S 'where' S Expr (S ',' S Expr)* (S ',')?)?
                 S ':' S Expr PropertyOption*
PropertyOption <- S 'with' S ('tolerance' / 'seed' / 'samples') S '=' S Expr
                / S 'with' S 'contract' S '=' S StringLit

The canonical property-option order is `tolerance`, `seed`, `samples`, then
every `contract`. Repeatable contracts retain their authored relative order.
The normal parser accepts another option order as a syntax-safe alias, and the
formatter rewrites it to this one order.

A property's optional `TypeBinders` is the declaration's complete explicit
binder list under §P4b/§P4c. It scopes every quantifier type, `where`
precondition, predicate body, and expression-valued property option. Duplicate
or forbidden binder names and undeclared type, dimension, or rank variables
reject exactly as they do for `def`; dtype bounds use the same
representation and validity rules. Omitting the list preserves the existing
monomorphic property spelling.

The comma and colon delimiters bound each property precondition. A binary
precondition therefore omits the redundant outer grouping pair used by the
general expression printer: `where x <= 1:` is canonical. The parser rejects
`where (x <= 1):`. Parentheses that group an operand remain meaningful and
accepted, as in `where (x + 1) <= y:`.

Property contract options are proof dependencies, not labels. Before using a
contract, the prover must bind every abstracted call to a linker-produced
declaration from a resolved dependency package; author-written names and local
declarations are not trusted, including names that imitate the linker's
internal spelling.

`std.quantile.monotonicity` is the source-visible contract for
`Nautilus.Stats.quantile_vec`. Its Tier-B lane accepts exactly the linker's
`pkg__nautilus__Nautilus__Stats__quantile_vec` declaration with the declared
`(&tensor[n, f32], f32) -> f32` surface. The tensor argument remains a compiler
AST identity and is never reconstructed from source or lowered as a scalar.
For two calls over the same compiler-bound dataset, the prover may introduce
fresh results and assume `p <= q => quantile(xs, p) <= quantile(xs, q)`.
Different datasets are not coupled. A requested quantile contract with no such
trusted call or no same-dataset call pair is unsupported, not a successful
proof. The source bridges for `std.quantile.range` and
`std.quantile.boundary` are not part of this lane and remain unsupported.

# ═══════════════════════════════════════════════════
#  FUNCTION DEFINITIONS
# ═══════════════════════════════════════════════════

FunDecl       <- 'def' S Ident TypeBinders? Params
                  ReturnType? EffectClause? S '=' S Expr

# The declaration binder list. It is unkinded (§P4b): a listed name
# resolves to a dimension variable, a precision type variable, or a
# general type variable according to its position. A bound restricts
# the binder to one dtype family (§P4b, spec/04-type-system.md §5.9);
# `TypeParams` on a `type` declaration has no bound production.
TypeBinders   <- '[' S TypeBinder (S ',' S TypeBinder)* (S ',')? S ']'
TypeBinder    <- Ident (S ':' S DtypeFamily)?
DtypeFamily   <- 'Float' / 'Int' / 'Numeric'
Params        <- '(' S (Param (S ',' S Param)* (S ',')?)? S ')'
Param         <- Ident (S ':' S TypeExpr)?
ReturnType    <- S '->' S TypeExpr
EffectClause  <- S '!' S '{' S (EffectExpr (S ',' S EffectExpr)* (S ',')?)? S '}'
EffectExpr    <- 'Diff' / 'Accum' / 'IO' / 'Test'
               / 'Resource' S '(' S StringLit (S ',')? S ')'

MacroDecl     <- 'macro' S Ident MacroParams S '=' S Expr
MacroParams   <- '(' S (Ident (S ',' S Ident)* (S ',')?)? S ')'
BindingDecl   <- ValueIdent (S ':' S TypeExpr)? S '=' S Expr
ValueIdent    <- Ident / [A-Z] ![a-zA-Z0-9_]

# ═══════════════════════════════════════════════════
#  TYPE EXPRESSIONS
# ═══════════════════════════════════════════════════

TypeExpr      <- TypeAtom (S '->' S TypeExpr)?

TypeAtom      <- 'tensor' '[' S DimList S ',' S PrecType (S ',')? S ']'
               / PrecType
               / 'unit'
               / '&' S TypeAtom
               / '(' S TypeExpr S ',' S ')'
               / '(' S TypeExpr S ',' S TypeExpr
                  (S ',' S TypeExpr)* (S ',')? S ')'
               / '(' S TypeExpr S ')'
               / TypeName TypeArgs?

TypeArgs      <- '[' S TypeArg (S ',' S TypeArg)* (S ',')? S ']'

# An IntLit type-application argument is the concrete dimension
# instantiation of a dimension-parameterized ADT (`Frame[2]` whose
# parameter reaches a tensor dimension slot). Bare type positions
# have no integer production. The checker rejects an argument whose
# kind or extent does not match the corresponding type parameter.
TypeArg       <- TypeExpr / IntLit

# Bare or module-qualified type name (`Mode`, `Demo.Dropout.Mode`).
TypeName      <- TypeIdent ('.' TypeIdent)*

PrecType      <- 'f32' / 'f64' / 'bf16' / 'f16'
               / 'i8' / 'i16' / 'i32' / 'i64'
               / 'bool' / 'string' / 'key'
               # The reserved names of spec/04-type-system.md §1.1.1 are
               # rejected at check time:
               # f8e4m3, f8e5m2, uint8/uint16/uint32/uint64, int4/uint4,
               # complex64/complex128, decimal128/decimal256. The short
               # unsigned spellings u8/u16/u32/u64 are not reserved at all.

DimList       <- DimExpr (S ',' S DimExpr)*
DimExpr       <- IntLit / Ident / '*' / '..' Ident

# ═══════════════════════════════════════════════════
#  EXPRESSIONS (Pratt parser)
# ═══════════════════════════════════════════════════

Expr          <- MatchExpr / IfExpr / FnExpr / WithHandler
               / PipeExpr

LetPattern    <- '(' S ')' / '(' S Ident S ',' S ')'
               / '(' S Ident S ',' S Ident (S ',' S Ident)* (S ',')? S ')'
               / Ident (S ':' S TypeExpr)?

MatchExpr     <- 'match' S Expr S 'with' S '{' S MatchArms S '}'
MatchArms     <- MatchArm (S MatchArm)*
MatchArm      <- '|' S Pattern Guard? S '=>' S Expr
Guard         <- S 'if' S Expr

IfExpr        <- 'if' S Expr S 'then' S Expr S 'else' S Expr
FnExpr        <- 'fn' S Params S '->' S Expr

# ── Operator expressions ──

# The [02-PIPE-2] grouping guard applies to each expression after parsing.
# Unparenthesized mixed operators or open-ended forms cannot contain '|>'.
PipeExpr      <- UpdateExpr (S '|>' S (CastPipeStage / UpdateExpr))*
CastPipeStage <- ('cast' / 'cast_trunc' / 'cast_saturate' / 'cast_wrap') S '(' S Ident S ')'
UpdateExpr    <- OrExpr (S 'with' S UpdateRecordBody)?
OrExpr        <- AndExpr (S '||' S AndExpr)*
AndExpr       <- CmpExpr (S '&&' S CmpExpr)*
CmpExpr       <- AddExpr (S CmpOp S AddExpr)?
AddExpr       <- MulExpr (S ('+' / '-') S MulExpr)*
MulExpr       <- UnaryExpr (S ('*' / '/' / '%') S UnaryExpr)*
UnaryExpr     <- ('-' / '!' / '&') S UnaryExpr / AnnotExpr
AnnotExpr     <- AccessExpr (S ':' S TypeExpr)?

# ── Postfix ──

AccessExpr    <- AppExpr AccessStep*
AccessStep    <- '.' IntLit                              # tuple index
               / '.' (Ident / TypeIdent) CallArgs?        # field / module path, optionally applied
CallArgs      <- '(' S ((Expr (S ',' S Expr)* (S ',' S AccumArg)? / AccumArg)
                  (S ',')?)? S ')'
AccumArg      <- 'accumulator' S '=' S PrecType  # spec/04 §5.7; `accumulator`
                                                  # stays an ordinary identifier
AppExpr       <- AtomExpr CallArgs?
               / TransformExpr CallArgs?
# A call whose callee is the reserved identifier 'to_tensor' parses its
# arguments as ordinary CallArgs; desugaring requires an optional second
# argument to be an identifier naming a dtype, a primitive or a
# dtype-bounded binder in scope, never a value (§P9, §P10b).

# ── Atoms ──

AtomExpr      <- '(' S ')'
               / '(' S Expr S ',' S ')'
               / '(' S Expr S ',' S Expr (S ',' S Expr)* (S ',')? S ')'
               / '(' S Expr S ')'
               / BlockExpr
               / DoExpr
               / ParExpr
               / WithHandler
               / QuoteExpr
               / ListExpr
               / RecordExpr
               / Literal
               / Ident
               / TypeIdent

BlockExpr     <- '{' S BlockBody S '}'
BlockBody     <- (BlockBinding Newline)+ Expr
BlockBinding  <- LetPattern S '=' S Expr
DoExpr        <- 'do' S '{' S Expr (S ';' S Expr)* (S ';')? S '}'
ParExpr       <- 'par' S '{' S Expr (S ';' S Expr)* (S ';')? S '}'
WithHandler   <- 'with' S 'device' S '(' S Expr (S ',')? S ')'
                  S HandlerBlock
HandlerBlock  <- '{' S (Expr / BlockBody) S '}'
# The tail Expr, like a BlockBinding value, is Sep-bounded: a top-level
# newline ends it unless the next line begins with one of the exact
# continuation tokens ('|>', 'then', 'else'), or the break is
# inside ()/[]/{}. There is exactly one tail (no `Expr (Sep Expr)*`), so a
# second top-level expression is a bare non-tail statement and is rejected
# — bind it with `_ = <expr>` or move it to tail position.

TransformExpr <- TransformKw S '(' S Expr
                  (S ',' S TransformArg)? (S ',')? S ')'
               / 'realize' / 'copy'
TransformKw   <- 'grad' / 'vmap' / 'jit' / 'realize'
               / 'cast' / 'cast_trunc' / 'cast_saturate' / 'cast_wrap' / 'copy'
TransformArg  <- PrecType / ('wrt' / 'axis') S '=' S Expr

RecordExpr    <- CtorName S RecordBody
RecordBody    <- '{' S (RecordField
                  (S ',' S RecordField)* (S ',')?)? S '}'
UpdateRecordBody <- '{' S RecordField
                      (S ',' S RecordField)* (S ',')? S '}'
RecordField   <- Ident S ':' S Expr / Ident

QuoteExpr     <- ('quote' / 'unquote' / 'splice') S '(' S Expr (S ',')? S ')'
ListExpr      <- '[' S (Expr (S ',' S Expr)* (S ',')?)? S ']'

# ═══════════════════════════════════════════════════
#  PATTERNS
# ═══════════════════════════════════════════════════

Pattern       <- Ident S '@' S Pattern / PatAtom

PatAtom       <- '(' S ')'
               / '(' S Pattern S ',' S ')'
               / '(' S Pattern S ',' S Pattern (S ',' S Pattern)* (S ',')? S ')'
               / '(' S Pattern S ')'
               / CtorName S '{' S (RecordPatField
                  (S ',' S RecordPatField)* (S ',')?)? S '}'
               / CtorName S '(' S Pattern
                  (S ',' S Pattern)* (S ',')? S ')'
               / CtorName
               / PatLiteral
               / '_'
               / Ident

PatLiteral    <- '-'? S (BareIntLit / BareFloatLit) / StringLit / BoolLit

# Bare or module-qualified constructor head (`Train`, `Demo.Dropout.Train`).
CtorName       <- TypeIdent ('.' TypeIdent)*

RecordPatField <- Ident S ':' S Pattern / Ident

# ═══════════════════════════════════════════════════
#  OPERATORS (for lookahead)
# ═══════════════════════════════════════════════════

CmpOp         <- '==' / '!=' / '<=' / '>=' / '<' / '>'
InfixOp       <- '|>' / '||' / '&&' / CmpOp
               / '+' / '-' / '*' / '/' / '%'

# ═══════════════════════════════════════════════════
#  LITERALS
# ═══════════════════════════════════════════════════

Literal       <- FloatLit / IntLit / BoolLit / StringLit

FloatLit      <- (DecimalFloat / ExponentFloat) FloatSuffix?
               / IntFloatBody FloatSuffix
BareFloatLit  <- DecimalFloat / ExponentFloat
DecimalFloat  <- DecDigitSeq '.' DecDigitSeq
ExponentFloat <- DecDigitSeq ('.' DecDigitSeq)? [eE] [+-]? DecDigitSeq
IntFloatBody  <- DecimalInt
IntLit        <- IntBody IntSuffix?
BareIntLit    <- IntBody
IntBody       <- HexInt / BinInt / DecimalInt
DecimalInt    <- '0' / [1-9] ('_'? [0-9])*
DecDigitSeq   <- [0-9] ('_'? [0-9])*
HexInt        <- '0' [xX] [0-9a-fA-F] ('_'? [0-9a-fA-F])*
BinInt        <- '0' [bB] [01] ('_'? [01])*
FloatSuffix   <- 'f32' / 'f64' / 'bf16' / 'f16'
IntSuffix     <- 'i8' / 'i16' / 'i32' / 'i64'
# After lexical recognition, a DECIMAL integer body MUST equal the canonical
# literal spelling after digit separators are removed; that is what rejects a
# redundant leading zero such as `007` or `007f64`. HexInt and BinInt are exempt
# and MUST instead decode to the same finite typed value the canonical printer
# emits. A float body carries neither constraint: every finite decimal body
# decodes to the value its canonical spelling round-trips to, so exponent forms,
# padded decimals, and bodies with digits past the shortest spelling are all
# accepted and normalized by the printer. FloatLit admits no radix body:
# IntFloatBody is decimal only, because an integer radix form carries no float
# suffix (spec/04-type-system.md §5.5). An IntFloatBody whose exact value rounds
# to infinity at its FloatSuffix width is rejected as non-finite (P10a).
# Suffix must immediately follow the digit sequence (no whitespace, no comment).
# Closed sets: any other identifier sequence directly adjacent to a numeric
# literal (e.g. `1.0xyz`, `42u8`, `1.0f8e4m3`) is a parse error per P10a.
BoolLit       <- 'true' / 'false'
StringLit     <- '"' StringChar* '"'
StringChar    <- '\\' [nrt0"\\]
               / UnicodeControlEscape
               / OrdinaryStringChar
UnicodeControlEscape <- '\\u{' [0-9a-fA-F]+ '}'
OrdinaryStringChar <- !('"' / '\\' / RawControl) .
RawControl    <- [\u0000-\u001f\u007f-\u009f]
# UnicodeControlEscape accepts one to six ASCII hexadecimal digits in either
# case and MUST encode a valid Unicode scalar. The printer chooses a printable
# character, named escape, or minimal lowercase control escape.

# ═══════════════════════════════════════════════════
#  IDENTIFIERS
# ═══════════════════════════════════════════════════

Ident         <- !Keyword [a-z_] [a-zA-Z0-9_]*
TypeIdent     <- !Keyword [A-Z] [a-zA-Z0-9]*

Keyword       <- ('def' / 'sig' / 'type' / 'dim'
               / 'macro'
               / 'match' / 'with' / 'fn' / 'module' / 'import'
               / 'export' / 'if' / 'then' / 'else' / 'grad'
               / 'vmap' / 'jit' / 'cast' / 'cast_trunc' / 'cast_saturate'
               / 'cast_wrap' / 'realize' / 'copy'
               / 'par' / 'true' / 'false' / 'tensor'
               / 'do' / 'quote' / 'unquote' / 'splice'
               / 'effect' / 'handler' / 'perform' / 'resume' / 'borrow'
               ) ![a-zA-Z0-9_]

# ═══════════════════════════════════════════════════
#  WHITESPACE & COMMENTS
# ═══════════════════════════════════════════════════

S             <- ([ \t\n\r] / Comment)*
Newline       <- '\n' / '\r\n' / '\r'
Comment       <- LineComment / BlockComment
LineComment   <- '--' (!Newline .)*
BlockComment  <- '{-' (BlockComment / !'-}' .)* '-}'
EOF           <- !.
```

---

## 5. Complete Desugaring Reference

Primed names (`a'`) denote recursive desugaring of subexpressions. All operators desugar to derived built-in function calls via `(app {} (var {} name) ...)`. The compiler lowers derived built-ins to RISC primitives during IR construction — the desugarer does NOT decompose them further.

### 5.1 Module Structure

```
module Foo.Bar                    ⟹  (module {} foo.bar <decls...>)
import Foo.Bar (baz, qux)        ⟹  (import {} foo.bar (baz qux))
import Foo.Bar (..)              ⟹  (import-all {} foo.bar)
import Foo.Bar                   ⟹  (import {} foo.bar ())
export (f, g, MyType)            ⟹  (export {} f g MyType)
dim batch, seq                   ⟹  (defdim {} batch) (defdim {} seq)
```

### 5.2 Declarations

```
sig f: f32 -> f32 -> f32
⟹  (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))

sig empty[p]: List[p]
empty = Nil
⟹  (defsig {} empty (p) (t-adt {} List (t-var {} p)))
    (def {} empty (var {} Nil))

def f(x: f32, y: f32) -> f32 = add(x, y)
⟹  (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
    (def {} f (fn {} (params {} (x {type: (t-var {} _)})
                           (y {type: (t-var {} _)}))
                   (app {} (var {} add) (var {} x) (var {} y))))

def f(x, y) = add(x, y)
⟹  (def {} f (fn {} (params {} x y) (app {} (var {} add) (var {} x) (var {} y))))

type Option[a] = | None | Some { value: a }
⟹  (deftype {} Option (a)
      (variant {} None)
      (variant {} Some (field {} value (t-var {} a))))

module Stats.Prob
@opaque
type Probability = | Probability { value: f32 }
⟹  (module {} stats.prob
      (deftype {opaque: true} Probability ()
        (variant {} Probability (field {} value (t-prim {} f32)))))

module Stats.Prob
@opaque
@invariant(p) (p.value >= 0.0) && (p.value <= 1.0)
type Probability = | Probability { value: f32 }
⟹  (module {} stats.prob
      (deftype {opaque: true,
                invariant: (fn {} (params {} p)
                  (app {} (var {} and)
                    (app {} (var {} gte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 0.0))
                    (app {} (var {} lte) (access {} (var {} p) value) (lit {type: (t-prim {} f32)} 1.0)))),
                invariant_amenability: "linear"}
        Probability ()
        (variant {} Probability (field {} value (t-prim {} f32)))))

type Weights = tensor[h, h, f32]
⟹  (typealias {} Weights () (t-tensor {} (d-name {} h) (d-name {} h) (t-prim {} f32)))
```

### 5.3 Expressions

```
-- Variables, literals
x                                 ⟹  (var {} x)
42                                ⟹  (lit {type: (t-prim {} i32)} 42)
3.14                              ⟹  (lit {type: (t-prim {} f32)} 3.14)
true                              ⟹  (lit {type: (t-prim {} bool)} true)
"hello"                           ⟹  (lit {type: (t-prim {} string)} "hello")
()                                ⟹  (lit {type: (t-unit {})} ())

-- Application
f(x, y)                           ⟹  (app {} (var {} f) x' y')
f(x, y, accumulator=p)            ⟹  (app {accumulator: (t-prim {} p)} (var {} f) x' y')

-- Arithmetic (all via derived built-ins)
a + b                             ⟹  (app {} (var {} add) a' b')
a - b                             ⟹  (app {} (var {} sub) a' b')
a * b                             ⟹  (app {} (var {} mul) a' b')
a / b                             ⟹  (app {} (var {} div) a' b')
a % b                             ⟹  (app {} (var {} mod) a' b')
-a                                ⟹  (app {} (var {} neg) a')

-- Comparison
a == b                            ⟹  (app {} (var {} eq) a' b')
a != b                            ⟹  (app {} (var {} neq) a' b')
a < b                             ⟹  (app {} (var {} cmplt) a' b')
a > b                             ⟹  (app {} (var {} gt) a' b')
a <= b                            ⟹  (app {} (var {} lte) a' b')
a >= b                            ⟹  (app {} (var {} gte) a' b')

-- Logical
a && b                            ⟹  (app {} (var {} and) a' b')
a || b                            ⟹  (app {} (var {} or) a' b')
!a                                ⟹  (app {} (var {} not) a')

-- Pipe
x |> f |> g                       ⟹  (app {} (var {} g) (app {} (var {} f) x'))

-- Control flow
if c then a else b                ⟹  (if {} c' a' b')
fn (x, y) -> body                 ⟹  (fn {} (params {} x y) body')

-- Block (sequential bindings)
{                                 ⟹  (let {} (bind x e1' y e2') body')
  x = e1
  y = e2
  body
}

-- Match
match e with {                    ⟹  (match {} e'
  | P1 => b1                           (arm {} P1' () b1')
  | P2 if g => b2                      (arm {} P2' g' b2'))
}

-- Tuples
(a, b, c)                         ⟹  (tuple {} a' b' c')
(a,)                              ⟹  (tuple {} a')
pair.0                            ⟹  (tuple-get {} (var {} pair) (lit {type: (t-prim {} i32)} 0))

-- Records
Foo { x: e1, y: e2 }             ⟹  (record {} Foo (kv {} x e1') (kv {} y e2'))
Foo { x, y }                      ⟹  (record {} Foo (kv {} x (var {} x)) (kv {} y (var {} y)))
foo with { x: e1, y }             ⟹  (record-update {} foo' (kv {} x e1') (kv {} y (var {} y)))
e.field                           ⟹  (access {} e' field)

-- Transforms
grad(f)                           ⟹  (grad {} f')
jit(f)                            ⟹  (jit {} f')
vmap(f, axis=n)                   ⟹  (vmap {} f' n')
vmap(f)                           ⟹  (vmap {} f' (lit {type: (t-prim {} i32)} 0))
cast(e, bf16)                     ⟹  (cast {} e' (t-prim {} bf16))
realize(e)                        ⟹  (realize {} e')
copy(e)                           ⟹  (copy {} e')
&x                                ⟹  (borrow {} (var {} x))
do { a; b }                       ⟹  (block {} a' b')
par { a; b }                      ⟹  (par {} a' b')
quote(e)                          ⟹  (quote {} e')
unquote(e)                        ⟹  (unquote {} e')
splice(e)                         ⟹  (splice {} e')
```

### 5.4 Type Expressions

```
f32                               ⟹  (t-prim {} f32)
bool                              ⟹  (t-prim {} bool)
string                            ⟹  (t-prim {} string)
tensor[batch, seq, f32]           ⟹  (t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))
tensor[a, b, f32]  (polymorphic)  ⟹  (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))
&tensor[batch, f32]               ⟹  (t-ref {} (t-tensor {} (d-name {} batch) (t-prim {} f32)))
A -> B -> C                       ⟹  (t-fn {} A' B' C')  -- flat, last is return
(A -> B) -> C                     ⟹  (t-fn {} (t-fn {} A' B') C')  -- arg is a function
Option[f32]                       ⟹  (t-adt {} Option (t-prim {} f32))
Frame[2]                          ⟹  (t-adt {} Frame (d-lit {} 2))
(f32, f32)                        ⟹  (t-tuple {} (t-prim {} f32) (t-prim {} f32))
unit                              ⟹  (t-unit {})
```

### 5.5 Patterns

```
x                                 ⟹  (pat-var {} x)
_                                 ⟹  (pat-wild {})
42                                ⟹  (pat-lit {} 42)
Some(x)                           ⟹  (pat-ctor {} Some (pat-var {} x))
Some(Some(x))                     ⟹  (pat-ctor {} Some (pat-ctor {} Some (pat-var {} x)))
Foo { x, y }                      ⟹  (pat-record {} Foo (kv {} x (pat-var {} x)) (kv {} y (pat-var {} y)))
(a, b)                            ⟹  (pat-tuple {} (pat-var {} a) (pat-var {} b))
x @ Some(_)                       ⟹  (pat-as {} x (pat-ctor {} Some (pat-wild {})))
```

---

## 6. Parsing and Disambiguation Rules

### 6.1 Flat application

An ordinary direct call has exactly one parenthesized argument list.
Juxtaposition and ungrouped chained calls are parse errors. A returned
function value is applied through a grouped callee, `(f(x))(y)`, which the
`AtomExpr CallArgs?` production represents without flattening the two calls.
Transform callees use the explicit `TransformExpr CallArgs?` production.

A call's argument list may end with the one named argument
`accumulator=<dtype>`, the explicit accumulator of spec/04 §5.7:
`sum(x, 0i32, accumulator=f64)`. It follows every positional argument and
desugars to the `app` node's `accumulator` metadata (spec/03 §1.1). Only a
call of the built-in `matmul`, `sum`, or `einsum` admits it; on any other
callee it is a type error, except that a macro call carrying it is rejected
earlier, during macro expansion ([02-MACRO-3]).

### 6.2 Bindings

Surf has one binding surface:

1. **Top level:** bare declarations such as `result = expr`.
2. **Inside a block `{ ... }`:** sequential bindings such as `x = expr` and `(a, b) = pair`.

There is no Surf `let ... in` expression form. `let` and `in` are ordinary identifiers.

### 6.3 Negative Literals vs Unary Minus

`-42` in expression position is always unary minus applied to `42`. The
literal itself is non-negative. `f -42` parses as `f - 42` (infix). Use parens
for negative arguments: `f(-42)`. During constant folding, `neg(42)` collapses
to a negative literal in Deep. Pattern position has no unary-expression node,
so minus plus an unsuffixed numeric token decodes directly to a negative
`pat-lit`; `-0` is rejected in favor of `0`, while `-0.0` preserves IEEE
negative zero.

### 6.4 Transform Recognition

`grad`, `vmap`, `jit`, `cast`, `cast_trunc`, `cast_saturate`, `cast_wrap`, `realize`, `copy` are keywords. In call position (`keyword(`), the parser emits a transform node. Bare usage (`g = grad`) is a parse error — transforms must always be applied.

`cast_trunc(x, T)`, `cast_saturate(x, T)` and `cast_wrap(x, T)` are the named lossy casts of [05-OP-6], [05-OP-23] and [05-OP-24]; each shares the `cast` node shape and differs only by carrying its `trunc`, `saturate` or `wrap` mode selector.

### 6.5 TypeIdent in Expression Position

Uppercase names in expression position are ADT constructors. Nullary: `None`. Record: `Adam { lr: 0.001 }`. Tuple-variant: `Some(x)`.

### 6.6 `type` Disambiguation

After `type Name =`, the parser checks if the next non-whitespace token is `|`. If yes → ADT (`TypeDecl`). If no → alias (`TypeAlias`).

---

## 7. Deep Vocabulary Boundary

`typealias`, `record-update`, and `pat-tuple` are active members of the closed
public Deep vocabulary. The complete 61-tag inventory and its compile-time
totality rule live in `spec/03-deep-syntax.md` §2 and §2.13; this Surf chapter
does not maintain a second count.

---

## 8. Example Programs

### 8.1 Linear Regression

```
module LinReg

dim features, samples

sig predict:
  tensor[samples, features, f32]
  -> tensor[features, 1, f32]
  -> tensor[1, f32]
  -> tensor[samples, 1, f32]

def predict(
  x: tensor[samples, features, f32],
  w: tensor[features, 1, f32],
  b: tensor[1, f32]
) -> tensor[samples, 1, f32] =
  add(matmul(x, w), insert(b, 0, samples))

def mse_loss(
  y_pred: tensor[samples, 1, f32],
  y_true: tensor[samples, 1, f32]
) -> tensor[f32] = {
  diff = sub(y_pred, y_true)
  sq = mul(diff, diff)
  mean(mean(sq, 1), 0)
}

def train_step(w, b, x, y, lr) = {
  loss_fn = fn (w_, b_) -> mse_loss(predict(x, w_, b_), y)
  (dw, db) = grad(loss_fn)(w, b)
  w_new = sub(w, mul(lr, dw))
  b_new = sub(b, mul(lr, db))
  (w_new, b_new)
}
```

### 8.2 MLP with Activation ADT

```
module Mlp

dim batch, input_dim, hidden_dim, output_dim

type Activation = | ReLU | Sigmoid

def activate(act: Activation, x: tensor[batch, hidden_dim, f32]) -> tensor[batch, hidden_dim, f32] =
  match act with {
    | ReLU    => relu(x)
    | Sigmoid => sigmoid(x)
  }

def forward(w1, b1, w2, b2, act, x) = {
  h = matmul(x, w1)
    |> add(b1)
    |> (fn (z) -> activate(act, z))
  add(matmul(h, w2), b2)
}
```

### 8.3 Records and Pattern Matching

```
module Optimizer

type Optimizer =
  | Sgd { lr: f32 }
  | Adam { lr: f32, beta1: f32, beta2: f32, eps: f32 }

default_adam =
  Adam { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8 }

def learning_rate(opt) =
  match opt with {
    | Sgd { lr }   => lr
    | Adam { lr }  => lr
  }

def scale_lr(opt, factor) =
  match opt with {
    | Sgd { lr } =>
        Sgd { lr: mul(lr, factor) }
    | Adam { lr, beta1, beta2, eps } =>
        Adam { lr: mul(lr, factor), beta1, beta2, eps }
  }
```

### 8.4 Dimension Polymorphism

```
module Linalg

def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] =
  permute(x, [1, 0])

def dot[n](x: tensor[n, f32], y: tensor[n, f32]) -> f32 =
  sum(mul(x, y))

def normalize[d](x: tensor[d, f32]) -> tensor[d, f32] = {
  norm = sqrt(sum(mul(x, x)))
  div(x, expand(norm, 0))
}
```

### 8.5 Training Loop with Grad

```
module Train

import Std.Io (println)
import Std.Iter (fold, range)   -- stdlib helpers, not built-ins

def train(model_w, model_b, data_x, data_y, lr, epochs) = {
  step = fn (wb, i) -> {
    (w, b) = wb
    loss_fn = fn (w_, b_) -> mse_loss(predict(data_x, w_, b_), data_y)
    (dw, db) = grad(loss_fn)(w, b)
    (sub(w, mul(lr, dw)), sub(b, mul(lr, db)))
  }
  fold(step, (model_w, model_b), range(0, epochs))
}
```

# Nomenclature

The Chelis ecosystem's naming conventions, organized by layer. This is
the canonical source of truth for identifier and filename conventions
across the chelis monorepo and all downstream shell repos
(`nautilus`, `coral`, `shoals`, `octant`).

The conventions documented here fall into three categories:

1. **Hard language constraints** (§1). Enforced by the parser, resolver,
   or backend. Immovable; not subject to style choice.
2. **Settled conventions** (§§2–10). Project-wide rules. Consistent
   across the ecosystem. Enforced by the `chelis lint` tool.
3. **Resolved decisions** (§11). Architectural questions surfaced
   during the May 2026 ecosystem naming pass and now closed. Recorded
   with their resolution so the rationale is visible.

A standalone lint tool (`chelis lint`) verifies adherence to the
settled conventions in §§2–10. CI runs it as a gate on every PR.
Blocking rule deviations are lint failures; advisory rules report
valid-but-non-preferred source without failing the gate.

---

## 1. Hard language constraints

The parser, resolver, and backends enforce these. Style choices live
inside them.

### 1.1 Surf identifier grammar

Surf identifiers split on first-character case at the lexer
(`crates/chelis-surf/src/lexer.rs`):

- Lowercase-leading → `Ident`. Used for values, functions, parameters,
  dimensions.
- Uppercase-leading → `TypeIdent`. Used for types, ADT constructors,
  module-ladder components.

Surf identifier charset is `[A-Za-z_][A-Za-z0-9_]*`. ASCII alphanumeric
plus underscore. **No hyphens.**

The lexer case-split is the *default* classification. An explicit
binding context overrides it for a name the user has explicitly bound,
the same way a def's `[..]` quantifier clause overrides the case-split
for type variables (`spec/02-surf-syntax.md` §P4a, where a quantified
PascalCase name such as `P` is a type variable, not a rigid ADT). The
single value-position override is:

- A **single ASCII-uppercase letter** (`S`, `K`, `T`, `N`, `P`, …) is a
  **value identifier** when it appears in a position that
  unambiguously binds or names a value: a top-level value-binding LHS
  (`S = ...`), a function or lambda parameter (`def payoff(S, K) = ...`),
  or a block binding (`{ S = expr ; ... }`). This admits finance/math
  notation as value names (chelis#437) without weakening the PascalCase
  convention.

The override is **single-letter only**. A multi-letter PascalCase name
(`Frame`, `Some`, `Foo`) remains a type or constructor everywhere; it is
never a value-binding LHS, parameter, or block binder, so `Foo = ...` is
still a parse error. In value-*reference* position a bare single-letter
uppercase name lexes as a constructor head and resolves to the in-scope
value binding when one exists (the resolver disambiguates
constructor-vs-value by environment lookup; see §3.2). Single-letter
uppercase in *type* position (a quantified `[..]` name, or a type
annotation) stays a type variable or type name — the value-binding
override applies only in the value-binding positions enumerated above.

### 1.2 Module-path lookup

`module Foo.Bar` resolves to `foo/bar.ch` (relative to the package's
`src/`). The desugarer lowercases each PascalCase component and joins
with `/`. Consequence: `.ch` filenames are forced lowercase or
snake_case to be importable.

### 1.3 Reserved keywords

28 active Surf keywords are lexically reserved, all lowercase:

```
def sig type dim macro match with fn module import export
if then else grad vmap jit realize copy tensor cast par do
quote unquote splice true false
```

Grammar-specific words such as `property`, `forall`, `where`, `opaque`, and
`invariant` are contextual rather than globally reserved. The future words
`effect`, `handler`, `perform`, `resume`, and `borrow` are also lexically
reserved but have no active production. Explicit borrow syntax is `&`.

### 1.4 Deep tag vocabulary

Closed vocabulary, hardcoded in `crates/chelis-deep/src/validate.rs`
and emitted by `crates/chelis-deep/src/printer.rs`. Authoritative list
in `spec/03-deep-syntax.md:262+`.

- **Single-token tags are bare lowercase**: `app`, `var`, `lit`, `def`,
  `sig`, `module`, `import`, `dim`, `deftype`, `variant`, `field`,
  `arm`, `meta`.
- **Compound tags use hyphens as the compound separator**: `t-fn`,
  `t-prim`, `t-tensor`, `t-var`, `pat-var`, `pat-lit`, `pat-ctor`,
  `d-name`, `d-var`, `d-lit`.

The vocabulary is closed and not user-controlled. The hyphen form
exists deliberately so the compound names read with visual structure;
see §1.5 for the corresponding grammar consequence.

### 1.5 Deep symbol grammar

Deep `Symbol` lexer accepts `[A-Za-z_][A-Za-z0-9_-]*`. Hyphens are
admitted so the closed compound-tag vocabulary in §1.4 can use them as
the compound separator.

The Surf-vs-Deep hyphen asymmetry is intentional structure, not a
defect: Surf is the human authoring surface where operator ambiguity
rules out hyphens; Deep is the compiler IR where compound tag names
follow a deliberate convention. User-defined Deep symbols (variable
names, function names) originate from Surf desugaring and inherit
Surf's no-hyphen rule by construction — there is no path by which a
hyphenated user symbol can enter Deep today.

The corresponding lint rule (§12) enforces the narrower invariant:
any Deep `Symbol` that is not in the closed tag vocabulary of §1.4
must satisfy the Surf identifier charset `[A-Za-z_][A-Za-z0-9_]*`.
The tag vocabulary is the allowlist; everything else passes through
the same constraint Surf imposes.

### 1.6 Formatter behavior

`chelis fmt` does not rewrite identifier case. Mixed styles survive
round-trips. The formatter is not the place to enforce style rules;
the lint is.

### 1.7 Backend symbol emission

> **[01-CID-1]** Every authored C function export other than source-level
> `main` SHALL use the compiler-owned symbol `chelis_fn_` followed by the
> lowercase hexadecimal UTF-8 bytes of its exact Chelis name. Source-level
> `main` SHALL use the module-qualified symbol specified by
> `spec/08-backends.md` §2. The mapping SHALL be injective for every legal
> Surf identifier, including C keywords, platform symbols, and names that
> themselves begin with `chelis_fn_`. The generated header and source SHALL
> use the same symbol. A Reef linker identity whose decoded source binding is
> exactly `main` retains that source-level exception; an ordinary authored
> identifier that merely ends in `__main` does not. The generated header
> and source SHALL carry the same versioned artifact envelope. The envelope
> binds the lowercase-hex UTF-8 program identity, the exact generated-source
> SHA-256 digest, and one declaration/export-block record per public
> definition. A declaration record carries the exact Chelis source identity,
> canonical C symbol, and canonical C declaration in lowercase-hex UTF-8,
> followed by declaration bytes that match the record exactly even when the
> declaration is multiline. Its source export block carries the export kind
> plus the same source identity, symbol, declaration, and a canonical
> definition commitment. That commitment hashes the exact enclosed C function
> definition bytes together with the translation unit's canonical source-local
> preprocessing-directive sequence. Validation SHALL parse the C
> translation unit structurally and prove that each block encloses exactly one
> externally linked function definition whose declarator and signature match
> the recorded symbol and declaration and whose exact definition bytes and
> preprocessing context match the recorded commitment. This structural parse,
> rather than C substring
> heuristics, determines definition boundaries and linkage. Downstream C code SHALL
> consume the generated declaration, and tooling that needs the Chelis-to-C
> association SHALL consume the envelope rather than reconstructing the
> symbol from source spelling. The declaration records and source export
> blocks SHALL form an exact set in program identity, source identity,
> canonical symbol, declaration, linkage, and exact definition. The header digest SHALL bind every
> source byte, so a partial header, an unmarked addition, a formatting or type
> mutation, or a source-name/symbol/definition reassociation is invalid before
> native execution. Recomputing the whole-source digest SHALL NOT authorize a
> changed public definition. Generated source SHALL NOT use direct or indirect
> preprocessor aliases to change a published definition's symbol or body.
> Every externally linked function definition other than the generated process
> entry `main` SHALL have exactly one export record. Translation-unit-private
> `static` helpers, including comment-separated multiline declarations, remain
> valid outside that set.

The source identifier itself is unchanged: no case rewriting or source-level
reservation is introduced. The compiler-owned spelling is the published C ABI
identity, not a renamed Chelis binding. Other backend-local user identifiers
remain subject to their target language's identifier rules.

---

## 2. Filesystem and manifest layer

### 2.1 Top-level repository directories

**Rule:** kebab-case.

Examples: `chelis`, `nautilus`, `coral`, `octant`, `shoals`. Single-word
repo names stay single-word; multi-word names use hyphens.

### 2.2 Rust crate directories and `Cargo.toml` `package.name`

**Rule:** kebab-case.

Examples: `chelis-backend-c`, `chelis-cli`, `chelis-effects`,
`chelis-runtime`, `tree-sitter-chelis-surf`.

Rust standard hyphen→underscore coercion applies to library names:
`chelis-python` (package) becomes `chelis_python` (lib symbol). This is
a Rust language norm. See §11.2 for the orchestrator decision to keep
this shoreline as documentation rather than rename.

### 2.3 `.rs` module files

**Rule:** snake_case (Rust standard).

Examples: `parser.rs`, `lexer.rs`, `span_merge.rs`, `optimize.rs`,
`pipeline.rs`, `reef_install_from_github.rs`.

### 2.4 `.ch` (Surf) source files

**Rule:** lowercase or snake_case (forced by §1.2).

Examples: `linalg.ch`, `pricing.ch`, `pattern_matching.ch`, `mlp.ch`,
`cubic_hermite.ch`. Never PascalCase, kebab, or mixed-case.

### 2.5 `.dp` (Deep) source files

**Rule:** snake_case.

Deep is generated, not authored. Test fixtures and translator outputs
follow snake_case naming. Examples: `simple_def.dp`, `hello_tensor.dp`,
`call_price.dp`.

### 2.6 Reef package manifests

**Rule:** Package `name` is kebab-case; `module_prefix` is PascalCase.

Examples:

| Package         | `name`         | `module_prefix` |
|-----------------|----------------|-----------------|
| chelis-std      | `chelis-std`   | `Std`           |
| nautilus        | `nautilus`     | `Nautilus`      |
| coral           | `coral`        | `Coral`         |
| shoals          | `shoals`       | `Shoals`        |
| octant          | `octant`       | `Octant`        |
| c-earchin       | `c-earchin`    | `CEarchin`      |
| capstone        | `capstone`     | `Capstone`      |

The `module_prefix` form is forced by Surf's case-split rule (§1.1):
identifiers must be uppercase-leading to be module-path components.
The short prefix `Std` is preferred over `ChelisStd` for the runtime
package.

`CEarchin` is the deliberate c-earchin exception: the capital `C` followed by
`Earchin` preserves the visual distinction between package prefix `c` and
domain name `earchin`. It is not an ALL-CAPS abbreviation.

### 2.7 Cross-shell dependency keys

**Rule:** kebab-case (matches the package `name`), with semantic-version
strings.

```toml
[dependencies]
chelis-std = { version = "0.1.0" }
nautilus   = { version = "0.5.0" }
coral      = { version = "0.5.0" }
```

### 2.8 Python files

**Rule:** snake_case (PEP 8 throughout).

Examples: `benchmark_parser_throughput.py`, `bump_compiler_pins.py`, `gen_goldens.py`,
`validate_book_examples.py`.

### 2.9 Shell scripts

**Rule:** Don't write them. Project policy (`AGENTS.md` / `CLAUDE.md` Scripting
Language Policy) prohibits shell scripts in favor of Python. Existing
shell scripts must be ported.

**One exception:** the chelisup bootstrap installer at
`crates/chelisup/bootstrap/chelisup.sh` (the `curl -fsSL ... | sh`
one-liner). It runs on a bare machine before any chelis, cargo, or
Python exists, so it cannot be written in any of those. It must be
minimal POSIX `sh`, `shellcheck`-clean, and test-covered (`sh -n` parse
plus `shellcheck`). It is the only entry on the `no-shell-scripts`
exception list; every other script remains Python.

### 2.10 CI workflow filenames

**Rule:** lowercase, no separator.

Examples: `ci.yml`, `release.yml`, `nightly.yml`. Identical across all
five ecosystem repos.

### 2.11 Hidden config conventions

| Path                       | Convention                                                  |
|----------------------------|-------------------------------------------------------------|
| `.github/workflows/*.yml`  | lowercase no-separator                                      |
| `.claude/skills/<name>/`   | kebab-case                                                  |
| `.claude/commands/*.md`    | kebab-case                                                  |
| `.claude/skills/*/SKILL.md`| SCREAMING_SNAKE_CASE (literal filename, see §8.3)           |

### 2.12 Tree-sitter grammars

**Rule:** kebab-case crate names. Two parallel grammars:
`tree-sitter-chelis-surf` and `tree-sitter-chelis-deep`.

---

## 3. Identifier conventions: Surf

### 3.1 Types and ADT constructors

**Rule:** PascalCase. Forced by Surf's case-split (§1.1).

Examples: `Frame`, `Column`, `GroupedFrame`, `Hamt`, `KeyValue`,
`YieldCurve`, `OrderBook`, `Decimal`, `Tokenizer`, `Json`, `JsonInt`,
`JsonObject`, `AggSum`, `RoundHalfEven`, `Activation`, `Relu`,
`Sigmoid`.

ADT constructors follow the same rule: `Some`, `None`, `Ok`, `Err`,
`Empty`, `Leaf`, `Collision`.

### 3.2 Functions and values

**Rule:** snake_case, with a single-letter math-notation carve-out.

Examples: `predict`, `loss`, `inner_product`, `matvec`, `from_pairs`,
`with_column`, `rolling_mean`, `golden_section_search`, `simpsons`.

**Single-letter math-notation carve-out (chelis#437).** A value binding
may be named with a single ASCII-uppercase letter (`S`, `K`, `T`, `N`,
`P`) in math-heavy code where that letter is the standard notation —
finance `S` (spot), `K` (strike), `T` (maturity), `N` (normal CDF), `P`
(probability). This is the value mirror of the §3.3 parameter register
and of the §P4a `[..]`-clause type-variable override: the explicit
binding makes the single-letter uppercase name a value. The carve-out is
single-letter only; multi-letter value names remain snake_case (a
multi-letter PascalCase name is a type or constructor, §3.1). The parser
accepts the single-letter uppercase value binder directly; in
value-*reference* position the name lexes as a constructor head and the
resolver binds it to the in-scope value
(`crates/chelis-types/src/infer.rs` resolves `var`/constructor heads by
environment lookup, so an uppercase reference with a value binding in
scope is a value, and an uppercase reference with no binding remains an
unknown-constructor error).

The carve-out does **not** extend to `def`/`sig` function names. An
*applied* uppercase head — `N(x)` — resolves to a constructor
application, so a function named `N` would be silently shadowed by
constructor resolution rather than called. Function names therefore stay
snake_case; the `surf-value-snake-case` lint keeps flagging an uppercase
`def` name. (A nullary value *reference* like `S` is unambiguous because
nothing is applied; an applied head is where the ambiguity bites.)

### 3.3 Parameters

**Rule:** Bimodal. Two registers, picked by the surrounding code's idiom.

- **Math-heavy code** (Shoals, Nautilus): single-letter parameters
  match mathematical notation. Both cases are admitted for the
  single-letter register: lowercase `s` (spot), `k` (strike), `r`
  (rate), `sigma`, `t`, `s0`, `mu`, `a`, `b`, `m`, `n`, `i`, `j`,
  `alpha`; and uppercase `S` (spot), `K` (strike), `T` (maturity), `N`
  (normal CDF), `P` (probability) where uppercase is the conventional
  notation (chelis#437). The uppercase form is single-letter only.
- **Data-processing code** (Coral, Std): descriptive snake_case.
  `col`, `col_name`, `df`, `values`, `entries`, `idx`, `hash`.

The bimodality is intentional: math code reads better in math notation,
data code reads better in descriptive prose. **No camelCase parameters
anywhere.** The narrow exception is index-plus-math-symbol composites
(`dW`, `tA`, `c1I`) which are still single-token identifiers under
Surf's lexer.

### 3.4 Dimensions

**Rule:** snake_case or single letter. No PascalCase or camelCase.

Examples: `seq`, `vocab`, `classes`, `batch`, `out_dim`, `hidden`,
`features`, `n`, `b`, `m`, `k`, `p`.

### 3.5 Function definition return-type syntax

**Rule:** When a `def` carries an explicit return type, write it with the
arrow form `def name(params) -> T = expr`. Do not use the colon form
`def name(params) : T = expr`.

```chelis
def softmax(x: Tensor[batch, vocab, f32]) -> Tensor[batch, vocab, f32] =
  exp(x) / sum(exp(x), dim = vocab)

def loss(p: f32, q: f32) -> f32 = -(p * log(q))
```

The arrow is the only canonical Surf spelling. The canonical parser rejects a
colon in result position; the explicit v0.18 migration path rewrites legacy
colon-form declarations. This visually pairs with parameter `:` annotations
without overloading the colon for two unrelated jobs (parameter binding vs.
function-result type). `chelis fmt` formats canonical Surf and does not act as
a dialect translator.

### 3.6 Pipe-first composition and first-argument stages

**Rule:** Pipe stages use first-argument insertion. In Surf,
`x |> f(y, z)` means `f(x, y, z)`, not `f(y, z, x)`.

```chelis
x |> normalize |> add(bias) |> relu
```

desugars as if written:

```chelis
relu(add(normalize(x), bias))
```

A bare stage (`x |> f`) passes the piped value as the only argument to
`f`. A call stage (`x |> f(y, z)`) inserts the piped value before the
written arguments. If a later argument position is intended, write an
explicit lambda:

```chelis
x |> fn (v) -> f(y, v)
```

Canonical Surf producers promote a nested application chain to this pipe form
only when the typed pipeline proof establishes a linear first-argument
dataflow chain. If that proof fails, the producer retains calls;
later-position insertion retains the explicit lambda. The equivalence is
limited to the proven chain and does not authorize token-only or untyped call
rewriting.

(This requirement is not fully implemented; see chelis#1171.)

The decompiler may compact a lambda stage back to call-stage sugar only
when the carried value is the first argument of the call. Naming rules
that refer to a function's principal or first argument, including the
type-suffix rule in §7.2, use this same interpretation for pipe stages.

---

## 4. Identifier conventions: Rust

**Rule:** Rust standard.

- Types and traits: `PascalCase`. Examples: `OctantError`, `SourceId`,
  `Span`, `FunctionRegistry`, `MetaMap`, `MetaExpr`.
- Functions and modules: `snake_case`. Examples: `parse_module`,
  `lower_to_dag`, `emit_c`.
- Constants: `SCREAMING_SNAKE_CASE`. Examples: `DEFAULT_MODEL`,
  `API_URL`, `RETRYABLE_CODES`, `FALLBACK_CHAIN`.

No deviations.

---

## 5. Identifier conventions: Python

**Rule:** PEP 8.

- Functions and modules: `snake_case`.
- Classes: `PascalCase`.
- Constants: `SCREAMING_SNAKE_CASE`.

Examples: `load_api_key`, `chelis_version`, `build_and_compile_canary`,
`PDFLinkExtractor`, `DEFAULT_MODEL`.

No deviations in Chelis-authored Python.

---

## 6. Module conventions

### 6.1 Module ladder structure

**Rule:** Modules form a ladder rooted at the shell's `module_prefix`.
Each component is a single PascalCase word or a Title-case compound.

Examples: `Nautilus.LinAlg`, `Coral.Frame`, `Shoals.Pricing`,
`Std.Tensor`, `School.Loss.CrossEntropy`, `School.Nn.RmsNorm`.

### 6.2 Compound styling: Title-case, not ALL-CAPS

**Rule:** When a module name component is a compound (multi-word or
abbreviation), use Title-case for each word. Do not use ALL-CAPS for
abbreviations.

Examples:

| Correct        | Incorrect       | Notes                          |
|----------------|-----------------|--------------------------------|
| `Hamt`         | `HAMT`          | hash-array-mapped trie         |
| `Io`           | `IO`            | input/output                   |
| `KlDiv`        | `KLDiv`         | KL divergence                  |
| `Json`         | `JSON`          | JSON format                    |
| `Lstm`         | `LSTM`          | recurrent network              |

The same rule applies to type constructors that name the abbreviation:
the `Hamt` constructor is correct; `HAMT` would be the violation.

### 6.3 Module-name PascalCase per component

**Rule:** Every component of a module ladder is PascalCase, including
each word inside a compound. Lowercase-after-first compounds are
violations.

Examples:

| Correct                          | Incorrect                       |
|----------------------------------|---------------------------------|
| `Nautilus.ExampleRootFind`       | `Nautilus.Examplerootfind`      |
| `Nautilus.ExampleOdeDemo`        | `Nautilus.Exampleodedemo`       |
| `Nautilus.ApiSmoke`              | `Nautilus.Apismoke`             |
| `Demo.HelloTensor`               | `Demo.Hellotensor`              |

This rule is the one already documented in `spec/02-surf-syntax.md:79`
("Module names are PascalCase. No nesting within a file.") applied
consistently to every component.

### 6.4 The runtime vs shell distinction

`chelis-std` is the language **runtime**, not a shell. It version-marches
with the compiler, ships bundled with the toolchain, and cannot be
substituted independently. The reef lockfile records it as
`LockSource::Bundled`. See `spec/design/chelis_canonical_reference.md`.

`chelis-runtime` is a separate Rust crate at `chelis/crates/chelis-runtime/`
supporting the C backend. Different artifact, different role.

The ecosystem's other reef packages are **shells**: distributable
libraries that build on `chelis-std`. The currently shipped shells
are `nautilus`, `coral`, `shoals`, and `octant`. Designed but not
yet shipped: `school`, `darwin`, `hull`, `hydrostatic`, `beacon`.

### 6.5 Module identity inside a reef package

**Rule:** A `.ch` source file inside a reef package declares exactly one
module, and that module's name is fixed by the package's `module_prefix`
(§2.6) together with the file's path beneath its source root. The file
does not choose its own name.

The name is built by taking `module_prefix`, then one component per path
segment beneath the source root, in order, with the file's extension
dropped. Each component is the path segment itself; the comparison
between the declared name and the derived name ignores case only. For a
package declaring `module_prefix = "Demo"`:

| File | Module it must declare |
|------|------------------------|
| `src/linalg.ch`      | `Demo.Linalg`      |
| `src/nn/linear.ch`   | `Demo.Nn.Linear`   |
| `src/io/json.ch`     | `Demo.Io.Json`     |

A source root other than `src` contributes its own name as the first
component after the prefix, so `properties/laws.ch` declares
`Demo.Properties.Laws`.

Three ways to violate the rule, each of which makes the package
uncompilable:

- the declared module is not rooted at `module_prefix`
  (`module Data` in a `Demo` package);
- the declared module is rooted at the prefix but does not derive from
  the file's path (`module Demo.Other` in `src/data.ch`);
- the file declares no module, or more than one.

**This rule is not a style preference, and a conforming implementation
has no discretion about it.** A file whose module identity disagrees
with its path cannot be placed in the package's module graph at all, so
the failure is a load-time rejection rather than a diagnostic the rest
of the pipeline can route around. It follows that a tool reporting on a
package's well-formedness must be able to reach this verdict, and must
not report success on a package that violates it. §12's blocking lint
gate is such a tool. `chelis fmt --check` is not: it judges one file's
formatting and claims nothing about the package that file belongs to.

---

## 7. Function naming patterns

### 7.1 Prefix-namespacing policy

**Rule:** Private/helper functions in a module get a prefix that's a
shorthand for the module's domain. Public functions don't.

The prefix is applied uniformly within the module to all internal
helpers. The prefix is short (two-to-four letters), distinctive, and
derived from the module's name or its primary subject matter.

Examples:

```chelis
// Nautilus.LinAlg: internal helpers prefixed with `la_`
def la_vec_sub(a, b) = ...
def la_vec_add(a, b) = ...
def la_basis_n_f32(n) = ...
def la_qr_build_q(...) = ...

// Nautilus.LinAlg: public functions with bare names
def transpose(m) = ...
def matmul_wrap(a, b) = ...
def gram(m) = ...
def qr_decompose(m) = ...
```

The rule covers both directions:

- A public function should not carry a module prefix. `bs_call_scalar`
  in `Shoals.Pricing` violates the rule because (a) it's public and
  (b) the prefix `bs_` is not the module's shorthand. The fix is
  `call_scalar`.
- A private helper should carry the prefix uniformly. If a `Nautilus.LinAlg`
  helper exists without the `la_` prefix, it's either a public function
  (rule: drop "helper" status) or a violation (rule: add the prefix).

The prefix is a domain shorthand, not the full module name. `Nautilus.LinAlg`
uses `la_*`; `Coral.Frame` uses no internal prefix today (its private
helpers, where they exist, would use `frame_*` or similar). The lint
flags inconsistency within a module: either all internal helpers have
the prefix or none do.

### 7.1.1 Model/algorithm sub-namespace prefixes

**Rule:** A function-name prefix may name a **model, algorithm,
mathematical object, or numerical method** within a module whose
domain hosts multiple coexisting variants of the same conceptual
operation. The prefix is short (two-to-four lowercase letters),
uniformly applied to every member of that variant family within the
module, and refers to a named mathematical object — not the module's
domain shorthand of §7.1.

§7.1.1 separates this case from the §7.1 helper-marker rule. The
distinction is:

- §7.1 prefix marks a private internal helper. The bare-name surface
  of the module is the public API.
- §7.1.1 prefix marks one of several public model/algorithm variants
  the module hosts. Each variant carries its own prefix uniformly;
  the module name describes the domain (`Pricing`, `Stochastic`),
  not any single variant.

Recognized model/algorithm sub-namespaces in the current ecosystem:

| Prefix | Meaning                            | Module(s)                       |
|--------|------------------------------------|---------------------------------|
| `bs_`  | Black-Scholes analytical pricing   | `Shoals.Pricing`                |
| `mc_`  | Monte-Carlo simulation pricing     | `Shoals.Pricing`                |
| `gbm_` | Geometric Brownian motion paths    | `Shoals.Stochastic`             |
| `fd_`  | Finite-difference numerical method | `Shoals.Properties.Greeks`      |
| `lm_`  | Levenberg-Marquardt fitting        | `Nautilus.CurveFit`             |
| `cg_`  | Conjugate gradient solver          | `Nautilus.LinAlg`               |
| `airy_`| Airy-function family               | `Nautilus.Special`              |
| `beta_`| Beta distribution / function       | `Nautilus.Distributions`        |
| `chi_` | Chi-squared distribution           | `Nautilus.Distributions`        |
| `det_` | Determinant of fixed-rank matrix   | `Nautilus.LinAlg`               |
| `eig_` | Eigenvalue helpers                 | `Nautilus.LinAlg`               |
| `inv_` | Matrix inverse of fixed rank       | `Nautilus.LinAlg`               |
| `erf_` | Error-function family              | `Nautilus.Special`              |

A prefix qualifies for §7.1.1 only when (a) it is uniformly applied
to every member of its family within the module, and (b) the family
names a coherent mathematical object. Single-use prefixes don't
qualify; inconsistent application within a family doesn't qualify
(those remain §7.1 violations).

The lint tracks these in `crates/chelis-lint/src/rules/prefix_namespace.rs`
under `MODEL_NAMESPACE_PREFIXES`. Adding a new model/algorithm
prefix requires both updating that table and updating this section's
recognized-list.

#### Math/ML well-known prefixes (domain-agnostic)

A parallel curated list of math/ML well-known prefixes accepts canonical
elementary-math, statistics, loss-function, activation-function, and
ML-lifecycle operation names regardless of the hosting module's domain
shorthand. These names are shared across the wider math/ML ecosystem
(numpy, scipy, torch, tensorflow) and identify the operation, not a
module-scoped variant family.

Recognized math/ML well-known prefixes:

| Prefix      | Family                                      |
|-------------|---------------------------------------------|
| `exp_`      | Elementary math: exponential                |
| `sin_`      | Elementary math: sine                       |
| `cos_`      | Elementary math: cosine                     |
| `tan_`      | Elementary math: tangent                    |
| `std_`      | Statistics: standard deviation              |
| `var_`      | Statistics: variance                        |
| `lin_`      | Linear interpolation / linear methods       |
| `mse_`      | Loss: mean squared error                    |
| `ce_`       | Loss: cross entropy                         |
| `kl_`       | Loss: Kullback-Leibler divergence           |
| `bce_`      | Loss: binary cross entropy                  |
| `relu_`     | Activation: rectified linear unit           |
| `gelu_`     | Activation: gaussian error linear unit      |
| `silu_`     | Activation: sigmoid linear unit             |
| `tanh_`     | Activation: hyperbolic tangent              |
| `fit_`      | ML lifecycle: fit a model                   |
| `pred_`     | ML lifecycle: predict from a model          |
| `eval_`     | ML lifecycle: evaluate a model              |

The lint tracks these in `crates/chelis-lint/src/rules/prefix_namespace.rs`
under `MATH_ML_WELL_KNOWN_PREFIXES`. The distinction from
`MODEL_NAMESPACE_PREFIXES` is scope: model-namespaces are tied to a
specific module domain (`bs_` in `Shoals.Pricing`); math/ML well-known
prefixes are domain-agnostic — a function named `mse_loss` is the same
operation whether it lives in `Hello.Loss`, `Nautilus.Stats`, or any
other module. Adding to this list requires updating both the const and
this section.

Longer well-known prefixes (`softmax_`, `sigmoid_`, `train_`) are 5+
characters and fall outside the short-prefix extractor's 2–4 character
window. They are recognized implicitly because the rule never extracts
them as a prefix in the first place. They are documented here for
clarity but require no entry in `MATH_ML_WELL_KNOWN_PREFIXES`.

### 7.2 Type-suffix policy

**Rule:** Type and shape suffixes describe the **element type or
shape of the principal argument**, never the container or dispatch
form.

#### Element-type suffixes

`_int`, `_f32`, `_bool`, `_string` indicate that the function is
monomorphized to that element type. They appear when a function has
type-specific behavior that wouldn't be captured by HM inference, or
when the function name benefits from being explicit about the type
it operates on.

```chelis
def parse_int(s: string) -> Option[i64] = ...
def parse_string(s: string) -> string = ...
def parse_bool(s: string) -> Option[bool] = ...
```

These suffixes are **not** used on functions that work generically
over the element type.

#### Shape suffixes

`_scalar`, `_vec` indicate the rank/structure of the principal
argument. `_scalar` is rank-0; `_vec` is rank-1. Not used when the
function works generically over rank.

```chelis
def softmax_vec[n](x: tensor[n, f32]) -> tensor[n, f32] = ...
def relu_scalar(x: f32) -> f32 = ...
```

#### Container-form distinction is never a type suffix

If a function operates on a different container type (e.g., a Frame
column rather than a raw tensor), use a distinct mechanism:

- A `_col` suffix denotes a column-form variant of the same
  operation, taking a Frame plus a column name.
- A different module namespace if the function genuinely belongs
  there (e.g., functions that primarily operate on Frames belong in
  `Coral.Frame.*`).
- A different verb if the operation differs structurally.

**Never** overload an element-type suffix to mean "dispatch form."

```chelis
// Correct
def is_nan[n](t: tensor[n, f32]) -> tensor[n, bool] = ...     // tensor variant
def is_nan_col[n](f: Frame, name: string) -> tensor[n, bool] = ... // column variant

// Incorrect: `_int` here means "Frame variant taking int column",
// which conflates element type with dispatch form.
def is_nan_int[n](f: Frame, name: string) -> tensor[n, bool] = ...
```

The historical Coral `*_int` family (`is_nan_int`, `any_nan_int`,
`count_nan_int`, `fill_nan_int`, `drop_nan_int`) is renamed to the
`*_col` form per this rule. The `_int` was extraneous: column dtype
is inferred when the column is fetched.

#### Parser/converter idiom (allowed)

A type-suffix may also describe **the type a function tests for or
produces**, not just the type of the principal argument, when the
function name begins with a parser/converter verb prefix:

- `parse_*`, `unwrap_*`, `try_*`, `is_some_*`, `as_*`, `from_*`, `to_*`

In these cases the suffix is naming what the function reads-out or
asserts about, not what its argument's element type is. The principal
argument is typically a string (or another carrier of the encoded
data).

```chelis
def parse_int(s: string) -> Option[i64] = ...    // tests/produces int
def unwrap_int(s: string) -> i64 = ...           // produces int (panicking)
def is_some_int(s: string) -> bool = ...           // tests for int
def is_some_float(s: string) -> bool = ...         // tests for float
def from_string_to_bool(s: string) -> bool = ...   // converter
```

The lint accepts this idiom when both conditions hold: the function
name starts with one of the recognized parser/converter prefixes, AND
the suffix names a recognized element type (`_int`, `_f32`, `_f64`,
`_bool`, `_string`).

---

## 8. Documentation conventions

### 8.1 Spec files

**Rule:** Numeric prefix + kebab-case for the top-level numbered
language specs in `chelis/spec/` only. The numbering reflects the
chelis monorepo's authoritative language-spec ordering.

```
00-context.md
01-nomenclature.md
02-surf-syntax.md
...
12-roadmap.md
```

Shell repos (`nautilus`, `coral`, `shoals`, `octant`) have their own
`spec/` directories holding per-shell implementation plans and design notes.
Those follow §8.2's snake_case rule, not §8.1's numbered-spec rule.

### 8.2 Design files

**Rule:** snake_case in `chelis/spec/design/` and in any shell repo's
top-level `spec/` directory.

Examples: `dtype_semantics.md`, `chelis_canonical_reference.md`,
`pre_release_validation.md`, `grad_eval_host_runtime.md`,
`runtime_abi.md` (shell repo implementation plan).

The historical kebab-case minority files (`grad-eval-host-runtime.md`,
`host-emit-hashmap-iteration-nondeterminism.md`, etc.) rename to
snake_case.

### 8.3 Per-shell `docs/`

**Rule:** snake_case for narrative documents. SCREAMING_SNAKE_CASE for
status reports.

Narrative documents (architecture descriptions, error-message catalogs,
extending guides):

```
architecture.md
error_messages.md
extending.md
supported_subset.md
```

Status reports (release notes, benchmark results, upstream-bug
tracking):

```
STATUS.md
RELEASES.md
UPSTREAM_BUGS.md
BENCHMARK_FINDINGS.md
EVAL_STARTUP_FINDINGS.md
```

The SCREAMING_SNAKE convention is grandfathered from the existing
nautilus/coral practice and applies project-wide for status reports.
Single-word status-report filenames (`SKILL.md`, `STATUS.md`) follow
the same SCREAMING_SNAKE rule and collapse visually to looking like
PascalCase.

#### Cargo-package-name exception

A narrative-docs filename may use kebab-case when its filename stem
matches the `name` of a Cargo package in the workspace. This carve-out
exists because Cargo package names are kebab-case by convention
(`c-earchin`, `chelis-runtime`, `chelis-cli`), and a documentation file
named for a specific Cargo crate (`docs/shells/c-earchin.md`) reads
more naturally with the package's own name shape than with a forced
snake-case rewrite.

The exception applies only to filenames that:

- match a Cargo `[package].name` in the workspace exactly (no fuzzy
  matching), and
- pass the kebab-case regex `^[a-z][a-z0-9-]*\.md$`.

Filenames that don't match a Cargo package fall under the base §8.3
rule. The lint enforces this in
`crates/chelis-lint/src/rules/doc_filename_convention.rs` by walking
the workspace for `Cargo.toml` files and reading their `[package].name`.

### 8.4 Versioned reports

**Rule:** snake_case with version markers in underscored form.

```
red_team_v0_4_0_pre_tag.md
red_team_o3.md
red_team_o4.md
red_team_pre_v0_1_0.md
```

The historical kebab+version style (`red-team-v0.2.0-final.md`)
renames forward.

### 8.5 mdBook book chapters (deliberate exception)

**Rule:** kebab-case. Applies to any path whose components include a
directory literally named `book/`. In the chelis ecosystem the
canonical location is `<repo>/docs/book/`; some shells may also use a
top-level `<repo>/book/`. Both are covered.

mdBook book chapters expect kebab-case URLs for stability across
renderers. This is a deliberate exception from the broader
documentation convention; the rest of the project uses snake_case.

```
first-program.md
cli.md
install.md
reef.md
effects.md
cg-solve.md
monte-carlo.md
```

The discriminator is path-based, not `book.toml`-anchored. A
`book.toml` sitting in `docs/` (or anywhere else in the ancestor
chain) does not retroactively promote sibling `docs/*.md` files to
§8.5. This is the issue #190 fix: previously the rule walked ancestors
looking for `book.toml`, which made the verdict for every `docs/*.md`
depend on unrelated filesystem state (adding or removing one
`book.toml` flipped every narrative doc between accepted and
rejected). Path-based opt-in keeps each verdict local to the file.

Shells that want mdBook content put it under `book/` or `docs/book/`.
This is the chelis-ecosystem convention; non-`book/` mdBook layouts
are out of scope for the lint.

The lint implements this in
`crates/chelis-lint/src/rules/doc_filename_convention.rs` by checking
whether the path's components contain the literal name `book`. The
check is cheap and depends only on the path, not on filesystem
state.

#### Tool-required exceptions inside any mdBook source tree

Two filenames inside an mdBook source tree are determined by mdBook
itself, not by this spec, and are exempt from the kebab-case rule:

- `SUMMARY.md` — the mdBook table-of-contents file. mdBook requires
  this literal filename.
- `README.md` — the mdBook chapter-index file. mdBook resolves the
  index page from this literal filename.

The exemption is for exactly these two filenames. Other uppercase
files inside an mdBook source tree are still violations of §8.5.

### 8.6 Public prose punctuation

New public strings should avoid em dashes. Prefer one of these fixes
when cleaning existing text:

- split the sentence into two sentences
- use a colon before an explanation
- use parentheses for a true aside
- use a comma or semicolon when the grammar calls for one
- use ASCII ` - ` only for a deliberately parenthetical break

The blocking `no-em-dash-in-public-strings` rule enforces this for
Surf, Deep, Rust, and Python string literals that are likely to reach
users as diagnostics, log messages, or public output. The v1 fixer is
deliberately narrow: `a — b` becomes `a. B`; paired parenthetical
dashes become commas; whitespace-asymmetric cases require manual
review. Markdown prose enforcement is queued until the active doc
corpus is cleaned. Do not add lint exceptions merely to preserve an em
dash in current-state docs. This rule does not prohibit syntax or
notation that is semantically meaningful in a spec, such as `->`, `|>`,
section references, or mathematical symbols.

#### Docstring exclusion (Python)

Python module, function, class, and method docstrings are narrative
prose, not user-facing strings. The rule excludes them.

The lint detects docstrings by shape: a Python triple-quoted string
(`"""..."""` or `'''...'''`) whose **opening triple-quote is the first
non-whitespace token on its line**. This catches all three canonical
docstring positions (module-top, after `def`, after `class`) and
excludes mid-line triple-quoted strings like
`print("""...""")` or `raise ValueError("""...""")`, which are
user-facing and still bound by the rule.

Rust `///` and `//!` doc comments are not string literals and are
already outside the rule's scope; no additional carve-out is needed.

---

## 9. Project-cutting conventions

### 9.1 Commit conventions

**Rule:** Conventional commits.

```
feat(reef): record lockfile remote origin
test(reef): cover parallel bootstrap installs
style(reef): give bootstrap tests descriptive names
docs(spec): define declarative naming
```

### 9.2 CI workflows

**Rule:** Three workflow files per repo, identical names across all
five repos.

```
.github/workflows/ci.yml
.github/workflows/release.yml
.github/workflows/nightly.yml
```

---

## 10. Test naming

### 10.1 Surf test functions

**Rule:** `def test_*` for unit tests; `def example_*` for illustrative
example functions in the same files.

```chelis
def test_softmax_sums_to_one() = ...
def example_simple_inference() = ...
```

### 10.2 Rust test functions

**Rule:** snake_case, descriptive.

```rust
#[test] fn roundtrip_simple_def() { ... }
#[test] fn span_id_unicode_preserved_bit_for_bit() { ... }
#[test] fn dim_params_produce_d_var() { ... }
```

### 10.3 Insta snapshot tests

**Rule:** `{context}__{section}__{test_name}.snap` with snake_case
components.

```
cli__error_format__parse_error_format.snap
cli__error_format__type_error_format.snap
parser__roundtrip__simple_def.snap
```

Insta's double-underscore separator is the canonical separator for
snapshot test files across the ecosystem.

---

## 11. Resolved decisions

These were open architectural questions during the May 2026 ecosystem
naming pass. Each is now closed with the orchestrator's resolution
recorded. The lint enforces the resolved rule.

### 11.1 Surf hyphen support — resolved: keep Deep grammar; document asymmetry as intentional

**Status:** closed.

The original framing surfaced this as a "latent round-trip risk":
Deep's `Symbol` lexer accepts hyphens (`[A-Za-z_][A-Za-z0-9_-]*`);
Surf does not. A first-pass plan proposed tightening the Deep lexer
to reject hyphens.

That plan was based on incorrect facts. The Deep tag vocabulary
(§1.4) deliberately uses hyphens as the compound separator: `t-fn`,
`t-prim`, `pat-ctor`, `pat-var`, `d-name`, `d-var`, `d-lit`. The
hardcoded list lives in `crates/chelis-deep/src/validate.rs`; the
printer at `crates/chelis-deep/src/printer.rs` emits these hyphenated
forms; checked-in `.dp` fixtures depend on parsing them. Closing the
loose side of the asymmetry would have required renaming the entire
compound-tag vocabulary and breaking every checked-in `.dp` fixture
plus the round-trip lexer test that explicitly asserts `x-y` lexes
as a single Symbol.

The actual resolution: **the Surf-vs-Deep hyphen asymmetry is by
design and documented as such.** Surf is the human authoring surface
where operator ambiguity rules out hyphens. Deep is the compiler IR
where compound tag names follow a deliberate naming convention that
gives the closed vocabulary its visual structure. User-defined Deep
symbols originate from Surf desugaring and inherit Surf's no-hyphen
rule by construction; there is no path by which a hyphenated user
symbol can enter Deep today.

The lint rule (§12) captures the actual narrower invariant: any Deep
`Symbol` that is not in the closed tag vocabulary of §1.4 must
satisfy the Surf identifier charset `[A-Za-z_][A-Za-z0-9_]*`. This
rule fires if a future Deep emitter accidentally produces a
hyphenated user symbol while leaving the legitimate compound-tag use
untouched.

The three candidate resolutions considered, for historical
visibility:

- (a) Allow hyphens in Surf with mandatory whitespace around `-`.
  Breaking parse change; rejected.
- (b) Allow hyphens in Surf in restricted positions. Complex;
  rejected.
- (c) Tighten the Deep lexer to reject hyphens. Initially preferred
  on the assumption that no hyphenated Deep symbols existed. Rejected
  once evidence showed the entire compound-tag vocabulary uses
  hyphens.
- (d, chosen) Document the asymmetry as intentional structure; keep
  both lexers as-is; rely on the user-symbol-charset lint rule.

### 11.2 Rust hyphen→underscore lib name — resolved: documented shoreline crossing

**Status:** closed.

`chelis-python` package name maps to `chelis_python` lib name per
Rust standard hyphen→underscore coercion. The orchestrator
resolution: **document this as a known shoreline crossing imposed
by Rust language norms; no rename.** §2.2 records the rule;
the implicit hyphen→underscore in lib symbols is the Rust
standard, not a project-specific deviation.

The two alternatives considered:

- Rename the package name to `chelis_python` so package and lib
  align (kebab → snake migration). Would have set precedent for the
  kebab convention being violated for Rust packages. Rejected.
- Rename the lib name to break the Rust standard (not actually
  possible without hacks). Rejected.

### 11.3 Allow vs keep semantics — resolved: distinct lint meanings

**Status:** closed.

The Surf/Deep lint cleanup uses two different terms intentionally:

- **Allow** means the construct is accepted as normal project style.
  The lint should not report it, and no migration pressure exists.
- **Keep** means checked-in source may remain as-is for compatibility,
  fixture coverage, or baseline preservation, but the construct is not
  preferred style for new human-authored source. A keep decision may
  still produce an advisory warning.

Keep decisions are not blocking-rule exceptions. Blocking-rule
exceptions still require an explicit rule id and a cross-reference to a
section of this spec. Advisory rules do not need path-glob exceptions
for existing corpus entries unless they are promoted into the blocking
registry later.

Inline source directives use the language's line-comment syntax:
`// chelis-lint: allow <rule>` or `// chelis-lint: keep <rule>` in
Surf/Rust-like files, `# chelis-lint: ...` in Python, and
`; chelis-lint: ...` in Deep. A directive suppresses or keeps the
diagnostic on the same line or the immediately following line. Deep
lint directive comments are ignored by the built-in style gate's
canonical-format comparison so the directive can suppress a lint
diagnostic without creating a formatting failure.

---

## 12. Enforcement

The `chelis lint` tool reads source files and reports violations of
the settled conventions in §§2–10 with `file:line` references, plus
the Deep user-symbol rule from §1.5 / §11.1, the def-arrow-form rule
from §3.5, and the pipe-stage interpretation from §3.6 where a rule
needs to reason about the principal argument. Blocking lint rules run
both as a standalone CI gate and as an implicit per-build gate:
`chelis build`, `chelis check`,
`chelis validate`, and `chelis eval --file` invoke `chelis fmt --check`
plus the relevant `chelis lint` rules on the input file before
running the front-end pipeline. Violations are blocking unless
explicitly waived in the style guide (e.g., the mdBook exception in
§8.5).

The lint is the persistent artifact: it prevents drift after a
cleanup pass. Without the lint, fixing today's outliers does not
prevent tomorrow's. CI invokes it directly via:

```
chelis lint --check
```

A successful run reports zero violations across every file in the
repo. A failing run lists the violations, each with the rule
reference and the file:line of the offending identifier or filename.

The build-time gate uses two escape hatches, neither of which CI may
use:

- `--allow-style-violations` — per-command opt-out emitted to stderr
  as a bypass warning. Reserved for emergency local builds and
  one-off migrations.
- `CHELIS_STYLE_GATE_DISABLE=1` — process-wide opt-out. Reserved for
  the integration-test corpus (tests that synthesize ad-hoc Surf to
  exercise type/effect/linearity behavior independently of style).

Severity behavior is part of the CLI contract:

| Severity | Source | User-facing output | Exit behavior |
|----------|--------|--------------------|---------------|
| Blocking violation | `registry::all_rules()` and formatter check | `path:line:col: rule_id (§ref): message`, or a formatter diagnostic | `chelis lint --check` exits nonzero; the built-in style gate blocks unless `--allow-style-violations` is passed |
| Warning/advisory | `registry::non_blocking_rules()` | prefixed with `warning:` or `advisory:` | never contributes to `chelis lint --check` failure; excluded from the built-in style gate |

`--allow-style-violations` bypasses only the built-in style gate. It
does not make parse, type, effect, validation, evaluation, or backend
errors non-fatal. Advisory warnings may still print on user-facing
commands after the gate is bypassed. `CHELIS_STYLE_GATE_DISABLE=1`
suppresses the gate and the fixture/test advisory pass and remains
reserved for tests.

`redundant-linearity-call` is advisory: `chelis lint` reports explicit
`copy()` and `drop()` source calls as warnings because implicit
linearity inserts equivalent IR nodes, and `chelis check` prints the
same warnings on user-facing runs. These warnings do not make
`chelis lint --check` fail.

Existing-corpus keep policy for `redundant-linearity-call`: checked-in
fixtures, migration examples, and baseline files may keep explicit
`copy()` or `drop()` when the call documents compatibility, preserves a
before/after baseline, or exercises legacy source behavior. New or
rewritten human-facing examples should use implicit linearity unless
the example is specifically teaching or testing the explicit forms. A
future promotion from advisory to blocking requires a separate cleanup
plan and updated docs before the registry changes.

`redundant-linearity-call` and `prefer-pipe-operator` do not expose
auto-fixes until the fixer can prove the rewrite preserves semantics.
For `copy()` / `drop()`, that proof requires the type and linearity
pipeline, not source-text matching. For pipe rewrites, that proof
requires knowing that the expression is a true first-argument dataflow
chain, not merely a call nested inside a sibling argument. Until that
semantic proof exists, `chelis lint --fix` must leave both warning
classes unchanged.

Exception entries inside the lint must carry a rule-id cross-reference
to a section of this document, not free-form prose. The schema:

```rust
struct Exception {
    pattern: GlobPattern,        // file or identifier glob the exception applies to
    rule_id: RuleId,             // the rule being waived
    cross_ref: SectionRef,       // section in spec/01-nomenclature.md that explains why
}
```

`cross_ref` is mandatory; entries without a resolvable spec section
are build-time errors. This discipline closes the loophole that lets
post-hoc justifications accrete in the lint config: every waiver has
to point at a documented rule that explicitly carves out the case.

### 12.1 Opaque Domain Construction

Types marked with `opaque: true` metadata participate in a
verified-constructor discipline. Outside the defining module, code must
obtain values of that type through exported constructor functions rather
than materializing the representation directly.

The AUTHORITATIVE gate is the type checker: opacity is enforced during
inference as `CheckErrorKind::OpaqueTypeViolation`, covering
construction, constructor references, pattern inspection, field
access, record update, casts, literal ascription, and out-of-module
references to unexported producer bindings. The full rejection set and
module-identity rules are normative in `spec/04-type-system.md` §2.5.

The lint rule `opaque-domain-construction` is kept as
defense-in-depth: per-file, no type context, fast editor/agent
feedback ahead of a full check, and its fail-closed
untyped-`record-update` arm complements the checker's deferred-target
ledger. The lint is fast feedback; the typing judgment is the
guarantee. The blocking lint rule rejects:

- Surf record construction of a marked ADT outside the defining module.
- Deep `record` construction of a marked ADT outside the defining module.
- Deep or Surf casts whose target is a marked domain type.
- Deep `record-update` when the node or updated value carries type
  metadata naming a marked domain type.
- Untyped Deep `record-update` outside every marked type's defining
  module when a marked type is in scope. This is fail-closed: without
  type metadata the rule cannot verify that the update is not
  materializing an opaque domain value.

The lint rule is a per-file construction-discipline gate layered
under the checker's constructor privacy. Downstream reports may cite
"checker-enforced opaque types plus lint defense-in-depth"; the
checker-level guarantee is the one specified in
`spec/04-type-system.md` §2.5.

Both layers deliberately allow direct construction inside the
defining module so the module can implement and prove its smart
constructors. Outside the defining module the checker rejects
constructor patterns and field access on the opaque type itself;
matching with irrefutable patterns and calling exported readers remain
available.

An opaque type may carry one declared invariant
(`@invariant(<binder>) <expr>`), extending the proven-constructor
discipline: a value of the type carries not only the provenance
guarantee (it originated inside the defining module) but the declared
boolean property of its representation. The invariant is recorded as
Deep metadata and checked for well-formedness at declaration time
(`spec/04-type-system.md` §2.5.1); the everyday checker never evaluates
it. The proven-constructor discipline is *discharged*, not merely
declared: `chelis prove` derives one obligation per exported producer of
the type (`opaque_invariants_rfc.md` D-PRODUCER / D-OBLIG) and proves —
at the SMT tier where the constructor lowers, otherwise by validated
sampling — that every produced value establishes the invariant. The
producer set is covered-or-rejected: an exported producer whose result
reaches the type through an unsupported container, or whose signature
hands caller-supplied code an unobligated value, is a declaration error,
so the discipline has no silent gaps.
If producer-set analysis cannot complete, the proof request SHALL report an
error; traversal exhaustion SHALL NOT mean that a type carries no obligation.

Advisory (non-blocking) lint rules support the invariant workflow:

- `opaque-without-invariant` (note): an opaque type with no declared
  `@invariant` carries only the construction guarantee; adding one lets
  `chelis prove` derive producer obligations.
- `invariant-float-equality`: exact `==` over a representation field in
  an invariant starves generation by design; the documented idiom is a
  tolerance band over a module constant.
- `unreachable-producer` (note): an opaque type with no exported
  producers is fully sealed and has an empty obligation set; the lint
  flags the dead declaration so the author either exports a producer or
  removes the type.
- `opaque-escape-site`: enumerates every in-module argument-egress site
  of a value of the type (or a function value capable of producing it)
  passed to an out-of-module callee, labelled by local provenance — a
  `note` when the value traces to a producer call or a type-T input of
  the enclosing function (locally attested), a `warning` when it traces
  to a raw construction or representation update (unattested). This is
  the audit surface for the explicit argument-egress trust caveat
  (`opaque_invariants_rfc.md` D-SOUND / D-LINT): return-egress is
  mechanically obligated, argument-egress is module-audited.

The authoritative design record is
`spec/design/opaque_invariants_rfc.md`.

### 12.2 Lint traversal exclusions

Whole-tree exclusions are stronger than diagnostic exceptions: a matched
nested path is pruned before classification, rule preparation, or rule
checks, so no rule sees it. They are permitted only for infrastructure,
build output, dependencies, generated artifacts, or immutable inputs that
should not be part of the editable lint corpus. They must not be added merely
to hide current violations.

A reef package's declared source roots are outside the shipped baseline's
reach. The baseline names directories that are infrastructure, build output
or vendored dependencies in a repository at large; beneath a package's `src`
or an `additional_sources` root the same names are source, because the reef
loader reads every `.ch` file there whatever the directory is called. Pruning
one would make the lint disagree with the build about which files the package
contains, and the disagreement is silent: no rule sees a file that is
compiled. A repository `chelis-lint.toml` entry is a deliberate local
declaration carrying its own cross-reference, so it continues to prune inside
a source root — excluding generated `.ch` is exactly what that mechanism is
for.

`chelis-lint` composes its shipped baseline policy with the nearest ancestor
`chelis-lint.toml`. Nearest-ancestor discovery resolves a relative lint
target against the invocation working directory before walking ancestors, so
relative and absolute spellings of the same target discover the same policy.
Repository patterns are gitignore-style and anchored to
the directory containing that file. The schema is versioned and strict:

```toml
version = 1
spec = "spec/01-nomenclature.md"

[[exclude]]
pattern = "path/to/generated/"
class = "generated"
cross_ref = "§12.2"
```

Every entry requires a pattern, a class from `infrastructure`, `build`,
`dependency`, `generated`, or `immutable`, and a cross-reference resolving
in the declared spec. Unknown fields, unsupported versions or classes,
invalid patterns, missing specs, and unresolved references fail lint before
traversal. Policy and spec paths are resolved before use: non-file or broken
policy paths and links escaping the policy root fail closed, while links that
remain inside the policy root are allowed. Loose targets without repository policy
receive only the shipped baseline.

The traversal engine must not consult `.gitignore`, `.ignore`, parent or
global Git configuration, `.git/info/exclude`, or hidden-file defaults. This
keeps local and CI scope identical and keeps hidden source such as
`.github/workflows/` visible unless Chelis policy explicitly excludes it.
An explicitly named file or directory overrides exclusion matching at
traversal depth zero; it must still be a regular file or directory (or a link
resolving to one) inside the policy root. An explicitly named root that
exists but fails that admission — a socket, FIFO, device, or other
non-regular entry, a link resolving to a different entry kind or outside the
policy root, or an unresolvable link — fails the lint invocation loudly with
the root path and rejection reason, matching the nonexistent-root failure; it
never produces a successful empty result. Only discovered (non-explicit)
inadmissible entries are silently omitted. Separately excluded descendants under
an explicit directory remain pruned. Non-explicit discovered entries must be
directories, regular files, or symlinks that resolve to the same entry kind
inside the policy root. Sockets, FIFOs, devices, and other special entries or
targets are omitted before a rule can open or read them. A symlink's target and
governed parents must remain policy-admitted. Broken links, links escaping the
policy root, and aliases into excluded content are omitted before any rule can
read them. Internal links to admitted regular files remain visible and are
classified by the link path.
Under an explicitly named excluded directory, that root's exclusion remains
overridden for an internal symlink target while separately excluded
descendants remain effective.

Traversal policy does not replace rule-specific `Exception` entries or inline
`allow` and `keep` directives. Those mechanisms act after a path has entered
the canonical corpus and retain their existing per-rule diagnostic or autofix
semantics. Rule-side catalogs and ancillary metadata must derive from the same
canonical entry set or pass a parent-aware traversal-policy admission check;
content under an excluded directory must not change an admitted entry's
verdict indirectly. When repository policy exists, its root bounds ancillary
workspace discovery: admitted sibling workspace manifests remain visible when
lint targets a subdirectory or explicit file, while machine-local ancestors
above the policy root cannot grant lint exceptions. A workspace crate entry's
kind is determined from its resolved metadata, so an internal symlinked crate
directory remains visible while a directory link resolving outside the policy
root is rejected. Governance follows the ancillary link path as well as its
resolved target: a link above the policy root remains machine-local even when
it points to an admitted file inside the root.

### 12.3 Recursive list cursor

`recursive-list-cursor` is advisory. It reports a self-recursive
definition whose recursive call advances one of that definition's own
parameters `p` through `skip`, in the argument position `p` itself
occupies in the parameter list.

That shape is a cursor walking a List one step at a time, and `skip`
returns a new List rather than a view into the old one, so the walk
allocates a fresh List per step and costs time quadratic in the List's
length. The linear form is a `fold` or `map` over the whole List, which
visits each element once. `Std.Io.Csv`'s row parser records the
rewrite; the general shape is to carry a scalar accumulator in a `fold`
and produce the result List with a single `map`.

The recursive argument advances `p` in either of two spellings. It may
be `skip(p, k)` written in the argument position itself, or it may be a
bare name that a local binding in scope at the call binds to
`skip(p, k)`:

```chelis
def walk(xs: List[i64], out: List[i64]) -> List[i64] = {
  rest = skip(xs, 1i64)
  if eq(len(xs), 0i64) then out else walk(rest, append(out, index(xs, 0i64)))
}
```

Both spellings are one defect and both are reported. Substitution is
one level deep and purely syntactic: the bound value must itself be a
direct `skip` of a parameter, so a name bound to another name resolves
no further. A binding is in scope for the bindings that follow it and
for its own block's body, and nowhere else.

The rule is deliberately narrow, and reports neither of these:

- `skip(xs, k)` outside a self-recursive call, including the single
  tail pass `fold(f, index(xs, 0), skip(xs, 1))` and a `skip` handed to
  a *different* function. One `skip` copies one List once.
- a self-recursive call whose cursor reaches the recursive argument
  through a call to another function, as in
  `apply_all(merge_once(tokens, skip(tokens, 1)), rules)`.

The second exclusion differs in kind from the first. Whether that call
advances a cursor depends on whether `merge_once` returns a suffix of
its argument, which is a fact about a different definition. A rule
reading one definition's syntax has no way to establish it, so
covering the form would mean giving the rule a different input rather
than a deeper substitution.

The rule also stays silent wherever its own reading would be a guess,
in three ways that are worth keeping distinct. A parameter the body
rebinds is no longer the value a recursive call walks, so that
parameter is not a cursor; the definition's other parameters are still
read, and a cursor among them is still reported. A definition that
rebinds its own name is skipped entirely, because its recursive calls
can no longer be identified by that name. And a name the body binds
more than once is never substituted. Each case errs toward silence,
which is the right direction for an advisory rule. Advisory severity
follows from a further limit: a ten-element List is not a defect, and
the rule cannot know the length.

`skip` is [05-OP-54]'s prefix removal. The one-argument `drop` is
[05-OP-67]'s linearity consume and is never this rule's subject.

### 12.4 Future rule queue

The following rules are intentionally queued, not currently part of
the blocking registry:

- Markdown prose punctuation: extend `no-em-dash-in-public-strings`
  from source string literals to active docs after existing current
  docs have been cleaned. Fixes should rewrite prose, not add path
  exceptions.
- `redundant-linearity-call` promotion review: decide after the
  implicit-linearity migration corpus is stable whether advisory
  warnings should remain permanent or become blocking for new source.
- Pipe-stage shape checks: if future syntax or decompiler work creates
  ambiguity around `x |> f(y)`, add coverage that preserves the
  first-argument semantics in §3.6 rather than accepting last-argument
  insertion.

### 12.5 Source the lint cannot parse

**Rule:** A `.ch` file the lint cannot parse is itself a blocking
violation. `chelis lint --check` must not report success on a file whose
contents it was unable to judge.

Rules that need an abstract syntax tree obtain it by parsing, and a rule
that cannot parse its input reaches no verdict and reports nothing. That
deferral is correct per rule: a rule must not invent a diagnosis from a
broken parse. It is wrong in aggregate: silence from every rule is
indistinguishable from a clean file, so without this rule the command
would report success over input no rule could read.

Exactly one rule therefore owns the parse verdict and reports the
failure; every other rule keeps deferring. This is what makes the
per-rule deferral safe rather than silent.

The requirement is on the lint, not on the formatter. `chelis fmt
--check` claims that the file it is named on is canonically formatted,
and claims nothing about any other file; it therefore does not discharge
this requirement, whatever it reports.

---

## 13. References

- `spec/02-surf-syntax.md`: Surf grammar, module-path mapping,
  reserved words.
- `spec/03-deep-syntax.md`: Deep tag vocabulary, symbol grammar.
- `spec/design/chelis_canonical_reference.md`: runtime-vs-shell
  distinction.
- `crates/chelis-surf/src/lexer.rs`: Surf identifier grammar.
- `crates/chelis-deep/src/lexer.rs`: Deep symbol grammar.
- `crates/chelis-deep/src/validate.rs`: closed Deep tag vocabulary.
- `crates/chelis-deep/src/printer.rs`: Deep tag emission.
- `crates/chelis-surf/src/format.rs`: formatter behavior re:
  identifier case.
- `crates/chelis-backend-c/src/emit.rs`: backend symbol emission.
- `crates/chelis-lint/`: lint implementation.
- `docs/archive/snapshots/ecosystem_naming_snapshot.md`: empirical snapshot of the
  May 2026 ecosystem state and the cleanup inventory.
- the `example-corpus` skill (`agent-skills/example-corpus/SKILL.md`) §Writing Surf: Surf code-style guidance.

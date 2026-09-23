# Unrepresentable AST Domain — Design Doc

**Issue:** #908  
**Delivery:** Stamped ingress and dedicated compiler annotations are implemented; #908 remains open through legacy-carrier retirement and the integration audit.
**Prerequisite:** #855 (merged 2026-07-29)

## Class Statement

> A structural token (Name) at a RuntimeExpr child position is
> unrepresentable — the constructor rejects it, and no accessor can
> return it from an expression traversal.

This is the domain-restructuring answer (technique three) to the defect
class #908 names: the Deep AST admits states no consumer can honestly
handle, because `Atom` unions structural tokens with value literals in
the same child sequence, `List.elements` carries the tag where traversal
visits it as an expression, and `List::tag()` returns `Option` whose
`None` conflates three cases.

## Acceptance Criteria

1. A Name atom at a RuntimeExpr child position is unrepresentable.
2. Bypass slots with declared expectations (Module, Match, Record, etc.)
   get StampError on undecodable heads, closing #858.
3. Form-expecting slots (RuntimeExpr, Type, EffectHandler) with
   undecodable heads produce a distinct `UnknownForm` variant that the
   checker scores (preserves fitness gradient).
4. `BareList` explicit in type forces stated disposition for [04-TOT-2]
   (technique two — explicit, not impossible; #908 amended).
5. Pre-expansion forms (defmacro) never enter the Deep AST.
6. `Atom::Tag` and `Atom::Keyword` deleted.

## The Cut — What Becomes True

| Before | After |
|--------|-------|
| Tag in child sequence, visitable as expression | Head in Node's private field, not traversable |
| `List::tag() -> Option<DeepTag>` conflating three cases | `Node::tag() -> DeepTag` (total); `BareList` and `UnknownForm` distinct variants |
| Name at RuntimeExpr admitted by type, caught at check time | Rejected by constructor; no accessor can yield it |
| Untagged top-level list silently passes (#858) | Module bypass expects declaration vocabulary → StampError |
| `defmacro` embedded in AST, every consumer disposes of it | Transported as `MacroDef`, never enters `Expr` |

## API Surface (Closed List)

### Constructor Blame Rule

- `Node::try_new` (returns Result → diagnostic): violation means the
  **user** wrote something wrong. Used wherever construction input could
  be incorrect due to user text — stamp pass, desugar, macro expansion.
- `Node::new` (panics — validates always in all build modes): violation
  means the **compiler** is wrong. Used for passes rewriting
  already-validated trees (prune, AD, transforms).
- **Rule:** if a violation at this site would be a compiler bug, use
  `new`. If it could be a user error, use `try_new`. New construction
  sites inherit this rule.
- Note: a rewrite panic is a live risk (recombining valid children can
  produce invalid output via arity or role mismatch), not a formality.

### Ways to Obtain a Node

1. `Node::try_new` (boundary — returns `Result<Node, NodeError>`)
2. `Node::new` (internal — panics on violation, validates always)
3. Custom `Deserialize` routes through `try_new`, surfaces failure as
   `serde::de::Error` (caller can catch and rebuild)

No other path constructs a `Node`.

### Ways to Observe Children

- Role-typed accessors: `expr_children()`, `binder_names()`,
  `type_children()`, `selectors()`
- Total traversal: `children_iter() -> impl Iterator<Item = ChildRef>`
  (the default migration target for `children(list)`)
- Indexed: `expr_child(i)` panics on role mismatch (consumer bug)
- **No** `&[Expr]` accessor. **No** `&mut` child access.
- Rebuilds reconstruct through the constructor.

### Construction Failure Channels

- `try_new` returns `NodeError` (tag, index, found, expected) →
  surfaced as `StampError` (distinct kind from `ParseError`)
- `new` panics with the same diagnostic information
- Both validate in all build modes (not debug-only)

### Arity

`arity_contract(tag: DeepTag) -> AritySpec` total over DeepTag. Gives
the legal child count (Fixed, AtLeast, Range). `try_new` rejects wrong
child count.

### Deserialization

Deserialize into shadow struct → call `try_new` → surface failure as
`D::Error`. The cache layer can catch and rebuild.

### Deny-Lint

`clippy::wildcard_enum_match_arm` activated per-crate in each migration
commit.

## Stamp Rule (Top-Down, Role-Directed)

The stamp pass converts `Vec<RawExpr>` → `Result<Vec<Expr>, StampError>`
via a top-down walk. The expectation comes from the **slot**, not from
inspecting element zero.

### Per-Role Stamp Table

| Role | Name atom? | List at this slot? |
|------|-----------|-------------------|
| RuntimeExpr | **StampError** | Vocabulary head → Node. Undecodable head → **`UnknownForm`** (checker rung, scored) |
| Type | Permitted | Vocabulary head → Node. Undecodable head → **StampError** (`UndecodableTypeHead`; see the 2026-08-24 amendment) |
| EffectHandler | Permitted | Vocabulary head → Node. Undecodable head → **`UnknownForm`** (form-expecting) |
| Syntax | Permitted | Known tag head + metadata map → Node ([03-ROLE-3]). Any other list → **BareList** |
| Binder | Permitted | Known tag head + metadata map → Node ([03-ROLE-3]). Any other list → **BareList** |
| Selector | Permitted | Known tag head + metadata map → Node ([03-ROLE-3]). Any other list → **BareList** |
| Bypass | Permitted | **Per-tag child expectation** (see below) |

### Strict/Lenient Reasoning Per Row

- **RuntimeExpr, EffectHandler** — lenient (→ UnknownForm). A typo'd
  head does not produce a vacuous pass; the checker scores it and the
  rest of the program is assessed. Leniency preserves the fitness
  gradient.
- **Type** — strict as shipped (→ StampError `UndecodableTypeHead`).
  This diverges from the original lenient row; the 2026-08-24 amendment
  records why the shipped rule stands and the table was corrected
  rather than the code.
- **Bypass with declared expectation** — strict (→ StampError). A wrong
  child in these slots gets **skipped** rather than scored — the walker
  doesn't enter it and the invariant is satisfied vacuously. That is
  #858, and hard rejection is the only disposition that prevents it.
- **Syntax/Binder/Selector** — structural, with vocabulary decode. A
  list whose head is a known tag word AND whose element 1 is a metadata
  map is a vocabulary node wherever it sits and decodes to Node;
  [03-ROLE-3]'s conjunction is the disambiguator, at any depth. Every
  other list stays BareList, because each half of the conjunction alone
  is legitimate structural data: a tag-word head without a map is an
  import name list (`(copy fill)`), and a map behind an ordinary name
  head is an annotated parameter (`(x {type: ...})`). Neither half may
  reinterpret the list on its own.

### Per-Tag Bypass Child Expectation

`bypass_child_expectation(tag, index) -> BypassExpectation` — total over
(tag, index) pairs where the role is Bypass.

Allowed-tag sets derived from **total classifiers** (exhaustive matches
with deny-lint active) rather than literal enumerations:

```rust
pub fn is_declaration_tag(tag: DeepTag) -> bool { /* exhaustive */ }
pub fn is_pattern_tag(tag: DeepTag) -> bool { /* exhaustive */ }
```

| Tag | Bypass children | Expectation |
|-----|----------------|-------------|
| Module(1+) | Declarations | `RequiresVocabulary(is_declaration_tag)` → StampError |
| Match(1+) | Arms | `RequiresVocabulary(Arm)` → StampError |
| Record(1+), RecordUpdate(1+) | Kvs | `RequiresVocabulary(Kv)` → StampError |
| Pipe(1+) | Stages | `FormExpecting` → UnknownForm |
| Let(0) | Bind | `RequiresVocabulary(Bind)` → StampError |
| Arm(0) | Pattern | `RequiresVocabulary(is_pattern_tag)` → StampError |
| PatCtor(1+), PatRecord(1+), PatTuple, PatAs(1+) | Sub-patterns | `RequiresVocabulary(is_pattern_tag)` → StampError |
| Kv(1) | Value | `FormExpecting` → UnknownForm |

Adding a new declaration or pattern tag breaks the build at the
classifier (deny-lint enforced exhaustive match).

## AST Shape

### Raw Stage (Parser Output)

```rust
pub enum RawExpr {
    Atom(RawAtom, Span),
    List(Vec<RawExpr>, Span),
    Map(Vec<(String, RawExpr)>, Span),
    MetaExpr { entries: Vec<(String, RawExpr)>, expr: Box<RawExpr>, span: Span },
}

pub enum RawAtom {
    Symbol(String),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}
// No Keyword (map keys are String; bare :kw is parse error)
// No Tag (decode happens at stamp, not parse)
```

### Stamped Stage (Post `stamp_to_typed`)

```rust
pub enum Expr {
    Atom(Atom, Span),
    Node(Node, Span),
    BareList(Vec<Expr>, Span),
    UnknownForm { head: String, meta: Metadata, children: Vec<Expr>, span: Span },
    Map(Metadata, Span),
    MetaExpr(MetaExpr, Span),
}

pub enum Atom {
    Name(String),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

pub enum ListHead { Node(DeepTag), Bare }

pub enum ChildRef<'a> {
    Expr(&'a Expr),
    Binder(&'a str),
    Selector(&'a str),
    Syntax(&'a Expr),
    Type(&'a Expr),
    EffectHandler(&'a Expr),
    Bypass(&'a Expr),
}

pub struct Node { /* private: tag: DeepTag, meta: Metadata, children: Vec<Expr> */ }
```

### Staging (Minimal — Technique C)

The parser emits `RawExpr`. The stamp pass converts to `Expr`. `RawExpr`
is kept permanently (the design requires it). This is technique C in
minimal form: two representations parameterized by stage.

## Pre-Expansion

Macros are **Surf-only** (confirmed: `.dp` strict parse rejects
`defmacro`; no `.dp` path runs expansion).

- Desugar returns `(Vec<Expr>, Vec<MacroDef>)`
- `expand_program(exprs: &[Expr], macros: &[MacroDef], options: &...)`
- Template placeholders are ordinary `(var {} param)` nodes (confirmed
  from live test `simple_macro_expands_to_base_tags_with_source_metadata`)
- Macro invocations are ordinary `App` nodes (live path recognizes
  `(app {} (var {} macroname) args...)`)
- `macro-invoke` dead (consumer at lib.rs:304, no producer; live tests
  use App path; confirm branch unexercised before deletion)
- Template body stamp entry: RuntimeExpr expected role.
  `arity_contract` checked against template as written.

```rust
pub struct MacroDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: Expr,
}
```

## Explicit Rejections

- **BareList + name-check for pre-expansion:** rejected. Reintroduces
  head-in-child-sequence dispatch by name inspection. Evidence:
  `macro-invoke`, `vmap-grad`, `drop`, `list` are dead (no producer).
- **`Node` unioning vocabulary + internal tags:** rejected. Creates
  `Option`-shaped disposition for every downstream consumer and
  permanently admits pre-expansion forms into post-expansion tree.
- **Content-inspection for head classification:** rejected. Classifying
  by element zero's value type reintroduces the defect at the
  classification layer.

## Scope Limitations

- The guarantee covers RuntimeExpr slots only for Name-atom rejection.
  Name is legitimate at Syntax, Binder, Selector, Type, EffectHandler,
  Bypass positions.
- BareList interiors are not role-typed.
- Bypass child that is itself a Node was validated at construction;
  validation recurses through Nodes. The genuine gap: Atom or BareList
  directly in a bypass slot — those are not role-typed by the parent.
- [04-TOT-2] addressed by technique two (BareList explicit, every match
  forces stated disposition), not technique three. Issue #908 amended.
- `infer_atom` Name arm: **kept as diagnostic** (#881). Not
  `unreachable!()`. Bypass interiors can route a Name to `infer_expr`
  via owning traversals that do not filter by role.
- `assert_decode_once_at_boundary`: **deleted**. The type replaced it
  (no representable input can fail it post-migration).

## Oracle Contract

1. **Headline criterion:** Name at RuntimeExpr slot → StampError
2. **#858 criterion:** misspelled tag at Module child → StampError
3. **#858 negative control:** legitimate params list and import list
   stamp clean
4. **Fitness gradient:** program with one unknown-head expression gets a
   scored result (UnknownForm → checker), not stamp failure
5. **Keyword:** bare `:kw` in expression position → ParseError
6. **Score-one control:** programs with Name at structural positions
   score 1.0

Mechanisms:
- trybuild (compile-fail): closed API surface (no struct literal, no raw
  child access)
- deny-lint: exhaustiveness (per-crate, per migration commit)
- Python oracle: behavioral (parse-rejection, stamp-rejection, scoring)

## Keyword Ladder Argument

Per-case enumeration of corpus members that move from check rung to
parse rung (to be filled during implementation):

| Corpus case | Old rung | New rung | Reason |
|-------------|----------|----------|--------|
| `dp_bare_keyword_body_no_defsig` | check | parse | keyword has no expression semantics; parse is earliest honest rejection |
| `dp_bare_keyword_body_unit_defsig` | check | parse | same |
| `dp_bare_keyword_toplevel_def` | check | parse | same |
| `dp_bare_keyword_unused_let_binding` | check | parse | same |

## Symbol Ladder Argument

The three bare-symbol corpus members moved rungs the same way when the
role-directed stamp landed ([03-ROLE-2]); recorded per case beside the
keyword table, and locked by the corpus test
`bare_atom_expression_position_rejects_at_parse`:

| Corpus case | Old rung | New rung | Reason |
|-------------|----------|----------|--------|
| `dp_bare_symbol_body_no_defsig` | check | parse (ingress) | a name is not an expression; the stamp rejection identifies the name and the `(var {} ...)` remediation |
| `dp_bare_symbol_body_unit_defsig` | check | parse (ingress) | same |
| `dp_bare_symbol_toplevel_def` | check | parse (ingress) | same |

## Unknown-Head Ladder Argument

Per-case enumeration of programs with unrecognized heads — fitness
consequence stated (to be filled during implementation):

| Case | Old behavior | New behavior | Fitness consequence |
|------|-------------|--------------|---------------------|
| Typo'd expression head (e.g. `apl` for `app`) | lenient parse + checker diagnosis, scored | `UnknownForm` + checker diagnosis, scored | **Unchanged** — same rung, same gradient |
| Typo'd declaration in Module | lenient parse, silently skipped (#858) | **StampError** | Hard rejection. Correct: silent skip was vacuous pass |
| Typo'd arm in Match | lenient parse, silently skipped | **StampError** | Hard rejection. Correct: same as Module |
| Typo'd type tag | checker diagnosis (unknown type tag) | `UnknownForm` + type resolver diagnosis | **Unchanged** — same rung |

## Task Breakdown

### Task 1: Design Doc — Constraining Sections

This file. All design forks resolved before implementation.

### Task 2: Move `child_stamp_role` to chelis-deep; write `arity_contract`; write `bypass_child_expectation`

- Move `ChildStampRole` and `child_stamp_role` to `chelis-deep/src/role.rs`
- Write `arity_contract(tag) -> AritySpec` total over DeepTag
- Write `bypass_child_expectation(tag, index) -> BypassExpectation` total
  over bypass (tag, index) pairs
- Write `is_declaration_tag`, `is_pattern_tag` classifiers (exhaustive,
  deny-lint)
- Completeness tests for all

**Demo:** `cargo nextest run -p chelis-deep -p chelis-types` green.

### Task 3: Introduce `RawExpr`, `Node`, `Expr::Node`, `Expr::BareList`, `Expr::UnknownForm`, `ChildRef`, custom Deserialize, trybuild

- `RawExpr`/`RawAtom` in `chelis-deep/src/raw.rs` (no Keyword, no Tag)
- `Node::try_new`: validates arity + per-role (Name rejected at
  RuntimeExpr only; permitted elsewhere)
- `Node::new`: same validation, panics (blame: compiler bug)
- Role-typed accessors, `ChildRef`, total iterator
- Custom Deserialize → shadow → try_new → D::Error
- trybuild: closed API surface (no struct literal, no raw access)
- Existing `Expr::node()` sites confirmed internal → `Node::new`

**Demo:** `cargo nextest run -p chelis-deep` green.

### Task 4: Remove `Atom::Keyword`; make bare `:kw` a parse error

- Remove variant; parser rejects outside metadata
- Ladder argument per corpus case (table above)
- Tag deletion deferred to Task 5

**Demo:** `cargo nextest run -p chelis-deep` green.

### Task 5: Rename `Atom::Symbol` → `Atom::Name`; delete `Atom::Tag`; retarget parser to `RawExpr`; write `stamp_to_typed`; extract defmacro

- Rename variant, delete `Atom::Tag`
- Retarget parser: `parse_str` deleted. New entry points:
  - `parse_raw(source) -> Result<Vec<RawExpr>, ParseError>`
  - `parse_and_stamp(source) -> Result<Vec<Expr>, StampError>`
- Write `stamp_to_typed(exprs: Vec<RawExpr>) -> Result<Vec<Expr>, StampError>`
  implementing the per-role table above
- Stamp-pass conversion: `RawExpr::MetaExpr` → `Expr::MetaExpr`,
  `RawExpr::Map` → `Expr::Map(Metadata)`
- `StampError` distinct from `ParseError`
- Desugar: `MacroDef` as separate return; uses `try_new`
- `expand_program` new signature
- Desugar's `sym()` → `name()`
- Unknown-head ladder per corpus case (table above)

**Tests:**
- Params list `(x y z)` at Binder → BareList (not error)
- Name at RuntimeExpr → StampError
- Unknown head at RuntimeExpr → UnknownForm (not StampError)
- Misspelled declaration in Module → StampError (#858)
- Name at Type slot → permitted
- Print/parse roundtrip with params list passes
- Desugar/expansion tests pass

**Demo:** `cargo nextest run -p chelis-deep -p chelis-surf -p chelis-macros` green.

### Task 6: Migrate chelis-effects and chelis-types

- `children(list)` → `children_iter()` (total, per-role disposition)
- `expr_children()` only where confirmed expression-only
- `infer_atom` Name arm kept as diagnostic
- Effects: `effect_set_expr` uses `Node::new` (internal — builds from
  EffectSet enum, not user text)
- Deny-lint per-crate

**Demo:** `cargo nextest run -p chelis-effects -p chelis-types` green.

### Task 7: Migrate chelis-ir, chelis-compiler-api, chelis-prove; remove dead guards

- Same migration pattern
- Remove dead guards (`vmap-grad` 4 sites, `macro-invoke` 1 site) —
  confirm unexercised before deletion
- Delete `assert_decode_once_at_boundary`
- Deny-lint per-crate

**Demo:** All green.

### Task 8: Migrate remaining crates; remove `Expr::List`

- Backends, lint, tide, reef, cli, e2e — per-crate commits
- Delete `Expr::List`, `struct List`, all deprecated accessors
- `RawExpr` kept (the design requires it)
- Cache format identity updated
- Deny-lint per-crate

**Demo:** `cargo nextest run --workspace --profile ci` green.

### Task 9: Wire the Oracle

- Python `scripts/unrepresentable_domain_oracle.py` (behavioral) - the
  authoritative oracle, run by `scripts/gate.py`'s `integration` stage
  (`heavy-e2e.yml` nightly/manual `integration-support` worker) and its `--local` subset, and locked
  there by `scripts/test_gate.py`. Acceptance is exit 0 with a final
  `ORACLE: PASS` line.
- Python `scripts/test_unrepresentable_domain_oracle.py` (unit tests).
  These patch the command runners, so they are evidence about the script's
  decision logic and never a substitute for running the oracle.
- trybuild (Task 3): API surface
- deny-lint (Tasks 6-8): exhaustiveness

### Task 10: Retrospective; amend #908; PR; self-review; merge

- Append retrospective to this doc
- Amend #908 issue ([04-TOT-2] technique two)
- `gh pr create`
- Fresh-context self-review against constraining sections
- Fix, push, merge on green

## Evidence Records

- `macro-invoke` dead: consumer at `chelis-macros/src/lib.rs:304`, no
  producer found. Live test `simple_macro_expands_to_base_tags_with_source_metadata`
  uses the `App`-recognition path. To be confirmed by branch-panic test.
- `vmap-grad` dead: 4 consumers in `host.rs`, no construction site.
- `drop` (internal tag) dead: 1 consumer in `infer.rs`, no producer.
- `list` (internal tag) dead: 2 consumers in `infer.rs`, no producer.
- Macros Surf-only: `.dp` path uses `parse_str_strict` which rejects
  unknown tags. No `.dp` path runs expansion.
- Template placeholders: `(var {} param)` nodes (confirmed from
  `substitute_expr` in expander — matches Var by name).
- `effect_set_expr` internal: builds from `EffectSet` enum (closed,
  compiler-computed). Effect names are fixed strings at Syntax slots.
- `child_stamp_role` has no type-crate dependencies: pure function over
  `(DeepTag, usize, usize)`.

## Amendments

**2026-08-24 (chelis#1088), Task 7's compiler-api half and one stale
evidence line.** Every public Deep text boundary in `chelis-compiler-api`
now consumes the role-stamped carrier, each field stamped in the role it
occupies: `parse_and_stamp_file` for a `.dp` program (the generic
`parse`/`check`/`decompile` door, the `prepare_source` pipeline door, and
every authoring `module` field), `parse_and_stamp` for a declaration
bundle, `parse_and_stamp_runtime_exprs` for a replacement body, and
`parse_and_stamp_tagged` for a field whose contract names one tag. The
named non-compiler-API readers moved with it
(`chelis-validate::validate_deep`, the `opaque-domain-construction` lint
rule, `chelis-e2e`'s snippet checker, `chelis-lsp`'s Deep document
analysis, and the `chelis-cli` style-gate fallback), so
`parse_str`/`parse_str_strict` are test-only spellings in the workspace.

The top-level rule this enforces is decided in the numbered spec, not
here: `spec/03-deep-syntax.md` §7.1 [03-PROG-1] enumerates the admissible
top-level forms, and [03-PROG-2] states the rejection contract. Chapter
03's PEG previously read `Program <- Spacing Node+ EOF`, which imposed no
top-level role restriction at all.

**The oracle is `scripts/unrepresentable_domain_oracle.py`**, wired into
`scripts/gate.py`'s `integration` stage and its `--local` pre-push subset.
Hosted CI runs that stage in the `integration-support` matrix of
`heavy-e2e.yml`, daily at 03:17 UTC and on manual dispatch. Ordinary PRs
do not run this exhaustive oracle.
That stage rather than `lint-and-unit` because two obligations run `cargo
nextest`, which the `lint-rust` worker deliberately does not install.
Acceptance is exit 0 with a final `ORACLE: PASS` line. Its obligations cover the compiler-API ingress:
obligation 3 drives [03-PROG-1] and [03-PROG-2] through the built `chelis`
binary over `.dp` fixtures, and obligation 5 executes the compiled
`crates/chelis-compiler-api/tests/phase3_stamped_ingress.rs` parity suite.
That Rust suite is evidence the oracle executes, not a second oracle;
`scripts/test_gate.py` locks the wiring so a future CI edit cannot drop
it silently.

The Evidence Record line "Macros Surf-only: `.dp` path uses
`parse_str_strict` which rejects unknown tags" states a true conclusion on
a reason that no longer holds, and did not hold for the pipeline door even
when it was written: the compile path's `.dp` ingress ran no
tag-vocabulary sweep. The conclusion stands for a different reason. `.dp`
ingress produces `PreparedProgram` without invoking
`chelis_macros::expand_program`, which only `prepare_surf_decls` calls, so
no `.dp` path runs expansion regardless of what its parser accepts. An
unknown head below a declaration is deliberately preserved as
`Expr::UnknownForm` on the generic parse door so the wire AST keeps its
identity and the checker owns the rejection; the authoring doors keep the
vocabulary sweep, so an unknown head still cannot reach a rewriter.

**2026-08-24 (chelis#885 / chelis#1023 §B row 1): the Type row diverged,
and the shipped strict rule stands.** The Per-Role table originally made an
undecodable head at a Type slot lenient (`UnknownForm`, "type resolver
diagnoses"). The implementation shipped strict: `stamp_type` rejects an
undecodable non-empty list head with `StampErrorKind::UndecodableTypeHead`
at ingress. The strict rule is the right one, and the table above now
records it: type syntax is a closed vocabulary with no user-extensible
heads, so unlike an expression head there is no later consumer whose
scoring the leniency would preserve — the type resolver's only possible
verdict on an unknown head is the same rejection, later and with less
location context. The table row and the reasoning bullet were corrected
rather than the code.

The same change set recorded four related decisions:

- **The Keyword design choice is RESOLVED.** #885's Form left "Keyword
  either becomes metadata-position-only or joins Name" open. `Atom::Keyword`
  is deleted; a colon-prefixed keyword is metadata-key syntax only,
  rejected by the parser outside metadata-key position (spec/03 §8.1). It
  did not join `Name`.
- **`Atom::Tag` deletion (Task 5 names it) is DEFERRED to chelis#1029**
  (blocked on chelis#1082): `Node::to_list` constructs it, 39 production
  `.to_list(` call sites and three normalize bridges consume it, so the
  deletion belongs to the same atomic change that removes `Expr::List`
  (#1023 §B row 5), not to the #885 contract slice. That change landed with
  chelis#1125's completion (decision row 23 of
  `spec/design/checker_totality.md`): `Atom::Tag`, `Expr::List`, `List`,
  `Node::to_list`, and every normalize bridge were removed together.
- **`spec/03-deep-syntax.md` §7.2 now owns the role contract** as
  numbered-spec text: [03-ROLE-1] (total (tag, index) → role
  classification; a bare identifier at a structural/type/effect-handler
  position is a name even when it spells a tag), [03-ROLE-2] (a bare
  identifier at an expression position rejects at ingress, identified per
  [03-PROG-2] with the `(var {} ...)` remediation), and [03-ROLE-3] (the
  metadata-map-at-element-1 disambiguator for lists at structural
  positions — neither the head alone nor the map alone reinterprets a
  structural list). The Per-Role Stamp Table above states the same rule;
  its Syntax/Binder/Selector rows shipped still reading "BareList without
  head decode" and were corrected to the decode conjunction in the PR
  #1319 review round.
- **The continuous oracle gained obligation 7** (`NAME_IN_EXPR_FIXTURES`):
  a bare identifier at def-body / fn-body / app-argument / bind-RHS
  positions rejects with the [03-ROLE-2] identification and remediation
  spelling, and the structural-name score-one controls extend to
  record-head / kv-key / access-field / export / deftype positions as the
  over-application guard. This closes #885's demand that its oracle not
  repeat the manual scratch-variant shape.

(Further divergences to be recorded here.)

## Metadata value domain (#1478, #1330, #1567)

`spec/03-deep-syntax.md` [03-META-1/2] owns annotation shapes, roles and
uniqueness. The AST represents each of the 30 compiler-owned keys with a
dedicated payload, and keeps producer extensions in a separate map. Text and
serde ingress decode raw entries once; consumers use typed getters and
role-aware traversal. Raw syntax views belong to the private codec and its
shape validator, not to downstream compiler APIs.

Payload construction enforces local shape. Node construction, deserialization
and transactional replacement enforce placement on the owning node. Complete
program boundaries additionally check parent/sibling relationships, including
Surf dimension groups. The legacy `Expr::List` structure these boundaries
once also had to defend against no longer exists (#1029, retired with #1125).
Runtime and type wrappers validate recursive roles, including children inside
otherwise validated nodes. Semantic type, effect and binder resolution
remains with the owning checkers: a well-shaped annotation does not establish
semantic agreement or make a producer's claim trustworthy.

The checker validates annotation placement before inference across direct,
library and context entry points. This produces one shape/placement diagnostic
before unrelated semantic failures. Tests that formerly forged invalid
metadata now exercise parser, constructor or deserializer rejection; checker
placement and ordinary legacy child-role attacks remain independent controls.

Surf declaration validation rejects repeated dtype bounds and conflicting
bound ownership before desugaring. The public round-trip normalizer returns
`Result` and validates its input before rewriting. Normalization preserves
bound-family distinctions and quantifier annotations, including present empty
containers, and removes derived annotations inside structural containers
through a separate annotation callback. It never submits those container
roots to a generic expression rewrite.

This is one shippable slice because changing the AST storage without migrating
all producers, readers, transforms and serialization boundaries together would
leave either uncompilable consumers or a second permissive annotation API.
It resolves the metadata instances above, not every remaining #908 obligation.

### Dedicated AST annotations and extensions

The [03-META-1/2] annotation contract uses private, sparse `Metadata` storage.
Each compiler-owned entry is a closed variant with a dedicated payload; its
variant determines its key. `ExtensionMap` is separate and rejects compiler
keys and the closed `surf_*` namespace. Default empty metadata allocates no storage.
Insertion rejects an existing key; replacement is an explicit operation.

Structural payloads own their structure: quantifiers contain binders,
preconditions contain runtime expressions, contracts contain strings, effect
sets contain names or resource strings, and dtype bounds contain distinct
binder names and family choices. Their container annotations and spans survive
transformations, including present-but-empty containers. Variable targets
preserve single-variable versus one-element-tuple spelling. Genuine runtime
and type payloads use immutable validated expression wrappers with fallible
reconstruction. Preserved macro source owns raw syntax data.

`Node`, `UnknownForm`, map expressions, prefix metadata and metadata inside
legacy lists all carry the same typed representation. Private text and serde
codecs retain the external spellings and core serde entry shape, collecting raw
entries before rejecting duplicates and decoding their roles. No consumer
reconstructs or queries a registered entry with a string key. Role-aware
visitors distinguish runtime expressions, types, binders, structural members
and preserved source, while opaque producer extensions are excluded; structural roots cannot become unit literals.
Rebuilding a node replaces coupled children and annotations atomically.

This representation concerns AST annotations only. It neither depends on nor
absorbs checked tensor storage metadata (#889) or exact capacity identity
(#888). Lowering interface and shared-fixture changes require explicit review
within that boundary. General legacy-list retirement (#1029), the remainder
of the accessor migration (#1125), and macro hygiene remain separate work.

The authoritative oracle remains
`.venv/bin/python scripts/unrepresentable_domain_oracle.py`. Its typed metadata
obligation covers all 30 registered keys, duplicate extensions, constructor
and replacement admission, old/current serde ingress, source-data exceptions,
role-preserving transformations, compile-fail API/privacy controls with
positive companions, and a source guard against iterator readers that compare registered keys as strings.
The #1567 scalar and tensor duplicate-stamp witnesses must reject in both
orders before check, evaluation, compiled execution or resugaring can select
one stamp. Exact single-stamp controls retain their numerical results.


## Opaque producer extensions (#1637)

[03-META-3] in `spec/03-deep-syntax.md` owns the data grammar and preservation
contract. `ExtensionData` stores a validated flat token tree in canonical
syntax, with its diagnostic span. The representation contains neither `Expr`
nor metadata containers and has no semantic traversal API. Keeping the tree
flat makes cloning, printing, serialization and destruction independent of
nesting depth. Numeric token spellings are syntax, never runtime numbers.

Raw parsing captures extensions before stamping and typed-literal expansion.
The raw stage distinguishes data explicitly; stamping data as a program form
fails. The private AST serde codec and compiler API wire output tag data
explicitly. Old expression-valued extension checkpoints are rejected and the
compiled-context, stdlib and library cache identities advance. Compiler-owned
payloads retain their existing validation and role-aware transforms.
The auxiliary Deep PEG validator sees a projection replacing admitted data
ranges with inert scalars, preserving byte and line positions. It checks program
structure without adding a second, narrower data grammar.

Semantic visitors exclude extensions. Metadata rewrites preserve both their
own extensions and those of originating expression/structural owners.
`Expr::try_inherit_extensions` provides transactional replacement/combination;
`ExtensionMap::try_merge` coalesces identical payloads and diagnoses conflicts.
Macro substitution and expansion preserve template, argument and call-site
owners through this boundary. Deep normalization retains data. Surf emission
rejects extensions it cannot represent before returning output.

Property discovery uses canonical `chelis_role` and `property_source_kind`;
`c_earchin_role` is producer-owned data. Producers emitting both forms already
have the canonical fields. Repository witnesses migrate with the compiler;
legacy-only producers must emit the canonical property schema.

The same continuously wired Deep-domain oracle remains authoritative. Its
compiled selection includes opaque-data admission, preservation, owner
combination and Surf rejection, with positive and negative controls. Existing
recursive compiler-syntax witnesses use registered expression fields instead
of relying on executable extensions. The oracle also tests CLI check/validate
parity for nested payloads that resemble malformed program syntax.

This is one dependency-coupled delivery slice: storage, ingress, wire format
and consumers must agree when extensions cease to be expressions. It does not
close #908, retire legacy Lists (#1029), fix macro binder representation
(#1320), or own runtime tensor metadata (#889) or capacity identity (#888).

### Structural enforcement techniques

Constructor validation prevents admission of an invalid payload through a
fallible boundary. Exhaustive dispatch forces each consumer to choose a
handling rule but does not prove that rule correct. Domain restructuring
removes the invalid combination from the representation: dedicated compiler
annotations carry their required shapes, while opaque data cannot appear as
an expression leaf. These techniques complement one another; the data/AST
boundary relies on all three, not on a consumer convention to ignore data.

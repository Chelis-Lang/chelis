# `deep-user-symbol-charset` allowlist gap

Diagnosis note for Item 3 of the 0.7.6 toolchain hygiene workstream.
Records the diff between the lint's `CLOSED_TAGS` allowlist and the
canonical Deep tag vocabulary, identifies the missing tags, and
recommends a structural follow-up for the orchestrator.

## Symptom

`chelis deep <program>` emits hyphenated compound tags from the closed
Deep tag vocabulary (e.g. `t-ref`), and a subsequent `chelis lint
--check` on that output rejects the program. The toolchain conflicts
with itself: emit canonical Deep, then refuse to consume canonical
Deep.

The user-reported failure shape is the `t-ref` tag, emitted whenever
Surf's `&T` read-only borrow type is desugared. The failing fixtures
in `crates/chelis-lint/tests/deep_user_symbol_charset_compound_tags.rs`
pin the precise output a user would see.

## Authoritative sources

- Lint rule and allowlist: `crates/chelis-lint/src/rules/deep_user_symbol_charset.rs`
  - `CLOSED_TAGS` constant: L31-92 (60 entries)
  - Regex accepting hyphens: L24 (`[A-Za-z_][A-Za-z0-9_-]*`)
  - Rejection logic: L131-144 (hyphenated token not in allowlist → violation)
- Canonical closed vocabulary: `crates/chelis-deep/src/validate.rs`
  - `VALID_TAGS` constant: L3-75 (61 entries)
- Specs (read-only — do not modify per workstream rules):
  - `spec/01-nomenclature.md` §1.4 (Deep tag vocabulary)
  - `spec/01-nomenclature.md` §1.5 (Deep symbol grammar)
  - `spec/03-deep-syntax.md` §2.5 (Type Expressions)
  - `spec/03-deep-syntax.md` §2.10 (61-tag count summary)

## Diff: `CLOSED_TAGS` vs canonical `VALID_TAGS`

Cross-referencing the lint's `CLOSED_TAGS` (60 entries) against
`chelis-deep/src/validate.rs:3-75` `VALID_TAGS` (61 entries):

| Canonical tag | In `CLOSED_TAGS`? | Category (spec §2.x) |
|---------------|-------------------|----------------------|
| `module`         | yes | 2.1 Module |
| `import`         | yes | 2.1 Module |
| `import-all`     | yes | 2.1 Module |
| `export`         | yes | 2.1 Module |
| `def`            | yes | 2.2 Declarations |
| `defsig`         | yes | 2.2 Declarations |
| `deftype`        | yes | 2.2 Declarations |
| `typealias`      | yes | 2.2 Declarations |
| `variant`        | yes | 2.2 Declarations |
| `field`          | yes | 2.2 Declarations |
| `defdim`         | yes | 2.2 Declarations |
| `fn`             | yes | 2.3 Expressions |
| `app`            | yes | 2.3 Expressions |
| `let`            | yes | 2.3 Expressions |
| `match`          | yes | 2.3 Expressions |
| `arm`            | yes | 2.3 Expressions |
| `if`             | yes | 2.3 Expressions |
| `var`            | yes | 2.3 Expressions |
| `lit`            | yes | 2.3 Expressions |
| `record`         | yes | 2.3 Expressions |
| `access`         | yes | 2.3 Expressions |
| `pipe`           | yes | 2.3 Expressions |
| `block`          | yes | 2.3 Expressions |
| `tuple`          | yes | 2.3 Expressions |
| `tuple-get`      | yes | 2.3 Expressions |
| `par`            | yes | 2.3 Expressions |
| `borrow`         | yes | 2.3 Expressions |
| `pat-var`        | yes | 2.4 Patterns |
| `pat-lit`        | yes | 2.4 Patterns |
| `pat-ctor`       | yes | 2.4 Patterns |
| `pat-tuple`      | yes | 2.4 Patterns |
| `pat-record`     | yes | 2.4 Patterns |
| `pat-wild`       | yes | 2.4 Patterns |
| `pat-as`         | yes | 2.4 Patterns |
| `record-update`  | yes | 2.3 Expressions (reserved) |
| `t-prim`         | yes | 2.5 Types |
| `t-fn`           | yes | 2.5 Types |
| `t-tensor`       | yes | 2.5 Types |
| **`t-ref`**      | **NO**  | **2.5 Types** |
| `t-adt`          | yes | 2.5 Types |
| `t-var`          | yes | 2.5 Types |
| `t-unit`         | yes | 2.5 Types |
| `t-tuple`        | yes | 2.5 Types |
| `d-name`         | yes | 2.6 Dimensions |
| `d-var`          | yes | 2.6 Dimensions |
| `d-lit`          | yes | 2.6 Dimensions |
| `grad`           | yes | 2.7 Transforms |
| `vmap`           | yes | 2.7 Transforms |
| `jit`            | yes | 2.7 Transforms |
| `realize`        | yes | 2.7 Transforms |
| `cast`           | yes | 2.7 Transforms |
| `copy`           | yes | 2.7 Transforms |
| `handle-effect`  | yes | 2.3 Expressions |
| `quote`          | yes | 2.8 Metaprogramming |
| `unquote`        | yes | 2.8 Metaprogramming |
| `splice`         | yes | 2.8 Metaprogramming |
| `params`         | yes | 2.9 Helpers |
| `bind`           | yes | 2.9 Helpers |
| `kv`             | yes | 2.9 Helpers |
| `effects`        | yes | 2.9 Helpers |
| `resource`       | yes | 2.9 Helpers |

### Missing tags

Exactly one tag is canonical-and-emitted but absent from `CLOSED_TAGS`:

- **`t-ref`** — read-only borrow type. Defined in
  `spec/03-deep-syntax.md` §2.5 (`(t-ref {} type)` — "Read-only borrow
  type"). Emitted by `chelis-surf/src/desugar.rs:1330`
  (`TypeExpr::Ref(...) -> node("t-ref", ...)`) whenever Surf source
  contains the `&T` borrow type. Consumed by
  `chelis-ir/src/host.rs:4633` and `chelis-ir/src/lower.rs:2076`.
  Validated as canonical by `chelis-deep/src/validate.rs:50`.

That's the entire allowlist gap. The fix is a one-line addition to
`CLOSED_TAGS`.

## Tag mentioned in plan but not actually emitted: `t-dims`

Plan §3.1's enumeration table includes `t-dims` alongside `t-ref`, but
the on-disk repo state shows:

- `t-dims` is **not** in `chelis-deep/src/validate.rs` `VALID_TAGS`.
- `t-dims` is **not** in `spec/03-deep-syntax.md` §2.5 / §2.6 / §2.10
  (the 61-tag closed vocabulary).
- `t-dims` does not appear in any Deep emitter (`chelis-surf/src/desugar.rs`,
  `chelis-deep/src/printer.rs`, or `chelis-ir/src/host.rs`). A
  ripgrep across the whole repo (excluding the lint test fixtures
  added by Commit 1 of this branch) returns only two hits, both in
  `chelis-ir/src/lower.rs` (L2164, L4258) where the IR lowerer
  defensively accepts `(t-dims {} ...)` as a packaging form for
  dimension lists when extracting reshape arguments.

Because no emitter produces `t-dims`, it does not surface as a real
allowlist gap. Adding it to `CLOSED_TAGS` would diverge the lint's
view of the closed vocabulary from the canonical `VALID_TAGS` list
and the spec. The fix therefore **does not add `t-dims`**; only
`t-ref` is added.

The defensive consumer in `chelis-ir` is an unrelated curiosity that
should be investigated separately. It is either dead code (no producer
exists) or vestigial from a prior IR experiment. Surfacing this
observation to the orchestrator (see recommendations below).

## Why the bug exists

`CLOSED_TAGS` (`crates/chelis-lint/src/rules/deep_user_symbol_charset.rs:31-92`)
is a hand-maintained mirror of `chelis-deep/src/validate.rs:3-75`'s
`VALID_TAGS`. The mirror was last synchronized before `t-ref` was
added to the canonical vocabulary, and there is no compile-time
constraint forcing the two lists to agree. The lint and the validator
hold parallel ground truths; one drifted, the other did not.

The plan summary in `spec/01-nomenclature.md` §1.4 says "Closed
vocabulary, hardcoded in `crates/chelis-deep/src/validate.rs` and
emitted by `crates/chelis-deep/src/printer.rs`. Authoritative list in
`spec/03-deep-syntax.md:262+`." — the lint is not named there.

## Recommendations (orchestrator decides whether to file §5 entries)

These are **recommendations** for the orchestrator. Per workstream
rules the diagnosis does not file `docs/gap_synthesis.md` entries
directly.

1. **Centralize the closed-tag vocabulary in one Rust constant**
   shared by `chelis-deep`, `chelis-ir`, and `chelis-lint`. Promote
   `VALID_TAGS` (currently private to `crates/chelis-deep/src/validate.rs`)
   to a public `pub const CLOSED_TAGS: &[&str]` re-exported by
   `chelis-deep`. Have `chelis-lint`'s `deep_user_symbol_charset` rule
   consume that constant directly, eliminating the hand-maintained
   mirror entirely. Risks: cross-crate dep cycle if not careful
   (currently `chelis-lint` does not depend on `chelis-deep`); adding
   the dep is the right call but should be reviewed for compile-time
   impact on the lint binary.

2. **Or: generate `CLOSED_TAGS` at build time from `spec/03-deep-syntax.md` §2.**
   A `build.rs` in `chelis-deep` (or a new shared `chelis-vocab` crate)
   could parse the §2 markdown tables and emit a Rust array. This makes
   the spec the single source of truth and removes both Rust mirrors.
   Higher implementation cost; higher payoff for cross-cutting
   vocabulary issues (RISC primitive list, effect names, etc.). See
   the recurring "closed-set facts need citation+spot-check" pattern.

3. **Investigate `chelis-ir/src/lower.rs`'s `t-dims` consumer.**
   Decide whether to (a) remove it as dead code, or (b) add `t-dims`
   to the canonical vocabulary (spec + `VALID_TAGS`) along with a
   matching emitter. Either way, the current state — a consumer with
   no producer and no spec entry — is a latent inconsistency that will
   confuse future maintainers. This is independent of the Item 3 fix.

None of these recommendations is filed as a §5 entry by this
diagnosis. The orchestrator decides scheduling and entry placement.

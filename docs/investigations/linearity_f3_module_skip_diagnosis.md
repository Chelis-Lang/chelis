# Linearity-F3: module-wrapped top-level defs skip the linearity check

Workstream: Linearity-F3. Branch: `fix/linearity-f3-pr1-warning-sweep`.
PR sequence: PR 1 (this branch) makes linearity run on module-wrapped
defs and emits violations as warnings; PR 2 (separately dispatched)
fixes the surfaced violations and flips the severity to errors.

Background. This bug was surfaced as a §5 follow-up in the V2-F4
closeout note (`var_rhs_aliased_fanout_v2_diagnosis.md`,
"Out-of-scope" section, item 2). It was held back from V2-F4 because
the visible symptom is "check passes" rather than "check rejects
valid code", and fixing it touches the active corpus.

Failing fixtures landed in 46b4c15:
- `crates/chelis-types/tests/linearity_module_wrapped.rs::module_wrapped_realize_then_borrow_emits_warning`
- `crates/chelis-types/tests/linearity_module_wrapped.rs::module_wrapped_consuming_call_then_borrow_emits_warning`
- `crates/chelis-types/tests/linearity_module_wrapped.rs::module_wrapped_multi_realize_then_borrow_emits_warning`

## Reproducer

Bare top level (rejects):

```chelis
x = to_tensor([1.0, 2.0, 3.0])
y = realize(x)
b = add(x, y)
```

`chelis check` emits:

```
"errors": [{"kind":"UseAfterConsume","message":"variable `x` was already consumed by realize at offset 0; later use at offset 0 is invalid","severity":0.9}]
```

Same statements wrapped in `module Test`:

```chelis
module Test

x = to_tensor([1.0, 2.0, 3.0])
y = realize(x)
b = add(x, y)
```

`chelis check` emits:

```
"score": 1, "errors": []
```

Score 1.0, no errors, no warnings. The linearity check did not run.

## Why the module-wrapped case accepts

`check_linearity` (`crates/chelis-types/src/linearity.rs:118-144`)
runs a pre-declare loop over `program.annotated_exprs()`:

```rust
for expr in program.annotated_exprs() {
    if let Expr::List(list, _) = expr
        && get_tag(list) == Some("def")
        && let Some(name) = children(list).first().and_then(symbol_name)
    {
        scope.declare(name, program.type_env().get(name).cloned());
    }
}
```

The filter is on the outermost expression's tag. For every idiomatic
source (every `.ch` begins with `module Foo`), the top-level
annotated expression is `(module {} name (defsig ...) (def ...)
...)`, not `(def ...)`. The pre-declare loop sees the `module` tag,
the filter fails, and **no top-level names are declared in `scope`**.

The main loop then walks `check_top_level` (lines 246-263) over each
top-level expression. `check_top_level` only handles `(def ...)`
directly; for any other tag it falls through to `check_expr`. For
`(module {} name ...)` the call routes through `check_expr` to the
`_` arm of the tag-match (line 316-320), which walks `children(list)`
recursively. The walk descends into each inner `(def name body)`,
then into its body, where intra-body linearity logic (fn parameters,
let-bindings, branch joins) DOES fire because `check_fn`, `check_let`,
`check_if`, `check_match` declare names into local scopes regardless
of how they were reached.

The gap is therefore at the **module-top-level scope only**. Inside
function bodies linearity is intact. The visible bugs are:

1. The V2-F4 binding-aliasing fix on `(def y (var x))` in
   `check_top_level` (lines 274-287) never runs for module-wrapped
   defs.
2. Top-level `consume_var_expr(x, ...)` calls from inner defs find
   `scope.top("x") = None` (never declared) and silently no-op,
   never recording a consume.
3. Subsequent borrow reads of `x` via `read_or_error` therefore
   never see a consume site, never emit `UseAfterConsume`, and the
   program "passes" linearity.

`check_linearity_with_context` (lines 170-243) has the same shape
and the same bug for both library and new-code modules.

## Fix plan for PR 1

1. Add a `module_path: Vec<String>` (or equivalent) flag to the
   `Checker` indicating whether the current walk is inside one or
   more module wrappers. (For PR 1, a single `in_module: bool` is
   sufficient — module nesting depth does not change severity.)
2. Replace the pre-declare loop with a helper that mirrors the
   `top_level_decl_items` pattern at
   `crates/chelis-types/src/infer.rs:1056-1074`: recurse through
   `(module {} name children...)` wrappers and pre-declare every
   nested `(def ...)`.
3. Extend `check_top_level` to detect a `(module {} name children...)`
   expression and recurse: walk each child as if it were itself a
   top-level expression, with `in_module = true`.
4. When `in_module` is true, route every push to `checker.errors`
   into `info.warnings` via `LinearityInfo::push_warning` instead.
   Concretely: add a `push_diagnostic(&mut self, e: CheckError)`
   helper on the `Checker` that dispatches based on `in_module`, and
   replace every `self.errors.push(e)` inside the body-walk path
   with `self.push_diagnostic(e)`. `consume_var_expr`,
   `read_or_error`, and `invalid_borrow` are the three diagnostic
   emitters.
5. Return `Ok(program.with_linearity(info))` when `checker.errors`
   is empty even if `info.warnings()` is non-empty.

Existing top-level bare-program behavior is untouched because the
pre-declare loop and walk for non-module expressions remain on the
original error-emitting path.

No `LinearityInfo` schema change is required beyond the
already-landed `warnings: Vec<CheckError>` field
(commit 46b4c15). `Checker` gains one boolean field; the new
`push_diagnostic` is a one-line dispatch.

The fix is local to `check_linearity` (and its `_with_context`
twin) and `Checker`'s emit helpers. No downstream consumer of
`LinearityInfo` (chelis-ir lowerer, eval, cli-cli) reads `warnings()`
today; the fix commit will wire the CLI to print warnings to stderr.

## PR 1 acceptance

- Three fixtures in
  `crates/chelis-types/tests/linearity_module_wrapped.rs` go from
  `#[ignore]`'d-and-failing to running-and-passing without manual
  fixture edits.
- The three sibling bare-program control tests in the same file
  continue to pass.
- `cargo test --workspace` is green.
- `cargo clippy --workspace --all-targets -- -D warnings` is green.
- `cargo fmt --all -- --check` is green.
- `chelis lint --check .` does not regress.
- The full corpus (`examples/`, `examples/illustrative/`,
  `packages/chelis-std/`, `crates/*/tests/fixtures/`) is inventoried
  for surfaced linearity warnings; the list is recorded in
  "Existing violations to fix in PR 2" below.

## PR 2 scope

PR 2 is the follow-on dispatch that closes the loop.

Plan:

1. The PR 1 corpus sweep found zero surfaced warnings (see
   "Existing violations to fix in PR 2" below), so the "fix every
   warning" step is a no-op for the in-tree corpus.
2. Flip the severity. The simplest path: drop the `in_module` flag
   from `Checker` and have module-recursive linearity route through
   the same `errors` vec as bare-top-level linearity. The
   `LinearityInfo::warnings` field becomes vestigial and can be
   removed in the same commit, along with `push_warning` and the
   manual `PartialEq` impl. The CLI stderr emit in
   `crates/chelis-cli/src/main.rs` is also dropped.
3. Sweep the corpus once more under `cargo run -p chelis-cli --bin
   chelis -- check` and `chelis lint --check .` to confirm no new
   errors appear.
4. Update the V2-F4 closeout doc and the spec notes flagging the
   latent gap to record that the gap is closed.

PR 2 must NOT land before PR 1. The branch order is enforced by
the dependency: PR 2's "delete the warning plumbing" step has
nothing to delete until PR 1 adds it.

## Existing violations to fix in PR 2

Inventory captured by running `cargo run -p chelis-cli --bin chelis
-- check --allow-style-violations` against every `.ch` file in
`examples/`, `examples/illustrative/`, `packages/` (chelis-std and
its tests), and `crates/*/tests/` (103 files total) after the PR 1
fix landed.

**Inventory: 0 surfaced linearity warnings.**

The executable corpus is already clean. Every `module Foo`-wrapped
program in the repository either uses function-body scope
(where intra-body linearity has always been intact) or uses safe
binding-aliasing chains (where the V2-F4 binding-tolerance applies
once the pre-declare loop sees the module-wrapped names).

This is good news for the workstream: PR 2's "fix surfaced
violations" step is a no-op, and PR 2 can collapse to a single
commit that flips warnings to errors and removes the warning-mode
plumbing.

A non-empty inventory was not the goal of PR 1 — the goal is to
remove the silent skip so that future module-wrapped programs that
DO violate linearity surface a warning rather than slipping
through. The corpus sweep confirms no in-tree program is hiding a
latent violation behind the bug.

| File | Line | Rule | Reason |
| ---- | ---- | ---- | ------ |
| _(none)_ | | | The 103-file sweep found zero surfaced warnings. |

Inventory script: `/tmp/lf3-pr1/sweep.py` in the local agent
worktree, ad-hoc. The sweep is reproducible from this branch via
`cargo build -p chelis-cli` followed by walking `find examples/
packages/ crates/ -name '*.ch'` and running `chelis check
--allow-style-violations <path>` on each, filtering stderr for
lines starting with `warning: linearity:`. The script is not
checked into the repo because the inventory step is a one-shot
PR 1 validation, not a recurring CI artifact.

## Out-of-scope / escalations

1. `ConsumeKind` typed-schema refactor — same as the predecessor
   notes (`var_rhs_let_fanout_diagnosis.md`,
   `var_rhs_aliased_fanout_v2_diagnosis.md`). The discrimination
   between aliasing and structural consumes remains string-based.
   A typed refactor is a candidate §5 follow-up; it is not required
   to surface the warnings.
2. Realize-then-borrow at top level — flagged in the V2-F4 closeout
   note (item 3 of its "Out-of-scope" section). Independently of
   the module-wrapping bug, the spec's Copy-insertion rules suggest
   that some realize-then-borrow shapes could be auto-corrected.
   Out of scope for this workstream; addressed via the spec-aligned
   review dispatch flagged in the V2-F4 note.
3. Module-nesting scope semantics — for PR 1 a single
   `in_module: bool` is sufficient because PR 2 collapses
   warnings back into errors. Whether each module-level scope is
   independent or shared is a separate semantic decision; the spec
   already treats top-level statements as conceptually flat (see
   `top_level_decl_items` at `infer.rs:1056-1074`), and PR 1 follows
   that precedent.

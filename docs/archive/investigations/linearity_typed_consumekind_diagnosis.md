# Linearity-F1 / F2 / AliasedConsume-F1 — diagnosis (W1 PR 1)

Owning plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`,
"W1 — Linearity layer correctness, PR 1".

Phase 0 spec lock: `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md`.

§5 entries closed by this PR (text edits filed by the orchestrator after
merge): `Linearity-F1`, `Linearity-F2`, `Linearity-AliasedConsume-F1`.

## Failure-mode map

Each fixture from `crates/chelis-types/tests/linearity_typed_consumekind.rs`
and `crates/chelis-types/tests/linearity_aliased_consume.rs`, mapped to
the code path that produces the observed behavior today and the
counterfactual after the fix.

### Fixture 1 — aliasing-consume control

```
def f(w: tensor[4, f32]) -> tensor[4, f32] =
  { y: tensor[4, f32] = w
    realize(y) }
```

Today:
- `check_let` (linearity.rs:506-526) sees `value = (var w)` and
  `expr_is_owned_linear(value)` is true; `consume_var_expr` records a
  `ConsumeSite { description: "binding `y` at offset N" }` against
  `w`.
- `realize(y)` triggers `check_realize` → `consume_var_expr` →
  `scope.consume(y, realize_site(...))`.
- No later use of `w` or `y`, so `read_or_error` is never called.
  Linearity passes.

After PR 1: the `ConsumeSite` recorded against `w` carries
`kind: ConsumeKind::Aliasing`; same overall outcome (passing) but the
discrimination is by typed field, not by string prefix.

### Fixture 2 — structural-consume control

```
def f(w: tensor[4, f32]) -> tensor[4, f32] =
  { y: tensor[4, f32] = realize(w)
    add(w, y) }
```

Today:
- `check_let` sees `value = (realize (var w))`; the `is_var_expr(value)`
  guard at L520 is false, so it falls into `check_expr(value, scope)`
  → `check_realize(expr, ...)` → `consume_var_expr(child, scope,
  realize_site(expr))`. `w`'s `BindingState` becomes
  `Consumed(ConsumeSite { description: "realize at offset N" })`.
- `add(w, y)` is a builtin where the first arg is read-only-borrowed
  (auto-borrow). `check_app` (L437-458) routes through `read_var_expr`
  → `read_or_error(w, ...)`.
- `read_or_error` (L745-787) finds `w` consumed; the description does
  *not* start with `"binding "`, so the function falls through to
  emit `UseAfterConsume`.

After PR 1: same path, but the test in `read_or_error` is
`matches!(site.kind, ConsumeKind::Aliasing)` instead of
`site.description.starts_with("binding ")`. Same outcome (failing).

### Fixture 3 — aliased-consume bypass (AliasedConsume-F1)

```
def f(w: tensor[4, f32]) -> tensor[4, f32] =
  { y: tensor[4, f32] = w
    z: tensor[4, f32] = realize(y)
    add(w, z) }
```

Today, silent pass:
- `check_let` for `y = w`: `consume_var_expr(value=(var w), ...,
  site={description: "binding `y` at offset N"})`. `w` is now
  `Consumed`.
- `check_let` for `z = realize(y)`: `check_expr(realize(y))` →
  `check_realize` → `consume_var_expr(y, scope, realize_site)`.
  `y` becomes `Consumed`. `w`'s scope entry is unaffected.
- `add(w, z)`: `read_var_expr(w)` → `read_or_error(w, ...)`. `w` is
  consumed with `description = "binding `y` at offset N"`, which
  *starts with* `"binding "` so `read_or_error` returns early on the
  Aliasing-consume tolerance path. **No error**.

The bug: `read_or_error`'s tolerance is correct for the alias
itself (a borrow of the alias-RHS source must not be flagged
because the IR shares the `Load` node), but it fails to track
through-to-source: when `realize(y)` happens, the alias of `y` is
*also* gone, and the tolerance for `w` is no longer applicable.

After PR 1: an alias mechanism (see "Lineage-tracking design") tells
`Checker` that `y` aliases `w`. When `realize(y)` consumes `y`
structurally, the consume forwards to `w`'s scope entry, replacing
the existing Aliasing-consume with a Structural one. The borrow at
`add(w, z)` then sees `w` consumed with `ConsumeKind::Structural`
and surfaces `UseAfterConsume` per the Fixture 2 path.

### Fixture 4 — tuple-destructure linearity (F2)

```
def f(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  { (a, b) = pair
    r1: tensor[4, f32] = realize(a)
    realize(a) }
```

Desugar at `crates/chelis-surf/src/desugar.rs:1135-1156` rewrites this
as (eliding span metadata):

```
(let (bind {} __chelis_tmp0 (var {} pair))
  (let (bind {} __chelis_tmp1 (tuple-get (var __chelis_tmp0) 0))
    (let (bind {} a (var {} __chelis_tmp1))
      (let (bind {} __chelis_tmp2 (tuple-get (var __chelis_tmp0) 1))
        (let (bind {} b (var {} __chelis_tmp2))
          ...)))))
```

Crucially, none of the `__chelis_tmp_N` bind values carries a `:type`
entry in its meta-map. The Var-pattern path at L1166-1171 calls
`inject_type_metadata` when an explicit type is supplied on the
binding, but the destructure path at L1135-1156 does not call it for
the synthesized intermediates.

Today, silent pass:
- `check_let` for `a = (var __chelis_tmp1)`: `expr_is_owned_linear`
  on `value` calls `expr_type` which reads
  `type_metadata((var __chelis_tmp1))`. The meta-map is empty, no
  `:type` entry. The fallback at `linearity.rs:790-792` looks up
  `__chelis_tmp1` in scope. The previous `let` bound `__chelis_tmp1`
  with whatever type `expr_type(tuple_index_expr(...))` returned — but
  `tuple-get` is generic and synthesized without a `:type` meta
  either. So both layers return `None`. `expr_is_owned_linear`
  returns false. **The consume on `__chelis_tmp1` is skipped**.
- `scope.declare(a, expr_type(value))` declares `a` with type `None`.
- `check_realize(realize(a))` → `consume_var_expr(a, ...)`.
  `expr_is_owned_linear(a)` reads `scope.ty("a")` which is `None`.
  Returns false; the consume on `a` is skipped.
- Same for the second `realize(a)`. No error.

After PR 1: the destructure desugar threads each tuple element's
type onto the synthesized `tuple_index_expr` and onto the
`__chelis_tmp_N` bind. The bind's value carries `:type` =
`(t-tensor {} (d-lit 4) (t-prim f32))`. `check_let` propagates the
type to `scope.declare(__chelis_tmp1, ...)` and then to
`scope.declare(a, ...)`. Subsequent consumes flow through the
already-tested paths. The second `realize(a)` now hits
`scope.consume(a, ...)` while `a` is already `Consumed(realize_site)`,
which today routes to the silent-fallthrough arm at L726 of
`consume_var_expr`. To surface this as a violation under PR 1, the
linearity check emits a *warning* on this case (see "Warning-emission
plumbing").

### Fixture 5 — tuple-destructure positive control

After the PR 1 fix threads type metadata, both `realize(a)` and
`realize(b)` consume distinct `BindingState`s exactly once. No
later use of either; no warning emitted; passes.

### Fixture 6 — mixed alias + destructure

Compounds Fixtures 3 and 4. After PR 1 wires both fixes, the
destructured `a` has a type, the alias `y = a` records an
Aliasing-consume on `a`, `realize(y)` forwards a Structural consume
to `a`, and `add(a, r)` borrows `a` through `read_or_error` which
finds `a` consumed with `ConsumeKind::Structural`. The violation
routes through the warning channel during the deprecation window.

## Lineage-tracking design

**Pick: Option B (HashMap-based alias map on `LinearScope`).**

Rationale: less invasive than introducing a third `BindingState`
variant. With Option B, `BindingState` keeps its `Live` / `Consumed`
shape and every existing site that pattern-matches on `BindingState`
keeps its current arms. The alias map is a side-table consulted only
at `consume_var_expr` time.

Shape:

```rust
#[derive(Debug, Clone, Default)]
struct LinearScope {
    bindings: HashMap<String, Vec<BindingState>>,
    types: HashMap<String, Vec<Option<Expr>>>,
    /// Alias map: when `let y = (var x)` is recorded as an
    /// `Aliasing` consume, also record `y -> x` here so that later
    /// structural consumes of `y` forward to `x`'s scope entry.
    /// Stacks parallel to `bindings` so that scoping behaves the
    /// same way (a re-let of `y` shadows the alias, and the previous
    /// alias re-emerges on `pop`).
    aliases: HashMap<String, Vec<String>>,
}
```

Operations to add:

- `LinearScope::record_alias(&mut self, alias: &str, source: &str)`:
  push `source` onto the stack at `aliases[alias]`. Called from
  `check_let` when `is_var_expr(value) && expr_is_owned_linear(value)`.
- `LinearScope::resolve_alias_chain(&self, name: &str) -> Option<&str>`:
  walk the alias stack to the underlying non-alias source. Bounded
  by chain length, which is bounded by source-program nesting depth.
  Returns `None` if `name` is not an alias.
- `LinearScope::declare` / `pop`: extend to push/pop the parallel
  alias stack so shadowing works as expected. Re-let of `y` with a
  non-var value pushes `None` (or just skips the aliases push) so
  the alias does not leak into the new binding.

Semantics in `consume_var_expr`:

```rust
fn consume_var_expr(&mut self, expr: &Expr, scope: &mut LinearScope, site: ConsumeSite) {
    // ... existing guards ...
    let name = var_name(expr).unwrap();
    // Forward structural consumes through the alias chain.  Aliasing
    // consumes do not forward (they only update the alias's own
    // scope entry).
    let target = match site.kind {
        ConsumeKind::Structural => scope
            .resolve_alias_chain(name)
            .map(|src| src.to_string())
            .unwrap_or_else(|| name.to_string()),
        ConsumeKind::Aliasing => name.to_string(),
    };
    match scope.top(&target) {
        Some(BindingState::Live { .. }) => scope.consume(&target, site),
        Some(BindingState::Consumed(consumed_at))
            if consumed_at.description.contains("closure capture")
                || consumed_at.description.contains("match scrutinee") => { ... },
        Some(BindingState::Consumed(consumed_at))
            if matches!(consumed_at.kind, ConsumeKind::Aliasing)
                && matches!(site.kind, ConsumeKind::Structural) =>
        {
            // The target was previously aliased; now we are
            // forwarding a structural consume to it.  Replace the
            // Aliasing record with the Structural one so subsequent
            // borrows of the target trip `read_or_error`.
            scope.consume(&target, site);
        }
        Some(BindingState::Consumed(_)) => {
            // implicit-Copy IR pass handles consuming fan-out;
            // borrow-after-consume is still caught by `read_or_error`.
        }
        None => {}
    }
}
```

Multi-level chains (`let z = y; let y = x; consume(z)`) work because
`resolve_alias_chain` walks until a non-alias name is found.

`read_or_error` itself does *not* need alias resolution: it is
called on a specific name at borrow time. The alias-consume forwarding
happens at consume time, so by the time a later borrow on the source
name fires, `scope.top(source)` already reflects the typed kind of
the most-recent consume.

## Warning-emission plumbing for the F2 destructure cascade

Mirror the F3 PR 1 pattern introduced in PR #65. The current
`Checker::push_diagnostic` at `linearity.rs:159-167` post-PR2 routes
unconditionally to `errors`. PR 1 reintroduces an
`in_destructure_warning_mode: bool` (or equivalent) on `Checker` set
true while walking nested expressions whose linearity violations are
*only* surfaced because of the type-metadata threading change.

The W1 PR 1 fix routes through a different gate than F3 did: F3's
gate was a `(module ...)` wrapper boundary. W1 PR 1's gate is the
post-fix discovery of consumes that the *pre-fix* desugarer would
have skipped. The simplest mechanical implementation:

- Tag the synthesized destructure tmp bindings with a `:meta` entry
  `destructure: true` in `desugar.rs:destructure_pattern`.
- `Checker::check_let` reads the entry on each bind. When it sees
  `destructure: true`, it sets `in_destructure_warning_mode = true`
  for the duration of the let body's recursive check, and restores
  on return.
- `push_diagnostic` routes to `info.warnings` when
  `in_destructure_warning_mode` is true.

Why this scope. The brief identified two adjacent failure classes
that share the destructure gate:

1. Direct double-consume of a destructured component (Fixture 4).
2. Aliased-consume through a destructured component (Fixture 6).

Both surface only because the destructure type-metadata threading
unblocked `expr_is_owned_linear`. Routing both through the warning
channel matches the cascade boundary: the in-tree corpus may have
latent violations of either shape that should be cleaned up before
the W2-cascade PR flips them to errors.

The aliased-consume bypass on non-destructured bindings (Fixture 3)
does *not* route through the warning channel — it is a hard error
from PR 1. Reason: the bug today is silent at *use*, not silent at
*surface*. The fix surfaces nothing new from the in-tree corpus
that was not already implied as a violation by the existing
single-binding linearity rules; it just makes the rules actually
enforce on aliased-via-alias consumes. The corpus sweep recorded
no instances of Fixture-3 shape today (every `let y = x; ...
realize(y); ... add(x, ...)` pattern in the corpus would already
trip the existing string-prefix check).

If a corpus sweep during PR review surfaces actual Fixture-3 shapes
that should not be errors, escalate to the orchestrator (per the
"escalate structural workarounds" rule). Do not silently widen the
warning channel to cover them.

## References

- Plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`
- Phase 0 spec lock: `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md`
- §5 entries: `docs/archive/reports/gap_synthesis.md` rows for `Linearity-F1`,
  `Linearity-F2`, `Linearity-AliasedConsume-F1`
- F3 PR 1 plumbing reference: commit `c7469d9` in `git log` —
  `LinearityInfo::warnings` field, `Checker::push_diagnostic`
  branching on `in_module`
- Code anchors:
  - `crates/chelis-types/src/linearity.rs:34-36` — `ConsumeSite` shape
  - `crates/chelis-types/src/linearity.rs:39-42` — `LinearScope`
    keyed by name only
  - `crates/chelis-types/src/linearity.rs:726` — string-prefix
    tolerance to replace
  - `crates/chelis-types/src/linearity.rs:330-336, 482-488, 519-522,
    575-577, 1200-1209, 1211-1215, 1217-1221, 1223-1231` — eight
    producer call-sites
  - `crates/chelis-surf/src/desugar.rs:1135-1156` —
    `destructure_pattern`
  - `crates/chelis-surf/src/desugar.rs:281-302` —
    `inject_type_metadata`
  - `crates/chelis-surf/src/desugar.rs:1158-1186` —
    `desugar_let_bindings` dispatch

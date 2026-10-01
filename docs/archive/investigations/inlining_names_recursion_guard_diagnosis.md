# `inlining_names` recursion-guard over-application — diagnosis

> **Superseded (chelis#620).** The refuse-on-reentry `inlining_names` set
> this doc diagnoses was replaced by depth-bounded unrolling
> (`inlining_depths` + `inlining_active` in
> `crates/chelis-ir/src/lower.rs`): a re-entrant call now inlines
> normally, statically-terminating recursion unrolls (its base case
> prunes via the static `if`/`match` condition fold), and a chain the
> pruning cannot bound errors loudly at a named cap instead of falling
> through to the silent return-last-arg fallback documented below. The
> fn-typed-parameter alias analysis in this doc (the `visited` set vs
> guard-placement distinction) still describes the live code.

Diagnoses **Inlining-F1** from the 0.7.6 toolchain hygiene workstream,
surfaced by PR #37's call-site parity test for fn-typed-parameter pipe
stages (`crates/chelis-ir/tests/pipe_evaluation.rs:422`–`425`).

## Bug shape (confirmed empirically)

Fixture: `crates/chelis-ir/tests/inlining_recursion_guard.rs`'s
`nested_fn_param_call_lowers_via_substituted_callable` (verified failing
on commit `5bd40a7`, the test-pin commit, before any guard change):

```
def doubler(x) = add(x, x)
def outer(f: tensor[3, f32] -> tensor[3, f32], x) = f(f(x))
// Call site:  outer(doubler, seed=[1, 2, 3])
// Expected:   doubler(doubler([1, 2, 3])) = [4, 8, 12]
// Actual:     [2, 4, 6]   (inner f(x) silently collapses to x)
```

The lowered DAG evaluates to `doubler(seed)` instead of
`doubler(doubler(seed))`. No panic, no diagnostic — silent data loss.

## Why the guard over-applies

The relevant code in `crates/chelis-ir/src/lower.rs`:

1. `try_lower_callable_app` (`lower.rs:2756`–`2802`) inserts a name into
   `self.inlining_names` *before* lowering the resolved callable body,
   then removes it after:

   ```rust
   let inlining_name = callable_ref_name(func).filter(|name| {
       self.local_callables.contains_key(name) || self.program_defs.contains_key(name)
   });
   if let Some(name) = inlining_name.as_ref() {
       self.inlining_names.insert(name.clone());
   }
   // ... lower the callable body, which may itself lower more apps ...
   if let Some(name) = inlining_name {
       self.inlining_names.remove(&name);
   }
   ```

   `callable_ref_name(func)` returns the **surface name at the call
   site**. For `f(x)` inside `outer`'s body, that's `"f"` — the
   parameter alias, not the resolved callable.

2. `resolve_callable_expr_inner`'s `var` arm (`lower.rs:2818`–`2848`)
   then rejects any subsequent `(var name)` lookup whose `name` is in
   `inlining_names`:

   ```rust
   if !visited.insert(name.clone()) || self.inlining_names.contains(&name) {
       return None;
   }
   ```

### Step-by-step for `outer(doubler, seed)`

| Step | Action | `inlining_names` after | `local_callables` after |
|------|--------|------------------------|-------------------------|
| 1 | Lower call `outer(doubler, seed)`. `try_lower_callable_app(outer, [doubler, seed])` resolves `outer` via `program_defs`. | `{"outer"}` | `{}` (caller scope) |
| 2 | `lower_plain_callable_app(outer_body, ...)`. Substitutes args. `doubler` is callable, so `local_callables["f"] = (var doubler)`; `seed` is tensor, goes in `bindings`. Begin lowering `outer`'s body `(app f (app f x))`. | `{"outer"}` | `{"f": (var doubler)}` |
| 3 | Outer `f` call: `try_lower_callable_app(f, [(app f x)])`. `callable_ref_name(func) = Some("f")`. `f` is in `local_callables` → filter passes → `inlining_names.insert("f")`. `resolve_callable_expr(f)` chains through `local_callables["f"] = (var doubler)` → `program_defs["doubler"] = fn_body` → `Plain(doubler_body)`. | `{"outer", "f"}` | `{"f": (var doubler)}` |
| 4 | `lower_plain_callable_app(doubler_body, [(app f x)])`. `callable_binding_expr((app f x))` returns `None` (not a `var`/`fn`). So the `f(x)` arg is lowered via `lower_expr` → `lower_app(f, [x])` → `try_lower_callable_app(f, [x])`. | `{"outer", "f"}` | `{"f": (var doubler), "x": ...}` (doubler's param `x` shadows outer's `x`) |
| 5 | Inner `f`: `resolve_callable_expr(f)` → `var` arm checks `self.inlining_names.contains("f")` → **true** → returns `None`. `try_lower_callable_app` returns `None`. `lower_app` falls through to "lower func and args, return last" (`lower.rs:2748`–`2753`). The args are `[(var x)]`, so the inner `f(x)` evaluates to whatever `x` resolves to — silently dropping the `f` application. | `{"outer", "f"}` | `{"f": (var doubler), "x": ...}` |

The guard fired at step 5 because the *call-site alias* `"f"` is in
`inlining_names`. But there is no actual recursion here: `f` resolves to
`doubler`, whose body doesn't reference `f` at all. The recursion guard
is checking the wrong thing.

## What the guard should check

The guard exists for **true self-recursion**: `def f(x) = f(x)`. There,
the resolved callable body references the same callable. The guard must
fire to prevent `lower_plain_callable_app` from entering the body
infinitely.

Discrimination criterion:

> The `inlining_names` set should track the **resolved terminal
> callable's name** (the name whose body we are *actually* lowering),
> not the call-site alias name. A recursive call is one that resolves
> back to a body we are already in the middle of lowering. An alias
> chain through `local_callables` to a *different* callable is not
> recursion.

For `outer(doubler, seed)`:

- At step 3, what we are inlining is `doubler`'s body (resolved from
  `f` via `local_callables["f"] → (var doubler) → program_defs["doubler"]`).
  The right name to track is `"doubler"`, not `"f"`.
- At step 5, the inner `f(x)`'s resolver walks the same chain:
  `f → (var doubler) → ...`. When the resolver tries to recurse into
  `(var doubler)`, that's the step where the guard should fire — and
  it will, if `inlining_names` contains `"doubler"`.

For `def f(x) = f(x)` (true recursion):

- At step 3, what we are inlining is `f`'s body (resolved from `f` via
  `program_defs["f"] = fn_body`). The terminal name is `"f"`.
- At step 5 the inner `f(x)`'s resolver walks `f → program_defs["f"] →
  fn_body`. When the resolver evaluates `(var f)` at the outer step,
  the guard checks `inlining_names.contains("f")` → fires. Correct.

The discrimination is automatic if we change *what name we insert*: use
the resolved-terminal name instead of the surface alias.

## Fix surface

Add a helper that walks the alias chain to find the terminal name (the
last `(var <name>)` in the chain that maps to either an `fn` literal in
`program_defs` or that has no further redirect). For an `fn` literal
passed directly (no name at all), the helper returns `None` — anonymous
fn literals cannot be re-encountered by name, so they never need
guarding.

Pseudocode for the helper:

```rust
fn resolve_inlining_target_name(&self, expr: &Expr) -> Option<String> {
    // Walk var → local_callables/program_defs chain, returning the name
    // of the var whose binding is an `fn` literal (i.e. the body we will
    // actually inline). Use a visited set to avoid alias-chain cycles.
    let mut visited = HashSet::new();
    let mut cursor = expr;
    loop {
        let name = callable_ref_name(cursor)?;
        if !visited.insert(name.clone()) {
            return None; // alias cycle — defensive
        }
        let next = self
            .local_callables
            .get(&name)
            .or_else(|| self.program_defs.get(&name));
        match next {
            Some(next_expr) if matches!(get_tag_of(next_expr), Some("var")) => {
                cursor = next_expr;
            }
            Some(_) => return Some(name),       // points at fn or other body
            None => return None,                // unresolved (parameter or unknown)
        }
    }
}
```

Then in `try_lower_callable_app` (`lower.rs:2768`):

```rust
let inlining_name = self.resolve_inlining_target_name(func);
```

`resolve_callable_expr_inner`'s guard at `lower.rs:2823` stays exactly
as-is. It checks the *current step's* `name` against `inlining_names`,
which now correctly contains the resolved-target name. For the
nested-fn-param case (step 5), the resolver walks `f → doubler`, and
when it recurses into `(var doubler)`, the next iteration's `name` is
`"doubler"` — not in `inlining_names` (which contains `"doubler"`? No
— wait):

Let me re-trace. With the fix at step 3: `resolve_inlining_target_name(f)`
walks `f → (var doubler) → ?`. `local_callables["doubler"]` not set,
`program_defs["doubler"]` exists and is the fn body (not a var). Loop
returns `Some("doubler")`. So `inlining_names = {"outer", "doubler"}`
after step 3.

At step 5 (inner `f(x)`): `resolve_inlining_target_name(f)` walks
`f → (var doubler) → ?` → returns `Some("doubler")`. Insert "doubler"
into inlining_names... wait, but it's already there. Should the insert
be a no-op or should we skip the call? Let me re-think.

Actually, the inner `f(x)` calls `try_lower_callable_app(f, [x])`. The
*resolution* via `resolve_callable_expr` happens first. That call walks
the var/local_callables/program_defs chain and checks `inlining_names`
at each var step:

1. `name = "f"` — guard: `inlining_names.contains("f")` → false. Pass.
2. Look up `local_callables["f"] = (var doubler)`. Recurse.
3. `name = "doubler"` — guard: `inlining_names.contains("doubler")` →
   **true** (set by step 3 of the outer trace). Return `None`.
4. `try_lower_callable_app` returns `None`. Fallback runs.

Wait — that means even with the fix, the inner `f(x)` still gets dropped!
Because `doubler` IS in `inlining_names`, and the resolver correctly
detects "we are already inlining doubler's body."

But that's wrong too — we *want* the inner `f(x)` to inline doubler
once more. The argument to the outer `doubler` is supposed to be
`doubler(x)`, evaluated *during* the lowering of the outer doubler call.
Each level of inlining is a fresh substitution. So while we are still
inlining doubler's body, we should be allowed to inline doubler again
at a different call site within that body — as long as the body
terminates.

Hmm. The infinite-recursion danger is specifically: lowering doubler's
body, encountering a call to doubler, which re-enters lowering doubler's
body, which encounters another call... and so on forever **if the body
unconditionally references itself**.

But `doubler`'s body is `add(x, x)`. It doesn't reference `doubler`.
The danger only manifests when the body recursively calls itself.

So actually the issue is more subtle. The guard *cannot* discriminate
"this call is fine" from "this call is recursive" by looking only at
names. The guard fires on first re-entry, which is correct for true
recursion but wrong for legitimate nested calls.

Wait — re-trace the bug more carefully. In the bug case (outer body
`f(f(x))`):

- Outer `f` call: `try_lower_callable_app(f, [f(x)])`. Inserts "f" into
  inlining_names. Resolves to `Plain(doubler_body)`. Calls
  `lower_plain_callable_app(doubler_body, [f(x)])`.
- `lower_plain_callable_app` lowers args first: the `f(x)` argument is
  lowered via `lower_expr` → `lower_app(f, [x])` → `try_lower_callable_app(f, [x])`.

The arg-lowering is the critical step. At the point where we lower
`f(x)`, we have NOT yet entered doubler's body. We're still in the
arg-substitution phase of `lower_plain_callable_app`. The body lowering
hasn't started.

So at step 5: the inner `f(x)` is being lowered as an *argument*, not
as part of doubler's body. The guard fires too early — it fires when
we've only *started* inlining doubler, but doubler's body hasn't begun
to lower yet.

OK so actually the right discrimination IS structural in time. The
guard should be: "are we *in the middle of lowering the body* of
callable X?" not "have we started a call to X anywhere up the stack?"

That said, the inner `f(x)` here resolves *through* `f`. The chain
`f → doubler`. If we tracked the resolved target as `doubler`, the guard
would fire on the inner `f(x)` too (because we're already in the middle
of the outer `doubler` call's `try_lower_callable_app` frame, which set
`inlining_names = {"doubler"}`).

But the *intent* of the inner `f(x)` is to apply doubler to x, producing
`doubler(x)`. That's a separate, finite, *non-recursive* call. The
outer doubler call eventually finishes; arg-lowering finishes
independently.

So the right fix is **not** "track resolved name" — it's **"the guard
should not fire on arg lowering, only on body lowering"**. In other
words: insert `inlining_names.insert(...)` *after* arg evaluation,
before body lowering. Then arg evaluation can freely call into the
same callable.

Looking at the code: `try_lower_callable_app` inserts BEFORE calling
`lower_plain_callable_app`. Inside `lower_plain_callable_app`, args are
evaluated first (`lower.rs:3073`–`3098`), then the body is lowered
(`lower.rs:3102`). The insertion site is too early.

## Revised fix surface

Move `inlining_names.insert(...)` to right before body lowering, and
`inlining_names.remove(...)` to right after. Two implementation choices:

**Option A (smaller diff): move the guard into each `lower_*_callable_app` body.**
Each of `lower_plain_callable_app`, `lower_vmap_callable_app`,
`lower_vmap_grad_callable_app`, `lower_grad_callable_app` takes a
`fn_expr`. Determine the inlining name from the resolved fn_expr (which
the helper above can do once given the `func` expr that resolved here),
pass it through, and insert/remove around the body-lowering step.

**Option B (preferred): keep the insert/remove in `try_lower_callable_app`,
but lower args *first* via a new pre-step.** Since `lower_plain_callable_app`
and friends own arg-lowering, we'd need to either pull arg-lowering up
(invasive) or inline a marker telling the body-vs-args boundary.

**Option C (smallest): split `lower_plain_callable_app` into "lower args"
and "lower body" phases.** Have `try_lower_callable_app` do:
`args_lowered = lower_args_only(...); inlining_names.insert(...);
result = lower_body_only(...); inlining_names.remove(...)`.

Option C is the surgical surface change. The other three
`lower_*_callable_app` functions also need the same treatment.

Looking at `lower_plain_callable_app` more carefully: the args-then-body
split is already structural — lines 3072–3099 handle args, lines
3100–3107 handle the body. The split is mechanical to extract.

Let me weigh: option A (insert/remove around body inside each
`lower_*_callable_app`) is also fine and arguably cleaner because each
callable shape can have its own body-lowering boundary.

Going with **Option A**. The shape:

1. `try_lower_callable_app` no longer touches `inlining_names`. It just
   resolves the callable and dispatches.

2. Each `lower_*_callable_app` accepts an optional `inlining_name`
   parameter (the resolved-terminal name). Right before lowering the
   body — *after* args are evaluated/bound — they do
   `self.inlining_names.insert(name.clone())`. After body lowering
   (before restoring `bindings`/`local_callables`/etc.), they remove.

3. The `inlining_name` parameter comes from a new helper
   `resolve_inlining_target_name(func)` called in `try_lower_callable_app`
   alongside `resolve_callable_expr(func)`. Same chain-walk logic.

For the recursion-guard discriminator in `resolve_callable_expr_inner`,
no change needed: the existing `inlining_names.contains(&name)` check
still fires when entering an inlining-in-progress callable from a fresh
recursion. With the insert site moved to "post-args," arg-lowering is
free to re-call the same callable.

## Verification against the test fixtures

**Target (`nested_fn_param_call_lowers_via_substituted_callable`):**

1. Outer `f(f(x))` call: `try_lower_callable_app(f, [f(x)])`. Resolves
   to `Plain(doubler_body)`. Resolved-terminal-name = `"doubler"`.
   Dispatches to `lower_plain_callable_app(doubler_body, [f(x)],
   inlining_name = Some("doubler"))`.
2. `lower_plain_callable_app` evaluates args first. The `f(x)` arg
   triggers nested `try_lower_callable_app(f, [x])`. At this point
   `inlining_names` is **empty** (the outer call hasn't inserted yet).
   Resolves to `Plain(doubler_body)`. Dispatches to
   `lower_plain_callable_app(doubler_body, [x], inlining_name = Some("doubler"))`.
3. Inner `lower_plain_callable_app` evaluates the single arg `x` (a
   tensor binding — no recursion). Inserts `"doubler"` into
   `inlining_names`. Lowers `doubler`'s body (`add(x, x)`) — no recursion
   inside. Removes `"doubler"`. Returns `add(x, x)` = `doubler(x)`.
4. Back in step 2's `lower_plain_callable_app`: the `f(x)` arg has now
   been lowered to `doubler(x)`. Args done. Inserts `"doubler"` into
   `inlining_names`. Lowers outer doubler's body (`add(x, x)` where
   `x = doubler(seed)`). Result: `add(doubler(seed), doubler(seed)) =
   doubler(doubler(seed)) = 4 * seed`. Removes `"doubler"`.
5. Result: `[4, 8, 12]`.

**Control (`true_self_recursion_still_rejected_by_inlining_guard`):**

1. `loop_self(seed)` call: `try_lower_callable_app(loop_self, [seed])`.
   Resolves to `Plain(loop_self_body)`. Terminal name = `"loop_self"`.
   Dispatches to `lower_plain_callable_app(loop_self_body, [seed],
   inlining_name = Some("loop_self"))`.
2. Args evaluated: `seed` is a tensor binding, no recursion.
3. Insert `"loop_self"` into `inlining_names`. Lower body `(app
   loop_self (var x))` (where `x = seed`).
4. Body's `try_lower_callable_app(loop_self, [x])`: `resolve_callable_expr(loop_self)`
   walks the `var` arm: guard checks `inlining_names.contains("loop_self")`
   → **true** → returns `None`. `try_lower_callable_app` returns `None`.
   Fallback: lower args and return last. Returns `x`.
5. Body lowered to `x`. Remove `"loop_self"` from `inlining_names`.
6. Result: `seed = [1, 2, 3]`. Same as today — the control still
   passes.

## Sibling-sweep findings

Walked the workspace for other places that mirror this pattern.

### `resolve_callable_expr_inner` (`lower.rs:2823`)

The dual guards `!visited.insert(name)` and `inlining_names.contains(&name)`:

- `visited` is per-resolution alias-cycle protection. Correct.
  Resolves a different concern (cycle within a single `resolve_callable_expr`
  walk, not across recursive lowering invocations).
- `inlining_names` is the cross-call inlining-in-progress guard. Bug
  is in *where it gets populated*, not in the check itself.

### `inline_ctx.local_callables.insert(...)` at `lower.rs:5907`

Audited: the `inline_ctx` block at `lower.rs:5876`–`5915` is inside
`#[cfg(test)]` (the `tests` module starting at `lower.rs:5201`). It is
a test scaffold, not a production inliner. Each `inline_ctx` is a
fresh `LowerCtx::new(...)` with empty `inlining_names`. Not affected
by the bug.

### `crates/chelis-ir/src/host.rs`

Searched for `inlining_names` and `inlining_name` in `host.rs`. No
matches. The host lane does not maintain a parallel recursion guard;
the IR-lane guard is the only one in the workspace.

### `lower_let` / `local_callables.insert` (`lower.rs:2561`, `lower.rs:2602`)

These insert into `local_callables` during let-binding. They don't
touch `inlining_names`. Not affected.

### Recommendation

Document the discriminator literally in code-adjacent comments at the
insert/remove sites and at the guard-check site. The bug surfaced
because the comment at the insert site (`lower.rs:2764`–`2767`) describes
the *intent* ("track it on the inlining stack") but doesn't define what
"it" means precisely — which makes the call-site-alias-vs-resolved-target
distinction invisible to a future reader.

## Surface size

- One new helper: `resolve_inlining_target_name`
  (~15 lines, mirrors the existing `var`-arm walk in
  `resolve_callable_expr_inner`).
- `try_lower_callable_app`: stop inserting/removing into
  `inlining_names`; instead pass the resolved-terminal-name through to
  the four `lower_*_callable_app` dispatch arms.
- Four `lower_*_callable_app` functions: accept an `inlining_name:
  Option<String>` parameter and wrap their body-lowering step with the
  insert/remove pair.
- Comments updated at the insert sites and the guard-check site to
  state the discriminator literally.

Estimated diff: ~60 lines net. No new state in `LowerCtx`. No spec edit.

## Sibling-sweep close-out

Audited during this diagnosis (see "Sibling-sweep findings" above):

- IR-lane: `try_lower_callable_app` is the only insert site for
  `inlining_names`. Only fix surface.
- Host-lane (`crates/chelis-ir/src/host.rs`): no parallel guard.
- Test-scaffold `inline_ctx` (`lower.rs:5876`–`5915`): fresh
  `LowerCtx::new`, immune.

No other code in the workspace participates in the recursion-guard
pattern. The fix surface in `try_lower_callable_app` and the four
`lower_*_callable_app` functions is complete.

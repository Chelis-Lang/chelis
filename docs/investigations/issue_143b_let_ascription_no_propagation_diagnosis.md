# Let-binding ascription not propagated to RHS diagnosis (chelis#143 sub-issue B)

Diagnosis pass for the second chelis#143 sub-bug surfaced during
PR #151 red-team: an ascribed let-binding's declared type is never
unified with its RHS expression during inference, so a generic builtin
RHS like `to_tensor(...)` leaves its output type variable free.

Cross-references:

- Issue: `chelis#159` (this sub-issue); parent `chelis#143`
- Sibling diagnosis: `docs/investigations/issue_143a_to_tensor_shape_erasure_diagnosis.md`
- Design note: `docs/investigations/issue_143_pad_sequences_to_design_note.md`
- Pinning test: `crates/chelis-types/tests/issue_143b_let_ascription_no_propagation.rs`
  (three regression tests; the formerly-`#[ignore]` probe is now passing
  under the prototype patch — see "Prototype validation" below)
- Prototype patch: `crates/chelis-types/src/infer.rs::infer_let` (~30 lines added)
- Spec: `spec/04-type-system.md` (no explicit rule for let-binding
  ascription propagation today; the existing §5.6 element-type
  narrowing covers only Position 1 *literal* RHS, not generic-builtin RHS)

The prototype patch lands in this branch — see "Prototype validation"
at the end.

## Bug surface

A let-binding ascription does not constrain its RHS expression's
inferred type during `infer_let`. Concretely:

```chelis
def caller() -> tensor[3, f32] = {
  a: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])
  b: tensor[5, f32] = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
  pair_id(&a, &b)
}
```

The user clearly states `a` has dims `[3]` and `b` has dims `[5]`. The
sig `pair_id: &tensor[n, f32] -> &tensor[n, f32] -> ...` demands a
shared `n`, so the call `pair_id(&a, &b)` should trip
`DimensionMismatch`. Today the check produces only an unrelated
return-type `TypeMismatch`:

```
[TypeMismatch] def 'caller' body doesn't match declared signature:
  body has type `() -> &tensor[Var(DimVar(40)), f32]`,
  declared type is `() -> tensor[Lit(3), f32]`
```

The sig dim var `Var(DimVar(40))` is unbound, exactly because `a` and
`b` carry `Type::Var(output_a)` and `Type::Var(output_b)` from
`to_tensor`'s `∀α β. α → β` scheme — never unified with the user's
ascription.

The parameter-ascription path is unaffected and behaves correctly (see
`issue_143b_let_ascription_no_propagation.rs::parameter_ascription_path_works_today`).
The bug is specifically in the let-binding ascription path.

## Bug site

The bug has **two distinct desugar paths**, both of which lose the
ascription before it reaches inference. The probe test in this branch
exercises the block-scoped path; the top-level path is a related but
not-yet-fixed sub-bug.

### 1a. Top-level let desugar emits a sibling `defsig`

`crates/chelis-surf/src/desugar.rs:664-682` (`Decl::LetDef` with
`ty: Some(t)`):

```rust
Decl::LetDef {
    name,
    ty: Some(t),
    value,
    ..
} => {
    // ... element-type contextual narrowing for list literals ...
    vec![
        node("defsig", vec![sym(name), desugar_type(t)]),
        node("def", vec![sym(name), body]),
    ]
}
```

The Deep program contains the declared type as a sibling `defsig`
node next to the `def`. The annotation pass at `infer.rs:45-99`
(`install_declared_sig_param_types`, `collect_defsig_param_types`)
picks up `defsig` entries but filters to `t-fn` only at line 88
(`if get_tag(fn_list) != Some("t-fn")`), so let-binding ascriptions
on non-fn types (tensor, prim, adt, etc.) are silently dropped.

### 1b. Block-scoped let desugar injects `"type"` metadata onto the RHS

`crates/chelis-surf/src/desugar.rs:1353-1393` (`desugar_let_bindings`):

```rust
if let Some(ty) = &binding.ty {
    out = bind_name_value(
        name,
        inject_type_metadata(value, desugar_type(ty)),
        out,
    );
}
```

`inject_type_metadata` (`desugar.rs:314-335`) embeds the type
expression as the `"type"` key of the RHS node's metadata map. No
sibling `defsig` is emitted for this path.

This is the path my probe test exercises: ascriptions appear inside
`{ ... }` blocks (function bodies), not as top-level declarations.

### 2. `infer_let` ignores both desugar paths

`crates/chelis-types/src/infer.rs:10708-10760`:

```rust
fn infer_let(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    ...
) -> Type {
    ...
    while i + 1 < bind_children.len() {
        if let Some(name) = symbol_name(&bind_children[i]) {
            let expr_ty = infer_expr(
                &bind_children[i + 1],
                &mut let_env,
                vg,
                subst,
                ...
            );
            let scheme = let_env.generalize(&expr_ty, subst);
            let_env.bind(name.to_string(), scheme);
        }
        i += 2;
    }
    ...
}
```

The function:

- Receives the binding name and the RHS expression.
- Calls `infer_expr` on the RHS with no expected-type hint.
- Generalizes the result and binds it.

There is no parameter for, or lookup of, the declared type. For the
**top-level** path (1a) the sibling `defsig` is filtered out by the
annotation pass's `t-fn` guard. For the **block-scoped** path (1b)
the `"type"` metadata sits on the RHS list node itself, but
`infer_expr` only consults that metadata for `lit` nodes
(`infer_lit` at `infer.rs:6087-6193`); for `app` / `var` / `let` /
other tag handlers, the metadata is silently ignored.

### 3. The parameter-ascription path uses a different mechanism

`crates/chelis-types/src/infer.rs:10542-10641` (`infer_def_body_with_sig`):
top-level `def` sites look up the declared sig type from the env, seed
the bare parameters with the declared types, and unify the body
against the declared return type. That's why
`def caller(a: &tensor[3, f32]) -> ...` works.

The let-binding ascription has equivalent declared-type information
available (the desugared `defsig`), but `infer_let` doesn't have an
analogous seeding step.

Per the Explore-agent investigation that mapped this, there is a
thread-local `DECLARED_SIG_PARAM_TYPES` map installed during the
annotation pass at the top of `infer.rs:45-53`. This map is queried
during top-level def inference for parameter types. A symmetric
local-bind map (or a direct lookup into the desugared sibling `defsig`)
would close the gap.

## Option survey

### (a) Teach `infer_let` to consult the desugared `defsig` — recommended

In `infer_let`, for each `(name, rhs)` pair, check whether a sibling
`defsig` exists for `name` in the surrounding scope. If yes, look up
its declared type and seed the RHS inference with that as the expected
type. After `infer_expr` returns the RHS's inferred type, unify it
against the declared type before generalizing.

Implementation sketches:

- The declared type for a top-level `defsig` is already discoverable
  via the env binding established during the annotation pass
  (`infer.rs:2543-2791`). Reuse that machinery: `env.lookup(name)`
  returns the declared scheme if one exists.
- For let-bindings inside a function body, the `defsig` emitted by
  desugar at `crates/chelis-surf/src/desugar.rs:680` lives in the same
  Deep block as the `def`. `infer_let` already iterates `bind_children`;
  it should accept an analogous `bind_sig_children` slice produced by
  the desugarer, or look the sig up via name.

Scope estimate: small-to-medium. ~30-50 lines in `infer_let` plus
either a desugar-side adjustment to expose the sibling sigs or an
env-lookup helper. New unit tests covering: ascribed let with literal
RHS (already works via §5.6 narrowing), ascribed let with generic
builtin RHS (the case this fixes), ascribed let with mismatched RHS
(should error), unascribed let (no change).

### (b) Emit ascribed lets as `(load name : T = expr)` instead of `(defsig ...) (def ...)`

A single Deep node that carries the ascription inline would make
inference trivial: the type checker sees a load with declared type T
and unifies the inferred RHS type against T at the load site. This is
a larger refactor — the desugarer's current shape is reused across
several callers, and the lint/format/decompile paths all depend on the
current Deep encoding.

Scope estimate: large. Multi-crate surface change. Rejected for this
sub-issue but worth noting if the Deep AST is ever re-evaluated for
unrelated reasons.

### (c) Defer (do nothing)

The user-facing footgun is real: `q: tensor[3, f32] = ...` quietly
fails to enforce the declared shape. Even an experienced user
inspecting the type-checked program would not see the ascription as
inert. Rejected.

## Recommendation

Pursue (a). Land it before sub-issue (A) so users have a working
workaround (manual ascription) while (A)'s desugar work is in
progress. The fix is local to `infer_let` and the desugar emission for
ascribed lets — no spec change required, since the property "an
ascription means what it says" is already implicit in the spec's
treatment of types.

Order of operations:

1. Add the ascription-lookup helper (or expose the sibling sig
   through `bind_children`).
2. Unify the RHS inferred type against the declared type before
   generalizing in `infer_let`.
3. Remove the `#[ignore]` from
   `crates/chelis-types/tests/issue_143b_let_ascription_no_propagation.rs::let_binding_ascription_should_propagate_to_to_tensor_rhs`
   and confirm it passes.
4. Add negative tests: ascribed let with explicitly-mismatched RHS
   (e.g. `q: tensor[3, f32] = some_function_that_returns_tensor[5, f32]`)
   must error.

## Interaction with sub-issue (A)

(B) fixed without (A) means users get a working workaround: write
ascriptions explicitly. Ergonomics regress relative to the desired
end state but the static catch lands.

(A) fixed without (B) leaves a smaller hole: any user who *does*
ascribe gets the ascription silently ignored when the RHS is generic.

Both should ship; (B) is the smaller change and unblocks the
workaround pattern, so land it first.

## Prototype validation

The block-scoped path (1b) is patched in this branch's prototype
commit. The fix lives in `crates/chelis-types/src/infer.rs::infer_let`
(~30 lines inside the existing `while i + 1 < bind_children.len()`
loop): after `infer_expr` returns the RHS type, the code looks for a
`"type"` key in the RHS list's metadata and, if present, converts
it to a `Type` via `deep_type_to_resolved_type` and `unify`s the
inferred type against the declared type before `generalize` + `bind`.
A failed unify pushes a `CheckError` carrying the binding name, the
declared type, and the inferred RHS type.

### Results

- **Probe test:** `crates/chelis-types/tests/issue_143b_let_ascription_no_propagation.rs::let_binding_ascription_propagates_to_to_tensor_rhs`
  was `#[ignore]`'d in the previous commit (asserted desired behavior
  that did not hold). Under the prototype patch the test passes, so
  the `#[ignore]` attribute has been dropped and the test name was
  rewritten to drop the now-stale "should" framing.
- **Repo gate:** `python3 scripts/gate.py` exits 0 on the prototype
  branch (3131/3131 nextest tests pass, lint/clippy/fmt clean).
- **Two latent test bugs surfaced and were fixed.** They were
  previously masked by the bug this prototype fixes:
  - `crates/chelis-types/tests/linearity.rs::tensor_to_scalar_does_not_consume_tensor_input`
    and `::tensor_to_scalar_still_flags_use_after_genuine_consume`
    both wrote `v: f64 = tensor_to_scalar(x)` where `x: tensor[f32]`.
    `tensor_to_scalar` has custom inference at `infer.rs:7323-7363`
    that returns the input tensor's precision (`f32` here), so the
    `f64` ascription was always wrong but silently dropped. The
    fix correctly reports `PrecisionMismatch`; the tests are updated
    to `v: f32 = ...` so they correctly exercise the linearity
    behavior they claim to test.
  - `crates/chelis-types/tests/signature_inference.rs::later_helper_does_not_retroactively_make_earlier_unconstrained_call_read_only`
    had `z: tensor[4, f32] = helper(a, b)` in a test whose name
    asserted "unconstrained call." Pre-fix the ascription was
    dropped so `z` stayed unconstrained; post-fix the ascription
    correctly propagates and constrains `z`, which through `add(a, z)`
    propagates to `a`, which then makes `a` read-only-inferrable.
    The test source was updated to drop the ascription on `z`,
    matching the test's stated intent. Test passes.

### Scope: what the prototype does and does not cover

**Covered (block-scoped lets, path 1b):**
- `def f(...) = { name: T = expr; ... }` ascriptions now propagate.
- Generic-builtin RHSes like `to_tensor(...)`, `pad_sequences_to(...)`,
  `tensor_to_scalar(...)`, etc. are correctly constrained by the
  ascription.
- A mismatch between ascription and inferred RHS now reports a
  diagnostic (`PrecisionMismatch`, `DimensionMismatch`, `TypeMismatch`
  per the unify outcome) with the binding name in the message.

**Not covered (top-level lets, path 1a) — tracked in chelis#162:**
- `name: T = expr` at module scope still routes through `Decl::LetDef`
  and emits a sibling `defsig`. The annotation pass filters those to
  `t-fn` only, so top-level non-fn ascriptions remain inert. A
  follow-up patch should either:
  - Broaden `collect_defsig_param_types` to also gather non-fn
    defsigs into a parallel map and have `infer_let` (or the top-level
    `def` handler) consult it; or
  - Change the top-level `Decl::LetDef` desugar to use
    `inject_type_metadata` like the block-scoped path, unifying the
    two paths.

The second option is cleaner — it unifies the two paths so future
work doesn't have to handle them separately. The risk is that other
passes (lint, decompile, format) may consume the sibling-defsig
shape and would need updating. Either way, the work is
straightforward once block-scoped is settled. Probe tests for the
top-level case live with the follow-up patch (chelis#162), not this
branch.

### Side observation

The micro-test `unify_dim_cannot_recover_lit_from_two_var_tensors`
added to `crates/chelis-types/src/unify.rs` (Phase 3) corroborates
that this fix shape is the right one. That test shows from first
principles that the sig machinery cannot manufacture a `Lit` from
`Var`/`Wildcard` inputs alone — any fix that wants the sig dim
contract to fire must inject the concrete dim *upstream*. For
ascribed let-bindings, "upstream" means right after the RHS
inference, which is exactly what this prototype does.

### Dim-name capture behavior (review comment 4)

A reviewer asked whether the prototype's call to
`deep_type_to_resolved_type(ty_expr, vg, adt_reg, &mut HashMap::new())`
correctly shares dim variables with the enclosing function signature,
or creates fresh ones per let-binding. The answer turns out to be
"neither, because the question is upstream of the wrong abstraction."

`deep_type_to_resolved_type`'s `&mut HashMap` argument is a
`HashMap<String, TypeVar>` — a **type variable** map. Dim variables
are not threaded through it. Dim names in user-written types parse
to `Dim::Name(String)` (`infer.rs:12007-12013`), i.e. string labels,
not capturing variables. At unification time (`unify.rs::unify_dim`):

- `Dim::Name(n1) ↔ Dim::Name(n2)` succeeds iff `n1 == n2`.
- `Dim::Name(n) ↔ Dim::Var(v)` binds `v := Name(n)`.

So `n` in a let-ascription is a label that unifies by string equality
for `Name ↔ Name` and binds any fresh `Var` to itself. It does NOT
capture the enclosing sig's `n` per-binding; cross-position contracts
within a sig are enforced by the sig's instantiation step (one fresh
dim var per quantified name per instantiation, shared across all
positions that named the same dim), not by let-ascription naming.

Observed behavior, pinned in
`crates/chelis-types/tests/issue_159_let_ascription_dim_capture.rs`:

1. **Matching sig dim names succeed.** Sig `&tensor[n, f32] -> &tensor[n, f32]`
   with body `x: &tensor[n, f32] = a; y: &tensor[n, f32] = b`
   typechecks cleanly. The shared sig dim var (instantiated once for
   both params) accepts the user's `Name("n")` on both bindings.

2. **Concrete-dim mismatches still error.** Declaring `x: &tensor[3, f32] = t`
   where `t: &tensor[2, f32]` reports `DimensionMismatch` via the
   chelis#159 diagnostic template.

3. **Dim names do NOT enforce sig-cross-position contracts via
   ascription naming.** With a sig `&tensor[n, f32] -> &tensor[m, f32]`
   (distinct dim names), writing both let-ascriptions as `n` does
   NOT produce a `DimensionMismatch` — both fresh sig vars
   independently bind to `Name("n")`. The user-intended contract
   (both `x` and `y` have the same dim) is not enforced. This is the
   pre-existing language semantics, not a bug introduced by chelis#159.

If the design ever needs to change so that let-ascription dim names
truly capture the enclosing sig's dim vars, the trip-wire is test 3
in that probe file. Today the test is a positive assertion of "no
error"; if the language semantics flip, the test would need to be
inverted along with the surrounding implementation.

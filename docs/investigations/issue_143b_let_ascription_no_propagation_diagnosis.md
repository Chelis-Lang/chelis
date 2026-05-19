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
  (one `#[ignore]` probe + two passing counter-probes)
- Spec: `spec/04-type-system.md` (no explicit rule for let-binding
  ascription propagation today; the existing §5.6 element-type
  narrowing covers only Position 1 *literal* RHS, not generic-builtin RHS)

No code changes in this commit.

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

Three pieces of evidence cooperate:

### 1. Desugar emits the ascription as a `defsig`

`crates/chelis-surf/src/desugar.rs:664-682`:

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

So the ascription survives desugar. The Deep program contains the
declared type as a sibling `defsig` next to the `def`.

### 2. `infer_let` ignores the declared sig

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

There is no parameter for, or lookup of, the declared sig type. The
ascription that desugar carefully preserved is never consulted.

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

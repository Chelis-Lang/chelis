# Issue #220: `extract_dim_list` reads Deep tag as dim name

## Symptom

`extract_dim_list` in `crates/chelis-ir/src/lower.rs` (the helper that
the IR `reshape` lowering arm calls to interpret the shape list) walks
`list.elements` directly and treats any `Atom::Symbol` as a dim name:

```rust
fn extract_dim_list(&self, expr: &Expr) -> Option<Vec<DimInfo>> {
    if let Expr::List(list, _) = expr {
        let mut dims = Vec::new();
        for elem in &list.elements {
            match elem {
                Expr::Atom(Atom::Int(n), _) => dims.push(DimInfo::Lit(*n as usize)),
                Expr::Atom(Atom::Symbol(name), _) => {
                    dims.push(DimInfo::Named(name.clone(), None));
                }
                _ => {}
            }
        }
        if !dims.is_empty() {
            return Some(dims);
        }
    }
    None
}
```

Surface syntax `[cast(2, int64), cast(3, int64)]` desugars to a
Cons-chain:

```
(app {} (var {} Cons)
        (cast {} (lit {} 2) (t-prim {} int64))
        (app {} (var {} Cons)
                (cast {} (lit {} 3) (t-prim {} int64))
                (var {} Nil)))
```

Element 0 of that outer `(app ...)` is the literal tag symbol `"app"`.
The walker pushes `DimInfo::Named("app", None)`, then bails. The
resulting `RiscOp::Reshape { new_shape: [Named("app", None)] }` carries
a bogus symbolic dim that downstream passes treat as a real dim
variable, eventually tripping `dag::symbolic_occurrences`'s panic
("symbolic dim 'app' is referenced by a non-Load node…").

## Reachability on current `main`

The original issue body's Surf reproducer does not currently trigger
because `expr_requires_host_runtime`
(`crates/chelis-ir/src/lower.rs:1203`) classifies any expression
containing an uppercase `(var Cons)` / `(var Nil)` as host-runtime. A
Cons-chain reshape arg therefore routes the entire enclosing def to
host eval (`chelis_host_reshape_tensor`), never invoking the IR
`reshape` arm that would call `extract_dim_list`. Re-running the issue
body's reproducer also hits an earlier `ArityMismatch` on
`grad(g, wrt=x)` (the keyword-arg form is no longer accepted by the
current type-checker arity rule), so the panic at `dag.rs:1108` is
masked twice over.

The defect is still reachable through paths that bypass the
host-runtime classifier:

1. `lower_subexpr_program` lowers a Deep expression directly via
   `LowerCtx::lower_expr`, without consulting `expr_requires_host_runtime`.
   The new test
   `crates/chelis-ir/tests/issue_220_extract_dim_list_cons_chain.rs`
   uses this entry point and reproduces the defect deterministically.
2. The crate's `parse_and_lower_unchecked` test helper does the same.
3. Future lowering paths or tooling that bypass the host-runtime
   filter, or any future shift that lets the IR `reshape` arm see
   Cons-chain args (e.g., an effects/host-runtime classifier
   refinement that no longer diverts Cons-chains), would activate the
   defect end-to-end.

The fix is therefore landed as defense in depth: the IR `reshape` arm
is public surface reachable by tooling and future paths, and
`extract_dim_list` is the only consumer of its shape-list arg.

## Fix shape

Rewrite `extract_dim_list` to:

1. Peel the Cons chain via a shared `collect_cons_chain` helper.
2. Interpret each head via `extract_int_for_dim`
   (`Atom::Int` / `(lit {} N)` / `(cast {} <int|lit> <prim>)`) or
   `symbolic_dim_var_name` (`(var {} <name>)`).
3. Return `None` for unrecognized shapes so the caller falls back to
   `ty.dims`.

The helper names and structure mirror the original PR #211 R3 patch
(commit `df84382` on the issue-199 branch, preserved as the local tag
`issue-199-r3-backup`); the lift verifies the patch still applies
cleanly against current `main` (the helpers `get_tag`, `children`, and
`symbol_name` exist; `expr_is_var_named` is replaced by the
already-present `is_app_of_builtin` so the new code does not duplicate
the existing builtin-check helper).

## Sibling sweep

After lifting the fix, the only direct caller of `extract_dim_list` in
`lower.rs` is the `"reshape"` arm at line 4680. The companion helper
`extract_pair_list` (line 5017) already handles both the raw
`(list ...)` form and the Cons-chain form via `cons_chain_pair_list`,
so it does not share the defect. `extract_usize_list` reads `args[1..]`
positionally (post-Cons-desugar) and is not affected.

Within the broader `lower.rs`, no other helper walks
`list.elements` of a Cons-chain `(app ...)` and matches `Atom::Symbol`
as a semantic value. The defect is unique to `extract_dim_list`.

## Oracle

`crates/chelis-ir/tests/issue_220_extract_dim_list_cons_chain.rs`. The
two test cases (`reshape_cons_chain_does_not_synthesize_tag_named_dim`,
`reshape_cons_chain_extracts_integer_dim_list`) fail on `main` before
the fix and pass after.

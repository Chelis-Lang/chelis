//! WS-5 Part B (walker-guard coverage): every self-recursive `deep::Expr` /
//! `deep::Pat` walker in `infer.rs` must carry the `stack_guard!` macro so the
//! recursive type checker bails before exhausting the native stack on a
//! deeply-nested program (WI-1, spec/design/verification_stack_master_plan.md
//! \u{00A7}4.1). This is a SOURCE-SCANNING invariant: it parses `infer.rs` with
//! `syn`, enumerates the functions that take a `&deep::Expr` / `&deep::Pat`
//! argument and call themselves, and asserts each body invokes `stack_guard!`.
//! A NEW unguarded recursive walker therefore fails this test rather than
//! silently reintroducing an unbounded-recursion stack-overflow surface.

use syn::visit::{self, Visit};
use syn::{Block, ExprMacro, File, ImplItemFn, ItemFn, Macro, Signature, Type};

/// The Rust source under analysis. Compiled in at build time so the test has
/// no filesystem dependency on the crate layout at run time.
const INFER_SRC: &str = include_str!("../src/infer.rs");

/// One analyzed walker function and the facts the scan derived about it.
#[derive(Debug)]
struct WalkerFn {
    name: String,
    /// Takes a `&deep::Expr` / `&deep::Pat` (the recursive AST surface).
    walks_deep_ast: bool,
    /// Calls itself somewhere in its body (self-recursion).
    self_recurses: bool,
    /// Body invokes the `stack_guard!` macro.
    has_stack_guard: bool,
}

impl WalkerFn {
    /// A function that descends the deep AST AND self-recurses is a walker
    /// that must be guarded.
    fn must_be_guarded(&self) -> bool {
        self.walks_deep_ast && self.self_recurses
    }
}

/// Does this type mention the recursive deep AST surface (`deep::Expr` or
/// `deep::Pat`)? The scan is textual over the type's tokens, which is robust
/// to `&`, `&mut`, `Option<&...>`, slices, and other wrappers -- any parameter
/// whose type references the deep AST node counts as "walks the deep AST".
fn type_mentions_deep_ast(ty: &Type) -> bool {
    let rendered = quote_type(ty);
    rendered.contains("deep :: Expr") || rendered.contains("deep :: Pat")
}

/// Render a type to a normalized token string (`syn`'s `ToTokens` spacing) so
/// the textual match above is independent of source whitespace.
fn quote_type(ty: &Type) -> String {
    use quote::ToTokens;
    ty.to_token_stream().to_string()
}

/// Visitor that records whether a function body (a) calls a given name and
/// (b) invokes `stack_guard!`.
struct BodyScan<'a> {
    own_name: &'a str,
    calls_own_name: bool,
    invokes_stack_guard: bool,
}

impl<'a, 'ast> Visit<'ast> for BodyScan<'a> {
    fn visit_macro(&mut self, mac: &'ast Macro) {
        if macro_is_stack_guard(mac) {
            self.invokes_stack_guard = true;
        }
        // A `stack_guard!` (or any) macro's own token stream can reference the
        // function name; still descend so a recursive call written inside
        // another macro (e.g. a logging macro) is seen.
        visit::visit_macro(self, mac);
    }

    fn visit_expr_macro(&mut self, node: &'ast ExprMacro) {
        if macro_is_stack_guard(&node.mac) {
            self.invokes_stack_guard = true;
        }
        visit::visit_expr_macro(self, node);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        // A self-call appears as a path whose final segment is the function's
        // own name. This also matches a bare `name(...)` call (single-segment
        // path) and a qualified `Self::name` / `module::name`.
        if let Some(last) = path.segments.last()
            && last.ident == self.own_name
        {
            self.calls_own_name = true;
        }
        visit::visit_path(self, path);
    }
}

fn macro_is_stack_guard(mac: &Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "stack_guard")
}

/// Analyze one function signature + body into a [`WalkerFn`]. The own-name
/// recursion scan deliberately runs on the body only (not the signature), so a
/// parameter named after the function does not count as a self-call.
fn analyze_fn(sig: &Signature, block: &Block) -> WalkerFn {
    let name = sig.ident.to_string();
    let walks_deep_ast = sig.inputs.iter().any(|arg| match arg {
        syn::FnArg::Typed(pat_type) => type_mentions_deep_ast(&pat_type.ty),
        syn::FnArg::Receiver(_) => false,
    });

    let mut scan = BodyScan {
        own_name: &name,
        calls_own_name: false,
        invokes_stack_guard: false,
    };
    scan.visit_block(block);
    let self_recurses = scan.calls_own_name;
    let has_stack_guard = scan.invokes_stack_guard;

    WalkerFn {
        name,
        walks_deep_ast,
        self_recurses,
        has_stack_guard,
    }
}

/// Collect every free function and impl method in the file, descending into
/// modules and impl blocks. Nested functions (closures aside) are reached via
/// the item visitor.
struct FnCollector {
    walkers: Vec<WalkerFn>,
}

impl<'ast> Visit<'ast> for FnCollector {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.walkers.push(analyze_fn(&node.sig, &node.block));
        visit::visit_item_fn(self, node);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        self.walkers.push(analyze_fn(&node.sig, &node.block));
        visit::visit_impl_item_fn(self, node);
    }
}

fn collect_walkers() -> Vec<WalkerFn> {
    let file: File = syn::parse_file(INFER_SRC).expect("infer.rs parses as Rust");
    let mut collector = FnCollector {
        walkers: Vec::new(),
    };
    collector.visit_file(&file);
    collector.walkers
}

/// Recursive deep walkers the scan flags as self-recursive but which are NOT
/// required to carry `stack_guard!`, each for a documented, category-coded
/// reason. This is the explicit, per-entry-justified allowlist (not free-form
/// prose): a new self-recursive deep walker that is NOT in this list and lacks
/// a guard fails `every_self_recursive_deep_walker_carries_stack_guard`, so the
/// "no new unguarded walker" property holds even though the set is not yet
/// complete.
///
/// Discovered while building this scan (WS-5): the historically-guarded set was
/// NOT the complete set of recursive deep walkers. The genuinely arbitrary-
/// depth production walkers below (`AST-DEPTH`) are a real WI-1 coverage gap and
/// are tracked for a guard follow-up; they are allowlisted here only so this
/// test can lock the regression boundary today rather than silently asserting a
/// completeness the tree does not have.
const GUARD_EXEMPT_WALKERS: &[(&str, &str)] = &[
    // FALSE-POSITIVE: not an AST-depth recursion. `visit` is a DFS over a
    // name-keyed call graph with a `visiting`/`visited` HashSet cycle guard;
    // its depth is bounded by the finite number of def names, not by AST
    // nesting, and it takes `&deep::Expr` only in lookup maps. It cannot
    // stack-overflow on a deep AST, so `stack_guard!` does not apply.
    (
        "visit",
        "FALSE-POSITIVE: cycle-guarded call-graph DFS, not AST-depth",
    ),
    // MODULE-ONLY: descends through `(module ...)` wrappers only (skips every
    // non-module node). Surf emits one module per file and reef strips
    // wrappers before inference, so these do not nest deeply in practice --
    // the same bounded-in-practice rationale the GUARDED
    // `collect_defsig_param_types` documents. Tracked for a uniform guard.
    ("push", "MODULE-ONLY: descends (module ...) wrappers only"),
    ("walk", "MODULE-ONLY: descends (module ...) wrappers only"),
    // TEST-HELPER: defined inside `#[cfg(test)] mod tests`, not production
    // checker code; not on the user-input path that the stack budget protects.
    (
        "missing_shape_sensitive_app",
        "TEST-HELPER: lives in #[cfg(test)] mod tests",
    ),
    // AST-DEPTH: genuine arbitrary-depth production walkers that SHOULD carry
    // `stack_guard!` (real WI-1 coverage gap, tracked for a follow-up). Listed
    // explicitly so the gap is visible rather than hidden, and so adding the
    // guard later simply removes the entry.
    (
        "type_expr_has_tensor_prec_var",
        "AST-DEPTH gap: recurses over t-fn/t-tuple/t-adt children",
    ),
    (
        "literal_static_value",
        "AST-DEPTH gap: recurses on (lit ...) children",
    ),
    (
        "tensor_dim_exprs_from_type_expr",
        "AST-DEPTH gap: recurses through nested (t-ref ...) wrappers",
    ),
    (
        "tensor_precision_expr",
        "AST-DEPTH gap: recurses through nested (t-ref ...) wrappers",
    ),
    (
        "tensor_dims_from_type_expr",
        "AST-DEPTH gap: recurses through nested (t-ref ...) wrappers",
    ),
    (
        "top_level_arm_is_irrefutable",
        "AST-DEPTH gap: recurses on (pat-as ...) inner pattern",
    ),
];

fn is_guard_exempt(name: &str) -> bool {
    GUARD_EXEMPT_WALKERS.iter().any(|(n, _)| *n == name)
}

#[test]
fn every_self_recursive_deep_walker_carries_stack_guard() {
    let walkers = collect_walkers();

    // The functions that must be guarded: they descend the deep AST and call
    // themselves. Any such function lacking `stack_guard!` AND not on the
    // documented exemption list is a regression -- in particular, a NEW
    // unguarded recursive walker (not yet allowlisted) fails here.
    let unguarded: Vec<&str> = walkers
        .iter()
        .filter(|w| w.must_be_guarded() && !w.has_stack_guard && !is_guard_exempt(&w.name))
        .map(|w| w.name.as_str())
        .collect();

    assert!(
        unguarded.is_empty(),
        "self-recursive deep::Expr/deep::Pat walkers missing a `stack_guard!` \
         (WI-1: every recursive checker walker must bail before the native \
         stack overflows -- add `stack_guard!(\"<name>\", <expr>, <bail>);` at \
         the top of each, or add a category-coded entry to \
         GUARD_EXEMPT_WALKERS with the reason it does not apply): {unguarded:?}"
    );

    // The allowlist must not rot: every exempt name must STILL be a flagged
    // walker. If a guard was added (or the function deleted/renamed), the stale
    // entry must be removed so the exemption set stays minimal and honest.
    let flagged: std::collections::HashSet<&str> = walkers
        .iter()
        .filter(|w| w.must_be_guarded() && !w.has_stack_guard)
        .map(|w| w.name.as_str())
        .collect();
    let stale: Vec<&str> = GUARD_EXEMPT_WALKERS
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !flagged.contains(name))
        .collect();
    assert!(
        stale.is_empty(),
        "GUARD_EXEMPT_WALKERS has stale entries no longer flagged as \
         unguarded recursive walkers (guard added, or fn renamed/removed) -- \
         delete them: {stale:?}"
    );

    // Pin the current guarded-walker set so the scan is proven to have teeth:
    // it must actually be classifying a non-trivial number of functions as
    // guarded walkers (not silently matching zero and passing vacuously).
    let guarded_walkers = walkers
        .iter()
        .filter(|w| w.must_be_guarded() && w.has_stack_guard)
        .count();
    assert!(
        guarded_walkers >= 20,
        "expected the scan to find the established set of guarded recursive deep \
         walkers (>= 20); found {guarded_walkers}. If walkers were intentionally \
         removed, lower this floor deliberately; a sudden drop means the scan \
         stopped recognizing them."
    );
}

#[test]
fn scan_detects_an_unguarded_recursive_deep_walker() {
    // Negative parity: the detector must FLAG a self-recursive deep walker that
    // lacks `stack_guard!`. Feed it a synthetic walker and assert the analysis
    // classifies it as must-be-guarded with no guard, which is exactly the
    // failure condition the coverage test above keys on.
    let src = r#"
        fn unguarded_walker(expr: &deep::Expr) -> bool {
            match expr {
                _ => unguarded_walker(expr),
            }
        }
    "#;
    let file: File = syn::parse_file(src).expect("synthetic source parses");
    let mut collector = FnCollector {
        walkers: Vec::new(),
    };
    collector.visit_file(&file);
    let walker = collector
        .walkers
        .iter()
        .find(|w| w.name == "unguarded_walker")
        .expect("the synthetic walker was collected");
    assert!(walker.walks_deep_ast, "it takes a &deep::Expr");
    assert!(walker.self_recurses, "it calls itself");
    assert!(!walker.has_stack_guard, "it has no stack_guard!");
    assert!(
        walker.must_be_guarded(),
        "so the coverage test would flag it"
    );
}

#[test]
fn scan_ignores_a_non_recursive_deep_consumer() {
    // A function that takes a `&deep::Expr` but does NOT call itself is not a
    // walker that needs a guard; the scan must not demand one (otherwise the
    // coverage test would be a constant false-positive on every leaf helper).
    let src = r#"
        fn leaf_consumer(expr: &deep::Expr) -> usize {
            expr.children.len()
        }
    "#;
    let file: File = syn::parse_file(src).expect("synthetic source parses");
    let mut collector = FnCollector {
        walkers: Vec::new(),
    };
    collector.visit_file(&file);
    let consumer = collector
        .walkers
        .iter()
        .find(|w| w.name == "leaf_consumer")
        .expect("the synthetic consumer was collected");
    assert!(consumer.walks_deep_ast);
    assert!(!consumer.self_recurses, "it does not call itself");
    assert!(
        !consumer.must_be_guarded(),
        "a non-recursive deep consumer needs no guard"
    );
}

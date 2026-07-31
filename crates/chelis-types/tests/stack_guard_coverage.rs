//! WS-5 Part B (walker-guard coverage): every self-recursive `deep::Expr` /
//! `deep::Pat` walker in the `infer` module tree must carry the `stack_guard!`
//! macro so the recursive type checker bails before exhausting the native
//! stack on a deeply-nested program (WI-1,
//! spec/design/verification_stack_master_plan.md \u{00A7}4.1). This is a
//! SOURCE-SCANNING invariant: it parses every module under `src/infer/` with
//! `syn`, enumerates the functions that take a `&deep::Expr` / `&deep::Pat`
//! argument and call themselves, and asserts each body invokes `stack_guard!`.
//! A NEW unguarded recursive walker therefore fails this test rather than
//! silently reintroducing an unbounded-recursion stack-overflow surface.
//!
//! The scan reads `src/infer/` at run time rather than `include_str!`-ing a
//! fixed list of modules: a hand-maintained list would drop a whole module's
//! walkers from coverage the moment someone added a file and forgot the entry,
//! and that loss would be silent. [`scan_covers_the_whole_inference_tree`]
//! pins that the directory walk actually found the tree.

use std::path::{Path, PathBuf};

use syn::visit::{self, Visit};
use syn::{
    Block, Expr, ExprCall, ExprMacro, ExprMethodCall, File, ImplItemFn, ItemFn, Macro, Signature,
    Type,
};

/// Root of the inference module tree under analysis.
fn infer_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/infer")
}

/// Every `.rs` file in the inference module tree, sorted for a stable scan
/// order.
fn infer_source_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_rs_files(&infer_src_dir(), &mut files);
    files.sort();
    files
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("read dir entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

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

    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if let Expr::Path(callee) = call.func.as_ref() {
            let segments = &callee.path.segments;
            let direct_call = segments.len() == 1;
            let self_associated_call = segments.len() == 2
                && segments
                    .first()
                    .is_some_and(|segment| segment.ident == "Self");
            if (direct_call || self_associated_call)
                && segments
                    .last()
                    .is_some_and(|segment| segment.ident == self.own_name)
            {
                self.calls_own_name = true;
            }
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if call.method == self.own_name
            && matches!(
                call.receiver.as_ref(),
                Expr::Path(receiver)
                    if receiver.path.segments.len() == 1
                        && receiver.path.segments[0].ident == "self"
            )
        {
            self.calls_own_name = true;
        }
        visit::visit_expr_method_call(self, call);
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
    let mut collector = FnCollector {
        walkers: Vec::new(),
    };
    for path in infer_source_files() {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let file: File = syn::parse_file(&source)
            .unwrap_or_else(|e| panic!("{} parses as Rust: {e}", path.display()));
        collector.visit_file(&file);
    }
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

/// The directory walk must actually reach the inference tree. Without this,
/// a wrong path or a renamed directory would make `collect_walkers` return an
/// empty set and every coverage assertion above would pass vacuously.
#[test]
fn scan_covers_the_whole_inference_tree() {
    let files = infer_source_files();
    assert!(
        !files.is_empty(),
        "the scan found no Rust source under {}; the inference tree moved and this \
         guard is no longer looking at it",
        infer_src_dir().display()
    );
    assert!(
        files
            .iter()
            .any(|p| p.file_name().is_some_and(|n| n == "mod.rs")),
        "the inference tree must have a root `mod.rs`; found: {files:?}"
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

#[test]
fn scan_ignores_a_qualified_delegation_with_the_same_terminal_name() {
    let src = r#"
        fn infer_program(exprs: &[deep::Expr]) -> Result {
            crate::session::infer_program(exprs)
        }
    "#;
    let file: File = syn::parse_file(src).expect("synthetic source parses");
    let mut collector = FnCollector {
        walkers: Vec::new(),
    };
    collector.visit_file(&file);
    let wrapper = collector
        .walkers
        .iter()
        .find(|walker| walker.name == "infer_program")
        .expect("the synthetic wrapper was collected");
    assert!(wrapper.walks_deep_ast);
    assert!(
        !wrapper.self_recurses,
        "a qualified delegation is not a call to the wrapper itself"
    );
}

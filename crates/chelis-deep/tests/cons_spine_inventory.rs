//! Structural guard for canonical list readers and helper-mediated recursion.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

const SHARED_READERS: &[&str] = &[
    "eval_cons_spine",
    "is_bracket_literal",
    "collect_cons_chain_for_shape",
    "untyped_chain_items",
    "resugar_finite_list",
    "cons_chain_int_pairs",
    "cons_chain_two_ints",
    "collect_shape_list_elements",
    "cons_chain_pair_list",
    "collect_cons_chain",
    "adt_cons_chain_values",
    "static_list_spine_items",
    "untyped_to_tensor_element",
    "lower_list_literal_items",
    "validate_static_cons_spine",
];

// Generic whole-expression evaluators recurse on every expression kind. They
// recognize Cons as one application, but do not walk a Cons tail specially.
// Pattern walkers likewise recurse over user-authored pattern trees rather
// than the compiler-authored expression/value spine guarded here.
const GENERIC_RECURSION: &[&str] = &[
    "validate_ir_expr",
    "pattern_matches_with_result_producer",
    "plan_host_pattern",
];

// A Cons spelling is not always a canonical spine recognizer. These sites
// build a spine, inspect a fixed two-cell pair, or walk a generic pattern.
// Keep file and function together so another same-named helper is not exempt.
const NON_SPINE_CONS_SITES: &[(&str, &str)] = &[
    (
        "crates/chelis-surf/src/resugar.rs",
        "resugar_declared_tensor_value",
    ),
    ("crates/chelis-ir/src/host.rs", "plan_host_pattern"),
    ("crates/chelis-ir/src/lower.rs", "rebuild_cons_chain"),
    ("crates/chelis-ir/src/lower.rs", "cons_two_int_pair"),
];

// These expression dispatch cycles recurse across arbitrary syntax. Their
// canonical Cons branch routes through a named ConsSpine reader above.
const GENERIC_DISPATCH_CYCLES: &[(&str, &str)] = &[
    (
        "crates/chelis-compiler-api/src/runtime/eval.rs",
        "eval_expr_inner",
    ),
    (
        "crates/chelis-ir/src/host.rs",
        "lower_host_expr_with_expected",
    ),
    ("crates/chelis-ir/src/lower.rs", "lower_expr"),
    (
        "crates/chelis-types/src/infer/validate.rs",
        "validate_ir_expr",
    ),
];

// These existing functions form audited generic-expression dispatch cycles.
// A new helper that joins one of those cycles is not exempt merely because it
// can reach a dispatcher: it must be reviewed and added here explicitly.
const AUDITED_GENERIC_DISPATCH_MEMBERS: &[(&str, &str)] = &[
    (
        "crates/chelis-compiler-api/src/runtime/eval.rs",
        r#"
        apply_def_kernel apply_resolved_callable apply_resolved_callable_under_result_claim
        apply_resolved_callable_with_arg_types apply_resolved_callable_with_arg_types_impl
        apply_staged_host_plan eval_access eval_app eval_app_under_result_claim eval_builtin
        eval_cast eval_checked_local_ascription_region eval_cons_spine eval_decoded eval_expr
        eval_if eval_if_condition eval_let eval_let_under_result_claim eval_let_with_claims
        eval_match eval_match_guard eval_match_under_result_claim eval_record eval_tuple_get
        eval_under_result_claim eval_var initialize_reached_values resolve_top_level
        "#,
    ),
    (
        "crates/chelis-ir/src/host.rs",
        r#"
        def_body_decision ensure_mono_specialization expr_calls_summary_rejecting_top_level_fn
        hoist_host_lane_tensor_bindings lower_access_host_expr lower_app_host_expr
        lower_checked_local_ascription_region lower_def_body_kernel
        lower_host_body_with_record_locals lower_host_callback lower_host_expr
        lower_host_expr_kind lower_host_expr_with_expected_opt lower_host_function
        lower_host_match_arm lower_inline_host_invocation lower_list_literal_items
        lower_match_host_expr lower_mono_specialized_function
        lower_named_retained_host_invocation lower_record_host_expr lower_recursive_generic_call
        lower_retained_host_invocation lower_staged_host_plan prepare_retained_payload_actuals
        top_level_fn_helper_summary_rejects
        "#,
    ),
    (
        "crates/chelis-ir/src/lower.rs",
        r#"
        extract_reshape_dim_list fold_shape_derived_static_size inline_initializer
        inline_program_value inline_trapping_value inline_value_reference
        input_axis_source_from_shape_arg lower_access lower_app lower_atom lower_block
        lower_branch_with_path lower_builtin_app lower_cast lower_copy lower_def
        lower_expr_node lower_expr_unclaimed lower_expr_with_claim lower_fn lower_grad
        lower_grad_callable_app lower_grad_callable_with_values lower_handle_effect
        lower_host_list_filter lower_host_list_fold lower_host_list_map lower_host_list_to_tensor
        lower_host_list_zip_map lower_identity lower_if lower_initializer lower_jit lower_let
        lower_match lower_node lower_one_bound lower_pair_bounds lower_par
        lower_plain_callable_app lower_plain_callable_with_values lower_realize lower_record
        lower_resolved_body lower_sequence_fallthrough lower_split_key lower_stride_bounds
        lower_tensor_concat lower_tuple lower_tuple_get lower_unrepresentable lower_var
        lower_vmap_callable_app lower_vmap_callable_with_nodes lower_vmap_grad_callable_app
        lower_vmap_grad_callable_with_nodes static_i64_from_expr_or_binding
        try_lower_callable_app try_lower_list_producer try_lower_staged_list_recurrence
        try_lower_staged_list_selection try_lower_static_list_concat
        "#,
    ),
    (
        "crates/chelis-types/src/infer/validate.rs",
        "validate_expression_metadata validate_static_cons_spine",
    ),
];

fn is_audited_generic_dispatch_member(source: &str, name: &str) -> bool {
    AUDITED_GENERIC_DISPATCH_MEMBERS
        .iter()
        .any(|(path, members)| {
            *path == source && members.split_whitespace().any(|member| member == name)
        })
}

#[derive(Default)]
struct BodyScan<'a> {
    function: &'a str,
    cons_literal: bool,
    self_call: bool,
    shared_spine: bool,
}

impl<'ast> Visit<'ast> for BodyScan<'_> {
    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        self.cons_literal |= lit.value() == "Cons";
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            self.self_call |= path
                .path
                .get_ident()
                .is_some_and(|callee| callee == self.function);
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.self_call |= call.method == self.function
            && matches!(call.receiver.as_ref(), syn::Expr::Path(path) if path.path.is_ident("self"));
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        self.shared_spine |= path.path.segments.iter().any(|s| s.ident == "ConsSpine");
        syn::visit::visit_expr_path(self, path);
    }
}

fn inspect<'a>(name: &'a str, body: &syn::Block) -> BodyScan<'a> {
    let mut scan = BodyScan {
        function: name,
        ..BodyScan::default()
    };
    scan.visit_block(body);
    scan
}

#[derive(Default)]
struct CallScan {
    cons_recognition: bool,
    calls: BTreeSet<CallRef>,
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
enum CallRef {
    Path(Vec<String>),
    SelfMethod(String),
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
enum FunctionKey {
    Free {
        module: Vec<String>,
        name: String,
    },
    Impl {
        module: Vec<String>,
        owner: String,
        name: String,
    },
}

impl FunctionKey {
    fn module(&self) -> &[String] {
        match self {
            Self::Free { module, .. } | Self::Impl { module, .. } => module,
        }
    }
}

impl<'ast> Visit<'ast> for CallScan {
    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        self.cons_recognition |= lit.value() == "Cons";
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref()
            && path.qself.is_none()
        {
            self.calls.insert(CallRef::Path(
                path.path
                    .segments
                    .iter()
                    .map(|part| part.ident.to_string())
                    .collect(),
            ));
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.cons_recognition |= call.method == "cons_parts";
        if matches!(call.receiver.as_ref(), syn::Expr::Path(path) if path.path.is_ident("self")) {
            self.calls
                .insert(CallRef::SelfMethod(call.method.to_string()));
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

struct FunctionFact {
    key: FunctionKey,
    name: String,
    scan: CallScan,
}

fn reaches_function(start: usize, target: usize, edges: &[Vec<usize>]) -> bool {
    let mut seen = BTreeSet::new();
    let mut work = vec![start];
    while let Some(index) = work.pop() {
        if !seen.insert(index) {
            continue;
        }
        if index == target {
            return true;
        }
        work.extend(edges[index].iter().copied());
    }
    false
}

fn source_module_path(source: &str) -> Vec<String> {
    let Some((_, relative)) = source.split_once("/src/") else {
        return Vec::new();
    };
    let mut parts = relative
        .trim_end_matches(".rs")
        .split('/')
        .map(str::to_string)
        .collect::<Vec<_>>();
    if matches!(
        parts.last().map(String::as_str),
        Some("lib" | "main" | "mod")
    ) {
        parts.pop();
    }
    parts
}

fn resolved_callees(
    fact: &FunctionFact,
    call: &CallRef,
    by_key: &BTreeMap<FunctionKey, Vec<usize>>,
) -> Vec<usize> {
    let module = fact.key.module();
    let mut keys = Vec::new();
    match call {
        CallRef::SelfMethod(name) => {
            if let FunctionKey::Impl { owner, .. } = &fact.key {
                keys.push(FunctionKey::Impl {
                    module: module.to_vec(),
                    owner: owner.clone(),
                    name: name.clone(),
                });
            }
        }
        CallRef::Path(path) => {
            let Some((name, qualifiers)) = path.split_last() else {
                return Vec::new();
            };
            if qualifiers.first().is_some_and(|part| part == "Self") {
                if qualifiers.len() == 1
                    && let FunctionKey::Impl { owner, .. } = &fact.key
                {
                    keys.push(FunctionKey::Impl {
                        module: module.to_vec(),
                        owner: owner.clone(),
                        name: name.clone(),
                    });
                }
            } else {
                let mut target_module = module.to_vec();
                let mut rest = qualifiers;
                if rest.first().is_some_and(|part| part == "crate") {
                    target_module.clear();
                    rest = &rest[1..];
                } else if rest.first().is_some_and(|part| part == "self") {
                    rest = &rest[1..];
                } else {
                    while rest.first().is_some_and(|part| part == "super") {
                        target_module.pop();
                        rest = &rest[1..];
                    }
                }
                target_module.extend(rest.iter().cloned());
                keys.push(FunctionKey::Free {
                    module: target_module,
                    name: name.clone(),
                });
                if let Some((owner, prefix)) = rest.split_last() {
                    let mut owner_module = if qualifiers.first().is_some_and(|part| part == "crate")
                    {
                        Vec::new()
                    } else {
                        let mut resolved = module.to_vec();
                        let skip = qualifiers.len() - rest.len();
                        for qualifier in &qualifiers[..skip] {
                            if qualifier == "super" {
                                resolved.pop();
                            }
                        }
                        resolved
                    };
                    owner_module.extend(prefix.iter().cloned());
                    keys.push(FunctionKey::Impl {
                        module: owner_module,
                        owner: owner.clone(),
                        name: name.clone(),
                    });
                }
            }
        }
    }
    keys.into_iter()
        .flat_map(|key| by_key.get(&key).into_iter().flatten().copied())
        .collect()
}

#[derive(Default)]
struct CallGraph {
    module: Vec<String>,
    impl_type: Option<String>,
    functions: Vec<FunctionFact>,
}

impl CallGraph {
    fn record(&mut self, name: &str, body: &syn::Block) {
        let mut scan = CallScan::default();
        scan.visit_block(body);
        let key = match &self.impl_type {
            Some(owner) => FunctionKey::Impl {
                module: self.module.clone(),
                owner: owner.clone(),
                name: name.to_string(),
            },
            None => FunctionKey::Free {
                module: self.module.clone(),
                name: name.to_string(),
            },
        };
        self.functions.push(FunctionFact {
            key,
            name: name.to_string(),
            scan,
        });
    }

    fn problems(&self, source: &str) -> Vec<String> {
        let mut by_key: BTreeMap<FunctionKey, Vec<usize>> = BTreeMap::new();
        for (index, fact) in self.functions.iter().enumerate() {
            by_key.entry(fact.key.clone()).or_default().push(index);
        }
        let edges = self
            .functions
            .iter()
            .map(|fact| {
                fact.scan
                    .calls
                    .iter()
                    .flat_map(|call| resolved_callees(fact, call, &by_key))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut problems = Vec::new();
        for (root, fact) in self.functions.iter().enumerate() {
            if GENERIC_RECURSION.contains(&fact.name.as_str()) {
                continue;
            }
            let mut seen = BTreeSet::new();
            let mut work = vec![root];
            let mut recursive = false;
            while let Some(index) = work.pop() {
                if !seen.insert(index) {
                    continue;
                }
                for &target in &edges[index] {
                    recursive |= target == root;
                    work.push(target);
                }
            }
            // Generic expression evaluators and lowerers form large call
            // cycles that dispatch Cons to the shared iterator. A new spine
            // reader has a different shape: a recursive cycle asks an
            // independent helper whether the current cell is Cons. Keep the
            // direct literal/self-recursion rule above for that simpler case.
            let audited_generic_dispatch = GENERIC_DISPATCH_CYCLES.iter().any(|(path, anchor)| {
                *path == source
                    && (fact.name == *anchor
                        || is_audited_generic_dispatch_member(source, &fact.name))
                    && seen.iter().copied().any(|candidate| {
                        self.functions[candidate].name == *anchor
                            && reaches_function(candidate, root, &edges)
                    })
            });
            let offending_recognizer = recursive
                .then(|| {
                    seen.iter().copied().find(|&candidate| {
                        let helper = &self.functions[candidate];
                        if !helper.scan.cons_recognition
                            || NON_SPINE_CONS_SITES.contains(&(source, helper.name.as_str()))
                        {
                            return false;
                        }
                        !reaches_function(candidate, root, &edges) || !audited_generic_dispatch
                    })
                })
                .flatten();
            if let Some(helper) = offending_recognizer {
                problems.push(format!(
                    "{source}:{} recursively walks Cons via {}",
                    fact.name, self.functions[helper].name
                ));
            }
        }
        problems
    }
}

impl<'ast> Visit<'ast> for CallGraph {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if let Some((_, items)) = &item.content {
            self.module.push(item.ident.to_string());
            for item in items {
                self.visit_item(item);
            }
            self.module.pop();
        }
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let previous = self.impl_type.take();
        self.impl_type = match item.self_ty.as_ref() {
            syn::Type::Path(path) => path.path.segments.last().map(|part| part.ident.to_string()),
            _ => None,
        };
        syn::visit::visit_item_impl(self, item);
        self.impl_type = previous;
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
    }
}

fn recursive_cons_walker_problems_in_file(parsed: &syn::File, source: &str) -> Vec<String> {
    let mut graph = CallGraph {
        module: source_module_path(source),
        ..CallGraph::default()
    };
    graph.visit_file(parsed);
    graph.problems(source)
}

fn recursive_cons_walker_problems(contents: &str, source: &str) -> Vec<String> {
    let parsed = syn::parse_file(contents).expect("parse source for Cons inventory");
    recursive_cons_walker_problems_in_file(&parsed, source)
}

#[derive(Default)]
struct Inventory {
    seen_shared: BTreeSet<String>,
    seen_audited_generic: BTreeSet<(String, String)>,
    problems: Vec<String>,
    source: String,
}

impl Inventory {
    fn record(&mut self, name: &str, body: &syn::Block) {
        let scan = inspect(name, body);
        if SHARED_READERS.contains(&name) {
            self.seen_shared.insert(name.to_string());
            if !scan.shared_spine {
                self.problems
                    .push(format!("{}:{name} must use ConsSpine", self.source));
            }
        }
        if is_audited_generic_dispatch_member(&self.source, name) {
            self.seen_audited_generic
                .insert((self.source.clone(), name.to_string()));
        }
        if scan.cons_literal && scan.self_call && !GENERIC_RECURSION.contains(&name) {
            self.problems
                .push(format!("{}:{name} recursively walks Cons", self.source));
        }
    }
}

impl<'ast> Visit<'ast> for Inventory {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
        syn::visit::visit_item_fn(self, item);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.record(&item.sig.ident.to_string(), &item.block);
        syn::visit::visit_impl_item_fn(self, item);
    }
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn canonical_cons_readers_share_the_iterator_and_no_direct_literal_reader_appears() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let mut inventory = Inventory::default();
    for crate_dir in fs::read_dir(root.join("crates")).expect("workspace crates") {
        let src = crate_dir.expect("crate entry").path().join("src");
        if !src.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        rust_sources(&src, &mut files);
        for file in files {
            inventory.source = file.strip_prefix(root).unwrap().display().to_string();
            let contents = fs::read_to_string(&file).expect("read Rust source");
            let parsed = syn::parse_file(&contents).expect("parse Rust source");
            inventory.visit_file(&parsed);
            inventory
                .problems
                .extend(recursive_cons_walker_problems_in_file(
                    &parsed,
                    &inventory.source,
                ));
        }
    }
    for reader in SHARED_READERS {
        if !inventory.seen_shared.contains(*reader) {
            inventory
                .problems
                .push(format!("missing shared reader: {reader}"));
        }
    }
    for (source, members) in AUDITED_GENERIC_DISPATCH_MEMBERS {
        for name in members.split_whitespace() {
            if !inventory
                .seen_audited_generic
                .contains(&(source.to_string(), name.to_string()))
            {
                inventory.problems.push(format!(
                    "stale audited generic dispatch member: {source}:{name}"
                ));
            }
        }
    }
    assert!(
        inventory.problems.is_empty(),
        "{}",
        inventory.problems.join("\n")
    );
}

#[test]
fn inventory_detects_a_new_recursive_cons_reader() {
    let parsed = syn::parse_file("fn new_reader(x: i32) { let _ = \"Cons\"; new_reader(x); }")
        .expect("valid Rust probe");
    let syn::Item::Fn(function) = &parsed.items[0] else {
        unreachable!()
    };
    let scan = inspect("new_reader", &function.block);
    assert!(scan.cons_literal && scan.self_call && !scan.shared_spine);
}

#[test]
fn inventory_does_not_call_another_types_same_named_method_self_recursion() {
    let parsed =
        syn::parse_file("fn new_reader(x: i32) { let _ = \"Cons\"; Other::new_reader(x); }")
            .expect("valid Rust probe");
    let syn::Item::Fn(function) = &parsed.items[0] else {
        unreachable!()
    };
    let scan = inspect("new_reader", &function.block);
    assert!(scan.cons_literal && !scan.self_call);
}

#[test]
fn inventory_detects_cons_recognition_and_recursion_split_across_helpers() {
    let source = r#"
        fn recognizes_cons(name: &str) -> bool { name == "Cons" }
        fn walk(node: &str) {
            if recognizes_cons(node) { continue_walk(node); }
        }
        fn continue_walk(node: &str) { walk(node); }
    "#;
    let problems = recursive_cons_walker_problems(source, "probe.rs");
    assert!(
        problems.iter().any(|problem| problem.contains("walk")),
        "mutual recursion through a separate recognizer must fail: {problems:?}"
    );
}

#[test]
fn inventory_detects_qualified_free_helpers_in_their_lexical_module() {
    let source = r#"
        mod reader {
            pub fn recognizes(name: &str) -> bool { name == "Cons" }
            pub fn walk(name: &str) {
                if self::recognizes(name) { continue_walk(name); }
            }
            fn continue_walk(name: &str) { self::walk(name); }
        }
    "#;
    let problems = recursive_cons_walker_problems(source, "probe.rs");
    assert!(
        problems.iter().any(|problem| problem.contains("walk")),
        "a qualified same-module helper cannot hide recursive Cons reading: {problems:?}"
    );
}

#[test]
fn inventory_distinguishes_qualified_free_modules_from_associated_functions() {
    let source = r#"
        mod reader {
            pub fn recognizes(name: &str) -> bool { name == "Cons" }
        }
        struct Other;
        impl Other { fn walk(name: &str) { let _ = name; } }
        fn walk(name: &str) {
            if reader::recognizes(name) { continue_walk(name); }
        }
        fn continue_walk(name: &str) { walk(name); }
    "#;
    let problems = recursive_cons_walker_problems(source, "probe.rs");
    assert!(
        problems.iter().any(|problem| problem.contains("walk")),
        "a qualified module function is part of the free call graph: {problems:?}"
    );
}

#[test]
fn inventory_resolves_super_and_crate_qualified_helpers() {
    let source = r#"
        mod reader {
            pub fn recognizes(name: &str) -> bool { name == "Cons" }
            mod nested {
                pub fn walk(name: &str) {
                    if super::recognizes(name) { continue_walk(name); }
                }
                fn continue_walk(name: &str) { crate::reader::nested::walk(name); }
            }
        }
    "#;
    let problems = recursive_cons_walker_problems(source, "probe.rs");
    assert!(
        problems.iter().any(|problem| problem.contains("walk")),
        "module-qualified helpers must stay in the call graph: {problems:?}"
    );
}

#[test]
fn inventory_detects_cons_recognition_inside_a_mutual_recursion_cycle() {
    let source = r#"
        fn walk(name: &str) {
            if name == "Cons" { continue_walk(name); }
        }
        fn continue_walk(name: &str) { walk(name); }
    "#;
    let problems = recursive_cons_walker_problems(source, "probe.rs");
    assert!(
        problems.iter().any(|problem| problem.contains("walk")),
        "mutual recursion containing Cons recognition must fail: {problems:?}"
    );
}

#[test]
fn inventory_rejects_a_recursive_reader_even_if_it_mentions_the_shared_iterator() {
    let source = r#"
        fn recognizes_cons(name: &str) -> bool { name == "Cons" }
        fn walk(name: &str) {
            let _ = ConsSpine::new(name);
            if recognizes_cons(name) { continue_walk(name); }
        }
        fn continue_walk(name: &str) { walk(name); }
    "#;
    let problems = recursive_cons_walker_problems(source, "probe.rs");
    assert!(
        problems.iter().any(|problem| problem.contains("walk")),
        "mentioning ConsSpine cannot exempt a recursive walker: {problems:?}"
    );
}

#[test]
fn inventory_rejects_a_reader_hidden_inside_an_audited_dispatch_cycle() {
    let source = r#"
        fn validate_ir_expr(node: &str) {
            if node == "Cons" { recursive_reader(node); }
        }
        fn recursive_reader(node: &str) {
            validate_ir_expr(node);
        }
    "#;
    let problems =
        recursive_cons_walker_problems(source, "crates/chelis-types/src/infer/validate.rs");
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("recursive_reader")),
        "an audited dispatcher cannot exempt a new recursive reader: {problems:?}"
    );
}

#[test]
fn inventory_accepts_nonrecursive_cons_helper_and_unrelated_recursion() {
    let source = r#"
        fn recognizes_cons(name: &str) -> bool { name == "Cons" }
        fn iterative_reader(name: &str) {
            let mut current = name;
            while recognizes_cons(current) { current = "Nil"; }
        }
        fn unrelated_recursion(value: usize) {
            if value > 0 { unrelated_recursion(value - 1); }
        }
    "#;
    assert!(
        recursive_cons_walker_problems(source, "probe.rs").is_empty(),
        "finite iteration and unrelated recursion are permitted"
    );
}

#[test]
fn inventory_detects_method_recursion_through_a_cons_helper() {
    let source = r#"
        struct Reader;
        impl Reader {
            fn recognizes(&self, name: &str) -> bool { name == "Cons" }
            fn walk(&self, name: &str) {
                if self.recognizes(name) { self.walk(name); }
            }
        }
    "#;
    let problems = recursive_cons_walker_problems(source, "probe.rs");
    assert!(
        problems.iter().any(|problem| problem.contains("walk")),
        "method helper plus recursive method must fail: {problems:?}"
    );
}

#[test]
fn inventory_does_not_confuse_another_types_method_with_recursion() {
    let source = r#"
        struct Other;
        impl Other { fn walk(name: &str) { let _ = name; } }
        fn walk(name: &str) {
            let _ = "Cons";
            Other::walk(name);
        }
    "#;
    assert!(recursive_cons_walker_problems(source, "probe.rs").is_empty());
}

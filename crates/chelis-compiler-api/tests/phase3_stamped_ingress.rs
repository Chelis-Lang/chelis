//! Compiler-API Deep ingress contract (chelis#1088, part of the chelis#908 /
//! chelis#731 successor-acceptance arc).
//!
//! Every public Deep *text* boundary in this crate consumes the role-stamped
//! carrier. Two properties follow, and both are checked here:
//!
//! 1. **No unstamped tree reaches a consumer.** A top-level form that is
//!    neither a `(module ...)` wrapper nor a declaration is an ingress
//!    rejection, not an `Expr::BareList` / `Expr::Atom` the consumer has to
//!    re-diagnose with hand-written structural guards.
//! 2. **The doors accept one language.** `parse`, `check`, `decompile`, the
//!    read-only authoring queries, and the mutating authoring edits all reject
//!    the same module texts at ingress, and all admit the same ones. The
//!    parity table below is the oracle for that: before chelis#1088 the
//!    generic and authoring doors ran two different ingress strengths and
//!    nothing tested that they agreed.
//!
//! Non-module fields carry their own roles: a replacement *body* is stamped as
//! a RuntimeExpr, a *declaration bundle* as declarations, and `new_params` as
//! exactly `(params ...)`. Each has a negative case here.

use chelis_compiler_api::compiler::{self, CompilerError};
use chelis_compiler_api::schema::{
    AddFunctionRequest, AddPropertyRequest, ChangeSignatureRequest, CheckRequest, DecompileRequest,
    DeepCallGraphRequest, DeepOutlineRequest, DeepReferencesRequest, ParseRequest, RenameRequest,
    ReplaceFunctionBodyRequest, ReplaceFunctionRequest, SourceKind,
};

/// A well-formed module every door accepts at ingress. It carries the
/// `target` function the mutating doors address, so a rejection from one of
/// them is never "no such function".
const VALID_MODULE: &str = r#"(module {}
  phase3.ingress
  (export {} target)
  (defsig {}
    target
    (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {}
    target
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (var {} x))))
"#;

/// A bare name at a RuntimeExpr slot: the canonical stamp rejection.
const BARE_NAME_BODY_MODULE: &str = "(module {} phase3.ingress (def {} target unwrapped_name))";

/// A top-level `(fn ...)`. It is a vocabulary-headed node, so the weaker
/// lenient ingress stamped it happily as a bare/syntax position; only the
/// declaration-role stamp rejects it.
const TOP_LEVEL_NON_DECLARATION: &str =
    "(fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))";

/// A top-level bare atom: not a node at all.
const TOP_LEVEL_ATOM: &str = "42";

/// A top-level untagged list (the chelis#858 input class).
const TOP_LEVEL_UNTAGGED_LIST: &str = "((var {} f) (var {} x))";

/// A top-level head outside the closed vocabulary.
const TOP_LEVEL_UNKNOWN_TAG: &str = "(future-form {} value)";

/// A declaration missing its metadata map at index 1.
const MISSING_META_MAP: &str = "(module {} phase3.ingress (def target (lit {} 1)))";

/// Every module text that must be rejected at ingress by every door.
const REJECTED_MODULES: &[(&str, &str)] = &[
    ("bare name at a RuntimeExpr slot", BARE_NAME_BODY_MODULE),
    ("top-level non-declaration node", TOP_LEVEL_NON_DECLARATION),
    ("top-level bare atom", TOP_LEVEL_ATOM),
    ("top-level untagged list", TOP_LEVEL_UNTAGGED_LIST),
    ("top-level unknown tag", TOP_LEVEL_UNKNOWN_TAG),
    ("declaration missing its metadata map", MISSING_META_MAP),
];

fn deep_parse_kind() -> chelis_vocab::DiagnosticKind {
    chelis_vocab::DiagnosticKind::DeepParseError
}

/// Assert an error is a Deep *ingress* rejection: one diagnostic, the Deep
/// parse kind, and a span (a stamped rejection points at the offending form).
fn assert_deep_ingress_rejection(error: &CompilerError, label: &str) {
    assert_eq!(
        error.errors.len(),
        1,
        "{label}: an ingress rejection is one diagnostic, got {:?}",
        error.errors
    );
    assert_eq!(
        error.errors[0].kind(),
        deep_parse_kind(),
        "{label}: wrong diagnostic kind ({})",
        error.errors[0].message
    );
    assert!(
        error.errors[0].span.is_some(),
        "{label}: an ingress rejection must carry a span"
    );
}

/// Assert an error is *not* a Deep ingress rejection. Used on the accepted
/// control: a door may still reject well-formed Deep for a semantic reason,
/// but never at the ingress boundary.
fn assert_not_a_deep_ingress_rejection(error: &CompilerError, label: &str) {
    assert!(
        error.errors.iter().all(|d| d.kind() != deep_parse_kind()),
        "{label}: well-formed Deep was rejected at ingress: {:?}",
        error.errors
    );
}

// ── The parity table ─────────────────────────────────────────────────

/// One public door, invoked so that only its ingress decision varies with the
/// module text it is handed.
type ModuleTextDoor = (&'static str, fn(&str) -> Result<(), CompilerError>);

/// Every public compiler-API door that takes whole-module Deep text.
fn module_text_doors() -> Vec<ModuleTextDoor> {
    fn parse_door(module: &str) -> Result<(), CompilerError> {
        compiler::parse(ParseRequest {
            source_kind: SourceKind::Deep,
            source: module.to_string(),
        })
        .map(|_| ())
    }
    fn check_door(module: &str) -> Result<(), CompilerError> {
        compiler::check(CheckRequest {
            source_kind: SourceKind::Deep,
            source: module.to_string(),
        })
        .map(|_| ())
    }
    fn decompile_door(module: &str) -> Result<(), CompilerError> {
        compiler::decompile(DecompileRequest {
            source: module.to_string(),
        })
        .map(|_| ())
    }
    fn outline_door(module: &str) -> Result<(), CompilerError> {
        compiler::deep_outline(DeepOutlineRequest {
            module: module.to_string(),
        })
        .map(|_| ())
    }
    fn references_door(module: &str) -> Result<(), CompilerError> {
        compiler::deep_references(DeepReferencesRequest {
            module: module.to_string(),
            symbol: "target".to_string(),
        })
        .map(|_| ())
    }
    fn call_graph_door(module: &str) -> Result<(), CompilerError> {
        compiler::deep_call_graph(DeepCallGraphRequest {
            module: module.to_string(),
        })
        .map(|_| ())
    }
    fn rename_door(module: &str) -> Result<(), CompilerError> {
        compiler::rename(RenameRequest {
            module: module.to_string(),
            function_name: "target".to_string(),
            new_name: "renamed".to_string(),
            preimage_sha256: None,
        })
        .map(|_| ())
    }
    fn replace_function_door(module: &str) -> Result<(), CompilerError> {
        compiler::replace_function(ReplaceFunctionRequest {
            module: module.to_string(),
            function_name: "target".to_string(),
            new_decls: REPLACEMENT_DECLS.to_string(),
            preimage_sha256: None,
        })
        .map(|_| ())
    }
    fn replace_function_body_door(module: &str) -> Result<(), CompilerError> {
        compiler::replace_function_body(ReplaceFunctionBodyRequest {
            module: module.to_string(),
            function_name: "target".to_string(),
            new_body: "(var {} x)".to_string(),
        })
        .map(|_| ())
    }
    fn change_signature_door(module: &str) -> Result<(), CompilerError> {
        compiler::change_signature(ChangeSignatureRequest {
            module: module.to_string(),
            function_name: "target".to_string(),
            new_defsig: REPLACEMENT_DEFSIG.to_string(),
            new_params: REPLACEMENT_PARAMS.to_string(),
            argument_order: vec!["x".to_string()],
            param_renames: Default::default(),
            preimage_sha256: None,
        })
        .map(|_| ())
    }
    fn add_function_door(module: &str) -> Result<(), CompilerError> {
        compiler::add_function(AddFunctionRequest {
            module: module.to_string(),
            new_decls: ADDED_DECLS.to_string(),
            insert_after_function: None,
        })
        .map(|_| ())
    }
    fn add_property_door(module: &str) -> Result<(), CompilerError> {
        compiler::add_property(AddPropertyRequest {
            module: module.to_string(),
            new_decls: ADDED_PROPERTY_DECLS.to_string(),
            insert_after_function: None,
        })
        .map(|_| ())
    }

    vec![
        ("parse", parse_door),
        ("check", check_door),
        ("decompile", decompile_door),
        ("deep_outline", outline_door),
        ("deep_references", references_door),
        ("deep_call_graph", call_graph_door),
        ("rename", rename_door),
        ("replace_function", replace_function_door),
        ("replace_function_body", replace_function_body_door),
        ("change_signature", change_signature_door),
        ("add_function", add_function_door),
        ("add_property", add_property_door),
    ]
}

const REPLACEMENT_DECLS: &str = r#"(defsig {}
  target
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
(def {}
  target
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (var {} x)))
"#;

const REPLACEMENT_DEFSIG: &str =
    "(defsig {} target (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))";

const REPLACEMENT_PARAMS: &str = "(params {} (x {type: (t-prim {} f32)}))";

const ADDED_DECLS: &str = r#"(defsig {}
  added
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
(def {}
  added
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (var {} x)))
"#;

const ADDED_PROPERTY_DECLS: &str = r#"(defsig {}
  added_property
  (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} bool)))
(def {chelis_role: "property",
       property_preconditions: (tuple {}),
       property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
       property_source_kind: "user"}
  added_property
  (fn {}
    (params {} (x {type: (t-prim {} f32)}))
    (app {} (var {} gte) (var {} x) (var {} x))))
"#;

#[test]
fn every_module_text_door_rejects_the_same_ingress_corpus() {
    for (label, module) in REJECTED_MODULES {
        for (door, invoke) in module_text_doors() {
            let Err(error) = invoke(module) else {
                panic!("{door} accepted `{label}`; every door shares one Deep ingress");
            };
            assert_deep_ingress_rejection(&error, &format!("{door} / {label}"));
        }
    }
}

#[test]
fn no_module_text_door_rejects_the_well_formed_control_at_ingress() {
    for (door, invoke) in module_text_doors() {
        if let Err(error) = invoke(VALID_MODULE) {
            assert_not_a_deep_ingress_rejection(&error, door);
        }
    }
}

// ── spec/03-deep-syntax.md §7.1 [03-PROG-1] / [03-PROG-2] ────────────
//
// The top-level form rule is decided by the numbered spec, and these cases
// are derived from its text rather than from what the implementation
// happens to do. [03-PROG-1] enumerates the admissible top-level forms;
// [03-PROG-2] states the rejection contract, which is what lets these
// assert the intended failure rather than merely any parse failure.

/// [03-PROG-1]'s admissible top-level forms: a `module` wrapper, each
/// declaration tag it names, and a mix of both spellings.
const ADMITTED_TOP_LEVEL_FORMS: &[(&str, &str)] = &[
    ("module wrapper", "(module {} m (def {} f (lit {} 1)))"),
    ("def", "(def {} f (lit {} 1))"),
    (
        "defsig",
        "(defsig {} f (t-fn {eff: (effects {})} (t-prim {} int32)))",
    ),
    ("deftype", "(deftype {} Color () (variant {} Red))"),
    ("typealias", "(typealias {} Scalar () (t-prim {} f32))"),
    ("defdim", "(defdim {} batch)"),
    ("import", "(import {} other.module (helper))"),
    ("import-all", "(import-all {} other.module)"),
    ("export", "(export {} f)"),
    (
        "module wrapper beside a bare declaration",
        "(module {} m (def {} inner (lit {} 1)))\n(def {} outer (lit {} 2))",
    ),
];

/// Everything [03-PROG-1] rejects, paired with the head [03-PROG-2]
/// requires the diagnostic to name.
const REJECTED_TOP_LEVEL_FORMS: &[(&str, &str, &str)] = &[
    // Headed forms.
    (
        "expression node",
        "(fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))",
        "`fn`",
    ),
    (
        "application node",
        "(app {} (var {} f) (var {} x))",
        "`app`",
    ),
    ("pattern node", "(pat-var {} x)", "`pat-var`"),
    ("type-expression node", "(t-prim {} f32)", "`t-prim`"),
    ("dimension-expression node", "(d-lit {} 4)", "`d-lit`"),
    (
        "structural child of deftype",
        "(variant {} Red)",
        "`variant`",
    ),
    (
        "helper node",
        "(params {} (x {type: (t-prim {} f32)}))",
        "`params`",
    ),
    (
        "node outside the closed vocabulary",
        "(future-form {} v)",
        "`future-form`",
    ),
    // Headless forms: the closed [03-PROG-2] class set, all nine.
    ("bare identifier", "some_name", "a bare identifier"),
    ("bare integer literal", "42", "a bare integer literal"),
    ("bare float literal", "1.5", "a bare float literal"),
    ("bare string literal", "\"text\"", "a bare string literal"),
    ("bare boolean literal", "true", "a bare boolean literal"),
    ("empty list", "()", "an empty list"),
    (
        "list without a tag symbol",
        "((var {} f) (var {} x))",
        "a list without a tag symbol",
    ),
    ("metadata map", "{key: 1}", "a metadata map"),
    (
        "metadata-annotated form",
        "^{:surf_literal_style \"explicit\"} (var {} x)",
        "a metadata-annotated form",
    ),
];

#[test]
fn parse_admits_exactly_the_top_level_forms_the_spec_enumerates() {
    for (label, source) in ADMITTED_TOP_LEVEL_FORMS {
        compiler::parse(ParseRequest {
            source_kind: SourceKind::Deep,
            source: source.to_string(),
        })
        .unwrap_or_else(|error| {
            panic!("[03-PROG-1] admits a {label} at top level, but parse rejected it: {error:?}")
        });
    }
}

#[test]
fn parse_rejects_every_other_top_level_form_naming_its_head() {
    for (label, source, identification) in REJECTED_TOP_LEVEL_FORMS {
        let error = require_rejection(
            compiler::parse(ParseRequest {
                source_kind: SourceKind::Deep,
                source: source.to_string(),
            }),
            &format!("[03-PROG-1] rejects a {label} at top level"),
        );
        assert_deep_ingress_rejection(&error, &format!("parse / {label}"));
        assert_identifies_the_offending_form(&error, identification, label);
        // [03-PROG-2]: reported at ingress, before anything else observes
        // the program.
        assert_eq!(error.stage, "parse", "{label}");
    }
}

#[test]
fn decompile_shares_the_same_top_level_acceptance_language() {
    // The second public operation the generic ingress backs. Its accepted
    // set is asserted as "no ingress rejection" rather than full success,
    // because resugaring can still reject a form for its own reasons.
    for (label, source) in ADMITTED_TOP_LEVEL_FORMS {
        if let Err(error) = compiler::decompile(DecompileRequest {
            source: source.to_string(),
        }) {
            assert_not_a_deep_ingress_rejection(&error, &format!("decompile / {label}"));
        }
    }
    for (label, source, identification) in REJECTED_TOP_LEVEL_FORMS {
        let error = require_rejection(
            compiler::decompile(DecompileRequest {
                source: source.to_string(),
            }),
            &format!("[03-PROG-1] rejects a {label} at top level"),
        );
        assert_deep_ingress_rejection(&error, &format!("decompile / {label}"));
        assert_identifies_the_offending_form(&error, identification, label);
        assert_eq!(error.stage, "parse", "{label}");
    }
}

/// [03-PROG-2]: the rejection identifies the offending form, by its head
/// symbol when it has one and by its syntactic class when it does not, and
/// never by a placeholder.
fn assert_identifies_the_offending_form(error: &CompilerError, identification: &str, label: &str) {
    let message = &error.errors[0].message;
    assert!(
        message.contains(identification),
        "[03-PROG-2] requires {label} to be identified as {identification}, got: {message}"
    );
    assert!(
        !message.contains('<') && !message.contains('>'),
        "[03-PROG-2] forbids a placeholder identification, got: {message}"
    );
}

/// `expect_err` whose panic names the spec obligation that was violated.
fn require_rejection<T>(outcome: Result<T, CompilerError>, context: &str) -> CompilerError {
    match outcome {
        Ok(_) => panic!("{context}, but it was accepted"),
        Err(error) => error,
    }
}

// ── Span exactness ───────────────────────────────────────────────────

#[test]
fn a_stamp_rejection_points_at_the_offending_form_not_the_whole_input() {
    // The lenient ingress re-wrapped every stamp failure as a parse error at
    // offset 0, so the caller could not locate the offending form. The
    // stamped ingress carries the real span.
    let expected = BARE_NAME_BODY_MODULE
        .find("unwrapped_name")
        .expect("fixture contains the offending name");
    let error = compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("a bare name at a RuntimeExpr slot is an ingress rejection");

    let span = error.errors[0].span.expect("stamp rejections carry a span");
    assert_eq!(
        span.offset, expected,
        "span must address the bare name: {}",
        error.errors[0].message
    );
    assert!(
        error.errors[0].message.contains("bare name"),
        "{}",
        error.errors[0].message
    );
}

// ── Unknown forms survive where they must ────────────────────────────

#[test]
fn generic_parse_wire_preserves_unknown_form_head_and_metadata() {
    // The generic Deep parse door deliberately runs no tag-vocabulary sweep:
    // an unknown head below a declaration stays an `UnknownForm` so the wire
    // AST preserves it and the checker owns the rejection.
    let parsed = compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: r#"(def {} f
  (future-form {sentinel_meta: "keep-me"}
    (lit {} 1)))"#
            .to_string(),
    })
    .expect("unknown forms remain available for downstream diagnostics");

    let wire = serde_json::to_string(&parsed.deep_ast.expect("Deep AST must be present"))
        .expect("wire AST must serialize");
    assert!(
        wire.contains("future-form"),
        "unknown head was lost: {wire}"
    );
    assert!(
        wire.contains("sentinel_meta"),
        "metadata key was lost: {wire}"
    );
    assert!(wire.contains("keep-me"), "metadata value was lost: {wire}");
}

#[test]
fn authoring_doors_still_reject_an_unknown_tag_below_a_declaration() {
    // The editing surfaces keep the vocabulary strictness `parse_str_strict`
    // supplied: an unknown head must not reach a rewriter, even though the
    // generic parse door preserves it.
    let module = r#"(module {}
  phase3.ingress
  (def {} target (future-form {} (lit {} 1))))"#;
    let error = compiler::deep_outline(DeepOutlineRequest {
        module: module.to_string(),
    })
    .expect_err("an unknown tag must not reach an authoring rewriter");
    assert_deep_ingress_rejection(&error, "deep_outline / unknown tag below a declaration");
    assert!(
        error.errors[0].message.contains("unknown tag"),
        "{}",
        error.errors[0].message
    );
}

// ── Non-module fields carry their own roles ──────────────────────────

#[test]
fn a_replacement_body_is_stamped_in_the_runtime_expression_role() {
    let error = compiler::replace_function_body(ReplaceFunctionBodyRequest {
        module: VALID_MODULE.to_string(),
        function_name: "target".to_string(),
        new_body: "unwrapped_name".to_string(),
    })
    .expect_err("a bare name is not a runtime expression");
    assert_deep_ingress_rejection(&error, "replace_function_body / bare name body");
    assert!(
        error.errors[0].message.contains("bare name"),
        "{}",
        error.errors[0].message
    );

    compiler::replace_function_body(ReplaceFunctionBodyRequest {
        module: VALID_MODULE.to_string(),
        function_name: "target".to_string(),
        new_body: "(var {} x)".to_string(),
    })
    .expect("a well-formed runtime expression is still accepted");
}

#[test]
fn a_replacement_declaration_bundle_must_be_declarations() {
    let error = compiler::add_function(AddFunctionRequest {
        module: VALID_MODULE.to_string(),
        new_decls: TOP_LEVEL_NON_DECLARATION.to_string(),
        insert_after_function: None,
    })
    .expect_err("`new_decls` is a declaration bundle, not an expression");
    assert_deep_ingress_rejection(&error, "add_function / non-declaration new_decls");
    assert!(
        error.errors[0].message.contains("expected declaration"),
        "{}",
        error.errors[0].message
    );

    compiler::add_function(AddFunctionRequest {
        module: VALID_MODULE.to_string(),
        new_decls: ADDED_DECLS.to_string(),
        insert_after_function: None,
    })
    .expect("a well-formed declaration bundle is still accepted");
}

#[test]
fn a_replacement_parameter_list_must_carry_the_params_tag() {
    let error = compiler::change_signature(ChangeSignatureRequest {
        module: VALID_MODULE.to_string(),
        function_name: "target".to_string(),
        new_defsig: REPLACEMENT_DEFSIG.to_string(),
        new_params: "(var {} x)".to_string(),
        argument_order: vec!["x".to_string()],
        param_renames: Default::default(),
        preimage_sha256: None,
    })
    .expect_err("`new_params` must be a `(params ...)` node");
    assert_deep_ingress_rejection(&error, "change_signature / non-params new_params");
    assert!(
        error.errors[0].message.contains("expected `params`"),
        "{}",
        error.errors[0].message
    );

    compiler::change_signature(ChangeSignatureRequest {
        module: VALID_MODULE.to_string(),
        function_name: "target".to_string(),
        new_defsig: REPLACEMENT_DEFSIG.to_string(),
        new_params: REPLACEMENT_PARAMS.to_string(),
        argument_order: vec!["x".to_string()],
        param_renames: Default::default(),
        preimage_sha256: None,
    })
    .expect("a well-formed `(params ...)` node is still accepted");
}

#[test]
fn a_replacement_signature_must_be_a_declaration() {
    let error = compiler::change_signature(ChangeSignatureRequest {
        module: VALID_MODULE.to_string(),
        function_name: "target".to_string(),
        new_defsig: "(var {} target)".to_string(),
        new_params: REPLACEMENT_PARAMS.to_string(),
        argument_order: vec!["x".to_string()],
        param_renames: Default::default(),
        preimage_sha256: None,
    })
    .expect_err("`new_defsig` must be a declaration");
    assert_deep_ingress_rejection(&error, "change_signature / non-declaration new_defsig");
    assert!(
        error.errors[0].message.contains("expected declaration"),
        "{}",
        error.errors[0].message
    );
}

// ── The drift guard: structural, not textual ─────────────────────────
//
// `parse_str` and `parse_str_strict` stamp every top-level form as a
// bare/syntax position with no declaration requirement, so a production call
// to either is a second, weaker Deep ingress: exactly the drift chelis#1088
// closed. Production Deep text boundaries route through
// `chelis_deep::parse_and_stamp_file` (a `.dp` program), `parse_and_stamp`
// (a declaration bundle), `parse_and_stamp_runtime_exprs` (an expression),
// or `parse_and_stamp_tagged` (one named tag).
//
// The guard analyses the Rust AST, not source lines. A line-substring guard
// cannot see this, and a reviewer demonstrated it on this very file:
//
//     use chelis_deep::parser::parse_str as parse_deep_text;
//     let exprs = parse_deep_text(source)?;
//
// The import line mentions the crate but not a call; the call line mentions
// neither the crate nor the function. Both lines pass a substring filter
// while the weak ingress is fully reopened. Resolving `use` trees is
// therefore the minimum correct analysis, and it is what the repository
// already does for the `reject_*` census in `phase3_gate_inventory.rs`.

mod weak_ingress {
    use std::collections::{BTreeMap, BTreeSet};
    use syn::visit::{self, Visit};

    /// The `chelis-deep` entry points whose top-level stamp is the lenient
    /// bare/syntax one.
    pub const FUNCTIONS: &[&str] = &["parse_str", "parse_str_strict", "stamp_exprs_lenient"];

    /// The `chelis-deep` modules those functions live in. A path is only an
    /// offence when its qualifier resolves to one of these, so importing
    /// `chelis_deep::parser` for `ParseError` stays legal.
    pub const MODULES: &[&str] = &["parser", "stamp_to_typed"];

    /// The owning crate. Anchoring on it is what keeps `chelis_surf`'s
    /// unrelated `parser::parse_str` (the Surf parser, which has no Deep
    /// top-level stamp to weaken) out of the findings.
    pub const CRATE: &str = "chelis_deep";

    fn is_weak_function(name: &str) -> bool {
        FUNCTIONS.contains(&name)
    }

    fn is_weak_module(name: &str) -> bool {
        MODULES.contains(&name)
    }

    /// A `#[cfg(test)]` item is test code, which may still build lenient
    /// fixtures. The guard governs production paths.
    fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|attr| {
            if !attr.path().is_ident("cfg") {
                return false;
            }
            let mut found = false;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("test") {
                    found = true;
                }
                Ok(())
            });
            found
        })
    }

    /// Local names bound to the owning crate itself, from `use chelis_deep
    /// as deep;`. Collected in their own pass because `use` items are
    /// order-independent: an alias may be declared below its first use.
    #[derive(Default)]
    struct CrateAliasCollector {
        aliases: BTreeSet<String>,
    }

    impl CrateAliasCollector {
        fn walk_tree(&mut self, tree: &syn::UseTree, depth: usize) {
            match tree {
                // `use chelis_deep as deep;` binds the crate only at the
                // root of the tree; `use a::chelis_deep as deep;` names some
                // other item.
                syn::UseTree::Rename(rename) if depth == 0 && rename.ident == CRATE => {
                    self.aliases.insert(rename.rename.to_string());
                }
                syn::UseTree::Group(group) => {
                    for item in &group.items {
                        self.walk_tree(item, depth);
                    }
                }
                syn::UseTree::Path(path) => self.walk_tree(&path.tree, depth + 1),
                _ => {}
            }
        }
    }

    impl<'ast> Visit<'ast> for CrateAliasCollector {
        fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
            self.walk_tree(&item.tree, 0);
        }

        fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
            if is_cfg_test(&item.attrs) {
                return;
            }
            visit::visit_item_mod(self, item);
        }
    }

    /// Pass 2: resolve `use` trees into local bindings.
    #[derive(Default)]
    struct UseCollector {
        in_owning_crate: bool,
        crate_aliases: BTreeSet<String>,
        /// local name -> the weak function it names.
        function_aliases: BTreeMap<String, String>,
        /// local name -> the weak module it names.
        module_aliases: BTreeMap<String, String>,
        findings: Vec<String>,
    }

    impl UseCollector {
        /// True when a path's root names the owning crate: spelled out, via
        /// a `use chelis_deep as ...` alias, or crate-relative inside
        /// `chelis-deep` itself.
        fn roots_at_owning_crate(&self, root: &str) -> bool {
            root == CRATE
                || self.crate_aliases.contains(root)
                || (self.in_owning_crate && matches!(root, "crate" | "self" | "super"))
        }

        /// True when `prefix` is an owning-crate path ending in a module
        /// that holds a weak entry point.
        fn prefix_is_weak_module(&self, prefix: &[String]) -> bool {
            let [root, .., last] = prefix else {
                return false;
            };
            self.roots_at_owning_crate(root) && is_weak_module(last)
        }

        /// True when `prefix` is rooted at the owning crate.
        fn prefix_is_owning_crate(&self, prefix: &[String]) -> bool {
            let [root, ..] = prefix else {
                return false;
            };
            self.roots_at_owning_crate(root)
        }

        fn record(&mut self, prefix: &[String], original: &str, local: &str) {
            // `use a::b::{self as x}` binds `b`, not an item called `self`.
            let (prefix, original) = if original == "self" {
                match prefix.split_last() {
                    Some((last, head)) => (head, last.as_str()),
                    None => return,
                }
            } else {
                (prefix, original)
            };

            if is_weak_function(original) && self.prefix_is_weak_module(prefix) {
                let path = prefix.join("::");
                self.function_aliases
                    .insert(local.to_string(), format!("{path}::{original}"));
                self.findings.push(format!(
                    "imports the weak Deep ingress `{path}::{original}` as `{local}`"
                ));
                return;
            }
            if is_weak_module(original) && self.prefix_is_owning_crate(prefix) {
                self.module_aliases
                    .insert(local.to_string(), original.to_string());
            }
        }

        fn walk_tree(&mut self, tree: &syn::UseTree, prefix: &mut Vec<String>) {
            match tree {
                syn::UseTree::Path(path) => {
                    prefix.push(path.ident.to_string());
                    self.walk_tree(&path.tree, prefix);
                    prefix.pop();
                }
                syn::UseTree::Name(name) => {
                    let ident = name.ident.to_string();
                    // `use a::b::{self}` binds `b` under its own name.
                    let local = if ident == "self" {
                        prefix.last().cloned().unwrap_or_else(|| ident.clone())
                    } else {
                        ident.clone()
                    };
                    self.record(prefix, &ident, &local);
                }
                syn::UseTree::Rename(rename) => {
                    self.record(
                        prefix,
                        &rename.ident.to_string(),
                        &rename.rename.to_string(),
                    );
                }
                syn::UseTree::Glob(_) => {
                    if self.prefix_is_weak_module(prefix) {
                        let path = prefix.join("::");
                        self.findings.push(format!(
                            "glob-imports `{path}::*`, which can introduce a weak Deep ingress \
                             under any local name"
                        ));
                    }
                }
                syn::UseTree::Group(group) => {
                    for item in &group.items {
                        self.walk_tree(item, prefix);
                    }
                }
            }
        }
    }

    impl<'ast> Visit<'ast> for UseCollector {
        fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
            let mut prefix = Vec::new();
            self.walk_tree(&item.tree, &mut prefix);
        }

        fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
            if is_cfg_test(&item.attrs) {
                return;
            }
            visit::visit_item_mod(self, item);
        }
    }

    /// Pass 3: check every path expression against the resolved bindings.
    struct CallChecker<'a> {
        bindings: &'a UseCollector,
        findings: Vec<String>,
    }

    impl<'ast> Visit<'ast> for CallChecker<'_> {
        fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
            if is_cfg_test(&item.attrs) {
                return;
            }
            visit::visit_item_mod(self, item);
        }

        fn visit_path(&mut self, path: &'ast syn::Path) {
            let segments: Vec<String> = path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect();
            if let Some(tail) = segments.last() {
                let rendered = segments.join("::");
                if segments.len() == 1 {
                    if let Some(target) = self.bindings.function_aliases.get(tail) {
                        self.findings
                            .push(format!("calls `{tail}`, which names `{target}`"));
                    }
                } else if is_weak_function(tail) {
                    let qualifier = &segments[segments.len() - 2];
                    let root = &segments[0];
                    // Either the path spells the owning crate (directly, via
                    // a crate alias, or crate-relative inside it), or its
                    // qualifier is a local alias for one of its weak modules.
                    let qualified =
                        is_weak_module(qualifier) && self.bindings.roots_at_owning_crate(root);
                    let aliased = self.bindings.module_aliases.contains_key(qualifier);
                    if qualified || aliased {
                        self.findings
                            .push(format!("names the weak Deep ingress `{rendered}`"));
                    }
                }
            }
            visit::visit_path(self, path);
        }
    }

    /// Report every way `source` reaches a weak Deep ingress from production
    /// code. An empty vector means the file is clean.
    ///
    /// `in_owning_crate` says whether the file belongs to `chelis-deep`,
    /// where `crate::parser::parse_str` names the same function that
    /// `chelis_deep::parser::parse_str` names elsewhere.
    ///
    /// **What this resolves:** fully qualified paths; `use` imports of a
    /// weak function under any local name; module aliases
    /// (`use chelis_deep::parser as p`), including the grouped and
    /// grouped-`self` spellings; crate aliases (`use chelis_deep as deep`);
    /// and glob imports of a weak module. Bindings are collected file-wide
    /// before paths are checked, so a `use` below its first use still
    /// resolves, and an inline module's import is treated as visible to the
    /// whole file. That last one over-approximates, which errs toward a
    /// false positive rather than a miss.
    ///
    /// **What it does not resolve, and cannot from one file:** a re-export
    /// chain, where module A does `pub use chelis_deep::parser::parse_str;`
    /// and module B in a *different* file calls `crate::a::parse_str`. The
    /// `pub use` itself is reported at its own site, so the chain cannot be
    /// introduced without one finding; a caller of an already-existing
    /// re-export in another file is the residual blind spot. Closing it
    /// needs cross-file name resolution, which is the visibility-restriction
    /// work under chelis#1029 rather than a source census.
    pub fn findings_in(source: &str, in_owning_crate: bool) -> Vec<String> {
        let file = match syn::parse_file(source) {
            Ok(file) => file,
            Err(error) => return vec![format!("source does not parse as Rust: {error}")],
        };
        let mut crate_aliases = CrateAliasCollector::default();
        crate_aliases.visit_file(&file);
        let mut bindings = UseCollector {
            in_owning_crate,
            crate_aliases: crate_aliases.aliases,
            ..UseCollector::default()
        };
        bindings.visit_file(&file);
        let mut checker = CallChecker {
            bindings: &bindings,
            findings: Vec::new(),
        };
        checker.visit_file(&file);
        let mut all = bindings.findings.clone();
        all.extend(checker.findings);
        all.sort();
        all.dedup();
        all
    }

    /// The common case: a consumer crate reaching for `chelis_deep`'s weak
    /// entry points.
    pub fn findings(source: &str) -> Vec<String> {
        findings_in(source, false)
    }
}

#[test]
fn no_production_source_reopens_the_weaker_deep_ingress() {
    let sources = workspace_production_sources();
    let mut offenders = Vec::new();
    for source in &sources {
        for finding in weak_ingress::findings_in(&source.text, source.in_owning_crate) {
            offenders.push(format!("{}: {finding}", source.label));
        }
    }
    assert!(
        offenders.is_empty(),
        "these production sites reopen the weaker Deep ingress (chelis#1088):\n{}",
        offenders.join("\n")
    );
}

// ── Mutation-style negative controls for the guard itself ────────────

#[test]
fn the_guard_catches_a_fully_qualified_weak_ingress_call() {
    let source = r#"
        fn ingest(text: &str) -> Vec<chelis_deep::Expr> {
            chelis_deep::parser::parse_str(text).unwrap()
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("chelis_deep::parser::parse_str")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_an_unqualified_imported_weak_ingress_call() {
    let source = r#"
        use chelis_deep::parser::parse_str;

        fn ingest(text: &str) -> Vec<chelis_deep::Expr> {
            parse_str(text).unwrap()
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(!findings.is_empty(), "an imported weak call must be caught");
    assert!(
        findings.iter().any(|finding| finding.contains("parse_str")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_an_aliased_weak_ingress_call() {
    // The reviewer's exact mutation on this branch.
    let source = r#"
        use chelis_deep::parser::parse_str as parse_deep_text;

        fn ingest(text: &str) -> Vec<chelis_deep::Expr> {
            parse_deep_text(text).unwrap()
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("parse_deep_text")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_module_aliased_weak_ingress_call() {
    let source = r#"
        use chelis_deep::parser as deep_parser;

        fn ingest(text: &str) -> Vec<chelis_deep::Expr> {
            deep_parser::parse_str_strict(text).unwrap()
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("deep_parser::parse_str_strict")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_glob_import_of_the_weak_ingress_module() {
    let source = r#"
        use chelis_deep::parser::*;

        fn ingest(text: &str) -> Vec<chelis_deep::Expr> {
            parse_str(text).unwrap()
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings.iter().any(|finding| finding.contains("glob")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_an_owning_crate_alias() {
    // Red-team mutation A, verbatim: `use chelis_deep as deep;` then
    // `deep::parser::parse_str`. The prior guard resolved module aliases but
    // not aliases of the crate itself, so this reopened the ingress green.
    let source = r#"
        #[allow(dead_code)]
        mod redteam_crate_alias_weak_ingress {
            use chelis_deep as deep;

            pub(super) fn ingest(source: &str) {
                let _ = deep::parser::parse_str(source);
            }
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("deep::parser::parse_str")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_grouped_self_module_alias() {
    // Red-team mutation B, verbatim: `use chelis_deep::parser::{self as
    // deep_parser};`. The grouped `self` names the module, which the prior
    // resolver read as an item literally called `self`.
    let source = r#"
        #[allow(dead_code)]
        mod redteam_grouped_module_alias_weak_ingress {
            use chelis_deep::parser::{self as deep_parser};

            pub(super) fn ingest(source: &str) {
                let _ = deep_parser::parse_str(source);
            }
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("deep_parser::parse_str")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_bare_grouped_self_module_import() {
    // One step past the review: `{self}` with no rename binds the module
    // under its own name, so the call spells `parser::parse_str` with no
    // `chelis_deep` on the line at all.
    let source = r#"
        use chelis_deep::parser::{self};

        fn ingest(source: &str) {
            let _ = parser::parse_str(source);
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("parser::parse_str")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_nested_group_module_rename() {
    // `use chelis_deep::{parser as p};` — the rename lives inside a group
    // rather than at the tail of a path.
    let source = r#"
        use chelis_deep::{DeepTag, parser as p};

        fn ingest(source: &str) -> Option<DeepTag> {
            let _ = p::parse_str_strict(source);
            None
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("p::parse_str_strict")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_crate_alias_combined_with_a_function_rename() {
    // Both indirections at once: alias the crate, then import the function
    // through the alias under a third name.
    let source = r#"
        use chelis_deep as deep;
        use deep::parser::parse_str as ingest_deep;

        fn ingest(source: &str) {
            let _ = ingest_deep(source);
        }
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("ingest_deep")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_public_re_export_of_the_weak_ingress() {
    // A `pub use` re-export is reported at its own site, which is what makes
    // the cross-file chain impossible to introduce without one finding.
    let source = r#"
        pub use chelis_deep::parser::parse_str;
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("imports the weak Deep ingress")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_catches_a_use_declared_below_its_first_call() {
    // `use` items are order-independent in Rust, so the resolver collects
    // every binding in the file before checking any path.
    let source = r#"
        fn ingest(source: &str) {
            let _ = deep::parser::parse_str(source);
        }

        use chelis_deep as deep;
    "#;
    let findings = weak_ingress::findings(source);
    assert!(
        findings
            .iter()
            .any(|finding| finding.contains("deep::parser::parse_str")),
        "{findings:?}"
    );
}

#[test]
fn the_guard_does_not_flag_an_unrelated_crate_named_parser_module() {
    // The anchoring control. `chelis_surf::parser::parse_str` is the Surf
    // parser: a different crate, no Deep top-level stamp to weaken. Flagging
    // it would make the guard unusable, since production code calls it
    // everywhere.
    let source = r#"
        use chelis_surf::parser::parse_str as parse_surf;

        fn ingest(source: &str) {
            let _ = parse_surf(source);
            let _ = chelis_surf::parser::parse_str(source);
        }
    "#;
    assert!(
        weak_ingress::findings(source).is_empty(),
        "{:?}",
        weak_ingress::findings(source)
    );
}

#[test]
fn the_guard_does_not_flag_an_alias_of_an_unrelated_crate() {
    // The crate-alias rule must anchor on `chelis_deep` specifically.
    let source = r#"
        use chelis_surf as surf;

        fn ingest(source: &str) {
            let _ = surf::parser::parse_str(source);
        }
    "#;
    assert!(
        weak_ingress::findings(source).is_empty(),
        "{:?}",
        weak_ingress::findings(source)
    );
}

#[test]
fn the_guard_accepts_the_sanctioned_stamped_ingress() {
    // The positive control. Every stamped entry point, a legitimate
    // `parser` import for its error type, and a `#[cfg(test)]` module that
    // does build a lenient fixture, must all pass.
    let source = r#"
        use chelis_deep::parser::ParseError;
        use chelis_deep::{parse_and_stamp, parse_and_stamp_file};

        fn ingest(text: &str) -> Result<Vec<chelis_deep::Expr>, ParseError> {
            let _decls = parse_and_stamp(text);
            let _one = chelis_deep::parse_and_stamp_runtime_exprs(text);
            let _tagged = chelis_deep::parse_and_stamp_tagged(text, chelis_deep::DeepTag::Params);
            Ok(parse_and_stamp_file(text).unwrap())
        }

        #[cfg(test)]
        mod tests {
            #[test]
            fn lenient_fixture() {
                let _ = chelis_deep::parser::parse_str("(lit {} 1)");
            }
        }
    "#;
    assert!(
        weak_ingress::findings(source).is_empty(),
        "{:?}",
        weak_ingress::findings(source)
    );
}

struct ProductionSource {
    label: String,
    text: String,
    in_owning_crate: bool,
}

/// The two `chelis-deep` modules that define the weak entry points. Their
/// own internal wiring (`parse_str` calls `parse`, which calls
/// `stamp_exprs_lenient`) IS the weak entry point, not a consumer choosing
/// the wrong door.
const OWNING_CRATE_DEFINING_MODULES: &[&str] = &["parser.rs", "stamp_to_typed.rs"];

/// Every workspace crate's `src/` tree, minus the file-backed modules a
/// `#[cfg(test)]` declaration pulls in.
///
/// Test modules are resolved structurally, from the `mod name;` declarations
/// themselves, not by a filename convention: a `#[cfg(test)] mod tests;` and
/// everything that module declares transitively is test code, whatever it is
/// named. `#[cfg(test)]` modules written inline are skipped by the analysis
/// itself.
fn workspace_production_sources() -> Vec<ProductionSource> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    fn walk(directory: &Path, out: &mut Vec<PathBuf>) {
        let mut entries: Vec<_> = std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
            .map(|entry| entry.expect("directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    /// The directory a file's `mod name;` declarations resolve against.
    fn module_directory(file: &Path) -> PathBuf {
        let parent = file
            .parent()
            .expect("a source file has a parent")
            .to_path_buf();
        match file.file_stem().and_then(|stem| stem.to_str()) {
            Some("lib") | Some("main") | Some("mod") => parent,
            Some(stem) => parent.join(stem),
            None => parent,
        }
    }

    /// Resolve `mod name;` to `<dir>/name.rs` or `<dir>/name/mod.rs`.
    fn resolve_module(directory: &Path, name: &str) -> Option<PathBuf> {
        let flat = directory.join(format!("{name}.rs"));
        if flat.is_file() {
            return Some(flat);
        }
        let nested = directory.join(name).join("mod.rs");
        nested.is_file().then_some(nested)
    }

    /// Top-level `mod name;` declarations and whether each is `#[cfg(test)]`.
    fn declared_file_modules(text: &str) -> Vec<(String, bool)> {
        let Ok(file) = syn::parse_file(text) else {
            return Vec::new();
        };
        file.items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Mod(module) if module.content.is_none() => {
                    let cfg_test = module.attrs.iter().any(|attr| {
                        if !attr.path().is_ident("cfg") {
                            return false;
                        }
                        let mut found = false;
                        let _ = attr.parse_nested_meta(|meta| {
                            if meta.path.is_ident("test") {
                                found = true;
                            }
                            Ok(())
                        });
                        found
                    });
                    Some((module.ident.to_string(), cfg_test))
                }
                _ => None,
            })
            .collect()
    }

    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("compiler-api lives under <workspace>/crates");
    let mut members: Vec<_> = std::fs::read_dir(crates)
        .expect("read the crates directory")
        .map(|entry| entry.expect("directory entry").path())
        .collect();
    members.sort();

    let mut files: Vec<PathBuf> = Vec::new();
    let mut owning_crate_files: BTreeSet<PathBuf> = BTreeSet::new();
    for member in &members {
        let source_root = member.join("src");
        if !source_root.is_dir() {
            continue;
        }
        let before = files.len();
        walk(&source_root, &mut files);
        if member.file_name().is_some_and(|name| name == "chelis-deep") {
            owning_crate_files.extend(files[before..].iter().cloned());
        }
    }

    let texts: BTreeMap<PathBuf, String> = files
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            (path.clone(), text)
        })
        .collect();

    // Seed with every `#[cfg(test)] mod name;`, then close transitively:
    // anything a test module declares is also test code.
    let mut test_only: BTreeSet<PathBuf> = BTreeSet::new();
    let mut frontier: Vec<PathBuf> = Vec::new();
    for (path, text) in &texts {
        let directory = module_directory(path);
        for (name, cfg_test) in declared_file_modules(text) {
            if cfg_test
                && let Some(child) = resolve_module(&directory, &name)
                && test_only.insert(child.clone())
            {
                frontier.push(child);
            }
        }
    }
    while let Some(path) = frontier.pop() {
        let Some(text) = texts.get(&path) else {
            continue;
        };
        let directory = module_directory(&path);
        for (name, _) in declared_file_modules(text) {
            if let Some(child) = resolve_module(&directory, &name)
                && test_only.insert(child.clone())
            {
                frontier.push(child);
            }
        }
    }

    let mut out = Vec::new();
    for path in files {
        if test_only.contains(&path) {
            continue;
        }
        let in_owning_crate = owning_crate_files.contains(&path);
        if in_owning_crate
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| OWNING_CRATE_DEFINING_MODULES.contains(&name))
        {
            continue;
        }
        let text = texts.get(&path).expect("every file was read").clone();
        out.push(ProductionSource {
            label: path.display().to_string(),
            text,
            in_owning_crate,
        });
    }
    assert!(
        out.len() > 100,
        "the workspace production census must be non-trivial, got {}",
        out.len()
    );
    out
}

#[test]
fn the_production_census_excludes_declared_test_modules_and_keeps_real_sources() {
    // A control on the census itself: a `#[cfg(test)] mod tests;` file is
    // excluded, a transitively declared child of one is excluded, and an
    // ordinary production module is kept.
    let sources = workspace_production_sources();
    let labels: Vec<&str> = sources.iter().map(|source| source.label.as_str()).collect();
    let has = |needle: &str| labels.iter().any(|label| label.ends_with(needle));

    assert!(
        has("chelis-compiler-api/src/compiler.rs"),
        "a production module must be censused"
    );
    assert!(
        !has("chelis-pred/src/tests.rs"),
        "a `#[cfg(test)] mod tests;` file must be excluded"
    );
    assert!(
        !has("chelis-types/src/infer/tests/more.rs"),
        "a module declared by a test module must be excluded transitively"
    );
    assert!(
        sources.iter().any(|source| source.in_owning_crate),
        "the owning crate must be censused with its crate-relative roots resolved"
    );
}

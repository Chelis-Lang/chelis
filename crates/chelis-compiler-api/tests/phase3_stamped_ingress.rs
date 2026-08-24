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

// ── The drift tripwire ───────────────────────────────────────────────

/// The weaker ingress spellings, by the exact call text a reviewer would see.
const WEAK_DEEP_INGRESS_CALLS: &[&str] = &[
    "parse_str_strict(",
    "parser::parse_str(",
    "stamp_exprs_lenient(",
];

#[test]
fn no_compiler_api_source_reopens_the_weaker_deep_ingress() {
    // chelis#1088's regression guard. `parse_str` and `parse_str_strict`
    // stamp every top-level form as a bare/syntax position, so a call to
    // either from this crate is a second, weaker Deep ingress: exactly the
    // drift this issue closed. New Deep text boundaries route through
    // `chelis_deep::parse_and_stamp_file` (a `.dp` program),
    // `parse_and_stamp` (a declaration bundle),
    // `parse_and_stamp_runtime_exprs` (an expression), or
    // `parse_and_stamp_tagged` (one named tag).
    let mut offenders = Vec::new();
    for (path, source) in crate_sources() {
        for (index, line) in source.lines().enumerate() {
            if !line.contains("chelis_deep") {
                continue;
            }
            for spelling in WEAK_DEEP_INGRESS_CALLS {
                if line.contains(spelling) {
                    offenders.push(format!("{path}:{}: {}", index + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these compiler-API sites reopen the weaker Deep ingress:\n{}",
        offenders.join("\n")
    );
}

fn crate_sources() -> Vec<(String, String)> {
    fn walk(directory: &std::path::Path, out: &mut Vec<(String, String)>) {
        let mut entries: Vec<_> = std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
            .map(|entry| entry.expect("directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                out.push((path.display().to_string(), text));
            }
        }
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    walk(&root, &mut out);
    assert!(!out.is_empty(), "the crate source tree must be readable");
    out
}

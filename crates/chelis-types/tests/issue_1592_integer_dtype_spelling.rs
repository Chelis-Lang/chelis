//! chelis#1592 checker ingress parity for canonical and retired signed integer
//! primitive spellings.

use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;
use chelis_types::types::{Prim, Type};

fn error_messages(program: &[chelis_deep::Expr]) -> Vec<String> {
    match check_ir_program(program) {
        Ok(_) => Vec::new(),
        Err(report) => report
            .errors
            .iter()
            .map(|error| error.message.clone())
            .collect(),
    }
}

#[test]
fn prim_identity_uses_the_literal_suffix_spelling_only() {
    for (name, prim) in [
        ("i8", Prim::Int8),
        ("i16", Prim::Int16),
        ("i32", Prim::Int32),
        ("i64", Prim::Int64),
    ] {
        assert_eq!(Prim::parse_name(name), Some(prim));
        assert_eq!(prim.name(), name);
        assert_eq!(Type::Prim(prim).to_string(), name);
    }
    for retired in ["int8", "int16", "int32", "int64"] {
        assert_eq!(
            Prim::parse_name(retired),
            None,
            "retired spellings are migration input, not aliases"
        );
        assert!(
            Prim::parse_interchange_name(retired).is_some(),
            "interchange keeps the ecosystem spelling"
        );
    }
    for language_name in ["i8", "i16", "i32", "i64"] {
        assert_eq!(
            Prim::parse_interchange_name(language_name),
            None,
            "versioned interchange must not admit the language spelling"
        );
    }
}

#[test]
fn retired_surf_names_reject_at_parser_ingress_with_the_versioned_migration_command() {
    for retired in ["int8", "int16", "int32", "int64"] {
        let source = format!("module P.M\nexport (f)\ndef f(x: {retired}) -> {retired} = x\n");
        let message = parse_str(&source)
            .expect_err("canonical Surf parser rejects retired dtype")
            .to_string();
        assert!(
            message.contains(retired) && message.contains("chelis migrate surf --from 0.18"),
            "`{retired}` must reject with actionable migration advice: {message}"
        );
    }
}

#[test]
fn retired_deep_primitive_and_type_variable_names_both_reject() {
    for (tag, command) in [
        ("t-prim", "chelis migrate deep --from 0.18"),
        ("t-var", "chelis migrate deep --from 0.18"),
    ] {
        let source = format!(
            "(module {{surf_path: \"P.M\"}}\n  p.m\n  (export {{}} f)\n  \
             (defsig {{}} f (t-fn {{}} ({tag} {{}} int64) ({tag} {{}} int64)))\n  \
             (def {{}} f (fn {{}} (params {{}} (x {{type: ({tag} {{}} int64)}})) \
             (var {{}} x))))\n"
        );
        let error =
            chelis_deep::parser::parse_and_stamp_file(&source).expect_err("Deep ingress rejects");
        let message = error.to_string();
        assert!(
            message.contains("int64") && message.contains(command),
            "retired `{tag}` spelling must reject with Deep migration advice: {message}"
        );
    }
}

#[test]
fn canonical_deep_integer_primitive_checks_clean() {
    let source = "(module {surf_path: \"P.M\"}\n  p.m\n  (export {} f)\n  \
                  (defsig {} f (t-fn {} (t-prim {} i64) (t-prim {} i64)))\n  \
                  (def {} f (fn {} (params {} (x {type: (t-prim {} i64)})) \
                  (var {} x))))\n";
    let program = chelis_deep::parser::parse_and_stamp_file(source).expect("Deep parses");
    assert!(
        error_messages(&program).is_empty(),
        "canonical Deep must check clean"
    );
}

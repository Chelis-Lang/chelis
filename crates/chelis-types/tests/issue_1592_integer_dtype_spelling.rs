//! chelis#1592 checker ingress parity for canonical and retired signed integer
//! primitive spellings.

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
fn language_and_interchange_integer_names_are_distinct() {
    for (language_name, interchange_name, prim) in [
        ("i8", "int8", Prim::Int8),
        ("i16", "int16", Prim::Int16),
        ("i32", "int32", Prim::Int32),
        ("i64", "int64", Prim::Int64),
    ] {
        assert_eq!(Prim::parse_name(language_name), Some(prim));
        assert_eq!(prim.name(), language_name);
        assert_eq!(Type::Prim(prim).to_string(), language_name);
        assert_eq!(prim.interchange_name(), interchange_name);
        assert_eq!(Prim::parse_interchange_name(interchange_name), Some(prim));
        assert_eq!(
            Prim::parse_interchange_name(language_name),
            None,
            "versioned interchange must not admit the language spelling"
        );
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
}

#[test]
fn retired_deep_primitive_and_type_variable_names_both_reject() {
    for tag in ["t-prim", "t-var"] {
        let source = format!(
            "(module {{surf_path: \"P.M\"}}\n  p.m\n  (export {{}} f)\n  \
             (defsig {{}} f (t-fn {{}} ({tag} {{}} int64) ({tag} {{}} int64)))\n  \
             (def {{}} f (fn {{}} (params {{}} (x {{type: ({tag} {{}} int64)}})) \
             (var {{}} x))))\n"
        );
        chelis_deep::parser::parse_and_stamp_file(&source).expect_err("Deep ingress rejects");
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

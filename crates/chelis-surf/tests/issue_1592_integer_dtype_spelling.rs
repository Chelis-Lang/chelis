//! chelis#1592: signed integer primitives have one canonical language
//! spelling in Surf, Deep, and literal suffixes.

use chelis_surf::desugar::desugar_program;
use chelis_surf::format::{format_source, migrate_source_v018};
use chelis_surf::parser::parse_str;

fn deep_of(source: &str) -> String {
    let declarations = parse_str(source).expect("Surf parses");
    chelis_deep::printer::print_canonical(&desugar_program(&declarations))
}

#[test]
fn canonical_integer_names_survive_surf_to_deep_in_every_type_position() {
    for name in ["i8", "i16", "i32", "i64"] {
        let source = format!(
            "module P.M\nexport (f)\n\
             sig f: tensor[n, {name}] -> tensor[n, {name}]\n\
             def f(x: tensor[n, {name}]) -> tensor[n, {name}] = cast(x, {name})\n"
        );
        let deep = deep_of(&source);
        assert!(
            deep.contains(&format!("(t-prim {{}} {name})")),
            "`{name}` must be the canonical Deep primitive: {deep}"
        );
        assert!(
            !deep.contains(&format!("int{}", &name[1..])),
            "the retired long spelling must not reach canonical Deep: {deep}"
        );
    }
}

#[test]
fn canonical_formatter_preserves_the_single_integer_spelling() {
    let source = "module P.M\nexport (f)\ndef f(x: i32) -> i64 = cast(x, i64)\n";
    let once = format_source(source).expect("format");
    let twice = format_source(&once).expect("format fixed point");
    assert_eq!(once, source);
    assert_eq!(twice, once);
}

#[test]
fn canonical_parser_rejects_retired_integer_names_in_every_type_edge() {
    for source in [
        "value: int64 = 1i64\n",
        "value: tensor[n, int32] = [1]\n",
        "value = cast(1, int16)\n",
        "value = 1 |> cast(int8)\n",
    ] {
        let error = parse_str(source).expect_err("retired dtype spelling must not parse");
        let rendered = error.to_string();
        assert!(
            rendered.contains("chelis migrate surf --from 0.18"),
            "{source:?}: {rendered}"
        );
    }
}

#[test]
fn v018_migration_rewrites_retired_integer_names_without_touching_identifiers() {
    let source = "module P.M\nexport (int64_value, f)\n\
                  int64_value: int64 = 1i64\n\
                  def f(x: tensor[n, int32]) -> int64 = cast(x.0, int64)\n";
    let migrated = migrate_source_v018(source).expect("legacy migration");
    assert!(migrated.contains("int64_value: i64"));
    assert!(migrated.contains("tensor[n, i32]"));
    assert!(migrated.contains("cast(x.0, i64)"));
    assert!(!migrated.contains(": int64"));
}

#[test]
fn typed_integer_literals_emit_the_same_name_as_their_suffix() {
    for name in ["i8", "i16", "i32", "i64"] {
        let deep = deep_of(&format!("value = 1{name}\n"));
        assert!(
            deep.contains(&format!("(t-prim {{}} {name})")),
            "literal suffix and primitive name must agree: {deep}"
        );
    }
}

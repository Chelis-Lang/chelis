//! Spec/03 §2.5.1 and dtype_semantics C6: resolved, recursive stdlib discovery.

use super::*;

fn sources(files: &[(&str, &str)]) -> Vec<Row> {
    let root = tempfile::tempdir().expect("fixture root");
    for (label, source) in files {
        let file = root.path().join(STD_SRC_REL).join(format!("{label}.ch"));
        fs::create_dir_all(file.parent().unwrap()).expect("fixture directory");
        fs::write(file, source).expect("fixture source");
    }
    stdlib_rows(root.path())
}

fn numeric(rows: &[Row], name: &str) -> bool {
    rows.iter()
        .any(|row| row.kind == "std-def-numeric" && row.id.starts_with(&format!("{name}: ")))
}

fn rejects(files: &[(&str, &str)], reason: &str) {
    let error = std::panic::catch_unwind(|| sources(files)).expect_err("invalid closure must fail");
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic");
    assert!(
        message.contains(reason),
        "expected {reason:?}, got {message}"
    );
}

#[test]
fn identical_short_names_remain_scoped_to_their_declaring_modules() {
    let rows = sources(&[
        (
            "a",
            "module Std.A\nexport (read)\ntype Value = | Value(f64)\ndef read(x: Value) -> Value = x",
        ),
        (
            "b",
            "module Std.B\nexport (read)\ntype Value = | Value(bool)\ndef read(x: Value) -> Value = x",
        ),
    ]);
    assert!(numeric(&rows, "a::read"));
    assert!(!numeric(&rows, "b::read"));
}

#[test]
fn imports_follow_qualified_selective_and_wildcard_type_names() {
    for (import, ty) in [
        ("import Std.Data", "Std.Data.Value"),
        ("import Std.Data (Value)", "Value"),
        ("import Std.Data (..)", "Value"),
    ] {
        let source = format!(
            "module Std.Use\n{import}\nexport (read)\ndef read(x: Option[List[{ty}]]) -> Option[List[{ty}]] = x"
        );
        let rows = sources(&[
            (
                "data",
                "module Std.Data\nexport (Value)\ntype Value = | Value(f64)",
            ),
            ("use", &source),
        ]);
        assert!(numeric(&rows, "use::read"), "{import}: {rows:?}");
        assert!(
            rows.iter()
                .find(|row| row.id.starts_with("use::read:"))
                .unwrap()
                .flags
                .is_empty()
        );
    }
}

#[test]
fn private_helpers_and_transparent_aliases_survive_relocation() {
    let consumer = "module Std.Use\nimport Std.Helper (Payload)\nexport (read)\ntype Public = | Public(Option[Payload])\ndef read(x: Public) -> Public = x";
    for helper_path in ["helper", "moved/helper"] {
        for (dtype, expected_numeric) in [("f64", true), ("bool", false)] {
            // The public alias reaches a private local declaration; it does
            // not make Hidden directly importable (spec/02 P2, P15; #1919).
            let helper = format!(
                "module Std.Helper\nexport (marker, Payload)\ntype Hidden = ({dtype}, bool)\ntype Payload = Hidden\ndef marker(x: bool) -> bool = x"
            );
            let rows = sources(&[(helper_path, &helper), ("use", consumer)]);
            assert_eq!(
                numeric(&rows, "use::read"),
                expected_numeric,
                "{helper_path}, {dtype}: {rows:?}"
            );
            assert!(!numeric(&rows, &format!("{helper_path}::marker")));
            rejects(
                &[
                    (helper_path, &helper),
                    ("use", &consumer.replace("Payload", "Hidden")),
                ],
                "module `Std.Helper` does not export `Hidden` for import into Std.Use",
            );
        }
    }
}

#[test]
fn import_failures_and_ambiguity_do_not_erase_numeric_capacity() {
    rejects(
        &[
            ("a", "module Std.A\ntype Other = | Value(f64)"),
            (
                "use",
                "module Std.Use\nimport Std.A\nexport (read)\ndef read(x: Std.A.Value) -> bool = true",
            ),
        ],
        "Value",
    );
    rejects(
        &[(
            "use",
            "module Std.Use\nimport Std.Absent\nexport (read)\ndef read(x: bool) -> bool = x",
        )],
        "unresolved import",
    );
    rejects(
        &[
            ("a", "module Std.A\ntype Value = | Value(f64)"),
            ("b", "module Std.B\ntype Value = | Value(bool)"),
            (
                "use",
                "module Std.Use\nimport Std.A (Value)\nimport Std.B (Value)\nexport (read)\ndef read(x: Value) -> Value = x",
            ),
        ],
        "ambiguous reference",
    );
    rejects(
        &[
            ("a", "module Std.A\ntype Value = | Value(f64)"),
            (
                "use",
                "module Std.Use\nimport Std.A\nexport (read)\ndef read(x: Std.A.Missing) -> bool = true",
            ),
        ],
        "Missing",
    );
}

#[test]
fn local_nominal_names_shadow_imported_names() {
    let rows = sources(&[
        ("a", "module Std.A\ntype Value = | Value(f64)"),
        (
            "use",
            "module Std.Use\nimport Std.A (Value)\nexport (read)\ntype Value = | Local(bool)\ndef read(x: Value) -> Value = x",
        ),
    ]);
    assert!(!numeric(&rows, "use::read"));
}

#[test]
fn aliases_substitute_generic_arguments_and_preserve_nonnumeric_companions() {
    let rows = sources(&[(
        "generic",
        "module Std.Generic\nexport (numbers, booleans)\ntype Box[a] = | Box(a)\ntype Wrapped[a] = Option[List[Box[a]]]\ndef numbers(x: Wrapped[f64]) -> Wrapped[f64] = x\ndef booleans(x: Wrapped[bool]) -> Wrapped[bool] = x",
    )]);
    assert!(numeric(&rows, "generic::numbers"));
    assert!(!numeric(&rows, "generic::booleans"));
}

#[test]
fn forward_and_recursive_aliases_reach_a_finite_numeric_fixed_point() {
    let rows = sources(&[(
        "recursive",
        "module Std.Recursive\nexport (forward, growing, booleans)\ntype First = Second\ntype Second = (f64, bool)\ntype Tree[a] = | Leaf(a) | Branch(List[Tree[List[a]]])\ntype Recursive[a] = (a, Recursive[a])\ndef forward(x: First) -> First = x\ndef growing(x: Tree[i64]) -> Tree[i64] = x\ndef booleans(x: Recursive[bool]) -> Recursive[bool] = x",
    )]);
    assert!(numeric(&rows, "recursive::forward"));
    assert!(numeric(&rows, "recursive::growing"));
    assert!(!numeric(&rows, "recursive::booleans"));
}

#[test]
fn mutual_nominal_recursion_propagates_numeric_fields() {
    let rows = sources(&[(
        "mutual",
        "module Std.Mutual\nexport (read)\ntype Left = | Left(Option[Right])\ntype Right = | Right(Left, f64)\ndef read(x: Left) -> Left = x",
    )]);
    assert!(numeric(&rows, "mutual::read"));
}

#[test]
fn unresolved_nominals_and_wrong_nominal_arguments_fail_closed() {
    for (signature, reason) in [
        ("Missing", "unknown nominal"),
        ("Box", "expects 1 argument"),
        ("Box[f64, bool]", "expects 1 argument"),
    ] {
        let source = format!(
            "module Std.Bad\nexport (read)\ntype Box[a] = | Box(a)\ndef read(x: {signature}) -> bool = true"
        );
        rejects(&[("bad", &source)], reason);
    }
    rejects(
        &[(
            "bad",
            "module Std.Bad\nexport (read)\ntype Matrix[p, n] = tensor[n, p]\ndef read(x: Matrix[f64, bool]) -> bool = true",
        )],
        "dimension",
    );
}

#[test]
fn nested_alias_fields_and_precision_variables_are_capacity() {
    let rows = sources(&[(
        "nested",
        "module Std.Nested\nexport (read, generic, ordinary, bounded)\ntype Hidden = (&tensor[3, f64], Dict[string, Option[(List[i64], bool)]])\ndef read(x: Hidden) -> Hidden = x\nsig generic[n, p]: tensor[n, p] -> p\ndef generic(x) = x\nsig ordinary[p]: p -> p\ndef ordinary(x) = x\nsig bounded[p: Float]: p -> p\ndef bounded(x) = x",
    )]);
    for name in ["read", "generic", "bounded"] {
        assert!(
            numeric(&rows, &format!("nested::{name}")),
            "{name}: {rows:?}"
        );
    }
    assert!(!numeric(&rows, "nested::ordinary"));
}

#[test]
fn stdlib_closure_preserves_all_final_registered_source_identities() {
    let expected: BTreeSet<_> = FINAL_NUMERIC_OPERATION_ROWS
        .iter()
        .filter(|registration| {
            matches!(
                registration.surface.kind,
                "std-def-numeric" | "std-adt-numeric"
            )
        })
        .map(|registration| {
            (
                registration.surface.kind.to_string(),
                registration.surface.id.to_string(),
            )
        })
        .collect();
    assert_eq!(
        expected
            .iter()
            .filter(|(kind, _)| kind == "std-def-numeric")
            .count(),
        228
    );
    assert_eq!(
        expected
            .iter()
            .filter(|(kind, _)| kind == "std-adt-numeric")
            .count(),
        14
    );
    let actual = stdlib_rows(&repo_root())
        .into_iter()
        .map(|row| (row.kind, row.id))
        .collect();
    assert_eq!(expected, actual);
}

#[test]
fn symbolic_tensor_precision_requires_an_adt_operation_row() {
    for (field, expected_flags) in [
        ("tensor[3, p]", Some(vec!["numeric-op"])),
        ("tensor[3, bool]", None),
        ("tensor[3, f64]", Some(vec!["float-carrier"])),
        ("p", None),
    ] {
        let source =
            format!("module Std.Constructor\nexport (Vector)\ntype Vector[p] = | Vector({field})");
        let rows = sources(&[("constructor", &source)]);
        match expected_flags {
            Some(flags) => {
                assert_eq!(rows.len(), 1, "{field}: {rows:?}");
                assert_eq!(rows[0].kind, "std-adt-numeric");
                assert!(
                    rows[0]
                        .id
                        .starts_with("constructor::Vector: (p) (variant {} Vector ")
                );
                assert_eq!(rows[0].flags, flags);
                let error = capacity_census_authority::classify_final_authority(
                    &authority_surface(&rows[0]),
                    final_authority_registries(),
                    &fs::read_to_string(repo_root().join(CONTROLLING_SPEC_REL)).unwrap(),
                )
                .expect_err("a new numeric constructor still needs exact semantic registration");
                assert!(
                    error.contains("descriptor has zero final matches"),
                    "{error}"
                );
            }
            None => assert!(rows.is_empty(), "{field}: {rows:?}"),
        }
    }
}

#[test]
fn imported_symbolic_precision_reaches_adt_rows_without_tainting_boolean_instances() {
    for field in ["Imported[p]", "Option[Imported[p]]", "Nested[p]"] {
        let source = format!(
            "module Std.Use\nimport Std.Data (Imported, Nested)\nexport (Vector, numbers, booleans)\ntype Vector[p] = | Vector({field})\nsig numbers: Vector[i32] -> Vector[i32]\ndef numbers(x) = x\nsig booleans: Vector[bool] -> Vector[bool]\ndef booleans(x) = x"
        );
        let rows = sources(&[
            (
                "data",
                "module Std.Data\ntype Precision[a] = tensor[3, a]\ntype Imported[a] = Precision[a]\ntype Nested[a] = | Nested(Option[Imported[a]])",
            ),
            ("use", &source),
        ]);
        for name in ["data::Nested", "use::Vector"] {
            let row = rows
                .iter()
                .find(|row| {
                    row.kind == "std-adt-numeric" && row.id.starts_with(&format!("{name}: "))
                })
                .unwrap_or_else(|| panic!("missing symbolic constructor {name}: {rows:?}"));
            assert_eq!(row.flags, ["numeric-op"], "{field}: {row:?}");
        }
        assert!(numeric(&rows, "use::numbers"), "{field}: {rows:?}");
        assert!(!numeric(&rows, "use::booleans"), "{field}: {rows:?}");
    }
}

#[test]
fn transparent_aliases_and_nominal_fields_have_distinct_carrier_flags() {
    let rows = sources(&[(
        "flags",
        "module Std.Flags\nexport (bare, tagged)\ntype Scalar = f64\ntype Box[a] = | Box(a)\ndef bare(x: Scalar) -> Scalar = x\ndef tagged(x: Box[Scalar]) -> Box[Scalar] = x",
    )]);
    assert!(
        rows.iter()
            .find(|row| row.id.starts_with("flags::bare:"))
            .unwrap()
            .flags
            .contains(&"float-carrier".to_string())
    );
    assert!(
        rows.iter()
            .find(|row| row.id.starts_with("flags::tagged:"))
            .unwrap()
            .flags
            .is_empty()
    );
}

#[test]
fn declared_surface_does_not_infer_expression_bodies_or_capture_type_binders() {
    let rows = sources(&[(
        "binders",
        "module Std.Binders\nexport (p, ordinary, bounded)\ndef p(x: f64) -> f64 = absent_body_symbol(x)\nsig ordinary[p]: p -> p\ndef ordinary(x) = x\nsig bounded[p: Float]: p -> p\ndef bounded(x) = x",
    )]);
    assert!(numeric(&rows, "binders::p"));
    assert!(!numeric(&rows, "binders::ordinary"));
    assert!(numeric(&rows, "binders::bounded"));
}

#[test]
fn generic_summaries_substitute_used_parameter_positions() {
    let rows = sources(&[(
        "positions",
        "module Std.Positions\nexport (numbers, booleans)\ntype Number = f64\ntype Second[a, b] = b\ntype Numeric = Second[bool, Number]\ntype Logical = Second[Number, bool]\ndef numbers(x: Numeric) -> Numeric = x\ndef booleans(x: Logical) -> Logical = x",
    )]);
    assert!(numeric(&rows, "positions::numbers"));
    assert!(!numeric(&rows, "positions::booleans"));
}

#[test]
fn alias_body_changes_capacity_without_rewriting_the_authored_identity() {
    let rows = |dtype| {
        sources(&[(
            "width",
            &format!(
                "module Std.Width\nexport (read)\ntype Scalar = {dtype}\ndef read(x: Scalar) -> Scalar = x"
            ),
        )])
    };
    let narrow = rows("f32");
    let wide = rows("f64");
    assert_eq!(narrow.len(), 1);
    assert_eq!(narrow[0].id, wide[0].id);
    assert_eq!(narrow[0].flags, wide[0].flags);
    assert!(rows("bool").is_empty());
}

#[test]
fn every_structural_payload_edge_has_numeric_and_boolean_parity() {
    for shape in [
        "&DTYPE",
        "(bool, DTYPE)",
        "DTYPE -> bool",
        "bool -> DTYPE",
        "tensor[3, DTYPE]",
        "List[DTYPE]",
        "Option[DTYPE]",
        "Dict[DTYPE, bool]",
        "Dict[bool, DTYPE]",
    ] {
        for (dtype, expected) in [("i32", true), ("bool", false)] {
            let ty = shape.replace("DTYPE", dtype);
            let source = format!(
                "module Std.Edge\nexport (read)\ntype Payload = {ty}\ndef read(x: Payload) -> Payload = x"
            );
            assert_eq!(
                numeric(&sources(&[("edge", &source)]), "edge::read"),
                expected,
                "{ty}"
            );
        }
    }
}

#[test]
fn mutually_recursive_aliases_substitute_their_payload_parameter() {
    let rows = sources(&[(
        "mutual_alias",
        "module Std.MutualAlias\nexport (numbers, booleans)\ntype Left[a] = Option[Right[a]]\ntype Right[a] = (a, Left[a])\ndef numbers(x: Left[i64]) -> Left[i64] = x\ndef booleans(x: Left[bool]) -> Left[bool] = x",
    )]);
    assert!(numeric(&rows, "mutual_alias::numbers"));
    assert!(!numeric(&rows, "mutual_alias::booleans"));
}

#[test]
fn tensor_alias_precision_is_substituted_before_numeric_classification() {
    for (argument, expected) in [
        ("bool", false),
        ("i32", true),
        ("p", true),
        ("Identity[p]", true),
    ] {
        let binders = if argument.contains('p') { "[p]" } else { "" };
        let source = format!(
            "module Std.Precision\nexport (read)\ntype Identity[a] = a\ntype Vector[a] = tensor[3, a]\nsig read{binders}: Vector[{argument}] -> Vector[{argument}]\ndef read(x) = x"
        );
        assert_eq!(
            numeric(&sources(&[("precision", &source)]), "precision::read"),
            expected,
            "{argument}"
        );
    }
}

#[test]
fn legacy_precision_spellings_remain_a_conservative_backstop() {
    for (binder, expected) in [
        ("p_float", true),
        ("p_int", true),
        ("p_numeric", true),
        ("q", true),
        ("Q", true),
        ("p", false),
    ] {
        let source = format!(
            "module Std.Precision\nexport (read)\nsig read[{binder}]: {binder} -> {binder}\ndef read(x) = x"
        );
        assert_eq!(
            numeric(&sources(&[("precision", &source)]), "precision::read"),
            expected,
            "{binder}"
        );
    }
}

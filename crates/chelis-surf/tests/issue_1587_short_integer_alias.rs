//! chelis#1587: an `iN` spelling in a TYPE position names its integer
//! primitive, not an implicit type variable.
//!
//! Before #1587, the desugarer's primitive list omitted `i8`..`i64`, so one
//! of these names fell through to the lexical case-split and became a
//! quantified `(t-var {} i64)`. `def ident(x: i64) -> i64 = x` then accepted
//! an f64 and returned f64, because `i64` was `forall a. a`. Issue #1592 later
//! made the already-recognized `iN` spellings canonical.
//!
//! Labels, established by reverting both source files to `4ad308501`:
//! `every_short_integer_name_maps_to_its_long_primitive`,
//! `the_alias_maps_in_every_type_position` and
//! `the_canonical_formatter_rewrites_the_alias` are REGRESSION tests, red there
//! and green here. `a_genuine_lowercase_name_still_quantifies` and
//! `formatting_the_alias_is_idempotent` are DISPOSITION LOCKS, green in both
//! states: the first guards the neighbour the mapping must not disturb, and
//! the second held trivially before because the formatter left the alias alone.
//!
//! `i8` and `i16` have no site anywhere in the tree, so their rows lock names
//! the corpus does not otherwise exercise. Issue #1854 later made every
//! declaration binder explicit; the ordinary-variable control uses `[a]`.

use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_source;
use chelis_surf::parser::parse_str;

fn deep_of(source: &str) -> String {
    let decls = parse_str(source).expect("parse");
    chelis_deep::printer::print_canonical(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
    )
}

#[test]
fn every_integer_name_maps_to_its_primitive() {
    for name in ["i8", "i16", "i32", "i64"] {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n"
        ));
        assert!(
            deep.contains(&format!("(t-prim {{}} {name})")),
            "`{name}` in a scalar type position must desugar to its primitive: {deep}"
        );
        assert!(
            !deep.contains(&format!("(t-var {{}} {name})")),
            "`{name}` must not become an implicit type variable: {deep}"
        );
    }
}

#[test]
fn the_alias_maps_in_every_type_position() {
    // Signature, parameter annotation, tensor precision slot, and cast target.
    let deep = deep_of(
        "module P.M\nexport (main)\n\
         sig scale[n]: tensor[n, i64] -> tensor[n, i64]\n\
         def scale(t) = t\n\
         def ann(x: i32) -> i64 = cast(x, i64)\n\
         def main() -> i64 = ann(1i32)\n",
    );
    assert!(deep.contains("i64") && deep.contains("i32"), "{deep}");
    for name in ["i64", "i32"] {
        assert!(
            deep.contains(&format!("(t-prim {{}} {name})"))
                && !deep.contains(&format!("(t-var {{}} {name})")),
            "`{name}` must be a primitive in every type position: {deep}"
        );
    }
}

#[test]
fn a_genuine_explicit_lowercase_binder_still_quantifies() {
    let deep = deep_of("module P.M\nexport (f)\nsig f[a]: a -> a\ndef f(x) = x\n");
    assert!(
        deep.contains("(t-var"),
        "`a` must stay a type variable: {deep}"
    );
}

#[test]
fn the_canonical_formatter_rewrites_the_alias() {
    // #1592 made the names exercised by #1587 canonical Surf spellings.
    let source = "module P.M\nexport (f, g)\n\
                  sig g[n]: tensor[n, i64] -> tensor[n, i64]\n\
                  def g(t) = t\n\
                  def f(x: i32) -> i64 = cast(x, i64)\n";
    let formatted = format_source(source).expect("format");
    assert!(
        formatted.contains("i64") && formatted.contains("i32"),
        "the formatter must print the canonical integer names: {formatted}"
    );
}

#[test]
fn formatting_the_alias_is_idempotent() {
    let source = "module P.M\nexport (f)\ndef f(x: i32) -> i64 = cast(x, i64)\n";
    let once = format_source(source).expect("format once");
    let twice = format_source(&once).expect("format twice");
    assert_eq!(once, twice, "formatter idempotence on an aliased source");
}

/// [02-P10b]: the alias must reach literal metadata, not just the signature.
/// Regression: before normalization at contextual ingress these emit t-prim iN.
/// A declared tensor type is the only context that makes a bracket literal a
/// tensor, so these are the declared positions 1 and 3.
#[test]
fn contextual_tensor_literals_normalize_aliases_in_declared_positions() {
    for name in ["i8", "i16", "i32", "i64"] {
        for source in [
            format!("xs: tensor[2, {name}] = [1, -2]\n"),
            format!("sig xs: tensor[2, {name}]\nxs = [1, -2]\n"),
            format!("def f() -> tensor[2, {name}] = [1, -2]\n"),
            format!("sig f: i32 -> tensor[2, {name}]\ndef f(n) = [1, -2]\n"),
            format!("def f() = {{\n  xs: tensor[2, {name}] = [1, -2]\n  xs\n}}\n"),
        ] {
            let deep = deep_of(&source);
            assert!(
                deep.contains(&format!("(t-prim {{}} {name})"))
                    && !deep.contains(&format!("(t-var {{}} {name})")),
                "the canonical integer name must reach literal type metadata: {source}\n{deep}"
            );
        }
    }
}

/// [02-P10b]: an unsuffixed scalar adopts an aliased cast target too.
/// Regression: before normalization the source retained its default i32.
#[test]
fn scalar_cast_literals_adopt_the_canonical_integer_target() {
    for name in ["i8", "i16", "i32", "i64"] {
        for literal in ["5", "-5"] {
            let deep = deep_of(&format!("out = cast({literal}, {name})\n"));
            assert!(
                deep.contains("surf_literal_style")
                    && deep.contains(&format!("(t-prim {{}} {name})")),
                "an unsuffixed scalar must adopt the canonical target: {deep}"
            );
        }
    }
}

/// Negative parity: a suffix or truncating cast never adopts the target.
#[test]
fn alias_normalization_preserves_nonadopting_cast_sources() {
    for (source, source_type) in [
        ("out = cast(5i32, i64)\n", "i32"),
        ("out = cast(1.5, i64)\n", "f32"),
        ("out = cast_trunc(1.5, i64)\n", "f32"),
    ] {
        let deep = deep_of(source);
        assert!(!deep.contains("unsuffixed"), "{source}\n{deep}");
        assert!(deep.contains(source_type), "{source}\n{deep}");
        assert!(deep.contains("i64"), "{source}\n{deep}");
    }
}

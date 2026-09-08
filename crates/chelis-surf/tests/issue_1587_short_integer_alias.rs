//! chelis#1587: a short integer spelling in a TYPE position is an accepted
//! input spelling for its `intN` primitive, not an implicit type variable.
//!
//! Before this, `spec/04-type-system.md` §5.8.1 named `i8`..`i64` as primitives
//! while the desugarer's primitive list held only `int8`..`int64`, so a short
//! name failed the primitive test, fell through to the lexical case-split, and
//! became a quantified `(t-var {} i64)`. `def ident(x: i64) -> i64 = x` then
//! accepted an f64 and returned f64, because `i64` was `forall a. a`.
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
//! the corpus does not otherwise exercise.

use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_source;
use chelis_surf::parser::parse_str;

fn deep_of(source: &str) -> String {
    let decls = parse_str(source).expect("parse");
    chelis_deep::printer::print_canonical(&desugar_program(&decls))
}

#[test]
fn every_short_integer_name_maps_to_its_long_primitive() {
    for (short, long) in [
        ("i8", "int8"),
        ("i16", "int16"),
        ("i32", "int32"),
        ("i64", "int64"),
    ] {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f(x: {short}) -> {short} = x\n"
        ));
        assert!(
            deep.contains(long),
            "`{short}` in a scalar type position must desugar to `{long}`: {deep}"
        );
        // No long spelling contains its short form as a substring
        // (`int64` has no `i64` in it), so this is exact.
        assert!(
            !deep.contains(short),
            "`{short}` must not survive into Deep in any form: {deep}"
        );
    }
}

#[test]
fn the_alias_maps_in_every_type_position() {
    // Signature, parameter annotation, tensor precision slot, and cast target.
    let deep = deep_of(
        "module P.M\nexport (main)\n\
         sig scale: tensor[n, i64] -> tensor[n, i64]\n\
         def scale(t) = t\n\
         def ann(x: i32) -> i64 = cast(x, i64)\n\
         def main() -> i64 = ann(1i32)\n",
    );
    assert!(deep.contains("int64") && deep.contains("int32"), "{deep}");
    for short in ["i64", "i32"] {
        assert!(
            !deep.contains(short),
            "no short spelling may reach Deep: {short} in {deep}"
        );
    }
}

#[test]
fn a_genuine_lowercase_name_still_quantifies() {
    // The neighbour the mapping must not disturb: a non-primitive lowercase
    // name in a sig is still an implicitly quantified type variable.
    let deep = deep_of("module P.M\nexport (f)\nsig f: a -> a\ndef f(x) = x\n");
    assert!(
        deep.contains("(t-var"),
        "`a` must stay a type variable: {deep}"
    );
}

#[test]
fn the_canonical_formatter_rewrites_the_alias() {
    // §P10-P12's model: the parser accepts a wider set than the formatter
    // emits. Without this the alias would be a second canonical Surf spelling.
    let source = "module P.M\nexport (f, g)\n\
                  sig g: tensor[n, i64] -> tensor[n, i64]\n\
                  def g(t) = t\n\
                  def f(x: i32) -> i64 = cast(x, i64)\n";
    let formatted = format_source(source).expect("format");
    assert!(
        formatted.contains("int64") && formatted.contains("int32"),
        "the formatter must print the canonical long names: {formatted}"
    );
    assert!(
        !formatted.contains("i64") && !formatted.contains("i32"),
        "no short spelling may survive formatting: {formatted}"
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
#[test]
fn contextual_tensor_literals_normalize_aliases_in_all_four_positions() {
    for short in ["i8", "i16", "i32", "i64"] {
        for source in [
            format!("xs: tensor[2, {short}] = [1, -2]\n"),
            format!("def f(x: tensor[2, {short}]) -> tensor[2, {short}] = x\nr = f([1, -2])\n"),
            format!("def f() -> tensor[2, {short}] = [1, -2]\n"),
            format!("xs = cast([1, -2], {short})\n"),
            format!("def f() = {{\n  xs: tensor[2, {short}] = [1, -2]\n  xs\n}}\n"),
            format!(
                "sig f: tensor[2, {short}] -> tensor[2, {short}]\ndef f(x) = x\nr = f([1, -2])\n"
            ),
        ] {
            let deep = deep_of(&source);
            assert!(
                !deep.contains(short),
                "an input alias must not survive in literal type metadata: {source}\n{deep}"
            );
        }
    }
}

/// [02-P10b]: an unsuffixed scalar adopts an aliased cast target too.
/// Regression: before normalization the source retained its default int32.
#[test]
fn scalar_cast_literals_adopt_the_canonical_alias_target() {
    for short in ["i8", "i16", "i32", "i64"] {
        for literal in ["5", "-5"] {
            let deep = deep_of(&format!("out = cast({literal}, {short})\n"));
            assert!(
                deep.contains("surf_literal_style") && !deep.contains(short),
                "an unsuffixed scalar must adopt the canonical target: {deep}"
            );
        }
    }
}

/// Negative parity: a suffix or truncating cast never adopts the target.
#[test]
fn alias_normalization_preserves_nonadopting_cast_sources() {
    for (source, source_type) in [
        ("out = cast(5i32, i64)\n", "int32"),
        ("out = cast(1.5, i64)\n", "f32"),
        ("out = cast_trunc(1.5, i64)\n", "f32"),
    ] {
        let deep = deep_of(source);
        assert!(!deep.contains("unsuffixed"), "{source}\n{deep}");
        assert!(deep.contains(source_type), "{source}\n{deep}");
        assert!(deep.contains("int64"), "{source}\n{deep}");
    }
}

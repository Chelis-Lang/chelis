//! Surf surface for `spec/04-type-system.md` §5.9 dtype-family bounds
//! (chelis#1417).
//!
//! Covers the §P4c grammar on both declaration forms, the `dtype_bounds` Deep
//! encoding of `spec/03-deep-syntax.md` §1.1, canonical formatting, and the
//! §0.1 round-trip laws for a bounded declaration. Every accepting case has
//! its rejecting mirror.

use chelis_deep::parser::parse_str_strict as deep_parse_strict;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_surf::resugar::resugar_program;

fn deep_text(source: &str) -> String {
    let decls = surf_parse(source).unwrap_or_else(|error| panic!("parse `{source}`: {error}"));
    print_canonical(&desugar_program(&decls).expect("Surf fixture must desugar"))
}

fn formatted(source: &str) -> String {
    let decls = surf_parse(source).unwrap_or_else(|error| panic!("parse `{source}`: {error}"));
    format_program(&decls)
}

// === §P4c grammar: accepted spellings ===

#[test]
fn sig_carries_a_binder_list_with_a_float_bound() {
    let text = deep_text("sig linspace[n, p: Float]: p -> p -> i64 -> tensor[n, p]");
    assert!(
        text.contains("dtype_bounds: {p: float}"),
        "expected a float bound on the defsig, got:\n{text}"
    );
}

#[test]
fn def_carries_a_binder_list_with_an_int_bound() {
    let text = deep_text("def arange_values[p: Int](current: p, stop: p) -> p = current");
    assert!(
        text.contains("dtype_bounds: {p: int}"),
        "expected an int bound on the synthesized defsig, got:\n{text}"
    );
}

#[test]
fn numeric_is_a_declarable_family() {
    let text = deep_text("sig total[p: Numeric]: p -> p -> p");
    assert!(
        text.contains("dtype_bounds: {p: numeric}"),
        "expected a numeric bound, got:\n{text}"
    );
}

#[test]
fn bounded_and_unbounded_binders_mix_in_one_list() {
    let text = deep_text("def scale[n, p: Float](x: tensor[n, p], k: p) -> tensor[n, p] = x");
    assert!(
        text.contains("dtype_bounds: {p: float}"),
        "expected only `p` bounded, got:\n{text}"
    );
    assert!(
        text.contains("(d-var {} n)"),
        "an unbounded binder in a dim slot stays a dim variable:\n{text}"
    );
}

#[test]
fn every_bounded_binder_occurrence_is_one_type_variable() {
    let text = deep_text("sig pick[p: Int]: p -> p -> p");
    assert_eq!(
        text.matches("(t-var {} p)").count(),
        3,
        "each occurrence resolves to the same named binder:\n{text}"
    );
}

#[test]
fn a_lambda_valued_binding_desugars_its_sig_binder_cast_target_as_t_var() {
    // chelis#1625: `recast = fn (v) -> cast(v, p)` under a standalone
    // `sig recast[p]: p -> p` must desugar the cast target as `(t-var {} p)`,
    // the same spelling the `def recast(value) -> p = cast(value, p)` form
    // produces, so the [04-DTYPE-1] classifier (which keys on a `t-var`
    // target) recognizes it. Before the fix, this desugared to
    // `(t-prim {} p)` and slipped past the checker at both ingresses.
    let declarations = surf_parse("sig recast[p]: p -> p\nrecast = fn (v) -> cast(v, p)")
        .expect("parse signature and lambda");
    let deep = chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
        &desugar_program(&declarations).expect("Surf fixture must desugar"),
    )
    .expect("normalize source spans");
    let text = chelis_deep::printer::print_canonical_flat(&deep);
    assert!(
        text.contains("(cast {} (var {} v) (t-var {} p))"),
        "expected the cast target to desugar as `(t-var {{}} p)`, got:\n{text}"
    );
    assert!(
        !text.contains("(t-prim {} p)"),
        "the cast target must not desugar as `(t-prim {{}} p)`:\n{text}"
    );
}

#[test]
fn a_lambda_valued_binding_does_not_rebind_a_primitive_sig_binder_name() {
    let error = surf_parse("sig recast[f32]: f32 -> f32\nrecast = fn (v) -> cast(v, f32)")
        .expect_err("an active primitive cannot enter the binder list");
    assert!(
        error
            .to_string()
            .contains("`f32` cannot be a declaration binder"),
        "the binder-list owner must reject the active primitive: {error}"
    );
}

#[test]
fn a_lambda_valued_binding_does_not_rebind_a_reserved_sig_binder_name() {
    let error = surf_parse("sig recast[u8]: u8 -> u8\nrecast = fn (v) -> cast(v, u8)")
        .expect_err("a reserved dtype cannot enter the binder list");
    assert!(
        error
            .to_string()
            .contains("`u8` cannot be a declaration binder"),
        "the binder-list owner must reject the reserved dtype: {error}"
    );
}

#[test]
fn a_declaration_without_a_bound_emits_no_bound_metadata() {
    // Canonical Deep for every pre-existing declaration is unchanged.
    let text = deep_text("def id[p](x: p) -> p = x");
    assert!(
        !text.contains("dtype_bounds"),
        "an unbounded binder list must not add metadata:\n{text}"
    );
}

#[test]
fn a_bound_forces_a_signature_even_with_no_other_annotation() {
    // The bound has no other carrier; without the synthesized `defsig` it
    // would be silently dropped.
    let text = deep_text("def f[p: Float](x) = x");
    assert!(
        text.contains("defsig") && text.contains("dtype_bounds: {p: float}"),
        "expected a synthesized bounded defsig, got:\n{text}"
    );
}

// === §P4c grammar: rejected spellings ===

#[test]
fn an_adt_name_is_not_a_bound() {
    let error = surf_parse("sig f[p: Tensor]: p -> p").expect_err("must reject");
    let rendered = error.to_string();
    assert!(
        rendered.contains("dtype family"),
        "expected a dtype-family diagnostic, got: {rendered}"
    );
}

#[test]
fn a_lowercase_family_spelling_is_rejected() {
    // Surf spells families PascalCase; the lowercase form is Deep's.
    surf_parse("sig f[p: float]: p -> p").expect_err("must reject a Deep spelling in Surf");
}

#[test]
fn a_primitive_name_is_not_a_bound() {
    surf_parse("sig f[p: f32]: p -> p").expect_err("must reject a concrete dtype as a bound");
}

#[test]
fn a_bound_requires_a_family_after_the_colon() {
    surf_parse("def f[p: ](x: p) -> p = x").expect_err("must reject a missing family");
}

#[test]
fn an_empty_binder_list_stays_rejected_on_a_sig() {
    surf_parse("sig f[]: i32 -> i32").expect_err("must reject `[]`");
}

// === Canonical formatting ===

#[test]
fn canonical_form_spells_a_bound_with_one_space() {
    assert_eq!(
        formatted("sig linspace[n,p:Float]: p -> p -> i64 -> tensor[n, p]").trim(),
        "sig linspace[n, p: Float]: p -> p -> i64 -> tensor[n, p]"
    );
}

#[test]
fn canonical_form_preserves_authored_binder_order() {
    assert_eq!(
        formatted("def scale[n, p: Float](x: tensor[n, p], k: p) -> tensor[n, p] = x").trim(),
        "def scale[n, p: Float](x: tensor[n, p], k: p) -> tensor[n, p] = x"
    );
}

#[test]
fn formatting_a_bounded_declaration_is_idempotent() {
    let once = formatted("sig arange[n, p: Int]: p -> p -> tensor[n, p]");
    let twice = formatted(&once);
    assert_eq!(once, twice, "chelis fmt must reach a fixed point");
}

// === spec/02 §0.1 round-trip laws ===

#[test]
fn bounded_deep_reparses_and_reprints_identically() {
    for source in [
        "sig arange[n, p: Int]: p -> p -> tensor[n, p]",
        "def arange_values[p: Int](current: p, stop: p) -> p = current",
        "def scale[n, p: Float](x: tensor[n, p], k: p) -> tensor[n, p] = x",
        "sig total[p: Numeric]: p -> p -> p",
    ] {
        let printed = deep_text(source);
        let reparsed = deep_parse_strict(&printed)
            .unwrap_or_else(|error| panic!("strict Deep re-parse of `{source}`: {error}"));
        assert_eq!(
            print_canonical(&reparsed),
            printed,
            "Deep round-trip is not stable for `{source}`"
        );
    }
}

#[test]
fn desugar_of_resugar_is_the_identity_on_bounded_declarations() {
    // spec/02 §0.1 law 1: `desugar(resugar(·))` on canonical Deep.
    for source in [
        "sig arange[n, p: Int]: p -> p -> tensor[n, p]",
        "def arange_values[p: Int](current: p, stop: p) -> p = current",
        "def scale[n, p: Float](x: tensor[n, p], k: p) -> tensor[n, p] = x",
        "def scale[p: Float](x: p) -> p = mul(x, cast(0.1, p))",
        // chelis#1625: a lambda-valued top-level binding whose type comes
        // from a standalone bounded sig must round-trip the same as the
        // equivalent `def` spelling.
        "sig recast[p: Float]: p -> p\nrecast = fn (v) -> cast(v, p)",
    ] {
        let decls = surf_parse(source).expect("parse");
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        let recovered =
            resugar_program(&deep).unwrap_or_else(|error| panic!("resugar `{source}`: {error}"));
        // `span` is derived surface provenance normalized away by
        // spec/03 §6.3.2, exactly as the canonical-Surf law harness does.
        assert_eq!(
            print_canonical(
                &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
                    &desugar_program(&recovered).expect("Surf fixture must desugar")
                )
                .expect("valid metadata for round-trip normalization")
            ),
            print_canonical(
                &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&deep)
                    .expect("valid metadata for round-trip normalization")
            ),
            "resugar/desugar is not the identity for `{source}`"
        );
    }
}

#[test]
fn resugaring_recovers_the_authored_bound_spelling() {
    let decls = surf_parse("sig arange[n, p: Int]: p -> p -> tensor[n, p]").expect("parse");
    let recovered = resugar_program(&desugar_program(&decls).expect("Surf fixture must desugar"))
        .expect("resugar");
    assert_eq!(
        format_program(&recovered).trim(),
        "sig arange[n, p: Int]: p -> p -> tensor[n, p]"
    );
}

#[test]
fn resugaring_an_unbounded_sig_preserves_its_complete_binder_list() {
    let decls = surf_parse("sig identity[n, p]: tensor[n, p] -> tensor[n, p]").expect("parse");
    let recovered = resugar_program(&desugar_program(&decls).expect("Surf fixture must desugar"))
        .expect("resugar");
    assert_eq!(
        format_program(&recovered).trim(),
        "sig identity[n, p]: tensor[n, p] -> tensor[n, p]"
    );
}

#[test]
fn a_malformed_deep_bound_fails_resugaring_closed() {
    deep_parse_strict("(defsig {dtype_bounds: {p: signed}} f (p) (t-var {} p))")
        .expect_err("unknown dtype families are rejected at ingress");
}

fn binder_deep(body: &str) -> Vec<chelis_deep::Expr> {
    deep_parse_strict(&format!(
        "(defsig {{dtype_bounds: {{p: float}}}} scale (p) (t-fn {{}} (t-var {{}} p) (t-var {{}} p)))\n\
         (def {{}} scale (fn {{}} (params {{}} (x {{type: (t-var {{}} p)}})) {body}))"
    ))
    .expect("Deep fixture")
}

#[test]
fn unary_minus_literal_source_preserves_its_adopting_cast_through_resugar() {
    let deep = desugar_program(
        &surf_parse("def scale[p: Float](x: p) -> p = cast(-0.1, p)").expect("Surf"),
    )
    .expect("Surf fixture must desugar");
    let printed = print_canonical(&deep);
    assert!(
        printed.contains("surf_literal_style: \"unsuffixed\", type: (t-var {} p)} -0.1)")
            && !printed.contains("} neg)"),
        "Surf unary syntax must become an unambiguous signed literal: {printed}"
    );
    let recovered = format_program(
        &resugar_program(&deep).expect("the adopting cast relation is representable"),
    );
    assert!(recovered.contains("cast(-0.1, p)"), "{recovered}");

    for source in [
        "def concrete(x: f64) -> f64 = cast(-0.1f64, f64)",
        "def bounded[p: Float](x: p) -> p = cast(-0.1f64, p)",
    ] {
        let printed = deep_text(source);
        assert!(
            printed.contains("} neg)") && printed.contains("} 0.1)"),
            "a suffixed negative keeps the ordinary unary application: {printed}"
        );
    }
}

#[test]
fn a_local_neg_call_is_not_a_negative_literal_source() {
    let text = deep_text(
        "def neg(x: f32) -> f32 = add(x, 1.0)\n\
         def scale[p: Float](x: p) -> p = cast(neg(0.1), p)",
    );
    assert!(
        text.matches("(app ").count() == 2 && text.contains("} neg)"),
        "the authored call must survive instead of sign-folding: {text}"
    );
}

#[test]
fn unrepresentable_binder_literal_provenance_fails_resugaring() {
    for deep in [
        deep_parse_strict("(defsig {} scale (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n(def {} scale (fn {} (params {} (x {type: (t-prim {} f32)})) (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1) (t-var {} p))))").expect("Deep fixture"),
        binder_deep("(lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1)"),
        deep_parse_strict(
            "(defsig {dtype_bounds: {p: float}} scale (p) (t-fn {} (t-fn {} (t-var {} p) (t-var {} p)) (t-var {} p) (t-var {} p)))\n\
             (def {} scale (fn {} (params {} (neg {type: (t-fn {} (t-var {} p) (t-var {} p))}) (x {type: (t-var {} p)}))\n\
               (cast {} (app {} (var {} neg) (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1)) (t-var {} p))))",
        )
        .expect("forged local neg Deep fixture"),
    ] {
        resugar_program(&deep).expect_err("unrepresentable binder provenance");
    }
    for marker in [
        "surf_literal_style: \"explicit\", ",
        "surf_literal_style: 1, ",
        "",
    ] {
        let source = format!(
            "(defsig {{dtype_bounds: {{p: float}}}} scale (p) (t-fn {{}} (t-var {{}} p) (t-var {{}} p))) (def {{}} scale (fn {{}} (params {{}} (x {{type: (t-var {{}} p)}})) (cast {{}} (lit {{{marker}type: (t-var {{}} p)}} 0.1) (t-var {{}} p))))"
        );
        match deep_parse_strict(&source) {
            Ok(deep) => {
                resugar_program(&deep).expect_err("unrepresentable binder provenance");
            }
            Err(error) => assert!(error.to_string().contains("surf_literal_style")),
        }
    }
}

#[test]
fn duplicate_authored_bounds_reject_before_deep_construction() {
    for source in [
        "sig f[p: Float, p: Int]: p -> p\ndef f(x) = x",
        "def f[p: Float, p: Int](x: p) -> p = x",
    ] {
        assert!(chelis_surf::parser::parse_str(source).is_err(), "{source}");
    }
    let declarations = chelis_surf::parser::parse_str("def f[p: Float](x: p) -> p = x").unwrap();
    chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
}

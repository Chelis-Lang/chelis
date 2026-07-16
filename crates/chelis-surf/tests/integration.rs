//! Integration tests: Surf parse → desugar → Deep print → Deep parse roundtrip.

use chelis_deep::parser::parse_str_strict as deep_parse_strict;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;

/// Parse Surf source, desugar to Deep, print Deep, re-parse Deep with strict
/// tag validation. Verifies the generated Deep uses only canonical 59-tag
/// vocabulary and round-trips through the Deep parser.
fn roundtrip(surf_source: &str) {
    // 1. Parse Surf
    let decls = surf_parse(surf_source).unwrap_or_else(|e| {
        panic!("Surf parse failed for:\n{surf_source}\nError: {e}");
    });

    // 2. Desugar to Deep
    let deep_exprs = desugar_program(&decls);
    assert!(
        !deep_exprs.is_empty(),
        "Desugarer produced no output for:\n{surf_source}"
    );

    // 3. Print Deep
    let deep_text = print_canonical(&deep_exprs);
    assert!(
        !deep_text.is_empty(),
        "Deep printer produced empty output for:\n{surf_source}"
    );

    // 4. Re-parse Deep with STRICT validation (unknown tags = error)
    let reparsed = deep_parse_strict(&deep_text).unwrap_or_else(|e| {
        panic!(
            "Deep strict re-parse failed.\nOriginal Surf:\n{surf_source}\nDeep output:\n{deep_text}\nError: {e}"
        );
    });

    // 5. Re-print and compare (Deep round-trip stability)
    let reprinted = print_canonical(&reparsed);
    assert_eq!(
        deep_text, reprinted,
        "Deep round-trip mismatch.\nFirst print:\n{deep_text}\nSecond print:\n{reprinted}"
    );
}

// === Fixture roundtrips ===

#[test]
fn roundtrip_simple_def() {
    roundtrip(include_str!("fixtures/simple_def.ch"));
}

#[test]
fn roundtrip_property_decl_desugars_to_property_metadata() {
    let source = r"
@property non_negative forall(x: f32) where x >= 0.0:
  x >= 0.0
  with samples = 3
";
    let decls = surf_parse(source).expect("property parses");
    let deep_exprs = desugar_program(&decls);
    let deep_text = print_canonical(&deep_exprs);
    assert!(deep_text.contains("chelis_role: \"property\""));
    assert!(deep_text.contains("property_source_kind: \"user\""));
    assert!(deep_text.contains("property_quantifiers: (params {}"));
    assert!(deep_text.contains("property_preconditions: (tuple {}"));
    assert!(deep_text.contains("property_samples"));
    deep_parse_strict(&deep_text).expect("property Deep validates");
}

#[test]
fn opaque_type_decl_desugars_to_metadata() {
    let source = r"
@opaque
type Probability = | Probability { value: f32 }
";
    let decls = surf_parse(source).expect("opaque type parses");
    let deep_exprs = desugar_program(&decls);
    let deep_text = print_canonical(&deep_exprs);
    assert!(deep_text.contains("opaque: true"));
    assert!(deep_text.contains("Probability"));
    // An opaque type without an invariant carries neither key.
    assert!(!deep_text.contains("invariant"));
    deep_parse_strict(&deep_text).expect("opaque type Deep validates");
}

/// Desugar a single Surf module and print canonical Deep.
fn desugar_to_deep(source: &str) -> String {
    let decls = surf_parse(source).expect("Surf parses");
    let deep_exprs = desugar_program(&decls);
    print_canonical(&deep_exprs)
}

#[test]
fn invariant_linear_desugars_to_three_metadata_keys() {
    let deep_text = desugar_to_deep(
        "module Stats.Prob\n@opaque\n\
         @invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
         type Probability = | Probability { value: f32 }",
    );
    assert!(deep_text.contains("opaque: true"));
    // Embedded predicate fn with the binder in a params node. The
    // canonical printer pretty-prints across lines, so assert on the
    // key + fn opener and the params node separately.
    assert!(deep_text.contains("invariant: (fn {}"));
    assert!(deep_text.contains("(params {} p)"));
    assert!(deep_text.contains("invariant_amenability: \"linear\""));
    // The embedded fn (a full Deep expr in the metadata map) round-trips
    // through the strict validator.
    deep_parse_strict(&deep_text).expect("invariant Deep validates strictly");
}

#[test]
fn invariant_polynomial_amenability_recorded() {
    let deep_text = desugar_to_deep(
        "module M\n@opaque\n@invariant(p) (p.value * p.value) <= 1.0\n\
         type Probability = | Probability { value: f32 }",
    );
    assert!(deep_text.contains("invariant_amenability: \"polynomial\""));
    deep_parse_strict(&deep_text).expect("polynomial invariant validates");
}

#[test]
fn invariant_transcendental_amenability_recorded() {
    let deep_text = desugar_to_deep(
        "module M\n@opaque\n@invariant(p) exp(p.value) <= 3.0\n\
         type Probability = | Probability { value: f32 }",
    );
    assert!(deep_text.contains("invariant_amenability: \"transcendental\""));
    deep_parse_strict(&deep_text).expect("transcendental invariant validates");
}

#[test]
fn simplex_tolerance_band_desugars_and_validates() {
    // W6 flagship declaration: sum over a literal-shape tensor field with
    // a module-constant tolerance band. Must classify linear and the
    // embedded predicate must validate strictly.
    let deep_text = desugar_to_deep(
        "module Stats.Simplex\n@opaque\n\
         @invariant(p) (sum(p.weights) >= (1.0 - eps)) && (sum(p.weights) <= (1.0 + eps))\n\
         type Simplex = | Simplex { weights: tensor[3, f32] }",
    );
    assert!(deep_text.contains("invariant: (fn {}"));
    assert!(deep_text.contains("(params {} p)"));
    assert!(deep_text.contains("invariant_amenability: \"linear\""));
    // The `sum` callee is present (it carries a span; match the suffix).
    assert!(deep_text.contains("sum)"));
    deep_parse_strict(&deep_text).expect("simplex invariant validates strictly");
}

#[test]
fn roundtrip_untyped_def() {
    roundtrip(include_str!("fixtures/untyped_def.ch"));
}

#[test]
fn roundtrip_operators() {
    roundtrip(include_str!("fixtures/operators.ch"));
}

#[test]
fn roundtrip_adt() {
    roundtrip(include_str!("fixtures/adt.ch"));
}

#[test]
fn roundtrip_block_binding_expr() {
    roundtrip(include_str!("fixtures/block_binding_expr.ch"));
}

#[test]
fn roundtrip_lambda() {
    roundtrip(include_str!("fixtures/lambda.ch"));
}

#[test]
fn roundtrip_pipe() {
    roundtrip(include_str!("fixtures/pipe.ch"));
}

#[test]
fn roundtrip_match() {
    roundtrip(include_str!("fixtures/match_expr.ch"));
}

// === Example programs roundtrip ===

#[test]
fn roundtrip_example_pattern_matching() {
    roundtrip(include_str!(
        "../../../examples/illustrative/pattern_matching.ch"
    ));
}

#[test]
fn roundtrip_example_pipeline() {
    roundtrip(include_str!("../../../examples/illustrative/pipeline.ch"));
}

#[test]
fn roundtrip_example_hello_tensor() {
    roundtrip(include_str!("../../../examples/hello_tensor.ch"));
}

#[test]
fn roundtrip_example_mnist() {
    roundtrip(include_str!("../../../examples/mnist.ch"));
}

#[test]
fn roundtrip_example_linreg() {
    roundtrip(include_str!("../../../examples/linreg.ch"));
}

#[test]
fn roundtrip_example_linear_regression() {
    roundtrip(include_str!(
        "../../../examples/illustrative/linear_regression.ch"
    ));
}

#[test]
fn roundtrip_example_mlp() {
    roundtrip(include_str!("../../../examples/illustrative/mlp.ch"));
}

// === Dim param desugaring ===

#[test]
fn dim_params_produce_d_var() {
    let decls = surf_parse("def transpose[batch, hidden](x: tensor[batch, hidden, f32]): tensor[hidden, batch, f32] = x").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    // batch and hidden are declared dim params → must be d-var, not d-name
    assert!(
        text.contains("(d-var {} batch)"),
        "expected d-var for 'batch', got:\n{text}"
    );
    assert!(
        text.contains("(d-var {} hidden)"),
        "expected d-var for 'hidden', got:\n{text}"
    );
    // No defdim — function dim params are polymorphic, not module-level
    assert!(
        !text.contains("defdim"),
        "function dim params should not emit defdim:\n{text}"
    );
}

// === Wildcard dimension ===

#[test]
fn wildcard_dimension_desugars_to_d_name_star() {
    let decls = surf_parse("def f(x: tensor[*, f32]): tensor[*, f32] = x").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(d-name {} *)"),
        "expected (d-name {{}} *) for wildcard dim, got:\n{text}"
    );
}

// === Specific output shape checks ===

#[test]
fn typed_def_emits_defsig() {
    let decls = surf_parse("def f(x: f32): f32 = x").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(text.contains("(defsig {} f"), "Missing defsig in:\n{text}");
    assert!(text.contains("(def {}\n  f"), "Missing def in:\n{text}");
}

#[test]
fn type_var_not_prim() {
    let decls = surf_parse("type Option[a] = | Some(a) | None").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(t-var {} a)"),
        "Expected t-var for type param 'a', got:\n{text}"
    );
}

#[test]
fn annotation_in_metadata() {
    let decls = surf_parse("def f(x) = x : f32").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("type: (t-prim {} f32)"),
        "Expected type annotation in metadata, got:\n{text}"
    );
}

#[test]
fn pat_lit_raw_value() {
    let decls =
        surf_parse("type B = | T | F\ndef f(x: f32): f32 = match x with { | 0 => 1 | _ => x }")
            .unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(pat-lit {} 0)"),
        "Expected raw value in pat-lit, got:\n{text}"
    );
}

#[test]
fn no_legacy_tags() {
    // Verify no legacy tags appear in desugared output
    let decls = surf_parse("def f(x: f32): f32 = x + 1").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        !text.contains("(sig "),
        "Legacy 'sig' tag found in:\n{text}"
    );
    assert!(
        !text.contains("(apply "),
        "Legacy 'apply' tag found in:\n{text}"
    );
    assert!(
        !text.contains("(: "),
        "Legacy ':' annotation tag found in:\n{text}"
    );
}

#[test]
fn effect_annotations_desugar_into_t_fn_metadata() {
    let decls = surf_parse("sig f: f32 -> f32 ! {Diff, Random, Resource(\"gpu:0\")}").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("{eff: (effects {} diff random (resource {} \"gpu:0\"))}"),
        "Expected effect metadata on t-fn, got:\n{text}"
    );
}

// #290: a function-typed *argument* — `(a -> b) -> c` — must desugar to a
// nested `t-fn` in argument position, distinct from the flat curried
// `a -> b -> c`. This is the type distinction the formatter and decompiler
// must preserve.
#[test]
fn hof_argument_sig_desugars_to_nested_t_fn() {
    let hof = surf_parse("sig f: (a -> b) -> c").unwrap();
    let hof_text = print_canonical(&desugar_program(&hof));
    assert!(
        hof_text.contains("(t-fn {} (t-fn {} (t-var {} a) (t-var {} b)) (t-var {} c))"),
        "HOF sig did not desugar to a nested t-fn; got:\n{hof_text}"
    );
    deep_parse_strict(&hof_text).expect("HOF sig Deep validates");

    let curried = surf_parse("sig f: a -> b -> c").unwrap();
    let curried_text = print_canonical(&desugar_program(&curried));
    assert!(
        curried_text.contains("(t-fn {} (t-var {} a) (t-var {} b) (t-var {} c))"),
        "curried sig did not desugar to a flat t-fn; got:\n{curried_text}"
    );

    // The Deep representations of the two sigs must differ.
    assert_ne!(
        hof_text, curried_text,
        "HOF and curried sigs collapsed to the same Deep type"
    );
}

// === Bare-keyword pipe stages (Item 2b / G11) ===
//
// The pipe-stage parser at `parser.rs::parse_pipe_stage` recognizes bare
// unary-builtin keyword tokens (`Realize`, `Copy`) and synthesizes an
// explicit lambda over a fresh `__chelis_pipe` parameter, so the
// desugarer produces the canonical `(pipe ... (fn (v) -> (realize v)))`
// Deep shape per spec `01-nomenclature.md` §3.6 first-argument insertion.
//
// Only the unary builtins have spec-meaningful bare pipe-stage form;
// keywords taking additional arguments (`grad`, `vmap`, `cast`, `jit`,
// `with`) keep their existing arg-form behavior, and the structural forms
// (`if`, `match`, `fn`, `par`) keep their existing rejection — see
// `docs/investigations/parser_pipe_bare_keyword_diagnosis.md`.

#[test]
fn bare_realize_as_pipe_stage_parses() {
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize";
    let decls = surf_parse(src).expect("bare `|> realize` should parse");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    // Canonical desugar: pipe with a synthesized unary lambda calling
    // the unary builtin on the piped value.
    assert!(
        text.contains("(pipe {"),
        "expected pipe node in desugared output, got:\n{text}"
    );
    assert!(
        text.contains("__chelis_pipe") && text.contains("realize"),
        "expected synthesized lambda body `(realize {{}} (var {{}} __chelis_pipe))`, got:\n{text}"
    );
    assert!(
        text.contains("(params {} __chelis_pipe)"),
        "expected fresh `__chelis_pipe` param, got:\n{text}"
    );
    // Deep round-trips with strict tag validation.
    let reparsed = deep_parse_strict(&text).expect("synthesized lambda Deep validates");
    assert_eq!(text, print_canonical(&reparsed));
}

#[test]
fn bare_copy_as_pipe_stage_parses() {
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy";
    let decls = surf_parse(src).expect("bare `|> copy` should parse");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(pipe {"),
        "expected pipe node in desugared output, got:\n{text}"
    );
    // Synthesized lambda over the piped value: `(fn ... (params __chelis_pipe)
    // (copy ... (var ... __chelis_pipe)))`. Metadata maps carry source spans
    // so we match on the structural body rather than literal `{}` markers.
    assert!(
        text.contains("(params {} __chelis_pipe)"),
        "expected fresh `__chelis_pipe` param, got:\n{text}"
    );
    assert!(
        text.contains("(copy {") && text.contains("(var {") && text.contains("__chelis_pipe))"),
        "expected `(copy {{...}} (var {{...}} __chelis_pipe))` body, got:\n{text}"
    );
    let reparsed = deep_parse_strict(&text).expect("synthesized lambda Deep validates");
    assert_eq!(text, print_canonical(&reparsed));
}

#[test]
fn chained_bare_keyword_with_named_pipe_stages_parses() {
    // Mixed pipe: bare keyword stage chained with a regular named-function
    // stage. Exercises the pipe-loop boundary (the parser must not consume
    // the next `|>` while parsing the bare keyword stage).
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize |> relu";
    let decls = surf_parse(src).expect("chained bare-keyword + named pipe should parse");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(pipe {"),
        "expected pipe node in desugared output, got:\n{text}"
    );
    // The bare-keyword stage produces the synthesized lambda.
    assert!(
        text.contains("__chelis_pipe") && text.contains("realize"),
        "expected bare-realize stage to expand to lambda body, got:\n{text}"
    );
    // The named-ident stage must remain a `var` reference (no lambda wrap).
    assert!(
        text.contains("relu)") && !text.contains("relu __chelis_pipe"),
        "expected named-ident pipe stage `(var ... relu)` to survive (no lambda wrap), got:\n{text}"
    );
}

#[test]
fn keyword_with_arg_form_in_pipe_stage_still_parses() {
    // Negative: the arg form `cast(value, type)` outside of pipe context
    // remains unchanged. The bare recognition in `parse_pipe_stage` does
    // not trigger because the token after `realize` in a hypothetical
    // `... |> realize(x)` form (or after `cast` in `cast(x, f32)`) is
    // `LParen`. This test exercises a non-pipe call followed by a bare
    // pipe stage to ensure neither path interferes with the other.
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = cast(x, f32) |> realize";
    let decls = surf_parse(src).expect("cast arg form + bare-realize pipe should parse");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    // The cast call retains its arg form.
    assert!(
        text.contains("(cast {"),
        "expected (cast {{...}} ...) in:\n{text}"
    );
    // The bare-realize stage still synthesizes the canonical lambda body.
    assert!(
        text.contains("__chelis_pipe"),
        "expected synthesized __chelis_pipe param, got:\n{text}"
    );
}

#[test]
fn fmt_compacts_bare_keyword_pipe_stages_back_to_keyword_form() {
    // Surf → Surf format round-trip: `x |> realize` and `x |> copy` must
    // re-emit as the bare keyword form (not the synthesized lambda). Spec
    // `01-nomenclature.md` §3.6 permits this compaction for any
    // single-arg-first-position pipe-stage lambda; Item 2b adds the unary
    // builtin keywords to the formatter's compact path.
    use chelis_surf::format::format_program;
    for (label, src) in [
        (
            "bare-realize",
            "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize",
        ),
        (
            "bare-copy",
            "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy",
        ),
        (
            "chained",
            "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize |> relu",
        ),
    ] {
        let decls = surf_parse(src).expect(label);
        let formatted = format_program(&decls);
        assert!(
            formatted.contains("|> realize") || formatted.contains("|> copy"),
            "{label}: expected bare-keyword pipe form in formatter output, got:\n{formatted}"
        );
        assert!(
            !formatted.contains("__chelis_pipe"),
            "{label}: formatter must compact synthesized lambdas, but `__chelis_pipe` leaked:\n{formatted}"
        );
        // Idempotency: formatting the formatted output produces the same
        // text (no further compaction or expansion).
        let reformatted = format_program(&surf_parse(&formatted).expect(label));
        assert_eq!(formatted, reformatted, "{label}: fmt is not idempotent");
    }
}

#[test]
fn unsupported_structural_keyword_pipe_stage_still_rejected() {
    // Bare structural keywords (`if`, `match`, `fn`, `par`) have no
    // callable-form interpretation per spec §3.6, so the parser must
    // still reject them. The `with` keyword (effect handler) is in the
    // same family.
    for kw in ["if", "match", "fn", "par", "with"] {
        let src = format!("def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> {kw}");
        let result = surf_parse(&src);
        assert!(
            result.is_err(),
            "expected parser to reject bare `|> {kw}` (no callable-form semantics), \
             but it parsed successfully: {src}"
        );
    }
}

// === Bare-keyword sibling-sweep extras (Item 2b H1/H2/H3) ===
//
// The pipe-stage fix in Item 2b (`parse_pipe_stage` synthesizes a lambda
// for `x |> realize` / `x |> copy`) intentionally left three sibling
// surfaces out of scope; see the "Sibling sweep" table in
// `docs/investigations/parser_pipe_bare_keyword_diagnosis.md`:
//
// H1: top-level bare unary-builtin reference, e.g. `f = realize`.
// H2: bare unary-builtin keyword as juxtaposition argument, e.g.
//     `apply_fn(realize)`.
// H3: one-arg `cast(type)` pipe-stage form, e.g. `x |> cast(f32)` which
//     per spec §3.6 ≡ `cast(x, f32)`.
//
// These three fixtures pin the failures. They flip to running in the
// fix commit (see `docs/investigations/pipe_autofix_and_bare_keyword_extras_diagnosis.md`).

#[test]
fn top_level_bare_unary_builtin_reference_parses() {
    // `f = realize` at top level binds the identifier `f` to the unary
    // builtin `realize`. Per the canonical lowering for bare pipe stages
    // (parser_pipe_bare_keyword_diagnosis.md), the natural η-expansion
    // is `fn (v) -> realize(v)`.
    for kw in ["realize", "copy"] {
        let src = format!("f = {kw}\n");
        let decls = surf_parse(&src)
            .unwrap_or_else(|e| panic!("bare top-level `{kw}` should parse, got: {e}"));
        assert_eq!(decls.len(), 1, "expected one decl for `{src}`");
        let deep = desugar_program(&decls);
        let text = print_canonical(&deep);
        // η-expanded body references the builtin.
        assert!(
            text.contains(kw),
            "{kw}: expected lowered output to mention `{kw}`, got:\n{text}"
        );
    }
}

#[test]
fn bare_unary_builtin_as_juxtaposition_argument_parses() {
    // `apply_fn(realize)` passes the unary builtin `realize` as a
    // function-valued argument. Same η-expansion as H1.
    for kw in ["realize", "copy"] {
        let src = format!("def f(x: tensor[3, f32]) -> tensor[3, f32] = apply_fn({kw})\n");
        let decls =
            surf_parse(&src).unwrap_or_else(|e| panic!("`apply_fn({kw})` should parse, got: {e}"));
        let deep = desugar_program(&decls);
        let text = print_canonical(&deep);
        // The `apply_fn` call must remain a single `app` of `apply_fn`
        // applied to the synthesized lambda (or builtin reference) — not
        // a parse of `apply_fn` as a zero-arg call.
        assert!(
            text.contains("apply_fn") && text.contains(kw),
            "{kw}: expected `apply_fn` call carrying `{kw}` body, got:\n{text}"
        );
    }
}

#[test]
fn one_arg_cast_pipe_stage_parses() {
    // Per spec §3.6, `x |> cast(f32)` ≡ `cast(x, f32)`: the piped value
    // fills the first slot, the type argument fills the second.
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> cast(f32)\n";
    let decls = surf_parse(src).expect("`x |> cast(f32)` should parse per spec §3.6");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(pipe {"),
        "expected pipe node in lowered output, got:\n{text}"
    );
    assert!(
        text.contains("(cast {"),
        "expected `(cast ...)` node receiving the piped value as first arg, got:\n{text}"
    );
    assert!(
        text.contains("f32"),
        "expected `f32` target type to survive lowering, got:\n{text}"
    );
}

#[test]
fn parsed_surf_expression_spans_enter_deep_metadata() {
    let source = "def f(x) = add(x, 1)";
    let decls = surf_parse(source).unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(app {span: \"surf:11..20\"}"),
        "Expected app node to carry its Surf byte range, got:\n{text}"
    );
    assert!(
        text.contains("(var {span: \"surf:11..14\"} add)"),
        "Expected callee var node to carry its Surf byte range, got:\n{text}"
    );
    assert!(
        text.contains("(lit {span: \"surf:18..19\", type: (t-prim {} int32)} 1)"),
        "Expected literal node to preserve both span and type metadata, got:\n{text}"
    );
}

#[test]
fn with_seed_desugars_to_handle_effect() {
    let decls = surf_parse("def f() = with seed(42) { dropout(x, 0.5) }").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(handle-effect {effect: random"),
        "Expected random handler node, got:\n{text}"
    );
}

#[test]
fn with_device_desugars_to_handle_effect() {
    let decls = surf_parse("def f() = with device(\"gpu:0\") { x }").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(handle-effect {effect: resource"),
        "Expected resource handler node, got:\n{text}"
    );
}

// === Finding 3a (0.7.6 red-team, PR #51): fmt non-idempotent on 3+ stage pipes ===
//
// `chelis fmt` must be a fixed point: `fmt(fmt(src)) == fmt(src)`. The red-team
// agent reported that a 3+ stage pipe like `x |> f |> g |> h` produces a
// multi-line braced form on pass 1, then a different one-line braced form on
// pass 2; `fmt --check` rejects the formatter's own pass-1 output.
//
// Pass 1 emits:
//   def f(...) -> ... = {
//     x
//     |> neg
//     |> abs
//     |> sigmoid
//   }
// Pass 2 collapses that re-parsed Block to:
//   def f(...) -> ... = { x
//   |> neg
//   |> abs
//   |> sigmoid }
//
// The fix changes the formatter's emit path for multi-stage pipes so the
// chosen layout survives a parse/emit cycle. These tests flip to running in
// the fix commit; see
// `docs/investigations/fmt_pipe_idempotency_diagnosis.md`.

/// `fmt(fmt(src))` must equal `fmt(src)`. Asserts a fixed point of the Surf
/// formatter for the given Surf source.
fn assert_fmt_idempotent(label: &str, src: &str) {
    use chelis_surf::format::format_program;
    let pass1 = format_program(
        &surf_parse(src)
            .unwrap_or_else(|e| panic!("{label}: pass-0 parse failed: {e}\nsrc:\n{src}")),
    );
    let pass2 = format_program(
        &surf_parse(&pass1)
            .unwrap_or_else(|e| panic!("{label}: pass-1 re-parse failed: {e}\npass1:\n{pass1}")),
    );
    assert_eq!(
        pass1, pass2,
        "{label}: fmt is not idempotent.\n--- pass1 ---\n{pass1}\n--- pass2 ---\n{pass2}"
    );
    // `fmt --check` semantics: pass-1 output is byte-identical to formatting
    // its own parse.
    let pass1_check = format_program(&surf_parse(&pass1).expect("pass-1 must remain parseable"));
    assert_eq!(
        pass1, pass1_check,
        "{label}: fmt --check would reject pass-1 output."
    );
}

#[test]
fn fmt_idempotent_three_stage_pipe() {
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> neg |> abs |> sigmoid\n";
    assert_fmt_idempotent("three-stage pipe", src);
}

#[test]
fn fmt_idempotent_four_stage_pipe() {
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> neg |> abs |> sigmoid |> tanh\n";
    assert_fmt_idempotent("four-stage pipe", src);
}

#[test]
fn fmt_idempotent_three_stage_pipe_with_sibling_decl() {
    // Mixed with a non-pipe top-level decl, to ensure the fix scopes to the
    // pipe layout and does not regress neighboring formatting.
    let src = "\
def g(y: tensor[3, f32]) -> tensor[3, f32] = relu(y)
def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> neg |> abs |> sigmoid
";
    assert_fmt_idempotent("three-stage pipe + sibling", src);
}

/// chelis#461: a compound expression (`if`/`match`/`fn`/`|>`/block) used as a
/// BINARY OPERAND must be parenthesized, or `fmt` is both non-idempotent and
/// MEANING-CHANGING -- the operator binds into the operand's tail on re-parse.
/// This pins idempotency AND meaning-preservation (the re-parsed AST equals the
/// once-formatted AST) for the if-as-operand shape that exposed the bug.
#[test]
fn fmt_if_as_binary_operand_is_idempotent_and_meaning_preserving() {
    use chelis_surf::format::format_program;
    let source = "def f(x: f32, y: f32) -> f32 = (if x > y then x else y) + 1.0\n";
    let decls1 = surf_parse(source).expect("parse");
    let pass1 = format_program(&decls1);
    // The if-operand is wrapped, so `+ 1.0` applies to the whole if-expression.
    assert!(
        pass1.contains("((if (x > y) then x else y) + 1.0)"),
        "the if operand must be parenthesized so + 1.0 binds outside the else: {pass1}"
    );
    let decls2 = surf_parse(&pass1).expect("pass-1 output re-parses");
    let pass2 = format_program(&decls2);
    // Idempotency: a second format is byte-identical.
    assert_eq!(pass1, pass2, "fmt must be idempotent on an if-as-operand");
    // Meaning preservation: the bug rendering -- in which the formatted text
    // re-parses so `+ 1.0` joins the else-branch -- must not occur. With the
    // fix, the once-formatted text re-parses to a form whose own canonical
    // formatting is the SAME text (a stable fixpoint), so the proposition the
    // formatter shows is the one the parser reads back.
    assert!(
        !pass1.contains("else y + 1.0"),
        "the operator must not bind into the else-branch on re-parse: {pass1}"
    );
}

/// Corpus sweep: every `.ch` file under `examples/`, `packages/chelis-std/`,
/// and `crates/chelis-surf/tests/fixtures/` must satisfy
/// `fmt(fmt(src)) == fmt(src)`. Locks the formatter-idempotency contract
/// across the executable and illustrative example surfaces so a future
/// regression in `format_block`, `format_pipe_layout`, or a sibling path
/// cannot reintroduce Finding 3a quietly.
#[test]
fn fmt_idempotent_on_repo_ch_corpus() {
    use chelis_surf::format::format_program;
    use std::path::PathBuf;

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");

    let roots = [
        repo_root.join("examples"),
        repo_root.join("packages/chelis-std"),
        repo_root.join("crates/chelis-surf/tests/fixtures"),
    ];

    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack: Vec<PathBuf> = roots.iter().filter(|p| p.exists()).cloned().collect();
    while let Some(dir) = stack.pop() {
        for entry in
            std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("ch") {
                files.push(path);
            }
        }
    }
    assert!(
        !files.is_empty(),
        "corpus sweep found zero .ch files; check root paths"
    );

    let mut failures: Vec<String> = Vec::new();
    for path in &files {
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let Ok(decls1) = surf_parse(&src) else {
            // Files that do not parse today are out of scope for the
            // formatter contract. The dedicated parser tests cover them.
            continue;
        };
        let pass1 = format_program(&decls1);
        let Ok(decls2) = surf_parse(&pass1) else {
            failures.push(format!(
                "{}: pass-1 output failed to re-parse",
                path.display()
            ));
            continue;
        };
        let pass2 = format_program(&decls2);
        if pass1 != pass2 {
            failures.push(format!("{}: fmt not idempotent", path.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "fmt idempotency corpus sweep found {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

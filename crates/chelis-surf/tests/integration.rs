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
    let source = r#"
@property non_negative forall(x: f32) where x >= 0.0:
  x >= 0.0
  with samples = 3
"#;
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

// === Bare-keyword pipe stages (Item 2b / G11) ===
//
// Pinned by failing tests until the parser accepts bare keyword tokens as
// pipe stages. The pipe-stage parser at `parser.rs:1000-1003` dispatches to
// `parse_prefix`, which routes every reserved keyword (Realize, Copy, Grad,
// Vmap, Jit, Par, Cast, If, Match, Fn, With) to a parser that unconditionally
// expects an `LParen` or `LBrace` next. The fix should let bare keyword forms
// produce a callable reference in pipe-stage context, matching the spec's
// first-argument insertion rule (`spec/01-nomenclature.md` §3.6).
//
// Only the keywords whose builtin form is **unary** have a meaningful bare
// pipe-stage semantics — currently `realize` and `copy`. Keywords that take
// additional arguments (`grad`, `vmap`, `cast`, `jit`, `with`, `par`) use the
// arg form (e.g. `x |> grad(f)`), and the structural forms (`if`, `match`,
// `fn`) have no bare-callable interpretation. The fixtures here cover the
// unary cases plus a chained-pipe canary.

#[test]
#[ignore = "parser rejects bare keyword pipe stage, see commit fix/parser-pipe-bare-keyword"]
fn bare_realize_as_pipe_stage_parses() {
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize";
    let decls = surf_parse(src).expect("bare `|> realize` should parse");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    // Canonical desugar: `(pipe {} (var {} x) <realize-callable>)`, matching
    // spec §3.6 first-argument insertion semantics for `realize`.
    assert!(
        text.contains("(pipe {}"),
        "expected pipe node in desugared output, got:\n{text}"
    );
}

#[test]
#[ignore = "parser rejects bare keyword pipe stage, see commit fix/parser-pipe-bare-keyword"]
fn bare_copy_as_pipe_stage_parses() {
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy";
    let decls = surf_parse(src).expect("bare `|> copy` should parse");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(pipe {}"),
        "expected pipe node in desugared output, got:\n{text}"
    );
}

#[test]
#[ignore = "parser rejects bare keyword pipe stage, see commit fix/parser-pipe-bare-keyword"]
fn chained_bare_keyword_with_named_pipe_stages_parses() {
    // Mixed pipe: bare keyword stage chained with a regular named-function
    // stage. Exercises the pipe-loop boundary (the parser must not consume
    // the next `|>` while parsing the bare keyword stage).
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize |> relu";
    let decls = surf_parse(src).expect("chained bare-keyword + named pipe should parse");
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(pipe {}"),
        "expected pipe node in desugared output, got:\n{text}"
    );
    // The named-ident stage must remain a `var` reference.
    assert!(
        text.contains("(var {} relu)"),
        "expected named-ident pipe stage (var relu) to survive, got:\n{text}"
    );
}

#[test]
fn bare_realize_pipe_stage_rejection_is_pinned() {
    // Regression pin for the current bug: parser rejects `x |> realize`. When
    // the fix lands, this test must be inverted (assert success) and the three
    // `#[ignore]` tests above must be flipped to running.
    let src = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize";
    let err = surf_parse(src).expect_err(
        "expected parser to reject `x |> realize` (current G11 behavior); \
         if this now parses successfully, the fix has landed — invert this test \
         and flip the ignored fixtures above to running",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("expected LParen"),
        "expected `expected LParen` diagnostic from bare-keyword rejection, got: {msg}"
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

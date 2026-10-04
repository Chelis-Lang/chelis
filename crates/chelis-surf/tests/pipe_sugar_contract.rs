//! Acceptance oracle for #3130: Surf pipes, one Deep application representation.
use chelis_deep::printer::print_canonical;
use chelis_surf::resugar::{normalize_deep_for_surface_roundtrip, resugar_program};
use chelis_surf::{desugar::desugar_program, format::format_source, parser::parse_str};

fn meaning(source: &str) -> String {
    let decls = parse_str(source).expect("Surf parses");
    let deep = desugar_program(&decls).expect("Surf desugars");
    print_canonical(&normalize_deep_for_surface_roundtrip(&deep).expect("valid Deep"))
}

#[test]
fn pipes_and_calls_have_identical_deep_before_checking() {
    for (pipe, call) in [
        ("x |> f", "f(x)"),
        ("x |> f(y, z)", "f(x, y, z)"),
        ("x |> f |> g(y)", "g(f(x), y)"),
        ("0.1 |> cast(f64)", "cast(0.1, f64)"),
        ("0.1f32 |> cast(f64)", "cast(0.1f32, f64)"),
        ("x |> cast_trunc(i32)", "cast_trunc(x, i32)"),
        ("x |> copy", "copy(x)"),
        ("x |> realize", "realize(x)"),
        ("x |> (fn (v) -> f(y, v))", "(fn (v) -> f(y, v))(x)"),
    ] {
        let actual = meaning(&format!("out = {pipe}\n"));
        assert!(!actual.contains("(pipe "), "{actual}");
        assert!(!actual.contains("surf_pipe_stage"), "{actual}");
        assert_eq!(actual, meaning(&format!("out = {call}\n")), "{pipe}");
    }
}

#[test]
fn transform_stages_use_the_same_callable_resolution_as_calls() {
    let declaration = "def f(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n";
    for (pipe, call) in [
        ("x |> grad(f)", "grad(f)(x)"),
        ("x |> grad(f, wrt=x)", "grad(f, wrt=x)(x)"),
        ("x |> vmap(f)", "vmap(f)(x)"),
        ("x |> jit(f)", "jit(f)(x)"),
    ] {
        assert_eq!(
            meaning(&format!("{declaration}out = {pipe}\n")),
            meaning(&format!("{declaration}out = {call}\n")),
            "{pipe}",
        );
    }
    for expression in ["x |> grad(f, wrt=missing)", "grad(f, wrt=missing)(x)"] {
        let declarations = parse_str(&format!("{declaration}out = {expression}\n")).unwrap();
        assert!(
            matches!(desugar_program(&declarations), Err(chelis_surf::desugar::DesugarError::UnknownGradParameter { parameter, .. }) if parameter == "missing"),
            "{expression}",
        );
    }
}

#[test]
fn every_mixed_operator_requires_grouping() {
    for operator in [
        "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=", "&&", "||",
    ] {
        let source = format!("out = a {operator} b |> f\n");
        let error = parse_str(&source)
            .expect_err("ungrouped mix must fail")
            .to_string();
        assert!(error.contains("group"), "{source}: {error}");
        parse_str(&format!("out = (a {operator} b) |> f\n")).expect("grouped seed");
        parse_str(&format!("out = a {operator} (b |> f)\n")).expect("grouped pipe");
    }
    for source in [
        "out = -x |> f\n",
        "out = !x |> f\n",
        "out = &x |> f\n",
        "out = if c then a else b |> f\n",
        "out = fn (v) -> v |> f\n",
        "out = x |> fn (v) -> v + y\n",
        "out = x: i32 |> f\n",
        "out = x |> f: i32\n",
        "out = x with { a: y } |> f\n",
        "out = match x |> f with { | _ => x }\n",
    ] {
        assert!(parse_str(source).is_err(), "{source}");
    }
}

#[test]
fn grouped_pipes_preserve_fmt_and_surface_retraction() {
    for source in [
        "out = (a * b) |> f\n",
        "out = (-x) |> f\n",
        "out = (if c then a else b) |> f\n",
        "out = x |> (fn (v) -> v + y)\n",
        "out = x |> f(a + b) |> g\n",
    ] {
        let formatted = format_source(source).expect("format");
        assert_eq!(format_source(&formatted).unwrap(), formatted);
        assert_eq!(meaning(source), meaning(&formatted));
        let deep = desugar_program(&parse_str(source).unwrap()).unwrap();
        let surf = chelis_surf::format::format_program(&resugar_program(&deep).unwrap());
        assert!(!surf.contains("|>"), "{surf}");
        assert_eq!(format_source(&surf).unwrap(), surf);
        assert_eq!(meaning(source), meaning(&surf));
    }
}

#[test]
fn removed_deep_forms_and_spelling_metadata_fail_closed() {
    for source in [
        "(pipe {} (var {} x) (var {} f))",
        "(fn {surf_pipe_stage: \"call-first\"} (params {} v) (var {} v))",
    ] {
        assert!(
            chelis_deep::parse_and_stamp_runtime_exprs(source).is_err(),
            "{source}"
        );
    }
}

#[test]
fn migration_preserves_literal_dtype_and_old_grouping() {
    let source = "out = 0.1 |> cast(f64)\n";
    let baseline = "(def {} out (pipe {} (lit {span: \"surf:6..9\", type: (t-prim {} f32)} 0.1) (fn {surf_pipe_stage: \"call-first\"} (params {} p) (cast {} (var {} p) (t-prim {} f64)))))";
    let migration = chelis_surf::pipe_migration::prepare(source, baseline).unwrap();
    assert_eq!(migration.source, "out = 0.1f32 |> cast(f64)\n");
    assert_eq!(
        meaning(&migration.source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&migration.baseline).unwrap())
    );
    assert_eq!(format_source(&migration.source).unwrap(), migration.source);
    assert!(chelis_surf::pipe_migration::prepare(source, "(def {} out (var {} x))").is_err());

    let migration = chelis_surf::pipe_migration::prepare(
        "out = 2.0 * x |> f\n",
        "(def {} out (pipe {} (app {} (var {} mul) (lit {span: \"surf:6..9\", type: (t-prim {} f32)} 2.0) (var {} x)) (var {} f)))",
    ).unwrap();
    assert_eq!(migration.source, "out = (2.0f32 * x) |> f\n");
    assert_eq!(
        meaning(&migration.source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&migration.baseline).unwrap())
    );
}

#[test]
fn migration_preserves_trailing_lambda_and_conditional_grouping() {
    for (source, previous) in [
        (
            "out = x |> fn (v) -> g(v) |> h\n",
            "(def {} out (pipe {} (var {} x) (fn {} (params {} v) (pipe {} (app {} (var {} g) (var {} v)) (var {} h)))))",
        ),
        (
            "out = if c then a else b |> f\n",
            "(def {} out (if {} (var {} c) (var {} a) (pipe {} (var {} b) (var {} f))))",
        ),
    ] {
        let migrated = chelis_surf::pipe_migration::prepare(source, previous).unwrap();
        assert_eq!(format_source(&migrated.source).unwrap(), migrated.source);
        assert_eq!(
            meaning(&migrated.source),
            print_canonical(&normalize_deep_for_surface_roundtrip(&migrated.baseline).unwrap()),
        );
    }
}

#[test]
fn macro_arguments_and_templates_preserve_pipe_grouping() {
    fn expanded(source: &str) -> String {
        let deep = desugar_program(&parse_str(source).unwrap()).unwrap();
        let deep =
            chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
                .unwrap()
                .into_exprs();
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep).unwrap())
    }
    for (pipe, call) in [
        (
            "macro apply(x) = x |> f\nout = apply(a + b)\n",
            "macro apply(x) = f(x)\nout = apply(a + b)\n",
        ),
        (
            "macro keep(x) = x\nout = keep(a + b) |> f\n",
            "macro keep(x) = x\nout = f(keep(a + b))\n",
        ),
        (
            "macro apply(x) = x |> cast(f64)\nout = apply(0.1f32)\n",
            "macro apply(x) = cast(x, f64)\nout = apply(0.1f32)\n",
        ),
        (
            "macro keep(x) = x\nout = keep(a |> f) |> g\n",
            "macro keep(x) = x\nout = g(keep(f(a)))\n",
        ),
    ] {
        assert_eq!(expanded(pipe), expanded(call));
    }
    assert!(parse_str("macro bad(x) = x + b |> f\n").is_err());
}

#[test]
fn invalid_programmatic_stage_fails_without_panicking() {
    use chelis_surf::ast::*;
    let span = chelis_deep::Span::new(7, 4);
    for syntax in [
        PipeStageSyntax::CallFirst,
        PipeStageSyntax::Cast(CastMode::Checked),
        PipeStageSyntax::Copy,
        PipeStageSyntax::Realize,
    ] {
        let expr = Expr::Pipe(
            Box::new(Expr::Var("x".into(), span)),
            vec![PipeStage {
                expression: Expr::Var("f".into(), span),
                syntax,
            }],
            span,
        );
        let error = chelis_surf::desugar::desugar_expr_only(&expr).unwrap_err();
        assert_eq!(error.span(), Some(span));
    }
}

#[test]
fn retired_head_rejection_preserves_binders_and_macro_source_data() {
    chelis_deep::parse_and_stamp_file(
        "(defsig {} f (pipe) (t-fn {} (t-var {} pipe) (t-var {} pipe)))",
    )
    .unwrap();
    chelis_deep::parse_and_stamp_file(
        "(def {source: (pipe {} historic)} x (lit {type: (t-prim {} i32)} 1))",
    )
    .unwrap();
    let raw =
        chelis_deep::parser::parse_pipe_migration_raw("(pipe {} (var {} x) (var {} f))").unwrap();
    let error = chelis_deep::stamp_runtime_exprs(raw).unwrap_err();
    assert!(error.to_string().contains("0.20") && error.to_string().contains("pipe"));
}

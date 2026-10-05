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
fn delimited_handlers_are_atoms_while_record_updates_require_grouping() {
    for (pipe, call) in [
        (
            "with device(\"cpu\") { x } |> f",
            "f(with device(\"cpu\") { x })",
        ),
        (
            "(with device(\"cpu\") { x }) |> f",
            "f(with device(\"cpu\") { x })",
        ),
        (
            "x |> with device(\"cpu\") { f }",
            "(with device(\"cpu\") { f })(x)",
        ),
        (
            "x |> (with device(\"cpu\") { f })",
            "(with device(\"cpu\") { f })(x)",
        ),
    ] {
        let source = format!("out = {pipe}\n");
        let expected = meaning(&format!("out = {call}\n"));
        assert_eq!(meaning(&source), expected, "{pipe}");
        let formatted = format_source(&source).unwrap();
        assert_eq!(meaning(&formatted), expected, "{formatted}");
        assert_eq!(format_source(&formatted).unwrap(), formatted);
    }
    for expression in [
        "r with { a: x } |> f",
        "r with\n{ a: x } |> f",
        "x |> r with { a: f }",
        "with device(\"cpu\") { a + b |> f }",
        "with device(\"cpu\") x |> f",
        "with unknown(\"cpu\") { x } |> f",
    ] {
        assert!(
            parse_str(&format!("out = {expression}\n")).is_err(),
            "{expression}"
        );
    }
    for expression in [
        "(r with { a: x }) |> f",
        "x |> (r with { a: f })",
        "with device(\"cpu\") { (a + b) |> f } |> g",
    ] {
        let source = format!("out = {expression}\n");
        let formatted = format_source(&source).unwrap();
        assert_eq!(meaning(&source), meaning(&formatted));
        assert_eq!(format_source(&formatted).unwrap(), formatted);
    }
}

#[test]
fn every_named_cast_stage_preserves_its_mode_and_rejects_a_mismatched_descriptor() {
    use chelis_deep::{CastMode, NamedCastMode};
    use chelis_surf::ast::{Decl, Expr, PipeStageSyntax};
    for (index, mode) in NamedCastMode::ALL.iter().enumerate() {
        let keyword = mode.keyword();
        let source = format!("out = x |> {keyword}(i8)\n");
        assert_eq!(
            meaning(&source),
            meaning(&format!("out = {keyword}(x, i8)\n"))
        );
        assert_eq!(format_source(&source).unwrap(), source);
        let mut declarations = parse_str(&source).unwrap();
        let Decl::LetDef {
            value: Expr::Pipe(_, stages, _),
            ..
        } = &mut declarations[0]
        else {
            panic!("expected authored pipe")
        };
        stages[0].syntax = PipeStageSyntax::Cast(CastMode::Named(
            NamedCastMode::ALL[(index + 1) % NamedCastMode::ALL.len()],
        ));
        assert!(
            matches!(
                desugar_program(&declarations),
                Err(chelis_surf::desugar::DesugarError::InvalidPipeStage { .. })
            ),
            "{keyword}: mismatched rung must fail closed"
        );
    }
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

fn callable_context_source(carried: &str, selector: &str) -> String {
    format!(
        "type Holder[a] = | Hold(a)\n\
         def f(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
         held = {carried}\n\
         chosen = match held with {{ | Hold(g) => g }}\n\
         out = grad(chosen, wrt={selector})\n"
    )
}

#[test]
fn contextual_declarations_normalize_before_callable_resolution() {
    use chelis_deep::{DeepTag, ExprCarrier};
    use chelis_surf::{ast::Decl, desugar::desugar_expr_in_program_scope};
    let mut results = Vec::new();
    for carried in ["f |> Hold", "Hold(f)"] {
        let declarations = parse_str(&callable_context_source(carried, "x")).unwrap();
        let Decl::LetDef { value, .. } = declarations.last().unwrap() else {
            panic!("output declaration");
        };
        let whole = desugar_program(&declarations).unwrap();
        let ExprCarrier::DecodedNode(DeepTag::Def, _, children) = whole.last().unwrap().carrier()
        else {
            panic!("output definition");
        };
        let contextual =
            desugar_expr_in_program_scope(&declarations[..declarations.len() - 1], value, &[])
                .expect("contextual pipe origins agree with whole-program origins");
        let actual = print_canonical(&normalize_deep_for_surface_roundtrip(&[contextual]).unwrap());
        let expected =
            print_canonical(&normalize_deep_for_surface_roundtrip(&[children[1].clone()]).unwrap());
        assert_eq!(actual, expected, "{carried}");
        results.push(actual);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn contextual_normalization_preserves_selector_and_shadow_rejections() {
    use chelis_surf::{
        ast::Decl,
        desugar::{DesugarError, desugar_expr_in_program_scope},
    };
    for carried in ["f |> Hold", "Hold(f)"] {
        let declarations = parse_str(&callable_context_source(carried, "missing")).unwrap();
        let Decl::LetDef { value, .. } = declarations.last().unwrap() else {
            panic!("output declaration");
        };
        for result in [
            desugar_program(&declarations).map(|_| ()),
            desugar_expr_in_program_scope(&declarations[..declarations.len() - 1], value, &[])
                .map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(DesugarError::UnknownGradParameter { parameter, .. }) if parameter == "missing"),
                "{carried}"
            );
        }
        let declarations = parse_str(&callable_context_source(carried, "x")).unwrap();
        let Decl::LetDef { value, .. } = declarations.last().unwrap() else {
            panic!("output declaration");
        };
        assert!(
            matches!(
                desugar_expr_in_program_scope(&declarations[..declarations.len() - 1], value, &["chosen".into()]),
                Err(DesugarError::UnresolvedGradTarget { target, .. }) if target == "chosen"
            ),
            "lexical shadow remains unresolved: {carried}"
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
        "out = x |> f.1\n",
        "out = x |> r.f\n",
        "out = r.f |> g\n",
        "out = M.f(x) |> g\n",
        "out = if c then x |> f else y\n",
        "out = fn (v) -> v |> f\n",
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
        "out = x |> (f.1)\n",
        "out = (x |> f).1\n",
        "out = (r.f) |> g\n",
        "out = (M.f(x)) |> g\n",
        "out = if c then (x |> f) else y\n",
        "out = fn (v) -> (v |> f)\n",
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
    assert_eq!(migration.source, "out = (2.0 * x) |> f\n");
    assert_eq!(
        meaning(&migration.source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&migration.baseline).unwrap())
    );
}

#[test]
fn migration_signed_seed_uses_the_signed_source_span_and_only_changed_dtypes() {
    let source = "out = cast(-1.0, f32) |> f\n";
    let previous = "(def {} out (pipe {} (cast {} (lit {span: \"surf:11..15\", type: (t-prim {} f32)} -1.0) (t-prim {} f32)) (var {} f)))";
    let migrated = chelis_surf::pipe_migration::prepare(source, previous).unwrap();
    assert_eq!(migrated.source, source);
    assert_eq!(
        meaning(source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&migrated.baseline).unwrap())
    );
    assert!(
        chelis_surf::pipe_migration::prepare(
            source,
            "(def {} out (pipe {} (var {} x) (var {} f)))"
        )
        .is_err()
    );
}

#[test]
fn migration_keeps_authored_first_argument_lambdas_as_values() {
    let source = "out = x |> (fn (v) -> h(v))\n";
    let previous = "(def {} out (pipe {} (var {} x) (fn {span: \"surf:12..26\", surf_pipe_stage: \"call-first\"} (params {} v) (app {} (var {} h) (var {} v)))))";
    let migrated = chelis_surf::pipe_migration::prepare(source, previous).unwrap();
    assert_eq!(
        meaning(source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&migrated.baseline).unwrap())
    );
    assert!(print_canonical(&migrated.baseline).contains("(fn "));
}

#[test]
fn migration_uses_validated_metadata_without_erasing_extensions() {
    let source = "out = 0.1 |> f\n";
    let previous = "(def {} out (pipe {} (lit {span: \"surf:6..9\", type: (t-prim {} f32), surf_literal_style: \"unsuffixed\", audit_note: \"literal\"} 0.1) (var {span: \"surf:13..14\"} f)))";
    let migration = chelis_surf::pipe_migration::prepare(source, previous).unwrap();
    assert_eq!(migration.source, "out = 0.1 |> f\n");
    let baseline = print_canonical(&migration.baseline);
    assert!(baseline.contains("audit_note"));
    assert!(baseline.contains("literal"));
    assert!(baseline.contains("surf:13..14"));
    assert!(baseline.contains("surf_literal_style"));
    for metadata in [
        "span: \"surf:6..9\", span: \"surf:6..9\", type: (t-prim {} f32)",
        "span: \"surf:6..9\", type: (t-prim {} f32), type: (t-prim {} f64)",
        "span: true, type: (t-prim {} f32)",
        "span: \"surf:6..9\", type: true",
    ] {
        let previous = format!("(def {{}} out (pipe {{}} (lit {{{metadata}}} 0.1) (var {{}} f)))");
        assert!(
            chelis_surf::pipe_migration::prepare(source, &previous).is_err(),
            "{metadata}"
        );
    }
}

#[test]
fn migration_preserves_metadata_owner_and_bind_value_placement() {
    let source = "out = {\n  x = 0.1 |> f\n  x\n}\n";
    let start = source.find("0.1").unwrap();
    let literal = format!(
        "(lit {{span: \"surf:{start}..{}\", type: (t-prim {{}} f32), surf_literal_style: \"unsuffixed\"}} 0.1)",
        start + 3
    );
    let previous = format!(
        "(def {{}} out (let {{}} (bind {{}} x (pipe {{surf_binding_type: \"inferred\"}} {literal} (var {{}} f))) (var {{}} x)))"
    );
    let migrated = chelis_surf::pipe_migration::prepare(source, &previous).unwrap();
    assert!(migrated.source.contains("0.1 |> f"));
    assert!(print_canonical(&migrated.baseline).contains("surf_binding_type"));
    assert_eq!(
        meaning(&migrated.source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&migrated.baseline).unwrap())
    );

    // Literal spelling stays on the literal, while binding annotations require
    // an actual bind-value parent; neither is admitted on a generic map.
    let source = "out = {\n  x = 0.1\n  x |> f\n}\n";
    let start = source.find("0.1").unwrap();
    let previous = format!(
        "(def {{}} out (let {{}} (bind {{}} x (lit {{span: \"surf:{start}..{}\", type: (t-prim {{}} f32), surf_literal_style: \"unsuffixed\", surf_binding_type: \"inferred\"}} 0.1)) (pipe {{}} (var {{}} x) (var {{}} f))))",
        start + 3
    );
    chelis_surf::pipe_migration::prepare(source, &previous).unwrap();

    for (source, previous, key) in [
        (
            "out = x |> f\n",
            "(def {} out (pipe {surf_binding_type: \"inferred\"} (var {} x) (var {} f)))",
            "surf_binding_type",
        ),
        (
            "out = x |> f\n",
            "(def {} out (pipe {} (block {surf_literal_style: \"unsuffixed\"} (var {} x)) (var {} f)))",
            "surf_literal_style",
        ),
        (
            "out = 0.1 |> f\n",
            "(def {} out (pipe {} (lit {surf_binding_type: \"inferred\", span: \"surf:6..9\", type: (t-prim {} f32)} 0.1) (var {} f)))",
            "surf_binding_type",
        ),
    ] {
        let error = match chelis_surf::pipe_migration::prepare(source, previous) {
            Ok(_) => panic!("accepted misplaced {key}: {previous}"),
            Err(error) => error,
        };
        assert!(error.contains(key), "{error}");
    }
}

#[test]
fn migration_validates_compacting_stage_annotations_before_erasure() {
    for (stage, body) in [
        ("f(y)", "(app {} (var {} f) (var {} p) (var {} y))"),
        ("cast(f64)", "(cast {} (var {} p) (t-prim {} f64))"),
        ("copy", "(copy {} (var {} p))"),
        ("realize", "(realize {} (var {} p))"),
    ] {
        let source = format!("out = x |> {stage}\n");
        let previous = |metadata: &str| {
            format!(
                "(def {{}} out (pipe {{}} (var {{}} x) (fn {{{metadata}, surf_pipe_stage: \"call-first\"}} (params {{}} p) {body})))"
            )
        };
        let migrated =
            chelis_surf::pipe_migration::prepare(&source, &previous("span: \"surf:11..20\""))
                .unwrap();
        assert_eq!(
            meaning(&migrated.source),
            print_canonical(&normalize_deep_for_surface_roundtrip(&migrated.baseline).unwrap())
        );
        for (metadata, key) in [
            ("span: true", "span"),
            ("span: \"a\", span: \"b\"", "span"),
            ("type: true", "type"),
            ("surf_literal_style: \"unsuffixed\"", "surf_literal_style"),
            ("surf_binding_type: \"inferred\"", "surf_binding_type"),
        ] {
            let error = match chelis_surf::pipe_migration::prepare(&source, &previous(metadata)) {
                Ok(_) => panic!("accepted malformed {key} on compacting stage {stage}"),
                Err(error) => error,
            };
            assert!(error.contains(key), "{stage}: {error}");
        }
    }
}

#[test]
fn migration_requires_node_maps_and_preserves_macro_source_data() {
    let source = "out = x |> f(y)\n";
    let previous = "(def {} out (pipe {} (var {} x) (fn {surf_pipe_stage: \"call-first\"} (params {} p) (app {} (var {} f) (var {} p) (var {} y)))))";
    let valid = chelis_surf::pipe_migration::prepare(source, previous).unwrap();
    assert_eq!(
        meaning(&valid.source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&valid.baseline).unwrap())
    );
    for (owner, pattern) in [
        ("fn", "(fn {surf_pipe_stage: \"call-first\"}"),
        ("pipe", "(pipe {}"),
    ] {
        for replacement in ["true", "(var {} x)", "()", ""] {
            let malformed = previous.replace(pattern, &format!("({owner} {replacement}"));
            let error = match chelis_surf::pipe_migration::prepare(source, &malformed) {
                Ok(_) => panic!("accepted non-map envelope for {owner}: {replacement}"),
                Err(error) => error,
            };
            assert!(error.contains("metadata map"), "{error}");
        }
    }
    for data in [
        "(pipe x y z)",
        "(fn true x y)",
        "(pipe (lit {span: true} 1) x y)",
    ] {
        let previous =
            format!("(def {{}} out (pipe {{}} (var {{}} x) (var {{source: {data}}} f)))");
        let migrated = chelis_surf::pipe_migration::prepare("out = x |> f\n", &previous).unwrap();
        assert!(print_canonical(&migrated.baseline).contains(data), "{data}");
    }
    for data in ["true", "(true x y)"] {
        let previous =
            format!("(def {{}} out (pipe {{}} (var {{}} x) (var {{source: {data}}} f)))");
        let error = match chelis_surf::pipe_migration::prepare("out = x |> f\n", &previous) {
            Ok(_) => panic!("accepted malformed source record: {data}"),
            Err(error) => error,
        };
        assert!(error.contains("source"), "{error}");
    }
}

#[test]
fn migration_preserves_structural_binder_roles_without_admitting_program_shapes() {
    let source =
        "sig identity[pipe: Numeric] : pipe -> pipe\ndef identity(x) = x\nout = 1i32 |> identity\n";
    let previous = "(defsig {dtype_bounds: {pipe: numeric}} identity (pipe) (t-fn {} (t-var {} pipe) (t-var {} pipe)))\n(def {} identity (fn {} (params {} x) (var {} x)))\n(def {} out (pipe {} (lit {type: (t-prim {} i32)} 1) (var {} identity)))";
    let migrated = chelis_surf::pipe_migration::prepare(source, previous).unwrap();
    assert_eq!(
        meaning(&migrated.source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&migrated.baseline).unwrap())
    );
    assert!(print_canonical(&migrated.baseline).contains("(pipe)"));
    for malformed in [
        previous.replace("pipe: numeric", "pipe: true"),
        previous.replace("out (pipe {}", "out (pipe true"),
    ] {
        assert!(chelis_surf::pipe_migration::prepare(source, &malformed).is_err());
    }
    // Quote's Syntax role still contains a program template to normalize.
    let quoted = chelis_surf::pipe_migration::prepare(
        "out = quote (x |> f)\n",
        "(def {} out (quote {} (pipe {} (var {} x) (var {} f))))",
    )
    .unwrap();
    assert_eq!(
        meaning(&quoted.source),
        print_canonical(&normalize_deep_for_surface_roundtrip(&quoted.baseline).unwrap())
    );
    assert!(
        chelis_surf::pipe_migration::prepare(
            "out = quote (x |> f)\n",
            "(def {} out (quote {} (pipe true (var {} x) (var {} f))))",
        )
        .is_err()
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

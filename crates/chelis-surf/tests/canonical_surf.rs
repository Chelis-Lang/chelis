//! Acceptance oracle for chelis#1032: one canonical Surf v0.19 grammar.
//!
//! Each positive/negative pair below is copied from
//! `spec/02-surf-syntax.md` section 0.2.  The canonical parser must reject
//! former aliases instead of relying on `chelis fmt` to translate them.

use chelis_deep::parser::parse_str as parse_deep;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::{format_expression, format_program, format_source, migrate_source_v018};
use chelis_surf::parser::parse_str;
use chelis_surf::resugar::{
    normalize_deep_for_surface_roundtrip, resugar_expression, resugar_program,
};

fn parses(source: &str) {
    parse_str(source)
        .unwrap_or_else(|error| panic!("canonical Surf did not parse: {error}\n{source}"));
}

fn rejects(source: &str) {
    if let Ok(ast) = parse_str(source) {
        panic!("legacy Surf alias unexpectedly parsed as {ast:#?}\n{source}");
    }
}

#[test]
fn canonical_declaration_and_expression_spellings_parse() {
    let canonical = [
        "def nullary() = value",
        "def identity(x: f32) -> f32 = x",
        "result = f(x, y)",
        "result = Some(x)",
        "result = {\n  x = f(a)\n  g(x)\n}",
        "result = par { f(x); g(y) }",
        "def unit_value() -> () = ()",
        "def log() ! { IO } = ()",
        "result = Point { x, y: other }",
        "result = Empty {}",
        "result = match value with { | Empty {} => 0 }",
        "result = vmap(f)",
        "result = vmap(f, axis=1)",
        "result = grad(f, wrt=x)",
        "result = (x,)",
        "result = 42i64",
        "result = 1.0f64",
        "result = 42i32",
        "result = 1.0f32",
        "@property grouped_operand forall(x: int32, y: int32) where (x + 1) <= y: true",
        "tiny = 5e-324",
        "huge = 1.7976931348623157e308",
    ];

    for source in canonical {
        parses(source);
    }
}

#[test]
fn extreme_float_spellings_are_shortest_and_formatter_stable() {
    let source = "tiny = 5e-324\nhuge = 1.7976931348623157e308\n";
    assert_eq!(format_source(source).expect("extreme floats parse"), source);
}

#[test]
fn non_finite_float_literals_are_rejected_with_a_targeted_diagnostic() {
    let error = parse_str("value = 1e400\n").expect_err("infinity has no Surf literal");
    assert!(
        error
            .to_string()
            .contains("non-finite numeric literal `1e400`"),
        "{error}"
    );
}

#[test]
fn every_deep_string_has_one_reparseable_canonical_surf_spelling() {
    let canonical = "value = \"\\u{8}\\u{1f}\\u{7f}\\u{85}\\0\\t\\n\\r\\\"\\\\\"\n";
    assert_eq!(
        format_source(canonical).expect("canonical controls parse"),
        canonical
    );

    let controls = "\u{8}\u{1f}\u{7f}\u{85}";
    let deep_source =
        format!("(def {{}} value (lit {{type: (t-prim {{}} string)}} \"{controls}\"))");
    let deep = parse_deep(&deep_source).expect("Deep control string parses");
    let surf = format_program(&resugar_program(&deep).expect("Deep control string resugars"));
    assert_eq!(surf, "value = \"\\u{8}\\u{1f}\\u{7f}\\u{85}\"\n");
    let redesugared =
        desugar_program(&parse_str(&surf).expect("resugared control string reparses"));
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
    );
}

#[test]
fn hand_authored_deep_without_optional_surface_origin_metadata_roundtrips() {
    for (deep_source, expected_surf) in [
        ("(import-all {} foo)", "import Foo (..)\n"),
        ("(defdim {} n)", "dim n\n"),
    ] {
        let deep = parse_deep(deep_source).expect("hand-authored Deep parses");
        let surf = format_program(&resugar_program(&deep).expect("public Deep resugars"));
        assert_eq!(surf, expected_surf);

        let redesugared = desugar_program(&parse_str(&surf).expect("resugared Surf reparses"));
        assert_eq!(
            print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
            print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
            "optional origin metadata must not break the Deep roundtrip for {deep_source}",
        );
    }
}

#[test]
fn roundtrip_normalization_retains_non_default_surface_origin_metadata() {
    let deep = parse_deep(concat!(
        "(import-all {surf_path: \"HTTP\"} http)\n",
        "(defdim {surf_dim_group_size: 2} rows)\n",
        "(defdim {} cols)\n",
    ))
    .expect("non-default surface metadata parses");

    let normalized = print_canonical(&normalize_deep_for_surface_roundtrip(&deep));

    assert!(normalized.contains("surf_path: \"HTTP\""), "{normalized}");
    assert!(
        normalized.contains("surf_dim_group_size: 2"),
        "{normalized}"
    );
}

#[test]
fn string_escape_aliases_and_raw_controls_are_rejected() {
    for source in [
        r#"value = "\u{08}""#,
        r#"value = "\u{0}""#,
        r#"value = "\u{9}""#,
        r#"value = "\u{a}""#,
        r#"value = "\u{d}""#,
        r#"value = "\u{22}""#,
        r#"value = "\u{5c}""#,
        r#"value = "\u{41}""#,
        r#"value = "\u{B}""#,
        "value = \"raw\tcontrol\"",
        "value = \"raw\u{8}control\"",
        "value = \"raw\u{7f}control\"",
    ] {
        rejects(source);
    }
}

#[test]
fn every_string_bearing_surface_uses_the_canonical_renderer() {
    let source = concat!(
        "def resource() ! { Resource(\"gpu\\\"\\\\\\t\\0\") } = ()\n",
        "@property escaped forall():\n",
        "  true\n",
        "  with contract = \"contract\\\"\\\\\\t\\0\"\n",
    );
    assert_eq!(
        format_source(source).expect("escaped strings format"),
        source
    );
}

#[test]
fn legacy_syntax_safe_aliases_are_rejected() {
    let aliases = [
        "def nullary = value",
        "def identity(x: f32): f32 = x",
        "result = f x",
        "result = Some x",
        "result = f(x)(y)",
        "result = { x = f(a); g(x) }",
        "result = { x }",
        "def unit_value() -> unit = ()",
        "def log() ! { io } = ()",
        "result = Point { x: x }",
        "result = vmap(f, 1)",
        "result = vmap(f, axis=0)",
        "result = grad(f, wrt=(x))",
        "result = 0x10",
        "result = 0b1010",
        "result = 1_000",
        "result = 1e3",
        "result = 42f32",
        "result = f(x,)",
        "result = [x,]",
        "result = Point { x, }",
        "result = (x, y,)",
        "def trailing_dim[a,](x) = x",
        "def trailing_effect() ! { IO, } = ()",
        "@property grouped forall(x: int32) where (x <= 1): true",
        "result = par { f(x); g(y); }",
        "result = do { f(x); g(y); }",
        "result = None()",
        "result = Some(x,)",
        "result = match x with { | Some(v,) => v }",
        "sig trailing_type: Option[int32,]",
        "type Trailing[a,] = | Trailing(a,)",
        "type Trailing = | Trailing(int32,)",
        "type TrailingRecord = | TrailingRecord { value: int32, }",
        "type EmptyAlias = | EmptyAlias()",
        "import Demo (value,)",
        "export (value,)",
        "result = grad(f,)",
        "result = grad(f, wrt=x,)",
        "result = grad(f, wrt=(x, y,))",
        "result = vmap(f,)",
        "result = vmap(f, axis=1,)",
        "result = cast(x, f64,)",
        "result = jit(f,)",
        "result = realize(x,)",
        "result = copy(x,)",
        "result = quote(x,)",
        "result = unquote(x,)",
        "result = splice(xs,)",
        "result = with seed(1,) { x }",
        "def resource() ! { Resource(\"gpu:0\",) } = ()",
        "typed: Option[] = None",
        "def empty_quantifiers[](x) = x",
        "type EmptyRecord = | EmptyRecord {}",
        "result = point with {}",
        "import Demo ()",
        "export ()",
        "result = match value with { | 42i64 => 0 }",
        "result = match value with { | 1.0f64 => 0 }",
        "result = match value with { | -9_223_372_036_854_775_808 => 0 }",
    ];

    for source in aliases {
        rejects(source);
    }
}

#[test]
fn direct_surface_forms_cover_deep_only_expression_tags() {
    let canonical = [
        "result = do { f(x); g(y) }",
        "result = point with { x: next_x, y }",
        "result = quote(f(x))",
        "result = unquote(ast)",
        "result = splice(items)",
    ];

    for source in canonical {
        parses(source);
    }
}

#[test]
fn record_construction_and_update_preserve_left_to_right_field_order() {
    let source =
        "a = Point { z: first(), a: second() }\nupdated = a with { y: third(), b: fourth() }";
    let decls = parse_str(source).expect("canonical record program parses");
    let deep = print_canonical(&desugar_program(&decls));

    let z = deep.find("(kv {} z").expect("z field exists");
    let a = deep.find("(kv {} a").expect("a field exists");
    assert!(z < a, "record fields must not be alphabetized:\n{deep}");

    let y = deep.find("(kv {} y").expect("y update field exists");
    let b = deep.find("(kv {} b").expect("b update field exists");
    assert!(
        y < b,
        "record-update fields must not be alphabetized:\n{deep}"
    );
}

#[test]
fn zero_field_deep_records_and_record_patterns_keep_their_distinct_surface_form() {
    let deep_source = concat!(
        "(def {} value (record {} Empty))\n",
        "(def {} matched (match {} (var {} value) ",
        "(arm {} (pat-record {} Empty) () (lit {type: (t-prim {} int32)} 1)) ",
        "(arm {} (pat-wild {}) () (lit {type: (t-prim {} int32)} 0))))\n",
    );
    let deep = parse_deep(deep_source).expect("zero-field Deep records parse");
    let surf = format_program(&resugar_program(&deep).expect("zero-field records resugar"));

    assert!(surf.contains("value = Empty {}"), "{surf}");
    assert!(surf.contains("| Empty {} => 1"), "{surf}");

    let redesugared = desugar_program(&parse_str(&surf).expect("resugared records reparse"));
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
    );
}

#[test]
fn explicitly_pure_function_round_trips_without_becoming_inferred() {
    let source = "def declared() ! {} = ()\n";
    let deep = desugar_program(&parse_str(source).expect("declared-pure Surf parses"));
    let surf = format_program(&resugar_program(&deep).expect("declared-pure Deep resugars"));

    assert_eq!(surf, source, "explicit `! {{}}` is a semantic declaration");

    let redesugared = desugar_program(&parse_str(&surf).expect("resugared Surf reparses"));
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
    );
}

#[test]
fn formatting_canonical_surf_is_idempotent() {
    let source = "def identity(x: f32) -> f32 = x\nresult = Point { x, y: other }\n";
    let once = format_source(source).expect("canonical Surf formats");
    let twice = format_source(&once).expect("formatted Surf reparses");

    assert_eq!(once, source);
    assert_eq!(twice, once);
}

#[test]
fn property_preconditions_are_formatter_fixed_points() {
    let source = "@property bounded forall(x: int32) where x <= 1:\n  true\n";
    let once = format_source(source).expect("canonical property parses");
    let twice = format_source(&once).expect("formatted property reparses");

    assert_eq!(once, source);
    assert_eq!(twice, once);
}

fn resugar_one(deep: &str) -> String {
    let mut nodes = parse_deep(deep).expect("Deep fixture parses");
    assert_eq!(nodes.len(), 1);
    format_expression(&resugar_expression(&nodes.remove(0)).expect("valid Deep resugars"))
}

#[test]
fn every_public_deep_expression_family_has_direct_canonical_surf() {
    let cases = [
        ("(fn {} (params {} x) (var {} x))", "fn (x) -> x"),
        (
            "(let {} (bind {} x (lit {} 1)) (app {} (var {} f) (var {} x)))",
            "{\n  x = 1\n  f(x)\n}",
        ),
        (
            "(if {} (var {} c) (var {} x) (var {} y))",
            "if c then x else y",
        ),
        (
            "(match {} (var {} x) (arm {} (pat-ctor {} Some (pat-var {} y)) () (var {} y)) (arm {} (pat-wild {}) () (lit {} 0)))",
            "match x with {\n  | Some(y) => y\n  | _ => 0\n}",
        ),
        (
            "(record {} Point (kv {} x (var {} x)) (kv {} y (lit {} 2)))",
            "Point { x, y: 2 }",
        ),
        ("(access {} (var {} point) x)", "point.x"),
        ("(pipe {} (var {} x) (var {} f) (var {} g))", "x |> f |> g"),
        (
            "(block {} (app {} (var {} f) (var {} x)) (app {} (var {} g) (var {} y)))",
            "do { f(x); g(y) }",
        ),
        ("(tuple {} (var {} x))", "(x,)"),
        ("(tuple-get {} (var {} pair) (lit {} 0))", "pair.0"),
        (
            "(record-update {} (var {} point) (kv {} x (var {} next_x)))",
            "point with { x: next_x }",
        ),
        ("(par {} (var {} x) (var {} y))", "par { x; y }"),
        ("(borrow {} (var {} x))", "&x"),
        ("(grad {} (var {} f))", "grad(f)"),
        ("(vmap {} (var {} f) (lit {} 0))", "vmap(f)"),
        ("(jit {} (var {} f))", "jit(f)"),
        ("(realize {} (var {} x))", "realize(x)"),
        ("(copy {} (var {} x))", "copy(x)"),
        ("(quote {} (var {} x))", "quote(x)"),
        ("(unquote {} (var {} x))", "unquote(x)"),
        ("(splice {} (var {} xs))", "splice(xs)"),
    ];

    for (deep, expected) in cases {
        assert_eq!(resugar_one(deep), expected, "Deep fixture: {deep}");
    }
}

#[test]
fn unit_and_empty_tuple_pattern_have_direct_canonical_surface_forms() {
    assert_eq!(
        resugar_one("(lit {type: (t-unit {})} ())"),
        "()",
        "unit literal metadata must not be mistaken for primitive metadata"
    );
    assert_eq!(
        resugar_one("(match {} (var {} x) (arm {} (pat-tuple {}) () (lit {} 0)))"),
        "match x with {\n  | () => 0\n}"
    );
    parses("result = match x with { | () => 0 }");
}

#[test]
fn nullary_constructors_are_bare_but_nullary_function_calls_keep_parentheses() {
    assert_eq!(resugar_one("(app {} (var {} None))"), "None");
    assert_eq!(resugar_one("(app {} (var {} f))"), "f()");

    let constructor_call = parse_deep("(app {} (var {} None))").expect("Deep app parses");
    let bare_constructor = parse_deep("(var {} None)").expect("Deep var parses");
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&constructor_call)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&bare_constructor))
    );
}

#[test]
fn default_type_suffixes_are_preserved_when_contextual_adoption_changes_meaning() {
    assert_eq!(
        resugar_one("(cast {} (lit {type: (t-prim {} f32)} 1.1) (t-prim {} f64))"),
        "cast(1.1f32, f64)"
    );
    assert_eq!(
        resugar_one("(cast {} (lit {type: (t-prim {} int32)} 42) (t-prim {} int64))"),
        "cast(42i32, int64)"
    );

    let source = concat!(
        "direct = cast(1.1, f64)\n",
        "direct_i = cast(42, int64)\n",
        "widened = cast(1.1f32, f64)\n",
        "widened_i = cast(42i32, int64)\n",
        "explicit_default_float = 1.0f32\n",
        "explicit_default_int = 42i32\n",
    );
    let parsed = parse_str(source).expect("all contextual forms parse");
    let desugared = desugar_program(&parsed);
    let deep = print_canonical(&desugared);
    assert!(deep.contains("(t-prim {} f64)"));
    assert!(deep.contains("(t-prim {} f32)"));
    assert_eq!(
        format_program(&resugar_program(&desugared).expect("contextual literals resugar")),
        source,
    );
}

#[test]
fn contextual_signed_minimum_preserves_its_adopted_int64_type() {
    let source = concat!(
        "contextual = cast(-9223372036854775808, int64)\n",
        "explicit = cast(-9223372036854775808i64, int64)\n",
    );
    let deep = desugar_program(&parse_str(source).expect("signed minima parse"));
    let surf = format_program(&resugar_program(&deep).expect("signed minima resugar"));
    assert_eq!(surf, source);
    let redesugared = desugar_program(&parse_str(&surf).expect("signed minima reparse"));
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
    );
}

#[test]
fn out_of_range_deep_integer_metadata_fails_closed_without_normalizer_panic() {
    let mut deep = parse_deep("(lit {type: (t-prim {} int32)} -9223372036854775808)")
        .expect("structurally accepted Deep literal parses");
    let normalized = normalize_deep_for_surface_roundtrip(&deep);
    assert!(print_canonical(&normalized).contains("-9223372036854775808"));
    let error = resugar_expression(&deep.remove(0))
        .expect_err("out-of-range typed integer must not emit invalid Surf");
    assert!(error.to_string().contains("compatible"), "{error}");
}

#[test]
fn multi_pair_deep_bind_resugars_as_ordered_surf_bindings() {
    assert_eq!(
        resugar_one(
            "(let {} (bind {} x (lit {} 1) y (lit {} 2)) (app {} (var {} f) (var {} x) (var {} y)))"
        ),
        "{\n  x = 1\n  y = 2\n  f(x, y)\n}"
    );
}

#[test]
fn inferred_block_literal_types_do_not_become_authored_annotations() {
    let source = "value = {\n  x = 1\n  y = 2\n  f(x, y)\n}\n";
    let deep = desugar_program(&parse_str(source).expect("untyped block parses"));
    let surf = format_program(&resugar_program(&deep).expect("untyped block resugars"));

    assert_eq!(surf, source);

    let typed = "value = {\n  x: int32 = 1\n  x\n}\n";
    let deep = desugar_program(&parse_str(typed).expect("typed block parses"));
    assert_eq!(
        format_program(&resugar_program(&deep).expect("typed block resugars")),
        typed,
    );
}

#[test]
fn empty_expression_and_type_tuples_normalize_to_unit_without_erasure() {
    let deep = parse_deep(concat!(
        "(defsig {} value (t-tuple {}))\n",
        "(def {} value (tuple {}))\n",
    ))
    .expect("empty Deep tuple forms parse");
    let surf = format_program(&resugar_program(&deep).expect("empty tuples resugar"));
    assert_eq!(surf, "value: () = ()\n");
    let redesugared = desugar_program(&parse_str(&surf).expect("unit Surf reparses"));
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared))
    );
}

#[test]
fn deep_negative_literals_normalize_to_surfs_unary_minus_shape() {
    for deep_source in [
        "(lit {type: (t-prim {} int32)} -42)",
        "(lit {type: (t-prim {} int8)} -128)",
        "(lit {type: (t-prim {} int64)} -9223372036854775808)",
        "(lit {type: (t-prim {} f32)} -1.5)",
        "(lit {type: (t-prim {} f64)} 42)",
        "(lit {type: (t-prim {} f64)} -42)",
        "(lit {type: (t-prim {} f64)} -0.0)",
        "(lit {type: (t-prim {} int64)} -9223372036854775808)",
        "(cast {} (lit {type: (t-prim {} f32)} -1.5) (t-prim {} f64))",
    ] {
        let deep = parse_deep(deep_source).expect("negative Deep literal parses");
        let surf = format_expression(
            &resugar_expression(&deep[0]).expect("negative Deep literal resugars"),
        );
        let wrapped = format!("value = {surf}\n");
        let redesugared = desugar_program(&parse_str(&wrapped).expect("resugared Surf reparses"));
        let original = parse_deep(&format!("(def {{}} value {deep_source})"))
            .expect("negative fixture wraps as a definition");
        assert_eq!(
            print_canonical(&normalize_deep_for_surface_roundtrip(&original)),
            print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
            "Deep fixture: {deep_source}; Surf: {surf}"
        );
    }
}

#[test]
fn negative_deep_literal_patterns_have_one_reparseable_surface_form() {
    for value in ["-42", "-1.5", "-0.0", "-9223372036854775808"] {
        let deep_source = format!(
            "(def {{}} value (match {{}} (var {{}} x) (arm {{}} (pat-lit {{}} {value}) () (lit {{type: (t-prim {{}} int32)}} 1)) (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} int32)}} 0))))"
        );
        let deep = parse_deep(&deep_source).expect("negative pattern fixture parses");
        let surf = format_program(&resugar_program(&deep).expect("negative pattern resugars"));
        let reparsed = parse_str(&surf)
            .unwrap_or_else(|error| panic!("negative pattern did not reparse: {error}\n{surf}"));
        let redesugared = desugar_program(&reparsed);
        assert_eq!(
            print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
            print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
            "Deep fixture: {deep_source}; Surf: {surf}"
        );
    }
}

#[test]
fn deep_application_callee_grouping_preserves_association_and_scope() {
    let cases = [
        (
            "(app {} (app {} (var {} f) (var {} x)) (var {} y))",
            "(f(x))(y)",
        ),
        (
            "(app {} (if {} (var {} c) (var {} f) (var {} g)) (var {} x))",
            "(if c then f else g)(x)",
        ),
        (
            "(app {} (fn {} (params {} x) (var {} x)) (var {} y))",
            "(fn (x) -> x)(y)",
        ),
        (
            "(app {} (app {} (var {} neg) (var {} f)) (var {} y))",
            "(-f)(y)",
        ),
    ];

    for (deep_source, expected_surf) in cases {
        let deep = parse_deep(deep_source).expect("Deep application parses");
        let surf =
            format_expression(&resugar_expression(&deep[0]).expect("Deep application resugars"));
        assert_eq!(surf, expected_surf, "Deep fixture: {deep_source}");
        let wrapped = format!("value = {surf}\n");
        let redesugared = desugar_program(&parse_str(&wrapped).expect("callee grouping reparses"));
        let original = parse_deep(&format!("(def {{}} value {deep_source})"))
            .expect("application fixture wraps as a definition");
        assert_eq!(
            print_canonical(&normalize_deep_for_surface_roundtrip(&original)),
            print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
            "Deep fixture: {deep_source}; Surf: {surf}"
        );
    }
}

#[test]
fn standalone_checked_definition_type_becomes_a_real_surf_signature() {
    let deep =
        parse_deep("(def {type: (t-prim {} int32)} value (lit {type: (t-prim {} int32)} 42))")
            .expect("checked standalone definition parses");
    let surf = format_program(&resugar_program(&deep).expect("checked definition resugars"));
    assert_eq!(surf, "value: int32 = 42\n");
    let redesugared = desugar_program(&parse_str(&surf).expect("typed binding reparses"));
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared))
    );
}

#[test]
fn operators_and_finite_lists_resugar_to_their_canonical_surface_forms() {
    let cases = [
        ("(app {} (var {} add) (var {} x) (var {} y))", "(x + y)"),
        ("(app {} (var {} neg) (var {} x))", "-x"),
        (
            "(app {} (var {} Cons) (var {} x) (app {} (var {} Cons) (var {} y) (var {} Nil)))",
            "[x, y]",
        ),
        (
            "(app {} (var {} Cons) (var {} x) (var {} tail))",
            "Cons(x, tail)",
        ),
    ];

    for (deep, expected) in cases {
        assert_eq!(resugar_one(deep), expected, "Deep fixture: {deep}");
    }
}

#[test]
fn canonical_program_round_trips_through_deep_and_the_shared_surf_ast() {
    let source = concat!(
        "module Demo.Core\n",
        "import Math\n",
        "import Data (..)\n",
        "import Util (map, fold)\n",
        "export (identity, answer)\n",
        "dim batch, seq\n",
        "type Option[a] =\n",
        "  | None\n",
        "  | Some(a)\n",
        "type Point =\n",
        "  | Point { x: f32, y: f32 }\n",
        "type Alias[a] = Option[a]\n",
        "sig standalone: f32 -> f32 ! { Diff }\n",
        "def identity(x: f32) -> f32 ! { Diff } = x\n",
        "answer: int32 = 42\n",
    );
    let surf = parse_str(source).expect("canonical program parses");
    let deep = desugar_program(&surf);
    let resugared = resugar_program(&deep).expect("all public Deep declarations resugar");
    let rendered = format_program(&resugared);

    assert_eq!(rendered, source);
    let reparsed = parse_str(&rendered).expect("resugared program reparses");
    assert_eq!(
        print_canonical(&desugar_program(&reparsed)),
        print_canonical(&deep)
    );
}

#[test]
fn opaque_invariants_properties_and_resource_effects_round_trip() {
    let source = format_source(concat!(
        "module Stats.Opaque\n",
        "@opaque\n",
        "@invariant(p) ((0.0 <= p.value) && (p.value <= 1.0))\n",
        "type Probability =\n",
        "  | Probability { value: f32 }\n",
        "def sample(p: Probability) -> Probability ! { Random, Resource(\"gpu:0\") } = with device(\"gpu:0\") { with seed(42i64) { p } }\n",
        "@property bounded forall(p: Probability) where 0.0 <= p.value:\n",
        "  (p.value <= 1.0)\n",
        "  with tolerance = 0.001\n",
        "  with seed = 42i64\n",
        "  with samples = 3\n",
        "  with contract = \"stats.probability.bounded\"\n",
    ))
    .expect("fixture canonicalizes");
    let surf = parse_str(&source).expect("canonical opaque/property program parses");
    let deep = desugar_program(&surf);
    let rendered = format_program(&resugar_program(&deep).expect("program resugars"));

    assert_eq!(rendered, source);
    let reparsed = parse_str(&rendered).expect("resugared program reparses");
    assert_eq!(
        print_canonical(&desugar_program(&reparsed)),
        print_canonical(&deep)
    );
}

#[test]
fn explicit_v018_migration_rewrites_aliases_and_preserves_comments() {
    let legacy = concat!(
        "-- source header\n",
        "def nullary = value\n",
        "def identity(x: f32): f32 ! { diff } = { f x; }\n",
        "result = f(x)(y)\n",
        "point = Point { x: x }\n",
        "mapped = vmap(f, 1)(xs)\n",
        "mapped_zero = vmap(f, axis=0)(xs)\n",
        "def unit_value(): unit = ()\n",
        "legacy_number = 0x10\n",
        "legacy_float = 42f32\n",
        "nullary_constructor = None()\n",
        "@property grouped forall(x: int32) where (x <= 1): true\n",
        "-- source footer\n",
    );
    let expected = concat!(
        "-- source header\n",
        "def nullary() = value\n",
        "def identity(x: f32) -> f32 ! { Diff } = f(x)\n",
        "result = f(x, y)\n",
        "point = Point { x }\n",
        "mapped = vmap(f, axis=1)(xs)\n",
        "mapped_zero = vmap(f)(xs)\n",
        "def unit_value() -> () = ()\n",
        "legacy_number = 16\n",
        "legacy_float = 42.0f32\n",
        "nullary_constructor = None\n",
        "@property grouped forall(x: int32) where x <= 1:\n",
        "  true\n",
        "-- source footer\n",
    );

    let migrated = migrate_source_v018(legacy).expect("legacy source migrates");

    assert_eq!(migrated, expected);
    parse_str(&migrated).expect("migration emits only canonical Surf");
    assert_eq!(format_source(&migrated).unwrap(), migrated);
}

#[test]
fn explicit_v018_migration_removes_trailing_separators_from_special_forms() {
    let legacy = concat!(
        "g0 = grad(f,)\n",
        "g1 = grad(f, wrt=(x, y,),)\n",
        "v0 = vmap(f,)\n",
        "v1 = vmap(f, axis=1,)\n",
        "c = cast(x, f64,)\n",
        "j = jit(f,)\n",
        "r = realize(x,)\n",
        "copied = copy(x,)\n",
        "q = quote(x,)\n",
        "u = unquote(x,)\n",
        "s = splice(xs,)\n",
        "seeded = with seed(1,) { x }\n",
        "def effectful() ! { Resource(\"gpu:0\",), } = ()\n",
    );
    let expected = concat!(
        "g0 = grad(f)\n",
        "g1 = grad(f, wrt=(x, y))\n",
        "v0 = vmap(f)\n",
        "v1 = vmap(f, axis=1)\n",
        "c = cast(x, f64)\n",
        "j = jit(f)\n",
        "r = realize(x)\n",
        "copied = copy(x)\n",
        "q = quote(x)\n",
        "u = unquote(x)\n",
        "s = splice(xs)\n",
        "seeded = with seed(1) { x }\n",
        "def effectful() ! { Resource(\"gpu:0\") } = ()\n",
    );

    assert_eq!(migrate_source_v018(legacy).unwrap(), expected);
}

#[test]
fn explicit_v018_migration_removes_decorative_empty_and_pattern_aliases() {
    let legacy = concat!(
        "typed: Option[] = None\n",
        "def identity[](x) = x\n",
        "type Empty[] = | Empty {}\n",
        "constructed = Empty {}\n",
        "matched = match value with { | Empty {} => 0 | 42i64 => 1 | -1.5f64 => 2 | -9223372036854775808i64 => 3 }\n",
        "unchanged = point with {}\n",
        "import Demo ()\n",
    );
    let expected = concat!(
        "typed: Option = None\n",
        "def identity(x) = x\n",
        "type Empty =\n",
        "  | Empty\n",
        "constructed = Empty {}\n",
        "matched = match value with {\n",
        "  | Empty {} => 0\n",
        "  | 42 => 1\n",
        "  | -1.5 => 2\n",
        "  | -9223372036854775808 => 3\n",
        "}\n",
        "unchanged = point\n",
        "import Demo\n",
    );

    let migrated = migrate_source_v018(legacy).expect("decorative aliases migrate");
    assert_eq!(migrated, expected);
    parse_str(&migrated).expect("migration emits canonical Surf");
    assert_eq!(format_source(&migrated).unwrap(), migrated);
}

#[test]
fn migration_rejects_ambiguous_interior_comment_attachment() {
    let legacy = "def f(x) = g({- belongs to the argument expression -} x)\n";

    let error = migrate_source_v018(legacy).expect_err("ambiguous attachment must not migrate");

    assert!(error.to_string().contains("inside a declaration"));
}

#[test]
fn migration_rejects_comments_inside_block_binding_expressions() {
    let legacy = concat!(
        "def f(x) = {\n",
        "  y = g({- belongs to the argument expression -} x)\n",
        "  y\n",
        "}\n",
    );

    let error = migrate_source_v018(legacy)
        .expect_err("an expression-interior comment must not be relocated to a block boundary");

    assert!(error.to_string().contains("inside a declaration"));
}

#[test]
fn migration_preserves_comments_attached_to_block_bindings() {
    let legacy = concat!(
        "def f(x) = {\n",
        "  -- compute the intermediate\n",
        "  y = g x\n",
        "  -- return it\n",
        "  y\n",
        "}\n",
    );
    let expected = concat!(
        "def f(x) = {\n",
        "  -- compute the intermediate\n",
        "  y = g(x)\n",
        "  -- return it\n",
        "  y\n",
        "}\n",
    );

    assert_eq!(migrate_source_v018(legacy).unwrap(), expected);
}

#[test]
fn migration_of_legacy_single_expression_function_blocks_is_a_fixed_point() {
    let legacy = concat!(
        "def choose(x: Option[int32]): int32 = {\n",
        "  match x with {\n",
        "    | Some(v) => v\n",
        "    | None => 0\n",
        "  }\n",
        "}\n",
    );
    let migrated = migrate_source_v018(legacy).expect("legacy wrapper migrates");

    assert_eq!(format_source(&migrated).unwrap(), migrated);
    assert_eq!(
        migrated,
        concat!(
            "def choose(x: Option[int32]) -> int32 =\n",
            "  match x with {\n",
            "    | Some(v) => v\n",
            "    | None => 0\n",
            "  }\n",
        )
    );
}

#[test]
fn pipe_call_stage_metadata_preserves_the_one_canonical_sugar() {
    let source = concat!(
        "ordinary = x |> f(y)\n",
        "casted = x |> cast(f64)\n",
        "later = x |> fn (v) -> f(y, v)\n",
    );
    let surf = parse_str(source).expect("canonical pipe stages parse");
    let deep = desugar_program(&surf);
    let printed = print_canonical(&deep);

    assert_eq!(
        printed.matches("surf_pipe_stage: \"call-first\"").count(),
        2
    );
    let resugared = format_program(&resugar_program(&deep).expect("pipe stages resugar"));
    assert_eq!(resugared, source);
    assert_eq!(
        print_canonical(&chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
            &deep
        )),
        print_canonical(&chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
            &desugar_program(&parse_str(&resugared).unwrap())
        ))
    );
}

#[test]
fn explicit_first_argument_pipe_lambda_alias_is_rejected() {
    rejects("result = x |> fn (v) -> f(v, y)");
    parses("result = x |> fn (v) -> f(y, v)");
}

#[test]
fn deep_surf_metadata_namespace_and_marker_values_are_closed() {
    let unknown = parse_deep("(var {surf_future: true} x)")
        .expect_err("unknown surf metadata key must be rejected at parse time");
    assert!(
        unknown
            .to_string()
            .contains("closed Surf metadata namespace")
    );

    let mut malformed = parse_deep(
        "(pipe {} (var {} x) (fn {surf_pipe_stage: \"later\"} (params {} v) (var {} v)))",
    )
    .expect("known key parses for value validation");
    let error = resugar_expression(&malformed.remove(0))
        .expect_err("unknown call-stage marker value must fail closed");
    assert!(error.to_string().contains("call-first"));

    let mut malformed =
        parse_deep("(lit {type: (t-prim {} f32), surf_literal_style: \"future\"} 1.0)")
            .expect("known literal-style key parses for value validation");
    let error = resugar_expression(&malformed.remove(0))
        .expect_err("unknown literal-style marker value must fail closed");
    assert!(error.to_string().contains("unsuffixed"));

    let malformed = parse_deep(concat!(
        "(let {} (bind {} x ",
        "(lit {type: (t-prim {} int32), surf_binding_type: \"future\"} 1)) ",
        "(var {} x))",
    ))
    .expect("known binding-style key parses for value validation");
    let error = resugar_expression(&malformed[0])
        .expect_err("unknown binding-style marker value must fail closed");
    assert!(error.to_string().contains("inferred"));

    for deep_source in [
        "(var {surf_literal_style: \"future\"} x)",
        "(var {surf_binding_type: \"future\"} x)",
        "(var {surf_binding_type: \"inferred\"} x)",
        "(var {surf_pipe_stage: \"call-first\"} x)",
        "(var {surf_path: \"Demo\"} x)",
        "(var {surf_dim_group_size: 1} x)",
    ] {
        let mut malformed = parse_deep(deep_source).expect("known metadata key parses");
        let rendered = print_canonical(&malformed);
        let Err(error) = resugar_expression(&malformed.remove(0)) else {
            panic!(
                "known Surf metadata unexpectedly resugared outside its declared placement: {deep_source}: {rendered}"
            );
        };
        assert!(
            error.to_string().contains("metadata"),
            "{deep_source}: {error}"
        );
    }

    let mut legacy_wrapper = parse_deep("^{:surf_literal_style \"explicit\"} (var {} x)")
        .expect("legacy metadata expression parses");
    let error = resugar_expression(&legacy_wrapper.remove(0))
        .expect_err("surface metadata on a legacy metadata wrapper must fail closed");
    assert!(error.to_string().contains("literal"));

    let malformed =
        parse_deep("(var {surf_binding_type: \"future\"} x)").expect("known metadata key parses");
    assert!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&malformed))
            .contains("surf_binding_type"),
        "normalization must not erase malformed Surf metadata before validation"
    );
}

#[test]
fn deep_names_that_cannot_be_surf_tokens_fail_closed() {
    for deep_source in [
        "(var {} foo-bar)",
        "(var {} foo.bar)",
        "(app {} (var {} foo-bar) (lit {} 1))",
        "(record {} Point (kv {} X (lit {} 1)))",
        "(match {} (var {} point) (arm {} (pat-record {} Point (kv {} X (pat-var {} x))) () (var {} x)))",
        "(def {} foo-bar (lit {} 1))",
    ] {
        let deep = parse_deep(deep_source).expect("structurally accepted Deep name parses");
        let result = if deep_source.starts_with("(def ") {
            resugar_program(&deep).map(|_| ())
        } else {
            resugar_expression(&deep[0]).map(|_| ())
        };
        let error = result.expect_err("invalid Deep name must not emit meaning-changing Surf");
        assert!(
            error.to_string().contains("identifier"),
            "{deep_source}: {error}"
        );
    }

    for deep_source in [
        "(var {} foo_bar)",
        "(var {} Some)",
        "(record {} Point (kv {} x (lit {} 1)))",
    ] {
        let deep = parse_deep(deep_source).expect("valid Deep name parses");
        resugar_expression(&deep[0])
            .unwrap_or_else(|error| panic!("valid Deep name failed: {deep_source}: {error}"));
    }
}

#[test]
fn roundtrip_normalization_removes_only_parameter_types_redundant_with_defsig() {
    let redundant = parse_deep(concat!(
        "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n",
        "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))\n",
    ))
    .expect("well-formed paired definition");
    let without_redundancy = parse_deep(concat!(
        "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n",
        "(def {} f (fn {} (params {} x) (var {} x)))\n",
    ))
    .expect("well-formed paired definition without redundant parameter metadata");

    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&redundant)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&without_redundancy))
    );

    let mismatched = parse_deep(concat!(
        "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n",
        "(def {} f (fn {} (params {} (x {type: (t-prim {} f64)})) (var {} x)))\n",
    ))
    .expect("structurally valid but mismatched parameter metadata");
    assert!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&mismatched))
            .contains("{type: (t-prim {} f64)}"),
        "normalization must not hide a semantic disagreement"
    );
}

#[test]
fn roundtrip_normalization_strips_only_enumerated_derived_metadata() {
    let deep = parse_deep(concat!(
        "(var {span: \"external\", loc: \"source\", ",
        "source: (app {} (var {} macro_call)), ",
        "effects: (effects {} IO), invariant_amenability: \"linear\", ",
        "eff: (effects {} IO), type: (t-prim {} f32), ",
        "surf_path: \"Module.Path\", surf_literal_style: \"explicit\", ",
        "surf_binding_type: \"inferred\"} x)\n",
    ))
    .expect("metadata-rich Deep parses");
    let normalized = print_canonical(&normalize_deep_for_surface_roundtrip(&deep));

    for derived in [
        "span:",
        "loc:",
        "source:",
        "effects:",
        "invariant_amenability:",
    ] {
        assert!(
            !normalized.contains(derived),
            "derived key survived: {normalized}"
        );
    }
    for semantic in [
        "eff:",
        "type:",
        "surf_path:",
        "surf_literal_style:",
        "surf_binding_type:",
    ] {
        assert!(
            normalized.contains(semantic),
            "semantic or misplaced surface key was erased: {normalized}"
        );
    }

    let legal_markers = parse_deep(concat!(
        "(bind {} x ",
        "(lit {surf_literal_style: \"explicit\", surf_binding_type: \"inferred\"} 1))\n",
    ))
    .expect("legal origin markers parse");
    let normalized_legal = print_canonical(&normalize_deep_for_surface_roundtrip(&legal_markers));
    assert!(!normalized_legal.contains("surf_literal_style:"));
    assert!(!normalized_legal.contains("surf_binding_type:"));
}

#[test]
fn deep_resugaring_recovers_dimension_and_precision_quantifiers() {
    let source = concat!(
        "def sort_values[n, p](values: &tensor[n, p], axis: int32)",
        " -> (tensor[n, p], tensor[n, int64]) = sort(values, axis)\n",
    );
    let deep = desugar_program(&parse_str(source).expect("quantified definition parses"));
    let resugared = format_program(&resugar_program(&deep).expect("quantifiers resugar"));
    let redesugared = desugar_program(&parse_str(&resugared).expect("resugared source parses"));

    assert_eq!(resugared, source);
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared))
    );
}

#[test]
fn tuple_destructuring_resugars_without_losing_linearity_markers() {
    let source = concat!(
        "def sum_pair(pair) = {\n",
        "  (x, y) = pair\n",
        "  (x + y)\n",
        "}\n",
    );
    let surf = parse_str(source).expect("tuple destructuring parses");
    let deep = desugar_program(&surf);
    let resugared = format_program(&resugar_program(&deep).expect("destructuring resugars"));
    let redesugared = desugar_program(&parse_str(&resugared).expect("resugared source parses"));

    assert_eq!(resugared, source);
    assert_eq!(
        print_canonical(&chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
            &deep
        )),
        print_canonical(&chelis_surf::resugar::normalize_deep_for_surface_roundtrip(
            &redesugared
        ))
    );
}

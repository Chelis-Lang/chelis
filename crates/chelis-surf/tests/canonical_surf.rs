//! Acceptance oracle for chelis#1032: one canonical Surf v0.19 output form.
//!
//! The normal parser accepts the harmless lexical and trailing-separator
//! aliases selected by `spec/02-surf-syntax.md`; the shared printer removes
//! those cosmetic distinctions. Semantic, ambiguous, and genuinely legacy
//! forms remain rejected outside the explicit v0.18 migration path.

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

fn assert_alias_formats_to(source: &str, expected: &str) {
    let authored = parse_str(source)
        .unwrap_or_else(|error| panic!("accepted Surf alias did not parse: {error}\n{source}"));
    let formatted = format_source(source)
        .unwrap_or_else(|error| panic!("accepted Surf alias did not format: {error}\n{source}"));
    assert_eq!(
        formatted, expected,
        "unexpected canonical output for {source:?}"
    );
    assert_eq!(
        format_source(&formatted).expect("canonical output reparses"),
        formatted,
        "canonical output must be a formatter fixed point",
    );

    let canonical = parse_str(&formatted).expect("canonical output parses");
    let authored_deep = desugar_program(&authored);
    let canonical_deep = desugar_program(&canonical);
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&authored_deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&canonical_deep)),
        "canonical formatting changed the Deep meaning of {source:?}",
    );
}

#[test]
fn canonical_declaration_and_expression_spellings_parse() {
    let canonical = [
        "def nullary() = value",
        "def identity(x: f32) -> f32 = x",
        "result = f(x, y)",
        "result = Some(x)",
        "result = None()",
        "result = None",
        "result = {\n  x = f(a)\n  g(x)\n}",
        "result = par { f(x); g(y) }",
        "def unit_value() -> unit = ()",
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
fn low_precedence_expressions_must_be_grouped_as_operator_operands() {
    rejects("result = x > fn (v) -> f(v)\n");
    parses("result = x > (fn (v) -> f(v))\n");
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
fn mismatched_surface_path_fails_closed() {
    let deep = parse_deep("(import-all {surf_path: \"Other\"} foo)")
        .expect("known surface metadata parses for contract validation");

    let error = resugar_program(&deep)
        .expect_err("surface path metadata must agree with the lowered Deep path");

    assert!(error.to_string().contains("lowered path child"), "{error}");
}

#[test]
fn normalization_retains_malformed_surface_path_marker() {
    let deep = parse_deep("(import-all {surf_path: \"FOO\"} FOO)")
        .expect("known surface metadata parses for normalization");

    let normalized = print_canonical(&normalize_deep_for_surface_roundtrip(&deep));

    assert!(
        normalized.contains("surf_path: \"FOO\""),
        "normalization must retain a marker that disagrees with a non-lowered path child: {normalized}",
    );
}

#[test]
fn dimension_group_marker_on_non_first_member_fails_closed() {
    let deep = parse_deep(concat!(
        "(defdim {surf_dim_group_size: 2} rows)\n",
        "(defdim {surf_dim_group_size: 1} cols)\n",
    ))
    .expect("known surface metadata parses for sequence validation");

    let error = resugar_program(&deep)
        .expect_err("only the first member of a dimension group may carry the marker");

    assert!(error.to_string().contains("first `defdim`"), "{error}");
}

#[test]
fn normalization_retains_misplaced_default_dimension_marker() {
    let deep = parse_deep(concat!(
        "(defdim {surf_dim_group_size: 2} rows)\n",
        "(defdim {surf_dim_group_size: 1} cols)\n",
    ))
    .expect("known surface metadata parses for normalization");

    let normalized = print_canonical(&normalize_deep_for_surface_roundtrip(&deep));

    assert!(
        normalized.contains("(defdim {surf_dim_group_size: 1} cols)"),
        "normalization must retain a default marker when it is misplaced: {normalized}",
    );
}

#[test]
fn valid_string_escape_aliases_format_to_one_spelling() {
    for (source, expected) in [
        (r#"value = "\u{08}""#, "value = \"\\u{8}\"\n"),
        (r#"value = "\u{0}""#, "value = \"\\0\"\n"),
        (r#"value = "\u{9}""#, "value = \"\\t\"\n"),
        (r#"value = "\u{a}""#, "value = \"\\n\"\n"),
        (r#"value = "\u{d}""#, "value = \"\\r\"\n"),
        (r#"value = "\u{22}""#, "value = \"\\\"\"\n"),
        (r#"value = "\u{5c}""#, "value = \"\\\\\"\n"),
        (r#"value = "\u{41}""#, "value = \"A\"\n"),
        (r#"value = "\u{B}""#, "value = \"\\u{b}\"\n"),
    ] {
        assert_alias_formats_to(source, expected);
    }
}

#[test]
fn raw_controls_and_invalid_escapes_are_rejected() {
    for source in [
        "value = \"raw\tcontrol\"",
        "value = \"raw\u{8}control\"",
        "value = \"raw\u{7f}control\"",
        r#"value = "\q""#,
        r#"value = "\u{}""#,
        r#"value = "\u{110000}""#,
    ] {
        rejects(source);
    }
}

#[test]
fn value_preserving_literal_aliases_format_canonically() {
    for (source, expected) in [
        ("result = 0x10", "result = 16\n"),
        ("result = 0b1010", "result = 10\n"),
        ("result = 1_000", "result = 1000\n"),
        ("result = 0x10i64", "result = 16i64\n"),
        ("result = 1e3", "result = 1000.0\n"),
        ("result = 1.0E+3", "result = 1000.0\n"),
        ("result = 1_0.5_0e+1", "result = 105.0\n"),
        (
            "result = match value with { | -9_223_372_036_854_775_808 => 0 }",
            "result = match value with {\n  | -9223372036854775808 => 0\n}\n",
        ),
    ] {
        assert_alias_formats_to(source, expected);
    }
}

#[test]
fn whitespace_before_parenthesized_calls_is_a_cosmetic_input_alias() {
    for (source, expected) in [
        ("result = f (x)", "result = f(x)\n"),
        ("result = Some (x)", "result = Some(x)\n"),
        ("result = f (x, y)", "result = f(x, y)\n"),
    ] {
        assert_alias_formats_to(source, expected);
    }
}

#[test]
fn trailing_separators_parse_and_format_to_a_fixed_point() {
    for source in [
        "result = f(x,)",
        "result = [x,]",
        "result = Point { x, }",
        "result = (x, y,)",
        "def trailing_dim[a,](x,) = x",
        "def trailing_effect() ! { IO, } = ()",
        "result = par { f(x); g(y); }",
        "result = do { f(x); g(y); }",
        "result = Some(x,)",
        "result = match x with { | Some(v,) => v }",
        "sig trailing_type: Option[int32,]",
        "type Trailing[a,] = | Trailing(a,)",
        "type Trailing = | Trailing(int32,)",
        "type TrailingRecord = | TrailingRecord { value: int32, }",
        "import Demo (value,)",
        "export (value,)",
        "result = grad(f,)",
        "result = grad(f, wrt=x,)",
        "result = grad(f, wrt=(x, y,))",
        "result = vmap(f,)",
        "result = vmap(f, axis=1,)",
        "result = cast(x, f64,)",
        "result = cast_trunc(x, int32,)",
        "result = jit(f,)",
        "result = realize(x,)",
        "result = copy(x,)",
        "result = quote(x,)",
        "result = unquote(x,)",
        "result = splice(xs,)",
        "result = with seed(1,) { x }",
        "def resource() ! { Resource(\"gpu:0\",) } = ()",
    ] {
        let formatted = format_source(source)
            .unwrap_or_else(|error| panic!("trailing separator did not parse: {error}\n{source}"));
        assert_ne!(
            formatted.trim_end(),
            source,
            "printer retained alias: {source}"
        );
        assert_eq!(
            format_source(&formatted).expect("formatted output reparses"),
            formatted,
            "trailing-separator output is not a fixed point for {source}",
        );
        assert!(
            !formatted.contains(",)")
                && !formatted.contains(",]")
                && !formatted.contains(", }")
                && !formatted.contains("; }"),
            "printer retained a trailing separator: {formatted}",
        );

        let authored_deep = desugar_program(&parse_str(source).expect("alias parses"));
        let formatted_deep =
            desugar_program(&parse_str(&formatted).expect("formatted output parses"));
        assert_eq!(
            print_canonical(&normalize_deep_for_surface_roundtrip(&authored_deep)),
            print_canonical(&normalize_deep_for_surface_roundtrip(&formatted_deep)),
            "separator removal changed Deep meaning for {source}",
        );
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
fn semantic_ambiguous_and_non_reviewed_legacy_aliases_are_rejected() {
    let aliases = [
        "def nullary = value",
        "def identity(x: f32): f32 = x",
        "result = f x",
        "result = Some x",
        "result = f(x)(y)",
        "result = { x = f(a); g(x) }",
        "result = { x }",
        "def unit_value() -> () = ()",
        "def log() ! { io } = ()",
        "result = Point { x: x }",
        "result = vmap(f, 1)",
        "result = vmap(f, axis=0)",
        "result = grad(f, wrt=(x))",
        "result = 42f32",
        "@property grouped forall(x: int32) where (x <= 1): true",
        "type EmptyAlias = | EmptyAlias()",
        "typed: Option[] = None",
        "def empty_quantifiers[](x) = x",
        "type EmptyRecord = | EmptyRecord {}",
        "result = point with {}",
        "import Demo ()",
        "export ()",
        "result = match value with { | 42i64 => 0 }",
        "result = match value with { | 1.0f64 => 0 }",
        "result = 1__0",
        "result = 1_",
        "result = 0x_10",
        "result = 0x10_",
        "result = 0b_10",
        "result = 1e3_",
    ];

    for source in aliases {
        rejects(source);
    }
}

#[test]
fn signed_minimum_magnitude_without_unary_minus_has_a_source_diagnostic() {
    let error = parse_str("value = 9223372036854775808i64")
        .expect_err("the positive magnitude above int64::MAX must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("only valid after unary `-`") && !message.contains("IntMinMagnitude"),
        "unexpected signed-minimum magnitude diagnostic: {message}"
    );
}

#[test]
fn direct_surface_forms_cover_deep_only_expression_tags() {
    let canonical = [
        "result = do { f(x); g(y) }",
        "result = point with { x: next_x, y }",
        "result = quote(f(x))",
        "result = unquote(ast)",
        "result = splice(items)",
        "where = 1",
    ];

    for source in canonical {
        parses(source);
    }
}

#[test]
fn future_keywords_remain_reserved_while_where_is_contextual() {
    parses("where = 1");
    for source in [
        "effect = 1",
        "handler = 1",
        "perform = 1",
        "resume = 1",
        "borrow = 1",
    ] {
        rejects(source);
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
fn constructor_references_calls_and_records_remain_structurally_distinct() {
    assert_eq!(resugar_one("(var {} None)"), "None");
    assert_eq!(resugar_one("(app {} (var {} None))"), "None()");
    assert_eq!(resugar_one("(record {} None)"), "None {}");
    assert_eq!(resugar_one("(app {} (var {} f))"), "f()");

    let constructor_call = parse_deep("(app {} (var {} None))").expect("Deep app parses");
    let bare_constructor = parse_deep("(var {} None)").expect("Deep var parses");
    assert_ne!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&constructor_call)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&bare_constructor))
    );

    for source in ["value = None\n", "value = None()\n", "value = None {}\n"] {
        assert_eq!(
            format_source(source).expect("constructor form parses"),
            source
        );
    }
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
    assert!(
        error
            .to_string()
            .contains("canonical atom/primitive pairing"),
        "{error}"
    );
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
fn authored_expression_annotation_on_a_binding_survives_structural_resugaring() {
    let source = "value = {\n  shaped = (reshape(x) : Result)\n  shaped\n}\n";
    let deep = desugar_program(&parse_str(source).expect("annotated binding parses"));
    let surf = format_program(&resugar_program(&deep).expect("annotated binding resugars"));

    assert_eq!(
        surf,
        "value = {\n  shaped: Result = reshape(x)\n  shaped\n}\n"
    );
    let redesugared = desugar_program(&parse_str(&surf).expect("resugared binding reparses"));
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        print_canonical(&normalize_deep_for_surface_roundtrip(&redesugared)),
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
    assert_eq!(surf, "value: unit = ()\n");
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
        "(lit {type: (t-prim {} f64), literal_source: integer} 42)",
        "(lit {type: (t-prim {} f64), literal_source: integer} -42)",
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
fn property_options_canonicalize_non_contract_options_before_ordered_contracts() {
    let authored = concat!(
        "@property ordered forall(x: f32):\n",
        "  x == x\n",
        "  with contract = \"first\"\n",
        "  with tolerance = 0.001\n",
        "  with contract = \"second\"\n",
        "  with seed = 42i64\n",
    );
    let canonical = concat!(
        "@property ordered forall(x: f32):\n",
        "  (x == x)\n",
        "  with tolerance = 0.001\n",
        "  with seed = 42i64\n",
        "  with contract = \"first\"\n",
        "  with contract = \"second\"\n",
    );

    assert_eq!(
        format_source(authored).expect("property formats"),
        canonical
    );
    let deep = desugar_program(&parse_str(canonical).expect("canonical property parses"));
    let resugared = format_program(&resugar_program(&deep).expect("property resugars"));
    assert_eq!(resugared, canonical);
}

#[test]
fn property_resugaring_rejects_provenance_that_surf_cannot_represent() {
    for metadata in [
        concat!(
            "property_source_kind: \"bridge:c-earchin\", ",
            "property_source_id: \"REQ-1\", "
        ),
        concat!(
            "property_source_kind: \"user\", ",
            "property_source_id: \"producer-local\", "
        ),
    ] {
        let source = format!(
            "(def {{chelis_role: \"property\", {metadata}\
             property_quantifiers: (params {{}} (x {{type: (t-prim {{}} f32)}})), \
             property_preconditions: (tuple {{}})}} \
             p (fn {{}} (params {{}} (x {{type: (t-prim {{}} f32)}})) \
             (lit {{type: (t-prim {{}} bool)}} true)))"
        );
        let deep = parse_deep(&source).expect("property Deep parses");
        let error = resugar_program(&deep).expect_err(
            "resugaring must not silently rewrite property provenance as user-authored",
        );
        assert!(
            error.to_string().contains("property_source"),
            "unexpected provenance error: {error}"
        );
    }
}

#[test]
fn property_resugaring_rejects_quantifiers_that_disagree_with_fn_parameters() {
    let deep = parse_deep(concat!(
        "(def {chelis_role: \"property\", property_source_kind: \"user\", ",
        "property_quantifiers: (params {} (y {type: (t-prim {} f32)})), ",
        "property_preconditions: (tuple {})} ",
        "p (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
    ))
    .expect("mismatched property Deep parses structurally");
    let error = resugar_program(&deep)
        .expect_err("property quantifiers must match the callable binder list exactly");
    assert!(
        error.to_string().contains("property_quantifiers") && error.to_string().contains("match"),
        "unexpected quantifier mismatch error: {error}"
    );
}

#[test]
fn property_resugaring_rejects_missing_required_precondition_metadata() {
    let deep = parse_deep(concat!(
        "(def {chelis_role: \"property\", property_source_kind: \"user\", ",
        "property_quantifiers: (params {} (x {type: (t-prim {} f32)}))} ",
        "p (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
    ))
    .expect("property without precondition metadata parses structurally");
    let error = resugar_program(&deep)
        .expect_err("required property metadata must not silently default to an empty tuple");
    assert!(
        error.to_string().contains("property_preconditions"),
        "unexpected missing-precondition error: {error}"
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
        "def unit_value() -> unit = ()\n",
        "legacy_number = 16\n",
        "legacy_float = 42.0f32\n",
        "nullary_constructor = None()\n",
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
fn operator_named_and_finite_list_pipe_stages_keep_their_call_stage_sugar() {
    // chelis#1197: the operator and finite-list sugars rewrite an
    // application into `Binary`, `Unary`, or `List`, none of which can
    // carry the `|> f(args)` stage form, so a stage calling `mul`, `cmplt`,
    // or `Cons` used to fail closed on Deep the desugarer itself emits.
    let source = concat!(
        "scaled = x |> mul(y)\n",
        "compared = x |> cmplt(y)\n",
        "listed = x |> Cons(Nil)\n",
        "chained = x |> fn (p) -> cast(p, f32) |> mul(cast(2.0, f32))\n",
    );
    let surf = parse_str(source).expect("operator-named pipe stages parse");
    let deep = desugar_program(&surf);
    let resugared = resugar_program(&deep).expect("operator-named pipe stages resugar");
    let rendered = format_program(&resugared);

    assert_eq!(rendered, source);
    assert_eq!(
        print_canonical(&normalize_deep_for_surface_roundtrip(&desugar_program(
            &parse_str(&rendered).expect("resugared pipe stages reparse")
        ))),
        print_canonical(&normalize_deep_for_surface_roundtrip(&deep)),
        "an operator-named pipe stage must survive Surf -> Deep -> Surf -> Deep",
    );
}

#[test]
fn operator_sugar_inside_a_call_stage_argument_is_still_applied() {
    // Only the stage application itself is held back from the operator
    // sugar. An operand of that application keeps its canonical operator
    // spelling.
    assert_eq!(
        resugar_one(concat!(
            "(pipe {} (var {} x) (fn {surf_pipe_stage: \"call-first\"} (params {} p) ",
            "(app {} (var {} mul) (var {} p) (app {} (var {} add) (var {} a) (var {} b)))))",
        )),
        "x |> mul((a + b))"
    );
}

#[test]
fn call_first_pipe_stage_bodies_that_cannot_carry_the_sugar_fail_closed() {
    for deep_source in [
        // The stage parameter is not the first argument, so the stage is not
        // first-argument insertion and has no call-stage spelling.
        concat!(
            "(pipe {} (var {} x) (fn {surf_pipe_stage: \"call-first\"} (params {} p) ",
            "(app {} (var {} mul) (var {} y) (var {} p))))",
        ),
        // The body is not an application at all.
        "(pipe {} (var {} x) (fn {surf_pipe_stage: \"call-first\"} (params {} p) (var {} p)))",
        // A call-first stage carries exactly one parameter.
        concat!(
            "(pipe {} (var {} x) (fn {surf_pipe_stage: \"call-first\"} (params {} p q) ",
            "(app {} (var {} mul) (var {} p) (var {} q))))",
        ),
        // The stage parameter occurs again in an operand. The sugar drops the
        // binder with the leading occurrence, so emitting `x |> add(p)` here
        // would leave the second `p` free for an enclosing binding to capture.
        concat!(
            "(pipe {} (var {} x) (fn {surf_pipe_stage: \"call-first\"} (params {} p) ",
            "(app {} (var {} add) (var {} p) (var {} p))))",
        ),
        // The same capture nested inside an operand.
        //
        // Both capture cases name an operator callee because that is the path
        // this test owns. An ordinary callee reaches the older strip arm, which
        // still frees a repeated parameter; that residue is chelis#1246.
        concat!(
            "(pipe {} (var {} x) (fn {surf_pipe_stage: \"call-first\"} (params {} p) ",
            "(app {} (var {} add) (var {} p) (app {} (var {} neg) (var {} p)))))",
        ),
    ] {
        let mut malformed = parse_deep(deep_source).expect("Deep fixture parses");
        let rendered = print_canonical(&malformed);
        let Err(error) = resugar_expression(&malformed.remove(0)) else {
            panic!(
                "a stage body that cannot carry the call-stage sugar must fail closed: {rendered}"
            );
        };
        assert!(
            error.to_string().contains("call-first stage"),
            "unexpected diagnostic for {deep_source}: {error}",
        );
    }
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
fn resugaring_rejects_parameter_and_function_types_that_conflict_with_defsig() {
    for source in [
        concat!(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n",
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f64)})) (var {} x)))\n",
        ),
        concat!(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n",
            "(def {} f (fn {type: (t-fn {} (t-prim {} f64) (t-prim {} f32))} ",
            "(params {} (x {type: (t-prim {} f32)})) (var {} x)))\n",
        ),
    ] {
        let deep = parse_deep(source).expect("conflicting checked Deep parses structurally");
        let error = resugar_program(&deep)
            .expect_err("conflicting checked types must fail instead of being overwritten");
        assert!(
            error.to_string().contains("matching") || error.to_string().contains("conflict"),
            "unexpected checked-type mismatch error: {error}"
        );
    }
}

#[test]
fn property_resugaring_rejects_signatures_that_conflict_with_property_types() {
    for signature_type in [
        "(t-fn {} (t-prim {} f64) (t-prim {} bool))",
        "(t-fn {} (t-prim {} f32) (t-prim {} f32))",
    ] {
        let source = format!(
            "(defsig {{}} p {signature_type})\n\
             (def {{chelis_role: \"property\", property_source_kind: \"user\", \
             property_quantifiers: (params {{}} (x {{type: (t-prim {{}} f32)}})), \
             property_preconditions: (tuple {{}})}} p \
             (fn {{}} (params {{}} (x {{type: (t-prim {{}} f32)}})) \
             (lit {{type: (t-prim {{}} bool)}} true)))\n"
        );
        let deep = parse_deep(&source).expect("conflicting property Deep parses structurally");

        let error = resugar_program(&deep)
            .expect_err("a property signature must not overwrite its quantifier or result types");
        assert!(
            error.to_string().contains("matching")
                || error.to_string().contains("conflict")
                || error.to_string().contains("returning `bool`"),
            "unexpected property-signature mismatch error: {error}"
        );
    }
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

#[test]
fn synthesized_destructuring_temporaries_do_not_capture_authored_names() {
    let source = concat!(
        "def keep_authored(a: int64) -> int64 = {\n",
        "  __chelis_tmp0 = a\n",
        "  _ = neg(a)\n",
        "  __chelis_tmp0\n",
        "}\n",
    );
    let deep = print_canonical(&desugar_program(
        &parse_str(source).expect("capture regression fixture parses"),
    ));

    assert_eq!(
        deep.matches("__chelis_tmp0").count(),
        2,
        "only the authored binder and its use may retain that name: {deep}"
    );
    assert!(
        deep.contains("__chelis_tmp1") && deep.contains("destructure: true"),
        "the synthesized discard binding must skip the authored name: {deep}"
    );
}

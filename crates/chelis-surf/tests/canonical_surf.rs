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
        "result = vmap(f)",
        "result = vmap(f, axis=1)",
        "result = grad(f, wrt=x)",
        "result = (x,)",
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
fn formatting_canonical_surf_is_idempotent() {
    let source = "def identity(x: f32) -> f32 = x\nresult = Point { x, y: other }\n";
    let once = format_source(source).expect("canonical Surf formats");
    let twice = format_source(&once).expect("formatted Surf reparses");

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
        "@property bounded forall(p: Probability) where (0.0 <= p.value):\n",
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
        "-- source footer\n",
    );

    let migrated = migrate_source_v018(legacy).expect("legacy source migrates");

    assert_eq!(migrated, expected);
    parse_str(&migrated).expect("migration emits only canonical Surf");
    assert_eq!(format_source(&migrated).unwrap(), migrated);
}

#[test]
fn migration_rejects_ambiguous_interior_comment_attachment() {
    let legacy = "def f(x) = g({- belongs to the argument expression -} x)\n";

    let error = migrate_source_v018(legacy).expect_err("ambiguous attachment must not migrate");

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
        "surf_path: \"Module.Path\"} x)\n",
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
    for semantic in ["eff:", "type:", "surf_path:"] {
        assert!(
            normalized.contains(semantic),
            "semantic key was erased: {normalized}"
        );
    }
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

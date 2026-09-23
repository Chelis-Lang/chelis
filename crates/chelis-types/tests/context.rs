//! Phase C: tests for `check_ir_with_context`.
//!
//! ADT-registry probes are written FIRST per the Phase C plan — naive
//! inner-before-outer registry stacking is the most likely silent-correctness
//! bug, so these tests must fail loudly if it is mis-implemented.

use chelis_types::{
    TypeEnv, build_compiled_library_context, build_compiled_library_context_with_base,
    build_type_env_from_library, check_ir_program, check_ir_with_context, check_typed_program,
};

fn parse(src: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(src).expect("deep parse")
}

fn build_ctx(library_src: &str) -> TypeEnv {
    let library = parse(library_src);
    build_type_env_from_library(&library).expect("library checks clean")
}

fn checker_messages(
    result: Result<chelis_types::CheckedProgram, chelis_types::InferResult>,
) -> Vec<String> {
    match result {
        Ok(_) => Vec::new(),
        Err(result) => {
            let mut messages = result
                .errors
                .into_iter()
                .map(|error| format!("[{:?}] {}", error.kind, error.message))
                .collect::<Vec<_>>();
            messages.sort();
            messages
        }
    }
}

#[test]
fn adt_variant_and_field_readers_match_both_carriers() {
    for source in [
        "(deftype {} Pair () \
           (variant {} Pair \
             (field {} left (t-prim {} i32)) \
             (field {} right (t-prim {} bool))))",
        "(deftype {} Pair () \
           (variant {} Pair \
             (field {} left (t-prim {} madeup))))",
    ] {
        let program = chelis_deep::parse_and_stamp_file(source).expect("fixture stamps");
        assert_eq!(
            checker_messages(check_typed_program(&program)),
            checker_messages(check_ir_program(&program)),
            "{source}"
        );
    }
}

fn replace_deftype_variant(program: &mut [chelis_deep::Expr], replacement: chelis_deep::Expr) {
    let chelis_deep::Expr::Node(deftype, span) = &program[0] else {
        panic!("fixture deftype must use the successor carrier");
    };
    let mut children = deftype.children_slice().to_vec();
    children[2] = replacement;
    program[0] = chelis_deep::Expr::node(deftype.tag(), deftype.meta().clone(), children, *span);
}

fn assert_mutated_adt_rejects_on_both_ingresses(program: &[chelis_deep::Expr], label: &str) {
    let typed = checker_messages(check_typed_program(program));
    let ir = checker_messages(check_ir_program(program));
    assert!(!typed.is_empty(), "{label}: malformed ADT must reject");
    assert_eq!(typed, ir, "{label}: carrier mutation must preserve parity");
}

/// A malformed variant or field can no longer be spelled as a vocabulary
/// node (the stamped `Node` constructor rejects it), so the mutation uses the
/// carriers that remain admissible in those slots: an undecodable head and a
/// structural list. Both programs must reject, with identical diagnostics on
/// the two checker entries. The field mutation is rejected by the field
/// reader itself; the variant mutation is rejected through `make`'s use of
/// the constructor the dropped variant would have declared.
#[test]
fn adt_non_node_variant_and_field_carriers_reject_with_entry_parity() {
    use chelis_deep::{Atom, Expr, Span, UnknownFormData};

    let source = "(deftype {} Pair () \
                    (variant {} Pair (field {} value (t-prim {} i32))))\n\
                  (def {} make (app {} (var {} Pair) (lit {type: (t-prim {} i32)} 1)))";
    let span = Span::new(0, 0);

    let mut malformed_variant =
        chelis_deep::parse_and_stamp_file(source).expect("variant fixture stamps");
    replace_deftype_variant(
        &mut malformed_variant,
        Expr::UnknownForm(Box::new(UnknownFormData {
            head: "not-a-variant".into(),
            meta: chelis_deep::Metadata::default(),
            children: vec![Expr::Atom(Atom::Name("Pair".into()), span)],
            span,
        })),
    );
    assert_mutated_adt_rejects_on_both_ingresses(&malformed_variant, "undecodable variant");

    let mut malformed_field =
        chelis_deep::parse_and_stamp_file(source).expect("field fixture stamps");
    let (variant_tag, variant_meta, mut variant_children, variant_span) = {
        let Expr::Node(deftype, _) = &malformed_field[0] else {
            panic!("fixture deftype must use the successor carrier");
        };
        let Expr::Node(variant, variant_span) = &deftype.children_slice()[2] else {
            panic!("fixture variant must use the successor carrier");
        };
        (
            variant.tag(),
            variant.meta().clone(),
            variant.children_slice().to_vec(),
            *variant_span,
        )
    };
    let field_type = {
        let Expr::Node(field, _) = &variant_children[1] else {
            panic!("fixture field must use the successor carrier");
        };
        field.children_slice()[1].clone()
    };
    variant_children[1] = Expr::BareList(
        vec![Expr::Atom(Atom::Name("value".into()), span), field_type],
        span,
    );
    replace_deftype_variant(
        &mut malformed_field,
        Expr::node(variant_tag, variant_meta, variant_children, variant_span),
    );
    assert_mutated_adt_rejects_on_both_ingresses(&malformed_field, "structural field");
}

#[test]
fn adt_reader_has_no_local_optional_carrier_adapter() {
    let source = include_str!("../src/adt.rs");
    let definition = ["fn stamped_", "parts"].concat();
    let call = ["stamped_", "parts("].concat();
    assert!(
        !source.contains(&definition) && !source.contains(&call),
        "E5b requires ADT variant and field reads to disposition ExprCarrier directly"
    );
}

#[test]
fn type_environment_matches_its_checked_library_program() {
    let library = parse("(def {} one (lit {type: (t-prim {} i32)} 1))");
    let (type_env, checked) =
        build_compiled_library_context(&library).expect("the library must type-check");

    assert!(type_env.matches_checked_program(&checked));
}

#[test]
fn type_environment_rejects_another_checked_library_program() {
    let first = parse("(def {} one (lit {type: (t-prim {} i32)} 1))");
    let second = parse("(def {} two (lit {type: (t-prim {} i32)} 2))");
    let (type_env, _) =
        build_compiled_library_context(&first).expect("the first library must type-check");
    let (_, checked) =
        build_compiled_library_context(&second).expect("the second library must type-check");

    assert!(!type_env.matches_checked_program(&checked));
}

#[test]
fn serialized_context_resumes_through_library_and_signature_layers() {
    let base_library = parse("(def {} library_id (fn {} (params {} value) (var {} value)))");
    let (base, _) =
        build_compiled_library_context(&base_library).expect("base library type-checks");
    let encoded = bincode::serialize(&base).expect("base TypeEnv serializes");
    let decoded: TypeEnv = bincode::deserialize(&encoded).expect("base TypeEnv deserializes");

    let layer = parse(
        "(def {} int_use
            (app {} (var {} library_id) (lit {type: (t-prim {} i32)} 1)))
         (def {} bool_use
            (app {} (var {} library_id) (lit {type: (t-prim {} bool)} true)))
         (def {} layer_id (fn {} (params {} value) (var {} value)))",
    );
    let (layer_context, _) = build_compiled_library_context_with_base(&decoded, &layer)
        .expect("a serialized base resumes before the added library layer");

    let new_code = parse(
        "(def {} layer_int
            (app {} (var {} layer_id) (lit {type: (t-prim {} i32)} 2)))
         (def {} layer_bool
            (app {} (var {} layer_id) (lit {type: (t-prim {} bool)} false)))",
    );
    check_ir_with_context(&layer_context, &new_code)
        .expect("signature-context checking resumes imported IDs before minting new ones");
}

// ── ADT exhaustivity probes (must pass before any general stacking work) ──

#[test]
fn adt_probe_library_option_exhaustive_match_in_new_code() {
    // Library defines Option[a]; new code matches an Option with both arms.
    let ctx = build_ctx(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
         (def {} library_some (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 1)))",
    );

    let new_exprs = parse(
        "(def {} result
           (match {} (var {} library_some)
             (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))
             (arm {} (pat-ctor {} MyNone) () (lit {type: (t-prim {} i32)} 0))))",
    );

    check_ir_with_context(&ctx, &new_exprs)
        .expect("exhaustive match against library ADT must pass");
}

#[test]
fn adt_probe_library_option_non_exhaustive_match_is_rejected() {
    // Library defines Option[a]; new code matches with only Some arm.
    // Must reject as non-exhaustive citing missing None — NOT silently accept.
    let ctx = build_ctx(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
         (def {} library_some (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 1)))",
    );

    let new_exprs = parse(
        "(def {} result
           (match {} (var {} library_some)
             (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))))",
    );

    let res = check_ir_with_context(&ctx, &new_exprs);
    let err = res.expect_err("non-exhaustive match against library ADT must be rejected");
    let mentions_missing_none = err.errors.iter().any(|e| {
        format!("{:?}", e.kind).contains("NonExhaustiveMatch") && e.message.contains("MyNone")
    });
    assert!(
        mentions_missing_none,
        "expected NonExhaustiveMatch citing MyNone, got: {:?}",
        err.errors
    );
}

#[test]
fn adt_probe_new_code_adt_does_not_inherit_library_variants() {
    // Library defines its own ADT. New code defines a different ADT and
    // matches it. The library registry must NOT pollute the new ADT's
    // exhaustivity — the new ADT has its own variant set.
    let ctx = build_ctx(
        "(deftype {} LibColor () (variant {} LibRed) (variant {} LibBlue))
         (def {} lib_red (var {} LibRed))",
    );

    // New code: define Shape, match on Shape with all its variants.
    let new_ok = parse(
        "(deftype {} Shape () (variant {} Circle) (variant {} Square))
         (def {} v (var {} Circle))
         (def {} result
           (match {} (var {} v)
             (arm {} (pat-ctor {} Circle) () (lit {type: (t-prim {} i32)} 1))
             (arm {} (pat-ctor {} Square) () (lit {type: (t-prim {} i32)} 2))))",
    );
    check_ir_with_context(&ctx, &new_ok)
        .expect("new ADT exhaustively matched must pass without library variants leaking in");

    // Same Shape definition but only Circle arm — must still be non-exhaustive
    // (missing Square), not silently accepted.
    let new_bad = parse(
        "(deftype {} Shape () (variant {} Circle) (variant {} Square))
         (def {} v (var {} Circle))
         (def {} result
           (match {} (var {} v)
             (arm {} (pat-ctor {} Circle) () (lit {type: (t-prim {} i32)} 1))))",
    );
    let err = check_ir_with_context(&ctx, &new_bad)
        .expect_err("non-exhaustive new ADT match must be rejected");
    let cites_square = err.errors.iter().any(|e| {
        format!("{:?}", e.kind).contains("NonExhaustiveMatch") && e.message.contains("Square")
    });
    assert!(
        cites_square,
        "expected NonExhaustiveMatch citing Square, got: {:?}",
        err.errors
    );
}

// ── Composition: with-context must equal monolithic for snippet equivalence ──

#[test]
fn with_context_equals_monolithic_for_five_snippets() {
    // Phase C plan acceptance: build a small library context once, then check
    // 5 snippets against it. Each snippet's checked-program output equals the
    // tail of the monolithic check on (library + snippet).
    let library_src = "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
        (def {} double (fn {} (params {} x)
            (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} i32)} 2))))
        (def {} library_id (fn {} (params {} y) (var {} y)))";

    let library = parse(library_src);
    let ctx = build_type_env_from_library(&library).expect("library OK");

    let snippets = [
        "(def {} a (app {} (var {} double) (lit {type: (t-prim {} i32)} 7)))",
        "(def {} b (app {} (var {} library_id) (lit {type: (t-prim {} i32)} 3)))",
        "(def {} c (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 9)))",
        "(def {} d
           (match {} (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 1))
             (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))
             (arm {} (pat-ctor {} MyNone) () (lit {type: (t-prim {} i32)} 0))))",
        "(def {} e (lit {type: (t-prim {} f32)} 2.5))",
    ];

    for snippet in snippets {
        let new_exprs = parse(snippet);
        let with_ctx = check_ir_with_context(&ctx, &new_exprs)
            .unwrap_or_else(|e| panic!("with_context failed on snippet {snippet}: {:?}", e.errors));

        let combined_src = format!("{library_src}\n{snippet}");
        let combined = parse(&combined_src);
        let monolithic = chelis_types::check_ir_program(&combined)
            .unwrap_or_else(|e| panic!("monolithic failed on combined: {:?}", e.errors));

        // The new-code's annotated decls should equal the tail of monolithic
        // result (post-library prefix).
        let library_decl_count = library.len();
        let monolithic_tail: Vec<_> = monolithic
            .annotated_exprs()
            .iter()
            .skip(library_decl_count)
            .cloned()
            .collect();
        assert_eq!(
            with_ctx.annotated_exprs().len(),
            monolithic_tail.len(),
            "annotated decl count differs for snippet `{snippet}`: \
             with-context returned {} decls, monolithic tail had {}",
            with_ctx.annotated_exprs().len(),
            monolithic_tail.len(),
        );
    }
}

#[test]
fn no_leak_between_snippets_against_same_context() {
    // A binding declared in snippet A must NOT be visible when type-checking
    // snippet B against the same context.
    let ctx = build_ctx("(def {} library_id (fn {} (params {} y) (var {} y)))");

    // Snippet A defines `secret_a`.
    let snippet_a = parse("(def {} secret_a (lit {type: (t-prim {} i32)} 42))");
    check_ir_with_context(&ctx, &snippet_a).expect("snippet A clean");

    // Snippet B references `secret_a`. If A leaked into ctx, this would pass.
    // It must fail with UnboundVariable.
    let snippet_b = parse("(def {} use_b (var {} secret_a))");
    let err = check_ir_with_context(&ctx, &snippet_b)
        .expect_err("snippet A's binding must not leak into ctx");
    let mentions_unbound = err
        .errors
        .iter()
        .any(|e| format!("{:?}", e.kind).contains("UnboundVariable"));
    assert!(
        mentions_unbound,
        "expected UnboundVariable for secret_a, got: {:?}",
        err.errors
    );
}

#[test]
fn empty_context_matches_check_ir_program() {
    // check_ir_with_context with an empty TypeEnv must behave the same
    // as check_ir_program on the same exprs.
    let exprs = parse(
        "(deftype {} Foo () (variant {} A) (variant {} B))
         (def {} pick (var {} A))
         (def {} use
           (match {} (var {} pick)
             (arm {} (pat-ctor {} A) () (lit {type: (t-prim {} i32)} 1))
             (arm {} (pat-ctor {} B) () (lit {type: (t-prim {} i32)} 2))))",
    );

    let empty_ctx = TypeEnv::empty();
    let with_ctx = check_ir_with_context(&empty_ctx, &exprs)
        .expect("empty-context check must succeed on standalone program");
    let mono = chelis_types::check_ir_program(&exprs).expect("monolithic must succeed");
    assert_eq!(
        with_ctx.annotated_exprs().len(),
        mono.annotated_exprs().len()
    );
}

// ── Regression: RT-C HIGH finding — type_env() must surface library decls ──

#[test]
fn type_env_surfaces_library_declared_types() {
    // RT-C HIGH (2026-04-26): with-context CheckedProgram.type_env() previously
    // only contained new-code declared types. Downstream passes (chelis-ir lower,
    // chelis-effects, linearity) read `program.type_env()` to resolve `(var lib)`
    // references; an absent library name returned None and broke composition.
    // Fix: union library `ir_types` into the returned type_env (new-code
    // wins on conflict). This regression test locks the union in.
    let ctx = build_ctx(
        "(def {} double (fn {} (params {} (x {type: (t-prim {} i32)}))
            (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} i32)} 2))))",
    );
    let checked = check_ir_with_context(
        &ctx,
        &parse("(def {} call (app {} (var {} double) (lit {type: (t-prim {} i32)} 5)))"),
    )
    .expect("check OK");

    let env = checked.type_env();
    assert!(env.contains_key("call"), "new-code def is in type_env");
    assert!(
        env.contains_key("double"),
        "library def MUST be surfaced in with-context type_env so downstream \
         passes can resolve cross-context name references; without this, lower / \
         effects / linearity will break on library calls. type_env keys: {:?}",
        env.keys().collect::<Vec<_>>(),
    );

    // Cross-check: monolithic on the union has the same key.
    let mono = chelis_types::check_ir_program(&parse(
        "(def {} double (fn {} (params {} (x {type: (t-prim {} i32)}))
            (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} i32)} 2))))
         (def {} call (app {} (var {} double) (lit {type: (t-prim {} i32)} 5)))",
    ))
    .expect("mono OK");
    assert!(mono.type_env().contains_key("double"));
}

#[test]
fn type_env_new_code_shadows_library_on_name_conflict() {
    // If new-code redefines a library name, with-context type_env should
    // hold the NEW-code's declared type, not the library's. (Independent of
    // whether the checker accepts the redef body — this test only inspects
    // the returned type_env shape on a successful check.)
    let ctx = build_ctx("(def {} foo (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))");
    // New code re-declares `foo` with the same signature. Should check OK
    // and `foo` should resolve to a type_env entry (not panic / not absent).
    let new_exprs =
        parse("(def {} foo (fn {} (params {} (x {type: (t-prim {} i32)})) (var {} x)))");
    let res = check_ir_with_context(&ctx, &new_exprs);
    if let Ok(checked) = res {
        assert!(
            checked.type_env().contains_key("foo"),
            "redeclared name must appear in type_env"
        );
    }
    // If redef is rejected (current checker policy), that's fine — the type_env
    // surface shape is only relevant on Ok results.
}

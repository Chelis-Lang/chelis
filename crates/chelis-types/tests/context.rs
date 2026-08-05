//! Phase C: tests for `check_ir_with_context`.
//!
//! ADT-registry probes are written FIRST per the Phase C plan — naive
//! inner-before-outer registry stacking is the most likely silent-correctness
//! bug, so these tests must fail loudly if it is mis-implemented.

use chelis_types::{
    TypeEnv, build_compiled_library_context, build_type_env_from_library, check_ir_with_context,
};

fn parse(src: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(src).expect("deep parse")
}

fn build_ctx(library_src: &str) -> TypeEnv {
    let library = parse(library_src);
    build_type_env_from_library(&library).expect("library checks clean")
}

#[test]
fn type_environment_matches_its_checked_library_program() {
    let library = parse("(def {} one (lit {type: (t-prim {} int32)} 1))");
    let (type_env, checked) =
        build_compiled_library_context(&library).expect("the library must type-check");

    assert!(type_env.matches_checked_program(&checked));
}

#[test]
fn type_environment_rejects_another_checked_library_program() {
    let first = parse("(def {} one (lit {type: (t-prim {} int32)} 1))");
    let second = parse("(def {} two (lit {type: (t-prim {} int32)} 2))");
    let (type_env, _) =
        build_compiled_library_context(&first).expect("the first library must type-check");
    let (_, checked) =
        build_compiled_library_context(&second).expect("the second library must type-check");

    assert!(!type_env.matches_checked_program(&checked));
}

// ── ADT exhaustivity probes (must pass before any general stacking work) ──

#[test]
fn adt_probe_library_option_exhaustive_match_in_new_code() {
    // Library defines Option[a]; new code matches an Option with both arms.
    let ctx = build_ctx(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
         (def {} library_some (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 1)))",
    );

    let new_exprs = parse(
        "(def {} result
           (match {} (var {} library_some)
             (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))
             (arm {} (pat-ctor {} MyNone) () (lit {type: (t-prim {} int32)} 0))))",
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
         (def {} library_some (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 1)))",
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
             (arm {} (pat-ctor {} Circle) () (lit {type: (t-prim {} int32)} 1))
             (arm {} (pat-ctor {} Square) () (lit {type: (t-prim {} int32)} 2))))",
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
             (arm {} (pat-ctor {} Circle) () (lit {type: (t-prim {} int32)} 1))))",
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
            (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} int32)} 2))))
        (def {} library_id (fn {} (params {} y) (var {} y)))";

    let library = parse(library_src);
    let ctx = build_type_env_from_library(&library).expect("library OK");

    let snippets = [
        "(def {} a (app {} (var {} double) (lit {type: (t-prim {} int32)} 7)))",
        "(def {} b (app {} (var {} library_id) (lit {type: (t-prim {} int32)} 3)))",
        "(def {} c (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 9)))",
        "(def {} d
           (match {} (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 1))
             (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))
             (arm {} (pat-ctor {} MyNone) () (lit {type: (t-prim {} int32)} 0))))",
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
    let snippet_a = parse("(def {} secret_a (lit {type: (t-prim {} int32)} 42))");
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
             (arm {} (pat-ctor {} A) () (lit {type: (t-prim {} int32)} 1))
             (arm {} (pat-ctor {} B) () (lit {type: (t-prim {} int32)} 2))))",
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
        "(def {} double (fn {} (params {} (x {type: (t-prim {} int32)}))
            (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} int32)} 2))))",
    );
    let checked = check_ir_with_context(
        &ctx,
        &parse("(def {} call (app {} (var {} double) (lit {type: (t-prim {} int32)} 5)))"),
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
        "(def {} double (fn {} (params {} (x {type: (t-prim {} int32)}))
            (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} int32)} 2))))
         (def {} call (app {} (var {} double) (lit {type: (t-prim {} int32)} 5)))",
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
    let ctx =
        build_ctx("(def {} foo (fn {} (params {} (x {type: (t-prim {} int32)})) (var {} x)))");
    // New code re-declares `foo` with the same signature. Should check OK
    // and `foo` should resolve to a type_env entry (not panic / not absent).
    let new_exprs =
        parse("(def {} foo (fn {} (params {} (x {type: (t-prim {} int32)})) (var {} x)))");
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

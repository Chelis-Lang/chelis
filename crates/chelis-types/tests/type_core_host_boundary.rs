//! Host authority, context isolation, and trusted cache fixtures for the type core.

use std::sync::Arc;

use chelis_types::{
    TypeEnv, build_compiled_library_context, check_ir_program, check_ir_with_context,
};

fn deep(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("valid Deep fixture")
}

fn accepted_snapshot(program: &chelis_types::CheckedProgram) -> String {
    let mut types: Vec<_> = program.type_env().iter().collect();
    types.sort_by_key(|(name, _)| *name);
    format!(
        "{}\n{types:?}\n{:?}\n{:?}",
        chelis_deep::printer::print_canonical(program.annotated_exprs()),
        program.signature_inference(),
        program.infer_stats(),
    )
}

#[test]
fn shared_context_checks_keep_bindings_and_diagnostics_separate() {
    let library = deep("(def {} library_id (fn {} (params {} value) (var {} value)))");
    let (context, checked_library) =
        build_compiled_library_context(&library).expect("valid library");
    let context = Arc::new(context);
    let bytes_before = bincode::serialize(context.as_ref()).expect("serialize trusted context");
    let sources = [
        "(def {} local_int (app {} (var {} library_id) (lit {type: (t-prim {} i32)} 3)))",
        "(def {} local_float (app {} (var {} library_id) (lit {type: (t-prim {} f32)} 2.5)))",
    ];
    let expected: Vec<_> = sources
        .iter()
        .map(|source| {
            accepted_snapshot(
                &check_ir_with_context(&context, &deep(source)).expect("isolated check"),
            )
        })
        .collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = sources
            .iter()
            .map(|source| {
                let context = Arc::clone(&context);
                scope.spawn(move || {
                    accepted_snapshot(
                        &check_ir_with_context(&context, &deep(source)).expect("concurrent check"),
                    )
                })
            })
            .collect();
        for (handle, expected) in handles.into_iter().zip(expected) {
            assert_eq!(handle.join().expect("thread return"), expected);
        }
    });
    let rejected = check_ir_with_context(&context, &deep("(def {} failed (var {} local_int))"))
        .expect_err("a prior check cannot add a library binding");
    assert!(
        rejected
            .errors
            .iter()
            .any(|error| error.message.contains("local_int"))
    );
    assert!(context.matches_checked_program(&checked_library));
    assert_eq!(
        bincode::serialize(context.as_ref()).expect("serialize unchanged context"),
        bytes_before
    );
}

#[test]
fn trusted_context_round_trip_retains_checked_product_and_high_water_marks() {
    let library = deep("(def {} library_id (fn {} (params {} value) (var {} value)))");
    let (context, checked_library) =
        build_compiled_library_context(&library).expect("valid library");
    let bytes = bincode::serialize(&context).expect("serialize trusted context");
    let decoded: TypeEnv = bincode::deserialize(&bytes).expect("decode trusted internal artifact");
    assert!(decoded.matches_checked_program(&checked_library));
    for source in [
        "(def {} result (app {} (var {} library_id) (lit {type: (t-prim {} i32)} 3)))",
        "(def {} result (app {} (var {} library_id) (lit {type: (t-prim {} f32)} 2.5)))",
    ] {
        let unit = deep(source);
        assert_eq!(
            accepted_snapshot(&check_ir_with_context(&context, &unit).expect("original check")),
            accepted_snapshot(&check_ir_with_context(&decoded, &unit).expect("cached check")),
        );
    }
    assert!(check_ir_program(&deep("(def {} result (var {} library_id))")).is_err());
}

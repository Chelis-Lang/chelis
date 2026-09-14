//! Checked library contexts transport family contracts without source-body validation.
use chelis_compiler_api::{
    LibraryContext, StdLibContext, build_library_context, build_stdlib_context,
};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{check_ir_with_context, errors::CheckErrorKind};

#[test]
fn checked_library_round_trips_preserve_generic_and_primitive_function_values() {
    let declarations = parse_str(
        "def average[p: Float](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n\
         def middle[q: Float](x: tensor[3, q]) -> tensor[q] = average(x)\n\
         primitive_average = mean\n",
    )
    .unwrap();
    let stdlib = build_stdlib_context(&declarations).unwrap();
    let stdlib_decoded: StdLibContext =
        bincode::deserialize(&bincode::serialize(&stdlib).unwrap()).unwrap();
    let empty = build_stdlib_context(&[]).unwrap();
    let dependency = build_library_context(&empty, &declarations)
        .unwrap()
        .unwrap();
    let dependency_decoded: LibraryContext =
        bincode::deserialize(&bincode::serialize(&dependency).unwrap()).unwrap();
    for env in [
        stdlib.type_env(),
        stdlib_decoded.type_env(),
        dependency.type_env(),
        dependency_decoded.type_env(),
    ] {
        for (dtype, values, accepts) in [
            ("f32", "[1.0f32, 2.0f32, 6.0f32]", true),
            ("int32", "[1i32, 2i32, 6i32]", false),
        ] {
            for expression in [
                format!("middle(to_tensor({values}))"),
                format!("{{ f = average\n f(to_tensor({values})) }}"),
                format!("primitive_average(to_tensor({values}), 0i32)"),
            ] {
                let source = format!("out: tensor[{dtype}] = {expression}\n");
                let verdict =
                    check_ir_with_context(env, &desugar_program(&parse_str(&source).unwrap()));
                if accepts {
                    verdict.unwrap_or_else(|errors| panic!("{source}: {errors:?}"));
                } else {
                    let errors = verdict.expect_err("an imported contract must reject int32");
                    assert!(
                        errors.errors.iter().any(|error| matches!(
                            error.kind,
                            CheckErrorKind::PrecisionMismatch
                        ) && error.message.contains("Float")),
                        "{source}: {errors:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn checked_library_round_trips_preserve_window_reduction_contracts() {
    let declarations = parse_str(
        "window_mean = reduce_window_mean\n\
         window_sum = reduce_window_sum\n",
    )
    .unwrap();
    let stdlib = build_stdlib_context(&declarations).unwrap();
    let stdlib_decoded: StdLibContext =
        bincode::deserialize(&bincode::serialize(&stdlib).unwrap()).unwrap();
    let empty = build_stdlib_context(&[]).unwrap();
    let dependency = build_library_context(&empty, &declarations)
        .unwrap()
        .unwrap();
    let dependency_decoded: LibraryContext =
        bincode::deserialize(&bincode::serialize(&dependency).unwrap()).unwrap();
    for env in [
        stdlib.type_env(),
        stdlib_decoded.type_env(),
        dependency.type_env(),
        dependency_decoded.type_env(),
    ] {
        for (source, accepted) in [
            (
                "out: tensor[2, f32] = window_mean(\
                   to_tensor([1.0f32, 2.0f32, 3.0f32]), [2i64], [1i64])\n",
                true,
            ),
            (
                "out: tensor[2, int32] = window_mean(\
                   to_tensor([1i32, 2i32, 3i32]), [2i64], [1i64])\n",
                false,
            ),
            (
                "out: tensor[2, int32] = window_sum(\
                   to_tensor([1i32, 2i32, 3i32]), [2i64], [1i64])\n",
                true,
            ),
            (
                "out: tensor[2, bool] = window_sum(\
                   to_tensor([true, false, true]), [2i64], [1i64])\n",
                false,
            ),
        ] {
            let verdict = check_ir_with_context(env, &desugar_program(&parse_str(source).unwrap()));
            if accepted {
                verdict.unwrap_or_else(|errors| panic!("{source}: {errors:?}"));
            } else {
                let errors = verdict.expect_err("an imported window contract must reject");
                assert!(
                    errors
                        .errors
                        .iter()
                        .any(|error| matches!(error.kind, CheckErrorKind::PrecisionMismatch)),
                    "{source}: {errors:?}"
                );
            }
        }
    }
}

#[test]
fn insufficient_authored_contracts_cannot_be_published_as_checked_libraries() {
    let empty = build_stdlib_context(&[]).unwrap();
    for (source, accepted) in [
        ("def g(x) = sin(x)\n", false),
        ("def g(x) = sin(x)\nout = g(0.0f32)\n", false),
        ("def make() = fn (x) -> sin(x)\n", false),
        ("def g(x: f32) -> f32 = sin(x)\n", true),
        ("def make() = fn (x: f32) -> sin(x)\n", true),
        ("primitive = sin\n", true),
        (
            "def source[p: Float]() -> p = cast(0.0f32, p)\ndef make() = source()\n",
            false,
        ),
        (
            "def source[p: Float]() -> p = cast(0.0f32, p)\ndef make() -> f32 = source()\n",
            true,
        ),
        ("def source() = sin\ndef make() = source()\n", true),
        ("def less[p](x: p, y: p) -> bool = lt(x, y)\n", false),
        (
            "def less[p: Numeric](x: p, y: p) -> bool = lt(x, y)\n",
            true,
        ),
    ] {
        let declarations = parse_str(source).unwrap();
        assert_eq!(
            build_stdlib_context(&declarations).is_ok(),
            accepted,
            "{source}"
        );
        assert_eq!(
            build_library_context(&empty, &declarations)
                .unwrap()
                .is_some(),
            accepted,
            "{source}"
        );
    }
    for binder in ["p", "p: Numeric", "p: Float"] {
        let declarations = parse_str(&format!(
            "def average[{binder}](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n"
        ))
        .unwrap();
        assert_eq!(
            build_stdlib_context(&declarations).is_ok(),
            binder == "p: Float"
        );
        assert_eq!(
            build_library_context(&empty, &declarations)
                .unwrap()
                .is_some(),
            binder == "p: Float"
        );
    }
    for (operation, binder, accepted) in [
        ("reduce_window_mean", "p", false),
        ("reduce_window_mean", "p: Numeric", false),
        ("reduce_window_mean", "p: Float", true),
        ("reduce_window_sum", "p", false),
        ("reduce_window_sum", "p: Numeric", true),
    ] {
        let declarations = parse_str(&format!(
            "def windowed[{binder}](x: tensor[3, p]) -> tensor[2, p] = \
             {operation}(x, [2i64], [1i64])\n"
        ))
        .unwrap();
        assert_eq!(
            build_stdlib_context(&declarations).is_ok(),
            accepted,
            "{operation} with {binder}"
        );
        assert_eq!(
            build_library_context(&empty, &declarations)
                .unwrap()
                .is_some(),
            accepted,
            "{operation} with {binder}"
        );
    }
}

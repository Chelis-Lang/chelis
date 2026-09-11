//! chelis#1512: a checked route's own validation must run on the operand it
//! actually receives, not on whatever the substitution happened to hold when
//! the route's node was inferred.
//!
//! The table below is four cells per route over the live producer measured on
//! `main` `e813415d0`: an unannotated lambda parameter bound by a later
//! application. (Resolved, valid) accepts, (resolved, invalid) rejects with
//! the route's own diagnostic, (unresolved, valid) accepts, and (unresolved,
//! invalid) rejects with the SAME diagnostic text. The fourth cell is the
//! issue; the third is its other half on the `app_shape` routes, which used
//! to publish the operand's own type and so rejected a correct declared
//! result while accepting the operand's shape.
//!
//! Every assertion here is a REGRESSION TEST unless its own comment says
//! otherwise: each was watched failing on `e813415d0` before the repair, and
//! the recorded pre-repair verdict is in the row's comment.
//!
//! Asserting the diagnostic TEXT rather than mere rejection is deliberate.
//! Suspending a decision relocates it, and anything the eager call path did
//! on its way to that decision can silently stop happening (measured three
//! times on chelis#1489's conversions). A cell that only checked "rejected"
//! would pass on a rejection from an entirely different rule.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn check(source: &str) -> Result<(), Vec<CheckError>> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    match check_typed_program(&desugar_program(&decls)) {
        Ok(_) => Ok(()),
        Err(report) => Err(report.errors),
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The same call with its operand supplied directly.
fn resolved(decl: &str, call: &str, result: &str) -> String {
    format!("def f({decl}) -> {result} = {}\n", call.replace('$', "x"))
}

/// The same call with its operand an unannotated lambda parameter, bound only
/// by the application on the line below it. The route runs while the operand
/// is still `Type::Var`.
fn unresolved(decl: &str, call: &str, result: &str) -> String {
    format!(
        "def f({decl}) -> {result} = {{\n  g = fn (t) -> {}\n  g(x)\n}}\n",
        call.replace('$', "t")
    )
}

/// One route's four cells. `valid_result` is the type the route's rule
/// actually produces; `invalid_call` is a call the rule rejects, and
/// `diagnostic` is the substring its own rejection prints.
struct Row {
    route: &'static str,
    decl: &'static str,
    valid_call: &'static str,
    valid_result: &'static str,
    invalid_call: &'static str,
    invalid_result: &'static str,
    diagnostic: &'static str,
}

fn run(row: &Row) {
    let Row {
        route,
        decl,
        valid_call,
        valid_result,
        invalid_call,
        invalid_result,
        diagnostic,
    } = row;

    // Cell 1 (resolved, valid): DISPOSITION LOCK. Green before the repair;
    // it is here so a repair that rejects correct programs is caught.
    check(&resolved(decl, valid_call, valid_result)).unwrap_or_else(|e| {
        panic!(
            "{route}: a valid resolved call must check:\n{}",
            summary(&e)
        )
    });

    // Cell 2 (resolved, invalid): DISPOSITION LOCK. This is the rule that was
    // always right and simply never ran; the cell pins its exact wording so
    // cell 4 has something to be equal to.
    let eager = check(&resolved(decl, invalid_call, invalid_result)).expect_err(&format!(
        "{route}: an invalid resolved call must be rejected"
    ));
    assert!(
        eager.iter().any(|e| e.message.contains(diagnostic)),
        "{route}: the resolved rejection must name its own rule, got:\n{}",
        summary(&eager)
    );

    // Cell 3 (unresolved, valid): REGRESSION TEST for the `app_shape` family,
    // DISPOSITION LOCK elsewhere. Before the repair the five window routes
    // published the OPERAND's type here, so a correct declared result was
    // rejected with a signature mismatch.
    check(&unresolved(decl, valid_call, valid_result)).unwrap_or_else(|e| {
        panic!(
            "{route}: a valid call over an unresolved operand must check:\n{}",
            summary(&e)
        )
    });

    // Cell 4 (unresolved, invalid): REGRESSION TEST. This is chelis#1512.
    // Before the repair every route below accepted this program at score 1
    // with an empty error list.
    let deferred = check(&unresolved(decl, invalid_call, invalid_result)).expect_err(&format!(
        "{route}: an invalid call over an unresolved operand must be rejected"
    ));
    assert!(
        deferred.iter().any(|e| e.message.contains(diagnostic)),
        "{route}: the deferred rejection must carry the SAME diagnostic as the resolved one \
         ({diagnostic:?}), got:\n{}",
        summary(&deferred)
    );
}

/// The `app_shape` window family: `permute`, `shrink`, `stride`, `pad` and
/// the `reduce_window_*` group. These five reach inference through `app.rs`
/// and their rules are replayed by `DeferredShapeRule::ShapeRoute`.
#[test]
fn app_shape_window_family_validates_a_late_bound_operand() {
    for row in [
        Row {
            route: "permute",
            decl: "x: tensor[3, 2, f32]",
            valid_call: "permute($, 1i32, 0i32)",
            valid_result: "tensor[2, 3, f32]",
            invalid_call: "permute($, 1i32, 0i32, 2i32)",
            invalid_result: "tensor[2, 3, f32]",
            diagnostic: "permute expects 2 axis indices for rank 2 tensor, got 3",
        },
        Row {
            route: "shrink",
            decl: "x: tensor[2, 4, f32]",
            valid_call: "shrink($, [[0i64, 1i64], [1i64, 3i64]])",
            valid_result: "tensor[1, 2, f32]",
            invalid_call: "shrink($, [[0i64, 1i64]])",
            invalid_result: "tensor[1, 4, f32]",
            diagnostic: "shrink expects 2 bounds pairs for rank 2 tensor, got 1",
        },
        Row {
            route: "stride",
            decl: "x: tensor[2, 4, f32]",
            valid_call: "stride($, 1i64, 2i64)",
            valid_result: "tensor[2, 2, f32]",
            invalid_call: "stride($, 1i64)",
            invalid_result: "tensor[2, 4, f32]",
            diagnostic: "stride expects 2 strides for rank 2 tensor, got 1",
        },
        Row {
            route: "pad",
            decl: "x: tensor[2, 4, f32]",
            valid_call: "pad($, [[0i64, 1i64], [0i64, 1i64]], 0.0f32)",
            valid_result: "tensor[3, 5, f32]",
            invalid_call: "pad($, [[0i64, 1i64]], 0.0f32)",
            invalid_result: "tensor[3, 4, f32]",
            diagnostic: "pad expects 2 padding pairs for rank 2 tensor, got 1",
        },
    ] {
        run(&row);
    }
}

/// `app_post` routes whose operand is a tensor.
#[test]
fn app_post_tensor_routes_validate_a_late_bound_operand() {
    for row in [
        Row {
            route: "tensor_to_scalar",
            decl: "x: tensor[3, f32]",
            valid_call: "tensor_to_scalar(sum($, 0i32))",
            valid_result: "f32",
            invalid_call: "tensor_to_scalar($)",
            invalid_result: "f32",
            diagnostic: "tensor_to_scalar expects a rank-0 tensor",
        },
        Row {
            route: "cumsum",
            decl: "x: tensor[2, 4, f32]",
            valid_call: "cumsum($, 0i32)",
            valid_result: "tensor[2, 4, f32]",
            invalid_call: "cumsum($, 7i32)",
            invalid_result: "tensor[2, 4, f32]",
            diagnostic: "cumsum axis 7 out of bounds for rank 2",
        },
        Row {
            route: "sort",
            decl: "x: tensor[2, 4, f32]",
            valid_call: "sort($, 0i32)",
            valid_result: "(tensor[2, 4, f32], tensor[2, 4, int64])",
            invalid_call: "sort($, 7i32)",
            invalid_result: "(tensor[2, 4, f32], tensor[2, 4, int64])",
            diagnostic: "sort axis 7 out of bounds for rank 2",
        },
    ] {
        run(&row);
    }
}

/// `app_post` routes whose rule is an operand-KIND check: the resolved arm
/// requires a `List`, a `Dict`, a string or a tensor, and the unresolved arm
/// used to accept anything at all.
#[test]
fn app_post_collection_and_host_routes_validate_a_late_bound_operand() {
    for row in [
        Row {
            route: "len",
            decl: "x: List[int32]",
            valid_call: "len($)",
            valid_result: "int64",
            invalid_call: "len(to_tensor($))",
            invalid_result: "int64",
            diagnostic: "len expects List or Dict input",
        },
        Row {
            route: "append",
            decl: "x: List[int32]",
            valid_call: "append($, 1i32)",
            valid_result: "List[int32]",
            invalid_call: "append(to_tensor($), 1i32)",
            invalid_result: "List[int32]",
            diagnostic: "append expects List input",
        },
        Row {
            route: "enumerate",
            decl: "x: List[int32]",
            valid_call: "enumerate($)",
            valid_result: "List[(int64, int32)]",
            invalid_call: "enumerate(to_tensor($))",
            invalid_result: "List[(int64, int32)]",
            diagnostic: "enumerate expects List input",
        },
        Row {
            route: "rank",
            decl: "x: tensor[3, f32]",
            valid_call: "rank($)",
            valid_result: "int32",
            invalid_call: "rank(to_list($))",
            invalid_result: "int32",
            diagnostic: "rank expects tensor input",
        },
        Row {
            route: "numel",
            decl: "x: tensor[3, f32]",
            valid_call: "numel($)",
            valid_result: "int64",
            invalid_call: "numel(to_list($))",
            invalid_result: "int64",
            diagnostic: "numel expects tensor input",
        },
        Row {
            route: "shape",
            decl: "x: tensor[3, f32]",
            valid_call: "shape($, 0i32)",
            valid_result: "int64",
            invalid_call: "shape(to_list($), 0i32)",
            invalid_result: "int64",
            diagnostic: "shape expects tensor input",
        },
        Row {
            route: "string_len",
            decl: "x: string",
            valid_call: "string_len($)",
            valid_result: "int64",
            invalid_call: "string_len(to_int($))",
            invalid_result: "int64",
            diagnostic: "string_len expects string input",
        },
    ] {
        run(&row);
    }
}

/// A `Type::Error` operand keeps its early return: the upstream failure is
/// reported once and the route adds nothing. REGRESSION TEST for the half of
/// the merged arm that must NOT change, and the negative twin of every cell
/// above.
#[test]
fn an_error_operand_still_suppresses_the_routes_own_diagnostic() {
    for (route, program) in [
        (
            "permute",
            "def f(x: tensor[3, f32]) -> tensor[3, f32] = permute(nope(x), 1i32, 0i32)\n",
        ),
        (
            "cumsum",
            "def f(x: tensor[3, f32]) -> tensor[3, f32] = cumsum(nope(x), 7i32)\n",
        ),
        ("len", "def f(x: List[int32]) -> int64 = len(nope(x))\n"),
    ] {
        let errors = check(program).expect_err("the unbound callee is an error");
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("unbound variable: nope")),
            "{route}: the upstream error must be reported:\n{}",
            summary(&errors)
        );
        assert!(
            !errors.iter().any(|e| e.message.contains(route)
                && !e.message.contains("unbound variable")
                && !e.message.contains("declared signature")),
            "{route}: an error operand must not also produce the route's own diagnostic:\n{}",
            summary(&errors)
        );
    }
}

/// An operand that never acquires an outer constructor is rejected at the
/// declaration boundary rather than accepted. REGRESSION TEST: before the
/// repair `permute` accepted this and `len` accepted its twin.
#[test]
fn an_operand_that_never_binds_is_rejected_at_the_declaration_boundary() {
    for (route, program) in [
        (
            "permute",
            "def f() -> int32 = {\n  g = fn (t) -> permute(t, 1i32, 0i32)\n  1i32\n}\n",
        ),
        (
            "len",
            "def f() -> int32 = {\n  g = fn (t) -> len(t)\n  1i32\n}\n",
        ),
    ] {
        let errors = check(program).expect_err("a never-bound operand is not a typed program");
        assert!(
            errors.iter().any(|e| e
                .message
                .contains(&format!("unresolved `{route}` shape obligation"))),
            "{route}: the declaration boundary must name the suspended route:\n{}",
            summary(&errors)
        );
    }
}

/// A declared result that neither candidate reading of the operand produces
/// is rejected once the operand binds. REGRESSION TEST: this is the
/// wrong-answer witness in the issue, where the declared signature, the
/// checker's stamp and the evaluated value were three different answers.
#[test]
fn a_declared_result_the_route_cannot_produce_is_rejected() {
    for (route, program) in [
        (
            "permute",
            "def f(x: tensor[3, 2, f32]) -> tensor[99, f32] = {\n  g = fn (t) -> permute(t, 1i32, 0i32)\n  g(x)\n}\n",
        ),
        (
            "cumsum",
            "def f(x: tensor[3, 2, f32]) -> tensor[99, f32] = {\n  g = fn (t) -> cumsum(t, 0i32)\n  g(x)\n}\n",
        ),
        (
            "pad",
            "def f(x: tensor[2, 4, f32]) -> tensor[99, f32] = {\n  g = fn (t) -> pad(t, [[0i64, 1i64], [0i64, 1i64]], 0.0f32)\n  g(x)\n}\n",
        ),
    ] {
        check(program).expect_err(&format!(
            "{route}: a declared result the route cannot produce must be rejected"
        ));
    }
}

/// The routes chelis#1489 already put on the deferred shape ledger:
/// `matmul`, the reductions, `expand`/`insert`, `layer_norm`, `conv` and
/// `scatter_elements`. Their own `Type::Var` arms are still written, but the
/// CALLER suspends the call before the rule can reach one with an unresolved
/// operand, so the rule always decides against a settled type.
///
/// DISPOSITION LOCK, every cell: all four were green on `e813415d0` before
/// this pull request touched anything. The test exists because the census
/// records these arms as `deferred_by_caller` and names this test as the
/// witness: if that suspension is ever removed, cell 4 goes red here rather
/// than the census silently continuing to claim the arm is covered.
#[test]
fn app_tensor_ledger_family_validates_a_late_bound_operand() {
    for row in [
        Row {
            route: "matmul",
            decl: "x: tensor[3, 2, f32], y: tensor[2, 4, f32], z: tensor[5, 4, f32]",
            valid_call: "matmul($, y)",
            valid_result: "tensor[3, 4, f32]",
            invalid_call: "matmul($, z)",
            invalid_result: "tensor[3, 4, f32]",
            diagnostic: "dimension mismatch: Lit(2) vs Lit(5)",
        },
        Row {
            route: "sum",
            decl: "x: tensor[3, 2, f32]",
            valid_call: "sum($, 0i32)",
            valid_result: "tensor[2, f32]",
            invalid_call: "sum($, 7i32)",
            invalid_result: "tensor[2, f32]",
            diagnostic: "sum axis 7 is out of bounds for rank 2 tensor",
        },
        Row {
            route: "expand",
            decl: "x: tensor[1, 3, f32]",
            valid_call: "expand($, 0i32, 4i64)",
            valid_result: "tensor[4, 3, f32]",
            invalid_call: "expand($, 9i32, 4i64)",
            invalid_result: "tensor[4, 3, f32]",
            diagnostic: "expand axis 9 is out of bounds for rank 2 tensor",
        },
        Row {
            route: "layer_norm",
            decl: "x: tensor[3, 2, f32], w: tensor[2, f32], v: tensor[5, f32]",
            valid_call: "layer_norm($, w, w, 0.001f32)",
            valid_result: "tensor[3, 2, f32]",
            invalid_call: "layer_norm($, v, v, 0.001f32)",
            invalid_result: "tensor[3, 2, f32]",
            diagnostic: "dimension mismatch: Lit(2) vs Lit(5)",
        },
        Row {
            route: "scatter_elements",
            decl: "x: tensor[3, f32], i: tensor[3, int64], u: tensor[3, f32], bad: tensor[2, f32]",
            valid_call: "scatter_elements($, i, u, 0i32)",
            valid_result: "tensor[3, f32]",
            invalid_call: "scatter_elements($, i, bad, 0i32)",
            invalid_result: "tensor[3, f32]",
            diagnostic: "scatter_elements updates shape must equal indices shape",
        },
        Row {
            route: "conv",
            decl: "x: tensor[1, 2, 4, 4, f32], w: tensor[3, 2, 3, 3, f32], bad: tensor[3, 5, 3, 3, f32]",
            valid_call: "conv($, w, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])",
            valid_result: "tensor[1, 3, 2, 2, f32]",
            invalid_call: "conv($, bad, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])",
            invalid_result: "tensor[1, 3, 2, 2, f32]",
            diagnostic: "Lit(2) vs Lit(5)",
        },
    ] {
        run(&row);
    }
}

/// `reshape`, which reached inference in a third spelling: a `Type::Var` arm
/// of its own that duplicated the tensor arm's tail and, when the input was
/// still free, published the OPERAND's type.
///
/// REGRESSION TEST, cells 3 and 4 both. Cell 3 is the sharper one: on
/// `e813415d0` a CORRECT `reshape` over a late-bound operand was REJECTED,
/// because the published operand type could not match the declared result.
#[test]
fn reshape_validates_a_late_bound_operand() {
    run(&Row {
        route: "reshape",
        decl: "x: tensor[3, 2, f32]",
        valid_call: "reshape($, [6i64])",
        valid_result: "tensor[6, f32]",
        invalid_call: "reshape($, [7i64])",
        invalid_result: "tensor[7, f32]",
        diagnostic: "reshape target has 7 elements but input tensor has 6",
    });
}

/// The census's `caught_downstream` rows: arms that DO admit a type variable,
/// where a later rule rejects the same program anyway. Each cell pins the
/// diagnostic that later rule actually prints, so removing the downstream
/// catch turns the census's claim red here.
///
/// DISPOSITION LOCK, every cell: measured on this head, not repaired by this
/// pull request. The wording is another rule's, deliberately: a
/// `caught_downstream` row asserts only that the program is rejected, never
/// that the route's own diagnostic is what the reader sees.
#[test]
fn a_route_whose_arm_admits_a_variable_is_still_rejected_by_a_later_rule() {
    for (route, program, diagnostic) in [
        (
            "and",
            "def f(x: tensor[3, f32], y: tensor[3, f32]) -> tensor[3, bool] = {\n  g = fn (t) -> and(t, y)\n  g(x)\n}\n",
            "and does not accept argument type tensor[3, f32] in this context",
        ),
        (
            "string_contains",
            "def f(x: int32) -> bool = {\n  g = fn (t) -> string_contains(t, \"a\")\n  g(x)\n}\n",
            "precision mismatch: expected string, got int32",
        ),
        (
            "dict_get",
            "def f(d: Dict[string, int32], k: int32) -> int32 = {\n  g = fn (t) -> dict_get(d, t)\n  g(k)\n}\n",
            "precision mismatch: expected string, got int32",
        ),
        (
            "dict_insert",
            "def f(d: Dict[string, int32], k: int32) -> Dict[string, int32] = {\n  g = fn (t) -> dict_insert(d, t, 1i32)\n  g(k)\n}\n",
            "precision mismatch: expected string, got int32",
        ),
        (
            "expand",
            "def f(x: tensor[1, f32], a: int32) -> tensor[4, f32] = {\n  g = fn (v) -> expand(x, 0i32, v)\n  g(a)\n}\n",
            "but no tensor in scope carries it",
        ),
        (
            "cast",
            "def f(x: List[int32]) -> f32 = {\n  g = fn (t) -> cast(t, f32)\n  g(x)\n}\n",
            "cast requires tensor or prim type",
        ),
        (
            "sum",
            "def f(x: tensor[3, 2, f32], a: int64) -> tensor[2, f32] = {\n  g = fn (v) -> sum(x, v)\n  g(a)\n}\n",
            "is neither a compile-time constant nor a named axis of the operand",
        ),
    ] {
        let errors = check(program).expect_err(&format!(
            "{route}: the program must still be rejected by a later rule"
        ));
        assert!(
            errors.iter().any(|e| e.message.contains(diagnostic)),
            "{route}: the downstream rule's diagnostic must be {diagnostic:?}, got:\n{}",
            summary(&errors)
        );
    }
}

/// The replay runs to a FIXPOINT, and this is why. Suspended calls are
/// replayed in ledger order, so a call whose operand is bound by a LATER
/// replay in the same pass would otherwise still be waiting when the pass
/// ends, and the declaration boundary would report it as never bound.
///
/// Here `g`'s operand is `h`'s result, which is itself a fresh variable until
/// `h`'s own suspended route replays. REGRESSION TEST: one pass leaves `g`
/// unresolved.
#[test]
fn a_chain_of_suspended_calls_settles_in_one_declaration() {
    let chained = "def f(x: tensor[3, 2, f32]) -> tensor[3, 2, f32] = {\n  \
                   g = fn (t) -> permute(t, 1i32, 0i32)\n  \
                   h = fn (u) -> permute(u, 1i32, 0i32)\n  \
                   a = h(x)\n  \
                   g(a)\n}\n";
    check(chained).unwrap_or_else(|e| {
        panic!(
            "a chain of suspended routes must settle, not report an unresolved obligation:\n{}",
            summary(&e)
        )
    });

    // NEGATIVE TWIN: the chain settling is not the same as the chain being
    // unchecked. Both `permute` rules still run, so the wrong declared result
    // is still rejected.
    let wrong = "def f(x: tensor[3, 2, f32]) -> tensor[2, 3, f32] = {\n  \
                 g = fn (t) -> permute(t, 1i32, 0i32)\n  \
                 h = fn (u) -> permute(u, 1i32, 0i32)\n  \
                 a = h(x)\n  \
                 g(a)\n}\n";
    check(wrong).expect_err("two transpositions return the original shape");
}

/// An operand that never binds is reported ONCE. The ledger holds one entry
/// per call, and the declaration boundary drains it once, so a suspended call
/// cannot accumulate a diagnostic per replay attempt.
///
/// REGRESSION TEST for the count, DISPOSITION LOCK for the wording.
#[test]
fn a_never_bound_operand_is_reported_exactly_once() {
    let never = "def f() -> int32 = {\n  g = fn (t) -> permute(t, 1i32, 0i32)\n  1i32\n}\n";
    let errors = check(never).expect_err("a never-bound operand is not a typed program");
    assert_eq!(
        errors.len(),
        1,
        "one suspended call is one diagnostic, got:\n{}",
        summary(&errors)
    );
    assert!(
        errors[0]
            .message
            .contains("unresolved `permute` shape obligation"),
        "the one diagnostic must name the suspended route, got {:?}",
        errors[0].message
    );
}

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

/// A SHAPE route's operand that never acquires an outer constructor is
/// rejected at the declaration boundary rather than accepted. These routes
/// cannot derive a result type at all without a shape, which is the
/// acceptance boundary the shape-computed builtins have carried since
/// chelis#1489.
///
/// The boundary is per route KIND, and the collection, host and string
/// routes deliberately do NOT behave this way; their twin is
/// `a_never_bound_generic_operand_is_accepted_and_validated_at_its_binding`
/// below.
///
/// REGRESSION TEST: before the repair `permute` accepted this.
#[test]
fn an_operand_that_never_binds_is_rejected_at_the_declaration_boundary() {
    for (route, program) in [
        (
            "permute",
            "def f() -> int32 = {\n  g = fn (t) -> permute(t, 1i32, 0i32)\n  1i32\n}\n",
        ),
        (
            "reshape",
            "def f() -> int32 = {\n  g = fn (t) -> reshape(t, [6i64])\n  1i32\n}\n",
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

/// A route whose unresolved operand is not the FIRST one, or whose valid and
/// invalid twins need different declarations, cannot use [`Row`]: that helper
/// substitutes one binding into both calls. These carry their three programs
/// written out instead.
struct Cell {
    route: &'static str,
    /// Resolved control: the rule that was always right, and its exact wording.
    resolved_invalid: &'static str,
    /// The operand binds after the route ran, and the call is invalid.
    late_invalid: &'static str,
    /// The same shape with a valid operand: the repair must not reject it.
    late_valid: &'static str,
    diagnostic: &'static str,
}

fn run_cell(cell: &Cell) {
    let Cell {
        route,
        resolved_invalid,
        late_invalid,
        late_valid,
        diagnostic,
    } = cell;

    // DISPOSITION LOCK. The resolved rejection pins the wording the late-bound
    // one has to equal.
    let eager = check(resolved_invalid).expect_err(&format!(
        "{route}: an invalid resolved call must be rejected"
    ));
    assert!(
        eager.iter().any(|e| e.message.contains(diagnostic)),
        "{route}: the resolved rejection must name its own rule, got:\n{}",
        summary(&eager)
    );

    // REGRESSION TEST. Accepted on `e813415d0` at score 1 with an empty error
    // list; this is the cell chelis#1512 is about.
    let late = check(late_invalid).expect_err(&format!(
        "{route}: an invalid call over a late-bound operand must be rejected"
    ));
    assert!(
        late.iter().any(|e| e.message.contains(diagnostic)),
        "{route}: the late-bound rejection must carry the SAME diagnostic as the resolved one \
         ({diagnostic:?}), got:\n{}",
        summary(&late)
    );

    // NEGATIVE TWIN, and the one that matters for blast radius: suspending the
    // call must not reject the correct program.
    check(late_valid).unwrap_or_else(|e| {
        panic!(
            "{route}: a valid call over a late-bound operand must check:\n{}",
            summary(&e)
        )
    });
}

/// The routes whose unresolved operand is a later argument, or whose rule
/// reads several operands at once. Each was MEASURED accepting its invalid
/// program on `e813415d0`.
#[test]
fn a_late_bound_secondary_operand_is_validated_too() {
    for cell in [
        Cell {
            route: "where",
            resolved_invalid: "def f(c: tensor[3, bool], a: tensor[3, f32], b: tensor[2, f32]) -> tensor[3, f32] = where(c, a, b)\n",
            late_invalid: "def f(c: tensor[3, bool], a: tensor[3, f32], b: tensor[2, f32]) -> tensor[3, f32] = {\n  g = fn (t) -> where(c, t, b)\n  g(a)\n}\n",
            late_valid: "def f(c: tensor[3, bool], a: tensor[3, f32], b: tensor[3, f32]) -> tensor[3, f32] = {\n  g = fn (t) -> where(c, t, b)\n  g(a)\n}\n",
            diagnostic: "where expects cond/both branches to have matching tensor shapes and branch precision",
        },
        Cell {
            route: "clamp",
            resolved_invalid: "def f(x: tensor[3, f32], lo: tensor[2, f32], hi: tensor[3, f32]) -> tensor[3, f32] = clamp(x, lo, hi)\n",
            late_invalid: "def f(x: tensor[3, f32], lo: tensor[2, f32], hi: tensor[3, f32]) -> tensor[3, f32] = {\n  g = fn (t) -> clamp(t, lo, hi)\n  g(x)\n}\n",
            late_valid: "def f(x: tensor[3, f32], lo: tensor[3, f32], hi: tensor[3, f32]) -> tensor[3, f32] = {\n  g = fn (t) -> clamp(t, lo, hi)\n  g(x)\n}\n",
            diagnostic: "clamp expects tensor input plus scalar-tensor or matching-shape tensor bounds of the same precision",
        },
        Cell {
            route: "index",
            resolved_invalid: "def f(xs: List[int32], k: f32) -> int32 = index(xs, k)\n",
            late_invalid: "def f(xs: List[int32], k: f32) -> int32 = {\n  g = fn (n) -> index(xs, n)\n  g(k)\n}\n",
            late_valid: "def f(xs: List[int32], k: int64) -> int32 = {\n  g = fn (n) -> index(xs, n)\n  g(k)\n}\n",
            diagnostic: "index expects integer index, got f32",
        },
        Cell {
            // `concat`'s List/List rule IS a unification of the element types,
            // so its own wording is the precision mismatch. Draft #1690
            // rewrites the SIBLING tensor-concat branch of this same match and
            // leaves this arm alone.
            route: "concat",
            resolved_invalid: "def f(x: List[int32], y: List[f32]) -> List[int32] = concat(x, y)\n",
            late_invalid: "def f(x: List[int32], y: List[f32]) -> List[int32] = {\n  g = fn (t) -> concat(x, t)\n  g(y)\n}\n",
            late_valid: "def f(x: List[int32], y: List[int32]) -> List[int32] = {\n  g = fn (t) -> concat(x, t)\n  g(y)\n}\n",
            diagnostic: "precision mismatch: expected int32, got f32",
        },
        Cell {
            route: "zip",
            resolved_invalid: "def f(x: List[int32], y: int32) -> List[(int32, int32)] = zip(x, y)\n",
            late_invalid: "def f(x: List[int32], y: int32) -> List[(int32, int32)] = {\n  g = fn (t) -> zip(x, t)\n  g(y)\n}\n",
            late_valid: "def f(x: List[int32], y: List[int32]) -> List[(int32, int32)] = {\n  g = fn (t) -> zip(x, t)\n  g(y)\n}\n",
            diagnostic: "zip expects List inputs, got List int32 and int32",
        },
        Cell {
            route: "dict_merge",
            resolved_invalid: "def f(d: Dict[string, int32], e: Dict[string, f32]) -> Dict[string, int32] = dict_merge(d, e)\n",
            late_invalid: "def f(d: Dict[string, int32], e: Dict[string, f32]) -> Dict[string, int32] = {\n  g = fn (t) -> dict_merge(d, t)\n  g(e)\n}\n",
            late_valid: "def f(d: Dict[string, int32], e: Dict[string, int32]) -> Dict[string, int32] = {\n  g = fn (t) -> dict_merge(d, t)\n  g(e)\n}\n",
            diagnostic: "precision mismatch: expected int32, got f32",
        },
        Cell {
            route: "split sizes",
            resolved_invalid: "def f(x: tensor[4, f32], s: List[f32]) -> List[tensor[2, f32]] = split(x, 0i32, s)\n",
            late_invalid: "def f(x: tensor[4, f32], s: List[f32]) -> List[tensor[2, f32]] = {\n  g = fn (t) -> split(x, 0i32, t)\n  g(s)\n}\n",
            late_valid: "def f(x: tensor[4, f32], s: List[int64]) -> List[tensor[2, f32]] = {\n  g = fn (t) -> split(x, 0i32, t)\n  g(s)\n}\n",
            diagnostic: "split expects List[int] sizes",
        },
        Cell {
            route: "split input",
            resolved_invalid: "def f(x: List[int32], s: List[int64]) -> List[tensor[2, f32]] = split(x, 0i32, s)\n",
            late_invalid: "def f(x: List[int32], s: List[int64]) -> List[tensor[2, f32]] = {\n  g = fn (t) -> split(t, 0i32, s)\n  g(x)\n}\n",
            late_valid: "def f(x: tensor[4, f32], s: List[int64]) -> List[tensor[2, f32]] = {\n  g = fn (t) -> split(t, 0i32, s)\n  g(x)\n}\n",
            diagnostic: "split expects tensor input and List[int] sizes",
        },
        Cell {
            route: "take",
            resolved_invalid: "def f(x: List[int32], k: f32) -> List[int32] = take(x, k)\n",
            late_invalid: "def f(x: List[int32], k: f32) -> List[int32] = {\n  g = fn (n) -> take(x, n)\n  g(k)\n}\n",
            late_valid: "def f(x: List[int32], k: int64) -> List[int32] = {\n  g = fn (n) -> take(x, n)\n  g(k)\n}\n",
            diagnostic: "take expects integer count, got f32",
        },
        Cell {
            // `drop` shares `take`'s arm; the cell is here because the
            // diagnostic interpolates the callee name and a shared arm that
            // named one of them would pass with the other silently wrong.
            route: "drop",
            resolved_invalid: "def f(x: List[int32], k: f32) -> List[int32] = drop(x, k)\n",
            late_invalid: "def f(x: List[int32], k: f32) -> List[int32] = {\n  g = fn (n) -> drop(x, n)\n  g(k)\n}\n",
            late_valid: "def f(x: List[int32], k: int64) -> List[int32] = {\n  g = fn (n) -> drop(x, n)\n  g(k)\n}\n",
            diagnostic: "drop expects integer count, got f32",
        },
        Cell {
            route: "chunk",
            resolved_invalid: "def f(x: List[int32], k: f32) -> List[List[int32]] = chunk(x, k)\n",
            late_invalid: "def f(x: List[int32], k: f32) -> List[List[int32]] = {\n  g = fn (n) -> chunk(x, n)\n  g(k)\n}\n",
            late_valid: "def f(x: List[int32], k: int64) -> List[List[int32]] = {\n  g = fn (n) -> chunk(x, n)\n  g(k)\n}\n",
            diagnostic: "chunk expects integer size, got f32",
        },
        Cell {
            // The window family suspended on its TENSOR operand only, so an
            // axis or step that bound late slipped past the route's own guard
            // while the tensor was already settled.
            route: "permute axis",
            resolved_invalid: "def f(x: tensor[3, 2, f32], a: int64) -> tensor[2, 3, f32] = permute(x, a, 0i32)\n",
            late_invalid: "def f(x: tensor[3, 2, f32], a: int64) -> tensor[2, 3, f32] = {\n  g = fn (v) -> permute(x, v, 0i32)\n  g(a)\n}\n",
            late_valid: "def f(x: tensor[3, 2, f32]) -> tensor[2, 3, f32] = {\n  g = fn (t) -> permute(t, 1i32, 0i32)\n  g(x)\n}\n",
            diagnostic: "permute expects int32 axis indices, got int64",
        },
        Cell {
            route: "stride step",
            resolved_invalid: "def f(x: tensor[2, 4, f32], a: int32) -> tensor[2, 2, f32] = stride(x, a, 2i64)\n",
            late_invalid: "def f(x: tensor[2, 4, f32], a: int32) -> tensor[2, 2, f32] = {\n  g = fn (v) -> stride(x, v, 2i64)\n  g(a)\n}\n",
            late_valid: "def f(x: tensor[2, 4, f32]) -> tensor[2, 2, f32] = {\n  g = fn (t) -> stride(t, 1i64, 2i64)\n  g(x)\n}\n",
            diagnostic: "stride expects int64 strides (write 2i64), got int32",
        },
    ] {
        run_cell(&cell);
    }
}

/// The other half of the per-route-kind boundary rule, and the reason it has
/// to be per kind at all.
///
/// A collection, host or string route's operand is NOT required to be a
/// tensor carrying a shape. A polymorphic definition legitimately calls
/// `dict_of`, `len` or `where` over a value that acquires its constructor
/// only at a call site, and `packages/chelis-std/src/io/json.ch` really does:
/// `parse_object` returns `Some((dict_of([]), ...))` under a declared
/// `Dict[string, Json]`, where the empty list literal's element type never
/// binds inside the declaration. Rejecting that at the boundary took the
/// whole standard library out.
///
/// So the obligation is discharged silently, and the route's own validation
/// runs at each binding instead. REGRESSION TEST for the acceptance halves,
/// which a boundary rule that did not distinguish route kinds rejected;
/// DISPOSITION LOCK for the rejection, which is the same cell the
/// collection-route table above already asserts.
#[test]
fn a_never_bound_generic_operand_is_accepted_and_validated_at_its_binding() {
    // Never applied: accepted, with no diagnostic at all.
    for (route, program) in [
        (
            "len",
            "def f() -> int32 = {\n  g = fn (t) -> len(t)\n  1i32\n}\n",
        ),
        (
            "where",
            "def f(c: tensor[3, bool], b: tensor[3, f32]) -> int32 = {\n  g = fn (t) -> where(c, t, b)\n  1i32\n}\n",
        ),
        (
            "index",
            "def f(xs: List[int32]) -> int32 = {\n  g = fn (n) -> index(xs, n)\n  1i32\n}\n",
        ),
    ] {
        check(program).unwrap_or_else(|e| {
            panic!(
                "{route}: a generic operand that never binds must be accepted, not reported:\n{}",
                summary(&e)
            )
        });
    }

    // The standard library's own shape, reduced to one declaration.
    check("def f() -> Dict[string, int32] = dict_of([])\n").unwrap_or_else(|e| {
        panic!(
            "dict_of: the declared result decides an empty literal's element type, and the \
             operand never binds:\n{}",
            summary(&e)
        )
    });

    // Applied once, validly: still accepted, and the route ran.
    check("def f(x: List[int32]) -> int64 = {\n  g = fn (t) -> len(t)\n  g(x)\n}\n")
        .unwrap_or_else(|e| panic!("len: a valid binding must check:\n{}", summary(&e)));

    // Applied once, invalidly: rejected with the ROUTE's own text. Accepting
    // the never-bound case is not the same as not checking; this is the cell
    // that separates the two.
    let errors =
        check("def f(x: tensor[3, f32]) -> int64 = {\n  g = fn (t) -> len(t)\n  g(x)\n}\n")
            .expect_err("len over a tensor must be rejected once the operand binds");
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("len expects List or Dict input")),
        "the binding must be validated by `len`'s own rule, got:\n{}",
        summary(&errors)
    );
}

/// A route can suspend on one operand and reject on ANOTHER in the same eager
/// pass. The replay re-enters the whole route, so without cancelling the
/// suspension that rejection is reported twice: the call is counted as two
/// failures, and the machine-facing `types` score moves with it.
///
/// REGRESSION TEST, round 1 P2-1. On the reviewed head each of these reported
/// its diagnostic TWICE; on `e813415d0`, which suspends nothing, each reports
/// once. Asserting the COUNT is the point: cell 4 of every table above uses
/// `.any(...)` and passes either way, which is exactly why nothing caught it.
#[test]
fn a_call_that_suspends_and_then_rejects_eagerly_reports_once() {
    for (route, program, diagnostic) in [
        (
            "string_slice",
            "def f(x: string) -> string = {\n  g = fn (t) -> string_slice(t, 0i64, 1.5f32)\n  g(x)\n}\n",
            "string_slice expects integer index arguments",
        ),
        (
            "string_concat",
            "def f(x: string) -> string = {\n  g = fn (t) -> string_concat(t, 5i64)\n  g(x)\n}\n",
            "string_concat expects string arguments",
        ),
    ] {
        let errors = check(program).expect_err(&format!("{route}: the call must be rejected"));
        let hits = errors
            .iter()
            .filter(|e| e.message.contains(diagnostic))
            .count();
        assert_eq!(
            hits,
            1,
            "{route}: the eager rejection must be reported once, not re-reported by the replay; \
             got {hits} copies in:\n{}",
            summary(&errors)
        );
    }

    // NEGATIVE TWIN. Cancelling a failed call's suspension must not cancel the
    // validation of a call that did NOT fail eagerly: the same route, invalid
    // only in the operand that binds late, still rejects.
    let errors = check(
        "def f(x: int32) -> string = {\n  g = fn (t) -> string_slice(t, 0i64, 1i64)\n  g(x)\n}\n",
    )
    .expect_err("string_slice over a late-bound non-string must be rejected");
    assert!(
        !errors.is_empty(),
        "the late-bound operand's own validation must still run"
    );
}

/// A dtype-admissibility route's four cells.
///
/// [`Cell`] above carries three of them. What makes these calls valid or
/// invalid is the OPERAND's own dtype rather than one of the call's other
/// arguments, so the declaration differs between the valid and invalid
/// programs and the fourth cell has to name its own program instead of
/// reusing the invalid one's declaration.
struct DtypeCell {
    cell: Cell,
    /// Resolved control on the accepting side: the route admits this dtype.
    resolved_valid: &'static str,
}

fn run_dtype_cell(row: &DtypeCell) {
    // Cell 1 (resolved, valid): DISPOSITION LOCK. Green before the repair; it
    // is here so a repair that rejects admissible dtypes is caught.
    check(row.resolved_valid).unwrap_or_else(|e| {
        panic!(
            "{}: a valid resolved call must check:\n{}",
            row.cell.route,
            summary(&e)
        )
    });
    run_cell(&row.cell);
}

/// chelis#1512's remaining class: the dtype-admissibility validators.
///
/// These three run at the head of `finish_unified_app`, before the route
/// dispatch that owns the deferral site, and each admitted a `Type::Var`
/// operand and walked away. Every `late_invalid` program below was MEASURED
/// accepted at score 1 with an empty error list on `6dbbbf2bc`, and every
/// `resolved_invalid` twin rejected there with the diagnostic named.
///
/// REGRESSION TEST for the `late_invalid` cell of every row; DISPOSITION LOCK
/// for the two valid cells.
#[test]
fn dtype_admissibility_validates_a_late_bound_operand() {
    for row in [
        // `operand_dtype_rejection`, reached from the TENSOR_OPS loop: the
        // transcendentals are float-only on a host scalar too.
        DtypeCell {
            cell: Cell {
                route: "sqrt",
                resolved_invalid: "def f(x: int32) -> int32 = sqrt(x)\n",
                late_invalid: "def f(x: int32) -> int32 = {\n  g = fn (t) -> sqrt(t)\n  g(x)\n}\n",
                late_valid: "def f(x: f32) -> f32 = {\n  g = fn (t) -> sqrt(t)\n  g(x)\n}\n",
                diagnostic: "sqrt does not accept argument type int32 in this context",
            },
            resolved_valid: "def f(x: f32) -> f32 = sqrt(x)\n",
        },
        // The logical operations are bool-only, and the arithmetic ones are
        // exactly not-bool ([04-NUM-4]). Both directions, so a repair cannot
        // satisfy one by widening the other.
        DtypeCell {
            cell: Cell {
                route: "and",
                resolved_invalid: "def f(x: int32) -> int32 = and(x, x)\n",
                late_invalid: "def f(x: int32) -> int32 = {\n  g = fn (t) -> and(t, t)\n  g(x)\n}\n",
                late_valid: "def f(x: bool) -> bool = {\n  g = fn (t) -> and(t, t)\n  g(x)\n}\n",
                diagnostic: "and does not accept argument type int32 in this context",
            },
            resolved_valid: "def f(x: bool) -> bool = and(x, x)\n",
        },
        DtypeCell {
            cell: Cell {
                route: "add",
                resolved_invalid: "def f(x: bool) -> bool = add(x, x)\n",
                late_invalid: "def f(x: bool) -> bool = {\n  g = fn (t) -> add(t, t)\n  g(x)\n}\n",
                late_valid: "def f(x: int32) -> int32 = {\n  g = fn (t) -> add(t, t)\n  g(x)\n}\n",
                diagnostic: "add on bool operands is not admitted",
            },
            resolved_valid: "def f(x: int32) -> int32 = add(x, x)\n",
        },
        // The reduction family's own dtype rule, reached from the first-argument
        // arm of `validate_numeric_and_reduction_arguments`.
        DtypeCell {
            cell: Cell {
                route: "mean",
                resolved_invalid:
                    "def f(x: tensor[3, int32]) -> tensor[int32] = mean(x, 0i32)\n",
                late_invalid: "def f(x: tensor[3, int32]) -> tensor[int32] = {\n  g = fn (t) -> mean(t, 0i32)\n  g(x)\n}\n",
                late_valid: "def f(x: tensor[3, f32]) -> tensor[f32] = {\n  g = fn (t) -> mean(t, 0i32)\n  g(x)\n}\n",
                diagnostic: "mean on operand precision `int32` is not admitted",
            },
            resolved_valid: "def f(x: tensor[3, f32]) -> tensor[f32] = mean(x, 0i32)\n",
        },
        DtypeCell {
            cell: Cell {
                route: "softmax",
                resolved_invalid:
                    "def f(x: tensor[3, int32]) -> tensor[3, int32] = softmax(x, 0i32)\n",
                late_invalid: "def f(x: tensor[3, int32]) -> tensor[3, int32] = {\n  g = fn (t) -> softmax(t, 0i32)\n  g(x)\n}\n",
                late_valid: "def f(x: tensor[3, f32]) -> tensor[3, f32] = {\n  g = fn (t) -> softmax(t, 0i32)\n  g(x)\n}\n",
                diagnostic: "softmax on operand precision `int32` is not admitted",
            },
            resolved_valid: "def f(x: tensor[3, f32]) -> tensor[3, f32] = softmax(x, 0i32)\n",
        },
        // `reject_inadmissible_operand_dtypes`. `uniform_like`'s template
        // parameter is a bare type variable in the builtin scheme, so a
        // late-bound operand REACHES the route still unresolved rather than
        // being bound by signature unification first.
        DtypeCell {
            cell: Cell {
                route: "uniform_like (non-float template)",
                resolved_invalid: "def f(x: tensor[3, int32]) -> tensor[3, int32] ! {Random} = uniform_like(x, 0.0f32, 1.0f32)\n",
                late_invalid: "def f(x: tensor[3, int32]) -> tensor[3, int32] ! {Random} = {\n  g = fn (t) -> uniform_like(t, 0.0f32, 1.0f32)\n  g(x)\n}\n",
                late_valid: "def f(x: tensor[3, f32]) -> tensor[3, f32] ! {Random} = {\n  g = fn (t) -> uniform_like(t, 0.0f32, 1.0f32)\n  g(x)\n}\n",
                diagnostic: "uniform_like expects a float tensor template",
            },
            resolved_valid: "def f(x: tensor[3, f32]) -> tensor[3, f32] ! {Random} = uniform_like(x, 0.0f32, 1.0f32)\n",
        },
        DtypeCell {
            cell: Cell {
                route: "uniform_like (non-tensor template)",
                resolved_invalid:
                    "def f(x: int32) -> int32 ! {Random} = uniform_like(x, 0.0f32, 1.0f32)\n",
                late_invalid: "def f(x: int32) -> int32 ! {Random} = {\n  g = fn (t) -> uniform_like(t, 0.0f32, 1.0f32)\n  g(x)\n}\n",
                late_valid: "def f(x: tensor[3, f32]) -> tensor[3, f32] ! {Random} = {\n  g = fn (t) -> uniform_like(t, 0.0f32, 1.0f32)\n  g(x)\n}\n",
                diagnostic: "uniform_like expects tensor template input",
            },
            resolved_valid: "def f(x: tensor[3, f32]) -> tensor[3, f32] ! {Random} = uniform_like(x, 0.0f32, 1.0f32)\n",
        },
        // `integer_binop_result_type`: the binary operators decide the call's
        // RESULT type, not only its admissibility, and their operands are three
        // independent type variables in the builtin scheme.
        DtypeCell {
            cell: Cell {
                route: "mod",
                resolved_invalid: "def f(x: int64) -> int64 = mod(x, 3i32)\n",
                late_invalid: "def f(x: int64) -> int64 = {\n  g = fn (t) -> mod(t, 3i32)\n  g(x)\n}\n",
                late_valid: "def f(x: int32) -> int32 = {\n  g = fn (t) -> mod(t, 3i32)\n  g(x)\n}\n",
                diagnostic: "mod requires matching integer arguments, got int64 and int32",
            },
            resolved_valid: "def f(x: int32) -> int32 = mod(x, 3i32)\n",
        },
        DtypeCell {
            cell: Cell {
                route: "bitand",
                resolved_invalid: "def f(x: int64) -> int64 = bitand(x, 3i32)\n",
                late_invalid: "def f(x: int64) -> int64 = {\n  g = fn (t) -> bitand(t, 3i32)\n  g(x)\n}\n",
                late_valid: "def f(x: int32) -> int32 = {\n  g = fn (t) -> bitand(t, 3i32)\n  g(x)\n}\n",
                diagnostic: "bitand requires matching integer arguments, got int64 and int32",
            },
            resolved_valid: "def f(x: int32) -> int32 = bitand(x, 3i32)\n",
        },
        // The shift operators, whose admissibility was two boolean
        // disjunctions rather than match arms until this repair.
        DtypeCell {
            cell: Cell {
                route: "shl",
                resolved_invalid: "def f(x: f32) -> f32 = shl(x, 1i32)\n",
                late_invalid: "def f(x: f32) -> f32 = {\n  g = fn (t) -> shl(t, 1i32)\n  g(x)\n}\n",
                late_valid: "def f(x: int32) -> int32 = {\n  g = fn (t) -> shl(t, 1i32)\n  g(x)\n}\n",
                diagnostic: "shl requires integer lhs and shift amount, got f32 and int32",
            },
            resolved_valid: "def f(x: int32) -> int32 = shl(x, 1i32)\n",
        },
        DtypeCell {
            cell: Cell {
                route: "shr",
                resolved_invalid: "def f(x: f32) -> f32 = shr(x, 1i32)\n",
                late_invalid: "def f(x: f32) -> f32 = {\n  g = fn (t) -> shr(t, 1i32)\n  g(x)\n}\n",
                late_valid: "def f(x: int32) -> int32 = {\n  g = fn (t) -> shr(t, 1i32)\n  g(x)\n}\n",
                diagnostic: "shr requires integer lhs and shift amount, got f32 and int32",
            },
            resolved_valid: "def f(x: int32) -> int32 = shr(x, 1i32)\n",
        },
    ] {
        run_dtype_cell(&row);
    }
}

/// The dtype half of the per-route-kind acceptance boundary.
///
/// A dtype-admissibility validator's operand is not required to carry a shape,
/// so a lambda that is never applied keeps its unconstrained result and is
/// accepted in silence, exactly as the collection, host and string routes are.
/// The validation is not lost: it runs at each binding instead, which is the
/// cell the table above asserts.
///
/// DISPOSITION LOCK: every program here was accepted on `6dbbbf2bc` and must
/// stay accepted. It is the cell that took the standard library out when a
/// first attempt at this repair rejected an operand that never binds.
#[test]
fn a_never_bound_dtype_operand_is_accepted() {
    for (route, program) in [
        (
            "sqrt",
            "def f() -> int32 = {\n  g = fn (t) -> sqrt(t)\n  1i32\n}\n",
        ),
        (
            "mod",
            "def f() -> int32 = {\n  g = fn (t) -> mod(t, 3i32)\n  1i32\n}\n",
        ),
        (
            "shl",
            "def f() -> int32 = {\n  g = fn (t) -> shl(t, 1i32)\n  1i32\n}\n",
        ),
        (
            "uniform_like",
            "def f() -> int32 = {\n  g = fn (t) -> uniform_like(t, 0.0f32, 1.0f32)\n  1i32\n}\n",
        ),
    ] {
        check(program).unwrap_or_else(|e| {
            panic!(
                "{route}: a dtype operand that never binds must be accepted, not reported:\n{}",
                summary(&e)
            )
        });
    }
}

/// An error operand keeps its early return here too: the upstream failure is
/// reported once and the dtype validator adds nothing on top of it.
///
/// A suspended call whose sibling operand is an error witness is the hazard:
/// the witness never binds, the readiness predicate ignores it, and a replay
/// would print the route's own diagnostic against a type the program never
/// really had. DISPOSITION LOCK on `6dbbbf2bc`'s verdict, and the negative
/// twin of every cell in the table above.
#[test]
fn an_error_operand_suppresses_the_dtype_routes_own_diagnostic() {
    for (route, program) in [
        ("sqrt", "def f(x: f32) -> f32 = sqrt(nope(x))\n"),
        ("mod", "def f(x: int32) -> int32 = mod(nope(x), 3i32)\n"),
        ("shl", "def f(x: int32) -> int32 = shl(nope(x), 1i32)\n"),
        (
            "add",
            "def f(x: int32) -> int32 = {\n  g = fn (t) -> add(t, nope(x))\n  g(x)\n}\n",
        ),
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

/// A dtype validator that suspends on one operand and then rejects on another
/// in the same eager pass reports that rejection ONCE.
///
/// `mod(t, 3.0f32)` reaches the rejecting arm while `t` is still a variable,
/// so the call both registers a suspension and fails, and a replay would
/// re-run the same validator against the same argument list. Asserting the
/// COUNT is the point: every cell above uses `.any(...)` and would pass either
/// way. REGRESSION TEST against the shape chelis#1512's round 1 measured on
/// the string routes.
#[test]
fn a_dtype_call_that_suspends_and_then_rejects_eagerly_reports_once() {
    let program = "def f(x: int32) -> int32 = {\n  g = fn (t) -> mod(t, 3.0f32)\n  g(x)\n}\n";
    let errors = check(program).expect_err("mod over a float shift amount must be rejected");
    let hits = errors
        .iter()
        .filter(|e| e.message.contains("mod requires matching integer arguments"))
        .count();
    assert_eq!(
        hits, 1,
        "the eager rejection must be reported once, not re-reported by the replay; \
         got {hits} copies in:\n{}",
        summary(&errors)
    );

    // NEGATIVE TWIN. Cancelling a failed call's suspension must not cancel a
    // call that did not fail eagerly: the same operator, invalid only in the
    // operand that binds late, still rejects.
    let errors = check("def f(x: int64) -> int64 = {\n  g = fn (t) -> mod(t, 3i32)\n  g(x)\n}\n")
        .expect_err("mod over a late-bound int64 must be rejected");
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("mod requires matching integer arguments")),
        "the late-bound operand's own validation must still run:\n{}",
        summary(&errors)
    );
}

/// The dtype routes whose late-bound form was ALREADY rejected before this
/// repair, each by a rule that runs ahead of the validator.
///
/// `trunc_div`, `div`, `dropout` and `test_assert_close_tensor` all declare a
/// concrete or restricted operand in the builtin scheme, so signature
/// unification binds the operand before `finish_unified_app` runs and the
/// validator never sees a variable. `sum`'s late-bound axis is caught by the
/// reduction's own axis rule.
///
/// DISPOSITION LOCK, not a regression test: every verdict here was measured on
/// `6dbbbf2bc` and must not move. Suspending a decision relocates it, and these
/// are the programs where an over-broad suspension would silently stop
/// rejecting.
#[test]
fn dtype_routes_caught_before_the_validator_keep_their_verdict() {
    for (route, program, diagnostic) in [
        (
            "trunc_div",
            "def f(x: f32) -> f32 = {\n  g = fn (t) -> trunc_div(t, 2.0f32)\n  g(x)\n}\n",
            "trunc_div on float operand precision `f32` is not admitted",
        ),
        (
            "div",
            "def f(x: int32) -> int32 = {\n  g = fn (t) -> div(t, 2i32)\n  g(x)\n}\n",
            "div on integer operand precision `int32` is not admitted",
        ),
        (
            "sum (non-tensor operand)",
            "def f(x: int32) -> int32 = {\n  g = fn (t) -> sum(t, 0i32)\n  g(x)\n}\n",
            "sum expects tensor input, got int32",
        ),
        (
            "dropout",
            "def f(x: tensor[3, int32]) -> tensor[3, int32] ! {Random} = {\n  g = fn (t) -> dropout(t, 0.5f32)\n  g(x)\n}\n",
            "tensor precision mismatch: f32 vs int32",
        ),
        (
            "test_assert_close_tensor",
            "def test_a(x: tensor[3, f32]) -> unit ! {Test} = {\n  g = fn (t) -> test_assert_close_tensor(t, t, 0.01f64, \"m\")\n  g(x)\n}\n",
            "tensor precision mismatch: f64 vs f32",
        ),
        (
            "sum (late-bound axis)",
            "def f(x: f32) -> tensor[f32] = {\n  g = fn (a) -> sum(to_tensor([1.0f32, 2.0f32, 3.0f32]), a)\n  g(x)\n}\n",
            "is neither a compile-time constant nor a named axis",
        ),
    ] {
        let errors = check(program).expect_err(&format!("{route}: this program must be rejected"));
        assert!(
            errors.iter().any(|e| e.message.contains(diagnostic)),
            "{route}: the verdict must not move, expected {diagnostic:?}, got:\n{}",
            summary(&errors)
        );
    }
}

/// The residual half of this class, tracked as chelis#1805: a tensor operand
/// whose PRECISION resolves after the route runs.
///
/// The suspension above waits on `shape_operand_awaits_binding`, which answers
/// true for `Type::Var` and `Type::Ref` alone. A `Type::Tensor` already carries
/// its outer constructor, so the ledger treats it as ready the moment it is
/// seen, whatever its precision variable still holds, and this ledger therefore
/// cannot carry the case. Repairing it needs either a readiness predicate that
/// also waits on a free `TensorPrec::Var`, which changes when every existing
/// entry replays, or a second ledger: a design decision rather than an
/// extension of the arm split, which is why chelis#1805 owns it.
///
/// NOT a disposition lock on desired behaviour. It pins a KNOWN HOLE so the
/// suite says out loud what this pull request does not close, and it is the
/// test chelis#1805 turns red on when it is repaired. Read a failure here as
/// that repair landing, and move the assertion to the rejecting side.
#[test]
fn a_late_bound_tensor_precision_is_not_validated_yet() {
    // The precision of the empty literal is a free `TensorPrec::Var` when the
    // reduction's dtype rule reads it; the declared result binds it to `int32`
    // afterwards, and nothing re-consults the policy.
    check("def f() -> tensor[int32] = mean(to_tensor([]), 0i32)\n").expect(
        "chelis#1805: an integer `mean` over a late-bound precision is still accepted; if this \
         now rejects, that issue is fixed and this assertion moves",
    );

    // The control, and the reason the row above is a hole rather than a policy:
    // the same operand with its precision settled first is rejected.
    let errors = check("def f(x: tensor[3, int32]) -> tensor[int32] = mean(x, 0i32)\n")
        .expect_err("a settled integer operand must be rejected");
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("mean on operand precision `int32` is not admitted")),
        "the resolved rejection must still name the dtype policy:\n{}",
        summary(&errors)
    );

    // The neighbouring route is NOT exposed, which is what makes chelis#1805 a
    // bounded residual rather than a blanket one: `softmax` preserves its
    // operand's precision, so signature unification binds it before the
    // validator runs.
    let errors = check("def f() -> tensor[3, int32] = softmax(to_tensor([]), 0i32)\n")
        .expect_err("softmax binds its operand precision through unification first");
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("softmax on operand precision `int32`")),
        "softmax must still reject through the same policy:\n{}",
        summary(&errors)
    );
}

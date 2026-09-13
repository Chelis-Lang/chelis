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

    // Round 1 P3-1. The shared harness above matches the diagnostic by
    // SUBSTRING, so a drift in the message tail, or a second diagnostic
    // appearing beside it, would pass. These routes relocate a decision rather
    // than restate it, so the two renderings are compared whole: same count,
    // same kind, same message. Measured equal on all eleven routes.
    let eager = check(row.cell.resolved_invalid).expect_err("checked by run_cell");
    let late = check(row.cell.late_invalid).expect_err("checked by run_cell");
    assert_eq!(
        summary(&eager),
        summary(&late),
        "{}: the late-bound call must produce the SAME diagnostics as the resolved one, whole \
         and in the same number, not merely a superset containing the fragment",
        row.cell.route
    );
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
                resolved_invalid: "def f(x: tensor[3, int32]) -> tensor[int32] = mean(x, 0i32)\n",
                late_invalid: "def f(x: tensor[3, int32]) -> tensor[int32] = {\n  g = fn (t) -> mean(t, 0i32)\n  g(x)\n}\n",
                late_valid: "def f(x: tensor[3, f32]) -> tensor[f32] = {\n  g = fn (t) -> mean(t, 0i32)\n  g(x)\n}\n",
                diagnostic: "mean on operand precision `int32` is not admitted",
            },
            resolved_valid: "def f(x: tensor[3, f32]) -> tensor[f32] = mean(x, 0i32)\n",
        },
        DtypeCell {
            cell: Cell {
                route: "softmax",
                resolved_invalid: "def f(x: tensor[3, int32]) -> tensor[3, int32] = softmax(x, 0i32)\n",
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
                resolved_invalid: "def f(x: int32) -> int32 ! {Random} = uniform_like(x, 0.0f32, 1.0f32)\n",
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
        .filter(|e| {
            e.message
                .contains("mod requires matching integer arguments")
        })
        .count();
    assert_eq!(
        hits,
        1,
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
        errors.iter().any(|e| e
            .message
            .contains("mod requires matching integer arguments")),
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

/// chelis#1805: a tensor operand whose PRECISION resolves after the route runs.
///
/// [04-INF-1] keeps a declared precision variable polymorphic, so
/// `def g[p](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)` is a legitimate
/// declaration and the defect is an INSTANTIATION at a dtype the family
/// excludes. `validate.rs::check_app_for_poly_op_constraint` already enforced
/// that, and already rejected `g(y)` for a `y` the enclosing definition
/// declares. It gave up whenever the argument's precision was not readable off
/// an annotation or that parameter scope, which is the hole: the three
/// spellings below reached the backend and the compiled C lane printed `f = 2`
/// for a true 2.333.
///
/// REGRESSION TEST, every rejecting row: each was measured ACCEPTED at score 1
/// on `0820ee28e`, with the `f = 2` C output recorded on that head. DISPOSITION
/// LOCK for the two accepting rows, measured accepted there and here: they are
/// what keeps the repair from becoming a declaration-time bound.
#[test]
fn a_late_bound_tensor_precision_is_rejected_at_the_instantiation() {
    const HELPER: &str = "def g[p](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n";
    const INT_TEXT: &str = "mean on operand precision `int32` is not admitted per the \
                            chelis#724 capability decision";

    for (label, program) in [
        (
            "a to_tensor literal argument",
            format!("{HELPER}f = g(to_tensor([1i32, 2i32, 4i32]))\n"),
        ),
        (
            "a nested call's declared result",
            format!(
                "{HELPER}def h() -> tensor[3, int32] = to_tensor([1i32, 2i32, 4i32])\n\
                 f = g(h())\n"
            ),
        ),
        (
            "a top-level value binding",
            format!("{HELPER}t = to_tensor([1i32, 2i32, 4i32])\nf = g(t)\n"),
        ),
        (
            "a tuple element of a declared parameter",
            format!(
                "{HELPER}def pick(pair: (tensor[3, int32], int32)) -> tensor[int32] = g(pair.0)\n\
                 f = pick((to_tensor([1i32, 2i32, 4i32]), 7i32))\n"
            ),
        ),
    ] {
        let Err(errors) = check(&program) else {
            panic!("{label}: the int32 instantiation must be rejected");
        };
        assert!(
            errors.iter().any(|e| e.message.contains(INT_TEXT)),
            "{label}: the rejection must name the instantiated dtype:\n{}",
            summary(&errors)
        );
    }

    // The declaration on its own stays polymorphic, and so does an f32
    // instantiation. Rejecting either is the over-rejection chelis#1805's
    // design proposed and this repair declined; `separate_sig_sdpa_checks_clean`
    // in `issue_773_arg_error_sibling_checking` is the program that settles it.
    check(HELPER).expect("an unbounded precision binder is a legitimate declaration");
    check(&format!(
        "{HELPER}f = g(to_tensor([1.0f32, 2.0f32, 4.0f32]))\n"
    ))
    .expect("an f32 instantiation of the same helper is well typed");
    check(&format!(
        "{HELPER}def pick(pair: (tensor[3, f32], int32)) -> tensor[f32] = g(pair.0)\n\
         f = pick((to_tensor([1.0f32, 2.0f32, 4.0f32]), 7i32))\n"
    ))
    .expect("the same tuple projection at f32 is admitted");
}

/// chelis#1805: the operand's constructor is known, its precision binds one
/// application later, and the decision is DEFERRED rather than taken at the
/// `mean` node.
///
/// `id_dt` gives `mean` a `tensor[3, ?p]` whose precision the lambda's own
/// application binds afterwards. Rejecting at the `mean` node would refuse the
/// float call below, which is legal; admitting for good is chelis#1805. The
/// ledger entry waits on the precision and the replay decides against the bound
/// one, which is why the integer call gets the ordinary CONCRETE diagnostic.
///
/// This is the half no call-site pass can reach: the lambda has no name for
/// `check_app_for_poly_op_constraint` to key on.
///
/// REGRESSION TEST for the integer half, measured ACCEPTED at score 1 on
/// `0820ee28e`. DISPOSITION LOCK for the float half, measured accepted there and
/// here: it is what proves the repair defers instead of rejecting eagerly.
#[test]
fn a_precision_that_binds_one_application_later_is_decided_on_binding() {
    const HELPER: &str = "def id_dt[p](x: tensor[3, p]) -> tensor[3, p] = x\n";

    let errors = check(&format!(
        "{HELPER}def f() -> tensor[int32] = \
         (fn (v) -> mean(id_dt(v), 0i32))(to_tensor([1i32, 2i32, 4i32]))\n"
    ))
    .expect_err("a precision that binds to int32 one application later must be rejected");
    assert!(
        errors.iter().any(|e| e.message.contains(
            "mean on operand precision `int32` is not admitted per the chelis#724 \
             capability decision"
        )),
        "the discharged rejection names the BOUND dtype, not the binder:\n{}",
        summary(&errors)
    );

    check(&format!(
        "{HELPER}def f() -> tensor[f32] = \
         (fn (v) -> mean(id_dt(v), 0i32))(to_tensor([1.0f32, 2.0f32, 4.0f32]))\n"
    ))
    .expect("a precision that binds to f32 one application later must be accepted");
}

/// chelis#1805: a binder that DOES declare a dtype-family bound is decided by
/// that family where the route runs, with no ledger entry and no call site.
///
/// [04-DTYPE-2] puts the bound in the binder list, so the route has everything
/// it needs at the `mean` node: `Float` admits, and `Int` and `Numeric` each
/// admit a dtype `mean` does not. The declaration alone is the witness, which is
/// why this half cannot be left to the instantiation pass.
///
/// The last assertion is the DISJOINTNESS property: a bounded binder that is
/// also badly instantiated is one defect and must be reported once.
/// `collect_defsig_dtype_bounds` is what makes the two enforcement points
/// disjoint, by handing every bounded variable to this one.
///
/// REGRESSION TEST for the two rejections, both measured ACCEPTED at score 1 on
/// `0820ee28e` with no call site at all, and for the single-report property,
/// measured as two diagnostics before `collect_defsig_dtype_bounds` existed.
/// DISPOSITION LOCK for the `Float` row, measured accepted there and here.
#[test]
fn a_bounded_precision_binder_is_decided_by_its_family_at_once() {
    check("def g[p: Float](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n")
        .expect("a `Float`-bounded binder is admitted by a float-only route");

    for (bound, gloss) in [
        ("Int", "the active signed integer dtypes"),
        ("Numeric", "the active numeric dtypes"),
    ] {
        let errors = check(&format!(
            "def g[p: {bound}](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n"
        ))
        .expect_err("a bound admitting an inadmissible dtype must be rejected");
        assert!(
            errors.iter().any(|e| e.message.contains(&format!(
                "mean on operand precision `p` (a precision variable bounded by dtype family \
                 `{bound}`, {gloss}) is not admitted per the chelis#724 capability decision"
            ))),
            "the `{bound}` bound must be named in the rejection:\n{}",
            summary(&errors)
        );
        assert!(
            errors.iter().any(|e| e
                .suggestions
                .iter()
                .any(|hint| hint
                    .contains("spec/04-type-system.md \u{a7}5.9 [04-DTYPE-2]: the binder's"))),
            "the rejection must carry the [04-DTYPE-2] repair hint:\n{}",
            summary(&errors)
        );
    }

    let errors = check(
        "def g[p: Numeric](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n\
         f = g(to_tensor([1i32, 2i32, 4i32]))\n",
    )
    .expect_err("a bounded binder badly instantiated must still be rejected");
    assert_eq!(
        errors
            .iter()
            .filter(|e| e.message.contains("mean on operand precision"))
            .count(),
        1,
        "one defect must be reported once, by the in-def decision alone:\n{}",
        summary(&errors)
    );
}

/// chelis#1805 across the family-policy routes, and the shape it must NOT touch.
///
/// A family policy is one stated over `Prim::is_float()` or
/// `Prim::is_integer()`: exactly the policies a `[p: Float]` or `[p: Int]` bound
/// satisfies. `add` is not one (it admits every numeric dtype) and must keep
/// accepting an unbounded binder at any instantiation, or the repair would
/// reject the dtype-polymorphic helpers the standard library is written from.
///
/// The `integer`/`float` qualifier words in the `div` and `trunc_div` rows are
/// asserted deliberately: the refactor that gave every family text one
/// implementation had to keep them on a concrete dtype and drop them for a
/// precision variable, and only a concrete row can prove the first half.
///
/// REGRESSION TEST for every rejecting row: each was measured ACCEPTED at score
/// 1 on `0820ee28e`. DISPOSITION LOCK for the two accepting rows.
#[test]
fn every_family_policy_route_rejects_an_inadmissible_instantiation() {
    for (body, elements, needle) in [
        (
            "sqrt(x)",
            "1i32, 4i32, 9i32",
            "sqrt on operand precision `int32` is not admitted per \
             spec/04-type-system.md \u{a7}5.4",
        ),
        (
            "div(x, x)",
            "1i32, 4i32, 9i32",
            "div on integer operand precision `int32` is not admitted per \
             spec/05-risc-primitives.md \u{a7}2.1",
        ),
        (
            "trunc_div(x, x)",
            "1.0f32, 4.0f32, 9.0f32",
            "trunc_div on float operand precision `f32` is not admitted \
             per spec/05-risc-primitives.md \u{a7}2.1",
        ),
        (
            "softmax(x, 0i32)",
            "1i32, 2i32, 4i32",
            "softmax on operand precision `int32` is not admitted per \
             spec/04-type-system.md \u{a7}5.4",
        ),
    ] {
        let program = format!(
            "def g[p](x: tensor[3, p]) -> tensor[3, p] = {body}\n\
             f = g(to_tensor([{elements}]))\n"
        );
        let Err(errors) = check(&program) else {
            panic!("`{body}` must reject its inadmissible instantiation");
        };
        assert!(
            errors.iter().any(|e| e.message.contains(needle)),
            "`{body}` must reject with its own text:\n{}",
            summary(&errors)
        );
    }

    // NOT a family policy: `add` admits every numeric dtype, so an integer
    // instantiation is well typed and must not be rejected.
    check(
        "def g[p](x: tensor[3, p]) -> tensor[3, p] = add(x, x)\n\
         f = g(to_tensor([1i32, 2i32, 4i32]))\n",
    )
    .expect("an operation with no family policy admits an integer instantiation");
    // A concrete float operand is untouched by all of this.
    check("def f(x: tensor[3, f32]) -> tensor[f32] = mean(x, 0i32)\n")
        .expect("a concrete float operand is still accepted");
}

/// chelis#1805, the arms it measured SAFE rather than repairing: a float-only
/// callee whose own builtin signature declares a `Float`-bounded precision
/// variable.
///
/// `dropout` and `test_assert_close_tensor` are the two. Unification propagates
/// the signature's bound onto the caller's own binder, so the call site rejects
/// every integer instantiation with the [04-DTYPE-2] family diagnostic. That is
/// why `operand_family_policy` does not list them and why their census rows are
/// `caught_downstream` rather than `deferred`: this test is the witness those
/// rows name.
///
/// DISPOSITION LOCK, every assertion: all four were measured with these exact
/// verdicts on `0820ee28e`, before the repair. Removing the downstream catch
/// turns this test red.
#[test]
fn a_signature_bounded_callee_is_caught_at_the_call_site() {
    const FAMILY: &str = "type variable bounded by dtype family `Float` (the active float dtypes) \
         cannot be instantiated at `int32`";

    check("def g[p](x: tensor[3, p], r: p) -> tensor[3, p] ! { Random } = dropout(x, r)\n")
        .expect("the signature's own bound makes the declaration well typed");

    let errors = check(
        "def g[p](x: tensor[3, p], r: p) -> tensor[3, p] ! { Random } = dropout(x, r)\n\
         def f() -> tensor[3, int32] ! { Random } = g(to_tensor([1i32, 2i32, 4i32]), 1i32)\n",
    )
    .expect_err("the integer instantiation must be rejected");
    assert!(
        errors.iter().any(|e| e.message.contains(FAMILY)),
        "dropout's caller must be caught by the propagated bound:\n{}",
        summary(&errors)
    );

    check(
        "def g[p](a: &tensor[3, p], b: &tensor[3, p], tol: p) -> unit ! { Test } = \
         test_assert_close_tensor(a, b, tol, \"l\")\n",
    )
    .expect("the signature's own bound makes the declaration well typed");

    let errors = check(
        "def g[p](a: &tensor[3, p], b: &tensor[3, p], tol: p) -> unit ! { Test } = \
         test_assert_close_tensor(a, b, tol, \"l\")\n\
         def test_call() -> unit ! { Test } = \
         g(&to_tensor([1i32, 2i32, 4i32]), &to_tensor([1i32, 2i32, 4i32]), 1i32)\n",
    )
    .expect_err("the integer instantiation must be rejected");
    assert!(
        errors.iter().any(|e| e.message.contains(FAMILY)),
        "test_assert_close_tensor's caller must be caught by the propagated bound:\n{}",
        summary(&errors)
    );
}

/// chelis#1805: a local binding SHADOWING a top-level one is read from the
/// local binding, in both directions.
///
/// The argument reader resolves a `(var name)` through the binding chain in
/// scope order. Reading the outer binding instead is wrong both ways round: it
/// rejects a legal program by naming a dtype no operand at that call carries,
/// and it accepts an illegal one because the shadowed binding happens to be
/// admissible.
///
/// REGRESSION TEST, both assertions, from round 1's P1-1 probes. On
/// `a36c7c581` the first was REJECTED naming `int32`, a dtype the call's
/// operand does not carry, and the second was ACCEPTED at score 1 with the
/// compiled C lane printing `f = 2` for a true 2.3333333.
#[test]
fn a_local_binding_shadowing_a_top_level_one_is_read_from_the_local() {
    const HELPER: &str = "def g[p](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n";

    check(&format!(
        "{HELPER}t = to_tensor([1i32, 2i32, 4i32])\n\
         def pick() -> tensor[f32] = {{\n  \
         t = to_tensor([1.0f32, 2.0f32, 4.0f32])\n  g(t)\n}}\n\
         f = pick()\n"
    ))
    .expect("the local f32 binding is what `g` receives, so the program is well typed");

    let errors = check(&format!(
        "{HELPER}t = to_tensor([1.0f32, 2.0f32, 4.0f32])\n\
         def pick() -> tensor[int32] = {{\n  \
         t = to_tensor([1i32, 2i32, 4i32])\n  g(t)\n}}\n\
         f = pick()\n"
    ))
    .expect_err("the local int32 binding is what `g` receives, so the program is rejected");
    assert!(
        errors.iter().any(|e| e.message.contains(
            "mean on operand precision `int32` is not admitted per the chelis#724 \
             capability decision"
        )),
        "the rejection must name the LOCAL binding's dtype:\n{}",
        summary(&errors)
    );
}

/// Binding references retain the scope in which their value was defined.
/// The two dtype orders catch false acceptance as well as false rejection.
#[test]
fn argument_precision_respects_sequential_and_top_level_binding_scopes() {
    const HELPER: &str = "def g[p](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n";
    for (first, second, rejects) in [
        ("[1.0f32, 2.0f32, 4.0f32]", "[1i32, 2i32, 4i32]", false),
        ("[1i32, 2i32, 4i32]", "[1.0f32, 2.0f32, 4.0f32]", true),
    ] {
        for program in [
            format!(
                "{HELPER}t = to_tensor({first})\n\
                 def pick() = {{\n u = t\n t = to_tensor({second})\n g(u)\n }}\nf = pick()\n"
            ),
            format!(
                "{HELPER}t = to_tensor({first})\nu = t\n\
                 def pick() = {{\n t = to_tensor({second})\n g(u)\n }}\nf = pick()\n"
            ),
        ] {
            let result = check(&program);
            if rejects {
                let errors = result.expect_err("the referenced binding carries int32");
                assert!(
                    errors.iter().any(|error| error
                        .message
                        .contains("mean on operand precision `int32` is not admitted")),
                    "{}",
                    summary(&errors)
                );
            } else {
                result.expect("the referenced binding carries f32");
            }
        }
    }
}

/// chelis#1805: bool arithmetic reached through an unbounded precision binder.
///
/// `BOOL_REJECTED_ARITH_OPS` is not a family policy, so nothing in this repair
/// decides it, but the argument reader feeds the pre-existing restricted-op
/// walk, which already covers those five operations. The narrowing is therefore
/// real and had no assertion, which is how it could regress silently while
/// chelis#1937 stayed open against a repaired witness.
///
/// REGRESSION TEST, every row: all five were measured ACCEPTED at score 1 on
/// `0820ee28e`, and the first is chelis#1937's verbatim witness.
#[test]
fn bool_arithmetic_through_an_unbounded_binder_is_rejected_at_the_instantiation() {
    for (op, body) in [
        ("add", "add(x, x)"),
        ("sub", "sub(x, x)"),
        ("mul", "mul(x, x)"),
        ("neg", "neg(x)"),
        ("floor_div", "floor_div(x, x)"),
    ] {
        let program = format!(
            "def g[p](x: tensor[3, p]) -> tensor[3, p] = {body}\n\
             def f() -> tensor[3, bool] = g(to_tensor([true, false, true]))\n"
        );
        let Err(errors) = check(&program) else {
            panic!("`{op}` on a bool instantiation must be rejected");
        };
        assert!(
            errors.iter().any(|e| e.message.contains(&format!(
                "{op} on bool operands is not admitted per the chelis#726 capability decision"
            ))),
            "`{op}` must reject with the chelis#726 text:\n{}",
            summary(&errors)
        );
        check(&format!(
            "def g[p](x: tensor[3, p]) -> tensor[3, p] = {body}\n\
             def f() -> tensor[3, int32] = g(to_tensor([1i32, 2i32, 4i32]))\n"
        ))
        .expect("the same arithmetic at int32 is admitted");
    }
}

/// The argument spellings chelis#1805's reader does NOT resolve, DISPOSITION
/// LOCKS so the residue is loud in the suite rather than silent.
///
/// The covered set is stated by construction: an argument whose precision the
/// reader resolves, reaching a restricted operation in the callee's own body
/// EXPRESSION. Each row below falls outside one half of that and needs a
/// mechanism this pull request does not add. All five were measured on
/// `0820ee28e` accepting at score 1 and printing `f = 2` on the compiled C lane
/// for a true 2.3333333, and all five still do.
///
/// A failure here means the owning issue was fixed.
#[test]
fn the_spellings_the_argument_reader_does_not_resolve_remain_unreached() {
    const HELPER: &str = "def g[p](x: tensor[3, p]) -> tensor[p] = mean(x, 0i32)\n";

    // chelis#1940: the helper reached as a function VALUE. It never appears in
    // callee position, so the call-site pass has nothing to key on.
    check(&format!(
        "{HELPER}def apply_it(fn_arg: (tensor[3, int32]) -> tensor[int32], y: tensor[3, int32]) \
         -> tensor[int32] = fn_arg(y)\n\
         f = apply_it(g, to_tensor([1i32, 2i32, 4i32]))\n"
    ))
    .expect("chelis#1940: a helper passed as a value is not reached");

    // chelis#1941: the substitution does not cross a second polymorphic def,
    // whether the second one wraps the call or supplies the argument.
    check(&format!(
        "{HELPER}def mid[p](x: tensor[3, p]) -> tensor[p] = g(x)\n\
         f = mid(to_tensor([1i32, 2i32, 4i32]))\n"
    ))
    .expect("chelis#1941: a two-def polymorphic chain is not reached");
    check(&format!(
        "{HELPER}def id2[q](y: tensor[3, q]) -> tensor[3, q] = y\n\
         f = g(id2(to_tensor([1i32, 2i32, 4i32])))\n"
    ))
    .expect("chelis#1941: a callee whose declared result is a bare binder is not reached");

    // chelis#1805: the restricted operation sits in a local lambda, so the
    // callee's parameter never reaches it by name and the body walk's
    // parameter-to-precision map does not apply.
    check(
        "def g[p](x: tensor[3, p]) -> tensor[p] = {\n  \
         h = fn (t) -> mean(t, 0i32)\n  h(x)\n}\n\
         f = g(to_tensor([1i32, 2i32, 4i32]))\n",
    )
    .expect("chelis#1805: a restricted op inside a local lambda is not reached");

    // chelis#1805: an ADT field argument. The reader can see the target's
    // declared type is an ADT by name, and the field's own type lives in a
    // declaration registry this pass does not hold.
    check(&format!(
        "type Box =\n  | Box {{ t: tensor[3, int32] }}\n\n\
         {HELPER}def pick(b: Box) -> tensor[int32] = g(b.t)\n\
         f = pick(Box {{ t: to_tensor([1i32, 2i32, 4i32]) }})\n"
    ))
    .expect("chelis#1805: an ADT field argument is not reached");
}

/// Round 1 P2-1: this pull request NARROWS acceptance on the empty tensor
/// literal, and the narrowing was neither recorded nor covered.
///
/// A shape- and dtype-preserving route ties its result to its operand, so a
/// declaration that names the result determines the operand as well. The call
/// is suspended while the operand is still a variable, the declaration binds
/// it, and the replay then consults the dtype policy against the bound type.
/// The three programs below were MEASURED accepted at score 1 on `6dbbbf2bc`
/// and are rejected here.
///
/// This is why the sibling above is a bounded residual rather than a blanket
/// one, and it strengthens `Closes #1512` rather than qualifying it: an empty
/// literal under a reducing route keeps its hole, an empty literal under a
/// preserving route does not.
///
/// REGRESSION TEST, every assertion. The earlier spelling of this file asserted
/// the `softmax` row inside the chelis#1805 lock with a mechanism sentence that
/// was false ("signature unification binds it before the validator runs";
/// unification did not bind it, and the base accepts the program).
#[test]
fn an_empty_literal_the_declared_result_determines_is_now_validated() {
    for (route, program, diagnostic) in [
        (
            "softmax",
            "def f() -> tensor[3, int32] = softmax(to_tensor([]), 0i32)\n",
            "softmax on operand precision `int32` is not admitted",
        ),
        (
            "sqrt",
            "def f() -> tensor[3, int32] = sqrt(to_tensor([]))\n",
            "sqrt on operand precision `int32` is not admitted",
        ),
        (
            "add",
            "def f() -> tensor[3, bool] = add(to_tensor([]), to_tensor([]))\n",
            "add on bool operands is not admitted",
        ),
    ] {
        let errors = check(program).expect_err(&format!(
            "{route}: an empty literal the declaration determines must now be validated"
        ));
        assert!(
            errors.iter().any(|e| e.message.contains(diagnostic)),
            "{route}: the narrowed rejection must name the dtype policy, got:\n{}",
            summary(&errors)
        );
    }

    // NEGATIVE TWIN. The narrowing is a dtype verdict, not a rejection of the
    // empty literal: the same programs at an admissible dtype still check.
    // DISPOSITION LOCK, accepted on `6dbbbf2bc` and here.
    for (route, program) in [
        (
            "softmax",
            "def f() -> tensor[3, f32] = softmax(to_tensor([]), 0i32)\n",
        ),
        ("sqrt", "def f() -> tensor[3, f32] = sqrt(to_tensor([]))\n"),
    ] {
        check(program).unwrap_or_else(|e| {
            panic!(
                "{route}: an admissible dtype over an empty literal must still check:\n{}",
                summary(&e)
            )
        });
    }

    // chelis#1836 NARROWS this twin further, and only for a REDUCING route.
    // `softmax` and `sqrt` preserve their operand's shape, so the declared
    // result determines the operand and the empty literal binds. `mean` does
    // not: its result has one fewer axis, so no declaration can supply the
    // operand's rank, the suspension is never discharged, and the obligation
    // reaches the declaration boundary.
    //
    // REGRESSION TEST. Measured on `6abca2406` as score 1 with `errors []` at
    // check and `error: to_tensor requires a resolved checked element dtype
    // [05-OP-57]` at eval: the acceptance this replaces was a check-clean
    // program that does not run, which is the defect class this ledger exists
    // to close rather than a capability being removed.
    let errors = check("def f() -> tensor[f32] = mean(to_tensor([]), 0i32)\n")
        .expect_err("a reducing route over an unresolved literal must be rejected");
    assert!(
        errors.iter().any(|e| e
            .message
            .starts_with("unresolved `mean` shape obligation at declaration boundary")),
        "the narrowing must be the boundary obligation:\n{}",
        summary(&errors)
    );
}

// ---------------------------------------------------------------------------
// chelis#1836: a shape-computed route whose operand resolves AFTER the route
// runs, reached through a provenance the lambda-ownership predicate did not
// recognize.
//
// The suspension chelis#1512 installed waited on
// `shape_operand_awaits_lambda_binding`: true only for a variable in
// `shape_lambda_tvars` or free in one of their substitutions. Three
// provenances are not owned that way -- a `pat-tuple` element on an
// unresolved scrutinee, a field access on an unresolved target, and a
// chelis#1577 gate's result -- so for them the route took its eager
// `Type::Var` arm, published the call's own result variable, and the
// declaration was free to bind that variable to any shape at all.
//
// Each cell below is the pair the issue names: the FALSE declared shape must
// be rejected with the direct spelling's own `DimensionMismatch` text, and
// the TRUE declared shape must still check. Asserting the text rather than
// mere rejection is the same discipline the rest of this file follows: a
// relocated decision can be reported by an unrelated rule.
// ---------------------------------------------------------------------------

/// The direct spelling's rejection, which every untied form must reproduce.
fn signature_mismatch(params: &str, body_result: &str, declared_result: &str) -> String {
    format!(
        "def 'probe' body doesn't match declared signature: body has type \
         `({params}) -> {body_result}`, declared type is `({params}) -> {declared_result}`"
    )
}

fn expect_exact_error(program: &str, expected: &str) {
    let errors = check(program).expect_err(&format!("this program must be rejected:\n{program}"));
    assert!(
        errors.iter().any(|e| e.message == expected),
        "the rejection must be the direct spelling's own text.\nexpected: {expected}\ngot:\n{}",
        summary(&errors)
    );
}

/// chelis#1836 cell 1: `sum` over a `match`-destructured operand.
///
/// `pattern_bindings`'s `pat-tuple` arm handed every sub-pattern
/// `vg.fresh_type()` and unified nothing, so `a` in `| (a, k) =>` descended
/// from `q` by name alone and never bound when `q` did. The repair unifies the
/// unresolved scrutinee with a tuple of one fresh element per sub-pattern --
/// the pattern fixes the arity -- so `a` IS `q`'s first element.
///
/// REGRESSION TEST for the first assertion: measured on `6abca2406` as score
/// 1 with `errors []`. DISPOSITION LOCK for the second and third: the true
/// declaration and the never-applied lambda were measured accepting there too,
/// and only the never-applied one moves (it is now an unresolved obligation,
/// which is the acceptance boundary chelis#1489 set for a shape-computed
/// route whose operand no application ever supplies).
#[test]
fn a_match_destructured_sum_operand_is_tied_to_the_bound_scrutinee() {
    let program = |declared: &str| {
        format!(
            "def apply_p[b](f: ((tensor[4, 3, f32], int32)) -> b, r: (tensor[4, 3, f32], int32)) -> b = f(r)\n\
             def probe(t: tensor[4, 3, f32]) -> {declared} = apply_p(fn (q) -> match q with {{ | (a, k) => sum(a, 0) }}, (t, 1i32))\n"
        )
    };

    expect_exact_error(
        &program("tensor[100, f32]"),
        &signature_mismatch("tensor[4, 3, f32]", "tensor[3, f32]", "tensor[100, f32]"),
    );

    // NEGATIVE TWIN: the shape the route actually produces still checks, so
    // the repair rejects a wrong claim rather than the destructuring form.
    check(&program("tensor[3, f32]"))
        .unwrap_or_else(|e| panic!("the true result shape must still check:\n{}", summary(&e)));

    // The operand that NEVER binds is the acceptance boundary, not a pass:
    // the lambda is never applied, so no application supplies `q`'s
    // constructor. Measured accepting at score 1 on `6abca2406`.
    let errors = check("def lam() = fn (q) -> match q with { | (a, k) => sum(a, 0) }\n")
        .expect_err("a never-applied shape-computed route must be rejected");
    assert!(
        errors.iter().any(|e| e
            .message
            .starts_with("unresolved `sum` shape obligation at declaration boundary")),
        "the boundary rejection must name the route:\n{}",
        summary(&errors)
    );
}

/// chelis#1836 cell 2: `matmul` over the same `match`-destructured operand.
///
/// The rule is a different one (`check_matmul_signature`, whose own eager
/// `Type::Var` arm is at the head of both operand reads), so it is a separate
/// cell rather than a spelling of cell 1.
///
/// REGRESSION TEST for the first assertion, measured score 1 `errors []` on
/// `6abca2406`; DISPOSITION LOCK for the second, accepted there and here.
#[test]
fn a_match_destructured_matmul_operand_is_tied_to_the_bound_scrutinee() {
    let program = |declared: &str| {
        format!(
            "def apply_p[b](f: ((tensor[4, 3, f32], int32)) -> b, r: (tensor[4, 3, f32], int32)) -> b = f(r)\n\
             def probe(t: tensor[4, 3, f32], w: tensor[3, 5, f32]) -> {declared} = apply_p(fn (q) -> match q with {{ | (a, k) => matmul(a, w) }}, (t, 1i32))\n"
        )
    };

    expect_exact_error(
        &program("tensor[100, 5, f32]"),
        &signature_mismatch(
            "tensor[4, 3, f32], tensor[3, 5, f32]",
            "tensor[4, 5, f32]",
            "tensor[100, 5, f32]",
        ),
    );

    // NEGATIVE TWIN.
    check(&program("tensor[4, 5, f32]")).unwrap_or_else(|e| {
        panic!(
            "matmul's true result shape must still check:\n{}",
            summary(&e)
        )
    });
}

/// chelis#1836 cell 3: `sum` over a field of an unresolved record target.
///
/// `infer_access`'s `Type::Var` arm publishes a fresh variable and records the
/// target on the deferred OPACITY ledger, which revisits the target but never
/// ties the projected field type to the field the target turns out to have.
/// The repair registers a `DeferredTypeDerivation::RecordField` beside the
/// existing tuple-projection entry, resolved by ADT field lookup when the
/// target binds.
///
/// This cell is why the widened readiness predicate is not the whole repair:
/// with the predicate alone the projected variable never binds and this
/// program is rejected as an unresolved obligation, which is false -- the
/// application does supply the target.
///
/// REGRESSION TEST for the first assertion, measured score 1 `errors []` on
/// `6abca2406`; DISPOSITION LOCK for the second.
#[test]
fn a_record_field_sum_operand_is_tied_to_the_bound_target() {
    let program = |declared: &str| {
        format!(
            "type Box =\n  | Box {{ x: tensor[4, 3, f32] }}\n\
             def apply_b[b](f: (Box) -> b, r: Box) -> b = f(r)\n\
             def probe(t: tensor[4, 3, f32]) -> {declared} = apply_b(fn (q) -> sum(q.x, 0), Box {{ x: t }})\n"
        )
    };

    expect_exact_error(
        &program("tensor[100, f32]"),
        &signature_mismatch("tensor[4, 3, f32]", "tensor[3, f32]", "tensor[100, f32]"),
    );

    // NEGATIVE TWIN.
    check(&program("tensor[3, f32]")).unwrap_or_else(|e| {
        panic!(
            "the true result shape over a record field must still check:\n{}",
            summary(&e)
        )
    });
}

/// chelis#1836 cell 4: `sum` over a chelis#1577 `copy` gate's result.
///
/// `copy` used to reject an unresolved operand eagerly, so no route downstream
/// of it ever saw a variable. chelis#1577 made it defer, and its result is a
/// fresh variable tied to the operand only through the gate ledger -- which
/// the ownership predicate did not follow. This is the cell that makes the
/// class newly reachable, and the one the issue asks to be closed before
/// chelis#1577 ships.
///
/// REGRESSION TEST for the first assertion, measured score 1 `errors []` on
/// `6abca2406`; DISPOSITION LOCK for the second and third.
#[test]
fn a_gated_copy_result_ties_its_sum_consumer_to_the_bound_operand() {
    let program = |declared: &str| {
        format!(
            "def apply_n[b](f: tensor[n, 3, f32] -> b, x: tensor[n, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 3, f32]) -> {declared} = apply_n(fn (v) -> sum(copy(v), 0), t)\n"
        )
    };

    expect_exact_error(
        &program("tensor[100, f32]"),
        &signature_mismatch("tensor[4, 3, f32]", "tensor[3, f32]", "tensor[100, f32]"),
    );

    // NEGATIVE TWIN.
    check(&program("tensor[3, f32]")).unwrap_or_else(|e| {
        panic!(
            "the true result shape behind a copy gate must still check:\n{}",
            summary(&e)
        )
    });

    // The unapplied form keeps `copy`'s own rejection AND gains the boundary
    // obligation: the gate never discharges, so its consumer never settles.
    // Measured on `6abca2406` reporting only the `copy` diagnostic.
    let errors = check("def lam() = fn (v) -> sum(copy(v), 0)\n")
        .expect_err("a never-applied gated route must be rejected");
    assert!(
        errors
            .iter()
            .any(|e| e.message.starts_with("copy requires tensor input, got ?")),
        "the existing copy rejection must not be lost:\n{}",
        summary(&errors)
    );
    assert!(
        errors.iter().any(|e| e
            .message
            .starts_with("unresolved `sum` shape obligation at declaration boundary")),
        "the consumer's own obligation must also be reported:\n{}",
        summary(&errors)
    );
}

/// NEGATIVE PARITY for chelis#1836's two ties: an ill-typed destructuring or
/// field read on a late-bound operand is now REPORTED, where it used to be
/// invisible.
///
/// Tying an unresolved scrutinee to the pattern's shape, and a projected field
/// to the field the target carries, makes four disagreements checkable that the
/// disconnected fresh variable absorbed. That is the tie working rather than a
/// side effect: a pattern or a field read that cannot agree with the value the
/// application supplies is a defect in the program, and nothing else in the
/// checker was positioned to notice.
///
/// Each program also reports the route's declaration-boundary obligation,
/// because the tie that failed is the one that would have settled the operand.
/// The assertions below name the PRIMARY diagnostic, and the last cell is the
/// control that nothing became noisy: a tie that succeeds still checks.
///
/// REGRESSION TEST, every rejecting assertion. All four were measured on
/// `6abca2406` at score 1 with `errors []`.
#[test]
fn an_ill_typed_destructure_or_field_read_on_a_late_bound_operand_is_now_reported() {
    for (case, program, kind, diagnostic) in [
        (
            "pattern arity disagrees with the scrutinee",
            "def apply_p[b](f: ((tensor[4, 3, f32], int32)) -> b, r: (tensor[4, 3, f32], int32)) -> b = f(r)\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[3, f32] = apply_p(fn (q) -> match q with { | (a, k, z) => sum(a, 0) }, (t, 1i32))\n",
            "ArityMismatch",
            "tuple length mismatch: 2 vs 3",
        ),
        (
            "tuple pattern on a scrutinee that binds to a tensor",
            "def apply_t[b](f: (tensor[4, 3, f32]) -> b, x: tensor[4, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[3, f32] = apply_t(fn (q) -> match q with { | (a, k) => sum(a, 0) }, t)\n",
            "TypeMismatch",
            "type mismatch: tensor[4, 3, f32] vs (",
        ),
        (
            "field read on a target that binds to a tensor",
            "def apply_t[b](f: (tensor[4, 3, f32]) -> b, x: tensor[4, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[3, f32] = apply_t(fn (q) -> sum(q.x, 0), t)\n",
            "TypeMismatch",
            "field access `.x` expects a record value, got a value of type `tensor[4, 3, f32]` (chelis#755)",
        ),
        (
            "unknown field on a target that binds to a record",
            "type Box =\n  | Box { x: tensor[4, 3, f32] }\n\
             def apply_b[b](f: (Box) -> b, r: Box) -> b = f(r)\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[3, f32] = apply_b(fn (q) -> sum(q.zzz, 0), Box { x: t })\n",
            "TypeMismatch",
            "unknown record field 'zzz' on Box",
        ),
    ] {
        let errors = check(program).expect_err(&format!("{case}: this program must be rejected"));
        assert!(
            errors
                .iter()
                .any(|e| format!("{:?}", e.kind) == kind && e.message.contains(diagnostic)),
            "{case}: the primary diagnostic must be [{kind}] {diagnostic:?}, got:\n{}",
            summary(&errors)
        );
    }

    // ONE diagnostic per mistake, not two. Round 1 P3-1: the derivation's
    // failing shapes bind `projected` to the reported error's witness, so the
    // suspended route sees a settled `Type::Error` operand and takes its
    // chelis#731 §C3 cascade-suppression arm instead of reaching the
    // declaration boundary. Before the fold each program below reported the
    // field defect AND `unresolved `sum` shape obligation at declaration
    // boundary`.
    //
    // REGRESSION TEST for the count. The field diagnostics themselves are
    // asserted above; this asserts that nothing follows them.
    for (case, program) in [
        (
            "field read on a target that binds to a tensor",
            "def apply_t[b](f: (tensor[4, 3, f32]) -> b, x: tensor[4, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[3, f32] = apply_t(fn (q) -> sum(q.x, 0), t)\n",
        ),
        (
            "unknown field on a target that binds to a record",
            "type Box =\n  | Box { x: tensor[4, 3, f32] }\n\
             def apply_b[b](f: (Box) -> b, r: Box) -> b = f(r)\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[3, f32] = apply_b(fn (q) -> sum(q.zzz, 0), Box { x: t })\n",
        ),
        (
            "multi-variant target",
            "type Shape =\n  | Circle { r: tensor[4, 3, f32] }\n  | Square { s: tensor[4, 3, f32] }\n\
             def apply_s[b](f: (Shape) -> b, r: Shape) -> b = f(r)\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[3, f32] = apply_s(fn (q) -> sum(q.r, 0), Circle { r: t })\n",
        ),
    ] {
        let errors = check(program).expect_err(&format!("{case}: must be rejected"));
        assert!(
            !errors.iter().any(|e| e
                .message
                .contains("shape obligation at declaration boundary")),
            "{case}: the field defect must not drag the route's boundary obligation with it:\n{}",
            summary(&errors)
        );
        assert_eq!(
            errors.len(),
            1,
            "{case}: one mistake must report one diagnostic, got:\n{}",
            summary(&errors)
        );
    }

    // CONTROL. Two field reads on one unresolved target both resolve, so the
    // ties report nothing when they agree. DISPOSITION LOCK: accepted on
    // `6abca2406` and here.
    check(
        "type Box =\n  | Box { x: tensor[4, 3, f32], y: tensor[4, 3, f32] }\n\
         def apply_b[b](f: (Box) -> b, r: Box) -> b = f(r)\n\
         def probe(t: tensor[4, 3, f32]) -> tensor[4, 3, f32] = apply_b(fn (q) -> add(q.x, q.y), Box { x: t, y: t })\n",
    )
    .unwrap_or_else(|e| {
        panic!(
            "two field reads on one late-bound target must still check:\n{}",
            summary(&e)
        )
    });
}

// ---------------------------------------------------------------------------
// chelis#1836 through chelis#1690's operand gates.
//
// chelis#1690 defers `gather`, `scatter`, `scatter_replace` and `trace` on
// their operand through `DeferredOperandGate::ShapeRoute`, whose suspended
// form publishes a FRESH result variable, in its own words "exactly as the
// `copy`/`cast` gates do". That is this issue's third provenance, so each of
// those routes became a new chelis#1836 instance the moment chelis#1690
// landed: a shape-computed consumer of the gate's result took its eager
// `Type::Var` arm and published a result the declaration could bind freely.
//
// EVIDENTIARY STATUS: REGRESSION TESTS, every rejecting assertion. Each false
// declaration was MEASURED at score 1 with `errors []` on `e0a482248`, this
// branch's base, with a binary built from that commit's
// `crates/chelis-types/src`. The true twins were measured accepting there and
// accept here, so they are DISPOSITION LOCKS: the repair rejects the wrong
// claim, not the form.
//
// One test per corpus row, so `route.untied.{gather,trace,scatter_replace}
// .gate` each name a receipt that fails alone.
// ---------------------------------------------------------------------------

/// chelis#1690's `gather` gate: `sum(gather(v, i, 0i32), 0)`.
#[test]
fn a_gather_gate_result_ties_its_sum_consumer_to_the_bound_operand() {
    let program = |declared: &str| {
        format!(
            "def apply_v[b](f: tensor[4, 3, f32] -> b, x: tensor[4, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 3, f32], i: tensor[2, int32]) -> {declared} = apply_v(fn (v) -> sum(gather(v, i, 0i32), 0), t)\n"
        )
    };

    // Measured score 1, `errors []` on `e0a482248`.
    expect_exact_error(
        &program("tensor[99, f32]"),
        &signature_mismatch(
            "tensor[4, 3, f32], tensor[2, int32]",
            "tensor[3, f32]",
            "tensor[99, f32]",
        ),
    );

    // NEGATIVE TWIN.
    check(&program("tensor[3, f32]")).unwrap_or_else(|e| {
        panic!(
            "gather's true result shape behind its gate must still check:\n{}",
            summary(&e)
        )
    });
}

/// chelis#1690's `trace` gate: `sum(trace(v, 0i32, 1i32), 0)`.
///
/// The rank-zero result is declared `tensor[f32]` and RENDERS as
/// `tensor[, f32]`, the spelling chelis#1148 and chelis#1355 pin. Both
/// spellings appear below deliberately: the declaration uses the source form,
/// the diagnostic quotes the rendered one.
#[test]
fn a_trace_gate_result_ties_its_sum_consumer_to_the_bound_operand() {
    let program = |declared: &str| {
        format!(
            "def apply_w[b](f: tensor[4, 4, 3, f32] -> b, x: tensor[4, 4, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 4, 3, f32]) -> {declared} = apply_w(fn (v) -> sum(trace(v, 0i32, 1i32), 0), t)\n"
        )
    };

    // Measured score 1, `errors []` on `e0a482248`.
    expect_exact_error(
        &program("tensor[99, f32]"),
        &signature_mismatch("tensor[4, 4, 3, f32]", "tensor[, f32]", "tensor[99, f32]"),
    );

    // NEGATIVE TWIN, at the rank-zero declaration.
    check(&program("tensor[f32]")).unwrap_or_else(|e| {
        panic!(
            "trace's true rank-zero result behind its gate must still check:\n{}",
            summary(&e)
        )
    });
}

/// chelis#1690's `scatter_replace` gate:
/// `sum(scatter_replace(v, i, u, 0i32), 0)`.
#[test]
fn a_scatter_replace_gate_result_ties_its_sum_consumer_to_the_bound_operand() {
    let program = |declared: &str| {
        format!(
            "def apply_v[b](f: tensor[4, 3, f32] -> b, x: tensor[4, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 3, f32], i: tensor[2, int32], u: tensor[2, 3, f32]) -> {declared} = apply_v(fn (v) -> sum(scatter_replace(v, i, u, 0i32), 0), t)\n"
        )
    };

    // Measured score 1, `errors []` on `e0a482248`.
    expect_exact_error(
        &program("tensor[99, f32]"),
        &signature_mismatch(
            "tensor[4, 3, f32], tensor[2, int32], tensor[2, 3, f32]",
            "tensor[3, f32]",
            "tensor[99, f32]",
        ),
    );

    // NEGATIVE TWIN.
    check(&program("tensor[3, f32]")).unwrap_or_else(|e| {
        panic!(
            "scatter_replace's true result shape behind its gate must still check:\n{}",
            summary(&e)
        )
    });
}

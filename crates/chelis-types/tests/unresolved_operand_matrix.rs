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

use chelis_deep::parser::parse_str;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn check(source: &str) -> Result<(), Vec<CheckError>> {
    let exprs = if source.trim_start().starts_with('[') {
        serde_json::from_str(source).expect("lossless AST fixture")
    } else {
        parse_str(source).expect("Deep fixture must parse")
    };
    match check_typed_program(&exprs) {
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

/// One route's four parser-produced core programs: resolved and late-bound
/// operands, each with a valid and invalid call. `diagnostic` identifies the
/// route's own rejection.
struct Row {
    route: &'static str,
    resolved_valid: &'static str,
    resolved_invalid: &'static str,
    late_valid: &'static str,
    late_invalid: &'static str,
    diagnostic: &'static str,
}

fn run(row: &Row) {
    let Row {
        route,
        resolved_valid,
        resolved_invalid,
        late_valid,
        late_invalid,
        diagnostic,
    } = row;

    // Cell 1 (resolved, valid): DISPOSITION LOCK. Green before the repair;
    // it is here so a repair that rejects correct programs is caught.
    check(resolved_valid).unwrap_or_else(|e| {
        panic!(
            "{route}: a valid resolved call must check:\n{}",
            summary(&e)
        )
    });

    // Cell 2 (resolved, invalid): DISPOSITION LOCK. This is the rule that was
    // always right and simply never ran; the cell pins its exact wording so
    // cell 4 has something to be equal to.
    let eager = check(resolved_invalid).expect_err(&format!(
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
    check(late_valid).unwrap_or_else(|e| {
        panic!(
            "{route}: a valid call over an unresolved operand must check:\n{}",
            summary(&e)
        )
    });

    // Cell 4 (unresolved, invalid): REGRESSION TEST. This is chelis#1512.
    // Before the repair every route below accepted this program at score 1
    // with an empty error list.
    let deferred = check(late_invalid).expect_err(&format!(
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
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_155.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_156.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_053.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_157.deep"
            ),
            diagnostic: "permute expects 2 axis indices for rank 2 tensor, got 3",
        },
        Row {
            route: "shrink",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_158.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_159.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_160.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_161.deep"
            ),
            diagnostic: "shrink expects 2 bounds pairs for rank 2 tensor, got 1",
        },
        Row {
            route: "stride",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_162.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_163.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_056.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_164.deep"
            ),
            diagnostic: "stride expects 2 strides for rank 2 tensor, got 1",
        },
        Row {
            route: "pad",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_165.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_166.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_167.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_168.deep"
            ),
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
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_169.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_170.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_171.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_172.deep"
            ),
            diagnostic: "tensor_to_scalar expects a rank-0 tensor",
        },
        Row {
            route: "cumsum",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_173.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_174.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_175.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_176.deep"
            ),
            diagnostic: "cumsum axis 7 out of bounds for rank 2",
        },
        Row {
            route: "sort",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_177.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_178.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_179.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_180.deep"
            ),
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
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_181.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_182.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_061.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_183.deep"
            ),
            diagnostic: "len expects List or Dict input",
        },
        Row {
            route: "append",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_184.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_185.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_186.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_187.deep"
            ),
            diagnostic: "append expects List input",
        },
        Row {
            route: "enumerate",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_188.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_189.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_190.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_191.deep"
            ),
            diagnostic: "enumerate expects List input",
        },
        Row {
            route: "rank",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_192.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_193.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_194.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_195.deep"
            ),
            diagnostic: "rank expects tensor input",
        },
        Row {
            route: "numel",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_196.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_197.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_198.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_199.deep"
            ),
            diagnostic: "numel expects tensor input",
        },
        Row {
            route: "shape",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_200.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_201.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_202.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_203.deep"
            ),
            diagnostic: "shape expects tensor input",
        },
        Row {
            route: "string_len",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_204.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_205.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_206.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_207.deep"
            ),
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_001.deep"),
        ),
        (
            "cumsum",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_002.deep"),
        ),
        (
            "len",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_003.deep"),
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_004.deep"),
        ),
        (
            "reshape",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_005.deep"),
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_006.deep"),
        ),
        (
            "cumsum",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_007.deep"),
        ),
        (
            "pad",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_008.deep"),
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
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_208.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_209.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_210.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_211.deep"
            ),
            diagnostic: "dimension mismatch: Lit(2) vs Lit(5)",
        },
        Row {
            route: "sum",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_212.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_213.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_214.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_215.deep"
            ),
            diagnostic: "sum axis 7 is out of bounds for rank 2 tensor",
        },
        Row {
            route: "expand",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_216.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_217.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_218.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_219.deep"
            ),
            diagnostic: "expand axis 9 is out of bounds for rank 2 tensor",
        },
        Row {
            route: "layer_norm",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_220.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_221.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_222.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_223.deep"
            ),
            diagnostic: "dimension mismatch: Lit(2) vs Lit(5)",
        },
        Row {
            route: "scatter_elements",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_224.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_225.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_226.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_227.deep"
            ),
            diagnostic: "scatter_elements updates shape must equal indices shape",
        },
        Row {
            route: "conv",
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_228.deep"
            ),
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_229.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_230.deep"),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_231.deep"
            ),
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
        resolved_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_232.deep"),
        resolved_invalid: include_str!(
            "fixtures/core/tests/unresolved_operand_matrix/case_233.deep"
        ),
        late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_234.deep"),
        late_invalid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_235.deep"),
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_009.deep"),
            "and does not accept argument type tensor[3, f32] in this context",
        ),
        (
            "string_contains",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_010.deep"),
            "precision mismatch: expected string, got i32",
        ),
        (
            "dict_get",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_011.deep"),
            "precision mismatch: expected string, got i32",
        ),
        (
            "dict_insert",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_012.deep"),
            "precision mismatch: expected string, got i32",
        ),
        (
            "expand",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_013.deep"),
            "but no tensor in scope carries it",
        ),
        (
            "cast",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_014.deep"),
            "cast requires tensor or prim type",
        ),
        (
            "sum",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_015.deep"),
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
    let chained = include_str!("fixtures/core/tests/unresolved_operand_matrix/case_016.deep");
    check(chained).unwrap_or_else(|e| {
        panic!(
            "a chain of suspended routes must settle, not report an unresolved obligation:\n{}",
            summary(&e)
        )
    });

    // NEGATIVE TWIN: the chain settling is not the same as the chain being
    // unchecked. Both `permute` rules still run, so the wrong declared result
    // is still rejected.
    let wrong = include_str!("fixtures/core/tests/unresolved_operand_matrix/case_017.deep");
    check(wrong).expect_err("two transpositions return the original shape");
}

/// An operand that never binds is reported ONCE. The ledger holds one entry
/// per call, and the declaration boundary drains it once, so a suspended call
/// cannot accumulate a diagnostic per replay attempt.
///
/// REGRESSION TEST for the count, DISPOSITION LOCK for the wording.
#[test]
fn a_never_bound_operand_is_reported_exactly_once() {
    let never = include_str!("fixtures/core/tests/unresolved_operand_matrix/case_004.deep");
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
/// invalid twins need different declarations, uses three independent fixtures
/// rather than [`Row`]'s four-cell comparison.
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

fn assert_family_error(errors: &[CheckError], family: &str, dtype: &str) {
    assert!(
        errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::PrecisionMismatch
        ) && error.message.contains(&format!("dtype family `{family}`"))
            && error
                .message
                .contains(&format!("cannot be instantiated at `{dtype}`"))),
        "{}",
        summary(errors)
    );
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
    if matches!(*route, "add" | "mean" | "sqrt" | "softmax" | "shl" | "shr") {
        assert_family_error(
            &late,
            match *route {
                "add" => "Numeric",
                "shl" | "shr" => "Int",
                _ => "Float",
            },
            match *route {
                "add" => "bool",
                "shl" | "shr" => "f32",
                _ => "i32",
            },
        );
    } else {
        assert!(
            late.iter().any(|e| e.message.contains(diagnostic)),
            "{route}: the late-bound rejection must carry the SAME diagnostic as the resolved one \
         ({diagnostic:?}), got:\n{}",
            summary(&late)
        );
    }

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
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_018.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_019.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_020.deep"),
            diagnostic: "where expects cond/both branches to have matching tensor shapes and branch precision",
        },
        Cell {
            route: "clamp",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_021.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_022.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_023.deep"),
            diagnostic: "clamp expects tensor input plus scalar-tensor or matching-shape tensor bounds of the same precision",
        },
        Cell {
            route: "index",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_024.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_025.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_026.deep"),
            diagnostic: "index expects i64 index, got f32",
        },
        Cell {
            // `concat`'s List/List rule IS a unification of the element types,
            // so its own wording is the precision mismatch. Draft #1690
            // rewrites the SIBLING tensor-concat branch of this same match and
            // leaves this arm alone.
            route: "concat",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_027.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_028.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_029.deep"),
            diagnostic: "precision mismatch: expected i32, got f32",
        },
        Cell {
            route: "zip",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_030.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_031.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_032.deep"),
            diagnostic: "zip expects List inputs, got List i32 and i32",
        },
        Cell {
            route: "dict_merge",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_033.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_034.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_035.deep"),
            diagnostic: "precision mismatch: expected i32, got f32",
        },
        Cell {
            route: "split sizes",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_036.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_037.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_038.deep"),
            diagnostic: "split expects List[int] sizes",
        },
        Cell {
            route: "split input",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_039.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_040.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_041.deep"),
            diagnostic: "split expects tensor input and List[int] sizes",
        },
        Cell {
            route: "take",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_042.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_043.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_044.deep"),
            diagnostic: "take expects integer count, got f32",
        },
        Cell {
            // `drop` shares `take`'s arm; the cell is here because the
            // diagnostic interpolates the callee name and a shared arm that
            // named one of them would pass with the other silently wrong.
            route: "drop",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_045.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_046.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_047.deep"),
            diagnostic: "drop expects integer count, got f32",
        },
        Cell {
            route: "chunk",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_048.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_049.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_050.deep"),
            diagnostic: "chunk expects integer size, got f32",
        },
        Cell {
            // The window family suspended on its TENSOR operand only, so an
            // axis or step that bound late slipped past the route's own guard
            // while the tensor was already settled.
            route: "permute axis",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_051.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_052.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_053.deep"),
            diagnostic: "permute expects i32 axis indices, got i64",
        },
        Cell {
            route: "stride step",
            resolved_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_054.deep"
            ),
            late_invalid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_055.deep"
            ),
            late_valid: include_str!("fixtures/core/tests/unresolved_operand_matrix/case_056.deep"),
            diagnostic: "stride expects i64 strides (write 2i64), got i32",
        },
    ] {
        run_cell(&cell);
    }
}

/// [04-INF-9] narrows the old route-kind boundary: a newly authored generic
/// `len` or `index` wrapper may not publish its unresolved operation-admission
/// contract, even when the local lambda is never applied. `where` retains the
/// non-collection disposition this matrix established, and `dict_of([])` may
/// still be decided by its declaration's explicit result type.
#[test]
fn never_bound_collection_contracts_reject_while_other_routes_keep_their_dispositions() {
    for (route, program) in [
        (
            "len",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_057.deep"),
        ),
        (
            "index",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_058.deep"),
        ),
    ] {
        let errors =
            check(program).expect_err("an authored generic collection contract must be rejected");
        let prefix =
            format!("unresolved `{route}` shape obligation in `f` at declaration boundary");
        assert!(
            errors.iter().any(|error| {
                matches!(
                    error.kind,
                    chelis_types::errors::CheckErrorKind::TypeMismatch
                ) && error.message.starts_with(&prefix)
            }),
            "{route}: the declaration-boundary rejection must name the collection operation:\n{}",
            summary(&errors)
        );
    }

    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_059.deep"
    ))
    .unwrap_or_else(|e| {
        panic!(
            "where: the unchanged non-collection route must remain accepted:\n{}",
            summary(&e)
        )
    });

    // The standard library's own shape, reduced to one declaration.
    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_060.deep"
    ))
    .unwrap_or_else(|e| {
        panic!(
            "dict_of: the declared result decides an empty literal's element type, and the \
             operand never binds:\n{}",
            summary(&e)
        )
    });

    // Applied once, validly: still accepted, and the route ran.
    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_061.deep"
    ))
    .unwrap_or_else(|e| panic!("len: a valid binding must check:\n{}", summary(&e)));

    // Applied once, invalidly: rejected with the ROUTE's own text. Accepting
    // the never-bound case is not the same as not checking; this is the cell
    // that separates the two.
    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_062.deep"
    ))
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_063.deep"),
            "string_slice expects integer index arguments",
        ),
        (
            "string_concat",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_064.deep"),
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
    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_065.deep"
    ))
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
    if matches!(
        row.cell.route,
        "sqrt" | "add" | "mean" | "softmax" | "shl" | "shr"
    ) {
        // The operation's checked family now travels with the inferred lambda.
        // Its use owns the error; no body replay recreates the direct spelling.
        assert_eq!(late.len(), 1, "{}", summary(&late));
        assert_family_error(
            &late,
            match row.cell.route {
                "add" => "Numeric",
                "shl" | "shr" => "Int",
                _ => "Float",
            },
            match row.cell.route {
                "add" => "bool",
                "shl" | "shr" => "f32",
                _ => "i32",
            },
        );
        return;
    }
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
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_066.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_067.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_068.deep"
                ),
                diagnostic: "sqrt does not accept argument type i32 in this context",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_069.deep"
            ),
        },
        // The logical operations are bool-only, and the arithmetic ones are
        // exactly not-bool ([04-NUM-4]). Both directions, so a repair cannot
        // satisfy one by widening the other.
        DtypeCell {
            cell: Cell {
                route: "and",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_070.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_071.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_072.deep"
                ),
                diagnostic: "and does not accept argument type i32 in this context",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_073.deep"
            ),
        },
        DtypeCell {
            cell: Cell {
                route: "add",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_074.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_075.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_076.deep"
                ),
                diagnostic: "add on bool operands is not admitted",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_077.deep"
            ),
        },
        // The reduction family's own dtype rule, reached from the first-argument
        // arm of `validate_numeric_and_reduction_arguments`.
        DtypeCell {
            cell: Cell {
                route: "mean",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_078.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_079.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_080.deep"
                ),
                diagnostic: "mean on operand precision `i32` is not admitted",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_081.deep"
            ),
        },
        DtypeCell {
            cell: Cell {
                route: "softmax",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_082.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_083.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_084.deep"
                ),
                diagnostic: "softmax on operand precision `i32` is not admitted",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_085.deep"
            ),
        },
        // `reject_inadmissible_operand_dtypes`. `uniform_like`'s template
        // parameter is a bare type variable in the builtin scheme, so a
        // late-bound operand REACHES the route still unresolved rather than
        // being bound by signature unification first.
        DtypeCell {
            cell: Cell {
                route: "uniform_like (non-float template)",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_086.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_087.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_088.deep"
                ),
                diagnostic: "uniform_like expects a float tensor template",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_089.deep"
            ),
        },
        DtypeCell {
            cell: Cell {
                route: "uniform_like (non-tensor template)",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_090.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_091.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_088.deep"
                ),
                diagnostic: "uniform_like expects tensor template input",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_089.deep"
            ),
        },
        // `integer_binop_result_type`: the binary operators decide the call's
        // RESULT type, not only its admissibility, and their operands are three
        // independent type variables in the builtin scheme.
        DtypeCell {
            cell: Cell {
                route: "mod",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_092.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_093.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_094.deep"
                ),
                diagnostic: "mod requires matching integer arguments, got i64 and i32",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_095.deep"
            ),
        },
        DtypeCell {
            cell: Cell {
                route: "bitand",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_096.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_097.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_098.deep"
                ),
                diagnostic: "bitand requires matching integer arguments, got i64 and i32",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_099.deep"
            ),
        },
        // The shift operators, whose admissibility was two boolean
        // disjunctions rather than match arms until this repair.
        DtypeCell {
            cell: Cell {
                route: "shl",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_100.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_101.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_102.deep"
                ),
                diagnostic: "shl requires integer lhs and shift amount, got f32 and i32",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_103.deep"
            ),
        },
        DtypeCell {
            cell: Cell {
                route: "shr",
                resolved_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_104.deep"
                ),
                late_invalid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_105.deep"
                ),
                late_valid: include_str!(
                    "fixtures/core/tests/unresolved_operand_matrix/case_106.deep"
                ),
                diagnostic: "shr requires integer lhs and shift amount, got f32 and i32",
            },
            resolved_valid: include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_107.deep"
            ),
        },
    ] {
        run_dtype_cell(&row);
    }
}

/// The dtype half of the per-route-kind acceptance boundary.
///
/// New unresolved Float/Int requirements cannot become implicit generic
/// contracts. Concrete body inference and operations outside this family
/// mechanism retain their previous disposition; do not reject all unused lambdas.
#[test]
fn never_bound_dtype_operands_do_not_publish_new_family_requirements() {
    for (route, program, concrete_program) in [
        (
            "sqrt",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_108.json"),
            Some(include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_108_concrete.json"
            )),
        ),
        (
            "mod",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_109.json"),
            Some(include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_109_concrete.json"
            )),
        ),
        (
            "shl",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_110.json"),
            Some(include_str!(
                "fixtures/core/tests/unresolved_operand_matrix/case_110_concrete.json"
            )),
        ),
        (
            "uniform_like",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_111.json"),
            None,
        ),
    ] {
        if let Some(concrete_program) = concrete_program {
            let family = if route == "sqrt" { "Float" } else { "Int" };
            let errors = check(program).expect_err("a new family requirement cannot escape");
            assert!(
                errors.iter().any(|error| {
                    matches!(
                        error.kind,
                        chelis_types::errors::CheckErrorKind::PrecisionMismatch
                    ) && error.message.contains(family)
                }),
                "{}",
                summary(&errors)
            );
            check(concrete_program).expect("the explicit concrete contract admits the operation");
            continue;
        }
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
        (
            "sqrt",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_112.deep"),
        ),
        (
            "mod",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_113.deep"),
        ),
        (
            "shl",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_114.deep"),
        ),
        (
            "add",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_115.deep"),
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
    let program = include_str!("fixtures/core/tests/unresolved_operand_matrix/case_116.deep");
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
    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_093.deep"
    ))
    .expect_err("mod over a late-bound i64 must be rejected");
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_117.deep"),
            "trunc_div on float operand precision `f32` is not admitted",
        ),
        (
            "div",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_118.deep"),
            "div on integer operand precision `i32` is not admitted",
        ),
        (
            "sum (non-tensor operand)",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_119.deep"),
            "sum expects tensor input, got i32",
        ),
        (
            "dropout",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_120.deep"),
            "tensor precision mismatch: f32 vs i32",
        ),
        (
            "test_assert_close_tensor",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_121.deep"),
            "tensor precision mismatch: f64 vs f32",
        ),
        (
            "sum (late-bound axis)",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_122.deep"),
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

/// Late precision instantiations retain their declared Float restriction.
/// The original unbounded witnesses were accepted on `0820ee28e` and printed
/// an integer-truncated mean. Under §3.1 they fail at their definition; these
/// bounded counterparts isolate restriction transport at the call site.
/// Historical test identities remain stable for the class oracle.
#[test]
fn a_late_bound_tensor_precision_is_rejected_at_the_instantiation() {
    const HELPER: &str =
        include_str!("fixtures/core/tests/unresolved_operand_matrix/case_123.deep");
    const INT_TEXT: &str = "type variable bounded by dtype family `Float` (the active float dtypes) cannot be instantiated at `i32`";

    for (label, program) in [
        (
            "a to_tensor literal argument",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_01.deep")
                .replace("@@HELPER@@", HELPER),
        ),
        (
            "a nested call's declared result",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_02.deep")
                .replace("@@HELPER@@", HELPER),
        ),
        (
            "a top-level value binding",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_03.deep")
                .replace("@@HELPER@@", HELPER),
        ),
        (
            "a tuple element of a declared parameter",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_04.deep")
                .replace("@@HELPER@@", HELPER),
        ),
    ] {
        let Err(errors) = check(&program) else {
            panic!("{label}: the i32 instantiation must be rejected");
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
    check(HELPER).expect("a sufficiently bounded precision binder remains polymorphic");
    check(
        &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_05.deep")
            .replace("@@HELPER@@", HELPER),
    )
    .expect("an f32 instantiation of the same helper is well typed");
    check(
        &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_06.deep")
            .replace("@@HELPER@@", HELPER),
    )
    .expect("the same tuple projection at f32 is admitted");
}

/// The operand's tensor constructor is known before its precision binds.
/// An inferred lambda operand keeps the Float requirement from mean; ordinary
/// application unification admits the float call and rejects the integer one.
/// No precision replay ledger or named-callee body lookup is involved.
#[test]
fn a_precision_that_binds_one_application_later_is_decided_on_binding() {
    const HELPER: &str =
        include_str!("fixtures/core/tests/unresolved_operand_matrix/case_125.deep");

    let errors = check(
        &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_07.deep")
            .replace("@@HELPER@@", HELPER),
    )
    .expect_err("a precision that binds to i32 one application later must be rejected");
    assert!(
        errors.iter().any(|e| e.message.contains(
            "type variable bounded by dtype family `Float` (the active float dtypes) cannot be instantiated at `i32`"
        )),
        "the discharged rejection names the BOUND dtype, not the binder:\n{}",
        summary(&errors)
    );

    check(
        &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_08.deep")
            .replace("@@HELPER@@", HELPER),
    )
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
    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_123.deep"
    ))
    .expect("a `Float`-bounded binder is admitted by a float-only route");

    for (bound, gloss) in [
        ("Int", "the active signed integer dtypes"),
        ("Numeric", "the active numeric dtypes"),
    ] {
        let errors = check(
            &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_09.deep")
                .replace("@@bound@@", &bound.to_ascii_lowercase()),
        )
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

    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_126.deep"
    ))
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

/// Float and Int operations reject instantiations outside their authored
/// contract; Numeric arithmetic continues to admit both integer and float
/// dtypes. Direct concrete calls retain their existing diagnostics.
#[test]
fn every_family_policy_route_rejects_an_inadmissible_instantiation() {
    for (body, elements, family, dtype) in [
        (
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_01.deep"),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_10.deep"),
            "Float",
            "i32",
        ),
        (
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_02.deep"),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_10.deep"),
            "Float",
            "i32",
        ),
        (
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_03.deep"),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_11.deep"),
            "Int",
            "f32",
        ),
        (
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_04.deep"),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_12.deep"),
            "Float",
            "i32",
        ),
    ] {
        let program =
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_10.deep")
                .replace("@@family@@", &family.to_ascii_lowercase())
                .replace("@@body@@", body)
                .replace("@@elements@@", elements);
        let errors = check(&program).expect_err("inadmissible instantiation");
        assert_family_error(&errors, family, dtype);
    }
    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_127.deep"
    ))
    .expect("Numeric admits integer arithmetic");
    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_081.deep"
    ))
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
         cannot be instantiated at `i32`";

    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_128.deep"
    ))
    .expect("the signature's own bound makes the declaration well typed");

    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_129.deep"
    ))
    .expect_err("the integer instantiation must be rejected");
    assert!(
        errors.iter().any(|e| e.message.contains(FAMILY)),
        "dropout's caller must be caught by the propagated bound:\n{}",
        summary(&errors)
    );

    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_130.deep"
    ))
    .expect("the signature's own bound makes the declaration well typed");

    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_131.deep"
    ))
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
/// `a36c7c581` the first was REJECTED naming `i32`, a dtype the call's
/// operand does not carry, and the second was ACCEPTED at score 1 with the
/// compiled C lane printing `f = 2` for a true 2.3333333.
#[test]
fn a_local_binding_shadowing_a_top_level_one_is_read_from_the_local() {
    const HELPER: &str =
        include_str!("fixtures/core/tests/unresolved_operand_matrix/case_123.deep");

    check(
        &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_11.deep")
            .replace("@@HELPER@@", HELPER),
    )
    .expect("the local f32 binding is what `g` receives, so the program is well typed");

    let errors = check(
        &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_12.deep")
            .replace("@@HELPER@@", HELPER),
    )
    .expect_err("the local i32 binding is what `g` receives, so the program is rejected");
    assert!(
        errors.iter().any(|e| e.message.contains(
            "type variable bounded by dtype family `Float` (the active float dtypes) cannot be instantiated at `i32`"
        )),
        "the rejection must name the LOCAL binding's dtype:\n{}",
        summary(&errors)
    );
}

/// Binding references retain the scope in which their value was defined.
/// The two dtype orders catch false acceptance as well as false rejection.
#[test]
fn argument_precision_respects_sequential_and_top_level_binding_scopes() {
    const HELPER: &str =
        include_str!("fixtures/core/tests/unresolved_operand_matrix/case_123.deep");
    for (first, second, rejects) in [
        (
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_13.deep"),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_12.deep"),
            false,
        ),
        (
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_12.deep"),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_13.deep"),
            true,
        ),
    ] {
        for program in [
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_13.deep")
                .replace("@@HELPER@@", HELPER)
                .replace("@@first@@", first)
                .replace("@@second@@", second),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_14.deep")
                .replace("@@HELPER@@", HELPER)
                .replace("@@first@@", first)
                .replace("@@second@@", second),
        ] {
            let result = check(&program);
            if rejects {
                let errors = result.expect_err("the referenced binding carries i32");
                assert!(
                    errors.iter().any(|error| error
                        .message
                        .contains("type variable bounded by dtype family `Float` (the active float dtypes) cannot be instantiated at `i32`")),
                    "{}",
                    summary(&errors)
                );
            } else {
                result.expect("the referenced binding carries f32");
            }
        }
    }
}

/// Numeric arithmetic excludes bool through its checked scheme. The original
/// unbounded forms were accepted on `0820ee28e`; the bounded forms below
/// isolate the call-site rejection and its integer positive control.
#[test]
fn bool_arithmetic_through_an_unbounded_binder_is_rejected_at_the_instantiation() {
    for (op, body) in [
        (
            "add",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_05.deep"),
        ),
        (
            "sub",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_06.deep"),
        ),
        (
            "mul",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_07.deep"),
        ),
        (
            "neg",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_08.deep"),
        ),
        (
            "floor_div",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_09.deep"),
        ),
    ] {
        let program =
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_15.deep")
                .replace("@@body@@", body);
        let Err(errors) = check(&program) else {
            panic!("`{op}` on a bool instantiation must be rejected");
        };
        assert_family_error(&errors, "Numeric", "bool");
        check(
            &include_str!("fixtures/core/tests/unresolved_operand_matrix/template_16.deep")
                .replace("@@body@@", body),
        )
        .expect("the same arithmetic at i32 is admitted");
    }
}

/// Former body-reader gaps now carry restrictions through ordinary checking:
/// higher-order values, a transitive wrapper, an identity argument, a local
/// lambda, and an ADT field. Each has an invalid integer and valid float case.
/// The original unbounded witnesses accepted on `0820ee28e` and printed a
/// truncated C mean; the CLI target also requires those originals to fail.
#[test]
fn the_spellings_the_argument_reader_does_not_resolve_remain_unreached() {
    // Historical identity retained for the class oracle. Every former residual
    // is now a rejection, paired with the same bounded program at f32.
    for (dtype, values, accepts) in [
        (
            "i32",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_12.deep"),
            false,
        ),
        (
            "f32",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_13.deep"),
            true,
        ),
    ] {
        let helper = include_str!("fixtures/core/tests/unresolved_operand_matrix/case_123.deep");
        for program in [
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_17.deep")
                .replace("@@helper@@", helper)
                .replace("@@dtype@@", dtype)
                .replace("@@values@@", values),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_18.deep")
                .replace("@@helper@@", helper)
                .replace("@@values@@", values),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_19.deep")
                .replace("@@helper@@", helper)
                .replace("@@values@@", values),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_20.deep")
                .replace("@@values@@", values),
            include_str!("fixtures/core/tests/unresolved_operand_matrix/template_21.deep")
                .replace("@@dtype@@", dtype)
                .replace("@@helper@@", helper)
                .replace("@@values@@", values),
        ] {
            let result = check(&program);
            if accepts {
                result.unwrap_or_else(|errors| panic!("{program}\n{}", summary(&errors)));
            } else {
                let errors = result.expect_err("checked restrictions survive every former gap");
                assert_family_error(&errors, "Float", dtype);
            }
        }
    }
}

/// An empty literal whose dtype is determined by the declared result must
/// satisfy the operation's family. The restriction travels on the operand,
/// without replaying a precision validator or inspecting the callee's body.
#[test]
fn an_empty_literal_the_declared_result_determines_is_now_validated() {
    for (route, program, _diagnostic) in [
        (
            "softmax",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_133.deep"),
            "softmax on operand precision `i32` is not admitted",
        ),
        (
            "sqrt",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_134.deep"),
            "sqrt on operand precision `i32` is not admitted",
        ),
        (
            "add",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_135.deep"),
            "add on bool operands is not admitted",
        ),
    ] {
        let errors = check(program).expect_err(&format!(
            "{route}: an empty literal the declaration determines must now be validated"
        ));
        assert_family_error(
            &errors,
            if route == "add" { "Numeric" } else { "Float" },
            if route == "add" { "bool" } else { "i32" },
        );
    }

    // NEGATIVE TWIN. The narrowing is a dtype verdict, not a rejection of the
    // empty literal: the same programs at an admissible dtype still check.
    // DISPOSITION LOCK, accepted on `6dbbbf2bc` and here.
    for (route, program) in [
        (
            "softmax",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_136.deep"),
        ),
        (
            "sqrt",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_137.deep"),
        ),
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
    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_138.deep"
    ))
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
        include_str!("fixtures/core/tests/unresolved_operand_matrix/template_22.deep")
            .replace("@@declared@@", declared)
    };

    expect_exact_error(
        &program("(t-tensor {} (d-lit {} 100) (t-prim {} f32))"),
        &signature_mismatch("tensor[4, 3, f32]", "tensor[3, f32]", "tensor[100, f32]"),
    );

    // NEGATIVE TWIN: the shape the route actually produces still checks, so
    // the repair rejects a wrong claim rather than the destructuring form.
    check(&program("(t-tensor {} (d-lit {} 3) (t-prim {} f32))"))
        .unwrap_or_else(|e| panic!("the true result shape must still check:\n{}", summary(&e)));

    // The operand that NEVER binds is the acceptance boundary, not a pass:
    // the lambda is never applied, so no application supplies `q`'s
    // constructor. Measured accepting at score 1 on `6abca2406`.
    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_140.deep"
    ))
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
        include_str!("fixtures/core/tests/unresolved_operand_matrix/template_23.deep")
            .replace("@@declared@@", declared)
    };

    expect_exact_error(
        &program("(t-tensor {} (d-lit {} 100) (d-lit {} 5) (t-prim {} f32))"),
        &signature_mismatch(
            "tensor[4, 3, f32], tensor[3, 5, f32]",
            "tensor[4, 5, f32]",
            "tensor[100, 5, f32]",
        ),
    );

    // NEGATIVE TWIN.
    check(&program(
        "(t-tensor {} (d-lit {} 4) (d-lit {} 5) (t-prim {} f32))",
    ))
    .unwrap_or_else(|e| {
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
        include_str!("fixtures/core/tests/unresolved_operand_matrix/template_24.deep")
            .replace("@@declared@@", declared)
    };

    expect_exact_error(
        &program("(t-tensor {} (d-lit {} 100) (t-prim {} f32))"),
        &signature_mismatch("tensor[4, 3, f32]", "tensor[3, f32]", "tensor[100, f32]"),
    );

    // NEGATIVE TWIN.
    check(&program("(t-tensor {} (d-lit {} 3) (t-prim {} f32))")).unwrap_or_else(|e| {
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
        include_str!("fixtures/core/tests/unresolved_operand_matrix/template_25.deep")
            .replace("@@declared@@", declared)
    };

    expect_exact_error(
        &program("(t-tensor {} (d-lit {} 100) (t-prim {} f32))"),
        &signature_mismatch("tensor[4, 3, f32]", "tensor[3, f32]", "tensor[100, f32]"),
    );

    // NEGATIVE TWIN.
    check(&program("(t-tensor {} (d-lit {} 3) (t-prim {} f32))")).unwrap_or_else(|e| {
        panic!(
            "the true result shape behind a copy gate must still check:\n{}",
            summary(&e)
        )
    });

    // The unapplied form keeps `copy`'s own rejection AND gains the boundary
    // obligation: the gate never discharges, so its consumer never settles.
    // Measured on `6abca2406` reporting only the `copy` diagnostic.
    let errors = check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_141.deep"
    ))
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_142.deep"),
            "ArityMismatch",
            "tuple length mismatch: 2 vs 3",
        ),
        (
            "tuple pattern on a scrutinee that binds to a tensor",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_143.deep"),
            "TypeMismatch",
            "type mismatch: tensor[4, 3, f32] vs (",
        ),
        (
            "field read on a target that binds to a tensor",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_145.deep"),
            "TypeMismatch",
            "field access `.x` expects a record value, got a value of type `tensor[4, 3, f32]` (chelis#755)",
        ),
        (
            "unknown field on a target that binds to a record",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_146.deep"),
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
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_145.deep"),
        ),
        (
            "unknown field on a target that binds to a record",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_146.deep"),
        ),
        (
            "multi-variant target",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_147.deep"),
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
    check(include_str!(
        "fixtures/core/tests/unresolved_operand_matrix/case_148.deep"
    ))
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
        include_str!("fixtures/core/tests/unresolved_operand_matrix/template_26.deep")
            .replace("@@declared@@", declared)
    };

    // Measured score 1, `errors []` on `e0a482248`.
    expect_exact_error(
        &program("(t-tensor {} (d-lit {} 99) (t-prim {} f32))"),
        &signature_mismatch(
            "tensor[4, 3, f32], tensor[2, i32]",
            "tensor[3, f32]",
            "tensor[99, f32]",
        ),
    );

    // NEGATIVE TWIN.
    check(&program("(t-tensor {} (d-lit {} 3) (t-prim {} f32))")).unwrap_or_else(|e| {
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
        include_str!("fixtures/core/tests/unresolved_operand_matrix/template_27.deep")
            .replace("@@declared@@", declared)
    };

    // Measured score 1, `errors []` on `e0a482248`.
    expect_exact_error(
        &program("(t-tensor {} (d-lit {} 99) (t-prim {} f32))"),
        &signature_mismatch("tensor[4, 4, 3, f32]", "tensor[, f32]", "tensor[99, f32]"),
    );

    // NEGATIVE TWIN, at the rank-zero declaration.
    check(&program("(t-tensor {} (t-prim {} f32))")).unwrap_or_else(|e| {
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
        include_str!("fixtures/core/tests/unresolved_operand_matrix/template_28.deep")
            .replace("@@declared@@", declared)
    };

    // Measured score 1, `errors []` on `e0a482248`.
    expect_exact_error(
        &program("(t-tensor {} (d-lit {} 99) (t-prim {} f32))"),
        &signature_mismatch(
            "tensor[4, 3, f32], tensor[2, i32], tensor[2, 3, f32]",
            "tensor[3, f32]",
            "tensor[99, f32]",
        ),
    );

    // NEGATIVE TWIN.
    check(&program("(t-tensor {} (d-lit {} 3) (t-prim {} f32))")).unwrap_or_else(|e| {
        panic!(
            "scatter_replace's true result shape behind its gate must still check:\n{}",
            summary(&e)
        )
    });
}

/// Spec/03 section 6.2 lexical binders cannot capture a substituted helper parameter.
/// Round 2: the four shadowing witnesses reject on 1dd0f547 but accept on
/// its base; alpha-renaming alone restores acceptance there.
#[test]
fn restricted_body_precision_follows_lexical_bindings() {
    for (case, program) in [
        (
            "body_local_shadow",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_149.deep"),
        ),
        (
            "body_lambda_shadow",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_150.deep"),
        ),
        (
            "body_callee_shadow",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_151.deep"),
        ),
        (
            "body_match_shadow",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_152.deep"),
        ),
        (
            "body_renamed_local",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_153.deep"),
        ),
        (
            "body_match_renamed",
            include_str!("fixtures/core/tests/unresolved_operand_matrix/case_154.deep"),
        ),
    ] {
        check(program).unwrap_or_else(|errors| panic!("{case}: {}", summary(&errors)));
    }
    // A captured parameter and its sequential alias retain the substituted
    // dtype, even after another binding reuses its name.
    for body in [
        include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_14.deep"),
        include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_15.deep"),
        include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_16.deep"),
    ] {
        for (dtype, values, rejects) in [
            (
                "i32",
                include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_12.deep"),
                true,
            ),
            (
                "f32",
                include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_13.deep"),
                false,
            ),
        ] {
            let program =
                include_str!("fixtures/core/tests/unresolved_operand_matrix/template_29.deep")
                    .replace("@@body@@", body)
                    .replace("@@values@@", values);
            let result = check(&program);
            if rejects {
                let errors = result.expect_err("the captured parameter still carries i32");
                assert_family_error(&errors, "Float", dtype);
            } else {
                result.expect("the captured float parameter remains admissible");
            }
        }
    }
}

/// [05-OP-43] and the operation atoms govern concrete instantiations as well
/// as declarations. Every policy family has an invalid and valid call here.
#[test]
fn every_family_route_checks_concrete_helper_instantiations() {
    for op in [
        "mean",
        "softmax",
        "exp",
        "log",
        "sin",
        "tan",
        "atan",
        "sqrt",
        "relu",
        "sigmoid",
        "tanh",
        "silu",
        "gelu",
        "recip",
        "div",
        "trunc_div",
    ] {
        let body = match op {
            "mean" | "softmax" => format!(
                "(app {{}} (var {{}} {op}) (var {{}} x) (lit {{type: (t-prim {{}} i32)}} 0))"
            ),
            "div" | "trunc_div" => format!("(app {{}} (var {{}} {op}) (var {{}} x) (var {{}} x))"),
            _ => format!("(app {{}} (var {{}} {op}) (var {{}} x))"),
        };
        let result_type = if op == "mean" {
            "(t-tensor {} (t-var {} p))"
        } else {
            "(t-tensor {} (d-lit {} 3) (t-var {} p))"
        };
        let family = if op == "trunc_div" { "Int" } else { "Float" };
        for (values, float) in [
            (
                include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_12.deep"),
                false,
            ),
            (
                include_str!("fixtures/core/tests/unresolved_operand_matrix/fragment_13.deep"),
                true,
            ),
        ] {
            let program =
                include_str!("fixtures/core/tests/unresolved_operand_matrix/template_30.deep")
                    .replace("@@family@@", &family.to_ascii_lowercase())
                    .replace("@@result_type@@", result_type)
                    .replace("@@body@@", &body)
                    .replace("@@values@@", values);
            let result = check(&program);
            if float == (op == "trunc_div") {
                let errors =
                    result.expect_err(&format!("{op}: inadmissible concrete instantiation"));
                assert_family_error(&errors, family, if float { "f32" } else { "i32" });
            } else {
                result.unwrap_or_else(|errors| panic!("{op}: {}", summary(&errors)));
            }
        }
    }
}

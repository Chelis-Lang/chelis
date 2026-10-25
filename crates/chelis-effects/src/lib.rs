use chelis_deep::DeepTag;

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
use chelis_deep::Span;
#[cfg(test)]
use chelis_deep::annotations::MetadataValue;
use chelis_deep::ast::Expr;
#[cfg(test)]
use chelis_deep::ast::Metadata;
use chelis_deep::{ExprCarrier, decode_effect_kind, string_literal};
#[cfg(test)]
use chelis_effects_core::EffectWorkProfile;
use chelis_effects_core::{DefEffectRows, EffectError, EffectErrorKind, infer_effect_rows};
#[cfg(test)]
use chelis_type_core::{Effect, EffectSet};
use chelis_types::{CheckedProgram, InferResult};
use chelis_vocab::EffectKind;
use chelis_vocab::EffectKindDecodeError;

pub mod realizability;

#[cfg(test)]
thread_local! {
    static EFFECT_WORK_PROFILE: RefCell<EffectWorkProfile> =
        RefCell::new(EffectWorkProfile::default());
}

#[cfg(test)]
fn reset_effect_work_profile() {
    EFFECT_WORK_PROFILE.with(|profile| *profile.borrow_mut() = EffectWorkProfile::default());
}

#[cfg(test)]
fn take_effect_work_profile() -> EffectWorkProfile {
    EFFECT_WORK_PROFILE.with(|profile| std::mem::take(&mut *profile.borrow_mut()))
}

fn analyze_effect_annotations(
    expressions: &[Expr],
    library: Option<&[Expr]>,
) -> Result<Vec<Expr>, Vec<EffectError>> {
    #[cfg(test)]
    {
        let (result, profile) =
            chelis_effects_core::analyze_effects_with_profile(expressions, library);
        EFFECT_WORK_PROFILE.with(|current| *current.borrow_mut() = profile);
        result
    }
    #[cfg(not(test))]
    {
        chelis_effects_core::analyze_effects(expressions, library)
    }
}

fn reconstruct_effect_annotations(
    program: &CheckedProgram,
    annotated_exprs: Vec<Expr>,
) -> Result<CheckedProgram, Vec<EffectError>> {
    program
        .try_with_effect_annotations(annotated_exprs)
        .map_err(type_totality_errors)
}

/// Compute the inferred effect row for every top-level `def` in a
/// checked program, keyed by def name.
///
/// This runs the same iterative-fixed-point inference that
/// [`check_program`] uses internally, but exposes the per-def effect
/// rows directly instead of folding them into validation. It performs
/// NO validation — callers that need handler-arity /
/// declared-vs-inferred checks must still call [`check_program`].
///
/// The intended consumer is `chelis check --show-inferred --json`,
/// which joins these effect rows with the type-level signature
/// inference so a machine consumer (Hull) can reconstruct each
/// function's `(Type, EffectRow)` without re-parsing a printer.
///
/// `program` must be the whole program. A caller holding only a package
/// extension must use [`def_effect_rows_in_context`] and pass the
/// library the extension was checked against; see chelis#606.
pub fn def_effect_rows(program: &CheckedProgram) -> DefEffectRows {
    def_effect_rows_in_context(None, program)
}

/// The reporting counterpart to [`check_effects_with_context`]: the
/// inferred effect rows of `program`, resolved against `library`.
///
/// An imported callee's effects propagate to its caller exactly as a
/// same-package callee's do (spec/04-type-system.md §7.1: a function's
/// inferred effect set is the union of the effects of the operations in
/// its body, and stdlib helpers inherit effects through calls). Nothing
/// in that rule is package-relative, so the row a package boundary
/// reports must equal the row enforcement discharges.
pub fn def_effect_rows_in_context(
    library: Option<&CheckedProgram>,
    program: &CheckedProgram,
) -> DefEffectRows {
    infer_effect_rows(
        program.annotated_exprs(),
        library.map(CheckedProgram::annotated_exprs),
    )
}

pub fn check_program(program: &CheckedProgram) -> Result<CheckedProgram, Vec<EffectError>> {
    let annotated_exprs = analyze_effect_annotations(program.annotated_exprs(), None)?;
    reconstruct_effect_annotations(program, annotated_exprs)
}

/// Check effects on `new_program` against an outer-scope library context.
///
/// This is the Phase D context-extension counterpart to [`check_program`].
/// `library_program` must already have been validated by [`check_program`];
/// its effect rows are treated as authoritative and are NOT re-validated.
///
/// The new-code effect rows are inferred against the union of the library's
/// effect map and the new-code's own iterative fixed-point pass. When new
/// code calls a library function, the library's effect row is inherited.
///
/// Validation passes (handler arity, declared-vs-inferred) run ONLY on the new code's annotated expressions. Library
/// validation already happened during the original [`check_program`] call.
///
/// The library's effect map is computed inside this function from
/// `library_program.annotated_exprs()`. It is NOT persisted across calls;
/// the function takes `&CheckedProgram` (not `&mut`) so subsequent calls
/// against the same `library_program` see a fresh effect map untouched by
/// any prior new-code inference.
///
/// ## Returned program
///
/// The returned [`CheckedProgram`] contains ONLY the new-code's annotated
/// expressions, with effect metadata composed against the library. The
/// `type_env()` is preserved verbatim from `new_program`.
pub fn check_effects_with_context(
    library_program: &CheckedProgram,
    new_program: &CheckedProgram,
) -> Result<CheckedProgram, Vec<EffectError>> {
    let annotated_exprs = analyze_effect_annotations(
        new_program.annotated_exprs(),
        Some(library_program.annotated_exprs()),
    )?;
    reconstruct_effect_annotations(new_program, annotated_exprs)
}

fn type_totality_errors(result: InferResult) -> Vec<EffectError> {
    if result.errors.is_empty() {
        return vec![EffectError {
            kind: EffectErrorKind::TypeTotality,
            message: "internal: checked-program reconstruction failed without a type diagnostic"
                .to_string(),
            suggestions: vec![],
        }];
    }
    result
        .errors
        .into_iter()
        .map(|error| EffectError {
            kind: EffectErrorKind::TypeTotality,
            message: error.message,
            suggestions: error.suggestions,
        })
        .collect()
}

/// Validate that the program's target-relevant constructs are admissible
/// for the requested build target.
///
/// Today this validates:
///   * resource-region pinning (host C admits only exact `cpu`; every other
///     selector requires a target capability that defines its semantics).
///
/// Per spec/04-type-system.md §1.1.3 the Metal target additionally
/// rejects FP64 because Apple Silicon GPUs lack FP64 ALUs (software
/// emulation explicitly out of scope). The spec names three rejection
/// surfaces: the CLI gate (`reject_unsupported_metal_ops` in
/// `chelis-cli`), the IR validation pass
/// (`chelis_ir::device_admission::validate_metal_admissible_precisions`), and
/// the codegen entry (`Emitter::require_metal_admissible` in
/// `chelis-backend-metal`). The IR pass walks the lowered DAG (which
/// is post-typecheck and out of scope here) and the CLI invokes it
/// between type-check and codegen so the f64-on-metal rejection
/// surfaces with the spec-pinned diagnostic at every entry point.
/// This function does not duplicate that DAG walk; it complements it
/// by validating handler-region pinning at the AST surface.
pub fn validate_build_target(
    program: &CheckedProgram,
    target: &str,
) -> Result<(), Vec<EffectError>> {
    validate_build_target_expressions(program.annotated_exprs(), target)
}

/// Check Resource regions in an emission scope selected from checked source.
///
/// This shares the whole-program validator's target classification. Callers
/// selecting an entry must include its transitive source dependencies before
/// lowering erases the handlers. This validation does not establish that an
/// arbitrary expression slice is well typed or is the correct emission scope.
pub fn validate_build_target_expressions(
    expressions: &[Expr],
    target: &str,
) -> Result<(), Vec<EffectError>> {
    let mut errors = Vec::new();
    for expr in expressions {
        validate_build_target_expr(expr, target, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_build_target_expr(expr: &Expr, target: &str, errors: &mut Vec<EffectError>) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            if tag == DeepTag::HandleEffect {
                validate_build_target_handler(
                    decode_effect_kind(metadata),
                    children,
                    target,
                    errors,
                );
            }
            metadata.visit_expressions(&mut |value, _| {
                validate_build_target_expr(value, target, errors)
            });
            for kid in children {
                validate_build_target_expr(kid, target, errors);
            }
        }
        ExprCarrier::MetadataMap(map) => {
            map.visit_expressions(&mut |value, _| {
                validate_build_target_expr(value, target, errors)
            });
        }
        ExprCarrier::MetadataExpression(meta) => {
            validate_build_target_expr(&meta.expr, target, errors);
            meta.metadata.visit_expressions(&mut |value, _| {
                validate_build_target_expr(value, target, errors)
            });
        }
        ExprCarrier::Atom(_) => {}
        ExprCarrier::StructuralList(children) => {
            for child in children {
                validate_build_target_expr(child, target, errors);
            }
        }
        ExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                validate_build_target_expr(child, target, errors);
            }
        }
    }
}

fn validate_build_target_handler(
    effect_kind: Result<EffectKind, EffectKindDecodeError<'_>>,
    kids: &[Expr],
    target: &str,
    errors: &mut Vec<EffectError>,
) {
    match effect_kind {
        Ok(EffectKind::Resource) => {
            if let Some(device) = kids.first().and_then(string_literal) {
                let ok = match target {
                    "c" => is_c_host_device_designator(device),
                    "hip" | "metal" => device.starts_with("gpu"),
                    _ => true,
                };
                if !ok {
                    errors.push(EffectError {
                        kind: EffectErrorKind::BuildTargetMismatch,
                        message: if target == "c" {
                            format!(
                                "`chelis build --target c` cannot satisfy resource region \
                                 `{device}`: host C accepts only exact `cpu`"
                            )
                        } else {
                            format!(
                                "`chelis build --target {target}` cannot satisfy resource region \
                                 `{device}`"
                            )
                        },
                        suggestions: match target {
                            "c" => vec![
                                "Use `with device(\"cpu\") { ... }` for host execution; \
                                 accelerator placement is a separate target capability \
                                 (chelis#2104)"
                                    .to_string(),
                            ],
                            "hip" | "metal" => vec![
                                "Use a GPU device such as `with device(\"gpu:0\") { ... }`"
                                    .to_string(),
                            ],
                            _ => vec![],
                        },
                    });
                }
            }
        }
        // `validate_handlers` is the owning structural diagnostic pass. Do
        // not assign target semantics after decode fails.
        Err(_) => {}
    }
}

/// The host-C selector contract owned by spec/04 [04-EFF-2].
///
/// This is deliberately a positive host classification. Any new accelerator
/// spelling therefore fails closed instead of inheriting host execution merely
/// because it lacks a known prefix.
fn is_c_host_device_designator(device: &str) -> bool {
    device == "cpu"
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::parser::parse_str;
    use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};

    fn checked(src: &str) -> CheckedProgram {
        let exprs = parse_str(src, &chelis_syntax_host::NativeStack).unwrap();
        let checked = chelis_types::check_ir_program(&exprs).unwrap();
        check_program(&checked).unwrap()
    }

    fn surf_checked(src: &str) -> CheckedProgram {
        let decls = parse_surf(src, &chelis_syntax_host::NativeStack).expect("surf parse");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        check_program(&checked).expect("effect check")
    }

    fn issue_1205_source(operations: usize, flat: bool) -> String {
        let module = if flat { "Flat" } else { "Nested" };
        let mut lines = vec![
            format!("module FrontEndPerformance.{module}N{operations}"),
            "def bc(c: f32) -> tensor[8, f32] = \
             reshape(insert(to_tensor([c]), 0, 8i64), [8i64])"
                .to_string(),
        ];
        if flat {
            lines.push(
                "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else {"
                    .to_string(),
            );
            let mut previous = "s".to_string();
            for index in 0..operations {
                lines.push(format!(
                    "  t{index} = mul(add({previous}, bc(cast(1.0, f32))), \
                     bc(cast(0.5, f32)))"
                ));
                previous = format!("t{index}");
            }
            lines.extend([format!("  st({previous}, add(i, 1i64))"), "}".to_string()]);
        } else {
            let mut body = "s".to_string();
            for _ in 0..operations {
                body = format!("mul(add({body}, bc(cast(1.0, f32))), bc(cast(0.5, f32)))");
            }
            lines.push(format!(
                "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else st({body}, add(i, 1i64))"
            ));
        }
        lines.push("r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)".to_string());
        lines.join("\n") + "\n"
    }

    fn issue_1205_effect_profile(operations: usize, flat: bool) -> EffectWorkProfile {
        std::thread::Builder::new()
            .name(format!("issue-1205-effects-{operations}"))
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let source = issue_1205_source(operations, flat);
                let decls = parse_surf(&source, &chelis_syntax_host::NativeStack)
                    .expect("#1205 surf fixture parses");
                let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
                    .expect("Surf fixture must desugar");
                let typed =
                    chelis_types::check_ir_program(&deep).expect("#1205 fixture type checks");
                reset_effect_work_profile();
                check_program(&typed).expect("#1205 fixture passes effect checking");
                take_effect_work_profile()
            })
            .expect("#1205 effect profile thread starts")
            .join()
            .expect("#1205 effect profile thread completes")
    }

    #[test]
    fn issue_1205_effect_clone_work_is_linear() {
        for flat in [false, true] {
            let mut previous = None;
            for operations in [20, 40, 80, 160] {
                let profile = issue_1205_effect_profile(operations, flat);
                let clone_work = profile.node_bridge_clone_nodes
                    + profile.list_rewrite_clone_nodes
                    + profile.top_level_body_clone_nodes;
                let total_work = clone_work
                    + profile.infer_expr_visits
                    + profile.annotation_expr_visits
                    + profile.metadata_rewrites;
                eprintln!(
                    "#1205 effects shape={} n={operations} profile={profile:?} clone_work={clone_work}",
                    if flat { "flat" } else { "nested" }
                );
                assert_eq!(
                    clone_work, 0,
                    "#1205 effects must not clone descendant-bearing expression trees"
                );
                if let Some(previous) = previous {
                    assert!(
                        total_work * 10 <= previous * 22,
                        "#1205 effect clone work must grow linearly: shape={} n={operations} \
                         previous={previous} current={total_work} profile={profile:?}",
                        if flat { "flat" } else { "nested" }
                    );
                }
                previous = Some(total_work);
            }
        }
    }

    #[test]
    fn effect_annotation_reconstruction_preserves_type_context() {
        let decls = parse_surf(
            "def add_one(x: i32) -> i32 = add(x, 1)",
            &chelis_syntax_host::NativeStack,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        let typed = chelis_types::check_ir_program(&deep).expect("type check");
        let expected_type_env = typed.type_env().clone();
        let expected_signatures = typed.signature_inference().clone();

        let reconstructed = check_program(&typed).expect("effect check");
        assert_eq!(reconstructed.type_env(), &expected_type_env);
        assert_eq!(reconstructed.signature_inference(), &expected_signatures);
    }

    #[test]
    fn core_success_cannot_hide_checked_reconstruction_failure() {
        let original = checked("(def {} answer (lit {type: (t-prim {} i32)} 1))");
        let other = checked("(def {} answer (lit {type: (t-prim {} i32)} 2))");
        let annotations = chelis_effects_core::analyze_effects(other.annotated_exprs(), None)
            .expect("portable analysis accepts the other program");
        let errors = reconstruct_effect_annotations(&original, annotations)
            .expect_err("portable success cannot authorize another program");
        assert!(!errors.is_empty());
        assert!(
            errors
                .iter()
                .all(|error| error.kind == EffectErrorKind::TypeTotality)
        );
    }

    #[test]
    fn effect_annotation_reconstruction_preserves_checked_local_owners() {
        let source = "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {\n f = if flag then is_nan else is_finite\n g: &tensor[2, f64] -> tensor[*, bool] = copy(f)\n g(x)\n}\n";
        let decls =
            parse_surf(source, &chelis_syntax_host::NativeStack).expect("Surf fixture parses");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture desugars");
        let typed = chelis_types::check_ir_program(&deep).expect("fixture type checks");
        assert_eq!(
            typed.local_ascriptions().len(),
            1,
            "fixture requires a real checked owner"
        );
        let reconstructed =
            check_program(&typed).expect("effect annotations preserve owner authority");
        assert_eq!(reconstructed.local_ascriptions(), typed.local_ascriptions());
        assert_eq!(reconstructed.type_env(), typed.type_env());
        assert_eq!(
            reconstructed.signature_inference(),
            typed.signature_inference()
        );
    }

    #[test]
    fn stamped_effect_annotation_preserves_the_checked_node_carrier() {
        let deep = chelis_deep::parse_and_stamp(
            "(def {} answer (fn {} (params {}) (lit {} 1)))",
            &chelis_syntax_host::NativeStack,
        )
        .expect("canonical Deep fixture stamps");
        let typed = chelis_types::check_typed_program(&deep).expect("type check");

        let reconstructed = check_program(&typed).expect("effect check");

        assert!(
            matches!(
                reconstructed.annotated_exprs().first(),
                Some(Expr::Node(..))
            ),
            "effect annotation must not normalize stamped checker output"
        );
    }

    #[test]
    fn stamped_declared_pure_function_rejects_inferred_io() {
        let deep = chelis_deep::parse_and_stamp(
            r#"(defsig {} entry
             (t-fn {eff: (effects {})}
               (t-prim {} string)
               (t-prim {} string)))
           (def {} entry
             (fn {}
               (params {} (x {type: (t-prim {} string)}))
               (app {} (var {} debug) (var {} x))))"#,
            &chelis_syntax_host::NativeStack,
        )
        .expect("canonical Deep fixture stamps");
        let typed = chelis_types::check_typed_program(&deep).expect("type check");

        let errors = check_program(&typed).expect_err("declared-pure IO body must reject");

        assert!(
            errors
                .iter()
                .any(|error| { error.message.contains("entry") && error.message.contains("IO") }),
            "effect diagnostic must name the function and missing effect: {errors:?}"
        );
    }

    #[test]
    fn undecodable_head_metadata_is_not_traversed_by_handler_validation() {
        fn stamped_def_body(source: &str) -> Expr {
            let exprs = chelis_deep::parse_and_stamp(source, &chelis_syntax_host::NativeStack)
                .expect("fixture stamps");
            let ExprCarrier::DecodedNode(DeepTag::Def, _, children) = exprs[0].carrier() else {
                panic!("fixture is a def");
            };
            children[1].clone()
        }

        fn expression_metadata(expr: Expr) -> Metadata {
            Metadata::from(MetadataValue::PropertySeed(
                chelis_deep::annotations::RuntimeExpression::try_new(expr)
                    .expect("handler is a runtime expression"),
            ))
        }

        fn unknown_form(metadata: Metadata, span: Span) -> Expr {
            Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                head: "future-wrapper".into(),
                meta: metadata,
                children: Vec::new(),
                span,
            }))
        }

        let span = Span::new(0, 0);
        let invalid_resource = stamped_def_body(
            "(def {} x
               (handle-effect {effect: resource}
                 (var {} device)
                 (lit {} 1)))",
        );
        let invalid_metadata = expression_metadata(invalid_resource);
        let unknown_errors =
            chelis_effects_core::analyze_effects(&[unknown_form(invalid_metadata, span)], None)
                .err()
                .unwrap_or_default();
        assert!(
            unknown_errors.is_empty(),
            "UnknownForm metadata was not traversed before the carrier migration: {unknown_errors:?}"
        );

        let gpu_resource = stamped_def_body(
            "(def {} x
               (handle-effect {effect: resource}
                 (lit {} \"gpu:0\")
                 (lit {} 1)))",
        );
        let resource_metadata = expression_metadata(gpu_resource);
        let mut unknown_target_errors = Vec::new();
        validate_build_target_expr(
            &unknown_form(resource_metadata, span),
            "c",
            &mut unknown_target_errors,
        );
        assert!(
            unknown_target_errors.is_empty(),
            "UnknownForm metadata was not traversed by target validation before migration: \
             {unknown_target_errors:?}"
        );
    }

    #[test]
    fn stamped_declared_pure_function_rejects_io_local_closure() {
        let deep = chelis_deep::parse_and_stamp(
            r#"(defsig {} entry
             (t-fn {eff: (effects {})}
               (t-prim {} string)
               (t-prim {} string)))
           (def {} entry
             (fn {}
               (params {} (x {type: (t-prim {} string)}))
               (let {}
                 (bind {} step
                   (fn {}
                     (params {} (y {type: (t-prim {} string)}))
                     (app {} (var {} debug) (var {} y))))
                 (app {} (var {} step) (var {} x)))))"#,
            &chelis_syntax_host::NativeStack,
        )
        .expect("canonical Deep fixture stamps");
        let typed = chelis_types::check_typed_program(&deep).expect("type check");

        let errors = check_program(&typed).expect_err("declared-pure local IO closure must reject");

        assert!(
            errors
                .iter()
                .any(|error| { error.message.contains("entry") && error.message.contains("IO") }),
            "effect diagnostic must name the function and local closure effect: {errors:?}"
        );
    }

    #[test]
    fn literal_resource_handler_crosses_the_type_effect_boundary() {
        let deep = parse_str(
            r#"(def {} value
               (handle-effect {effect: resource}
                 (lit {type: (t-prim {} string)} "cpu")
                 (lit {type: (t-prim {} i32)} 1)))"#,
            &chelis_syntax_host::NativeStack,
        )
        .expect("Deep handler fixture parses");
        let typed = chelis_types::check_ir_program(&deep).expect("type boundary accepts");
        check_program(&typed).expect("effects boundary accepts literal handler");
    }

    #[test]
    fn nonliteral_resource_handler_is_rejected_once_by_the_effect_owner() {
        let deep = parse_str(
            "(def {} value (handle-effect {effect: resource} \
         (var {} computed_handler) (lit {type: (t-prim {} i32)} 1)))",
            &chelis_syntax_host::NativeStack,
        )
        .expect("Deep handler fixture parses");
        let typed = chelis_types::check_ir_program(&deep)
            .expect("handler payload is owned by the effects gate");
        let errors = check_program(&typed).expect_err("nonliteral handler must reject");
        assert_eq!(errors.len(), 1, "one effects owner diagnostic: {errors:?}");
        assert_eq!(errors[0].kind, EffectErrorKind::InvalidHandler);
        assert!(
            errors[0]
                .message
                .contains("requires a string literal device"),
            "{errors:?}"
        );
    }

    // `def_effect_rows` is the public seam `chelis check --show-inferred
    // --json` reads to attach a structured effect row to each function.
    // Positive: a function calling `debug` carries IO. Negative parity:
    // a pure function carries an EMPTY row (present and empty, not
    // missing), so a consumer can tell "pure" from "unknown".
    #[test]
    fn def_effect_rows_reports_io_and_empty_rows() {
        let program = surf_checked(
            r#"
def logged(msg: string) -> string = debug(msg)
def pure_add(x: i64, y: i64) -> i64 = add(x, y)
"#,
        );
        let rows = def_effect_rows(&program);
        let logged = rows.get("logged").expect("logged effect row present");
        assert!(logged.contains(&Effect::Io));
        assert_eq!(logged.iter().count(), 1);

        let pure_add = rows.get("pure_add").expect("pure_add effect row present");
        assert!(
            pure_add.is_empty(),
            "pure function must have an empty effect row, got {pure_add}"
        );
    }

    #[test]
    fn keyed_dropout_root_checks_clean() {
        // The retired unhandled-`Random` root check has no successor: a keyed
        // draw at a value root needs no handler.
        let exprs = parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} dropout)
           (app {} (var {} key_from_seed) (lit {type: (t-prim {} i64)} 7))
           (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
            &chelis_syntax_host::NativeStack,
        )
        .unwrap();
        let checked = chelis_types::check_ir_program(&exprs).unwrap();
        check_program(&checked).expect("a keyed draw root needs no handler");
    }

    #[test]
    fn build_target_rejects_gpu_region_for_c() {
        let program = checked(
            "(def {} x
               (handle-effect {effect: resource}
                 (lit {type: (t-prim {} string)} \"gpu:0\")
                 (lit {type: (t-prim {} i32)} 1)))",
        );
        let errors = validate_build_target(&program, "c").unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.kind == EffectErrorKind::BuildTargetMismatch)
        );
    }

    #[test]
    fn c_target_accepts_only_explicit_host_resource_designators() {
        let program = surf_checked("def main() -> i32 = with device(\"cpu\") { 1 }\n");
        validate_build_target(&program, "c").expect("exact cpu host region");

        for device in [
            "cpu:0",
            "cpu:author-device",
            "cpu:socket_9",
            "cpu:HOST_2",
            "cuda:0",
            "metal",
            "rocm",
            "xpu:1",
            "Gpu:0",
            "gpu:0",
            "host",
            "",
            "cpu:",
            "cpu:two words",
            "cpu:/0",
        ] {
            let program = surf_checked(&format!(
                "def main() -> i32 = with device(\"{device}\") {{ 1 }}\n"
            ));
            let errors = validate_build_target(&program, "c").expect_err(device);
            assert_eq!(errors.len(), 1, "{device}: {errors:?}");
            assert_eq!(
                errors[0].kind,
                EffectErrorKind::BuildTargetMismatch,
                "{device}"
            );
            assert_eq!(
                errors[0].message,
                format!(
                    "`chelis build --target c` cannot satisfy resource region `{device}`: \
                     host C accepts only exact `cpu`"
                ),
                "{device}"
            );
            assert_eq!(
                errors[0].suggestions,
                vec![
                    "Use `with device(\"cpu\") { ... }` for host execution; accelerator \
                     placement is a separate target capability (chelis#2104)"
                        .to_string()
                ],
                "{device}"
            );
        }
    }

    #[test]
    fn c_target_checks_each_nested_resource_region() {
        for source in [
            r#"def main() -> i32 = with device("cpu") { with device("cuda:0") { 1 } }"#,
            r#"def main() -> i32 = with device("metal") { with device("cpu") { 1 } }"#,
            r#"def main() -> i32 = with device("cpu") { with device("cpu:author-device") { 1 } }"#,
            r#"def main() -> i32 = with device("cpu:socket_9") { with device("cpu") { 1 } }"#,
        ] {
            let program = surf_checked(source);
            let errors = validate_build_target(&program, "c").expect_err(source);
            assert_eq!(errors.len(), 1, "{source}: {errors:?}");
            assert_eq!(errors[0].kind, EffectErrorKind::BuildTargetMismatch);
        }

        let accepted =
            surf_checked(r#"def main() -> i32 = with device("cpu") { with device("cpu") { 1 } }"#);
        validate_build_target(&accepted, "c").expect("nested host regions");

        let two_rejected = surf_checked(
            r#"def main() -> i32 = with device("cpu:author-device") { with device("cpu:socket_9") { 1 } }"#,
        );
        let errors = validate_build_target(&two_rejected, "c").expect_err("two rejected regions");
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert_eq!(
            errors
                .iter()
                .map(|error| error.message.as_str())
                .collect::<Vec<_>>(),
            [
                "`chelis build --target c` cannot satisfy resource region \
                 `cpu:author-device`: host C accepts only exact `cpu`",
                "`chelis build --target c` cannot satisfy resource region \
                 `cpu:socket_9`: host C accepts only exact `cpu`",
            ]
        );
    }

    #[test]
    fn checked_program_does_not_emit_internal_type_override_metadata() {
        let program = checked(
            "(def {} f
               (fn {} (params {} (x {type: (t-prim {} string)}))
                 (app {} (var {} debug) (var {} x))))",
        );
        let text = chelis_deep::printer::print_canonical(program.annotated_exprs());
        assert!(
            !text.contains("__type_override"),
            "effect checker should not leak internal metadata, got:\n{text}"
        );
        assert!(
            text.contains("effects"),
            "expected inferred effect metadata on checked fn body, got:\n{text}"
        );
    }

    #[test]
    fn map_propagates_io_effect_from_callback() {
        let program = surf_checked(
            r#"
def emit(x: i64) -> i64 = debug(add(x, cast(1, i64)))
xs: List[i64] = [cast(1, i64), cast(2, i64)]
ys = map(emit, xs)
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("ys")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on map result, got {:?}",
            inferred.get("ys")
        );
    }

    #[test]
    fn fold_propagates_io_effect_from_inline_callback() {
        let program = surf_checked(
            r#"
xs: List[i64] = [cast(1, i64), cast(2, i64)]
total = fold(fn (acc: i64, x: i64) -> debug(add(acc, x)), cast(0, i64), xs)
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("total")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on fold result, got {:?}",
            inferred.get("total")
        );
    }

    #[test]
    fn scan_propagates_io_effect_from_inline_callback() {
        let program = surf_checked(
            r#"
xs: List[i64] = [cast(1, i64), cast(2, i64)]
totals = scan(fn (acc: i64, x: i64) -> debug(add(acc, x)), cast(0, i64), xs)
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("totals")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on scan result, got {:?}",
            inferred.get("totals")
        );
    }

    #[test]
    fn partition_propagates_io_effect_from_callback() {
        let program = surf_checked(
            r#"
def keep(x: i64) -> bool = gt(debug(x), cast(1, i64))
xs: List[i64] = [cast(1, i64), cast(2, i64)]
buckets = partition(keep, xs)
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("buckets")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on partition result, got {:?}",
            inferred.get("buckets")
        );
    }

    #[test]
    fn flat_map_propagates_io_effect_from_callback() {
        let program = surf_checked(
            r#"
xs: List[i64] = [cast(1, i64), cast(2, i64)]
ys = flat_map(fn (x: i64) -> debug([x, add(x, cast(10, i64))]), xs)
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("ys")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on flat_map result, got {:?}",
            inferred.get("ys")
        );
    }

    /// [04-INF-9] (chelis#3149): a builtin named as a value performs its
    /// call's effect, as a lambda that calls it does, so an alias cannot shed
    /// `IO` or `Test`. Before the fix `reader` and `caller` were pure.
    #[test]
    fn a_builtin_named_as_a_value_performs_its_call_effect() {
        let program = surf_checked(
            r#"
reader = read_file
def caller(path: string) -> string = reader(path)
def local_alias(path: string) -> string = {
  op = read_file
  op(path)
}
checker = test_assert
def via_lambda(path: string) -> string = {
  op = fn (p: string) -> read_file(p)
  op(path)
}
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        for (name, effect) in [
            ("reader", Effect::Io),
            ("caller", Effect::Io),
            ("local_alias", Effect::Io),
            ("via_lambda", Effect::Io),
            ("checker", Effect::Test),
        ] {
            assert!(
                inferred
                    .get(name)
                    .is_some_and(|effects| effects.contains(&effect)),
                "expected {effect:?} on `{name}`, got {:?}",
                inferred.get(name)
            );
        }
    }

    /// [04-INF-9]: for every builtin admitted as a value, an alias performs
    /// exactly the effect a call performs, read from the registry rather than
    /// a list here. The type-level contract of the same aliases is compared
    /// with the direct call in chelis-types (`builtin_value_contract_tests`).
    #[test]
    fn every_admitted_builtin_value_performs_exactly_its_call_effect() {
        let (env, _) = chelis_types::builtin_env();
        let mut checked = 0;
        for name in chelis_types::builtin_env_names() {
            let scheme = env.lookup(name).expect("bound builtin has a scheme");
            let Some(decl) = chelis_types::builtin_decl(name) else {
                continue;
            };
            if !chelis_types::builtin_value_contract_carried(decl, scheme) {
                continue;
            }
            let program = surf_checked(&format!("alias = {}\n", decl.name));
            let inferred = infer_effect_rows(program.annotated_exprs(), None);
            let mut expected = EffectSet::new();
            if let Some(effect) = chelis_effects_core::builtin_call_effect(decl.name) {
                expected.insert(effect);
            }
            assert_eq!(
                inferred.get("alias").cloned().unwrap_or_default(),
                expected,
                "`alias = {}` must perform exactly the call's effect",
                decl.name
            );
            checked += 1;
        }
        assert!(checked > 0, "no builtin is admitted as a value");
    }

    /// [04-INF-9]: a lexical binder that reuses an effectful builtin's name
    /// is an ordinary value, whether a parameter, a lambda parameter, a `let`
    /// name, a destructured name or a match binder, and referencing or calling
    /// it performs no builtin effect. The builtin itself, named outside the
    /// binder's scope, still performs its effect. Before the fix each
    /// shadowed function inferred `IO` or `Test`.
    #[test]
    fn a_lexical_binder_named_after_a_builtin_performs_no_builtin_effect() {
        let program = surf_checked(
            r#"
def param(read_file: string) -> string = read_file
def param_call(read_file: string -> string, p: string) -> string = read_file(p)
def lambda_param(p: string) -> string = {
  f = fn (print: string) -> print
  f(p)
}
def let_name(p: string) -> string = {
  debug = p
  debug
}
def destructured(p: (string, string)) -> string = {
  (read_file, other) = p
  read_file
}
def matched(o: Option[string]) -> string = match o with {
  | Some(test_assert) => test_assert
  | None => ""
}
def outside(p: string) -> string = {
  f = fn (read_file: string) -> read_file
  _ = f(p)
  read_file(p)
}
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        for name in [
            "param",
            "param_call",
            "lambda_param",
            "let_name",
            "destructured",
            "matched",
        ] {
            assert!(
                inferred.get(name).is_none_or(|effects| effects.is_empty()),
                "`{name}` binds a builtin's name and performs no effect, got {:?}",
                inferred.get(name)
            );
        }
        assert!(
            inferred
                .get("outside")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "`read_file` outside the lambda's scope is the builtin, got {:?}",
            inferred.get("outside")
        );
    }

    #[test]
    fn file_io_builtins_infer_io_but_mmap_reads_stay_pure() {
        let program = surf_checked(
            r#"
mapped = mmap_file("dataset.txt")
prefix = mmap_read(mapped, cast(0, i64), cast(4, i64))
width = mmap_len(mapped)
weights = mmap_tensor(mapped, cast(0, i64), cast(1, i64), i32)
text = mmap_text(mapped, cast(0, i64), cast(4, i64))
digest = mmap_sha256(mapped, cast(0, i64), cast(4, i64))
contents = read_file("dataset.txt")
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        // [05-OP-80] and [05-OP-81]: the reads of an open mapping are pure.
        for root in ["weights", "text", "digest"] {
            assert!(
                inferred.get(root).is_none_or(|effects| effects.is_empty()),
                "expected {root} to stay pure, got {:?}",
                inferred.get(root)
            );
        }
        assert!(
            inferred
                .get("mapped")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on mmap_file root, got {:?}",
            inferred.get("mapped")
        );
        assert!(
            inferred
                .get("prefix")
                .is_none_or(|effects| effects.is_empty()),
            "expected mmap_read to stay pure, got {:?}",
            inferred.get("prefix")
        );
        assert!(
            inferred
                .get("width")
                .is_none_or(|effects| effects.is_empty()),
            "expected mmap_len to stay pure, got {:?}",
            inferred.get("width")
        );
        assert!(
            inferred
                .get("contents")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on read_file root, got {:?}",
            inferred.get("contents")
        );
    }

    #[test]
    fn process_run_infers_io_but_pure_binding_stays_pure() {
        // Hull subprocess exec: a binding whose value applies process_run
        // acquires Effect::Io (mirroring read_file), while an adjacent pure
        // arithmetic binding stays effect-free.
        let program = surf_checked(
            r#"
result = process_run("echo", ["hi"])
pure_value = add(cast(1, i64), cast(2, i64))
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("result")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on process_run root, got {:?}",
            inferred.get("result")
        );
        assert!(
            inferred
                .get("pure_value")
                .is_none_or(|effects| effects.is_empty()),
            "expected pure arithmetic binding to stay effect-free, got {:?}",
            inferred.get("pure_value")
        );
    }

    #[test]
    fn clock_reads_infer_io_but_pure_binding_stays_pure() {
        // [05-OP-75]: each clock read is an IO source; an adjacent pure
        // binding stays effect-free.
        let program = surf_checked(
            r#"
wall = clock_wall_read()
mono = clock_monotonic_read()
pure_value = add(1i64, 2i64)
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        for root in ["wall", "mono"] {
            assert!(
                inferred
                    .get(root)
                    .is_some_and(|effects| effects.contains(&Effect::Io)),
                "expected IO effect on clock root {root}, got {:?}",
                inferred.get(root)
            );
        }
        assert!(
            inferred
                .get("pure_value")
                .is_none_or(|effects| effects.is_empty()),
            "expected pure arithmetic binding to stay effect-free, got {:?}",
            inferred.get("pure_value")
        );
    }

    #[test]
    fn declared_pure_function_cannot_read_a_clock() {
        // [05-OP-75] and spec/04 section 7: a function whose declared effects
        // omit IO cannot read either clock, directly or through a caller.
        // Declaring IO admits the same bodies.
        for clock in ["clock_wall_read", "clock_monotonic_read"] {
            for (declared, admitted) in [("{}", false), ("{Test}", false), ("{IO}", true)] {
                let decls = parse_surf(
                    &format!(
                        "def read() -> (i64, i64) ! {declared} = {clock}()\n\
                     def outer() -> (i64, i64) ! {declared} = read()\n"
                    ),
                    &chelis_syntax_host::NativeStack,
                )
                .expect("surf parse");
                let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
                    .expect("Surf fixture must desugar");
                let checked = chelis_types::check_ir_program(&deep).expect("type check");
                let verdict = check_program(&checked);
                if admitted {
                    assert!(verdict.is_ok(), "{clock} under ! {declared}: {verdict:?}");
                    continue;
                }
                let errors = verdict.expect_err("a clock read needs a declared IO effect");
                for function in ["read", "outer"] {
                    assert!(
                        errors.iter().any(|error| {
                            error.kind == EffectErrorKind::UnhandledEffect
                                && error.message.contains(&format!("`{function}`"))
                                && error.message.contains("IO")
                        }),
                        "expected UnhandledEffect on {function} mentioning IO for {clock} \
                         under ! {declared}, got {errors:?}"
                    );
                }
            }
        }
    }

    // ----- Phase 3t.1: Test effect propagation and enforcement -----

    #[test]
    fn test_assert_adds_test_effect_to_calling_fn() {
        // A top-level def whose body calls test_assert should acquire Effect::Test.
        let program = surf_checked(
            r#"
def my_check() -> unit = test_assert(true, "ok")
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("my_check")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "expected Test effect on my_check, got {:?}",
            inferred.get("my_check")
        );
    }

    #[test]
    fn test_assert_close_tensor_propagates_test_effect() {
        let program = surf_checked(
            r#"
def close() -> unit =
  test_assert_close_tensor(
    to_tensor([1.0, 2.0], f32),
    to_tensor([1.0, 2.0], f32),
    0.01,
    "close"
  )
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("close")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "expected Test effect on close(), got {:?}",
            inferred.get("close")
        );
    }

    #[test]
    fn test_effect_propagates_transitively_through_calls() {
        let program = surf_checked(
            r#"
def inner() -> unit = test_assert(true, "inner")
def outer() -> unit = inner()
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("outer")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "expected Test effect to transit inner -> outer, got {:?}",
            inferred.get("outer")
        );
    }

    #[test]
    fn with_device_does_not_handle_test_effect() {
        // A `resource` handler handles no inferred effect: `with device(...)`
        // must NOT remove Test.
        let program = surf_checked(
            r#"
def sealed() -> unit =
  with device("cpu") { test_assert(true, "inside-handler") }
"#,
        );
        let inferred = infer_effect_rows(program.annotated_exprs(), None);
        assert!(
            inferred
                .get("sealed")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "with device(...) must not swallow the Test effect, got {:?}",
            inferred.get("sealed")
        );
    }

    #[test]
    fn empty_effect_sig_rejects_test_assert_caller() {
        // `def g() -> unit ! {} = test_assert(...)` should fail: body inferred Test,
        // sig declares no effects.
        let decls = parse_surf(
            r#"
        def g() -> unit ! {} = test_assert(true, "leak")
        "#,
            &chelis_syntax_host::NativeStack,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        let errors = check_program(&checked)
            .expect_err("def with `! {}` that calls test_assert must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect && error.message.contains("Test")
            }),
            "expected UnhandledEffect mentioning Test, got {errors:?}"
        );
    }

    #[test]
    fn empty_effect_sig_rejects_transitive_test_caller() {
        // Caller has no assert in its body, but it calls a function that does.
        // The `! {}` signature on g must still reject this.
        let decls = parse_surf(
            r#"
        def f() -> unit = test_assert(true, "x")
        def g() -> unit ! {} = f()
        "#,
            &chelis_syntax_host::NativeStack,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        let errors =
            check_program(&checked).expect_err("transitive caller with `! {}` must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect
                    && error.message.contains("g")
                    && error.message.contains("Test")
            }),
            "expected UnhandledEffect on g with Test, got {errors:?}"
        );
    }

    #[test]
    fn declared_test_effect_accepts_test_assert_caller() {
        // When the caller honestly declares `! { Test }`, no error.
        let decls = parse_surf(
            r#"
        def test_ok() -> unit ! {Test} = test_assert(true, "ok")
        "#,
            &chelis_syntax_host::NativeStack,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        check_program(&checked).expect("declared Test should accept test_assert caller");
    }

    #[test]
    fn declared_io_effect_still_rejects_test_effect_leakage() {
        // A sig with `! { IO }` is not a superset of `{ Test }`. Adding a test_assert
        // call inside an IO-only function should still be rejected — the effect
        // systems are orthogonal axes of the effect row.
        let decls = parse_surf(
            r#"
        def leak() -> unit ! {IO} = test_assert(true, "sneak")
        "#,
            &chelis_syntax_host::NativeStack,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        let errors =
            check_program(&checked).expect_err("IO-declared fn must not silently acquire Test");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect && error.message.contains("Test")
            }),
            "expected UnhandledEffect mentioning Test, got {errors:?}"
        );
    }

    // ----- Module-wrapped whole-program effect soundness -----
    //
    // A `.dp` MODULE wraps its decls in `(module ...)`. The whole-program
    // effect validators must descend into that wrapper so a module-wrapped
    // `chelis check` agrees with the flattened build/eval path. Without the
    // descent, a declared-pure function whose body performs IO would be
    // accepted module-wrapped but rejected flattened.

    /// Type-check a module-WRAPPED Deep program (the shape `chelis check` runs
    /// on a `.dp` MODULE), preserving the `(module ...)` wrapper.
    fn typed_module(src: &str) -> CheckedProgram {
        let decls = parse_surf(src, &chelis_syntax_host::NativeStack).expect("surf parse");
        let deep = desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar");
        chelis_types::check_typed_program(&deep).expect("type check")
    }

    #[test]
    fn module_wrapped_declared_pure_body_does_io_is_rejected() {
        // A declared-pure function whose body calls a file-IO builtin must be
        // rejected module-wrapped, the same as flattened.
        let checked = typed_module(
            r#"module Frag.Io
export (entry)
def entry(path: string) -> string ! { } = read_file(path)
"#,
        );
        let errors = check_program(&checked)
            .expect_err("module-wrapped declared-pure body performing IO must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect
                    && error.message.contains("entry")
                    && error.message.contains("IO")
            }),
            "expected UnhandledEffect on entry mentioning IO, got {errors:?}"
        );
    }

    #[test]
    fn module_wrapped_pure_program_checks_clean() {
        // The negative-parity case: a pure module must still check clean after
        // the descent change (no effects -> no rejection).
        let checked = typed_module(
            r#"module Frag.Pure
export (entry)
def helper(x: f32) -> f32 = add(x, x)
def entry(x: f32) -> f32 = helper(mul(x, x))
"#,
        );
        check_program(&checked).expect("pure module-wrapped program must check clean");
    }

    #[test]
    fn module_wrapped_honest_io_signature_checks_clean() {
        // A module-wrapped function that honestly declares `! { IO }` must
        // check clean: the descent fix tightens the unsound-accept path only,
        // not honest code.
        let checked = typed_module(
            r#"module Frag.Honest
export (entry)
def entry(path: string) -> string ! { IO } = read_file(path)
"#,
        );
        check_program(&checked).expect("honest IO module-wrapped program must check clean");
    }
}

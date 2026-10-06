// Tests only: Rust std functions on the clippy disallowed list compute
// reference or input values here; the list holds production code to
// chelis-crmath (chelis#2957).
#![allow(clippy::disallowed_methods)]
use super::host_ops::*;
use super::transforms::*;
use super::*;
use chelis_deep::DeepTag;

/// chelis#399, chelis#2889: a reef-linked ADT value keeps its linker name as
/// its identity and stores its declared source spelling, which both eval
/// exits (the human renderer and the `--json` `ExecutionValue` ABI surface)
/// print. The spelling keeps an authored `__`, so a type's two constructors
/// `Foo__Bar` and `Bar` stay distinct; a constructor without a linker name
/// stores its own name.
#[test]
fn eval_renders_the_stored_constructor_source_spelling() {
    let deftype = chelis_deep::parser::parse_str(
        "(deftype {} Pkg__app__Demo__Shapes__Shape () \
         (variant {} Pkg__app__Demo__Shapes__Foo__Bar (field {} value (t-prim {} i64))) \
         (variant {} Pkg__app__Demo__Shapes__Bar (field {} value (t-prim {} i64))))",
    )
    .expect("parse deftype fixture");
    let names = collect_constructor_source_names(&deftype);
    assert_eq!(
        names
            .get("Pkg__app__Demo__Shapes__Foo__Bar")
            .map(String::as_str),
        Some("Foo__Bar")
    );
    assert_eq!(
        names.get("Pkg__app__Demo__Shapes__Bar").map(String::as_str),
        Some("Bar")
    );
    let linked = RuntimeValue::Adt {
        ctor: "Pkg__app__Demo__Shapes__Foo__Bar".to_string(),
        source_name: "Foo__Bar".to_string(),
        fields: vec![RuntimeValue::Unit].into(),
        field_names: None,
    };
    assert_eq!(render_value(&linked), "Foo__Bar(())");
    match runtime_value_to_schema(&linked).expect("schema") {
        crate::schema::ExecutionValue::Adt { ctor, .. } => assert_eq!(ctor, "Foo__Bar"),
        _ => panic!("expected ExecutionValue::Adt"),
    }
    let bare = RuntimeValue::Adt {
        ctor: "None".to_string(),
        source_name: "None".to_string(),
        fields: vec![].into(),
        field_names: None,
    };
    assert_eq!(render_value(&bare), "None");
}

fn checked_surf(source: &str) -> CheckedProgram {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    chelis_types::check_ir_program(&exprs).expect("ir check")
}

/// chelis#1125: authoring normalization preserves legal structural parameter
/// lists, so both runtime readers must accept an annotated `Expr::BareList`
/// without admitting malformed name or metadata layouts.
#[test]
fn annotated_bare_list_parameter_preserves_runtime_name_and_declared_type() {
    use chelis_deep::Span;
    use chelis_deep::annotations::{MetadataValue, TypeSyntax};

    let span = Span::new(0, 0);
    let declared_type = Expr::node(
        DeepTag::TPrim,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("f64".to_string()), span)],
        span,
    );
    let parameter = Expr::BareList(
        vec![
            Expr::Atom(Atom::Name("x".to_string()), span),
            Expr::Map(
                Metadata::from(MetadataValue::Type(
                    TypeSyntax::try_new(declared_type.clone()).expect("valid declared type"),
                )),
                span,
            ),
        ],
        span,
    );

    assert_eq!(runtime_param_name(&parameter), Some("x"));
    assert_eq!(param_decl_type_expr(&parameter), Some(&declared_type));

    let malformed_name = Expr::BareList(
        vec![
            Expr::Atom(Atom::Int(0), span),
            Expr::Map(Metadata::default(), span),
        ],
        span,
    );
    assert_eq!(runtime_param_name(&malformed_name), None);
    assert_eq!(param_decl_type_expr(&malformed_name), None);
}

#[test]
fn runtime_function_params_reject_nonparameter_carriers_without_filtering() {
    use chelis_deep::Span;
    use chelis_deep::ast::UnknownFormData;

    let span = Span::new(0, 0);
    let invalid_params = [
        Expr::UnknownForm(Box::new(UnknownFormData {
            head: "x".to_string(),
            meta: Metadata::default(),
            children: vec![],
            span,
        })),
        Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("x".to_string()), span)],
            span,
        ),
    ];

    for parameter in invalid_params {
        assert_eq!(
            runtime_param_name(&parameter),
            None,
            "only source parameter roles may produce runtime binders"
        );
        let function = Expr::node(
            DeepTag::Fn,
            Metadata::default(),
            vec![
                Expr::node(DeepTag::Params, Metadata::default(), vec![parameter], span),
                Expr::node(
                    DeepTag::Lit,
                    Metadata::default(),
                    vec![Expr::Atom(Atom::Int(1), span)],
                    span,
                ),
            ],
            span,
        );
        assert_eq!(
            issue_1125_eval_raw_expr(&function)
                .expect_err("an invalid parameter must reject the function"),
            "fn params malformed",
            "an invalid parameter must reject instead of disappearing"
        );
    }
}

fn issue_1125_eval_raw_expr(expr: &Expr) -> Result<RuntimeValue, String> {
    let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    let mut ctx = EvalContext {
        bindings: Frame::new(),
        result_producer: None,
        binding_types: UnordMap::new(),
        precision_bindings: UnordMap::new(),
        declaration_values: UnordMap::new(),
        named_axis_route_cache: UnordMap::new(),
        named_axis_route_visiting: UnordSet::new(),
        program: ProgramScope::new(UnordMap::new(), UnordMap::new()),
        declared_signatures: UnordMap::new(),
        adt_fields: UnordMap::new(),
        constructor_names: UnordMap::new(),
        adt_registry: chelis_types::adt::AdtRegistry::default(),
        tensor_bindings: &empty_tensors,
        session: None,
        active_declaration_names: Vec::new(),
        def_kernels: UnordMap::new(),
        transcript: Vec::new(),
        transcript_capture: None,
        resolving_top_levels: Vec::new(),
        cancel: None,
        system: system::EvalSystemBoundary::permissive(),
        failure_kind: RuntimeFailureKind::Ordinary,
        activation_extents: Default::default(),
    };
    ctx.eval_expr(expr)
}

fn issue_1125_eval_checked_root(
    checked: &CheckedProgram,
    exprs: &[Expr],
    root: &str,
) -> Result<RuntimeValue, String> {
    let empty_tensors = UnordMap::new();
    let mut definitions = UnordMap::new();
    register_top_level_defs(
        exprs,
        &BTreeMap::new(),
        None,
        &mut definitions,
        &mut Vec::new(),
        false,
    );
    let mut signatures = UnordMap::new();
    register_declared_signatures(exprs, &mut signatures);
    let mut ctx = EvalContext {
        bindings: Frame::new(),
        result_producer: None,
        binding_types: UnordMap::new(),
        precision_bindings: UnordMap::new(),
        declaration_values: UnordMap::new(),
        named_axis_route_cache: UnordMap::new(),
        named_axis_route_visiting: UnordSet::new(),
        program: ProgramScope::new(
            definitions,
            checked
                .type_env()
                .iter()
                .map(|(name, ty)| (name.clone(), ty.clone()))
                .collect(),
        ),
        declared_signatures: signatures,
        adt_fields: UnordMap::new(),
        constructor_names: UnordMap::new(),
        adt_registry: checked.adt_registry().clone(),
        tensor_bindings: &empty_tensors,
        session: Some(chelis_ir::host::HostLoweringSession::new(checked)),
        active_declaration_names: Vec::new(),
        def_kernels: UnordMap::new(),
        transcript: Vec::new(),
        transcript_capture: None,
        resolving_top_levels: Vec::new(),
        cancel: None,
        system: system::EvalSystemBoundary::permissive(),
        failure_kind: RuntimeFailureKind::Ordinary,
        activation_extents: Default::default(),
    };
    ctx.resolve_top_level(root)
}

fn issue_1125_gradient_values(value: &RuntimeValue) -> Vec<f64> {
    match value {
        RuntimeValue::Scalar(payload) => vec![payload.as_f64_lossy()],
        RuntimeValue::Tensor(tensor) => tensor.value.to_f64_lossy_vec(),
        _ => panic!("expected one scalar or tensor gradient"),
    }
}

#[test]
fn runtime_grad_wrt_reads_the_checked_program() {
    let checked = checked_surf(
        "def pair(x: f32, y: f32) -> f32 = mul(x, y)\n\
         out = grad(pair, wrt=y)(2.0f32, 3.0f32)\n",
    );

    let gradient = issue_1125_eval_checked_root(&checked, checked.exprs(), "out").expect("grad");

    assert_eq!(issue_1125_gradient_values(&gradient), vec![2.0]);
}

/// chelis#1125: an explicitly typed binding stamps `surf_binding_type` on the
/// bound `grad` node, and spec/03 section 1.1 admits that key only on a bind
/// value. Applying the captured transform splices the node into the callee
/// slot of a synthesized `app`, where the node gate refuses the key, so the
/// splice must leave the binding origin behind.
#[test]
fn runtime_grad_bound_with_an_explicit_type_applies() {
    let checked = checked_surf(
        "def sq(x: f32) -> f32 = mul(x, x)\n\
         out = {\n\
           g: (f32) -> f32 = grad(sq)\n\
           g(3.0f32)\n\
         }\n",
    );

    let gradient = issue_1125_eval_checked_root(&checked, checked.exprs(), "out").expect("grad");

    assert_eq!(issue_1125_gradient_values(&gradient), vec![6.0]);
}

/// The twin of `runtime_grad_bound_with_an_explicit_type_applies` for the
/// differentiated function: an explicitly typed local `fn` carries
/// `surf_binding_type` into the captured closure that the transform installs
/// as a program definition for lowering.
#[test]
fn runtime_grad_of_a_function_bound_with_an_explicit_type_applies() {
    let checked = checked_surf(
        "out = {\n\
           f: (f32) -> f32 = fn (x: f32) -> mul(x, x)\n\
           grad(f)(3.0f32)\n\
         }\n",
    );

    let gradient = issue_1125_eval_checked_root(&checked, checked.exprs(), "out").expect("grad");

    assert_eq!(issue_1125_gradient_values(&gradient), vec![6.0]);
}

#[test]
fn runtime_closure_preserves_the_original_checked_function_carrier() {
    let span = chelis_deep::Span::new(4, 12);
    let successor = Expr::node(
        DeepTag::Fn,
        Metadata::default(),
        vec![
            Expr::node(DeepTag::Params, Metadata::default(), vec![], span),
            Expr::node(
                DeepTag::Lit,
                Metadata::default(),
                vec![Expr::Atom(Atom::Int(1), span)],
                span,
            ),
        ],
        span,
    );
    let RuntimeValue::Closure {
        checked_function: successor_checked,
        ..
    } = issue_1125_eval_raw_expr(&successor).expect("successor closure")
    else {
        panic!("successor function must evaluate to a closure")
    };

    assert!(matches!(successor_checked.as_ref(), Expr::Node(..)));
    assert_eq!(
        chelis_deep::printer::print_canonical_flat(&[successor_checked.as_ref().clone()]),
        chelis_deep::printer::print_canonical_flat(&[successor])
    );
}

#[test]
fn runtime_eval_reads_a_decoded_literal_node() {
    use chelis_deep::Span;

    let span = Span::new(4, 12);
    let value = Expr::Atom(Atom::Int(7), span);
    let successor = Expr::node(DeepTag::Lit, Metadata::default(), vec![value], span);

    let successor = issue_1125_eval_raw_expr(&successor).expect("successor literal evaluates");
    assert_eq!(render_value(&successor), "7");
}

#[test]
fn runtime_match_patterns_read_decoded_nodes() {
    use chelis_deep::Span;

    let span = Span::new(4, 12);
    let node = |tag, children| Expr::node(tag, Metadata::default(), children, span);
    let successor = node(
        DeepTag::Match,
        vec![
            node(DeepTag::Lit, vec![Expr::Atom(Atom::Int(1), span)]),
            node(
                DeepTag::Arm,
                vec![
                    node(DeepTag::PatLit, vec![Expr::Atom(Atom::Int(1), span)]),
                    Expr::BareList(vec![], span),
                    node(DeepTag::Lit, vec![Expr::Atom(Atom::Int(42), span)]),
                ],
            ),
        ],
    );
    let successor = issue_1125_eval_raw_expr(&successor);
    assert_eq!(successor.as_ref().map(render_value), Ok("42".to_string()));
}

/// `(match {} (lit {} scrutinee) (arm {} (pat-var {} flag) guard (lit {} 1))
/// (arm {} (pat-wild {}) () fallback))`: the first arm's guard is `guard`,
/// and the second arm is guardless.
fn guarded_match(scrutinee: bool, guard: Expr, fallback: Expr) -> Expr {
    let span = chelis_deep::Span::new(4, 12);
    let node = |tag, children| Expr::node(tag, Metadata::default(), children, span);
    node(
        DeepTag::Match,
        vec![
            node(DeepTag::Lit, vec![Expr::Atom(Atom::Bool(scrutinee), span)]),
            node(
                DeepTag::Arm,
                vec![
                    node(
                        DeepTag::PatVar,
                        vec![Expr::Atom(Atom::Name("flag".to_string()), span)],
                    ),
                    guard,
                    node(DeepTag::Lit, vec![Expr::Atom(Atom::Int(1), span)]),
                ],
            ),
            node(
                DeepTag::Arm,
                vec![
                    node(DeepTag::PatWild, vec![]),
                    Expr::BareList(vec![], span),
                    fallback,
                ],
            ),
        ],
    )
}

fn raw_var(name: &str) -> Expr {
    let span = chelis_deep::Span::new(4, 12);
    Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name(name.to_string()), span)],
        span,
    )
}

fn raw_int(value: i64) -> Expr {
    let span = chelis_deep::Span::new(4, 12);
    Expr::node(
        DeepTag::Lit,
        Metadata::default(),
        vec![Expr::Atom(Atom::Int(value), span)],
        span,
    )
}

/// REGRESSION TEST ([04-PAT-2], chelis#2445). The guard reads the binding
/// its own pattern introduced, and a `false` guard passes control to the next
/// arm. Before the fix the guard slot was never read, so the first arm won for
/// both scrutinees.
#[test]
fn runtime_match_guard_reads_its_binding_and_false_tries_the_next_arm() {
    let selected = issue_1125_eval_raw_expr(&guarded_match(true, raw_var("flag"), raw_int(2)));
    assert_eq!(selected.as_ref().map(render_value), Ok("1".to_string()));
    let skipped = issue_1125_eval_raw_expr(&guarded_match(false, raw_var("flag"), raw_int(2)));
    assert_eq!(skipped.as_ref().map(render_value), Ok("2".to_string()));
}

/// REGRESSION TEST ([04-PAT-2]). A skipped arm's bindings are discarded: the
/// next arm cannot read `flag`. Before the fix the first arm was selected, so
/// the fallback never ran.
#[test]
fn runtime_match_guard_that_is_false_discards_its_bindings() {
    let result = issue_1125_eval_raw_expr(&guarded_match(false, raw_var("flag"), raw_var("flag")));
    let error = result.expect_err("the skipped arm's `flag` must not reach the next arm");
    assert!(error.contains("flag"), "{error}");
}

/// REGRESSION TEST, negative parity ([04-PAT-2]). A guard that cannot be
/// evaluated fails the match rather than reading as `false`, and a guard that
/// is not a `bool` is rejected rather than read as truthy. Before the fix both
/// programs returned the first arm's `1`.
#[test]
fn runtime_match_guard_that_fails_or_is_not_bool_fails_the_match() {
    let unbound = issue_1125_eval_raw_expr(&guarded_match(true, raw_var("missing"), raw_int(2)));
    let error = unbound.expect_err("a failing guard must fail the match");
    assert!(error.contains("missing"), "{error}");
    let not_bool = issue_1125_eval_raw_expr(&guarded_match(true, raw_int(7), raw_int(2)));
    let error = not_bool.expect_err("a non-bool guard must fail the match");
    assert!(error.contains("match arm guard must be bool"), "{error}");
}

#[test]
fn runtime_pattern_reader_matches_every_decoded_pattern() {
    use chelis_deep::Span;

    let span = Span::new(4, 12);
    let name = |value: &str| Expr::Atom(Atom::Name(value.to_string()), span);
    let bool_lit = |value| {
        Expr::node(
            DeepTag::PatLit,
            Metadata::default(),
            vec![Expr::Atom(Atom::Bool(value), span)],
            span,
        )
    };
    let cases = [
        (
            RuntimeValue::Bool(true),
            Expr::node(
                DeepTag::PatVar,
                Metadata::default(),
                vec![name("bound")],
                span,
            ),
        ),
        (
            RuntimeValue::Bool(true),
            Expr::node(DeepTag::PatWild, Metadata::default(), vec![], span),
        ),
        (RuntimeValue::Bool(true), bool_lit(true)),
        (
            RuntimeValue::Adt {
                ctor: "Some".to_string(),
                source_name: "Some".to_string(),
                fields: vec![RuntimeValue::Bool(true)].into(),
                field_names: None,
            },
            Expr::node(
                DeepTag::PatCtor,
                Metadata::default(),
                vec![
                    name("Some"),
                    Expr::node(
                        DeepTag::PatVar,
                        Metadata::default(),
                        vec![name("item")],
                        span,
                    ),
                ],
                span,
            ),
        ),
        (
            RuntimeValue::Adt {
                ctor: "Point".to_string(),
                source_name: "Point".to_string(),
                fields: vec![RuntimeValue::Bool(true)].into(),
                field_names: Some(vec!["x".to_string()]),
            },
            Expr::node(
                DeepTag::PatRecord,
                Metadata::default(),
                vec![
                    name("Point"),
                    Expr::node(
                        DeepTag::Kv,
                        Metadata::default(),
                        vec![name("x"), bool_lit(true)],
                        span,
                    ),
                ],
                span,
            ),
        ),
        (
            RuntimeValue::Tuple(
                vec![
                    RuntimeValue::Bool(true),
                    RuntimeValue::String("ok".to_string()),
                ]
                .into(),
            ),
            Expr::node(
                DeepTag::PatTuple,
                Metadata::default(),
                vec![
                    bool_lit(true),
                    Expr::node(DeepTag::PatWild, Metadata::default(), vec![], span),
                ],
                span,
            ),
        ),
    ];
    let adt_fields = UnordMap::new();

    for (value, successor) in cases {
        let mut successor_bindings = Frame::new();
        assert_eq!(
            pattern_matches(&value, &successor, &mut successor_bindings, &adt_fields),
            Ok(true)
        );
    }

    for invalid in [
        Expr::BareList(vec![name("not-a-pattern")], span),
        Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
            head: "future-pattern".to_string(),
            meta: Metadata::default(),
            children: vec![],
            span,
        })),
        Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![name("not-a-pattern")],
            span,
        ),
    ] {
        let mut bindings = Frame::new();
        assert_eq!(
            pattern_matches(
                &RuntimeValue::Bool(true),
                &invalid,
                &mut bindings,
                &adt_fields
            ),
            Ok(false)
        );
        assert!(bindings.is_empty());
    }
}

#[test]
fn runtime_eval_rejects_each_nonruntime_carrier_explicitly() {
    use chelis_deep::Span;
    use chelis_deep::ast::{MetaExpr, UnknownFormData};

    let span = Span::new(4, 12);
    let cases = [
        (
            Expr::BareList(vec![Expr::Atom(Atom::Name("item".to_string()), span)], span),
            "a structural bare list is not a runtime expression",
        ),
        (
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "future-form".to_string(),
                meta: Metadata::default(),
                children: vec![],
                span,
            })),
            "unknown form `future-form` is not a runtime expression",
        ),
        (
            Expr::Atom(Atom::Name("leaf".to_string()), span),
            "bare atom is not a runtime expression",
        ),
    ];
    for (expr, expected) in cases {
        assert_eq!(
            issue_1125_eval_raw_expr(&expr).expect_err("nonruntime carrier must reject"),
            expected
        );
    }

    assert!(matches!(
        issue_1125_eval_raw_expr(&Expr::Map(Metadata::default(), span)),
        Ok(RuntimeValue::Unit)
    ));
    assert!(matches!(
        issue_1125_eval_raw_expr(&Expr::MetaExpr(
            MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(Expr::Map(Metadata::default(), span)),
            },
            span,
        )),
        Ok(RuntimeValue::Unit)
    ));
}

#[test]
fn runtime_nested_owner_readers_reject_malformed_children() {
    use chelis_deep::Span;
    use chelis_deep::ast::UnknownFormData;

    let span = Span::new(12, 18);
    let name = |value: &str| Expr::Atom(Atom::Name(value.to_string()), span);
    let unknown = || {
        Expr::UnknownForm(Box::new(UnknownFormData {
            head: "future-field".to_string(),
            meta: Metadata::default(),
            children: vec![],
            span,
        }))
    };

    let mut adt_fields = UnordMap::new();
    adt_fields.insert("Point".to_string(), vec!["x".to_string()]);
    let mut bindings = Frame::new();
    let malformed_pattern = Expr::node(
        DeepTag::PatRecord,
        Metadata::default(),
        vec![name("Point"), unknown()],
        span,
    );
    assert_eq!(
        pattern_matches(
            &RuntimeValue::Adt {
                ctor: "Point".to_string(),
                source_name: "Point".to_string(),
                fields: vec![RuntimeValue::Bool(true)].into(),
                field_names: Some(vec!["x".to_string()]),
            },
            &malformed_pattern,
            &mut bindings,
            &adt_fields,
        )
        .expect_err("a malformed pat-record child must not disappear"),
        "pat-record field must be a decoded `kv` node"
    );
    assert!(bindings.is_empty());

    let malformed_record = Expr::node(
        DeepTag::Record,
        Metadata::default(),
        vec![name("Point"), unknown()],
        span,
    );
    assert_eq!(
        issue_1125_eval_raw_expr(&malformed_record)
            .expect_err("a malformed record child must not disappear"),
        "record field must be a decoded `kv` node"
    );

    let malformed_match = Expr::node(
        DeepTag::Match,
        Metadata::default(),
        vec![
            Expr::node(
                DeepTag::Lit,
                Metadata::default(),
                vec![Expr::Atom(Atom::Bool(true), span)],
                span,
            ),
            unknown(),
        ],
        span,
    );
    assert_eq!(
        issue_1125_eval_raw_expr(&malformed_match)
            .expect_err("a malformed match arm must not disappear"),
        "match arm must be a decoded `arm` node"
    );
}

/// chelis#1829: the interpreter entry derives each definition's kernel
/// decision once, so the summary probes behind `def_kernel` are bounded by the
/// number of definitions rather than expanding the call graph as a tree.
///
/// Two rows, and each stops the other going vacuous. The tensor row measures
/// the BOUND, because it still reaches the probe. The scalar row is the
/// REGRESSION CLASS, because #1829's own provoking definitions are non-tensor,
/// and after chelis#1835's predicate reorder its count is exactly zero.
///
/// Evidentiary status: REGRESSION TEST for the entry point. On the base this
/// function armed nothing, so the scalar fixture builds hundreds of thousands
/// of summaries (measured: 398,574 at depth 12) and takes minutes.
///
/// The fixture uses one program rather than a composed library plus caller,
/// because `CheckedProgram::compose` requires a `library_proof_id` that only
/// the real context pipeline mints and a lib unit test cannot. The exposure
/// #1693 actually opened, an imported library definition reaching this probe,
/// is covered end to end by the `csv_io`, `json_io` and `issue_1314_json_bigint`
/// CLI rows. What this pins is the part those rows cannot: an exact probe count
/// at the entry that arms the scope.
///
/// The negative twin is on the base rather than a mutation: the scalar row's
/// 398,574 against its 0 here, and the tensor row's own `probes > 0` guard,
/// which fired on CI when chelis#1835's reorder first made the scalar fixture
/// stop reaching the probe.
/// One evaluation of `source`, returning its `result` rendering and the number
/// of kernel-decision summary probes the interpreter ran.
///
/// A large stack so that an unmemoized run reports the probe COUNT rather than
/// overflowing: the probe recurses once per call-graph level per call site, and
/// the default test stack aborts the whole process at depth 12.
fn issue_1829_entry_probe_count(label: &'static str, source: String) -> (String, u64) {
    std::thread::Builder::new()
        .name(format!("issue-1829-entry-{label}"))
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            let checked = checked_surf(&source);
            let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
            let inputs = HostEvaluationInputs {
                roots: &empty_tensors,
                bindings: None,
            };
            chelis_ir::host::reset_host_summary_probe_builds();
            let outcome = evaluate_host_program_with_library_and_types(
                &checked, None, None, inputs, None, None,
            )
            .expect("#1829 fanout fixture evaluates");
            let probes = chelis_ir::host::host_summary_probe_builds();
            let result = outcome
                .host_bindings
                .get("result")
                .map(render_value)
                .expect("#1829 fixture binds `result`");
            (result, probes)
        })
        .expect("#1829 entry probe thread starts")
        .join()
        .expect("#1829 entry probe thread completes")
}

/// A fan-out chain of `depth` definitions, each calling the next from two
/// argument positions, returning tensors or scalars.
fn issue_1829_entry_source(depth: usize, tensor_result: bool) -> String {
    let mut source = String::new();
    if tensor_result {
        for level in 0..depth {
            source.push_str(&format!(
                "def f{level}(x: tensor[4, f32]) -> tensor[4, f32] = \
                 add(f{next}(x), f{next}(x))\n",
                next = level + 1
            ));
        }
        source.push_str(&format!(
            "def f{depth}(x: tensor[4, f32]) -> tensor[4, f32] = mul(x, x)\n"
        ));
        source.push_str("seed = reshape(insert(to_tensor([cast(1.0, f32)]), 0, 4i64), [4i64])\n");
    } else {
        for level in 0..depth {
            source.push_str(&format!(
                "def f{level}(x: i64) -> i64 = add(f{next}(x), f{next}(x))\n",
                next = level + 1
            ));
        }
        source.push_str(&format!("def f{depth}(x: i64) -> i64 = x\n"));
        source.push_str("seed = cast(1, i64)\n");
    }
    source.push_str("result = f0(seed)\n");
    source
}

#[test]
fn issue_1829_interpreter_entry_bounds_kernel_decision_probes() {
    const DEPTH: usize = 12;
    let definitions = u64::try_from(DEPTH + 1).expect("definition count fits");

    // Row 1, the BOUND. A tensor-returning chain reaches the probe, so the
    // bound is measured rather than asserted over nothing. This row exists in
    // this shape because chelis#1835's predicate reorder made the probe the
    // last host-lane predicate asked: the scalar chain below no longer reaches
    // it, and a receipt whose only fixture stopped reaching it would pass
    // while measuring zero work. Its own `probes > 0` guard caught exactly
    // that, on CI.
    let (tensor_result, tensor_probes) =
        issue_1829_entry_probe_count("tensor", issue_1829_entry_source(DEPTH, true));
    eprintln!("#1829 entry tensor definitions={definitions} probes={tensor_probes}");
    assert_eq!(
        tensor_result, "tensor(shape=[4], data=[4096.0, 4096.0, 4096.0, 4096.0])",
        "#1829 tensor fixture must still compute the right answer"
    );
    assert!(
        tensor_probes > 0,
        "#1829: the fixture must actually reach the kernel-decision probe, or this receipt \
         would pass without measuring anything"
    );
    assert!(
        tensor_probes <= definitions,
        "#1829: the interpreter entry must build at most one summary per definition; \
         {definitions} definitions produced {tensor_probes} builds"
    );

    // Row 2, the REGRESSION CLASS. #1829's own class is the scalar chain,
    // because the definitions that provoked it are non-tensor (`Std.Io.Json`).
    // Keeping it is what stops row 1's retyping from quietly dropping the
    // class this receipt was written for. Its count is now exactly zero, and
    // that is the public-entry receipt for the reorder: a definition whose
    // declared result cannot be a kernel pays no probe at all. On the base it
    // was 398,574.
    let (scalar_result, scalar_probes) =
        issue_1829_entry_probe_count("scalar", issue_1829_entry_source(DEPTH, false));
    eprintln!("#1829 entry scalar definitions={definitions} probes={scalar_probes}");
    assert_eq!(
        scalar_result, "4096",
        "#1829 scalar fixture must still compute the right answer"
    );
    assert_eq!(
        scalar_probes, 0,
        "#1829/#1835: a non-tensor definition must pay no kernel-decision probe; the reorder \
         asks the probe after the declared result type, so this chain reaches none"
    );
}

/// Evaluate `source` and return its `result` binding with the number of
/// definition kernel plannings performed.
fn def_kernel_plannings(source: &str) -> (String, u64) {
    let checked = checked_surf(source);
    let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    let inputs = HostEvaluationInputs {
        roots: &empty_tensors,
        bindings: None,
    };
    super::eval::take_def_kernel_plannings();
    let outcome =
        evaluate_host_program_with_library_and_types(&checked, None, None, inputs, None, None)
            .expect("#2392 fixture evaluates");
    let plannings = super::eval::take_def_kernel_plannings();
    let result = outcome
        .host_bindings
        .get("result")
        .map(render_value)
        .expect("#2392 fixture binds `result`");
    (result, plannings)
}

/// chelis#2392: a recursive program applies its helpers many times.
/// `def_kernel` used to skip its memo beneath a recursive caller and re-plan
/// every applied helper per application. A helper that draws no Random is
/// planned once however many times the recursion applies it. chelis#2413
/// gives a drawing helper the same memo: its draw keys read the frame it is
/// evaluated with, so its kernel no longer depends on the stream position.
/// chelis#2405 retired the inherited execution exclusion this counted
/// beneath, so the receipt now counts every planning.
///
/// Evidentiary status: REGRESSION TEST for the non-drawing row (it fails on
/// the #2392 base, where the count grows with the depth) and for the drawing
/// row (it fails on the #2413 base, which re-planned per application).
#[test]
fn issue_2392_kernel_under_recursion_is_planned_once_per_helper() {
    let program = |depth: i64| {
        format!(
            "def double(x: tensor[4, f32]) -> tensor[4, f32] = add(x, x)\n\
             def walk(n: i64, x: tensor[4, f32]) -> tensor[4, f32] = if eq(n, 0i64) then x else walk(sub(n, 1i64), double(x))\n\
             result = tensor_to_scalar(sum(walk({depth}i64, to_tensor([1.0f32, 0.0f32, 0.0f32, 0.0f32])), 0i32))\n"
        )
    };
    let (shallow, shallow_plannings) = def_kernel_plannings(&program(4));
    let (deep, deep_plannings) = def_kernel_plannings(&program(12));
    assert_eq!(shallow, "16.0");
    assert_eq!(deep, "4096.0");
    assert!(
        shallow_plannings >= 1,
        "the fixture must reach the kernel decision"
    );
    assert_eq!(
        shallow_plannings, deep_plannings,
        "plannings must not grow with the number of applications"
    );
}

/// chelis#2393: short-name resolution through the terminal index must agree
/// with the linear scan it replaced, `terminal_name_matches` over every key
/// with the exactly-one-match rule, for exact, short, qualified, ambiguous and
/// absent spellings, and the index is built once per scope however many
/// names are resolved.
///
/// Evidentiary status: DISPOSITION LOCK for the resolution table (the rule is
/// unchanged) and REGRESSION TEST for the build count (the base had no index
/// and scanned on every ask).
#[test]
fn issue_2393_terminal_index_resolves_like_the_scan() {
    let keys = [
        "A__x",
        "B.x",
        "A.b__y",
        "y",
        "C__y__z",
        "Pkg__lib__Mod__w",
        "Pkg.lib.Mod.v",
        "u",
        "Other__u",
        "plain",
    ];
    let span = chelis_deep::Span::new(0, 0);
    let defs = keys
        .iter()
        .map(|key| {
            (
                (*key).to_owned(),
                Expr::Atom(chelis_deep::ast::Atom::Int(0), span),
            )
        })
        .collect::<UnordMap<_, _>>();
    let scope = ProgramScope::new(defs.clone(), UnordMap::new());
    let scan = |name: &str| {
        let sorted = defs.to_sorted();
        let mut matches = sorted
            .into_iter()
            .filter(|(key, _)| terminal_name_matches(key, name))
            .map(|(key, _)| key.clone());
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    };
    let mut queries = keys.iter().map(|key| (*key).to_owned()).collect::<Vec<_>>();
    for short in ["x", "y", "z", "w", "v", "u", "b__y", "plain", "missing", ""] {
        queries.push(short.to_owned());
        queries.push(format!("Q__{short}"));
        queries.push(format!("Q.{short}"));
    }
    super::program_scope::take_terminal_index_builds();
    for query in &queries {
        assert_eq!(
            scope.resolve_def_key(query).map(str::to_owned),
            if defs.contains_key(query.as_str()) {
                Some(query.clone())
            } else {
                scan(query)
            },
            "resolution of `{query}`"
        );
    }
    // Spot-check the table itself so a shared bug in both readings shows.
    assert_eq!(scope.resolve_def_key("x"), None, "`x` is ambiguous");
    assert_eq!(scope.resolve_def_key("z"), Some("C__y__z"));
    assert_eq!(scope.resolve_def_key("Q.w"), Some("Pkg__lib__Mod__w"));
    assert_eq!(scope.resolve_def_key("v"), Some("Pkg.lib.Mod.v"));
    assert_eq!(scope.resolve_def_key("u"), Some("u"), "an exact key wins");
    assert_eq!(scope.resolve_def_key("missing"), None);
    assert_eq!(
        super::program_scope::take_terminal_index_builds(),
        1,
        "the index is built once per scope, not once per ask"
    );
}

/// chelis#2204: an anonymous `fn` captures the whole enclosing binding frame
/// and the list combinators clone the callback once per element, so before
/// this fix every element deep-copied every binding in scope, including
/// bindings the callback never mentions. A `fold` was quadratic in whatever
/// happened to be in scope.
///
/// Counted receipt in the shape of chelis#1835's
/// `host_summary_probe_builds`: `frame_value_copies` counts binding
/// entries deep-copied by frame clones. The fixture binds one unused list and
/// folds one closure over `applications` elements; the asymptotic promise is
/// that the copies do not grow with the application count, asserted as a
/// comparison between two application counts rather than a machine budget.
///
/// Evidentiary status: REGRESSION TEST, proven failing first. With the
/// counter and this test in place but the frame representation unchanged
/// (`clone_frame`/`clone_callable` over the by-value `UnordMap` frame), the
/// receipt read 201 copies at 100 applications and 801 at 400: two frame
/// copies per element plus the capture. After the fix both read 0.
///
/// The nested-let companion fixture keeps the receipt honest: a block that
/// shadows an enclosing local must still copy that local when it saves the
/// frame, so a counter that stopped measuring would fail there rather than
/// pass vacuously here.
#[test]
fn issue_2204_frame_copies_do_not_scale_with_closure_applications() {
    fn evaluate(source: &str, binding: &str) -> (u64, String) {
        let checked = checked_surf(source);
        let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
        let inputs = HostEvaluationInputs {
            roots: &empty_tensors,
            bindings: None,
        };
        super::frame::reset_frame_value_copies();
        let outcome =
            evaluate_host_program_with_library_and_types(&checked, None, None, inputs, None, None)
                .expect("#2204 fixture evaluates");
        let copies = super::frame::frame_value_copies();
        let value = outcome
            .host_bindings
            .get(binding)
            .map(render_value)
            .unwrap_or_else(|| panic!("#2204 fixture binds `{binding}`"));
        (copies, value)
    }
    fn fold_fixture(applications: usize) -> String {
        // `unused` is in scope and never read by the closure; before the fix
        // it was copied on every application anyway.
        format!(
            "result = {{\n  unused = range(cast(0, i64), cast(64, i64))\n  \
             fold(fn (acc: i64, x: i64) -> add(acc, x), cast(0, i64), \
             range(cast(0, i64), cast({applications}, i64)))\n}}\n"
        )
    }

    let (small, small_result) = evaluate(&fold_fixture(100), "result");
    let (large, large_result) = evaluate(&fold_fixture(400), "result");
    eprintln!(
        "#2204 receipt: 100 applications copied {small} entries, 400 applications copied {large}"
    );
    assert_eq!(
        small_result, "4950",
        "#2204: 100-element fold must still sum correctly"
    );
    assert_eq!(
        large_result, "79800",
        "#2204: 400-element fold must still sum correctly"
    );
    assert!(
        large <= small,
        "#2204: frame copies must not grow with closure applications; 100 applications \
         copied {small} binding entries, 400 applications copied {large}"
    );

    // Companion: the counter is live. Entering a nested block saves the
    // enclosing frame, and that frame holds one local, so exactly that copy
    // is observed.
    let (nested, nested_result) = evaluate(
        "result = {\n  a = cast(1, i64)\n  b = {\n    c = cast(2, i64)\n    add(a, c)\n  }\n  b\n}\n",
        "result",
    );
    assert_eq!(
        nested_result, "3",
        "#2204: nested-let companion must still compute"
    );
    assert!(
        nested >= 1,
        "#2204: the frame-copy counter must observe the nested block's frame save, or this \
         receipt would pass without measuring anything"
    );
}

/// chelis#2335: a read-only list builtin reads its list argument in place.
///
/// `expect_list_arg` copied the whole list for every list builtin, so a
/// `fold` that reads one element of an `n`-element list with `index` on each
/// of `n` steps copied `n * (n + 1)` elements: `n` per `index` and `n` for
/// the fold's own list. Counted receipt in the shape of chelis#2204's:
/// `element_copies` counts container elements copied out of a shared
/// sequence, and the bound is the fold's own `n`, at two list lengths.
///
/// Evidentiary status: REGRESSION TEST, proven failing first. With the
/// counter and this test in place and `expect_list_arg` still copying, the
/// receipt read 10100 copies at 100 elements and 160400 at 400. After the
/// fix it reads 100 and 400.
///
/// The companion keeps the receipt honest: `append` to a list another binding
/// still holds must copy it, so a counter that stopped measuring fails there
/// rather than passing vacuously here.
#[test]
fn issue_2335_read_only_list_builtins_do_not_copy_the_list() {
    fn evaluate(source: &str) -> (u64, String) {
        let checked = checked_surf(source);
        let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
        let inputs = HostEvaluationInputs {
            roots: &empty_tensors,
            bindings: None,
        };
        super::shared_values::reset_element_copies();
        let outcome =
            evaluate_host_program_with_library_and_types(&checked, None, None, inputs, None, None)
                .expect("#2335 fixture evaluates");
        let copies = super::shared_values::element_copies();
        let value = outcome
            .host_bindings
            .get("result")
            .map(render_value)
            .expect("#2335 fixture binds `result`");
        (copies, value)
    }
    fn index_fixture(length: usize) -> String {
        format!(
            "result = {{\n  xs = range(0i64, {length}i64)\n  \
             fold(fn (acc: i64, i: i64) -> add(acc, index(xs, i)), 0i64, range(0i64, {length}i64))\n}}\n"
        )
    }
    fn append_fixture(length: usize) -> String {
        format!(
            "result = {{\n  xs = range(0i64, {length}i64)\n  ys = append(xs, 7i64)\n  \
             add(len(xs), len(ys))\n}}\n"
        )
    }

    let (small, small_result) = evaluate(&index_fixture(100));
    let (large, large_result) = evaluate(&index_fixture(400));
    eprintln!("#2335 receipt: index over 100 elements copied {small}, over 400 copied {large}");
    assert_eq!(
        small_result, "4950",
        "#2335: the 100-element sum is unchanged"
    );
    assert_eq!(
        large_result, "79800",
        "#2335: the 400-element sum is unchanged"
    );
    // The fold hands each element of its own list to the callback, one copy
    // per element; every `index` beyond that must copy nothing.
    assert!(
        small <= 100 && large <= 400,
        "#2335: `index` must not copy the list it reads; 100 elements copied {small}, \
         400 copied {large}"
    );

    let (shared_small, small_len) = evaluate(&append_fixture(100));
    let (shared_large, large_len) = evaluate(&append_fixture(400));
    assert_eq!(small_len, "201", "#2335: append companion computes");
    assert_eq!(large_len, "801", "#2335: append companion computes");
    assert!(
        shared_small >= 100 && shared_large >= 400,
        "#2335: appending to a list another binding holds must copy it, or this receipt \
         would pass without measuring anything; copied {shared_small} and {shared_large}"
    );
}

/// chelis#2204: a closure parameter shadows a captured binding of the same
/// name at the interpreter level, not only inside `Frame`'s own unit tests.
/// Red-team round 1 on chelis#2208 inverted `Frame::get` to prefer the
/// outermost scope and the whole crate stayed green except `frame.rs`'s
/// tests; this fixture is the interpreter-level lock. Under that inversion
/// `g(3)` returns the captured `n = 7`, giving 77 instead of 37.
#[test]
fn issue_2204_closure_parameter_shadows_captured_binding() {
    let checked = checked_surf(
        "result = {\n  n = cast(7, i64)\n  g = fn (n: i64) -> n\n  add(mul(g(cast(3, i64)), cast(10, i64)), n)\n}\n",
    );
    let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    let inputs = HostEvaluationInputs {
        roots: &empty_tensors,
        bindings: None,
    };
    let outcome =
        evaluate_host_program_with_library_and_types(&checked, None, None, inputs, None, None)
            .expect("#2204 shadowing fixture evaluates");
    let result = outcome
        .host_bindings
        .get("result")
        .map(render_value)
        .expect("#2204 shadowing fixture binds `result`");
    assert_eq!(
        result, "37",
        "#2204: the closure parameter `n` must shadow the captured `n`; 77 means the captured \
         scope won the lookup"
    );
}

fn manifest_entry_with_path(
    name: &str,
    def_name: &str,
    path: Vec<chelis_types::manifest::RootPathStep>,
) -> chelis_types::manifest::RootEntry {
    chelis_types::manifest::RootEntry {
        name: name.to_string(),
        path,
        def_name: def_name.to_string(),
        ty: Expr::Atom(
            chelis_deep::ast::Atom::Name("unknown".to_string()),
            chelis_deep::Span::new(0, 0),
        ),
        lane: chelis_types::types::Lane::Host,
        required_inputs: std::collections::BTreeSet::new(),
        reasons: Vec::new(),
    }
}

#[test]
fn manifest_root_lookup_follows_recursive_list_adt_path() {
    use chelis_types::manifest::RootPathStep::Adt;

    let entry = manifest_entry_with_path("items.1.0", "items", vec![Adt(1), Adt(0)]);
    let bindings = UnordMap::from([(
        "items".to_string(),
        RuntimeValue::List(vec![RuntimeValue::int_lit(1), RuntimeValue::int_lit(2)].into()),
    )]);

    let value = lookup_runtime_value_for_manifest_root(&entry, &bindings, &UnordMap::new())
        .expect("Cons tail/head path must select the second list item");
    assert_eq!(render_value(&value), "2");
}

#[test]
fn manifest_root_lookup_rejects_a_path_step_for_the_wrong_runtime_shape() {
    use chelis_types::manifest::RootPathStep::Tuple;

    let entry = manifest_entry_with_path("value.0", "value", vec![Tuple(0)]);
    let bindings = UnordMap::from([("value".to_string(), RuntimeValue::Bool(true))]);

    assert!(
        lookup_runtime_value_for_manifest_root(&entry, &bindings, &UnordMap::new()).is_none(),
        "an unavailable owed component must not fall back to the base value"
    );
}

#[test]
fn keyed_uniform_like_evaluates() {
    let checked = checked_surf(
        r#"
x = tensor_to_scalar(
  uniform_like(
    key_from_seed(7i64),
    trace(pad_sequences_to([[0.0]], cast(1, i64), cast(0.0, f32)), cast(0, i32), cast(1, i32)),
    0.0,
    1.0
  )
)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("keyed host program should evaluate");
    let value = outcome.host_bindings.get("x").expect("x binding");
    match value.as_f64() {
        Some(v) => assert!((0.0..=1.0).contains(&v), "got {v}"),
        None => panic!("expected float result, got {value:?}"),
    }
}

#[test]
fn ownership_drop_is_unit_and_does_not_shadow_list_drop_runtime() {
    let checked = checked_surf(
        r#"
x = {
  t = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  values = to_list(t)
  actual = index(values, cast(0, i64))
  _ = drop(t)
  actual
}
y = index(skip([cast(10, i64), cast(20, i64)], cast(1, i64)), cast(0, i64))
"#,
    );

    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("ownership drop and list drop should both evaluate");
    assert!(matches!(
        outcome.host_bindings.get("x").and_then(RuntimeValue::as_f64),
        Some(v) if (v - 1.0).abs() < f64::EPSILON
    ));
    assert_eq!(
        outcome
            .host_bindings
            .get("y")
            .and_then(RuntimeValue::as_i64),
        Some(20),
    );
}

// ----- chelis#1558: [04-DTYPE-1] owns the unbounded cast target -----
//
// Every row below used to assert that the host interpreter accepted
// `cast(<variable>, p)` under an unbounded binder and actualized `p` at each
// call site's own concrete dtype. [04-DTYPE-1] says a primitive type position
// SHALL name an active primitive and that the TYPE CHECKER rejects anything
// else as a cast target, and [04-DTYPE-2] says an unbounded binder is not a
// primitive. So those programs never reach the interpreter, and the rows
// encoded an implementation convenience rather than a decided rule.
//
// Each row therefore became two: a negative control asserting the check-time
// rejection, and a bounded-binder twin that keeps the actualization coverage,
// because `[p_int: Int]` is a legitimate primitive position under [04-DTYPE-2]
// and still specializes per call site. The twins are the reason the inversion
// loses no behavioural coverage.

/// Assert the [04-DTYPE-1] check-time rejection, naming the binder and its
/// owning declaration. The interpreter is never invoked: a rejected program has
/// no checked form to evaluate.
fn expect_unbounded_cast_target_rejection(source: &str, binder: &str, owner: &str) {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let errors = chelis_types::check_ir_program(&exprs)
        .expect_err("an unbounded binder is not a primitive type position");
    let subject = format!("cast target `{binder}` in `{owner}` does not name an active primitive");
    assert!(
        errors
            .errors
            .iter()
            .any(|error| error.message.contains(&subject) && error.message.contains("04-DTYPE-1")),
        "expected the [04-DTYPE-1] rejection naming `{binder}` in `{owner}`; got [{}]",
        errors
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    );
}

/// chelis#1558 negative control. **Regression test**: red before the checker
/// enforced [04-DTYPE-1] for a variable source, when this program checked at
/// 1.0 and the interpreter actualized `p_int` per call site.
///
/// The binder is declared by a `sig` with no bound, which is the form that most
/// looks like a dtype parameter and is not one.
#[test]
fn unbounded_sig_declared_cast_target_is_rejected_at_check_time() {
    expect_unbounded_cast_target_rejection(
        r#"
sig recast_int[p_int]: p_int -> List[p_int] -> p_int
def recast_int(value, witness) = cast(value, p_int)
i16_value = recast_int(cast(257, i16), [cast(0, i16)])
"#,
        "p_int",
        "recast_int",
    );
}

/// chelis#1558 positive twin of the row above. **Disposition lock** on the
/// behaviour the old row protected: a cast target that IS a legitimate
/// primitive position actualizes at each call site's own concrete dtype. Green
/// before and after; only its binder gained the bound [04-DTYPE-2] requires.
#[test]
fn bounded_cast_target_uses_each_calls_concrete_precision() {
    let checked = checked_surf(
        r#"
sig recast_int[p_int: Int]: p_int -> List[p_int] -> p_int
def recast_int(value, witness) = cast(value, p_int)
sig recast_float[p_float: Float]: p_float -> List[p_float] -> p_float
def recast_float(value, witness) = cast(value, p_float)
i16_value = recast_int(cast(257, i16), [cast(0, i16)])
i64_value = recast_int(cast(4294967297, i64), [cast(0, i64)])
f32_value = recast_float(cast(1.5, f32), [cast(0.0, f32)])
f64_value = recast_float(cast(1.5, f64), [cast(0.0, f64)])
"#,
    );

    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("bounded cast targets should actualize at each call");
    for (name, expected) in [
        ("i16_value", Prim::Int16),
        ("i64_value", Prim::Int64),
        ("f32_value", Prim::F32),
        ("f64_value", Prim::F64),
    ] {
        let RuntimeValue::Scalar(payload) = outcome.host_bindings.get(name).expect(name) else {
            panic!("{name} should be a scalar")
        };
        assert_eq!(payload.dtype(), expected, "{name} dtype");
    }
}

#[test]
fn tensor_cast_without_actualized_target_rejects_instead_of_reusing_source_dtype() {
    let error = eval_deep_with_bindings(
        "cast(value, p)",
        &[("value", tensor_value(Prim::F32, vec![2], vec![1.0, 2.0]))],
    )
    .expect_err("a missing dtype binding must not become an identity cast");
    assert!(error.contains("cast target `p`"), "{error}");
}

/// chelis#1558: this row's subject is the PRECISION MISMATCH, not the binder,
/// so it keeps its subject on a bounded binder where the program is still
/// admitted far enough to reach it. **Disposition lock**, green before and
/// after. On an unbounded binder [04-DTYPE-1] now fires first and the mismatch
/// is never reported, which is what the negative control below records.
#[test]
fn bounded_cast_target_does_not_accept_conflicting_precisions() {
    let source = r#"
def choose_and_cast[p_int: Int](left: p_int, right: p_int) -> p_int = cast(left, p_int)
value = choose_and_cast(cast(1, i16), cast(2, i64))
"#;
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let errors = chelis_types::check_ir_program(&exprs)
        .expect_err("one bounded precision cannot actualize to two concrete dtypes");
    assert!(
        errors.errors.iter().any(|error| {
            error.kind.diagnostic_name() == "PrecisionMismatch"
                && error.expected.is_none()
                && error.got.is_none()
                && error.span_offset == source.rfind("choose_and_cast(")
                && error.message.contains("choose_and_cast")
                && error.message.contains("i16")
                && error.message.contains("i64")
        }),
        "the shared binder must reject both concrete precisions at the call without \
         fabricating a declared dtype direction"
    );
}

/// chelis#1558 negative control for the same program with the bound removed.
/// **Regression test**: red before this change, when it checked far enough to
/// report only the precision mismatch. It now fails earlier, on the target.
#[test]
fn unbounded_cast_target_is_rejected_before_any_precision_mismatch() {
    expect_unbounded_cast_target_rejection(
        r#"
def choose_and_cast[p_int](left: p_int, right: p_int) -> p_int = cast(left, p_int)
value = choose_and_cast(cast(1, i16), cast(2, i64))
"#,
        "p_int",
        "choose_and_cast",
    );
}

/// chelis#1558 negative control. **Regression test**: a callee binder nested
/// inside a caller binder is still an unbounded binder in a primitive position.
#[test]
fn unbounded_cast_target_in_nested_calls_is_rejected_at_check_time() {
    expect_unbounded_cast_target_rejection(
        r#"
def inner[p_int](value: p_int, witness: List[p_int]) -> p_int = cast(value, p_int)
def outer[p_int](witness: List[p_int]) -> i64 =
  inner(cast(4294967297, i64), [cast(0, i64)])
value = outer([cast(7, i16)])
"#,
        "p_int",
        "inner",
    );
}

/// chelis#1558 positive twin. **Disposition lock**: a callee's bounded binder
/// specializes independently of its caller's, which is what the old row proved
/// and a bound does not change.
#[test]
fn bounded_cast_target_uses_fresh_specialization_for_nested_calls() {
    let checked = checked_surf(
        r#"
def inner[p_int: Int](value: p_int, witness: List[p_int]) -> p_int = cast(value, p_int)
def outer[p_int: Int](witness: List[p_int]) -> i64 =
  inner(cast(4294967297, i64), [cast(0, i64)])
value = outer([cast(7, i16)])
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("callee binders must specialize independently of caller binders");
    let RuntimeValue::Scalar(payload) = outcome.host_bindings.get("value").expect("value") else {
        panic!("value should be a scalar")
    };
    assert_eq!(payload.dtype(), Prim::Int64);
}

/// chelis#1558 negative control. **Regression test**: an empty container gives
/// the binder no witness, and the rejection does not depend on one.
#[test]
fn unbounded_cast_target_with_an_empty_container_is_rejected_at_check_time() {
    expect_unbounded_cast_target_rejection(
        r#"
def empty_witness[p_int](items: List[p_int], value: i64) -> p_int =
  cast(value, p_int)
def make_i16() -> i16 = empty_witness([], cast(257, i64))
value = make_i16()
"#,
        "p_int",
        "empty_witness",
    );
}

/// chelis#1558 positive twin. **Disposition lock**: the checked call result
/// still supplies the specialization when the container is empty.
#[test]
fn bounded_cast_target_uses_contextual_specialization_for_empty_container() {
    let checked = checked_surf(
        r#"
def empty_witness[p_int: Int](items: List[p_int], value: i64) -> p_int =
  cast(value, p_int)
def make_i16() -> i16 = empty_witness([], cast(257, i64))
value = make_i16()
"#,
    );

    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("the checked call result must supply the empty container specialization");
    let RuntimeValue::Scalar(payload) = outcome.host_bindings.get("value").expect("value") else {
        panic!("value should be a scalar")
    };
    assert_eq!(payload.dtype(), Prim::Int16);
}

/// chelis#1558 negative control. **Regression test**: an ADT argument carrying
/// the concrete dtype does not make an unbounded binder a primitive position.
#[test]
fn unbounded_cast_target_behind_an_adt_argument_is_rejected_at_check_time() {
    expect_unbounded_cast_target_rejection(
        r#"
def option_witness[p_int](item: Option[p_int], value: i64) -> p_int =
  cast(value, p_int)
value = option_witness(Some(cast(0, i16)), cast(257, i64))
"#,
        "p_int",
        "option_witness",
    );
}

/// chelis#1558 positive twin. **Disposition lock**: a checked `Option` argument
/// retains its concrete specialization through a bounded binder.
#[test]
fn bounded_cast_target_uses_checked_adt_specialization() {
    let checked = checked_surf(
        r#"
def option_witness[p_int: Int](item: Option[p_int], value: i64) -> p_int =
  cast(value, p_int)
value = option_witness(Some(cast(0, i16)), cast(257, i64))
"#,
    );

    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("the checked Option argument must retain its concrete specialization");
    let RuntimeValue::Scalar(payload) = outcome.host_bindings.get("value").expect("value") else {
        panic!("value should be a scalar")
    };
    assert_eq!(payload.dtype(), Prim::Int16);
}

/// chelis#1558 negative control. **Regression test**: passing the generic def
/// as a `map` callback does not exempt its cast target.
#[test]
fn unbounded_cast_target_in_a_map_callback_is_rejected_at_check_time() {
    expect_unbounded_cast_target_rejection(
        r#"
def recast[p_int](value: p_int) -> p_int = cast(value, p_int)
values = map(recast, [cast(127, i8)])
value = index(values, cast(0, i64))
"#,
        "p_int",
        "recast",
    );
}

/// chelis#1558 positive twin. **Disposition lock**: the checked `map` callback
/// type still specializes the bounded closure.
#[test]
fn bounded_cast_target_survives_map_callback_specialization() {
    let checked = checked_surf(
        r#"
def recast[p_int: Int](value: p_int) -> p_int = cast(value, p_int)
values = map(recast, [cast(127, i8)])
value = index(values, cast(0, i64))
"#,
    );

    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("the checked map callback type must specialize the generic closure");
    let RuntimeValue::Scalar(payload) = outcome.host_bindings.get("value").expect("value") else {
        panic!("value should be a scalar")
    };
    assert_eq!(payload.dtype(), Prim::Int8);
}

/// chelis#1558 negative control. **Regression test**: the `fold` accumulator
/// edge is no different from the `map` element edge.
#[test]
fn unbounded_cast_target_in_a_fold_callback_is_rejected_at_check_time() {
    expect_unbounded_cast_target_rejection(
        r#"
def keep_left[p_int](left: p_int, right: p_int) -> p_int = cast(left, p_int)
value = fold(keep_left, cast(127, i8), [cast(1, i8)])
"#,
        "p_int",
        "keep_left",
    );
}

/// chelis#1558 positive twin. **Disposition lock**: the checked `fold` callback
/// type still specializes the bounded closure.
#[test]
fn bounded_cast_target_survives_fold_callback_specialization() {
    let checked = checked_surf(
        r#"
def keep_left[p_int: Int](left: p_int, right: p_int) -> p_int = cast(left, p_int)
value = fold(keep_left, cast(127, i8), [cast(1, i8)])
"#,
    );

    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("the checked fold callback type must specialize the generic closure");
    let RuntimeValue::Scalar(payload) = outcome.host_bindings.get("value").expect("value") else {
        panic!("value should be a scalar")
    };
    assert_eq!(payload.dtype(), Prim::Int8);
}

/// chelis#1558 negative control. **Regression test**, and the row whose cause
/// was misread: the program's only literal cast to a binder is `cast(0, p_int)`
/// inside `nonnegative[p_int: Int]`, a BOUNDED binder that PR #1545's literal
/// rule always accepted. What this program trips is its three unbounded defs,
/// each casting a variable. `keep_left_hof` is named because it is the first
/// such declaration the checker reaches.
#[test]
fn unbounded_cast_target_across_higher_order_edges_is_rejected_at_check_time() {
    expect_unbounded_cast_target_rejection(
        r#"
def nonnegative[p_int: Int](value: p_int) -> bool =
  gte(cast(value, p_int), cast(0, p_int))
def keep_left_hof[p_int](left: p_int, right: p_int) -> p_int =
  cast(left, p_int)
scanned = scan(keep_left_hof, cast(7, i8), [cast(1, i8)])
"#,
        "p_int",
        "keep_left_hof",
    );
}

/// chelis#1558 positive twin. **Disposition lock**: every higher-order callback
/// edge preserves its checked specialization when the binder carries a bound.
/// This is the row that proves the inversion costs no behavioural coverage, so
/// it keeps all five edges the old row exercised.
#[test]
fn bounded_cast_target_survives_every_higher_order_callback_edge() {
    let checked = checked_surf(
        r#"
def nonnegative[p_int: Int](value: p_int) -> bool =
  gte(cast(value, p_int), cast(0, p_int))
def keep_left_hof[p_int: Int](left: p_int, right: p_int) -> p_int =
  cast(left, p_int)
def singleton[p_int: Int](value: p_int) -> List[p_int] = [cast(value, p_int)]
def keep_state[p_int: Int](state: p_int, index: i64) -> p_int = cast(state, p_int)

filtered = filter(nonnegative, [cast(-1, i8), cast(2, i8)])
scanned = scan(keep_left_hof, cast(7, i8), [cast(1, i8)])
partitioned = partition(nonnegative, [cast(-1, i8), cast(2, i8)])
flattened = flat_map(singleton, [cast(3, i8)])
generated = tensor_scan(cast(9, i8), keep_state, cast(2, i64))
"#,
    );

    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("every HOF callback edge must preserve its checked specialization");
    for name in ["filtered", "scanned", "flattened"] {
        let RuntimeValue::List(items) = outcome.host_bindings.get(name).expect(name) else {
            panic!("{name} should be a list")
        };
        assert!(items.iter().all(
            |item| matches!(item, RuntimeValue::Scalar(payload) if payload.dtype() == Prim::Int8)
        ));
    }
    let RuntimeValue::Tensor(generated) =
        outcome.host_bindings.get("generated").expect("generated")
    else {
        panic!("generated should be a tensor")
    };
    assert_eq!(generated.precision, Prim::Int8);
    assert!(matches!(
        outcome.host_bindings.get("partitioned"),
        Some(RuntimeValue::Tuple(parts)) if parts.len() == 2
    ));
}

// ----- Phase 3t.1: test_assert_* builtins -----

#[test]
fn test_assert_true_returns_unit() {
    let checked = checked_surf(r#"x = test_assert(true, "ok")"#);
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("test_assert(true, ...) should evaluate to Ok");
    let value = outcome.host_bindings.get("x").expect("x binding");
    assert!(matches!(value, RuntimeValue::Unit), "got {value:?}");
}

#[test]
fn test_assert_false_returns_err_with_label() {
    let checked = checked_surf(r#"x = test_assert(false, "my-label")"#);
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("test_assert(false, ...) should surface as host Err");
    assert!(
        err.contains("assert failed") && err.contains("my-label"),
        "expected 'assert failed' and 'my-label' in error, got: {err}"
    );
}

#[test]
fn generic_test_assert_eq_float_mismatch_includes_actual_and_expected() {
    let checked = checked_surf(r#"x = test_assert_eq(1.0, 2.0, "label")"#);
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("mismatched f32 assert should surface as host Err");
    assert!(err.contains("1") && err.contains("2"), "got: {err}");
    assert!(err.contains("label"), "expected label in error, got: {err}");
}

#[test]
fn generic_test_assert_eq_float_match_returns_unit() {
    let checked = checked_surf(r#"x = test_assert_eq(1.5, 1.5, "same")"#);
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("matched f32 assert should evaluate");
    let value = outcome.host_bindings.get("x").expect("x binding");
    assert!(matches!(value, RuntimeValue::Unit), "got {value:?}");
}

#[test]
fn generic_test_assert_eq_int_match_and_mismatch() {
    let ok = checked_surf(r#"x = test_assert_eq(cast(3, i64), cast(3, i64), "i")"#);
    let outcome = evaluate_host_program(&ok, &UnordMap::new()).expect("int match should eval");
    assert!(matches!(
        outcome.host_bindings.get("x"),
        Some(RuntimeValue::Unit)
    ));

    let bad = checked_surf(r#"x = test_assert_eq(cast(3, i64), cast(5, i64), "i")"#);
    let err =
        evaluate_host_program(&bad, &UnordMap::new()).expect_err("int mismatch should surface Err");
    assert!(
        err.contains("3") && err.contains("5") && err.contains("i"),
        "got: {err}"
    );
}

#[test]
fn generic_test_assert_eq_bool_match_and_mismatch() {
    let ok = checked_surf(r#"x = test_assert_eq(true, true, "b")"#);
    evaluate_host_program(&ok, &UnordMap::new()).expect("bool match should eval");

    let bad = checked_surf(r#"x = test_assert_eq(true, false, "b")"#);
    let err = evaluate_host_program(&bad, &UnordMap::new())
        .expect_err("bool mismatch should surface Err");
    assert!(
        err.contains("true") && err.contains("false") && err.contains("b"),
        "got: {err}"
    );
}

#[test]
fn generic_test_assert_eq_string_match_and_mismatch() {
    let ok = checked_surf(r#"x = test_assert_eq("hi", "hi", "s")"#);
    evaluate_host_program(&ok, &UnordMap::new()).expect("string match should eval");

    let bad = checked_surf(r#"x = test_assert_eq("foo", "bar", "s")"#);
    let err = evaluate_host_program(&bad, &UnordMap::new())
        .expect_err("string mismatch should surface Err");
    assert!(
        err.contains("foo") && err.contains("bar") && err.contains("s"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_reports_first_mismatch_index() {
    // Use 4-element tensors [1.0, 2.0, 3.0, 4.0] vs [1.0, 2.0, 99.0, 4.0]:
    // index 2 is the first mismatch.
    let checked = checked_surf(
        r#"
actual: tensor[4, f32] = to_tensor([1.0, 2.0, 3.0, 4.0], f32)
expected: tensor[4, f32] = to_tensor([1.0, 2.0, 99.0, 4.0], f32)
x = test_assert_close_tensor(actual, expected, 0.001, "close")
"#,
    );
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("close-tensor mismatch should surface Err");
    assert!(
        err.contains("at index 2") && err.contains("close"),
        "expected 'at index 2' and label, got: {err}"
    );
    assert!(err.contains("99") && err.contains('3'), "got: {err}");
}

#[test]
fn test_assert_close_tensor_match_returns_unit() {
    let checked = checked_surf(
        r#"
actual: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0], f32)
expected: tensor[3, f32] = to_tensor([1.001, 2.001, 3.001], f32)
x = test_assert_close_tensor(actual, expected, 0.01, "close")
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("close-tensor within tol should evaluate to Unit");
    assert!(matches!(
        outcome.host_bindings.get("x"),
        Some(RuntimeValue::Unit)
    ));
}

#[test]
fn test_assert_close_tensor_zero_tol_passes_bit_exact() {
    // Regression: tol = 0 with identical data must pass, not report a false mismatch.
    let checked = checked_surf(
        r#"
actual: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0], f32)
expected: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0], f32)
x = test_assert_close_tensor(actual, expected, 0.0, "bit-exact")
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("zero-tol bit-exact equality should pass");
    assert!(matches!(
        outcome.host_bindings.get("x"),
        Some(RuntimeValue::Unit)
    ));
}

#[test]
fn test_assert_close_tensor_zero_tol_rejects_any_delta() {
    let checked = checked_surf(
        r#"
actual: tensor[2, f32] = to_tensor([1.0, 2.0], f32)
expected: tensor[2, f32] = to_tensor([1.0, 2.00001], f32)
x = test_assert_close_tensor(actual, expected, 0.0, "strict")
"#,
    );
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("zero-tol with any delta must fail");
    assert!(
        err.contains("at index 1") && err.contains("strict"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_nan_actual_fails() {
    // Regression: NaN in actual must fail. (NaN - x).abs() is NaN, which silently
    // passes the old `>= tol` check. sqrt(-1.0) produces NaN.
    let checked = checked_surf(
        r#"
nan_val: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, nan_val], f32)
expected: tensor[2, f32] = to_tensor([1.0, 2.0], f32)
x = test_assert_close_tensor(actual, expected, 0.01, "nan-actual")
"#,
    );
    let err =
        evaluate_host_program(&checked, &UnordMap::new()).expect_err("NaN in actual must fail");
    assert!(
        err.contains("at index 1") && err.contains("NaN"),
        "expected NaN-aware diagnostic, got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_nan_expected_fails() {
    let checked = checked_surf(
        r#"
nan_val: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, 2.0], f32)
expected: tensor[2, f32] = to_tensor([1.0, nan_val], f32)
x = test_assert_close_tensor(actual, expected, 0.01, "nan-expected")
"#,
    );
    let err =
        evaluate_host_program(&checked, &UnordMap::new()).expect_err("NaN in expected must fail");
    assert!(
        err.contains("at index 1") && err.contains("NaN"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_negative_tol_rejected() {
    let checked = checked_surf(
        r#"
actual: tensor[2, f32] = to_tensor([1.0, 2.0], f32)
expected: tensor[2, f32] = to_tensor([1.0, 2.0], f32)
x = test_assert_close_tensor(actual, expected, -0.001, "neg-tol")
"#,
    );
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("negative tol must be rejected");
    assert!(
        err.contains("invalid tolerance") && err.contains("neg-tol"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_nan_tol_rejected() {
    let checked = checked_surf(
        r#"
nan_tol: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, 2.0], f32)
expected: tensor[2, f32] = to_tensor([1.0, 4.0], f32)
x = test_assert_close_tensor(actual, expected, nan_tol, "nan-tol")
"#,
    );
    let err =
        evaluate_host_program(&checked, &UnordMap::new()).expect_err("NaN tol must be rejected");
    assert!(
        err.contains("invalid tolerance") && err.contains("nan-tol"),
        "got: {err}"
    );
}

/// Evaluate one Deep expression against hand-supplied runtime bindings.
///
/// The `test_assert_close_tensor` eval arm guards operands the checker has
/// already constrained: `reject_inadmissible_operand_dtypes` requires the
/// tolerance to carry the tensor's own float dtype, so no CHECKED program
/// reaches the arm's own dtype, shape, and tolerance rejections. They exist
/// for dynamically-constructed calls, and their whole job is to name the value
/// they were handed, so the runtime library's `eval_expr` entry rather than
/// the checked pipeline is where their text is observable.
fn eval_deep_with_bindings(
    surf_expr: &str,
    args: &[(&str, RuntimeValue)],
) -> Result<String, String> {
    let source = format!("probe = {surf_expr}\n");
    let decls = chelis_surf::parser::parse_str(&source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let Expr::Node(def, _) = &exprs[0] else {
        panic!("desugaring a top-level binding yields one def form");
    };
    // `(def {} <name> <body>)`: the body is the second child.
    let body = def.children_slice()[1].clone();

    let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    let mut ctx = EvalContext {
        bindings: Frame::new(),
        result_producer: None,
        binding_types: UnordMap::new(),
        precision_bindings: UnordMap::new(),
        declaration_values: UnordMap::new(),
        named_axis_route_cache: UnordMap::new(),
        named_axis_route_visiting: UnordSet::new(),
        program: ProgramScope::new(UnordMap::new(), UnordMap::new()),
        declared_signatures: UnordMap::new(),
        adt_fields: UnordMap::new(),
        constructor_names: UnordMap::new(),
        adt_registry: chelis_types::adt::AdtRegistry::default(),
        tensor_bindings: &empty_tensors,
        session: None,
        active_declaration_names: Vec::new(),
        def_kernels: UnordMap::new(),
        transcript: Vec::new(),
        transcript_capture: None,
        resolving_top_levels: Vec::new(),
        cancel: None,
        system: system::EvalSystemBoundary::permissive(),
        failure_kind: RuntimeFailureKind::Ordinary,
        activation_extents: Default::default(),
    };
    for (name, value) in args {
        ctx.bindings.insert((*name).to_string(), value.clone());
        ctx.binding_types.insert((*name).to_string(), None);
    }
    ctx.eval_expr(&body).map(|value| render_value(&value))
}

#[test]
fn test_assert_close_tensor_f32_uses_f32_subtraction() {
    // These are exact f32 values. Their exact widened difference is
    // 0.9999999701976741, but f32 subtraction rounds it to the tolerance
    // 0.9999999403953552. [05-OP-35] therefore accepts the f32 pair; an f64
    // funnel rejects it.
    let actual = tensor_value(Prim::F32, vec![1], vec![2.980232594040899e-8]);
    let expected = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let tolerance = scalar_of(Prim::F32, 0.9999999403953552);
    let rendered = eval_deep_with_bindings(
        r#"test_assert_close_tensor(actual, expected, tolerance, "f32-width")"#,
        &[
            ("actual", actual),
            ("expected", expected),
            ("tolerance", tolerance),
        ],
    )
    .expect("the f32-width comparison must accept the rounded difference");
    assert_eq!(rendered, "()");
}

#[test]
fn test_assert_close_tensor_f64_keeps_the_f64_boundary_distinct() {
    let actual = tensor_value(Prim::F64, vec![1], vec![2.980232594040899e-8]);
    let expected = tensor_value(Prim::F64, vec![1], vec![1.0]);
    let tolerance = scalar_of(Prim::F64, 0.9999999403953552);
    let error = eval_deep_with_bindings(
        r#"test_assert_close_tensor(actual, expected, tolerance, "f64-width")"#,
        &[
            ("actual", actual),
            ("expected", expected),
            ("tolerance", tolerance),
        ],
    )
    .expect_err("the exact f64 difference is greater than the tolerance");
    assert!(error.contains("at index 0") && error.contains("f64-width"));
}

#[test]
fn test_assert_close_tensor_mismatch_uses_own_width_canonical_digits() {
    for dtype in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        let actual = tensor_value(dtype, vec![1], vec![0.1]);
        let expected = tensor_value(dtype, vec![1], vec![0.2]);
        let tolerance = scalar_of(dtype, 0.0);
        let error = eval_deep_with_bindings(
            r#"test_assert_close_tensor(actual, expected, tolerance, "digits")"#,
            &[
                ("actual", actual),
                ("expected", expected),
                ("tolerance", tolerance),
            ],
        )
        .expect_err("the values differ with zero tolerance");
        assert_eq!(
            error,
            "assert_close_tensor (digits): at index 0 expected 0.2, got 0.1, tol 0.0",
            "{} diagnostic must render every value at its stored width",
            dtype.name()
        );
    }
}

#[test]
fn test_assert_close_tensor_special_mismatch_uses_canonical_spellings() {
    let actual = tensor_value(Prim::F32, vec![1], vec![f64::NAN]);
    let expected = tensor_value(Prim::F32, vec![1], vec![f64::INFINITY]);
    let tolerance = scalar_of(Prim::F32, 0.0);
    let error = eval_deep_with_bindings(
        r#"test_assert_close_tensor(actual, expected, tolerance, "special-digits")"#,
        &[
            ("actual", actual),
            ("expected", expected),
            ("tolerance", tolerance),
        ],
    )
    .expect_err("NaN is never close");
    assert_eq!(
        error,
        "assert_close_tensor (special-digits): at index 0 expected inf, got NaN, tol 0.0 (NaN is never close)"
    );
}

#[test]
fn test_assert_close_tensor_invalid_tolerance_uses_own_width_canonical_digits() {
    let actual = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let expected = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let tolerance = scalar_of(Prim::F32, -0.1);
    let error = eval_deep_with_bindings(
        r#"test_assert_close_tensor(actual, expected, tolerance, "tol-digits")"#,
        &[
            ("actual", actual),
            ("expected", expected),
            ("tolerance", tolerance),
        ],
    )
    .expect_err("a negative tolerance is invalid");
    assert_eq!(
        error,
        "assert_close_tensor (tol-digits): invalid tolerance -0.1 (must be finite and non-negative)"
    );
}

#[test]
fn test_assert_close_tensor_accepts_each_active_float_dtype() {
    for dtype in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        let actual = tensor_value(dtype, vec![2], vec![-0.0, 1.0]);
        let expected = tensor_value(dtype, vec![2], vec![0.0, 1.001]);
        let tolerance = scalar_of(dtype, 0.01);
        eval_deep_with_bindings(
            r#"test_assert_close_tensor(actual, expected, tolerance, "matrix")"#,
            &[
                ("actual", actual),
                ("expected", expected),
                ("tolerance", tolerance),
            ],
        )
        .unwrap_or_else(|error| panic!("{} own-width comparison failed: {error}", dtype.name()));
    }
}

#[test]
fn test_assert_close_tensor_rejects_infinite_tolerance() {
    let actual = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let expected = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let tolerance = scalar_of(Prim::F32, f64::INFINITY);
    let error = eval_deep_with_bindings(
        r#"test_assert_close_tensor(actual, expected, tolerance, "infinite-tol")"#,
        &[
            ("actual", actual),
            ("expected", expected),
            ("tolerance", tolerance),
        ],
    )
    .expect_err("the tolerance must be finite");
    assert!(error.contains("invalid tolerance") && error.contains("infinite-tol"));
}

#[test]
fn test_assert_close_tensor_runtime_shape_mismatch_uses_decimal_shape_rendering() {
    let actual = tensor_value(Prim::F32, vec![2], vec![1.0, 2.0]);
    let expected = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let tolerance = scalar_of(Prim::F32, 0.0);
    let error = eval_deep_with_bindings(
        r#"test_assert_close_tensor(actual, expected, tolerance, "shape")"#,
        &[
            ("actual", actual),
            ("expected", expected),
            ("tolerance", tolerance),
        ],
    )
    .expect_err("different tensor shapes must fail before comparison");
    assert!(error.contains("shape mismatch, expected [1], got [2]"));
}

#[test]
fn test_assert_close_tensor_runtime_non_float_tolerance_uses_canonical_renderer() {
    let actual = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let expected = tensor_value(Prim::F32, vec![1], vec![1.0]);
    let tolerance = scalar_of(Prim::Int32, 0.0);
    let error = eval_deep_with_bindings(
        r#"test_assert_close_tensor(actual, expected, tolerance, "dtype")"#,
        &[
            ("actual", actual),
            ("expected", expected),
            ("tolerance", tolerance),
        ],
    )
    .expect_err("a non-float tolerance must fail defensively at runtime");
    assert!(error.contains("expected float tolerance, got i32 0"));
}

#[test]
fn test_assert_close_tensor_infinity_and_signed_zero_contract() {
    for dtype in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        let actual = tensor_value(dtype, vec![3], vec![f64::INFINITY, f64::NEG_INFINITY, -0.0]);
        let expected = tensor_value(dtype, vec![3], vec![f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let tolerance = scalar_of(dtype, 0.0);
        eval_deep_with_bindings(
            r#"test_assert_close_tensor(actual, expected, tolerance, "special")"#,
            &[
                ("actual", actual),
                ("expected", expected),
                ("tolerance", tolerance),
            ],
        )
        .unwrap_or_else(|error| panic!("{} equal infinities/zeros failed: {error}", dtype.name()));

        let actual = tensor_value(dtype, vec![2], vec![f64::INFINITY, 1.0]);
        let expected = tensor_value(dtype, vec![2], vec![f64::NEG_INFINITY, f64::INFINITY]);
        let finite_tolerance = match dtype {
            Prim::F16 => 65_504.0,
            Prim::Bf16 | Prim::F32 => 3.0e38,
            Prim::F64 => 1.0e300,
            _ => unreachable!("test loops active float dtypes"),
        };
        let tolerance = scalar_of(dtype, finite_tolerance);
        let error = eval_deep_with_bindings(
            r#"test_assert_close_tensor(actual, expected, tolerance, "nonfinite")"#,
            &[
                ("actual", actual),
                ("expected", expected),
                ("tolerance", tolerance),
            ],
        )
        .expect_err("opposite or finite/infinite pairs are never close");
        assert!(error.contains("at index 0") && error.contains("nonfinite"));
    }
}

#[test]
fn test_assert_close_tensor_arm_has_no_lossy_f64_tensor_funnel() {
    let source = include_str!("eval.rs");
    let arm = source
        .split_once("\"test_assert_close_tensor\" =>")
        .expect("assert-close dispatch arm")
        .1
        .split_once("\n            \"debug\" =>")
        .expect("next dispatch arm")
        .0;
    assert!(arm.contains("StorageView::F32"));
    assert!(arm.contains("StorageView::F16"));
    assert!(arm.contains("StorageView::Bf16"));
    assert!(arm.contains("StorageView::F64"));
    assert!(
        !arm.contains("to_f64_lossy_vec"),
        "[05-OP-35] forbids a whole-tensor f64 comparison funnel"
    );
}

// ----- N2 fix: matmul / permute / sum host evaluator coverage -----
//
// Pins the closure of the upstream-reported gap: native `chelis test`
// erroring with `unsupported builtin 'matmul'` / `'permute'` / `'sum'`
// when those primitives appear in a test's dependency graph.

fn first_tensor_data(outcome: &RuntimeOutcome, name: &str) -> Vec<f64> {
    match outcome.host_bindings.get(name) {
        Some(RuntimeValue::Tensor(t)) => t.value.to_f64_lossy_vec().clone(),
        other => panic!("expected tensor binding {name}, got {other:?}"),
    }
}

fn first_tensor_shape(outcome: &RuntimeOutcome, name: &str) -> Vec<usize> {
    match outcome.host_bindings.get(name) {
        Some(RuntimeValue::Tensor(t)) => t.value.shape.clone(),
        other => panic!("expected tensor binding {name}, got {other:?}"),
    }
}

#[test]
fn host_runtime_matmul_2x2_identity_passthrough() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, i64), cast(0.0, f32))
b = pad_sequences_to([[cast(3.0, f32), cast(5.0, f32)], [cast(7.0, f32), cast(11.0, f32)]], cast(2, i64), cast(0.0, f32))
y = matmul(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("matmul should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![3.0, 5.0, 7.0, 11.0]);
}

// IEEE-754 corner cases for `Div` and `Recip` on the runtime
// evaluator path. A `mul(a, exp(neg(log(b))))` decomposition
// would NaN on every non-positive operand below; these tests
// pin the IEEE-correct outputs and serve as regression guards.
#[test]
fn host_runtime_div_negative_divisor_returns_finite_value() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(5.0, f32)])
b = to_tensor([cast(-2.0, f32)])
y = div(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("div with negative divisor should evaluate");
    // IEEE: 5 / -2 = -2.5 (an `exp(neg(log(-2)))` decomposition
    // would NaN here).
    assert_eq!(first_tensor_data(&outcome, "y"), vec![-2.5]);
}

#[test]
fn host_runtime_div_by_positive_zero_is_positive_infinity() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(1.0, f32)])
b = to_tensor([cast(0.0, f32)])
y = div(a, b)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &UnordMap::new()).expect("div by zero should evaluate");
    let v = first_tensor_data(&outcome, "y");
    assert_eq!(v.len(), 1);
    assert!(
        v[0].is_infinite() && v[0] > 0.0,
        "expected +inf, got {}",
        v[0]
    );
}

#[test]
fn host_runtime_div_negative_one_by_zero_is_negative_infinity() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(-1.0, f32)])
b = to_tensor([cast(0.0, f32)])
y = div(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new()).expect("-1/0 should evaluate");
    let v = first_tensor_data(&outcome, "y");
    assert_eq!(v.len(), 1);
    assert!(
        v[0].is_infinite() && v[0] < 0.0,
        "expected -inf, got {}",
        v[0]
    );
}

#[test]
fn host_runtime_div_zero_by_zero_is_nan() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(0.0, f32)])
b = to_tensor([cast(0.0, f32)])
y = div(a, b)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &UnordMap::new()).expect("0/0 should evaluate (NaN)");
    let v = first_tensor_data(&outcome, "y");
    assert_eq!(v.len(), 1);
    assert!(v[0].is_nan(), "expected NaN, got {}", v[0]);
}

// chelis#458 regression lock (eval/fold side). On chelis 0.9.0 a mixed
// `(f32, i32)` numeric op silently const-folded when both operands were
// literals: `div(1.0, 4)` evaluated to 0.25 via a `_ => Prim::F32`
// re-precisioning fallback, while the bound-variable form errored at
// type-check — a check↔eval inconsistency (the #458 defect). Per
// spec/04-type-system.md §5.1 (no implicit promotion) and §5.2 (casts are
// always explicit) the mixed pair is a type error, so the literal form
// must NOT fold to 0.25. These two tests pin both barriers: the
// type-checker rejects the literal pair before eval, AND the host scalar
// dispatch itself rejects a mixed pair rather than re-precisioning it.

/// The full check→eval pipeline rejects `div(1.0, 4)` at type-check, so
/// the const-fold to 0.25 is never reached. `4` is an `i32` literal
/// (§5.3), making this the same mixed `(f32, i32)` pair as the
/// bound-variable form.
#[test]
fn issue_458_div_f32_over_i32_literal_rejected_before_eval_not_folded() {
    let decls = chelis_surf::parser::parse_str("out = div(1.0, 4)").expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let err = chelis_types::check_ir_program(&exprs)
        .expect_err("chelis#458: div(1.0, 4) is a mixed (f32, i32) pair and must be a type error");
    assert!(
        err.errors.iter().any(|e| matches!(
            e.kind,
            chelis_types::errors::CheckErrorKind::PrecisionMismatch
        )),
        "spec §5.1: mixed (f32, i32) div must surface a PrecisionMismatch; got: {:?}",
        err.errors
    );
}

/// Runtime backstop: even if a future type-checker hole let a mixed
/// `(f32, i32)` scalar pair reach the host evaluator, the closed kernel
/// dispatcher
/// must reject it rather than re-precisioning to a 0.25 float fold (the old
/// 0.9.0 `_ => Prim::F32` fallback). The mixed pair is neither int-int nor
/// float-float, so it falls to the `_ => Err(..)` catch-all.
#[test]
fn issue_458_closed_binop_dispatch_rejects_mixed_f32_i32_not_folds_to_quarter() {
    let lhs = RuntimeValue::scalar_like_float(chelis_types::types::Prim::F32, 1.0)
        .expect("f32 scalar 1.0");
    let rhs =
        RuntimeValue::scalar_like_int(chelis_types::types::Prim::Int32, 4).expect("i32 scalar 4");
    let result = numeric_binop(
        &[lhs, rhs],
        Some(chelis_types::IntBinOp::TruncDiv),
        Some(chelis_types::FloatBinOp::Div),
    );
    assert!(
        result.is_err(),
        "chelis#458 / spec §5.1: a mixed (f32, i32) scalar div must be rejected by the \
         host dispatch, NOT silently re-precisioned and folded to 0.25; got Ok({result:?})"
    );
}

#[test]
fn host_runtime_recip_negative_value_is_negative_reciprocal() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(-2.0, f32)])
y = recip(a)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &UnordMap::new()).expect("recip(-2.0) should evaluate");
    // IEEE: 1 / -2 = -0.5 (an `exp(neg(log(-2)))` decomposition
    // would NaN here).
    assert_eq!(first_tensor_data(&outcome, "y"), vec![-0.5]);
}

#[test]
fn host_runtime_matmul_2x3_3x2_basic() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, i64), cast(0.0, f32))
b = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(1.0, f32)]], cast(2, i64), cast(0.0, f32))
y = matmul(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("rectangular matmul should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    // Row 0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    // Row 1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
    assert_eq!(first_tensor_data(&outcome, "y"), vec![4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn host_runtime_permute_2x2_transpose_swaps_off_diagonal() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]], cast(2, i64), cast(0.0, f32))
y = permute(a, 1, 0)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("permute should evaluate under host runtime");
    // Row-major: original [[1,2],[3,4]] -> transpose [[1,3],[2,4]]
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn host_runtime_permute_2x3_transpose_to_3x2() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, i64), cast(0.0, f32))
y = permute(a, 1, 0)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("rectangular permute should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3, 2]);
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    assert_eq!(
        first_tensor_data(&outcome, "y"),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn host_runtime_sum_axis1_reduces_2x3_to_2() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, i64), cast(0.0, f32))
y = sum(a, cast(1, i32))
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("sum should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![6.0, 15.0]);
}

#[test]
fn host_runtime_sum_axis0_reduces_2x3_to_3() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, i64), cast(0.0, f32))
y = sum(a, cast(0, i32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &UnordMap::new()).expect("sum on axis 0 should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![5.0, 7.0, 9.0]);
}

// Issue Chelis-Lang/chelis#163: `sum` on f32 must use the stride-4 ILP
// cascade (torch's CPU `row_sum`) reduction order, not the previous
// strict left-fold. The 11-element reflected-pad sequence below is
// the issue's exact reproducer pattern: the same multiset summed in
// two different orderings produces the same result under stride-4
// (matches torch/numpy) but differs by 1 ULP under left-fold.
//
// Source values: torch.rand(6) with manual_seed(0); each f32 value
// is expressed as the f64 string that round-trips back to the same
// f32 bit pattern via the Surf `cast(_, f32)` path.
#[test]
fn host_runtime_sum_f32_uses_pairwise_order_for_issue_163_repro() {
    // right-pad: [v0, v1, v2, v3, v4, v5, v4, v3, v2, v1, v0]
    let checked_right = checked_surf(
        r#"
seq = to_tensor([
cast(0.49625658988952637, f32),
cast(0.7682217955589294, f32),
cast(0.08847743272781372, f32),
cast(0.13203048706054688, f32),
cast(0.30742114782333374, f32),
cast(0.6340786814689636, f32),
cast(0.30742114782333374, f32),
cast(0.13203048706054688, f32),
cast(0.08847743272781372, f32),
cast(0.7682217955589294, f32),
cast(0.49625658988952637, f32)
])
y = sum(seq, cast(0, i32))
"#,
    );
    let outcome_right = evaluate_host_program(&checked_right, &UnordMap::new())
        .expect("right-pad reflected sum should evaluate");
    let right = first_tensor_data(&outcome_right, "y");
    assert_eq!(right.len(), 1);
    // Stride-4 ILP cascade f32 result. Matches torch's CPU
    // `row_sum` bit-exactly for n <= 16; coincides with
    // `numpy.sum` only because this specific 11-element multiset
    // happens to round the same way under both the stride-4
    // cascade and numpy's pairwise tree — the two algorithms
    // disagree in general (numpy uses a divide-and-conquer
    // pairwise tree with 128-element blocks). The old strict
    // left-fold would have produced 4.218894004821777 here — a
    // 1-ULP drift that the parity harness now no longer needs to
    // carve out (issue #163 acceptance criterion).
    // Bit patterns rather than f32 decimal literals: clippy's
    // `excessive_precision` lint would rewrite the source
    // literals to shorter decimals that round to the SAME bits
    // but obscure intent. This regression-lock IS about exact
    // bits, so encode them directly.
    let stride4 = 0x4087012d_u32; // = 4.218893527984619 -> f32 (stride-4 cascade)
    let left_fold = 0x4087012e_u32; // = 4.218894004821777_f32 (old left-fold)
    assert_eq!(
        (right[0] as f32).to_bits(),
        stride4,
        "expected stride-4 cascade result; got {}",
        right[0]
    );
    // Negative regression-lock: the test must also assert the OLD
    // left-fold result is NOT produced, so a future change that
    // accidentally reverts to a left-fold (or to a different
    // tree shape that lands on the old value) fails loudly here.
    assert_ne!(
        (right[0] as f32).to_bits(),
        left_fold,
        "regression: result matches the old left-fold value 4.218894004821777, \
         which the stride-4 cascade was supposed to replace"
    );

    // Same multiset, left-pad ordering. Both stride-4 and the old
    // left-fold happen to agree here — pinning to prove parity stays
    // intact across the algorithm change.
    let checked_left = checked_surf(
        r#"
seq = to_tensor([
cast(0.30742114782333374, f32),
cast(0.13203048706054688, f32),
cast(0.08847743272781372, f32),
cast(0.7682217955589294, f32),
cast(0.49625658988952637, f32),
cast(0.49625658988952637, f32),
cast(0.7682217955589294, f32),
cast(0.08847743272781372, f32),
cast(0.13203048706054688, f32),
cast(0.30742114782333374, f32),
cast(0.6340786814689636, f32)
])
y = sum(seq, cast(0, i32))
"#,
    );
    let outcome_left = evaluate_host_program(&checked_left, &UnordMap::new())
        .expect("left-pad reflected sum should evaluate");
    let left = first_tensor_data(&outcome_left, "y");
    assert_eq!(left.len(), 1);
    assert_eq!(
        (left[0] as f32).to_bits(),
        stride4,
        "left-pad ordering must produce same result as right-pad under stride-4; got {}",
        left[0]
    );
}

/// Trace is diagonal followed by [05-OP-30]'s canonical adjacent-pair tree.
/// This cancellation sequence distinguishes it from both retired orders.
#[test]
fn host_runtime_trace_f32_uses_canonical_balanced_tree() {
    let diag = [1e20_f64, 1.0, -1e20_f64, 1.0, 1.0];
    let mut m = vec![0.0_f64; 5 * 5];
    for (i, &v) in diag.iter().enumerate() {
        m[i * 5 + i] = v;
    }
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![5, 5], m),
        precision: Prim::F32,
    };
    let out = tensor_trace_value(&tensor, 0, 1).expect("trace must evaluate");
    assert_eq!(out.value.shape, Vec::<usize>::new(), "trace is a scalar");
    assert_eq!(
        (out.value.to_f64_lossy_vec()[0] as f32).to_bits(),
        1.0_f32.to_bits()
    );
}

/// The f64 host lane uses the same canonical tree at f64 arithmetic width.
#[test]
fn host_runtime_trace_f64_uses_canonical_balanced_tree() {
    let diag = [1e300_f64, 1.0, -1e300_f64, 1.0, 1.0];
    let mut m = vec![0.0_f64; 5 * 5];
    for (i, &v) in diag.iter().enumerate() {
        m[i * 5 + i] = v;
    }
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![5, 5], m),
        precision: Prim::F64,
    };
    let out = tensor_trace_value(&tensor, 0, 1).expect("trace must evaluate");
    assert_eq!(out.value.shape, Vec::<usize>::new(), "trace is a scalar");
    assert_eq!(out.value.to_f64_lossy_vec()[0].to_bits(), 1.0_f64.to_bits());
}

/// [05-RWIN]: windowed extrema select the first NaN in row-major order and
/// preserve its exact stored representation.
#[test]
fn host_runtime_reduce_window_max_min_preserve_first_nan_bits() {
    let first_nan = f32::from_bits(0xffc1_2345);
    let second_nan = f32::from_bits(0x7fc5_4321);
    // The input is stored bits, not an arithmetic result, which `from_wide`
    // would finalize to the canonical NaN.
    let stored = [first_nan, 1.0, 2.0, second_nan]
        .map(|value| chelis_types::scalar_from_f64("test", Prim::F32, f64::from(value)).unwrap());
    let tensor = RuntimeTensorValue::from_scalars(Prim::F32, vec![4], &stored);
    let max = tensor_reduce_window_host(
        &tensor,
        &[2],
        &[2],
        ReduceWindowOp::Max,
        "reduce_window_max",
    )
    .expect("reduce_window max must evaluate");
    assert_eq!(max.value.shape, vec![2]);
    let chelis_types::dtype_semantics::StorageView::F32(max_values) = max.value.storage().view()
    else {
        panic!("window extrema must preserve f32 storage");
    };
    assert_eq!(
        max_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![first_nan.to_bits(), second_nan.to_bits()],
    );
    let min = tensor_reduce_window_host(
        &tensor,
        &[2],
        &[2],
        ReduceWindowOp::Min,
        "reduce_window_min",
    )
    .expect("reduce_window min must evaluate");
    let chelis_types::dtype_semantics::StorageView::F32(min_values) = min.value.storage().view()
    else {
        panic!("window extrema must preserve f32 storage");
    };
    assert_eq!(
        min_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![first_nan.to_bits(), second_nan.to_bits()],
    );
}

/// [05-OP-30] and section 4.1 require the f32 adjacent-pair tree.
/// For [2^24, forty ones, -2^24], only the first pair loses its unit:
/// the final two subtrees are 2^24 + 30 and -2^24 + 9, giving 39.
/// An f64 reference (40) and a left fold (0) both implement different graphs.
#[test]
fn host_runtime_matmul_f32_preserves_canonical_accumulator_tree() {
    let mut lhs_row = vec![16_777_216.0_f64];
    lhs_row.extend(std::iter::repeat_n(1.0_f64, 40));
    lhs_row.push(-16_777_216.0_f64);
    let k = lhs_row.len(); // 42
    let lhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1, k], lhs_row),
        precision: Prim::F32,
    };
    let rhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![k, 1], vec![1.0_f64; k]),
        precision: Prim::F32,
    };
    let out = tensor_matmul_host(&lhs, &rhs).expect("matmul must evaluate");
    assert_eq!(
        out.value.to_f64_lossy_vec(),
        vec![39.0_f64],
        "f32 matmul must execute its canonical accumulator tree"
    );
}

/// One `ij,j->i` einsum row against a ones vector: the result is the sum of
/// the row's products, each formed at the operand's §5.7.1 default
/// accumulator and summed in the C runtime's balanced order.
fn einsum_row_sum(precision: Prim, row: Vec<f64>) -> RuntimeTensorValue {
    let k = row.len();
    let lhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1, k], row),
        precision,
    };
    let rhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![k], vec![1.0_f64; k]),
        precision,
    };
    tensor_einsum_value("ij,j->i", &lhs, &rhs).expect("einsum must evaluate")
}

/// chelis#3041: eval accumulated einsum in f64 while `chelis_tensor_einsum`
/// accumulates at the default accumulator ([05-OP-51]: never an unrequested
/// f64 graph). Each witness's f64 sum rounds differently from the f32 one.
#[test]
fn host_runtime_einsum_accumulates_at_the_default_accumulator_like_the_c_runtime() {
    // f32: balanced pairs (2^24 + 1) + (1 + 1) = 2^24 + 2; f64 gave 2^24 + 4.
    let out = einsum_row_sum(Prim::F32, vec![16_777_216.0, 1.0, 1.0, 1.0]);
    assert_eq!(out.precision, Prim::F32);
    assert_eq!(out.value.to_f64_lossy_vec(), vec![16_777_218.0]);
    // 2^24 + forty ones - 2^24: the balanced f32 tree loses one unit.
    let mut row = vec![16_777_216.0_f64];
    row.extend(std::iter::repeat_n(1.0_f64, 40));
    row.push(-16_777_216.0);
    assert_eq!(
        einsum_row_sum(Prim::F32, row).value.to_f64_lossy_vec(),
        vec![39.0]
    );
    // bf16: the f32 total 1 + 2^-8 is a tie that narrows to 1.0 (0x3f80);
    // f64 kept 2^-30 and narrowed up to 0x3f81.
    let out = einsum_row_sum(Prim::Bf16, vec![1.0, 2.0_f64.powi(-8), 2.0_f64.powi(-30)]);
    assert_eq!(out.precision, Prim::Bf16);
    assert_eq!(out.value.to_f64_lossy_vec(), vec![1.0]);
    // f16: 2^-24 is half an f32 unit at 1, so the f32 total is the tie
    // 1 + 2^-11, which narrows to 1.0 (0x3c00); f64 narrowed up to 0x3c01.
    let out = einsum_row_sum(Prim::F16, vec![1.0, 2.0_f64.powi(-11), 2.0_f64.powi(-24)]);
    assert_eq!(out.precision, Prim::F16);
    assert_eq!(out.value.to_f64_lossy_vec(), vec![1.0]);
}

/// An integer contraction's overflow traps as the contraction, as
/// `chelis_tensor_einsum` and `chelis_tensor_trace` report it, whether the
/// product (`[[65536]]·[65536]`) or the balanced sum
/// (`[[2147483647, 1]]·[1, 1]`) leaves the i32 accumulator.
#[test]
fn host_runtime_contraction_overflow_traps_under_the_contraction_name() {
    let tensor = |shape: Vec<usize>, values: Vec<f64>| {
        RuntimeTensorValue::from_wide("test", Prim::Int32, shape, values)
            .expect("i32 fixtures are in range")
    };
    let sum = tensor_einsum_value(
        "ij,j->i",
        &tensor(vec![1, 2], vec![2_147_483_647.0, 1.0]),
        &tensor(vec![2], vec![1.0, 1.0]),
    );
    assert_eq!(
        sum.err().as_deref(),
        Some("numeric trap: overflow in einsum at i32")
    );
    let product = tensor_einsum_value(
        "ij,j->i",
        &tensor(vec![1, 1], vec![65_536.0]),
        &tensor(vec![1], vec![65_536.0]),
    );
    assert_eq!(
        product.err().as_deref(),
        Some("numeric trap: overflow in einsum at i32")
    );
    let trace = tensor_trace_value(
        &tensor(vec![2, 2], vec![2_147_483_647.0, 0.0, 0.0, 1.0]),
        0,
        1,
    );
    assert_eq!(
        trace.err().as_deref(),
        Some("numeric trap: overflow in trace at i32")
    );
}

#[test]
fn host_runtime_einsum_accepts_the_legal_rank_zero_grammar() {
    let lhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![], vec![2.0]),
        precision: Prim::F32,
    };
    let rhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![], vec![3.0]),
        precision: Prim::F32,
    };
    let output = tensor_einsum_value(",->", &lhs, &rhs).expect("rank-zero einsum is legal");
    assert_eq!(output.value.shape, Vec::<usize>::new());
    assert_eq!(output.value.to_f64_lossy_vec(), vec![6.0]);
}

/// A zero extent means zero elements wherever the zero sits, so both shapes
/// below describe the same empty operand and the derived reduction count is
/// zero for both. A left-to-right checked fold reaches `BIG * BIG` first,
/// which is not an i64, and so accepted one permutation while rejecting the
/// other. The C runtime holds the same invariant in
/// `crates/chelis-runtime/tests/op33_int64_extent_domain.rs`.
#[test]
fn host_runtime_einsum_zero_extent_acceptance_does_not_depend_on_axis_order() {
    const BIG: usize = 4_000_000_000;
    // The zero's axis is the only difference between the two cases, and the
    // equation names it, so diagnostics identify the case by equation rather
    // than by Debug-printing the extents (faithful_observation.md B2.4: no
    // third formatter in an observation exit surface).
    for (shape, equation) in [
        (vec![BIG, 0, BIG], "abc,def->b"),
        (vec![BIG, BIG, 0], "abc,def->c"),
    ] {
        let operand = RuntimeTensorValue {
            value: IrTensorValue::from_vec(shape, Vec::new()),
            precision: Prim::F32,
        };
        let output = tensor_einsum_value(equation, &operand, &operand)
            .unwrap_or_else(|error| panic!("host einsum `{equation}` must evaluate: {error}"));
        assert_eq!(
            output.value.shape,
            vec![0],
            "host einsum `{equation}` must produce an empty result"
        );
    }
}

/// [05-OP-33]: `diagonal` "keeps source axis order with the second axis
/// removed", so its output holds one coordinate per RETAINED source axis and
/// the diagonal's own coordinate sits at axis1's position AFTER that removal.
/// Reading that vector at the SOURCE axis number picked up a neighbouring
/// axis's coordinate and panicked when axis1 was the last source axis, and the
/// remaining coordinates were not shifted past the slot the diagonal occupies.
/// `trace` then has to reduce the axis the diagonal was written into, which
/// `min(axis1, axis2)` names only for an adjacent pair.
///
/// Every expectation is the closed form of the ramp `in[i][j][k] = 4i + 2j + k`
/// evaluated at the coordinates the atom names. The C runtime holds the same
/// invariants in `crates/chelis-runtime/tests/op33_diagonal_axis_mapping.rs`;
/// this is the host lane's half, and the two lanes agreeing is the point:
/// before this repair they were bug-compatible, so eval-vs-C parity was green
/// on a wrong answer.
#[test]
fn host_runtime_diagonal_and_trace_map_every_axis_pair_to_source_coordinates() {
    let ramp = |shape: Vec<usize>| {
        let count: usize = shape.iter().product();
        RuntimeTensorValue {
            value: IrTensorValue::from_vec(
                shape,
                (0..count).map(|index| index as f64).collect::<Vec<_>>(),
            ),
            precision: Prim::F32,
        }
    };

    /// One axis-pair case: source shape, the two axes, and the output shape
    /// and elements [05-OP-33] requires.
    struct AxisPairCase {
        label: &'static str,
        shape: Vec<usize>,
        axis1: i64,
        axis2: i64,
        out_shape: Vec<usize>,
        elements: Vec<f64>,
    }
    let case =
        |label, shape: Vec<usize>, axis1, axis2, out_shape: Vec<usize>, elements| AxisPairCase {
            label,
            shape,
            axis1,
            axis2,
            out_shape,
            elements,
        };

    let diagonal_cases = vec![
        case("rank2-forward", vec![2, 2], 0, 1, vec![2], vec![0.0, 3.0]),
        case("rank2-reversed", vec![2, 2], 1, 0, vec![2], vec![0.0, 3.0]),
        case(
            "rank2-negative-axes-reversed",
            vec![2, 2],
            -1,
            -2,
            vec![2],
            vec![0.0, 3.0],
        ),
        case(
            "rank3-leading-pair",
            vec![2, 2, 2],
            0,
            1,
            vec![2, 2],
            vec![0.0, 1.0, 6.0, 7.0],
        ),
        case(
            "rank3-trailing-pair",
            vec![2, 2, 2],
            1,
            2,
            vec![2, 2],
            vec![0.0, 3.0, 4.0, 7.0],
        ),
        case(
            "rank3-straddling-forward",
            vec![2, 2, 2],
            0,
            2,
            vec![2, 2],
            vec![0.0, 2.0, 5.0, 7.0],
        ),
        case(
            "rank3-straddling-reversed",
            vec![2, 2, 2],
            2,
            0,
            vec![2, 2],
            vec![0.0, 5.0, 2.0, 7.0],
        ),
        case(
            "rank4-straddling-reversed",
            vec![2, 2, 2, 2],
            2,
            0,
            vec![2, 2, 2],
            vec![0.0, 1.0, 10.0, 11.0, 4.0, 5.0, 14.0, 15.0],
        ),
        // Distinct extents everywhere, so a misrouted coordinate changes the
        // OUTPUT SHAPE and not merely the values. `infer_diagonal_result_type`
        // derives the declared type the same way, so a disagreement here is a
        // checked type the interpreter does not honor.
        case(
            "rank4-distinct-extents-reversed",
            vec![2, 3, 2, 5],
            2,
            0,
            vec![3, 2, 5],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 35.0, 36.0, 37.0, 38.0, 39.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 45.0, 46.0, 47.0, 48.0, 49.0, 20.0, 21.0, 22.0, 23.0, 24.0, 55.0, 56.0, 57.0,
                58.0, 59.0,
            ],
        ),
        case(
            "non-square-forward",
            vec![3, 2],
            0,
            1,
            vec![2],
            vec![0.0, 3.0],
        ),
        case(
            "non-square-reversed",
            vec![3, 2],
            1,
            0,
            vec![2],
            vec![0.0, 3.0],
        ),
        // A zero selected extent is deliberately absent here and present in
        // the C runtime's sibling file, because the two lanes disagree on it
        // and the host half is chelis#1347, not this repair. `tensor_numel`
        // reports one element for a zero-extent shape, so `linear_to_indices`
        // divides by that zero: `tensor_diagonal_value(shape [0, 3], 0, 1)`
        // panics "attempt to calculate the remainder with a divisor of zero"
        // at host_ops.rs, where `chelis_tensor_diagonal` returns the empty
        // result [05-OP-33] owes. Nothing in this test's own repair touches
        // that path.
    ];
    for probe in diagonal_cases {
        let label = probe.label;
        let output = tensor_diagonal_value(&ramp(probe.shape), probe.axis1, probe.axis2)
            .unwrap_or_else(|error| panic!("host diagonal `{label}` must evaluate: {error}"));
        assert_eq!(
            output.value.shape, probe.out_shape,
            "host diagonal `{label}` output shape"
        );
        assert_eq!(
            output.value.to_f64_lossy_vec(),
            probe.elements,
            "host diagonal `{label}` elements"
        );
    }

    let trace_cases = vec![
        case("rank2-forward", vec![2, 2], 0, 1, Vec::new(), vec![3.0]),
        case("rank2-reversed", vec![2, 2], 1, 0, Vec::new(), vec![3.0]),
        case(
            "rank3-leading-pair",
            vec![2, 2, 2],
            0,
            1,
            vec![2],
            vec![6.0, 8.0],
        ),
        case(
            "rank3-leading-pair-reversed",
            vec![2, 2, 2],
            1,
            0,
            vec![2],
            vec![6.0, 8.0],
        ),
        case(
            "rank3-trailing-pair",
            vec![2, 2, 2],
            1,
            2,
            vec![2],
            vec![3.0, 11.0],
        ),
        case(
            "rank3-straddling-forward",
            vec![2, 2, 2],
            0,
            2,
            vec![2],
            vec![5.0, 9.0],
        ),
        case(
            "rank3-straddling-reversed",
            vec![2, 2, 2],
            2,
            0,
            vec![2],
            vec![5.0, 9.0],
        ),
        case(
            "rank4-straddling-reversed",
            vec![2, 2, 2, 2],
            2,
            0,
            vec![2, 2],
            vec![10.0, 12.0, 18.0, 20.0],
        ),
        // `infer_trace_result_type` removes both source axes and declares
        // [3, 5]; reducing the diagonal's axis 0 instead would yield [2, 5].
        case(
            "rank4-distinct-extents-reversed",
            vec![2, 3, 2, 5],
            2,
            0,
            vec![3, 5],
            vec![
                35.0, 37.0, 39.0, 41.0, 43.0, 55.0, 57.0, 59.0, 61.0, 63.0, 75.0, 77.0, 79.0, 81.0,
                83.0,
            ],
        ),
    ];
    for probe in trace_cases {
        let label = probe.label;
        let output = tensor_trace_value(&ramp(probe.shape), probe.axis1, probe.axis2)
            .unwrap_or_else(|error| panic!("host trace `{label}` must evaluate: {error}"));
        assert_eq!(
            output.value.shape, probe.out_shape,
            "host trace `{label}` output shape"
        );
        assert_eq!(
            output.value.to_f64_lossy_vec(),
            probe.elements,
            "host trace `{label}` elements"
        );
    }
}

/// Negative parity for the case above: the axis domain still fails closed, and
/// an equal pair is rejected whichever spelling produces it.
#[test]
fn host_runtime_diagonal_and_trace_reject_equal_and_out_of_range_axes() {
    let operand = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![2, 2, 2], (0..8).map(|i| i as f64).collect()),
        precision: Prim::F32,
    };
    for (label, axis1, axis2) in [
        ("equal-axes", 1_i64, 1_i64),
        ("equal-axes-normalized", 2, -1),
        ("axis-out-of-range", 0, 3),
        ("negative-axis-out-of-range", 0, -4),
    ] {
        assert!(
            tensor_diagonal_value(&operand, axis1, axis2).is_err(),
            "host diagonal `{label}` must be rejected"
        );
        assert!(
            tensor_trace_value(&operand, axis1, axis2).is_err(),
            "host trace `{label}` must be rejected"
        );
    }
}

/// An empty operand's axis decomposition is never read, and computing it
/// anyway overflows `usize` on the prefix product or spins an empty loop. The
/// C runtime holds the same invariant in
/// `crates/chelis-runtime/tests/op33_empty_tensor_axis_decomposition.rs`; this
/// is the host lane's half. The extents are chosen so the prefix product is
/// `2^64` exactly.
#[test]
fn host_runtime_empty_operands_skip_their_axis_decomposition() {
    const BIG: usize = 1 << 32;
    let operand = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![BIG, BIG, 0], Vec::new()),
        precision: Prim::F32,
    };

    let scanned = tensor_cumsum_value(&operand, 2).expect("empty cumsum must evaluate");
    assert_eq!(scanned.value.shape, vec![BIG, BIG, 0]);
    assert_eq!(scanned.value.len(), 0);

    let sorted = tensor_sort_value(&operand, 2).expect("empty sort must evaluate");
    // Diagnostics name the expectation rather than Debug-printing the value:
    // this module is an observation exit surface, and
    // faithful_observation.md B2.4 admits no third formatter.
    match sorted {
        RuntimeValue::Tuple(items) => {
            assert_eq!(items.len(), 2, "sort must return two results");
            for item in items {
                match item {
                    RuntimeValue::Tensor(tensor) => assert_eq!(tensor.value.len(), 0),
                    _ => panic!("sort must return tensors"),
                }
            }
        }
        _ => panic!("sort must return a tuple"),
    }
}

/// A zero extent means zero elements, so a host operation over one owes an
/// empty result rather than a panic. `tensor_numel` used to clamp the count to
/// one, which handed every caller a phantom element: `linear_to_indices` then
/// divided by the zero extent, and the shorter paths built a `picks` vector one
/// longer than the storage its own shape declares.
///
/// The C runtime returned the empty result correctly throughout, so each case
/// below was also a lane divergence on a legal program (chelis#1347).
#[test]
fn host_runtime_zero_extent_operands_return_empty_results_rather_than_panicking() {
    fn empty(shape: Vec<usize>) -> RuntimeTensorValue {
        RuntimeTensorValue {
            value: IrTensorValue::from_vec(shape, Vec::new()),
            precision: Prim::F32,
        }
    }

    // diagonal: the case that surfaced this, at both zero positions.
    for (shape, axis1, axis2, expected) in [
        (vec![0_usize, 3], 0_i64, 1_i64, vec![0_usize]),
        (vec![3, 0], 0, 1, vec![0]),
        (vec![2, 0, 3], 0, 2, vec![2, 0]),
    ] {
        let out = tensor_diagonal_value(&empty(shape.clone()), axis1, axis2)
            .unwrap_or_else(|error| panic!("diagonal over a zero extent must evaluate: {error}"));
        assert_eq!(out.value.shape, expected);
        assert_eq!(out.value.len(), 0);
    }

    // trace reduces the diagonal, so it inherits the same path.
    let traced = tensor_trace_value(&empty(vec![2, 0, 3]), 0, 2)
        .unwrap_or_else(|error| panic!("trace over a zero extent must evaluate: {error}"));
    assert_eq!(traced.value.shape, vec![0_usize]);
    assert_eq!(traced.value.len(), 0);

    // The count short-circuits a zero rather than folding past it, so the
    // other extents never multiply. Exercised through the operation rather
    // than the private helper: `diagonal` over axes (0, 1) of these shapes
    // asks for the count of an output whose remaining extents would reach
    // 2^64 before reaching the trailing zero.
    const BIG: usize = 1 << 32;
    for shape in [
        vec![BIG, BIG, BIG, 0],
        vec![BIG, BIG, 0, BIG],
        vec![0, BIG, BIG, BIG],
    ] {
        let out = tensor_diagonal_value(&empty(shape.clone()), 0, 1)
            .unwrap_or_else(|error| panic!("diagonal over a huge empty shape: {error}"));
        assert_eq!(out.value.len(), 0);
        assert!(
            out.value.shape.contains(&0),
            "an empty operand owes an empty result"
        );
    }
}

#[test]
fn host_runtime_einsum_rejects_non_lowercase_labels() {
    let operand = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1], vec![1.0]),
        precision: Prim::F32,
    };
    for equation in ["I,I->", "_,i->"] {
        assert!(
            tensor_einsum_value(equation, &operand, &operand).is_err(),
            "host einsum accepted non-lowercase equation `{equation}`"
        );
    }
}

#[test]
fn host_runtime_einsum_rejects_duplicate_output_labels() {
    let operand = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1], vec![1.0]),
        precision: Prim::F32,
    };
    assert!(
        tensor_einsum_value("i,i->ii", &operand, &operand).is_err(),
        "host einsum accepted a duplicate output label"
    );
}

// PR #168 review LOW #5: NaN/Inf/n<4 edge-case coverage for the
// stride-4 ILP cascade. Tail handling (n < 4 where the lane-fill
// doesn't complete a full cycle) and special-value propagation
// are load-bearing invariants of the cascade; without these
// tests, a future change to the tail loop or lane combine could
// silently regress them.

/// Stride-4 reference for n < 4: the lanes are assigned naturally
/// (acc0=x[0], acc1=x[1], acc2=x[2]); the combine is
/// `(acc0 + acc1) + (acc2 + 0.0)` which equals a left-fold for
/// n ≤ 3. The expected bit pattern is therefore the straight-
/// forward sum.
#[test]
fn host_runtime_sum_f32_n1_n2_n3_bit_exact() {
    for (n, expr, expected) in [
        (1, "to_tensor([cast(1.5, f32)])", 1.5_f32),
        (2, "to_tensor([cast(1.5, f32), cast(0.25, f32)])", 1.75_f32),
        (
            3,
            "to_tensor([cast(1.5, f32), cast(0.25, f32), cast(0.125, f32)])",
            1.875_f32,
        ),
    ] {
        let src = format!(
            "seq = {expr}\n\
             y = sum(seq, cast(0, i32))\n"
        );
        let checked = checked_surf(&src);
        let outcome = evaluate_host_program(&checked, &UnordMap::new())
            .unwrap_or_else(|_| panic!("n={n} sum should evaluate"));
        let result = first_tensor_data(&outcome, "y");
        assert_eq!(result.len(), 1);
        assert_eq!(
            (result[0] as f32).to_bits(),
            expected.to_bits(),
            "n={n}: expected {expected}, got {}",
            result[0]
        );
    }
}

#[test]
fn host_runtime_sum_f32_propagates_nan() {
    // A single NaN anywhere in the input must propagate to the
    // final result. Pinned bit-exactly so a future change to the
    // lane combine that hides NaN through e.g. min/max can't slip
    // by.
    let checked = checked_surf(
        r#"
seq = to_tensor([
cast(1.0, f32),
cast(2.0, f32),
cast(0.0, f32) / cast(0.0, f32),
cast(4.0, f32),
cast(5.0, f32)
])
y = sum(seq, cast(0, i32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &UnordMap::new()).expect("nan-bearing sum should evaluate");
    let result = first_tensor_data(&outcome, "y");
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] as f32).is_nan(),
        "sum with NaN must propagate NaN; got {}",
        result[0]
    );
}

#[test]
fn host_runtime_sum_f32_inf_plus_neg_inf_is_nan() {
    // +Inf + -Inf is IEEE-754 NaN. The stride-4 cascade must
    // produce this regardless of which lanes the two infinities
    // land in (`x[0]` and `x[1]` here land in acc0/acc1; under
    // stride-4 the cascade still adds them and the result is NaN).
    let checked = checked_surf(
        r#"
seq = to_tensor([
cast(1.0, f32) / cast(0.0, f32),
cast(-1.0, f32) / cast(0.0, f32),
cast(2.0, f32),
cast(3.0, f32)
])
y = sum(seq, cast(0, i32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &UnordMap::new()).expect("inf-pair sum should evaluate");
    let result = first_tensor_data(&outcome, "y");
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] as f32).is_nan(),
        "sum with +Inf and -Inf must produce NaN; got {}",
        result[0]
    );
}

#[test]
fn host_runtime_matmul_shared_axis_mismatch_errors() {
    // Build a 2x3 and a 2x2 — shared axis is 3 vs 2, must fail.
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, i64), cast(0.0, f32))
b = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, i64), cast(0.0, f32))
y = matmul(a, b)
"#,
    );
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("matmul shared-axis mismatch must fail");
    assert!(
        err.contains(
            "matmul shared axis disagrees: lhs [2, 3] has 3 at axis 1, rhs [2, 2] has 2 at axis 0\n\
             numeric trap: domain in matmul at i64"
        ),
        "expected matmul's Domain trap (spec/04 section 4.7), got: {err}"
    );
}

// ----- expand / softmax host evaluator coverage (#38 follow-up) -----
//
// Pins the closure of the second host-runtime gap from the N2 fix:
// `chelis test` / `chelis eval` erroring with `unsupported builtin
// 'expand'` / `'softmax'` when those primitives appear in a test's
// dependency graph (School.Nn.Linear, School.Nn.Attention, School.Loss.CrossEntropy).

#[test]
fn host_runtime_expand_inserts_new_leading_axis() {
    // Linear.forward calls `expand(b, 0, batch)` where `b` is a 1-D
    // bias [out_dim] and the output is [batch, out_dim]. Pin that.
    let checked = checked_surf(
        r#"
b = to_tensor([cast(10.0, f32), cast(100.0, f32)])
y = insert(b, cast(0, i32), cast(3, i64))
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("insert should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3, 2]);
    // Three replicas of [10, 100].
    assert_eq!(
        first_tensor_data(&outcome, "y"),
        vec![10.0, 100.0, 10.0, 100.0, 10.0, 100.0]
    );
}

#[test]
fn host_runtime_expand_inserts_trailing_axis() {
    // axis == rank inserts a new last axis.
    let checked = checked_surf(
        r#"
b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
y = insert(b, cast(1, i32), cast(2, i64))
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("insert at trailing axis should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    // [1, 2] expanded along new last axis with count 2 -> [[1,1],[2,2]].
    assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn host_runtime_insert_singleton_input_adds_an_axis() {
    // Bucket 4a regression: `insert(b: tensor[1, f32], 0, count)` must
    // produce shape `[count, 1]`, matching the checker. Previously one
    // host function served both names and detected `in_shape[axis] == 1`
    // to replicate the singleton in-place, producing `[count]` and
    // diverging from `chelis check` on `examples/linreg.ch`. With one
    // result shape per operation the two names route to two host
    // functions and neither guesses (spec/04-type-system.md section
    // 4.7.2).
    let checked = checked_surf(
        r#"
b = to_tensor([cast(7.0, f32)])
y = insert(b, cast(0, i32), cast(4, i64))
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("insert([1], 0, 4) should evaluate under host runtime");
    assert_eq!(
        first_tensor_shape(&outcome, "y"),
        vec![4, 1],
        "INSERT semantics: rank-1 [1] insert at axis 0 with count 4 must produce rank-2 [4, 1]",
    );
    assert_eq!(first_tensor_data(&outcome, "y"), vec![7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn host_runtime_expand_negative_count_errors() {
    // chelis#1277 Slice A admits zero and continues to reject negative
    // runtime extents. Derive -1 from shape metadata so the checker cannot
    // fold it and this test reaches the host-runtime defense in depth.
    let checked = checked_surf(
        r#"
b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
negative_count = sub(shape(b, cast(0, i32)), cast(3, i64))
y = insert(b, cast(0, i32), negative_count)
"#,
    );
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("insert with negative count must fail");
    // chelis#1802: the trap renders as every lane renders it, the context
    // line and then [04-NUM-9]'s trap line.
    assert!(
        err.contains(
            "insert target extent at axis 0 is negative: -1\nnumeric trap: domain in insert at i64"
        ),
        "expected the Domain trap for a negative extent, got: {err}"
    );
}

// ----- to_tensor nested-list (Bucket 4b) -----

#[test]
fn host_runtime_to_tensor_accepts_2d_float_literal() {
    // Bucket 4b: previously rejected with
    // "to_tensor expects numeric or bool List elements, got List f32".
    let checked = checked_surf(
        r#"
y = to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]])
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("to_tensor of [[1,2],[3,4]] must evaluate to a rank-2 tensor");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn host_runtime_to_tensor_accepts_3d_float_literal() {
    // 2x2x2 cube — exercises 3-deep recursion in
    // `nested_list_to_tensor_data`.
    let checked = checked_surf(
        r#"
y = to_tensor([
  [[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]],
  [[cast(5.0, f32), cast(6.0, f32)], [cast(7.0, f32), cast(8.0, f32)]]
])
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("to_tensor of 2x2x2 nested list must evaluate to a rank-3 tensor");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2, 2]);
    assert_eq!(
        first_tensor_data(&outcome, "y"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn host_runtime_to_tensor_rejects_ragged_runtime_rows() {
    // Known ragged literals reject during checking. Hide row lengths behind
    // a callable to retain the separate host-runtime rejection obligation.
    let source = r#"
def row(short: bool) -> List[f32] = if short then [3.0f32] else [1.0f32, 2.0f32]
y = to_tensor([row(false), row(true)])
"#;
    let rectangular = checked_surf(&source.replace("row(true)", "row(false)"));
    let outcome = evaluate_host_program(&rectangular, &UnordMap::new())
        .expect("rectangular runtime rows must form a tensor");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 2.0, 1.0, 2.0]);
    let checked = checked_surf(source);
    let err = evaluate_host_program(&checked, &UnordMap::new())
        .expect_err("ragged nested list must fail to_tensor");
    assert!(
        err.contains(
            "to_tensor children disagree in shape: child 0 has [2], child 1 has [1]\n\
             numeric trap: domain in to_tensor at i64"
        ),
        "expected ragged-shape diagnostic, got: {err}"
    );
}

#[test]
fn host_runtime_softmax_uniform_input_is_uniform_output() {
    // softmax of all-zeros along axis 0 of length 3 is [1/3, 1/3, 1/3].
    let checked = checked_surf(
        r#"
x = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
y = softmax(x, cast(0, i32))
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("softmax should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3]);
    let data = first_tensor_data(&outcome, "y");
    for value in &data {
        // chelis#729 Phase 1: the f32-typed softmax result finalizes at
        // f32, so each element is exactly the f32 image of 1/3.
        assert_eq!(
            *value,
            (1.0f32 / 3.0f32) as f64,
            "uniform softmax element should be f32(1/3), got {value}"
        );
    }
}

#[test]
fn host_runtime_softmax_two_class_matches_reference() {
    // softmax([1.0, 0.0], 0) = [exp(1)/(exp(1)+1), 1/(exp(1)+1)]
    //                         ≈ [0.7310585, 0.2689414]
    let checked = checked_surf(
        r#"
x = to_tensor([cast(1.0, f32), cast(0.0, f32)])
y = softmax(x, cast(0, i32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &UnordMap::new()).expect("softmax 2-class should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2]);
    let data = first_tensor_data(&outcome, "y");
    let e = 1.0_f64.exp();
    let expected = [e / (e + 1.0), 1.0 / (e + 1.0)];
    for (got, want) in data.iter().zip(expected.iter()) {
        assert!(
            (*got - *want).abs() < 1e-6,
            "softmax 2-class mismatch: got {got}, want {want}"
        );
    }
}

#[test]
fn host_runtime_softmax_numerical_stability_handles_large_inputs() {
    // Without the max-subtraction trick, exp(1000) would overflow to
    // inf and produce NaN. The stable lowering must still produce
    // a normalized distribution.
    let checked = checked_surf(
        r#"
x = to_tensor([cast(1000.0, f32), cast(1000.0, f32)])
y = softmax(x, cast(0, i32))
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("softmax with large inputs should remain numerically stable");
    let data = first_tensor_data(&outcome, "y");
    assert_eq!(data.len(), 2);
    for value in &data {
        assert!(
            (*value - 0.5).abs() < 1e-9,
            "softmax([1000,1000]) should be uniform 0.5, got {value}"
        );
    }
}

#[test]
fn host_runtime_softmax_positive_infinity_yields_nan_like_torch() {
    // #173: `softmax([+Inf, 0, 0])` must return all-NaN, matching torch
    // (verified: torch 2.x CPU `torch.softmax([inf,1,2]) == [nan,nan,nan]`).
    // The standard max-shift formula computes `exp(+Inf - +Inf) = exp(NaN)
    // = NaN`, which is what the C backend, IR evaluator, and spec'd
    // lowering all already produce. The host runtime previously
    // special-cased +Inf to a `1/K` one-hot ("natural limit"), silently
    // diverging from torch and every other Chelis lane; that special-case
    // is removed.
    let inf = f64::INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![inf, 0.0, 0.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("+Inf softmax must not error in host runtime");
    assert_eq!(out.value.len(), 3);
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "softmax of a slice containing +Inf must be all-NaN (torch parity, #173); got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_two_positive_infinities_yield_nan_like_torch() {
    // #173: torch returns all-NaN for any +Inf-containing slice, including
    // multiple +Inf (`torch.softmax([inf,inf,2]) == [nan,nan,nan]`). The
    // prior `1/K`-split special-case is removed.
    let inf = f64::INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![inf, inf, 0.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("two +Inf softmax must not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "softmax of a slice with multiple +Inf must be all-NaN (torch parity, #173); got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_all_negative_infinity_yields_nan_like_torch() {
    // #173: an all-`-Inf` slice yields `exp(-Inf - -Inf) = exp(NaN) = NaN`
    // under the standard formula, which is what torch returns
    // (`torch.softmax([-inf,-inf,-inf]) == [nan,nan,nan]`) and what the C
    // backend / IR evaluator already produce. The host runtime previously
    // special-cased this to uniform `1/N`; that special-case is removed.
    let neg_inf = f64::NEG_INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![neg_inf, neg_inf, neg_inf]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("all -Inf softmax must not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "softmax of an all-(-Inf) slice must be all-NaN (torch parity, #173); got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_mixed_negative_infinity_is_finite_like_torch() {
    // #173 positive control: a slice with `-Inf` but NO `+Inf` has a
    // finite max, so `exp(-Inf - max) = 0` at the masked position and the
    // standard formula produces a valid (non-NaN) distribution. This is
    // the legitimate attention-masking case and must NOT regress to NaN.
    // torch: `torch.softmax([1, -inf, 2]) == [0.2689, 0.0, 0.7311]`.
    let neg_inf = f64::NEG_INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![1.0, neg_inf, 2.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("mixed -Inf softmax must not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| !v.is_nan()),
        "mixed -Inf (no +Inf) softmax must be finite, not NaN; got {:?}",
        out.value.to_f64_lossy_vec()
    );
    assert!(
        out.value.to_f64_lossy_vec()[1].abs() < 1e-9,
        "the -Inf position must be 0"
    );
    assert!(
        (out.value.to_f64_lossy_vec()[0] - 0.268_941_4).abs() < 1e-5
            && (out.value.to_f64_lossy_vec()[2] - 0.731_058_6).abs() < 1e-5,
        "mixed -Inf softmax must match torch [0.2689, 0, 0.7311]; got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_nan_input_propagates_nan() {
    // NaN is contagious by spec; matches PyTorch / NumPy / JAX behavior.
    // We pin this so a future refactor doesn't accidentally mask it.
    let nan = f64::NAN;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![nan, 0.0, 0.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("NaN softmax does not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "NaN must propagate to every output element; got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_axis_out_of_bounds_errors() {
    // Issue #216: cast-wrapped out-of-bounds softmax axis is now
    // caught at infer time (the checker peels the `cast(N, i32)`
    // wrapper via `extract_int_for_dim` and applies the rank-bounds
    // check). Pre-fix the cast hid the literal from
    // `extract_int_literal` and the rejection only fired in the
    // host-runtime defense-in-depth layer. The user-facing contract
    // is unchanged (the program is still rejected); only the layer
    // emitting the diagnostic moved upstream.
    let src = r#"
x = to_tensor([cast(1.0, f32), cast(2.0, f32)])
y = softmax(x, cast(5, i32))
"#;
    let res = chelis_types::check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(src).expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let infer_err = res.expect_err("softmax with out-of-bounds axis must fail infer-time check");
    let joined = infer_err
        .errors
        .iter()
        .map(|e| e.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("softmax") && joined.contains("out of bounds"),
        "expected softmax axis-bounds diagnostic, got: {joined}"
    );
}

// ----------------------------------------------------------------
// #172: Max/Min reductions PROPAGATE NaN, matching torch
// (`torch.max`/`torch.min` of any NaN-containing slice return NaN, at
// every position). A naive `value > best` drops NaN, which made the C
// backend's SIMD `chelis_max_f32` position-dependent and diverged from
// torch on both lanes. The eval lane is the reference, so it must agree.
// ----------------------------------------------------------------

#[test]
fn host_runtime_max_reduce_propagates_nan_like_torch() {
    let nan = f64::NAN;
    // NaN at each position must still yield NaN (position-independent).
    for pos in 0..3 {
        let mut data = vec![1.0, 2.0, 3.0];
        data[pos] = nan;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], data),
            precision: Prim::F32,
        };
        let out = tensor_reduce_host(&tensor, 0, ReduceOp::Max).expect("max reduce");
        assert!(
            out.value.to_f64_lossy_vec()[0].is_nan(),
            "max_reduce of a slice with NaN@{pos} must be NaN (torch parity, #172); got {:?}",
            out.value.to_f64_lossy_vec()
        );
    }
}

#[test]
fn host_runtime_min_reduce_propagates_nan_like_torch() {
    let nan = f64::NAN;
    for pos in 0..3 {
        let mut data = vec![1.0, 2.0, 3.0];
        data[pos] = nan;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], data),
            precision: Prim::F32,
        };
        let out = tensor_reduce_host(&tensor, 0, ReduceOp::Min).expect("min reduce");
        assert!(
            out.value.to_f64_lossy_vec()[0].is_nan(),
            "min_reduce of a slice with NaN@{pos} must be NaN (torch parity, #172); got {:?}",
            out.value.to_f64_lossy_vec()
        );
    }
}

#[test]
fn host_runtime_max_reduce_no_nan_is_unchanged() {
    // POSITIVE control: a NaN-free slice still reduces normally — the
    // NaN-propagation fix must not perturb ordinary max/min.
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![1.0, 3.0, 2.0]),
        precision: Prim::F32,
    };
    let max = tensor_reduce_host(&tensor, 0, ReduceOp::Max).expect("max reduce");
    let min = tensor_reduce_host(&tensor, 0, ReduceOp::Min).expect("min reduce");
    assert_eq!(max.value.to_f64_lossy_vec(), vec![3.0]);
    assert_eq!(min.value.to_f64_lossy_vec(), vec![1.0]);
}

// ----------------------------------------------------------------
// WS-A0 acceptance test (c): the dtype-tagged Scalar payload must
// reject mismatched (dtype, bits) pairs at construction time. The
// public constructor `RuntimeValue::scalar` is the only checked
// path; the typed convenience constructors (`int_lit`, `float_lit`,
// `i64`, `float64`, `scalar_like_*`) are implementation-internal
// and statically-correct by construction.
// ----------------------------------------------------------------

#[test]
fn scalar_construction_is_sealed_and_width_checked() {
    // chelis#729 Phase 1 re-authoring: the old dtype/bits mismatch this
    // test enumerated is now UNREPRESENTABLE - the sealed ScalarValue's
    // storage variant IS the dtype, and the only constructors are the
    // dtype-semantics module's finalize/ingress chokepoints. What remains
    // testable at this level is the width discipline those constructors
    // enforce: out-of-width integers trap instead of wrapping, and every
    // constructed scalar reports the dtype it was finalized at.
    let trapped = RuntimeValue::scalar_like_int(Prim::Int8, 200)
        .expect_err("i8 cannot hold 200; finalize must trap, not wrap");
    assert!(
        trapped.contains("overflow"),
        "the trap must carry the branded overflow diagnostic, got: {trapped}"
    );
    let ok = RuntimeValue::scalar_like_int(Prim::Int8, 127).expect("127 fits i8");
    match ok {
        RuntimeValue::Scalar(payload) => assert_eq!(payload.dtype(), Prim::Int8),
        other => panic!("expected RuntimeValue::Scalar, got {other:?}"),
    }
}

// ----------------------------------------------------------------
// RT-1 finding (C1), history: the original RT-1 test exercised
// `RuntimeValue::Scalar { dtype: F16, bits: ScalarBits::F32(_) }`
// directly, then C1 sealed the payload so the literal no longer
// compiled, and chelis#729 Phase 1 removed the (dtype, bits) pair
// entirely - the storage variant IS the dtype.
// ----------------------------------------------------------------

/// RT-1 closed finding C1, strengthened by chelis#729 Phase 1: the
/// dtype/bits mismatch the pre-fix red team constructed is now
/// UNREPRESENTABLE, not merely rejected. `RuntimeValue::Scalar` wraps the
/// sealed `chelis_types::ScalarValue`, whose storage variant IS the
/// dtype; there is no (dtype, bits) pair to disagree, and the only
/// constructors are the dtype-semantics module's finalize/ingress
/// chokepoints. This pin documents the strengthening and asserts the
/// module constructor reports the dtype it stored at.
#[test]
fn scalar_payload_dtype_is_the_storage_variant() {
    let value =
        chelis_types::scalar_from_f64("test", Prim::F16, 1.5).expect("1.5 finalizes at f16");
    assert_eq!(value.prim(), Prim::F16);
    let payload = match RuntimeValue::from_scalar_value(value) {
        RuntimeValue::Scalar(payload) => payload,
        other => panic!("expected RuntimeValue::Scalar, got {other:?}"),
    };
    assert_eq!(payload.dtype(), Prim::F16);
    assert_eq!(payload.as_f64_lossy(), 1.5);
}

/// chelis#2413 (B4): a key element, such as the one element of a rank-0 key
/// tensor a staged host plan captures as a scalar, becomes the key variant,
/// never a numeric scalar, so a draw reads it as its key and every
/// diagnostic renders it. `split(key_from_seed(7))`'s left key is
/// `aa3896172f9a3213` in `key_ref.py`.
///
/// Evidentiary status: REGRESSION TEST. At `b005bb19b` the constructor
/// returned a `Scalar` holding the key and describing it panicked in
/// `element_ref` ("a random key has no observation form").
#[test]
fn a_key_element_is_the_key_variant_and_describes_without_panicking() {
    let seed = chelis_types::scalar_from_i64("test", Prim::Int64, 7).expect("7 is an i64");
    let (key, _) = chelis_types::RandomKey::from_seed(seed)
        .expect("every i64 seeds a key")
        .split();
    let value = RuntimeValue::from_scalar_value(chelis_types::ScalarValue::from_key(key));
    assert!(matches!(value, RuntimeValue::Key(inner) if inner == key));
    assert_eq!(describe_value(&value), "key(aa3896172f9a3213)");
    assert_eq!(describe_argument(Some(&value)), "key(aa3896172f9a3213)");
}

fn numeric_scalar(prim: Prim, integer: i64, float: f64) -> RuntimeValue {
    if prim.is_integer() {
        RuntimeValue::from_scalar_value(
            chelis_types::scalar_from_i64("test", prim, integer)
                .expect("test integer fits its dtype"),
        )
    } else {
        RuntimeValue::from_scalar_value(
            chelis_types::scalar_from_f64("test", prim, float)
                .expect("float finalization is total"),
        )
    }
}

/// chelis#729 Phase 1, dtype-semantics C3: scalar values nested in the
/// execution wire are dtype-tagged carriers too. Lists, tuples, and ADTs may
/// not silently widen all integer values to i64 or all floats to f64.
#[test]
fn execution_wire_nested_numeric_scalars_keep_their_dtype_tags() {
    let cases = [
        (Prim::Int8, "int8"),
        (Prim::Int16, "int16"),
        (Prim::Int32, "int32"),
        (Prim::Int64, "int64"),
        (Prim::F16, "f16"),
        (Prim::Bf16, "bf16"),
        (Prim::F32, "f32"),
        (Prim::F64, "f64"),
    ];

    for (prim, expected_tag) in cases {
        let scalar = numeric_scalar(prim, 7, 1.5);
        let containers = [
            RuntimeValue::List(vec![scalar.clone()].into()),
            RuntimeValue::Tuple(vec![scalar.clone()].into()),
            RuntimeValue::Dict(
                vec![(RuntimeValue::String("value".to_string()), scalar.clone())].into(),
            ),
            RuntimeValue::Adt {
                ctor: "Boxed".to_string(),
                source_name: "Boxed".to_string(),
                fields: vec![scalar].into(),
                field_names: Some(vec!["value".to_string()]),
            },
        ];

        for container in containers {
            let encoded = runtime_value_to_schema(&container).expect("wire encode");
            let json = serde_json::to_value(encoded).expect("serialize execution value");
            let nested = json
                .get("value")
                .and_then(|value| value.as_array())
                .and_then(|values| values.first())
                .or_else(|| {
                    json.get("fields")
                        .and_then(|value| value.as_array())
                        .and_then(|values| values.first())
                })
                .or_else(|| {
                    json.get("entries")
                        .and_then(|value| value.as_array())
                        .and_then(|entries| entries.first())
                        .and_then(|entry| entry.get("value"))
                })
                .expect("container has one nested scalar");
            assert_eq!(
                nested
                    .get("value")
                    .and_then(|value| value.get("dtype"))
                    .and_then(|value| value.as_str()),
                Some(expected_tag),
                "nested numeric scalar must keep its own wire tag: {json}"
            );
        }
    }
}

/// Positive width matrix for the list bridges covered by Phase 1 eval
/// adoption. The checker derives the exact element dtype, so runtime
/// construction must preserve it for `to_tensor`, `pad_sequences`, and
/// `pad_sequences_to`.
#[test]
fn list_tensor_bridges_preserve_every_numeric_dtype() {
    let dtypes = [
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::F16,
        Prim::Bf16,
        Prim::F32,
        Prim::F64,
    ];

    for prim in dtypes {
        let value = numeric_scalar(prim, 7, if prim == Prim::F64 { 1e100 } else { 1.5 });
        // Float elements move into the buffer as the stored scalar itself.
        let expected_float = match &value {
            RuntimeValue::Scalar(payload) => payload.value(),
            _ => panic!("numeric_scalar must build a numeric scalar"),
        };
        let (_, tensor_data) =
            nested_list_to_tensor_data(std::slice::from_ref(&value), prim, &[Some(1)])
                .expect("to_tensor list ingress");
        match tensor_data {
            ListTensorData::Int(values) => assert_eq!(values, vec![7]),
            ListTensorData::Float(values) => assert_eq!(values, vec![expected_float]),
        }

        let sequences = [RuntimeValue::List(vec![value.clone()].into())];
        let (padded_prim, padded_data, _, _) =
            pad_sequences_value(&sequences, &value).expect("pad_sequences ingress");
        assert_eq!(
            padded_prim, prim,
            "pad_sequences must preserve the input dtype"
        );
        match padded_data {
            ListTensorData::Int(values) => assert_eq!(values, vec![7]),
            ListTensorData::Float(values) => assert_eq!(values, vec![expected_float]),
        }

        let (padded_to_prim, padded_to_data, _) =
            pad_sequences_to_value(&sequences, 2, &value).expect("pad_sequences_to ingress");
        assert_eq!(
            padded_to_prim, prim,
            "pad_sequences_to must preserve the input dtype"
        );
        match padded_to_data {
            ListTensorData::Int(values) => assert_eq!(values, vec![7, 7]),
            ListTensorData::Float(values) => {
                assert_eq!(values, vec![expected_float, expected_float])
            }
        }
    }
}

/// Negative-test parity for the exact list bridge: same-family dtypes are
/// still heterogeneous. Accepting i8 beside i16 or f64 beside f32 would
/// perform an implicit cast that the checker never authorized.
#[test]
fn list_tensor_bridges_reject_same_family_dtype_substitution() {
    let int8 = numeric_scalar(Prim::Int8, 7, 0.0);
    let int16 = numeric_scalar(Prim::Int16, 7, 0.0);
    assert!(
        nested_list_to_tensor_data(&[int8.clone(), int16.clone()], Prim::Int8, &[Some(2)]).is_err(),
        "to_tensor must reject heterogeneous integer widths"
    );
    assert!(
        pad_sequences_value(&[RuntimeValue::List(vec![int8].into())], &int16).is_err(),
        "pad_sequences must reject a different integer pad dtype"
    );

    let f64_value = numeric_scalar(Prim::F64, 0, 1e100);
    let f32_pad = numeric_scalar(Prim::F32, 0, 0.0);
    assert!(
        pad_sequences_to_value(&[RuntimeValue::List(vec![f64_value].into())], 2, &f32_pad).is_err(),
        "pad_sequences_to must reject a different float pad dtype"
    );
}

/// E2 (WS-A0 RT-1 fixup): `prim_from_name` must panic on the
/// `"f8e4m3"` token with the §1.1.1 message rather than producing
/// `Some(Prim::F8e4m3)` and letting the deferred dtype flow into
/// runtime classification.
#[test]
#[should_panic(expected = "f8e4m3 is deferred per spec/04-type-system.md §1.1.1")]
fn prim_from_name_panics_on_f8e4m3_per_spec_1_1_1() {
    let _ = prim_from_name("f8e4m3");
}

/// Negative-parity twin: the active dtype names still resolve.
#[test]
fn prim_from_name_resolves_active_dtype_names() {
    for (name, expected) in &[
        ("f32", Prim::F32),
        ("f64", Prim::F64),
        ("f16", Prim::F16),
        ("bf16", Prim::Bf16),
        ("i8", Prim::Int8),
        ("i16", Prim::Int16),
        ("i32", Prim::Int32),
        ("i64", Prim::Int64),
        ("bool", Prim::Bool),
    ] {
        assert_eq!(
            prim_from_name(name),
            Some(*expected),
            "active dtype name `{name}` must still resolve"
        );
    }
}

/// E1 (WS-A0 RT-1 fixup): the old float-binop fallback silently
/// re-precisioned mixed narrow floats to F32. The type checker must reject
/// `(Bf16, F16)` and `(F16, Bf16)` per spec/04-type-system.md §5.1, while
/// the closed dtype kernel is the runtime backstop.
#[test]
fn type_checker_rejects_mixed_narrow_float_binop_per_spec_5_1() {
    let src = r#"
def main() -> bf16 = add(cast(1.0, bf16), cast(1.0, f16))
"#;
    let res = chelis_types::check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(src).expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    assert!(
        res.is_err(),
        "spec §5.1 forbids implicit precision promotion; \
         `add(_:bf16, _:f16)` must be rejected by the type checker"
    );
}

#[test]
fn scalar_constructor_accepts_every_active_numeric_dtype() {
    // Sanity sibling (re-authored for chelis#729 Phase 1): every active
    // numeric dtype constructs through the module's ingress constructors
    // and reports itself back from the sealed storage.
    for (dtype, value) in [
        (Prim::Int8, 7.0),
        (Prim::Int16, 123.0),
        (Prim::Int32, -42.0),
        (Prim::Int64, 1_000_000.0),
        (Prim::F32, 1.5),
        (Prim::F64, 2.5),
        (Prim::F16, 0.25),
        (Prim::Bf16, 0.25),
    ] {
        let v = RuntimeValue::scalar_like_float(dtype, value)
            .unwrap_or_else(|e| panic!("{} must construct: {e}", dtype.name()));
        match v {
            RuntimeValue::Scalar(payload) => {
                assert_eq!(payload.dtype(), dtype);
                assert_eq!(payload.as_f64_lossy(), value);
            }
            other => panic!("expected RuntimeValue::Scalar, got {other:?}"),
        }
    }
}

#[test]
fn integer_spelled_f32_scalar_finalizes_without_f64_intermediate() {
    let checked = checked_surf(
        r#"
def probe() -> f32 = sub(
  cast(18014399583223809, f32),
  cast(18014398509481984, f32)
)
root = probe()
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new())
        .expect("integer-spelled f32 scalar must evaluate");
    let root = outcome.host_bindings.get("root").expect("root binding");
    let RuntimeValue::Scalar(payload) = root else {
        panic!("expected scalar root");
    };
    assert_eq!(payload.dtype(), Prim::F32);
    assert_eq!(
        (payload.as_f64_lossy() as f32).to_bits(),
        0x4F00_0000,
        "single rounding keeps the low-side operand one f32 ulp above the control; double rounding produces zero"
    );
}

// ===========================================================================
// chelis#732 Phase 1: render_value's [05-OBS] behavior at the unit level.
// The end-to-end exits are locked by the CLI observation harness; these pin
// the renderer's edge policy directly.
// ===========================================================================

fn tensor_value(precision: Prim, shape: Vec<usize>, data: Vec<f64>) -> RuntimeValue {
    RuntimeValue::Tensor(
        RuntimeTensorValue::from_wide("test", precision, shape, data)
            .expect("test fixtures carry in-domain values"),
    )
}

/// Integer/bool tensor elements render per their tag's class
/// ([05-OBS-2]): integers lose the `.0`, bools print true/false.
#[test]
fn render_value_tensor_elements_follow_tag_class() {
    assert_eq!(
        render_value(&tensor_value(Prim::Int8, vec![3], vec![127.0, -127.0, 0.0])),
        "tensor(shape=[3], data=[127, -127, 0])"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![2], vec![1.0, 0.0])),
        "tensor(shape=[2], data=[true, false])"
    );
    // Float elements render at the stored f64 width (the chelis#717 width
    // note, spec/05 section 8.1).
    assert_eq!(
        render_value(&tensor_value(
            Prim::F64,
            vec![2],
            vec![0.30000000000000004, -0.0]
        )),
        "tensor(shape=[2], data=[0.30000000000000004, -0.0])"
    );
}

/// Tag-vs-bits disagreements print the BITS (spec/05 section 8.1): an
/// i64-tagged slot holding 187.5 (the live `mean`-of-i64 example,
/// chelis#724 territory) renders 187.5, never a truncated 187; a
/// bool-tagged slot holding 2.0 renders 2.0, never `true`; an
/// i8-tagged slot holding 400 renders 400.0's bits faithfully rather
/// than a wrapped/saturated lie.
#[test]
fn tag_storage_disagreement_is_unconstructible() {
    // chelis#729 Phase 1 re-authoring: this test used to pin FAITHFUL
    // MISREPORTING - an integer/bool-tagged buffer holding out-of-domain
    // f64 bits rendered the stored bits. Per-dtype sealed storage makes
    // that state unrepresentable: the construction chokepoint traps at
    // finalize instead, so nothing out-of-domain can reach the renderer.
    let trapped = RuntimeTensorValue::from_wide("test", Prim::Int64, vec![1], vec![187.5])
        .expect_err("a fractional value in an i64 buffer must Domain-trap");
    assert!(
        trapped.contains("domain"),
        "branded domain trap, got: {trapped}"
    );
    let trapped = RuntimeTensorValue::from_wide("test", Prim::Bool, vec![1], vec![2.0])
        .expect_err("2.0 in a bool buffer must Domain-trap");
    assert!(
        trapped.contains("domain"),
        "branded domain trap, got: {trapped}"
    );
    let trapped = RuntimeTensorValue::from_wide("test", Prim::Int8, vec![1], vec![400.0])
        .expect_err("400 in an i8 buffer must Overflow-trap");
    assert!(
        trapped.contains("overflow"),
        "branded overflow trap, got: {trapped}"
    );
}

/// [05-OBS-4]: a rank-0 tensor renders as its single element, bare, at
/// the renderer level - the wrapper is not an exit form.
#[test]
fn render_value_rank_zero_tensor_renders_bare() {
    assert_eq!(
        render_value(&tensor_value(Prim::F64, vec![], vec![0.1])),
        "0.1"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Int32, vec![], vec![7.0])),
        "7"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![], vec![1.0])),
        "true"
    );
}

/// [05-OBS-5]: tensor renders truncate after 32 elements with the
/// `, ...` marker; a 32-element tensor renders in full.
#[test]
fn render_value_truncates_at_32_with_marker() {
    let data: Vec<f64> = (1..=33).map(f64::from).collect();
    let rendered = render_value(&tensor_value(Prim::F64, vec![33], data));
    let visible: Vec<String> = (1..=32).map(|i| format!("{i}.0")).collect();
    assert_eq!(
        rendered,
        format!("tensor(shape=[33], data=[{}, ...])", visible.join(", "))
    );

    let full: Vec<f64> = (1..=32).map(f64::from).collect();
    let rendered = render_value(&tensor_value(Prim::F64, vec![32], full));
    assert!(
        !rendered.contains("..."),
        "a 32-element tensor renders in full: {rendered}"
    );
}

/// Scalar payloads render at their OWN width ([05-OBS-2]): the f32
/// scalar prints its shortest-at-f32 digits, not the f64 image.
#[test]
fn render_value_scalars_render_at_own_width() {
    let f32_scalar = RuntimeValue::scalar_like_float(Prim::F32, 0.1f32 as f64).expect("scalar");
    assert_eq!(render_value(&f32_scalar), "0.1");
    let f16_scalar = RuntimeValue::scalar_like_float(Prim::F16, 2048.0).expect("scalar");
    assert_eq!(render_value(&f16_scalar), "2048.0");
    let i64_scalar = RuntimeValue::scalar_like_int(Prim::Int64, 9007199254740993).expect("scalar");
    assert_eq!(render_value(&i64_scalar), "9007199254740993");
    let f64_scalar = RuntimeValue::scalar_like_float(Prim::F64, f64::MAX).expect("scalar");
    assert_eq!(render_value(&f64_scalar), "1.7976931348623157e308");
}

// ===========================================================================
// RT792 probes (PR #792 fresh-context red team, adopted): the tag-vs-bits
// arithmetic edges of render_tensor_element, locked at the boundaries.
// ===========================================================================

/// int_or_bits boundary sweep at +/-2^63 (red-team authored): stored
/// exactly -2^63 sits inside the exact-i64 guard and prints the integer;
/// stored exactly +2^63 (where f64 has no i64 twin) is outside and prints
/// the faithful f64 bits, never a saturated integer; the largest f64
/// below 2^63 prints exactly. Non-integral / non-finite stored values
/// under an integer tag print the f64 bits.
#[test]
fn rt792_render_value_int64_tag_pow63_boundaries() {
    // chelis#729 Phase 1 re-authoring: in-range integral values store and
    // render exactly; the out-of-range and fractional rows that used to
    // render as faithful f64 bits now trap at the construction chokepoint
    // (see tag_storage_disagreement_is_unconstructible).
    assert_eq!(
        render_value(&tensor_value(
            Prim::Int64,
            vec![1],
            vec![-9223372036854775808.0]
        )),
        "tensor(shape=[1], data=[-9223372036854775808])"
    );
    assert_eq!(
        render_value(&tensor_value(
            Prim::Int64,
            vec![1],
            vec![9223372036854774784.0]
        )),
        "tensor(shape=[1], data=[9223372036854774784])"
    );
}

/// A bool-tagged slot storing -0.0 prints the BITS (`-0.0`), never
/// `false`: the sign bit is stored state a boolean rendering would
/// launder, and the slot is source-reachable via
/// `print(neg(to_tensor([false, true])))`. Red-team finding F6, fixed;
/// this is the tightened lock (the red team's original test pinned the
/// laundering behavior as evidence). A bool-tagged 2.0 / 0.5 prints the
/// bits per the tag-vs-bits rule; +0.0 stays `false`; an int-tagged -0.0
/// prints `0` (integers have no signed zero - the sign bit there is an
/// f64-backing-store artifact, not integer state).
#[test]
fn rt792_render_value_bool_tag_negative_zero_is_false() {
    // chelis#729 Phase 1 re-authoring: -0.0 equals 0, a member of bool's
    // {0, 1} set, so it finalizes to false (the old faithful-misreport
    // "-0.0" row required a tag/storage disagreement that is now
    // unrepresentable; non-members like 2.0 and 0.5 trap at the
    // chokepoint instead - see tag_storage_disagreement_is_unconstructible).
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![1], vec![-0.0])),
        "tensor(shape=[1], data=[false])"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![1], vec![0.0])),
        "tensor(shape=[1], data=[false])"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Int32, vec![1], vec![-0.0])),
        "tensor(shape=[1], data=[0])"
    );
}

// ---------------------------------------------------------------------------
/// [05-OP-1]'s f32 lane: an f32 operand rounds at its OWN width and the
/// result stays f32 -- checker and eval agree on the dtype, and the value
/// is the correctly-rounded decimal rounding of the f32's exact binary
/// value. 2.675f32 is exactly 2.67499995231628417968750, so 2 places
/// rounds DOWN to 2.67 (then stored as the nearest f32).
#[test]
fn round_to_f32_lane_preserves_dtype_and_rounds_at_own_width() {
    let checked = checked_surf(
        r#"
x = round_to(2.675f32, 2)
y = round_to(2.675f64, 2)
"#,
    );
    let outcome = evaluate_host_program(&checked, &UnordMap::new()).expect("should evaluate");
    match outcome.host_bindings.get("x") {
        Some(RuntimeValue::Scalar(payload)) => {
            assert_eq!(payload.dtype(), Prim::F32, "f32 in, f32 out");
            assert_eq!(payload.as_f64_lossy(), f64::from(2.67f32));
        }
        other => panic!("expected scalar, got {other:?}"),
    }
    match outcome.host_bindings.get("y") {
        Some(RuntimeValue::Scalar(payload)) => {
            assert_eq!(payload.dtype(), Prim::F64, "f64 in, f64 out");
            assert_eq!(payload.as_f64_lossy(), 2.67);
        }
        other => panic!("expected scalar, got {other:?}"),
    }
}

// ===========================================================================
// FO-DIAG: the diagnostic-rendering boundary (chelis#997,
// spec/design/faithful_observation.md §C1.6 / §B2.4).
//
// §C1.6: "a diagnostic must not launder what it reports." These pin the
// boundary beside `render_value` -- `describe_value`, `describe_argument`,
// `describe_fields` -- as the ONE way a runtime, JSON, or CSV payload
// becomes diagnostic text, and pin the negative half: the derived-`Debug`
// spellings must not reappear in any of it.
//
// Focused run:
//   cargo nextest run -p chelis-compiler-api --lib -E 'test(fo_diag)'
// ===========================================================================

/// Derived-`Debug` fingerprints. A diagnostic containing any of these is
/// rendering the Rust representation instead of the value.
const DEBUG_SPELLINGS: &[&str] = &[
    "Scalar(",
    "ScalarPayload",
    "ScalarValue",
    "bits:",
    "Bits::",
    "F16(",
    "Bf16(",
    "F32(",
    "F64(",
    "I32(",
    "I64(",
    "RuntimeValue::",
    "field_names",
    "Adt {",
    "IrTensorValue",
];

/// The measured pre-migration renderings of a stored f16/bf16 `0.1`:
/// `half`'s `Debug` forwards to `to_f32()`, so it reported `0.099975586`
/// and `0.100097656` for values every exit renders as `0.1`. These are
/// PREFIXES of those strings, so a `half` version bump that shifts the
/// trailing digits still trips the assertions, while the correct `0.1`
/// rendering can never contain either one.
const WIDENED_F16_IMAGE: &str = "0.0999755";
const WIDENED_BF16_IMAGE: &str = "0.1000976";

#[track_caller]
fn assert_no_debug_spelling(rendered: &str, what: &str) {
    for needle in DEBUG_SPELLINGS {
        assert!(
            !rendered.contains(needle),
            "{what} leaked the derived-Debug spelling `{needle}`: {rendered}"
        );
    }
}

fn scalar_of(dtype: Prim, value: f64) -> RuntimeValue {
    RuntimeValue::scalar_like_float(dtype, value).expect("in-domain scalar fixture")
}

fn int_scalar_of(dtype: Prim, value: i64) -> RuntimeValue {
    RuntimeValue::scalar_like_int(dtype, value).expect("in-domain scalar fixture")
}

fn adt(ctor: &str, fields: Vec<RuntimeValue>) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: ctor.to_string(),
        source_name: ctor.to_string(),
        fields: fields.into(),
        field_names: None,
    }
}

/// A scalar renders as `<dtype> <canonical digits>`: the dtype comes from
/// the sealed storage variant, and the digits are `format_element`'s, so a
/// diagnostic and an exit channel agree on every payload.
#[test]
fn fo_diag_scalars_carry_their_dtype_and_own_width_digits() {
    // Exact i64 above 2^53 -- the [#723] case. 9007199254740993 has no
    // f64 twin, so a rendering that funnels through double reports
    // ...992.
    let big = int_scalar_of(Prim::Int64, 9_007_199_254_740_993);
    assert_eq!(describe_value(&big), "i64 9007199254740993");
    assert!(
        !describe_value(&big).contains("9007199254740992"),
        "the f64 image must not reach the diagnostic channel"
    );

    assert_eq!(describe_value(&int_scalar_of(Prim::Int32, -5)), "i32 -5");
    assert_eq!(describe_value(&int_scalar_of(Prim::Int8, 127)), "i8 127");

    // Own-width floats: the f32 prints its shortest-at-f32 digits, and the
    // f64 holding the same f32's image prints the wider, different string.
    assert_eq!(
        describe_value(&scalar_of(Prim::F32, 0.1f32 as f64)),
        "f32 0.1"
    );
    assert_eq!(
        describe_value(&scalar_of(Prim::F64, 0.1f32 as f64)),
        "f64 0.10000000149011612"
    );
    assert_eq!(
        describe_value(&scalar_of(Prim::F64, f64::MAX)),
        "f64 1.7976931348623157e308"
    );
    assert_eq!(describe_value(&scalar_of(Prim::F64, -0.0)), "f64 -0.0");

    for value in [&big, &scalar_of(Prim::F32, 0.1f32 as f64)] {
        assert_no_debug_spelling(&describe_value(value), "a scalar diagnostic");
    }
}

/// The half widths are where derived `Debug` actively lies: `half::f16`'s
/// `Debug` forwards to `to_f32()`, so it reports the WIDENED image rather
/// than the shortest string that parses back to the stored f16. The
/// pre-migration rendering of the first two fixtures below, measured:
///
/// ```text
/// [Scalar(ScalarPayload { value: ScalarValue { bits: F16(0.099975586) } }),
///  Scalar(ScalarPayload { value: ScalarValue { bits: Bf16(0.100097656) } })]
/// ```
///
/// One stored value, two answers: `0.099975586` in a diagnostic and `0.1`
/// at every exit. That is the §B2.4 harm itself, so these assertions pin
/// the own-width digits and the absence of the widened image.
#[test]
fn fo_diag_half_widths_are_not_reported_through_their_f32_image() {
    let f16 = scalar_of(Prim::F16, 0.1);
    assert_eq!(describe_value(&f16), "f16 0.1");
    // The needle is the shared PREFIX of the measured widened image
    // (`0.099975586`), not the whole string: a `half` bump that shifts the
    // trailing digits must still trip this, and the correct rendering
    // (`0.1`) can never contain it.
    assert!(
        !describe_value(&f16).contains(WIDENED_F16_IMAGE),
        "the f32 image of the stored f16 must not reach the diagnostic"
    );

    let bf16 = scalar_of(Prim::Bf16, 0.1);
    assert_eq!(describe_value(&bf16), "bf16 0.1");
    assert!(!describe_value(&bf16).contains(WIDENED_BF16_IMAGE));

    // And the same value inside a malformed ADT field list, which is how a
    // JSON/CSV shape diagnostic reports it.
    assert_eq!(describe_fields(&[f16, bf16]), "[f16 0.1, bf16 0.1]");
}

/// Bools, strings, and the nonnumeric controls: each names its kind, and
/// a string is quoted so an empty or control-carrying value is visible.
#[test]
fn fo_diag_bools_strings_and_nonnumeric_controls() {
    assert_eq!(describe_value(&RuntimeValue::Bool(true)), "bool true");
    assert_eq!(describe_value(&RuntimeValue::Bool(false)), "bool false");
    // A bool is never reported as a number ([05-OBS-3]).
    assert!(!describe_value(&RuntimeValue::Bool(true)).contains('1'));

    assert_eq!(
        describe_value(&RuntimeValue::String(String::new())),
        "string \"\""
    );
    assert_eq!(
        describe_value(&RuntimeValue::String("a\nb\"c".to_string())),
        "string \"a\\nb\\\"c\""
    );

    assert_eq!(describe_value(&RuntimeValue::Unit), "()");
    assert_eq!(
        describe_value(&RuntimeValue::MappedFile(vec![1, 2, 3])),
        "<mapped-file:3>"
    );
    let checked_function = chelis_deep::parser::parse_str(
        "(fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))",
    )
    .expect("parse checked function fixture")
    .remove(0);
    assert_eq!(
        describe_value(&RuntimeValue::Closure {
            checked_function: Box::new(checked_function),
            params: vec!["x".to_string()],
            param_types: vec![None],
            return_type: None,
            checked_signature: None,
            invocation_contracts: Box::default(),
            body: chelis_deep::ast::Expr::Atom(
                chelis_deep::ast::Atom::Bool(false),
                chelis_deep::Span::new(0, 0)
            ),
            env: Frame::new(),
            precision_env: UnordMap::new(),
            def_name: None,
        }),
        "<closure>"
    );
}

/// Nested structure is `render_value`'s output verbatim -- the diagnostic
/// channel and the exit channel are the same grammar below the top-level
/// kind tag, so nested payloads cannot disagree.
#[test]
fn fo_diag_nested_structure_delegates_to_the_canonical_renderer() {
    let list = RuntimeValue::List(
        vec![
            int_scalar_of(Prim::Int64, 9_007_199_254_740_993),
            scalar_of(Prim::F32, 0.1f32 as f64),
            RuntimeValue::Bool(true),
        ]
        .into(),
    );
    assert_eq!(
        describe_value(&list),
        format!("list {}", render_value(&list))
    );
    assert_eq!(describe_value(&list), "list [9007199254740993, 0.1, true]");

    let tuple = RuntimeValue::Tuple(vec![int_scalar_of(Prim::Int32, 1), RuntimeValue::Unit].into());
    assert_eq!(describe_value(&tuple), "tuple (1, ())");

    // Tensors, dicts, and ADTs already name their own shape, so they are
    // the canonical rendering unchanged.
    let tensor = tensor_value(
        Prim::F32,
        vec![2],
        vec![f64::from(0.1f32), f64::from(0.2f32)],
    );
    assert_eq!(describe_value(&tensor), render_value(&tensor));
    assert_eq!(
        describe_value(&tensor),
        "tensor(shape=[2], data=[0.1, 0.2])"
    );

    let nested = adt(
        "JList",
        vec![RuntimeValue::List(
            vec![
                adt(
                    "JInt",
                    vec![int_scalar_of(Prim::Int64, 9_007_199_254_740_993)],
                ),
                adt("JNum", vec![scalar_of(Prim::F64, 1e-7)]),
            ]
            .into(),
        )],
    );
    assert_eq!(describe_value(&nested), render_value(&nested));
    assert_eq!(
        describe_value(&nested),
        "JList([JInt(9007199254740993), JNum(1e-7)])"
    );
    assert_no_debug_spelling(&describe_value(&nested), "a nested Json diagnostic");
}

/// `describe_fields` tags every field, because a malformed-shape
/// diagnostic has to say WHY the shape was rejected: `[i32 5]` under a
/// `JNum` names the reason an untagged `[5]` does not.
#[test]
fn fo_diag_field_lists_tag_every_field() {
    assert_eq!(describe_fields(&[]), "[]");
    assert_eq!(describe_fields(&[int_scalar_of(Prim::Int32, 5)]), "[i32 5]");
    assert_eq!(
        describe_fields(&[
            scalar_of(Prim::F32, 0.1f32 as f64),
            RuntimeValue::Bool(true),
            RuntimeValue::String("x".to_string()),
        ]),
        "[f32 0.1, bool true, string \"x\"]"
    );
}

/// A missing argument slot reads as a missing argument, not as a Rust
/// `Option` spelling.
#[test]
fn fo_diag_argument_slots_name_an_absent_argument() {
    assert_eq!(describe_argument(None), "nothing");
    let value = int_scalar_of(Prim::Int32, 7);
    assert_eq!(describe_argument(Some(&value)), describe_value(&value));
    assert_no_debug_spelling(&describe_argument(None), "an absent argument");
}

/// Truncation is the boundary's, not the call site's: one cap, applied
/// once, on a char boundary, and stated rather than silent. A rendering
/// that fits is byte-identical to the untruncated form.
#[test]
fn fo_diag_truncation_is_owned_by_the_boundary() {
    let short = RuntimeValue::List(vec![int_scalar_of(Prim::Int32, 1)].into());
    assert_eq!(describe_value(&short), "list [1]");
    assert!(!describe_value(&short).contains("elided"));

    let long = RuntimeValue::List(
        (0..200)
            .map(|i| int_scalar_of(Prim::Int32, i))
            .collect::<Vec<_>>()
            .into(),
    );
    let rendered = describe_value(&long);
    assert!(
        rendered.starts_with("list [0, 1, 2, "),
        "the head of the payload survives: {rendered}"
    );
    assert!(
        rendered.ends_with(" more bytes elided)"),
        "the cut is stated, not silent: {rendered}"
    );

    // Multibyte: the cut lands on a char boundary (a panic here is the
    // failure), and a long multibyte string still truncates.
    let wide = RuntimeValue::String("\u{1F600}".repeat(200));
    let rendered = describe_value(&wide);
    assert!(rendered.ends_with(" more bytes elided)"));
    assert!(rendered.starts_with("string \"\u{1F600}"));
}

/// [05-OP-57]: checked metadata, never empty payloads, supplies dtype and
/// hidden extents. Invalid data or missing witnesses cannot choose defaults.
#[test]
fn list_tensor_bridges_require_checked_dtype_and_empty_shape_evidence() {
    let (shape, data) = nested_list_to_tensor_data(&[], Prim::F64, &[Some(0), Some(3)]).unwrap();
    assert_eq!(shape, vec![0, 3]);
    assert!(matches!(data, ListTensorData::Float(values) if values.is_empty()));
    assert!(nested_list_to_tensor_data(&[], Prim::F64, &[Some(0), None]).is_err());
    assert!(nested_list_to_tensor_data(&[], Prim::F64, &[Some(1)]).is_err());
    assert!(nested_list_to_tensor_data(&[], Prim::String, &[Some(0)]).is_err());
    let value = numeric_scalar(Prim::Int8, 1, 0.0);
    assert!(nested_list_to_tensor_data(&[value], Prim::Int16, &[Some(1)]).is_err());
}

/// A routed named-axis reduction must not re-fold the whole program.
///
/// Preparing a subexpression lowering context folds the pipes in every
/// definition it admits (chelis#1923). Named-axis routing built one of those
/// contexts per routed reduction, so a package with `chelis-std` in scope
/// re-folded all of `chelis-std` on every call (chelis#2207). The context is
/// now a program-scoped fact of the evaluation context, so the fold is
/// bounded by the program rather than by the number of lowering ingresses.
///
/// The receipt is a count, not a wall clock, so it cannot flake under machine
/// load. `chelis_ir::lower::program_context_preparations` rises once per prepared
/// context, which is once per whole-program fold.
///
/// Failing first, measured on this exact test with only the memo in
/// `ProgramScope::routing_lowering_context` bypassed so a context is prepared
/// per ingress as it was before: 1 routed reduction cost 2 whole-program
/// folds and 40 cost 41, so the assertion below reported "40 routed
/// reductions cost 41 whole-program definition folds, 1 costs 2". The 39
/// extra folds for 39 extra reductions are one apiece, which is the defect.
/// With the memo restored both counts are 2.
mod issue_2207_routing_lowering_context {
    use crate::compiler::{eval_selected, wire_values};
    use crate::schema::{EvalRequest, SourceKind};

    /// A program whose recursion applies one named-axis reduction per step.
    ///
    /// `sum(t, rows)` names a *dimension* of the operand rather than a
    /// runtime value, which is what selects the routing lane; an integer axis
    /// takes the ordinary host path and exercises none of this.
    fn source(routed_reductions: u32) -> String {
        format!(
            "def rowsum(t: &tensor[rows, f32]) -> f32 = tensor_to_scalar(sum(t, rows))\n\
             def build() -> tensor[8, f32] = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
             def repeat(n: i64, acc: f32) -> f32 = {{\n\
             t = build()\n\
             if n <= 0i64 then acc else repeat(n - 1i64, acc + rowsum(&t))\n\
             }}\n\
             answer = repeat({routed_reductions}i64, 0.0f32)\n"
        )
    }

    /// Evaluate the program and return its whole-program fold passes.
    ///
    /// Two other things are asserted. The answer, because a program that
    /// failed to evaluate would fold nothing and report a flattering zero.
    /// And the routing count, because the ordinary host path computes the
    /// same sum: a fixture that stopped reaching the named-axis lane would
    /// satisfy the bound below without ever exercising it.
    fn fold_passes_for(routed_reductions: u32) -> u64 {
        chelis_ir::lower::reset_program_context_preparations();
        super::super::named_axis::reset_named_axis_routes();
        let result = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source(routed_reductions),
                bindings: Default::default(),
            },
            &["answer".to_string()],
        )
        .expect("the routed program evaluates");
        let passes = chelis_ir::lower::program_context_preparations();
        assert_eq!(
            super::super::named_axis::named_axis_routes(),
            u64::from(routed_reductions),
            "the fixture must reach the named-axis routing lane once per reduction"
        );
        assert_eq!(
            serde_json::to_value(&result.roots[0].value).unwrap(),
            serde_json::to_value(wire_values::scalar_f32(routed_reductions as f32 * 8.0)).unwrap(),
            "{routed_reductions} routed reductions over a length-8 tensor of ones"
        );
        passes
    }

    #[test]
    fn routed_named_axis_reductions_prepare_the_context_once() {
        // A recursive fixture in a debug evaluator needs the stack the other
        // recursion tests here take, without a runner environment flag.
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let one = fold_passes_for(1);
                let many = fold_passes_for(40);
                eprintln!(
                    "whole-program context preparations: 1 routed reduction = {one}, 40 = {many}"
                );
                assert_eq!(
                    many, one,
                    "40 routed reductions cost {many} whole-program definition folds, 1 costs {one}"
                );
            })
            .expect("spawn the deep-stack evaluator thread")
            .join()
            .expect("the deep-stack evaluator thread completes");
    }
}

/// The rank-polymorphic local-site parity oracle must reach named-axis
/// routing. A host-only execution could satisfy the CLI answer while leaving
/// that route free to drop checker-owned claims again (chelis#3092).
#[test]
fn rank_polymorphic_local_sites_reach_the_named_axis_route() {
    use crate::compiler::eval_selected;
    use crate::schema::{EvalRequest, SourceKind};

    for (later_size, must_trap) in [(3, false), (4, true)] {
        super::named_axis::reset_named_axis_routes();
        let source = format!(
            "def f[r, h](v: &tensor[..r, f32]) -> tensor[..r, h, f32] = {{\n\
             \x20 a: tensor[..r, h, f32] = insert(v, h, 3i64)\n\
             \x20 b: tensor[..r, h, f32] = insert(v, h, {later_size}i64)\n\
             \x20 _ = b\n\
             \x20 a\n\
             }}\n\
             out = f(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n"
        );
        let result = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source,
                bindings: Default::default(),
            },
            &["out".to_string()],
        );
        assert_eq!(
            super::named_axis::named_axis_routes(),
            1,
            "the rank-polymorphic call must reach the checked route"
        );
        if must_trap {
            let error = result.expect_err("the disagreeing later site traps");
            assert!(
                error.errors.iter().any(|diagnostic| diagnostic
                    .message
                    .contains("extent `h`: claimed = 3, insert axis 2 = 4")),
                "the disagreement must identify the realized axis and expected extent"
            );
        } else {
            assert!(result.is_ok(), "the agreeing later site executes");
        }
    }
}

/// chelis#2439: a `grad` application prepared a fresh subexpression lowering
/// context, which copies and folds every definition in the program, standard
/// library included. The context is now a fact of the evaluation context, so
/// the folds are bounded by the program rather than by the number of
/// applications. Counted, like chelis#2207's receipt above, not timed.
///
/// Failing first, measured on this test with only the memo in
/// `ProgramScope::transform_lowering_context` bypassed: 1 application cost 3
/// whole-program context preparations and 40 cost 42, one per application. With the memo
/// both counts are 2.
mod issue_2439_transform_lowering_context {
    use crate::compiler::{eval_selected, wire_values};
    use crate::schema::{EvalRequest, SourceKind};

    /// A program whose recursion applies one `grad` per step.
    fn source(applications: u32) -> String {
        format!(
            "def loss(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
             def repeat(n: i64, acc: f32) -> f32 = {{\n\
             g = grad(loss)(to_tensor([1.0f32, 1.0f32]))\n\
             if n <= 0i64 then acc else repeat(n - 1i64, acc + tensor_to_scalar(sum(g, 0i32)))\n\
             }}\n\
             answer = repeat({applications}i64, 0.0f32)\n"
        )
    }

    /// Evaluate the program and return its whole-program fold passes, after
    /// checking the answer, so an evaluation that failed cannot report a
    /// flattering zero.
    fn fold_passes_for(applications: u32) -> u64 {
        chelis_ir::lower::reset_program_context_preparations();
        let result = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source(applications),
                bindings: Default::default(),
            },
            &["answer".to_string()],
        )
        .expect("the grad program evaluates");
        let passes = chelis_ir::lower::program_context_preparations();
        // d/dx sum(x * x) at [1, 1] is [2, 2], so each application adds 4.
        assert_eq!(
            serde_json::to_value(&result.roots[0].value).unwrap(),
            serde_json::to_value(wire_values::scalar_f32(applications as f32 * 4.0)).unwrap(),
            "{applications} grad applications"
        );
        passes
    }

    #[test]
    fn repeated_grad_applications_prepare_the_context_once() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let one = fold_passes_for(1);
                let many = fold_passes_for(40);
                eprintln!(
                    "whole-program context preparations: 1 grad application = {one}, 40 = {many}"
                );
                assert_eq!(
                    many, one,
                    "40 grad applications cost {many} whole-program definition folds, 1 costs {one}"
                );
            })
            .expect("spawn the deep-stack evaluator thread")
            .join()
            .expect("the evaluator thread finishes");
    }
}

/// chelis#2592: a clone shares containers while preserving field order,
/// dict pairs, and value semantics after a write.
#[test]
fn runtime_value_clone_copies_every_container_in_order() {
    let original = RuntimeValue::Adt {
        ctor: "Record".to_string(),
        source_name: "Record".to_string(),
        fields: vec![
            RuntimeValue::List(vec![RuntimeValue::int64(1), RuntimeValue::int64(2)].into()),
            RuntimeValue::Tuple(vec![RuntimeValue::Bool(true), RuntimeValue::Unit].into()),
            RuntimeValue::Dict(
                vec![
                    (
                        RuntimeValue::String("a".to_string()),
                        RuntimeValue::int64(3),
                    ),
                    (
                        RuntimeValue::String("b".to_string()),
                        RuntimeValue::List(Vec::new().into()),
                    ),
                ]
                .into(),
            ),
        ]
        .into(),
        field_names: Some(vec!["xs".to_string(), "pair".to_string(), "d".to_string()]),
    };
    let mut copy = original.clone();
    let names = |value: &RuntimeValue| match value {
        RuntimeValue::Adt {
            ctor, field_names, ..
        } => (ctor.clone(), field_names.clone()),
        _ => panic!("the copy of a data-type value is a data-type value"),
    };
    assert_eq!(names(&copy), names(&original));
    assert_eq!(
        render_value(&copy),
        "Record([1, 2], (true, ()), dict(a: 3, b: []))"
    );
    if let RuntimeValue::Adt { fields, .. } = &mut copy
        && let RuntimeValue::List(items) = &mut fields[0]
    {
        items.push(RuntimeValue::int64(9));
    }
    assert_eq!(
        render_value(&original),
        "Record([1, 2], (true, ()), dict(a: 3, b: []))"
    );
}

/// chelis#2619: a transform's closures are closure-converted against their
/// own environments. A closure several others reach is staged once, however
/// many paths reach it: a chain where each closure calls the previous two
/// reached the first ones along Fibonacci-many paths, which took 75 s and
/// 8.4 GB at depth 18 before staged values were keyed by value.
///
/// Failing first, measured with only the `FrameCaptures::staged` lookup
/// bypassed: depth 12 staged 431 values; with it, 13.
mod issue_2619_shared_capture_staging {
    use crate::compiler::eval_selected;
    use crate::schema::{EvalRequest, SourceKind};

    fn source(depth: usize) -> String {
        let mut body = String::from(
            "  w = to_tensor([1.0f32, 1.0f32])\n  f0 = fn (x: tensor[2, f32]) -> mul(x, w)\n  f1 = fn (x: tensor[2, f32]) -> mul(x, w)\n",
        );
        for k in 2..depth {
            body.push_str(&format!(
                "  f{k} = fn (x: tensor[2, f32]) -> add(f{}(x), f{}(x))\n",
                k - 1,
                k - 2
            ));
        }
        format!(
            "out = {{\n{body}  grad(fn (x: tensor[2, f32]) -> sum(f{}(x), 0i32))(to_tensor([1.0f32, 2.0f32]))\n}}\n",
            depth - 1
        )
    }

    #[test]
    fn a_closure_many_closures_reach_is_staged_once() {
        super::super::transforms::reset_staged_frame_values();
        let result = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source(12),
                bindings: Default::default(),
            },
            &["out".to_string()],
        )
        .expect("the closure chain evaluates");
        let staged = super::super::transforms::staged_frame_values();
        // f11 is Fib(12) = 144 copies of w, so its gradient is [144, 144].
        assert_eq!(
            serde_json::to_value(&result.roots[0].value).unwrap()["value"]["data"]["bits"],
            serde_json::json!(["43100000", "43100000"])
        );
        assert!(
            staged <= 16,
            "a depth-12 closure chain staged {staged} frame values; one per closure and value is 13"
        );
    }
}

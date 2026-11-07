use super::*;
use chelis_deep::role::{ChildStampRole, child_stamp_role};
use chelis_deep::tag::DeepTag;

/// Parse Surf and hand back the canonical Deep the lowerer receives.
///
/// The oracle builds every fixture this way and never constructs an `Expr`
/// by hand. Round 2 is why: the previous version built its typed-parameter
/// fixture as an `UnknownForm`, a shape the parser does not produce at a
/// binder position, so its typed leg was green against a spelling that
/// never occurs while the real one -- an unstamped list whose head is the
/// name -- fell through undecoded. An oracle over a representation nobody
/// emits proves nothing about the representation everybody emits.
fn deep_program(source: &str) -> Vec<Expr> {
    let decls = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("surf parse");
    chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
        .expect("Surf fixture must desugar")
}

/// The body of the named `def`, which is the subtree the hoist walks.
fn def_body(program: &[Expr], name: &str) -> Expr {
    crate::test_support::with_operation(|operation| {
        let (_, body) = find_top_level_def_named(program, name, operation).expect("the def");
        let Some((DeepTag::Fn, _, kids)) = stamped_parts(body) else {
            panic!("a def body is a `fn` node");
        };
        kids.get(1).expect("the fn body").clone()
    })
}

fn bound_names(source: &str, def: &str) -> Result<Vec<String>, String> {
    let program = deep_program(source);
    let mut out = UnordSet::new();
    names_bound_in(&def_body(&program, def), &mut out)?;
    Ok(out.into_sorted())
}

/// One Surf spelling that binds `shadowed` inside `f`'s body, named by the
/// binder form it exercises. Every entry is real source the parser accepts.
fn binder_spellings() -> Vec<(&'static str, String, &'static str)> {
    vec![
            (
                "typed fn parameter",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (shadowed: tensor[1, f32]) -> mul(shadowed, shadowed))(x)\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "untyped fn parameter",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (shadowed) -> mul(shadowed, shadowed))(x)\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "two typed fn parameters",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (shadowed: tensor[1, f32], other: tensor[1, f32]) -> mul(shadowed, other))(x, x)\n"
                    .to_string(),
                "shadowed",
            ),
            (
                // A typed parameter whose NAME is also a Deep tag spelling.
                // `stamped_parts` would read `(params {type: ..})` as a
                // vocabulary node and collect nothing, so the desugarer emits
                // `^{:type ..} params` instead and the reader's `MetaExpr` arm
                // is what keeps this whole family correct. Round 3 found that
                // arm unguarded: mutating it left every other row green while
                // shipping round 2's defect for this spelling.
                "typed fn parameter named after a Deep tag",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 (fn (params: tensor[1, f32]) -> mul(params, params))(x)\n"
                    .to_string(),
                "params",
            ),
            (
                "let binding",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = {\n  shadowed = mul(x, x)\n  shadowed\n}\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "match pattern binder",
                "type Holder =\n  | Holder { v: tensor[1, f32] }\n\n\
                 def f(x: tensor[1, f32]) -> tensor[1, f32] = \
                 match Holder { v: x } with { | Holder { v: shadowed } => mul(shadowed, shadowed) }\n"
                    .to_string(),
                "shadowed",
            ),
            (
                "pipe stage over a let binder",
                "def f(x: tensor[1, f32]) -> tensor[1, f32] = {\n  \
                 shadowed = x |> mul(x)\n  shadowed\n}\n"
                    .to_string(),
                "shadowed",
            ),
        ]
}

/// The vocabulary oracle for chelis#1266's hoist, built through the parser.
///
/// Every binder spelling Surf can write must yield its name, because the
/// hoist substitutes a local for a projection and must never do so under a
/// binder it could not see. A spelling that yields nothing is either a
/// silent wrong answer or, with the fail-closed exit, a definition whose
/// hoist is abandoned and whose C lane then refuses.
///
/// EVIDENTIARY STATUS: regression test. Measured RED at `2570da8d1` on the
/// "typed fn parameter" and "two typed fn parameters" rows, where the
/// reader refused with "a binder position of `fn` at index 0 carries a
/// spelling this walk cannot read a name from" and bound nothing.
#[test]
fn every_surf_binder_spelling_yields_its_name() {
    for (form, source, binder) in binder_spellings() {
        let read = bound_names(&source, "f");
        let names = read.unwrap_or_else(|error| panic!("{form}: the reader refused: {error}"));
        assert!(
            names.iter().any(|name| name == binder),
            "{form}: bound {names:?}, which does not include `{binder}`"
        );
    }
}

/// The role table is the authority for WHICH positions bind, and it must
/// keep naming the forms above. A tag that gains a binder position joins
/// this set on its own.
///
/// EVIDENTIARY STATUS: disposition lock on the representation choice.
#[test]
fn the_role_table_declares_the_binder_positions_the_reader_uses() {
    let mut tags = Vec::new();
    for tag in DeepTag::ALL {
        for arity in [1usize, 2, 3] {
            for index in 0..arity {
                if child_stamp_role(tag, index, arity) == ChildStampRole::Binder
                    && !tags.contains(&tag)
                {
                    tags.push(tag);
                }
            }
        }
    }
    for required in [DeepTag::Params, DeepTag::Bind, DeepTag::PatVar, DeepTag::Fn] {
        assert!(
            tags.contains(&required),
            "`{}` must declare a binder position: {:?}",
            required.as_str(),
            tags.iter().map(|t| t.as_str()).collect::<Vec<_>>()
        );
    }
}

/// The fail-closed half: a binder position carrying a spelling the reader
/// cannot decode refuses, so the caller abandons the hoist rather than
/// treating the name as unbound.
///
/// This one is built by hand deliberately, because its whole subject is a
/// shape the parser does NOT produce. It is the only hand-built fixture
/// here, and it asserts the refusal path rather than a decode.
///
/// EVIDENTIARY STATUS: disposition lock.
#[test]
fn an_undecodable_binder_position_refuses_instead_of_reading_no_name() {
    let span = chelis_deep::span::Span::new(0, 0);
    // An UNSTAMPED form whose head is not a name: no closed tag matches
    // it, so it is not a container the walk descends, and nothing in it
    // spells a binder.
    let unreadable = Expr::Node(
        Box::new(chelis_deep::node::Node::new(
            DeepTag::Params,
            chelis_deep::Metadata::default(),
            vec![Expr::BareList(
                vec![Expr::Atom(Atom::Int(1.into()), span)],
                span,
            )],
        )),
        span,
    );
    let mut out = UnordSet::new();
    let read = names_bound_in(&unreadable, &mut out);
    assert!(
        read.is_err(),
        "a binder position holding a node with no name must refuse, got {out:?}"
    );
    assert!(
        read.unwrap_err().contains("params"),
        "the refusal names the construct"
    );
}

#[test]
fn inlined_result_claims_come_only_from_authored_literal_results() {
    let body_ty = HostTypeTerm::Tensor(TensorType {
        dims: vec![DimInfo::Lit(9)],
        precision: Prim::F32,
    });
    let body = || HostExpr::new(HostExprKind::Var("produced".into(), body_ty.clone()));

    assert!(matches!(
        retain_inlined_result_claim(body(), None).kind,
        HostExprKind::Var(..)
    ));

    let named_signature = (
        Vec::new(),
        HostTypeTerm::Tensor(TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        }),
    );
    assert!(matches!(
        retain_inlined_result_claim(body(), Some(&named_signature)).kind,
        HostExprKind::Var(..)
    ));

    let literal_result = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let literal_signature = (Vec::new(), HostTypeTerm::Tensor(literal_result.clone()));
    let retained = retain_inlined_result_claim(body(), Some(&literal_signature));
    let HostExprKind::ResultClaimScope { plan, ty, .. } = retained.kind else {
        panic!("an authored literal result must retain a claim scope");
    };
    assert_eq!(plan.result(), &literal_result);
    assert_eq!(
        ty, body_ty,
        "body inference must not rewrite the claim plan"
    );
}

#[test]
fn retained_result_uses_the_first_formal_kind_for_its_witness() {
    fn claim(list_first: bool, nested_pattern: bool) -> HostResultRequirementPlan {
        let tensor = TensorType {
            dims: vec![DimInfo::Named("seq".into(), None)],
            precision: Prim::F32,
        };
        let tensor_term = HostTypeTerm::Tensor(tensor.clone());
        let list_term = if nested_pattern {
            HostTypeTerm::Tuple(vec![tensor_term.clone()])
        } else {
            HostTypeTerm::List(Box::new(tensor_term.clone()))
        };
        let nested_claim = nested_pattern.then(|| {
            let syntax = host_type_syntax(&list_term, chelis_deep::Span::new(0, 0)).unwrap();
            Arc::new(
                crate::claim_pattern::ClaimPattern::derive(
                    &syntax,
                    &chelis_checker_core::adt::AdtRegistry::default(),
                )
                .unwrap(),
            )
        });
        let direct = HostParam {
            name: "direct".into(),
            ty: tensor_term.clone(),
        };
        let list = HostParam {
            name: "list".into(),
            ty: list_term.clone(),
        };
        let formals = if list_first {
            vec![list, direct]
        } else {
            vec![direct, list]
        };
        let direct_position = usize::from(list_first);
        let entry = HostExpr::new(HostExprKind::SignatureEntry {
            contract: EntryContract::from_params(&formals),
            plan: SignatureEntryPlan::new([HostTensorInput {
                name: "direct".into(),
                ty: tensor.clone(),
            }]),
            args: vec![HostExpr::new(HostExprKind::Var(
                "prepared_direct".into(),
                tensor_term.clone(),
            ))],
            positions: vec![direct_position],
            lists: vec![HostListEntry {
                position: 1 - direct_position,
                name: "list".into(),
                ty: list_term.clone(),
                value: HostExpr::new(HostExprKind::Var("prepared_list".into(), list_term)),
                claim: nested_claim,
            }],
        });
        let result_ty = HostTypeTerm::Tensor(tensor.clone());
        let retained = HostExpr::new(HostExprKind::RetainedInvocation {
            forwarding: None,
            bindings: vec![HostBinding {
                name: "entry".into(),
                display_name: None,
                display_roots: Vec::new(),
                ty: HostTypeTerm::Unit,
                value: entry,
            }],
            body: Box::new(HostExpr::new(HostExprKind::Var(
                "produced".into(),
                result_ty.clone(),
            ))),
            ty: result_ty,
        });
        let scoped = retain_actualized_result_claim(retained, Some(&tensor));
        let HostExprKind::RetainedInvocation { body, .. } = scoped.kind else {
            panic!("retained invocation lost its boundary");
        };
        let HostExprKind::ResultClaimScope { plan, .. } = body.kind else {
            panic!("retained result lost its obligation");
        };
        plan.axes()[0].1.clone()
    }

    for nested_pattern in [false, true] {
        assert!(matches!(
            claim(false, nested_pattern),
            HostResultRequirementPlan::NamedDirect {
                source,
                prepared,
                axis: 0,
                ..
            } if source == "direct" && prepared == "prepared_direct"
        ));
        assert!(matches!(
            claim(true, nested_pattern),
            HostResultRequirementPlan::NamedList { state: 0, .. }
        ));
    }
}

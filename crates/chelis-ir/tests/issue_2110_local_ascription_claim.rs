//! chelis#2110: authored local tensor ascriptions become producer-owned
//! runtime obligations without being confused with inferred type metadata.

use chelis_ir::dag::{Dag, DimInfo, ExtentWitnessSite, RiscOp, RtAxis, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::optimize::{common_subexpr_eliminate, dead_code_eliminate};
use chelis_ir::specialize::specialize_for_exact_arithmetic;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;
use chelis_types::types::Prim;

fn lower(source: &str) -> chelis_ir::Dag {
    let decls = parse_str(source).expect("Surf parse");
    let checked = check_ir_program(&desugar_program(&decls).expect("Surf fixture must desugar"))
        .unwrap_or_else(|report| {
            panic!("type check failed: {:?}", report.errors);
        });
    chelis_ir::host::lower_named_tensor_entry_dag(&checked, "f").expect("named tensor entry lowers")
}

fn checked(source: &str) -> chelis_types::CheckedProgram {
    let decls = parse_str(source).expect("Surf parse");
    check_ir_program(&desugar_program(&decls).expect("Surf fixture must desugar")).unwrap_or_else(
        |report| {
            panic!("type check failed: {:?}", report.errors);
        },
    )
}

fn local_claims(dag: &chelis_ir::Dag) -> Vec<(usize, u64, String, String, i32)> {
    dag.nodes()
        .iter()
        .filter_map(|node| {
            let RiscOp::ExtentWitness {
                site:
                    ExtentWitnessSite::LocalAscriptionClaim {
                        ascription_id,
                        binding,
                        claim,
                        axis: chelis_ir::dag::RtAxis::Lit(axis),
                    },
                ..
            } = &node.op
            else {
                return None;
            };
            Some((
                node.id.0,
                *ascription_id,
                binding.clone(),
                claim.clone(),
                *axis,
            ))
        })
        .collect()
}

fn direct_source(body: &str) -> String {
    format!("def f(x: tensor[*, f32]) -> tensor[*, f32] = {{\n  {body}\n}}\n")
}

fn runtime_branch_source() -> &'static str {
    "def f(flag: bool, x: tensor[*, f32]) -> tensor[*, f32] = \
     if flag then {\n  \
       y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
       y\n\
     } else x\n"
}

fn runtime_branch_inlined_helper_source() -> &'static str {
    "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
       y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
       y\n\
     }\n\
     def f(flag: bool, x: tensor[*, f32]) -> tensor[*, f32] = \
       if flag then helper(x) else x\n"
}

fn runtime_bool(value: bool) -> TensorValue {
    TensorValue::finalize_from_wide_int(
        "issue_2110_runtime_branch",
        Prim::Bool,
        vec![],
        vec![i64::from(value)],
    )
    .expect("typed scalar Bool")
}

#[test]
fn untaken_runtime_branch_does_not_execute_its_local_ascription_guard() {
    let dag = lower(runtime_branch_source());
    let token = chelis_ir::dag::NodeId(local_claims(&dag)[0].0);
    let owner = dag
        .nodes()
        .iter()
        .find(|node| node.shape_deps.contains(&token))
        .expect("runtime-branch initializer owns its local claim");
    // The claim's activation is its carrier's owner activation, the arm's
    // scalar Bool path (spec/10 section 3.2), and nothing else carries it.
    let activation = owner
        .owner
        .activation
        .expect("a path-local claim's carrier runs under the arm's activation");
    let activation = dag.get(activation).unwrap();
    assert!(
        activation.output_type.dims.is_empty() && activation.output_type.precision == Prim::Bool,
        "a path-local claim has one exact scalar Bool activation"
    );
    assert!(
        chelis_ir::verify::verify(&dag).is_empty(),
        "{:?}",
        chelis_ir::verify::verify(&dag)
    );
    let values = eval_tensor_with(&dag, |name| match name {
        "flag" => Some(runtime_bool(false)),
        "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
        _ => None,
    })
    .expect("the untaken branch's disagreeing local claim is not observed");
    let root = *dag.roots().last().expect("entry root");
    assert_eq!(values[&root].shape, vec![3]);
    assert_eq!(values[&root].to_f64_lossy_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn selected_runtime_branch_executes_its_local_ascription_guard() {
    let dag = lower(runtime_branch_source());
    let error = eval_tensor_with(&dag, |name| match name {
        "flag" => Some(runtime_bool(true)),
        "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
        _ => None,
    })
    .expect_err("the selected branch's disagreeing local claim must trap");
    assert_eq!(
        error,
        "extent `2`: claimed = 2, pad axis 0 = 3\n\
         numeric trap: domain in pad at i64"
    );
}

#[test]
fn untaken_runtime_branch_does_not_execute_an_inlined_helpers_local_ascription_guard() {
    let dag = lower(runtime_branch_inlined_helper_source());
    let values = eval_tensor_with(&dag, |name| match name {
        "flag" => Some(runtime_bool(false)),
        "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
        _ => None,
    })
    .expect("the untaken helper branch's disagreeing local claim is not observed");
    let root = *dag.roots().last().expect("entry root");
    assert_eq!(values[&root].shape, vec![3]);
    assert_eq!(values[&root].to_f64_lossy_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn selected_runtime_branch_executes_an_inlined_helpers_local_ascription_guard() {
    let dag = lower(runtime_branch_inlined_helper_source());
    let error = eval_tensor_with(&dag, |name| match name {
        "flag" => Some(runtime_bool(true)),
        "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
        _ => None,
    })
    .expect_err("the selected helper branch's disagreeing local claim must trap");
    assert_eq!(
        error,
        "extent `2`: claimed = 2, pad axis 0 = 3\n\
         numeric trap: domain in pad at i64"
    );
}

/// The claimed arm in the `else` position: the join's condition takes its
/// extent from that arm, whose zeros have the claimed extent 2, so only the
/// `where` rule that an unselected branch is not shape-checked (decisions
/// section 25) lets the taken `then` value through at its own extent.
///
/// Evidentiary status: DISPOSITION LOCK at 096daea8c (the untaken `pad` was
/// computed from `x` there); REGRESSION TEST against 987b79e2e, where the
/// zeros met `x` at the join and failed the shape check.
#[test]
fn an_untaken_else_arms_claimed_result_is_not_read_at_the_join() {
    let dag = lower(
        "def f(flag: bool, x: tensor[*, f32]) -> tensor[*, f32] = \
         if flag then x else {\n  \
           y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
           y\n\
         }\n",
    );
    let run = |flag| {
        eval_tensor_with(&dag, |name| match name {
            "flag" => Some(runtime_bool(flag)),
            "x" => Some(TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0])),
            _ => None,
        })
    };
    let values = run(true).expect("the untaken else arm's claim is not observed");
    let root = *dag.roots().last().expect("entry root");
    assert_eq!(values[&root].shape, vec![3]);
    assert_eq!(values[&root].to_f64_lossy_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(
        run(false).expect_err("the selected else arm's claim traps"),
        "extent `2`: claimed = 2, pad axis 0 = 3\n\
         numeric trap: domain in pad at i64"
    );
}

#[test]
fn lowering_matches_the_authored_binding_and_attaches_one_exact_site_to_its_initializer() {
    let dag = lower(&direct_source(
        "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y",
    ));
    let claims = local_claims(&dag);
    assert_eq!(claims.len(), 1, "exactly one authored obligation: {dag:?}");
    assert_eq!(claims[0].1, 0);
    assert_eq!(claims[0].2, "y");
    assert_eq!(claims[0].3, "2");
    assert_eq!(claims[0].4, 0);

    let token = chelis_ir::dag::NodeId(claims[0].0);
    let owners = dag
        .nodes()
        .iter()
        .filter(|node| node.shape_deps.contains(&token))
        .collect::<Vec<_>>();
    assert_eq!(owners.len(), 1, "one initializer owns the token");
    assert!(
        matches!(owners[0].op, RiscOp::Pad { .. }),
        "the pad initializer, not the later binding use, owns the guard: {:?}",
        owners[0]
    );
    let error = eval_tensor_with(&dag, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]))
    })
    .expect_err("the producer-owned local ascription must trap");
    assert_eq!(
        error,
        "extent `2`: claimed = 2, pad axis 0 = 3\n\
         numeric trap: domain in pad at i64"
    );
}

#[test]
fn aliasing_and_dead_value_elimination_retain_the_initializer_obligation() {
    let dag = lower(&direct_source(
        "raw = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y: tensor[2, f32] = raw\n  \
         x",
    ));
    assert_eq!(local_claims(&dag).len(), 1, "dead except for its trap");
    assert!(
        dag.nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Pad { .. })
                && node.shape_deps.iter().any(|dependency| {
                    matches!(
                        dag.get(*dependency).map(|node| &node.op),
                        Some(RiscOp::ExtentWitness {
                            site: ExtentWitnessSite::LocalAscriptionClaim { .. },
                            ..
                        })
                    )
                })),
        "the aliased pad remains live as the initializer owner: {dag:?}"
    );
}

#[test]
fn rebuild_cse_dce_and_specialization_preserve_the_exact_site() {
    let source = direct_source("y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y");
    let original = lower(&source);
    for (pass, dag) in [
        ("cse", common_subexpr_eliminate(&original)),
        ("dce", dead_code_eliminate(&original)),
        ("specialize", specialize_for_exact_arithmetic(&original)),
    ] {
        let claims = local_claims(&dag);
        assert_eq!(claims.len(), 1, "{pass} must retain one site");
        assert_eq!(&claims[0].1, &0);
        assert_eq!(&claims[0].2, "y");
        assert_eq!(&claims[0].3, "2");
        assert_eq!(&claims[0].4, &0);
    }
}

#[test]
fn fusion_preserves_the_exact_local_site_on_the_rebuilt_initializer() {
    fn fusion_dag(claimed: bool) -> Dag {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Named("*".into(), None)],
            precision: Prim::F32,
        };
        let claim = claimed.then(|| {
            dag.add_node(
                decl,
                RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::LocalAscriptionClaim {
                        ascription_id: 0,
                        binding: "y".into(),
                        claim: "2".into(),
                        axis: RtAxis::Lit(0),
                    },
                    parameter: String::new(),
                    axis: RtAxis::Lit(0),
                    requirements: vec![
                        chelis_types::scalar_from_i64("test", Prim::Int64, 2).unwrap(),
                    ],
                    claims: Vec::new(),
                },
                Vec::new(),
                TensorType {
                    dims: Vec::new(),
                    precision: Prim::Int64,
                },
                None,
            )
        });
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            Vec::new(),
            ty.clone(),
            None,
        );
        let add = dag.add_node(decl, RiscOp::Add, vec![input, input], ty.clone(), None);
        if let Some(claim) = claim {
            dag.add_shape_dep(add, claim);
        }
        let neg = dag.add_node(decl, RiscOp::Neg, vec![add], ty, None);
        dag.add_root(neg);
        dag
    }

    let unclaimed = fusion_dag(false);
    assert!(
        chelis_ir::fuse::fuse(&unclaimed)
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
        "the unclaimed control must exercise fusion: {unclaimed:?}"
    );

    let original = fusion_dag(true);
    let fused = dead_code_eliminate(&chelis_ir::fuse::fuse(&original));
    let claims = local_claims(&fused);
    assert_eq!(
        claims.len(),
        1,
        "fusion must retain one exact site: {fused:?}"
    );
    let token = chelis_ir::dag::NodeId(claims[0].0);
    let owners = fused
        .nodes()
        .iter()
        .filter(|node| node.shape_deps.contains(&token))
        .collect::<Vec<_>>();
    assert_eq!(owners.len(), 1, "the fused initializer must own the site");
    assert!(
        matches!(owners[0].op, RiscOp::Add),
        "the claimed initializer must retain its primitive attribution: {:?}",
        owners[0]
    );
    let error = eval_tensor_with(&fused, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]))
    })
    .expect_err("fusion must not erase the local trap");
    assert_eq!(
        error,
        "extent `2`: claimed = 2, add axis 0 = 3\n\
         numeric trap: domain in add at i64"
    );
}

#[test]
fn grad_pruning_preserves_a_dead_except_for_trap_local_site() {
    let original = lower(
        "def f(x: tensor[*, f32]) -> tensor[f32] = {\n  \
         y: tensor[2, f32] = exp(neg(x))\n  \
         sum(y, 0)\n\
         }\n",
    );
    let output = original.roots()[0];
    let input = original
        .nodes()
        .iter()
        .find(|node| matches!(&node.op, RiscOp::Load { name } if name == "x"))
        .expect("input load")
        .id;
    let differentiated =
        chelis_ir::grad::grad_dag_checked(&original, output, &[input]).expect("gradient lowering");
    assert_eq!(
        local_claims(&differentiated.dag).len(),
        1,
        "grad's forward-pruning rebuild must retain the initializer trap"
    );
}

#[test]
fn ownership_lowering_keeps_the_claim_live_through_its_initializer_owner() {
    let dag = lower(&direct_source(
        "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y",
    ));
    let claims = local_claims(&dag);
    let token = chelis_ir::dag::NodeId(claims[0].0);
    let owner = dag
        .nodes()
        .iter()
        .find(|node| node.shape_deps.contains(&token))
        .expect("initializer owner")
        .id;
    let owned = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(dag).expect("ownership lowering"),
    )
    .expect("ownership verification");
    let plan = chelis_ir::ownership::plan_c_storage(owned).expect("C storage plan");
    let slot = plan.slot_for_node(token).expect("claim token storage");
    assert!(
        plan.slot(slot).expect("claim slot").last_use_index() >= owner.0,
        "ownership lowering must keep the claim live until its producer reads it"
    );
}

#[test]
fn inferred_metadata_does_not_create_a_local_ascription_site() {
    let dag = lower(&direct_source(
        "inferred = pad(x, [[0i64, 0i64]], 0.0f32)\n  inferred",
    ));
    assert!(local_claims(&dag).is_empty());
}

#[test]
fn host_evaluation_kernel_carries_the_same_local_ascription_site() {
    let source = direct_source("y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y");
    let checked = checked(&source);
    let session = chelis_ir::host::HostLoweringSession::new(&checked);
    let kernel = chelis_ir::host::host_def_kernel(&session, "f")
        .expect("kernel decision")
        .expect("tensor kernel");
    assert_eq!(
        local_claims(&kernel.dag).len(),
        1,
        "the actual Eval kernel must retain the checker-owned site: {:?}",
        kernel.dag
    );
}

#[test]
fn staged_host_partition_retains_the_local_ascription_site() {
    let source = "def f[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = reshape(x, [numel(source)])\n  \
                  y\n\
                  }\n";
    let checked = checked(source);
    let session = chelis_ir::host::HostLoweringSession::new(&checked);
    let kernel = chelis_ir::host::host_def_kernel(&session, "f")
        .expect("kernel decision")
        .expect("tensor kernel");
    assert!(
        kernel.staged.is_some(),
        "numel-fed reshape must exercise the staged host partition"
    );
    assert_eq!(
        local_claims(&kernel.dag).len(),
        1,
        "the partitioned kernel must retain the checker-owned local site: {:?}",
        kernel.dag
    );
}

#[test]
fn each_inlined_activation_gets_a_distinct_one_owner_local_claim() {
    let source = "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  def f(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  left = helper(x)\n  \
                  right = helper(x)\n  \
                  add(left, right)\n\
                  }\n";
    let dag = lower(source);
    let claims = local_claims(&dag);
    assert_eq!(claims.len(), 2, "one claim per helper activation: {dag:?}");
    for (token, _, binding, claim, axis) in claims {
        assert_eq!(binding, "y");
        assert_eq!(claim, "2");
        assert_eq!(axis, 0);
        let token = chelis_ir::dag::NodeId(token);
        assert_eq!(
            dag.nodes()
                .iter()
                .filter(|node| node.shape_deps.contains(&token))
                .count(),
            1,
            "each activation token has exactly one initializer owner"
        );
    }
}

#[test]
fn vmap_shifts_the_local_claim_and_initializer_axis_together() {
    let source = "def helper(x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  }\n\
                  def f(xs: tensor[batch, *, f32]) -> tensor[batch, *, f32] = \
                  vmap(helper)(xs)\n";
    let dag = lower(source);
    let claims = local_claims(&dag);
    assert_eq!(claims.len(), 1, "one vmapped local obligation: {dag:?}");
    assert_eq!(claims[0].2, "y");
    assert_eq!(claims[0].3, "2");
    assert_eq!(claims[0].4, 1, "the prepended batch axis shifts the claim");
    assert!(
        chelis_ir::verify::verify(&dag).is_empty(),
        "{:?}",
        chelis_ir::verify::verify(&dag)
    );
}

#[test]
fn inserted_output_axis_and_observed_source_axis_are_independently_exact() {
    let dag = lower(
        "def f[c, a, h, w](g: &tensor[c, f32], x: &tensor[a, c, h, w, f32]) -> tensor[c, h, w, f32] = {\n  \
         step1: tensor[c, h, f32] = insert(g, 1, shape(x, 2))\n  \
         step2: tensor[c, h, w, f32] = insert(step1, 2, shape(x, 3))\n  \
         step2\n\
         }\n",
    );
    let coordinates = dag
        .nodes()
        .iter()
        .filter_map(|node| {
            let RiscOp::ExtentWitness {
                site:
                    ExtentWitnessSite::LocalAscriptionClaim {
                        axis: RtAxis::Lit(claimed),
                        ..
                    },
                axis: RtAxis::Lit(observed),
                ..
            } = &node.op
            else {
                return None;
            };
            Some((*claimed, *observed))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        coordinates,
        vec![(1, 2), (2, 3)],
        "an insert's authored output axis and its source-tensor observation \
         are separate coordinates: {dag:#?}"
    );
    assert!(
        chelis_ir::verify::verify(&dag).is_empty(),
        "{:?}",
        chelis_ir::verify::verify(&dag)
    );
}

#[test]
fn malformed_local_claim_roles_are_rejected_by_the_native_verifier() {
    let direct = lower(&direct_source(
        "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y",
    ));
    let token = chelis_ir::dag::NodeId(local_claims(&direct)[0].0);
    assert!(chelis_ir::verify::verify(&direct).is_empty());
    for mutation in 0..7 {
        let mut dag = direct.clone();
        let decl = dag.nodes()[0].owner.decl;
        match mutation {
            0 => {
                let RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::LocalAscriptionClaim { binding, .. },
                    ..
                } = &mut dag.node_mut(token).unwrap().op
                else {
                    unreachable!()
                };
                binding.clear();
            }
            1 => {
                let RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::LocalAscriptionClaim { claim, .. },
                    ..
                } = &mut dag.node_mut(token).unwrap().op
                else {
                    unreachable!()
                };
                claim.clear();
            }
            2 => {
                let RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::LocalAscriptionClaim { axis, .. },
                    ..
                } = &mut dag.node_mut(token).unwrap().op
                else {
                    unreachable!()
                };
                *axis = chelis_ir::dag::RtAxis::Lit(1);
            }
            3 => {
                let RiscOp::ExtentWitness { axis, .. } = &mut dag.node_mut(token).unwrap().op
                else {
                    unreachable!()
                };
                *axis = chelis_ir::dag::RtAxis::Lit(1);
            }
            4 => {
                let RiscOp::ExtentWitness { claims, .. } = &mut dag.node_mut(token).unwrap().op
                else {
                    unreachable!()
                };
                claims.push(chelis_ir::dag::ExtentClaim {
                    claim: "forbidden-entry-claim".into(),
                    requirement_declares: false,
                });
            }
            5 => {
                let owners = dag.nodes().iter().map(|node| node.id).collect::<Vec<_>>();
                for owner in owners {
                    dag.node_mut(owner)
                        .unwrap()
                        .shape_deps
                        .retain(|dependency| *dependency != token);
                }
            }
            6 => {
                let owner = dag
                    .nodes()
                    .iter()
                    .find(|node| node.shape_deps.contains(&token))
                    .unwrap()
                    .id;
                let ty = dag.get(owner).unwrap().output_type.clone();
                let duplicate = dag.add_node(decl, RiscOp::Copy, vec![owner], ty, None);
                dag.add_shape_dep(duplicate, token);
            }
            _ => unreachable!(),
        }
        let errors = chelis_ir::verify::verify(&dag);
        assert!(
            !errors.is_empty(),
            "malformed local role {mutation} was accepted"
        );
    }

    let named = lower(
        "def f[n](anchor: tensor[n, f32], x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
         y: tensor[n, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
         y\n\
         }\n",
    );
    let named_token = chelis_ir::dag::NodeId(local_claims(&named)[0].0);
    let mut missing_declaration_identity = named;
    let RiscOp::ExtentWitness { parameter, .. } = &mut missing_declaration_identity
        .node_mut(named_token)
        .unwrap()
        .op
    else {
        unreachable!()
    };
    parameter.clear();
    assert!(
        !chelis_ir::verify::verify(&missing_declaration_identity).is_empty(),
        "a named local claim must retain its declaring parameter identity"
    );
}

/// The node carrying `source`'s one local claim, and the claim's guard site.
fn claim_carrier_and_site(
    source: &str,
) -> (
    chelis_ir::Dag,
    chelis_ir::dag::NodeId,
    chelis_ir::axis_sources::LocalGuardClaim,
) {
    let dag = lower(source);
    let claims = local_claims(&dag);
    assert_eq!(claims.len(), 1, "{dag:#?}");
    let token = chelis_ir::dag::NodeId(claims[0].0);
    let carrier = dag
        .nodes()
        .iter()
        .find(|node| node.shape_deps.contains(&token))
        .expect("a claim token has its carrier")
        .id;
    let sites = chelis_ir::axis_sources::local_dim_guard_sites(&dag).unwrap();
    let [(_, site)] = sites.as_slice() else {
        panic!("{sites:#?}");
    };
    (dag, carrier, site.clone())
}

/// spec/10 section 3.2 (#2413): an arm's local claim is checked under the
/// owner activation of the node carrying it, the one carrier of that fact.
/// The carrier holds no Bool dependency that could name a second
/// activation, and inside a `grad` body spliced into the arm the activation
/// is still the arm's, not the body's own (empty) branch path.
///
/// Evidentiary status: REGRESSION TEST for the `grad` row (at 224414e1f its
/// site had no activation); DISPOSITION LOCK for the direct arm, whose
/// activation was already the arm's, through a Bool shape dependency.
#[test]
fn an_arms_local_claim_is_checked_under_its_carriers_owner_activation() {
    let direct = "def f(flag: bool, x: tensor[*, f32]) -> tensor[*, f32] = \
                  if flag then {\n  \
                  y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                  y\n\
                  } else x\n";
    let spliced = "def h(x: tensor[*, f32]) -> tensor[f32] = {\n  \
                   y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                   sum(y, 0i32)\n\
                   }\n\
                   def f(flag: bool, x: tensor[*, f32]) -> tensor[*, f32] = \
                   if flag then grad(h)(x) else x\n";
    for (row, source) in [("direct", direct), ("grad", spliced)] {
        let (dag, carrier, site) = claim_carrier_and_site(source);
        let carrier = dag.get(carrier).unwrap();
        assert!(site.activation.node().is_some(), "{row}: {dag:#?}");
        assert_eq!(
            site.activation.node(),
            carrier.owner.activation,
            "{row}: {dag:#?}"
        );
        assert_eq!(site.activation.claimed(), carrier.id, "{row}: {dag:#?}");
        assert!(
            carrier.shape_deps.iter().all(|dependency| {
                let dependency = dag.get(*dependency).unwrap();
                !(dependency.output_type.dims.is_empty()
                    && dependency.output_type.precision == Prim::Bool)
            }),
            "{row}: {dag:#?}"
        );
    }
}

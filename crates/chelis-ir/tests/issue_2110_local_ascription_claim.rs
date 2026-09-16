//! chelis#2110: authored local tensor ascriptions become producer-owned
//! runtime obligations without being confused with inferred type metadata.

use chelis_ir::dag::{ExtentWitnessSite, RiscOp};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::optimize::{common_subexpr_eliminate, dead_code_eliminate};
use chelis_ir::specialize::specialize_for_exact_arithmetic;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::check_ir_program;

fn lower(source: &str) -> chelis_ir::Dag {
    let decls = parse_str(source).expect("Surf parse");
    let checked = check_ir_program(&desugar_program(&decls)).unwrap_or_else(|report| {
        panic!("type check failed: {:?}", report.errors);
    });
    chelis_ir::host::lower_named_tensor_entry_dag(&checked, "f").expect("named tensor entry lowers")
}

fn checked(source: &str) -> chelis_types::CheckedProgram {
    let decls = parse_str(source).expect("Surf parse");
    check_ir_program(&desugar_program(&decls)).unwrap_or_else(|report| {
        panic!("type check failed: {:?}", report.errors);
    })
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
         numeric trap: domain in pad at int64"
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
    let execution =
        chelis_ir::evaluation::RandomExecutionContext::new(chelis_ir::host::RandomLoweringState {
            seed: None,
            counter: 0,
        });
    let plan = chelis_ir::host::host_def_evaluation_plan(&session, "f", &execution)
        .expect("kernel decision")
        .expect("tensor kernel");
    assert_eq!(
        local_claims(&plan.kernel_for_inspection().dag).len(),
        1,
        "the actual Eval kernel must retain the checker-owned site: {:?}",
        plan.kernel_for_inspection().dag
    );
}

#[test]
fn staged_host_partition_retains_the_local_ascription_site() {
    let source = "def f(source: tensor[m, f32], x: tensor[n, f32]) -> tensor[*, f32] = {\n  \
                  y: tensor[2, f32] = reshape(x, [numel(source)])\n  \
                  y\n\
                  }\n";
    let checked = checked(source);
    let session = chelis_ir::host::HostLoweringSession::new(&checked);
    let execution =
        chelis_ir::evaluation::RandomExecutionContext::new(chelis_ir::host::RandomLoweringState {
            seed: None,
            counter: 0,
        });
    let plan = chelis_ir::host::host_def_evaluation_plan(&session, "f", &execution)
        .expect("kernel decision")
        .expect("tensor kernel");
    assert!(
        plan.kernel_for_inspection().staged.is_some(),
        "numel-fed reshape must exercise the staged host partition"
    );
    assert_eq!(
        local_claims(&plan.kernel_for_inspection().dag).len(),
        1,
        "the partitioned kernel must retain the checker-owned local site: {:?}",
        plan.kernel_for_inspection().dag
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
fn malformed_local_claim_roles_are_rejected_by_the_native_verifier() {
    let direct = lower(&direct_source(
        "y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  y",
    ));
    let token = chelis_ir::dag::NodeId(local_claims(&direct)[0].0);
    assert!(chelis_ir::verify::verify(&direct).is_empty());
    for mutation in 0..7 {
        let mut dag = direct.clone();
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
                let duplicate = dag.add_node(RiscOp::Copy, vec![owner], ty, None);
                dag.add_shape_dep(duplicate, token);
            }
            _ => unreachable!(),
        }
        assert!(
            !chelis_ir::verify::verify(&dag).is_empty(),
            "malformed local role {mutation} was accepted"
        );
    }

    let named = lower(
        "def f(anchor: tensor[n, f32], x: tensor[*, f32]) -> tensor[*, f32] = {\n  \
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

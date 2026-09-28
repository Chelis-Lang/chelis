//! chelis#1889: a caller-side checked result label is a view of one returned
//! axis. It must not merge that axis with a different binder from the inlined
//! helper activation or with another physical axis of a rank-polymorphic
//! argument.

use chelis_ir::axis_sources::{EntryExtentGuard, entry_extent_guards};
use chelis_ir::dag::{RiscOp, RtAxis};
use chelis_ir::eval::{TensorValue, eval_tensor_with};

type NamedAxis = (String, usize);
type CallerNamedClaim = (String, NamedAxis, NamedAxis);

fn checked_surf(source: &str) -> chelis_types::CheckedProgram {
    let _linked = chelis_types::install_linked_program_guard();
    let declarations = chelis_surf::parser::parse_str(source).expect("Surf parse");
    let expanded = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expansion")
    .into_exprs();
    let checked = chelis_types::check_ir_program(&expanded).unwrap_or_else(|report| {
        panic!(
            "type check failed: {:?}",
            report
                .errors
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
        )
    });
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

fn named_entry(source: &str, name: &str) -> chelis_ir::Dag {
    chelis_ir::host::lower_named_tensor_entry_dag(&checked_surf(source), name)
        .expect("named tensor entry lowers")
}

fn compiled_dags(source: &str) -> Vec<chelis_ir::Dag> {
    let compiled = chelis_ir::host::try_lower_compiled_program(&checked_surf(source))
        .expect("compiled host program lowers");
    let mut dags = compiled.dag.into_iter().collect::<Vec<_>>();
    if let Some(host) = compiled.host {
        dags.extend(
            host.global_tensor_helpers
                .into_iter()
                .map(|helper| helper.dag),
        );
        dags.extend(
            host.functions
                .into_iter()
                .flat_map(|function| function.tensor_helpers)
                .map(|helper| helper.dag),
        );
    }
    dags.into_iter()
        .map(|dag| chelis_ir::specialize::specialize_for_exact_arithmetic(&dag))
        .collect()
}

fn load_axis(dag: &chelis_ir::Dag, endpoint: (chelis_ir::dag::NodeId, usize)) -> (String, usize) {
    let RiscOp::Load { name } = &dag.get(endpoint.0).expect("guard load").op else {
        panic!("entry guard endpoint is not a load: {endpoint:?}\n{dag:#?}");
    };
    (name.as_ref().to_owned(), endpoint.1)
}

fn caller_named_claims(dag: &chelis_ir::Dag) -> Vec<CallerNamedClaim> {
    dag.nodes()
        .iter()
        .flat_map(|node| {
            let RiscOp::ExtentWitness {
                site: chelis_ir::dag::ExtentWitnessSite::Caller,
                parameter,
                axis: RtAxis::Lit(axis),
                claims,
                ..
            } = &node.op
            else {
                return Vec::new();
            };
            claims
                .iter()
                .zip(node.inputs.iter().skip(1))
                .filter_map(|(claim, required)| {
                    let RiscOp::ExtentWitness {
                        site: chelis_ir::dag::ExtentWitnessSite::Caller,
                        parameter: required_parameter,
                        axis: RtAxis::Lit(required_axis),
                        ..
                    } = &dag.get(*required)?.op
                    else {
                        return None;
                    };
                    Some((
                        claim.claim.clone(),
                        (
                            required_parameter.clone(),
                            usize::try_from(*required_axis).ok()?,
                        ),
                        (parameter.clone(), usize::try_from(*axis).ok()?),
                    ))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn caller_result_label_does_not_merge_an_unrelated_helper_parameter_binder() {
    let dag = named_entry(
        "def keep[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[d, f32] = copy(x)\n\
         def bridge(x: tensor[fixed, f32]) -> tensor[fixed, f32] = \
           keep(x, to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        "bridge",
    );

    let false_guard = entry_extent_guards(&dag).into_iter().find(|guard| {
        let EntryExtentGuard::Named {
            claim,
            canonical,
            observed,
        } = guard
        else {
            return false;
        };
        let canonical = load_axis(&dag, *canonical);
        let observed = load_axis(&dag, *observed);
        claim == "fixed"
            && ((canonical == ("x".to_owned(), 0) && observed == ("gain".to_owned(), 0))
                || (canonical == ("gain".to_owned(), 0) && observed == ("x".to_owned(), 0)))
    });
    assert!(
        false_guard.is_none(),
        "the caller's `fixed` view must not import the helper's independent \
         `gain: tensor[fixed]` binder: {false_guard:?}\n{dag:#?}"
    );

    let output = eval_tensor_with(&dag, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![2], vec![4.0, 5.0]))
    })
    .expect("the unrelated gain extent does not constrain the returned x axis");
    let root = *dag.roots().last().expect("bridge root");
    assert_eq!(output[&root].shape, vec![2]);
}

#[test]
fn authored_repeated_binder_still_enforces_one_runtime_witness() {
    let dag = named_entry(
        "def same[d](x: tensor[d, f32], y: tensor[d, f32]) -> tensor[d, f32] = copy(x)\n",
        "same",
    );

    let shared = entry_extent_guards(&dag).into_iter().find(|guard| {
        let EntryExtentGuard::Named {
            claim,
            canonical,
            observed,
        } = guard
        else {
            return false;
        };
        let canonical = load_axis(&dag, *canonical);
        let observed = load_axis(&dag, *observed);
        claim == "d"
            && ((canonical == ("x".to_owned(), 0) && observed == ("y".to_owned(), 0))
                || (canonical == ("y".to_owned(), 0) && observed == ("x".to_owned(), 0)))
    });
    assert!(
        shared.is_some(),
        "an explicitly repeated authored binder remains one runtime witness:\n{dag:#?}"
    );

    let error = eval_tensor_with(&dag, |name| match name {
        "x" => Some(TensorValue::from_vec(vec![2], vec![1.0, 2.0])),
        "y" => Some(TensorValue::from_vec(vec![3], vec![3.0, 4.0, 5.0])),
        _ => None,
    })
    .expect_err("inconsistent witnesses for an authored binder must trap");
    assert!(
        error.contains("extent `d`"),
        "the shared-binder diagnostic remains exact: {error}"
    );
}

#[test]
fn authored_repeated_rank_spread_enforces_each_positional_runtime_witness() {
    let dag = named_entry(
        "def same_rank[rest](x: &tensor[..rest, f32], y: &tensor[..rest, f32]) -> tensor[..rest, f32] = copy(x)\n\
         def bridge(x: tensor[*, *, f32], y: tensor[*, *, f32]) -> tensor[*, *, f32] = \
           same_rank(x, y)\n",
        "bridge",
    );

    let claims = caller_named_claims(&dag);
    for (axis, label) in [(0, "rest[0]"), (1, "rest[1]")] {
        assert!(
            claims.iter().any(|claim| {
                claim
                    == &(
                        label.to_owned(),
                        ("x".to_owned(), axis),
                        ("y".to_owned(), axis),
                    )
                    || claim
                        == &(
                            label.to_owned(),
                            ("y".to_owned(), axis),
                            ("x".to_owned(), axis),
                        )
            }),
            "a repeated rank spread keeps one positional witness per axis: \
             missing {label} in {claims:#?}\n{dag:#?}"
        );
    }
}

#[test]
fn rank_polymorphic_result_label_does_not_merge_distinct_argument_axes() {
    let dag = named_entry(
        "def add_axis[rest](x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = \
           insert(x, one, 1i64)\n\
         def a2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, one, f32] = add_axis(x)\n",
        "a2",
    );

    let false_guard = entry_extent_guards(&dag).into_iter().find(|guard| {
        let EntryExtentGuard::Named {
            claim,
            canonical,
            observed,
        } = guard
        else {
            return false;
        };
        let canonical = load_axis(&dag, *canonical);
        let observed = load_axis(&dag, *observed);
        claim == "seq"
            && ((canonical == ("x".to_owned(), 0) && observed == ("x".to_owned(), 1))
                || (canonical == ("x".to_owned(), 1) && observed == ("x".to_owned(), 0)))
    });
    assert!(
        false_guard.is_none(),
        "the returned `seq` label must not rename the independent `batch` \
         axis into the same entry class: {false_guard:?}\n{dag:#?}"
    );

    let output = eval_tensor_with(&dag, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    })
    .expect("batch and seq remain distinct");
    let root = *dag.roots().last().expect("a2 root");
    assert_eq!(output[&root].shape, vec![2, 3, 1]);
}

#[test]
fn merged_root_does_not_join_a_checked_label_to_the_helpers_sibling_binder() {
    let dags = compiled_dags(
        "def pkg__mylib__Mylib__Axes__keep_rank[rest](x: tensor[..rest, f32], gain: tensor[fixed, f32]) -> tensor[..rest, f32] = copy(x)\n\
         def bridge(x: tensor[fixed, f32]) -> tensor[fixed, f32] = \
           pkg__mylib__Mylib__Axes__keep_rank(x, to_tensor([1.0f32, 2.0f32, 3.0f32]))\n\
         out = bridge(to_tensor([4.0f32, 5.0f32]))\n",
    );
    for dag in &dags {
        let claims = caller_named_claims(dag);
        assert!(
            claims.iter().all(|claim| {
                claim
                    != &(
                        "fixed".to_owned(),
                        ("x".to_owned(), 0),
                        ("gain".to_owned(), 0),
                    )
                    && claim
                        != &(
                            "fixed".to_owned(),
                            ("gain".to_owned(), 0),
                            ("x".to_owned(), 0),
                        )
            }),
            "the retained helper activation must not turn the caller's \
             checked `fixed` result view into an authored x/gain equality: \
             {claims:#?}\n{dag:#?}"
        );
        let false_guard = entry_extent_guards(dag).into_iter().find(|guard| {
            let EntryExtentGuard::Named {
                claim,
                canonical,
                observed,
            } = guard
            else {
                return false;
            };
            let canonical = load_axis(dag, *canonical);
            let observed = load_axis(dag, *observed);
            claim == "fixed"
                && ((canonical == ("x".to_owned(), 0) && observed == ("gain".to_owned(), 0))
                    || (canonical == ("gain".to_owned(), 0) && observed == ("x".to_owned(), 0)))
        });
        assert!(
            false_guard.is_none(),
            "compiled host lowering must not join helper `gain: fixed` to \
             returned `x` through the caller's label: {false_guard:?}\n{dag:#?}"
        );
    }
}

#[test]
fn merged_rank_polymorphic_root_keeps_batch_and_seq_in_distinct_classes() {
    let dags = compiled_dags(
        "def add_axis[rest](x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = \
           insert(x, one, 1i64)\n\
         def widen[pre, post, c](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32] = \
           insert(x, c, 3i64, seq)\n\
         def a1(x: &tensor[seq, f32]) -> tensor[seq, one, f32] = add_axis(x)\n\
         def a2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, one, f32] = add_axis(x)\n\
         def w_lead[c](x: &tensor[seq, hidden, f32]) -> tensor[c, seq, hidden, f32] = widen(x)\n\
         def w_mid[c](x: &tensor[batch, seq, f32]) -> tensor[batch, c, seq, f32] = widen(x)\n\
         out1 = a1(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n\
         out2 = a2(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n\
         outl = w_lead(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n\
         outm = w_mid(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n",
    );
    for dag in &dags {
        let claims = caller_named_claims(dag);
        assert!(
            claims.iter().all(|claim| {
                claim != &("seq".to_owned(), ("x".to_owned(), 0), ("x".to_owned(), 1))
                    && claim != &("seq".to_owned(), ("x".to_owned(), 1), ("x".to_owned(), 0))
            }),
            "the retained helper activation must not turn a rank-spread \
             result view into an authored batch/seq equality: {claims:#?}\n{dag:#?}"
        );
        let false_guard = entry_extent_guards(dag).into_iter().find(|guard| {
            let EntryExtentGuard::Named {
                claim,
                canonical,
                observed,
            } = guard
            else {
                return false;
            };
            let canonical = load_axis(dag, *canonical);
            let observed = load_axis(dag, *observed);
            claim == "seq"
                && ((canonical == ("x".to_owned(), 0) && observed == ("x".to_owned(), 1))
                    || (canonical == ("x".to_owned(), 1) && observed == ("x".to_owned(), 0)))
        });
        assert!(
            false_guard.is_none(),
            "compiled rank-polymorphic host helper keeps batch and seq distinct: \
             {false_guard:?}\n{dag:#?}"
        );
    }
}

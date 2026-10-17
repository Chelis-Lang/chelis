//! chelis#2622: the semantic view of a stamped type applies the
//! substitution to a linear number of type nodes.
//!
//! The view applied the substitution again at every level of the type it
//! walked, so one stamp of depth `d` visited about `d^2 / 2` nodes.
//!
//! The oracle counts the nodes `Subst::apply` visits. The count is
//! deterministic for a fixed program. Doubling the depth must grow it by at
//! most 2.5 times: linear growth gives about 2, quadratic about 4.

use super::*;
use crate::unify::take_apply_visits;

const DEPTHS: [usize; 4] = [64, 128, 256, 512];

fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

fn i32_type() -> &'static str {
    "(t-prim {} i32)"
}

/// One deep type and almost no expression nodes: an identity function whose
/// parameter is annotated with a tuple type `depth` deep. Only the stamps'
/// semantic view sees the depth.
fn annotated_parameter(depth: usize) -> String {
    let leaf = i32_type();
    let ty = format!("(t-tuple {{}} {leaf} ").repeat(depth) + leaf + &")".repeat(depth);
    format!("(def {{}} f (fn {{}} (params {{}} (x {{type: {ty}}})) (var {{}} x)))\n")
}

fn on_measuring_thread<R: Send + 'static>(work: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(work)
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

fn apply_visits(source: &str) -> usize {
    let program = chelis_deep::parse_and_stamp_file(source).expect("the probe parses");
    let _ = take_apply_visits();
    let checked = check_typed_program(&program);
    let visits = take_apply_visits();
    checked.expect("the probe checks");
    visits
}

fn assert_linear(label: &str, generate: fn(usize) -> String) {
    let visits = on_measuring_thread(move || DEPTHS.map(|depth| apply_visits(&generate(depth))));
    eprintln!("{label}: apply visits at depth {DEPTHS:?}: {visits:?}");
    for pair in visits.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "checking a {label} applied the substitution superlinearly in its depth: {visits:?}"
        );
    }
}

#[test]
fn issue_2622_a_deep_annotated_type_applies_linearly() {
    assert_linear("deep annotated parameter", annotated_parameter);
}

#[test]
fn issue_2622_the_detector_rejects_a_semantic_view_that_reapplies_per_level() {
    // Negative control: the removed algorithm, which applied the substitution
    // to the whole type and then again to each child it walked into.
    fn reapply_per_level(subst: &Subst, ty: &Type) -> Type {
        match subst.apply(ty) {
            Type::Tuple(items) => Type::Tuple(
                items
                    .iter()
                    .map(|item| reapply_per_level(subst, item))
                    .collect(),
            ),
            other => other,
        }
    }
    fn nested(depth: usize) -> Type {
        (0..depth).fold(Type::Prim(Prim::Int32), |inner, _| {
            Type::Tuple(vec![Type::Prim(Prim::Int32), inner])
        })
    }
    let subst = Subst::new();
    let mut visits = Vec::new();
    for depth in [256, 512] {
        let ty = nested(depth);
        let _ = take_apply_visits();
        let _ = reapply_per_level(&subst, &ty);
        visits.push(take_apply_visits());
    }
    assert!(
        !grows_linearly(visits[0], visits[1]),
        "the detector accepted a semantic view that re-applies per level: {visits:?}"
    );
    // And the semantic view the checker uses is linear on the same types.
    let mut linear = Vec::new();
    for depth in [256, 512] {
        let ty = nested(depth);
        let _ = take_apply_visits();
        assert_eq!(subst.semantic_type(&ty), ty);
        linear.push(take_apply_visits());
    }
    assert!(
        grows_linearly(linear[0], linear[1]),
        "the semantic view visited superlinearly many nodes: {linear:?}"
    );
}

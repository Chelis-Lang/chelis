//! chelis#3819: checking nested constructor applications applies the
//! substitution to a linear number of type nodes.
//!
//! `Some(Some(... Some(1)))` binds a few variables at each level to the type
//! built so far, which at depth `k` has size about `k`. Two walks made that
//! superlinear. The occurs check applied the substitution again at every
//! level of the type it searched, so one binding cost the sum of its
//! subtrees' sizes, cubic over the chain. And about nine sites apply the
//! substitution to the type each level builds, walking every finished
//! subtree again, quadratic over the chain. The occurs check now applies
//! once and searches the result, and applying returns a child it has found
//! ground without walking it.
//!
//! The oracle counts the nodes `Subst::apply` visits, which is deterministic
//! for a fixed program. Doubling the depth must grow it by at most 2.5
//! times: linear growth gives about 2, quadratic about 4.

use super::*;
use chelis_type_core::take_apply_visits;

const DEPTHS: [usize; 4] = [64, 128, 256, 512];

fn grows_linearly(small: usize, large: usize) -> bool {
    large * 2 <= small * 5
}

fn literal() -> &'static str {
    "(lit {type: (t-prim {} i32)} 1)"
}

/// The issue's witness: `Some` applied `depth` times to a literal.
fn constructor_chain(depth: usize) -> String {
    let body = "(app {} (var {} Some) ".repeat(depth) + literal() + &")".repeat(depth);
    format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
}

/// The same chain with a tuple at every level, so each level's type has two
/// children, one of them already finished.
fn constructor_tuple_chain(depth: usize) -> String {
    let leaf = literal();
    let body = format!("(app {{}} (var {{}} Some) (tuple {{}} {leaf} ").repeat(depth)
        + leaf
        + &"))".repeat(depth);
    format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
}

/// The chain over a parameter whose type is a variable, so every level's
/// type stays open until the declaration generalizes.
fn open_constructor_chain(depth: usize) -> String {
    let body = "(app {} (var {} Some) ".repeat(depth) + "(var {} x)" + &")".repeat(depth);
    format!("(def {{}} f (fn {{}} (params {{}} (x {{type: (t-var {{}} _)}})) {body}))\n")
}

fn apply_visits(source: &str) -> usize {
    let program = chelis_deep::parse_and_stamp_file(source, &chelis_syntax_host::NativeStack)
        .expect("the probe parses");
    let _ = take_apply_visits();
    let checked = check_typed_program(&program);
    let visits = take_apply_visits();
    checked.expect("the probe checks");
    visits
}

fn sweep(generate: fn(usize) -> String) -> [usize; DEPTHS.len()] {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || DEPTHS.map(|depth| apply_visits(&generate(depth))))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

/// A program shape, by name, and its generator.
type Shape = (&'static str, fn(usize) -> String);

#[test]
fn issue_3819_nested_constructor_applications_apply_linearly() {
    let shapes: [Shape; 2] = [
        ("constructor chain", constructor_chain),
        ("constructor chain over tuples", constructor_tuple_chain),
    ];
    let mut superlinear = Vec::new();
    for (label, generate) in shapes {
        let visits = sweep(generate);
        eprintln!("{label}: apply visits at depth {DEPTHS:?}: {visits:?}");
        if visits
            .windows(2)
            .any(|pair| !grows_linearly(pair[0], pair[1]))
        {
            superlinear.push(format!("{label}: {visits:?}"));
        }
    }
    assert!(
        superlinear.is_empty(),
        "checking nested constructor applications applied the substitution superlinearly: \
         {superlinear:?}"
    );
}

#[test]
fn issue_3819_an_open_constructor_chain_applies_at_most_quadratically() {
    // Every level's type holds the parameter's variable, so no level is
    // ground and each is applied in full: quadratic, about 4 per doubling.
    // The occurs check's per-level re-application made it cubic, about 8.
    let visits = sweep(open_constructor_chain);
    eprintln!("constructor chain over a variable: apply visits at depth {DEPTHS:?}: {visits:?}");
    for pair in visits.windows(2) {
        assert!(
            pair[1] * 2 <= pair[0] * 9,
            "checking an open constructor chain applied the substitution cubically: {visits:?}"
        );
    }
}

/// A tuple chain `depth` deep with a free variable at every level, which
/// applying cannot skip as ground.
fn open_chain(depth: usize, first: u32) -> Type<ErrorWitness> {
    (0..depth).fold(Type::Prim(Prim::Int32), |inner, level| {
        Type::tuple(vec![Type::Var(TypeVar(first + level as u32)), inner])
    })
}

#[test]
fn issue_3819_one_occurs_check_visits_the_type_once() {
    // Binding a variable to a type of size `n` applies the substitution to
    // that type once for the occurs check, whatever its depth.
    let mut visits = Vec::new();
    for depth in DEPTHS {
        let ty = open_chain(depth, 1);
        let mut subst = Subst::<ErrorWitness>::new();
        let _ = take_apply_visits();
        unify(&Type::Var(TypeVar(0)), &ty, &mut subst).expect("the binding is acyclic");
        visits.push(take_apply_visits());
    }
    eprintln!("one binding: apply visits at depth {DEPTHS:?}: {visits:?}");
    for pair in visits.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "one binding applied the substitution superlinearly in the type's size: {visits:?}"
        );
    }
    // The check still finds the variable at any depth.
    let deepest = DEPTHS[DEPTHS.len() - 1];
    let cyclic = open_chain(deepest, 1);
    let mut subst = Subst::<ErrorWitness>::new();
    let error = unify(&Type::Var(TypeVar(deepest as u32)), &cyclic, &mut subst)
        .expect_err("the variable occurs at the bottom of the chain");
    assert!(
        matches!(error.kind, TypeErrorKind::OccursCheck),
        "{error:?}"
    );
}

#[test]
fn issue_3819_a_ground_child_is_returned_without_a_walk() {
    // The first application walks a ground chain and records its children;
    // applying the result again visits only the node that holds them.
    let ground = (0..DEPTHS[DEPTHS.len() - 1]).fold(Type::Prim(Prim::Int32), |inner, _| {
        Type::tuple(vec![Type::Prim(Prim::Int32), inner])
    });
    let subst = Subst::<ErrorWitness>::new();
    let _ = take_apply_visits();
    let applied = subst.apply(&ground);
    let first = take_apply_visits();
    assert_eq!(applied, ground);
    let again = subst.apply(&applied);
    let second = take_apply_visits();
    assert_eq!(again, ground);
    assert!(
        first > DEPTHS[0],
        "the first application walks the chain: {first}"
    );
    assert_eq!(
        second, 1,
        "applying a recorded ground chain walked it again"
    );
    // A copy of the substitution shares what was recorded.
    let copy = subst.clone();
    let _ = copy.apply(&applied);
    assert_eq!(
        take_apply_visits(),
        1,
        "a copy walked a recorded ground chain"
    );
}

#[test]
fn issue_3819_the_detector_rejects_an_occurs_check_that_reapplies_per_level() {
    // Negative control: the removed occurs check, which applied the
    // substitution again at every level of the type it searched.
    fn occurs_reapplying(v: TypeVar, ty: &Type<ErrorWitness>, subst: &Subst<ErrorWitness>) -> bool {
        match subst.apply(ty) {
            Type::Var(found) => found == v,
            Type::Tuple(items) => items.iter().any(|item| occurs_reapplying(v, item, subst)),
            _ => false,
        }
    }
    let subst = Subst::<ErrorWitness>::new();
    let visits = DEPTHS.map(|depth| {
        let ty = open_chain(depth, 1);
        let _ = take_apply_visits();
        assert!(!occurs_reapplying(TypeVar(0), &ty, &subst));
        take_apply_visits()
    });
    assert!(
        visits
            .windows(2)
            .all(|pair| !grows_linearly(pair[0], pair[1])),
        "the detector accepted an occurs check that re-applies per level: {visits:?}"
    );
}

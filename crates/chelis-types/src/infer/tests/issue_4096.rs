//! spec/04 §3.1 and §4.5.2: selected constructor contracts own their equations.
use super::*;

fn equation_chain_work(count: u32, reverse: bool) -> usize {
    let subst = Subst::new();
    let mut equations = (0..count)
        .map(|index| ResultConstraint::Annotation {
            actual: Type::Var(TypeVar(index)),
            declared: Type::Var(TypeVar(index + 1)),
        })
        .collect::<Vec<_>>();
    if reverse {
        equations.reverse();
    }
    for equation in equations {
        subst.record_result_constraint(equation);
    }
    // An unrelated component must never become evidence for the requested type.
    subst.record_result_constraint(ResultConstraint::Annotation {
        actual: Type::Var(TypeVar(count + 10)),
        declared: Type::Var(TypeVar(count + 11)),
    });
    let mut product = InferenceProduct::default();
    product.import_result_constraints(&subst);
    take_result_replay_work();
    let connected = product.result_equations_for(&Type::Var(TypeVar(count)), &subst);
    assert_eq!(connected.len(), count as usize);
    assert!(
        connected
            .iter()
            .flat_map(ResultConstraint::types)
            .all(|ty| { matches!(ty, Type::Var(variable) if variable.0 <= count) })
    );
    take_result_replay_work()[0]
}

#[test]
fn result_equation_reachability_is_independent_of_registration_order() {
    for reverse in [false, true] {
        let small = equation_chain_work(32, reverse);
        let large = equation_chain_work(128, reverse);
        assert!(small > 0);
        assert!(large <= small * 5, "{small} -> {large}, reverse={reverse}");
    }
}

// spec/04 sections 3.1 and 4.5.1: list elements keep rank, precision and
// rigid-axis constraints; a concrete tensor join does not owe a second,
// unbound generic constructor application.
fn tensor_lists(count: usize) -> String {
    let mut source = String::from("def probe[n, c, h, w](x: tensor[n, c, h, w, f32]) -> i64 = {\n");
    for index in 0..count {
        source.push_str(&format!("  values_{index} = [x, x, x]\n"));
    }
    source.push_str("  0i64\n}\n");
    source
}

fn replay_work(source: &str) -> usize {
    take_result_replay_work();
    checked_surf(source);
    take_result_replay_work().iter().sum()
}

#[test]
fn concrete_tensor_lists_do_not_accumulate_generic_result_replays() {
    let small = replay_work(&tensor_lists(32));
    let large = replay_work(&tensor_lists(128));
    assert!(
        small > 0,
        "generic final Cons still checks its result origin"
    );
    assert!(
        large <= small * 5,
        "32 -> 128 tensor lists: {small} -> {large}"
    );
}

fn checks(source: &str) -> bool {
    let declarations =
        chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    check_ir_program(&deep).is_ok()
}

#[test]
fn tensor_join_preserves_rank_precision_and_rigid_axes() {
    assert!(checks(
        "values = [to_tensor([1.0f32]), to_tensor([2.0f32, 3.0f32])]"
    ));
    assert!(!checks(
        "values = [to_tensor([1.0f32]), to_tensor([[2.0f32]])]"
    ));
    assert!(!checks(
        "values = [to_tensor([1.0f32]), to_tensor([2.0f64])]"
    ));
    assert!(!checks(
        "def probe[n, m](x: tensor[n, f32], y: tensor[m, f32]) = [x, y]"
    ));
}

#[test]
fn tensor_join_preserves_singleton_empty_and_wildcard_directions() {
    for (left, right) in [(1, 3), (3, 1), (0, 3), (3, 0), (0, 0)] {
        assert!(checks(&format!(
            "def probe(x: tensor[{left}, f32], y: tensor[{right}, f32]) = [x, y]"
        )));
        if left != right {
            assert!(!checks(&format!(
                "def probe[n](x: tensor[{left}, f32], y: tensor[{right}, f32]) -> List[tensor[n, f32]] = [x, y]"
            )));
        }
    }
    for (left, right) in [("2", "*"), ("*", "2")] {
        assert!(checks(&format!(
            "def probe(x: tensor[{left}, f32], y: tensor[{right}, f32]) = [x, y]"
        )));
        assert!(!checks(&format!(
            "def probe[n](x: tensor[{left}, f32], y: tensor[{right}, f32]) -> List[tensor[n, f32]] = [x, y]"
        )));
    }
}

#[test]
fn constructor_routes_check_eager_producers_and_pending_gradients() {
    assert!(checks(
        "def probe() -> i64 = { values = [to_tensor([1.0f32]), reshape(to_tensor([2.0f32]), [1i64])]\n 0i64\n }"
    ));
    assert!(!checks(
        "def probe() -> i64 = { values = [to_tensor([1.0f32]), reshape(to_tensor([2.0f32]), [2i64])]\n 0i64\n }"
    ));
    assert!(!checks(
        "def probe() = { g = grad(fn (x) -> 1.0f32)\n [g, grad(fn (x: f32) -> 1.0f32)]\n }"
    ));
    assert!(checks(
        "def probe() = { g = grad(fn (x) -> 1.0f32)\n value = g(2.0f32)\n [g, grad(fn (x: f32) -> 1.0f32)]\n }"
    ));
}

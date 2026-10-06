//! A list literal is a `Cons` chain, but its checker work must scale with its elements.
//! Result-origin equalities remain required, including for Grad values in lists.

use super::*;
use std::time::Instant;

fn check_surf(source: &str) -> Result<CheckedProgram, InferResult> {
    let declarations = chelis_surf::parser::parse_str(source).expect("Surf fixture parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture desugars");
    check_ir_program(&deep)
}

fn list_replay_work(elements: usize) -> [usize; 5] {
    let items = vec!["0.0f64"; elements].join(", ");
    let source = format!("probe = to_tensor([{items}])");
    let _ = take_result_replay_work();
    let checked = check_surf(&source).expect("homogeneous numeric list checks");
    assert!(checked.infer_stats().typed_nodes > elements);
    take_result_replay_work()
}

fn callable_list_replay_work(elements: usize) -> [usize; 5] {
    let items = vec!["grad(fn (w: f32) -> 1.0f32)"; elements].join(", ");
    let source = format!("probe = [{items}]");
    let _ = take_result_replay_work();
    check_surf(&source).expect("explicitly typed gradient list checks");
    take_result_replay_work()
}

fn nested_list_replay_work(rows: usize, columns: usize) -> [usize; 5] {
    let row = format!("[{}]", vec!["0.0f64"; columns].join(", "));
    let source = format!("probe = to_tensor([{}])", vec![row; rows].join(", "));
    let _ = take_result_replay_work();
    check_surf(&source).expect("nested numeric list checks");
    take_result_replay_work()
}

#[test]
fn numeric_list_replay_work_is_linear() {
    let small = list_replay_work(100);
    let large = list_replay_work(1000);
    let small_total: usize = small.iter().sum();
    let large_total: usize = large.iter().sum();
    assert!(
        small_total > 0,
        "the work counter must observe result-origin checks"
    );
    assert!(
        large_total <= small_total * 12,
        "100 -> 1000 elements: {small:?} -> {large:?} replay visits"
    );
}

#[test]
fn numeric_list_check_time_scales_with_element_count() {
    let source = |elements| {
        format!(
            "probe = to_tensor([{}])",
            vec!["0.0f64"; elements].join(", ")
        )
    };
    let small_source = source(100);
    let large_source = source(1000);
    check_surf(&small_source).expect("warm the checker before timing");

    let mut small_times = [0; 3];
    let mut large_times = [0; 3];
    for index in 0..3 {
        let start = Instant::now();
        check_surf(&small_source).expect("100-element list checks");
        small_times[index] = start.elapsed().as_nanos();

        let start = Instant::now();
        check_surf(&large_source).expect("1000-element list checks");
        large_times[index] = start.elapsed().as_nanos();
    }
    small_times.sort_unstable();
    large_times.sort_unstable();

    assert!(
        large_times[1] <= small_times[1] * 30,
        "100 -> 1000 elements took {small_times:?} -> {large_times:?} ns"
    );
}

#[test]
fn callable_list_replay_work_is_linear() {
    let small = callable_list_replay_work(100);
    let large = callable_list_replay_work(200);
    let small_total: usize = small.iter().sum();
    let large_total: usize = large.iter().sum();
    assert!(small_total > 0);
    assert!(
        large_total <= small_total * 3,
        "100 -> 200 elements: {small:?} -> {large:?} replay visits"
    );
}

#[test]
fn nested_numeric_list_replay_work_is_linear() {
    let small = nested_list_replay_work(20, 20);
    let large = nested_list_replay_work(40, 20);
    let small_total: usize = small.iter().sum();
    let large_total: usize = large.iter().sum();
    assert!(small_total > 0);
    assert!(
        large_total <= small_total * 3,
        "20 -> 40 rows: {small:?} -> {large:?} replay visits"
    );
}

#[test]
fn delayed_list_equality_still_rejects_mixed_elements() {
    assert!(check_surf("probe = [0.0f64, true]").is_err());
    assert!(check_surf("probe = to_tensor([0.0f64, true])").is_err());
    assert!(check_surf("probe = to_tensor([[0.0f64], [true]])").is_err());
}

#[test]
fn list_equality_cannot_choose_an_unresolved_grad_parameter() {
    let prefix =
        "def main() = {\n g = grad(fn (w) -> 1.0f32)\n known = grad(fn (w: f32) -> 1.0f32)\n";
    assert!(check_surf(&format!("{prefix} [g, known]\n}}")).is_err());
    assert!(check_surf(&format!("{prefix} value = g(2.0f32)\n [g, known]\n}}")).is_ok());
}

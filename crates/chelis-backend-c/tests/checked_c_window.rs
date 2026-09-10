//! [05-OP-33]/[05-RWIN-1]: window geometry is checked before storage.
const SOURCE: &str = include_str!("../src/emit.rs");
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let tail = &source[start..];
    &tail[..tail.find("\n    fn ").unwrap_or(tail.len())]
}
fn checked(source: &str) -> bool {
    let Some(_) = source.find("fn emit_window_plan(") else {
        return false;
    };
    let plan = method(source, "emit_window_plan");
    [
        "chelis_tensor_window_plan(",
        "emit_runtime_dim_sites(",
        "chelis_window_check_target(",
        "chelis_window_count(",
    ]
    .iter()
    .all(|token| plan.contains(token))
        && ["emit_reduce_window", "emit_reduce_window_grad"]
            .iter()
            .all(|name| {
                let body = method(source, name);
                body.find("emit_window_plan(")
                    .is_some_and(|p| body.find("emit_slot_wrapper(").is_some_and(|s| p < s))
                    && body.contains("chelis_window_index(")
                    && body.contains("chelis_window_plan_release(")
                    && ![
                        ".product()",
                        "chelis_flat_to_indices(",
                        "chelis_indices_to_flat(",
                        "full_indices[",
                        "out_indices[",
                        "for (int __w",
                    ]
                    .iter()
                    .any(|token| body.contains(token))
            })
        && method(source, "emit_reduce_window_grad")
            .find("chelis_window_check_tensor(")
            .is_some_and(|p| {
                p < method(source, "emit_reduce_window_grad")
                    .find("emit_slot_wrapper(")
                    .unwrap()
            })
}
#[test]
fn windows_delegate_complete_geometry_before_storage() {
    assert!(checked(SOURCE));
}
#[test]
fn window_projection_and_submission_bypass_controls_fail() {
    for token in [
        "chelis_tensor_window_plan(",
        "chelis_window_check_target(",
        "chelis_window_count(",
        "chelis_window_check_tensor(",
        "chelis_window_index(",
        "chelis_window_plan_release(",
    ] {
        assert!(!checked(&SOURCE.replace(token, "REMOVED(")), "{token}");
    }
}

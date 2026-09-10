//! [05-OP-33]/[05-MOV-1]: movement loops project indices without rank scratch.
const SOURCE: &str = include_str!("../src/emit.rs");
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("    fn {name}(")).expect(name);
    let rest = &source[start..];
    &rest[..rest.find("\n    }\n").expect("method end") + 7]
}
fn checked(source: &str) -> bool {
    for (name, factory) in [
        ("emit_permute", "chelis_tensor_permute_plan("),
        ("emit_expand", "chelis_tensor_expand_plan("),
        ("emit_pad", "self.emit_affine_plan("),
        ("emit_shrink", "self.emit_affine_plan("),
        ("emit_stride", "self.emit_affine_plan("),
    ] {
        let body = method(source, name);
        let before = body.find(factory);
        let allocate = body.find("self.emit_slot_wrapper(");
        if !before.zip(allocate).is_some_and(|(a, b)| a < b) {
            return false;
        }
        for required in [
            "chelis_movement_count(",
            "chelis_movement_index(",
            "chelis_movement_plan_release(",
        ] {
            if !body.contains(required) {
                return false;
            }
        }
        for retired in [
            "coordinates[",
            "out_indices[",
            "in_indices[",
            "chelis_tensor_unravel_index(",
            "chelis_tensor_flat_index(",
            "chelis_tensor_affine_index(",
        ] {
            if body.contains(retired) {
                return false;
            }
        }
    }
    true
}
#[test]
fn all_five_movement_emitters_use_checked_plans_without_rank_scratch() {
    assert!(checked(SOURCE));
}
#[test]
fn missing_movement_projection_or_release_fails_the_adoption_control() {
    for removed in [
        "chelis_tensor_permute_plan(",
        "chelis_tensor_expand_plan(",
        "self.emit_affine_plan(",
        "chelis_movement_count(",
        "chelis_movement_index(",
        "chelis_movement_plan_release(",
    ] {
        assert!(!checked(&SOURCE.replace(removed, "REMOVED(")), "{removed}");
    }
}

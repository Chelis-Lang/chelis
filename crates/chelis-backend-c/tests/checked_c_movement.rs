//! Bounded permutation/expansion adoption; native cases live in exec_compile.
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("    fn {name}(")).expect(name);
    let rest = &source[start..];
    &rest[..rest.find("\n    }\n").expect("method end") + 7]
}

fn validate(source: &str) -> Result<(), String> {
    let shape = method(source, "emit_affine_plan");
    for required in [
        "chelis_tensor_affine_plan(",
        "self.emit_runtime_dim_sites(",
        "chelis_movement_check_target(",
    ] {
        if !shape.contains(required) {
            return Err(format!("missing affine shape obligation: {required}"));
        }
    }
    for name in ["emit_pad", "emit_shrink", "emit_stride"] {
        let body = method(source, name);
        let check = body
            .find("self.emit_affine_plan(")
            .ok_or_else(|| format!("{name}: missing affine shape validation"))?;
        if check >= body.find("self.emit_slot_wrapper(").expect("allocation") {
            return Err(format!("{name}: late affine shape validation"));
        }
        for retired in [
            "chelis_flat_to_indices(",
            "chelis_indices_to_flat(",
            "src_indices[",
            "in_indices[",
            "dst_indices[",
        ] {
            if body.contains(retired) {
                return Err(format!("{name}: raw affine coordinate authority"));
            }
        }
        if !body.contains("chelis_movement_index(") || !body.contains("chelis_movement_count(") {
            return Err(format!("{name}: missing checked affine coordinates"));
        }
    }
    for (name, check) in [
        ("emit_permute", "chelis_tensor_permute_plan("),
        ("emit_expand", "chelis_tensor_expand_plan("),
    ] {
        let body = method(source, name);
        let checked = body
            .find(check)
            .ok_or_else(|| format!("{name}: missing movement validation"))?;
        if checked >= body.find("self.emit_slot_wrapper(").expect("allocation") {
            return Err(format!("{name}: late movement validation"));
        }
        for retired in [
            "chelis_flat_to_indices(",
            "chelis_indices_to_flat(",
            "int64_t in_indices[",
            "int64_t out_indices[",
        ] {
            if body.contains(retired) {
                return Err(format!("{name}: raw coordinate authority"));
            }
        }
        for checked in [
            "chelis_movement_index(",
            "chelis_movement_count(",
            "chelis_movement_check_target(",
            "chelis_movement_plan_release(",
        ] {
            if !body.contains(checked) {
                return Err(format!("{name}: missing checked coordinates"));
            }
        }
    }
    Ok(())
}

#[test]
fn permutation_and_expansion_use_checked_metadata_before_allocation() {
    validate(include_str!("../src/emit.rs")).unwrap();
}

#[test]
fn movement_control_rejects_removed_checks_and_raw_index_helpers() {
    let source = include_str!("../src/emit.rs");
    validate(source).unwrap();
    for removed in [
        "chelis_tensor_affine_plan(",
        "chelis_movement_check_target(",
    ] {
        assert!(
            validate(&source.replace(removed, "erased_obligation"))
                .unwrap_err()
                .contains("missing affine shape obligation")
        );
    }
    for name in ["emit_pad", "emit_shrink", "emit_stride"] {
        let body = method(source, name);
        let missing = body.replace("self.emit_affine_plan(", "self.retired_affine_shape(");
        assert!(
            validate(&source.replace(body, &missing))
                .unwrap_err()
                .contains("missing affine shape validation")
        );
        let allocation = "self.emit_slot_wrapper(id, ty);";
        let late = body.replace(allocation, "").replacen(
            "        let a = inputs[0].0;",
            &format!("        {allocation}\n        let a = inputs[0].0;"),
            1,
        );
        assert!(
            validate(&source.replace(body, &late))
                .unwrap_err()
                .contains("late affine shape validation")
        );
    }
    for name in ["emit_permute", "emit_expand"] {
        let body = method(source, name);
        let allocation = "self.emit_slot_wrapper(id, ty);";
        let late = body.replace(allocation, "").replacen(
            "        let a = inputs[0].0;",
            &format!("        {allocation}\n        let a = inputs[0].0;"),
            1,
        );
        assert!(
            validate(&source.replace(body, &late))
                .unwrap_err()
                .contains("late movement validation")
        );
    }
    for (from, to, reason) in [
        (
            "chelis_tensor_permute_plan(",
            "retired_check(",
            "missing movement validation",
        ),
        (
            "chelis_tensor_expand_plan(",
            "retired_check(",
            "missing movement validation",
        ),
        (
            "chelis_movement_index(t{id}_movement,",
            "chelis_indices_to_flat(in_indices, t{a}_strides, t{a}_rank); chelis_movement_index(t{id}_movement,",
            "coordinate authority",
        ),
        (
            "chelis_movement_plan_release(t{id}_movement);",
            "int64_t in_indices[1]; chelis_movement_plan_release(t{id}_movement);",
            "coordinate authority",
        ),
    ] {
        assert!(source.contains(from), "production mutation anchor {from}");
        assert!(
            validate(&source.replace(from, to))
                .unwrap_err()
                .contains(reason)
        );
    }
}

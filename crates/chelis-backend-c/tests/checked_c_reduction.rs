//! The checked plan must dominate allocation in every migrated reduction emitter.
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let rest = &source[source.find(&format!("    fn {name}(")).expect(name)..];
    &rest[..rest.find("\n    }\n").expect("method end") + 7]
}

fn validate(source: &str) -> Result<(), String> {
    for name in [
        "emit_count",
        "emit_reduce_sum_general",
        "emit_reduce_extreme",
        "emit_reduce_simple",
        "emit_reduce_argcmp",
        "emit_fused_reduce",
    ] {
        let body = method(source, name);
        let check = body
            .find("self.emit_reduction_plan(")
            .ok_or(format!("{name}: missing plan"))?;
        if check >= body.find("self.emit_slot_wrapper(").expect("allocation") {
            return Err(format!("{name}: late plan"));
        }
        if !body.contains("self.projection_index(id")
            || !body.contains("chelis_reduction_plan_release(")
        {
            return Err(format!("{name}: missing checked index or release"));
        }
        for raw in [
            "chelis_flat_to_indices(",
            "chelis_indices_to_flat(",
            "malloc(",
            "full_indices[",
            "__count_n_{id} *=",
            "chelis_reduction_index(",
        ] {
            if body.contains(raw) {
                return Err(format!("{name}: raw metadata authority"));
            }
        }
    }
    let max = method(source, "emit_reduce_max");
    if !max.contains("self.emit_reduce_extreme(") {
        return Err("emit_reduce_max: missing checked extrema delegation".into());
    }
    let dispatch = method(source, "emit_node");
    for required in [
        "self.emit_reduce_max(id, *axis",
        "self.emit_reduce_extreme(id, *axis",
    ] {
        if !dispatch.contains(required) {
            return Err(format!("emit_node: missing checked dispatch {required}"));
        }
    }
    let wrapper = method(source, "emit_sum_level");
    if !wrapper.contains("self.emit_sum_level_lanes(id, axis_size, precision, false);")
        || ["malloc(", "sizeof(", "chelis_alloc("]
            .iter()
            .any(|raw| wrapper.contains(raw))
    {
        return Err("unchecked Sum scratch delegation".into());
    }
    let scratch = method(source, "emit_sum_level_lanes");
    for required in [
        "__sum_n_{id} = {axis_size};",
        "if blocked {",
        "__sum_shape_{id}[2] = {{ __sum_n_{id}, 8 }};",
        "chelis_alloc(2, __sum_shape_{id}, {dtype})",
        "} else {",
        "chelis_alloc(1, &__sum_n_{id}, {dtype})",
        "chelis_tensor_begin_write(__sum_scratch_{id})",
        "chelis_tensor_write_view(__sum_guard_{id}).data",
    ] {
        if scratch.matches(required).count() != 1 {
            return Err(format!("unchecked Sum scratch: {required}"));
        }
    }
    if scratch.contains("malloc(")
        || scratch.contains("sizeof(")
        || scratch.matches("chelis_alloc(").count() != 2
    {
        return Err("unchecked Sum scratch owner".into());
    }
    let ordered = [
        "__sum_n_{id} = {axis_size};",
        "if blocked {",
        "__sum_shape_{id}[2] = {{ __sum_n_{id}, 8 }};",
        "chelis_alloc(2, __sum_shape_{id}, {dtype})",
        "} else {",
        "chelis_alloc(1, &__sum_n_{id}, {dtype})",
        "chelis_tensor_begin_write(__sum_scratch_{id})",
        "chelis_tensor_write_view(__sum_guard_{id}).data",
    ]
    .map(|token| scratch.find(token).unwrap());
    if !ordered.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("Sum scratch domain/allocation/write order".into());
    }
    let inlined = method(source, "emit_inlined_product_sum");
    let check = inlined
        .find("self.emit_reduction_plan(")
        .ok_or("missing inlined Sum plan")?;
    let output = inlined
        .find("self.emit_slot_wrapper(")
        .ok_or("missing inlined Sum output")?;
    let leaves = inlined
        .find("self.emit_sum_level_lanes(")
        .ok_or("missing inlined Sum scratch")?;
    if check >= output || output >= leaves || !inlined.contains("ty, \"CHELIS_REDUCE_SUM\", true)")
    {
        return Err("inlined Sum unchecked scratch domain".into());
    }
    let plan = method(source, "emit_reduction_plan");
    for required in [
        "chelis_tensor_reduction_plan(",
        "chelis_shape_reduction_plan(",
        "chelis_reduction_check_target(",
        "self.emit_runtime_dim_sites(",
        "chelis_reduction_check_scratch(",
        "self.emit_projection_terms(",
        "chelis_reduction_term(",
    ] {
        if !plan.contains(required) {
            return Err(format!("missing plan obligation {required}"));
        }
    }
    Ok(())
}

#[test]
fn reduction_consumers_require_checked_domains_before_allocation() {
    validate(include_str!("../src/emit.rs")).unwrap();
}

#[test]
fn bypassing_each_consumer_and_scratch_owner_is_rejected() {
    let source = include_str!("../src/emit.rs");
    validate(source).unwrap();
    for name in [
        "emit_count",
        "emit_reduce_sum_general",
        "emit_reduce_extreme",
        "emit_reduce_simple",
        "emit_reduce_argcmp",
        "emit_fused_reduce",
    ] {
        let body = method(source, name);
        for original in [
            "self.emit_reduction_plan(",
            "self.projection_index(id",
            "chelis_reduction_plan_release(",
        ] {
            assert!(
                validate(&source.replace(body, &body.replace(original, "removed("))).is_err(),
                "{name}: {original}"
            );
        }
        // A per-element runtime index call is rejected even beside the projection.
        let per_element = body.replacen(
            "self.projection_index(id",
            "chelis_reduction_index(); self.projection_index(id",
            1,
        );
        assert!(
            validate(&source.replace(body, &per_element)).is_err(),
            "{name}: per-element index"
        );
    }
    let max = method(source, "emit_reduce_max");
    assert!(
        validate(&source.replace(max, &max.replace("self.emit_reduce_extreme(", "removed(")))
            .is_err(),
        "emit_reduce_max: checked extrema delegation"
    );
    let dispatch = method(source, "emit_node");
    for required in [
        "self.emit_reduce_max(id, *axis",
        "self.emit_reduce_extreme(id, *axis",
    ] {
        assert!(
            validate(&source.replace(dispatch, &dispatch.replace(required, "removed("))).is_err(),
            "emit_node: checked reduction dispatch {required}"
        );
    }
    assert!(
        validate(&source.replace("chelis_alloc(1, &__sum_n_{id}", "malloc(1, &__sum_n_{id}"))
            .is_err()
    );
    let wrapper = method(source, "emit_sum_level");
    assert!(
        validate(&source.replace(
            wrapper,
            &wrapper.replace(
                "self.emit_sum_level_lanes(id, axis_size, precision, false);",
                "self.emit_sum_level_lanes(id, axis_size, precision, true);",
            )
        ))
        .is_err(),
        "scalar scratch must delegate to one lane"
    );
    let scratch = method(source, "emit_sum_level_lanes");
    for original in [
        "__sum_n_{id} = {axis_size};",
        "if blocked {",
        "__sum_shape_{id}[2] = {{ __sum_n_{id}, 8 }};",
        "chelis_alloc(2, __sum_shape_{id}, {dtype})",
        "chelis_alloc(1, &__sum_n_{id}, {dtype})",
        "chelis_tensor_begin_write(__sum_scratch_{id})",
        "chelis_tensor_write_view(__sum_guard_{id}).data",
    ] {
        assert!(
            validate(&source.replace(scratch, &scratch.replace(original, "removed("))).is_err(),
            "scratch owner: {original}"
        );
    }
    let reversed = scratch
        .replace(
            "__sum_shape_{id}[2] = {{ __sum_n_{id}, 8 }};",
            "__domain_placeholder__",
        )
        .replace(
            "chelis_alloc(2, __sum_shape_{id}, {dtype})",
            "__sum_shape_{id}[2] = {{ __sum_n_{id}, 8 }};",
        )
        .replace(
            "__domain_placeholder__",
            "chelis_alloc(2, __sum_shape_{id}, {dtype})",
        );
    assert!(
        validate(&source.replace(scratch, &reversed)).is_err(),
        "domain must precede vector allocation"
    );
    let inlined = method(source, "emit_inlined_product_sum");
    for original in [
        "self.emit_reduction_plan(",
        "self.emit_sum_level_lanes(",
        "ty, \"CHELIS_REDUCE_SUM\", true)",
    ] {
        assert!(
            validate(&source.replace(inlined, &inlined.replace(original, "removed("))).is_err(),
            "inlined scratch: {original}"
        );
    }
}

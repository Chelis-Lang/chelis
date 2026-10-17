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
    // chelis#3700: the canonical Sum tree streams through stack locals, so
    // none of its emitters allocates, and it holds one slot per level of an
    // admitted i64 leaf count.
    for name in [
        "emit_carry_tree_open",
        "emit_carry_tree_push",
        "emit_carry_tree_total",
        "emit_sum_store",
    ] {
        let body = method(source, name);
        if [
            "malloc(",
            "sizeof(",
            "chelis_alloc(",
            "chelis_tensor_begin_write(",
        ]
        .iter()
        .any(|raw| body.contains(raw))
        {
            return Err(format!("{name}: Sum tree scratch allocation"));
        }
    }
    let open = method(source, "emit_carry_tree_open");
    if !open.contains("{name}_level[64][8];") || !open.contains("{name}_level[64];") {
        return Err("Sum tree levels are not bounded by the i64 leaf count".into());
    }
    let inlined = method(source, "emit_inlined_product_sum");
    let check = inlined
        .find("self.emit_reduction_plan(")
        .ok_or("missing inlined Sum plan")?;
    let output = inlined
        .find("self.emit_slot_wrapper(")
        .ok_or("missing inlined Sum output")?;
    let leaves = inlined
        .find("self.emit_carry_tree_open(")
        .ok_or("missing inlined Sum tree")?;
    if check >= output || output >= leaves {
        return Err("inlined Sum unchecked tree domain".into());
    }
    let checked_domain = inlined[check..output]
        .split_whitespace()
        .collect::<String>()
        .replace(",)", ")");
    if !checked_domain.contains("ty,\"CHELIS_REDUCE_SUM\")") {
        return Err("inlined Sum unchecked tree domain".into());
    }
    let plan = method(source, "emit_reduction_plan");
    for required in [
        "chelis_tensor_reduction_plan(",
        "chelis_shape_reduction_plan(",
        "chelis_reduction_check_target(",
        "self.emit_runtime_dim_sites(",
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
    for name in [
        "emit_carry_tree_open",
        "emit_carry_tree_push",
        "emit_carry_tree_total",
        "emit_sum_store",
    ] {
        let body = method(source, name);
        for raw in ["malloc(", "chelis_alloc(", "chelis_tensor_begin_write("] {
            let allocating = body.replacen("{\n", &format!("{{\n        {raw});\n"), 1);
            assert!(
                validate(&source.replace(body, &allocating)).is_err(),
                "{name}: {raw}"
            );
        }
    }
    let open = method(source, "emit_carry_tree_open");
    assert!(
        validate(&source.replace(
            open,
            &open.replace("{name}_level[64];", "{name}_level[32];")
        ))
        .is_err(),
        "the tree must keep one level per bit of an i64 leaf count"
    );
    let inlined = method(source, "emit_inlined_product_sum");
    for original in ["self.emit_reduction_plan(", "self.emit_carry_tree_open("] {
        assert!(
            validate(&source.replace(inlined, &inlined.replace(original, "removed("))).is_err(),
            "inlined tree: {original}"
        );
    }
}

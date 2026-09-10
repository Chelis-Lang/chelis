//! The checked plan must dominate allocation in every migrated reduction emitter.
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let rest = &source[source.find(&format!("    fn {name}(")).expect(name)..];
    &rest[..rest.find("\n    }\n").expect("method end") + 7]
}

fn validate(source: &str) -> Result<(), String> {
    for name in [
        "emit_count",
        "emit_reduce_sum_general",
        "emit_reduce_max",
        "emit_reduce_max_reduced_f",
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
        if !body.contains("chelis_reduction_index(")
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
        ] {
            if body.contains(raw) {
                return Err(format!("{name}: raw metadata authority"));
            }
        }
    }
    let scratch = method(source, "emit_sum_level");
    if scratch.contains("malloc(")
        || scratch.contains("sizeof(")
        || !scratch.contains("chelis_alloc(")
    {
        return Err("unchecked Sum scratch".into());
    }
    let plan = method(source, "emit_reduction_plan");
    for required in [
        "chelis_tensor_reduction_plan(",
        "chelis_shape_reduction_plan(",
        "chelis_reduction_check_target(",
        "self.emit_runtime_dim_sites(",
        "chelis_reduction_check_scratch(",
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
        "emit_reduce_max",
        "emit_reduce_max_reduced_f",
        "emit_reduce_simple",
        "emit_reduce_argcmp",
        "emit_fused_reduce",
    ] {
        let body = method(source, name);
        for original in [
            "self.emit_reduction_plan(",
            "chelis_reduction_index(",
            "chelis_reduction_plan_release(",
        ] {
            assert!(
                validate(&source.replace(body, &body.replace(original, "removed("))).is_err(),
                "{name}: {original}"
            );
        }
    }
    assert!(
        validate(&source.replace("chelis_alloc(1, &__sum_n_{id}", "malloc(1, &__sum_n_{id}"))
            .is_err()
    );
}

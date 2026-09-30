//! Checked sparse iteration must precede every affected allocation and access.
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let rest = &source[source.find(&format!("    fn {name}(")).expect(name)..];
    &rest[..rest.find("\n    }\n").unwrap() + 7]
}
fn sparse_arm<'a>(source: &'a str, name: &str) -> &'a str {
    let emit_node = method(source, "emit_node");
    let start = &emit_node[emit_node
        .find(&format!("            RiscOp::{name} {{"))
        .expect(name)..];
    let end = "\n            }\n";
    &start[..start.find(end).expect(name) + end.len()]
}
fn validate(source: &str) -> Result<(), String> {
    for (name, op, batch) in [
        (
            "Gather",
            "SparseEmission::Gather",
            "batch_rank: *batch_rank",
        ),
        (
            "ScatterAdd",
            "SparseEmission::Add",
            "batch_rank: *batch_rank",
        ),
        (
            "Scatter",
            "SparseEmission::Replace",
            "batch_rank: *batch_rank",
        ),
        (
            "ScatterElements",
            "SparseEmission::Elements",
            "batch_rank: 0",
        ),
    ] {
        let body = sparse_arm(source, name);
        if !body.contains("self.emit_sparse_checked(")
            || !body.contains("axis: *axis")
            || !body.contains(batch)
            || !body.contains(op)
        {
            return Err(format!("{name}: missing exact checked delegation"));
        }
    }
    let body = method(source, "emit_sparse_checked");
    let alloc = body.find("self.emit_slot_wrapper(").ok_or("allocation")?;
    for check in [
        "chelis_tensor_sparse_plan(",
        "chelis_sparse_check_target(",
        "chelis_sparse_count(",
    ] {
        if body.find(check).ok_or(check)? >= alloc {
            return Err(format!("late {check}"));
        }
    }
    for check in [
        "chelis_sparse_index_slot(",
        "chelis_sparse_data_index(",
        "chelis_sparse_plan_release(",
        "_byte_capacity != 0",
    ] {
        if !body.contains(check) {
            return Err(format!("missing {check}"));
        }
    }
    for raw in [
        "dim_product_expr",
        "sizeof(",
        "_before",
        "_after",
        "_stride *=",
        "abort();",
    ] {
        if body.contains(raw) {
            return Err(format!("raw sparse authority {raw}"));
        }
    }
    Ok(())
}

#[test]
fn sparse_dag_consumers_delegate_to_one_checked_domain() {
    validate(include_str!("../src/emit.rs")).unwrap();
}

#[test]
fn sparse_dag_dispatch_mutations_are_rejected() {
    let source = include_str!("../src/emit.rs");
    validate(source).unwrap();
    for name in ["Gather", "ScatterAdd", "Scatter", "ScatterElements"] {
        let arm = sparse_arm(source, name);
        for (from, to) in [
            ("self.emit_sparse_checked(", "self.emit_sparse_unchecked("),
            ("axis: *axis", "axis: 0"),
            ("batch_rank: *batch_rank", "batch_rank: 0"),
        ] {
            if !arm.contains(from) {
                continue;
            }
            let changed = arm.replacen(from, to, 1);
            assert!(
                validate(&source.replacen(arm, &changed, 1)).is_err(),
                "{name}: {from}"
            );
        }
    }
}

fn validate_host(source: &str) -> Result<(), String> {
    let body = method(source, "assign_sparse_summary");
    let allocation = body.find("= chelis_alloc(").ok_or("allocation")?;
    for check in [
        "chelis_tensor_sparse_plan(",
        "chelis_sparse_check_target(",
        "chelis_sparse_count(",
    ] {
        if body.find(check).ok_or(check)? >= allocation {
            return Err(format!("late {check}"));
        }
    }
    let loop_body = method(source, "emit_sparse_summary_body");
    for check in [
        "chelis_sparse_index_slot(",
        "chelis_sparse_data_index(",
        "chelis_tensor_byte_count(",
        "chelis_sparse_plan_release(",
    ] {
        if !loop_body.contains(check) {
            return Err(format!("missing {check}"));
        }
    }
    for raw in [
        "sparse_dim_product(",
        "sizeof(",
        "_before",
        "_after",
        "abort();",
    ] {
        if loop_body.contains(raw) {
            return Err(format!("raw sparse host authority {raw}"));
        }
    }
    Ok(())
}

#[test]
fn sparse_host_summaries_use_checked_submission_and_indexing() {
    let source = include_str!("../src/host_emit.rs");
    validate_host(source).unwrap();
    for obligation in [
        "chelis_tensor_sparse_plan(",
        "chelis_sparse_check_target(",
        "chelis_sparse_count(",
        "chelis_sparse_index_slot(",
        "chelis_sparse_data_index(",
        "chelis_tensor_byte_count(",
        "chelis_sparse_plan_release(",
    ] {
        let assignment = method(source, "assign_sparse_summary");
        let body = if assignment.contains(obligation) {
            assignment
        } else {
            method(source, "emit_sparse_summary_body")
        };
        assert!(body.contains(obligation));
        let changed = body.replacen(obligation, "bypassed_obligation(", 1);
        assert!(
            validate_host(&source.replacen(body, &changed, 1)).is_err(),
            "{obligation}"
        );
    }
}

#[test]
fn bypassing_sparse_projection_or_submission_is_rejected() {
    let source = include_str!("../src/emit.rs");
    validate(source).unwrap();
    for obligation in [
        "chelis_tensor_sparse_plan(",
        "chelis_sparse_check_target(",
        "chelis_sparse_count(",
        "chelis_sparse_index_slot(",
        "chelis_sparse_data_index(",
        "chelis_sparse_plan_release(",
        "_byte_capacity != 0",
    ] {
        let body = method(source, "emit_sparse_checked");
        assert!(body.contains(obligation));
        let changed = body.replacen(obligation, "bypassed_obligation(", 1);
        assert!(
            validate(&source.replacen(body, &changed, 1)).is_err(),
            "{obligation}"
        );
    }
}

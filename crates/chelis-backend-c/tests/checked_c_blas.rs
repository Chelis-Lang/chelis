//! [05-OP-33]: generated matrix submissions delegate domains to runtime metadata.
const DAG: &str = include_str!("../src/emit.rs");
const HOST: &str = include_str!("../src/host_emit.rs");
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let tail = &source[start..];
    let end = tail.find("\n    fn ").unwrap_or(tail.len());
    &tail[..end]
}
fn checked_dag(source: &str) -> bool {
    let plan = method(source, "emit_matmul_plan");
    [
        "chelis_tensor_matmul_plan(",
        "emit_runtime_dim_sites(",
        "chelis_matmul_check_target(",
        "chelis_matmul_check_vendor(",
        "chelis_matmul_check_scratch(",
    ]
    .iter()
    .all(|required| plan.contains(required))
        && ["emit_blas_matmul", "emit_blas_matmul_reduced_f"]
            .iter()
            .all(|name| {
                let body = method(source, name);
                body.find("emit_matmul_plan(").is_some_and(|plan| {
                    body.find("emit_slot_wrapper(")
                        .is_some_and(|alloc| plan < alloc)
                }) && body.contains("chelis_matmul_index(")
                    && body.contains("chelis_matmul_plan_release(")
                    && !body.contains("malloc(")
                    && !body.contains("_offset +=")
                    // A batch's matrix is one contiguous run: its checked first
                    // index locates every element without a per-element call.
                    && !body.contains("chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_i)")
            })
}
#[test]
fn blas_dag_paths_use_checked_domains_before_storage() {
    assert!(checked_dag(DAG));
}
#[test]
fn blas_domain_bypass_controls_are_rejected() {
    for required in [
        "chelis_tensor_matmul_plan(",
        "chelis_matmul_check_target(",
        "chelis_matmul_check_vendor(",
        "chelis_matmul_check_scratch(",
        "chelis_matmul_index(",
        "chelis_matmul_plan_release(",
    ] {
        assert!(
            !checked_dag(&DAG.replace(required, "REMOVED(")),
            "{required}"
        );
    }
    let body = method(DAG, "emit_blas_matmul_reduced_f");
    let per_element = body.replacen(
        "chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0));",
        "chelis_scalar_from_bits(CHELIS_DTYPE_I64, t{id}_i));",
        1,
    );
    assert_ne!(per_element, body);
    assert!(!checked_dag(&DAG.replace(body, &per_element)));
}
#[test]
fn blas_host_summary_uses_checked_submission_and_offsets() {
    let body = method(HOST, "assign_blas_matmul_summary");
    let allocation = body.find("= chelis_alloc(").unwrap();
    for required in [
        "chelis_tensor_matmul_plan(",
        "chelis_matmul_check_target(",
        "chelis_matmul_check_vendor(",
    ] {
        assert!(
            body.find(required).is_some_and(|check| check < allocation),
            "{required}"
        );
    }
    for required in [
        "chelis_matmul_batch_count(",
        "chelis_matmul_index(",
        "chelis_matmul_plan_release(",
    ] {
        assert!(body.contains(required), "{required}");
    }
    assert!(!body.contains("_offset +=") && !body.contains("chelis_host_tensor_stride("));
}

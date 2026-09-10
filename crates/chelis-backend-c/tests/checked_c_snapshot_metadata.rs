//! C2.2: shape queries consume checked metadata before any storage repurpose.
#[test]
fn generated_metadata_uses_runtime_queries_without_rank_sized_stack_scratch() {
    let emit = include_str!("../src/emit.rs");
    let host = include_str!("../src/host_emit.rs");
    assert!(!emit.contains("int64_t t{id}_shape[t{id}_rank"));
    assert!(!emit.contains("int64_t t{id}_strides[t{id}_rank"));
    assert!(!host.contains("int64_t shape[rank > 0 ? rank : 1]"));
    assert!(host.contains("chelis_tensor_alloc_like("));
    let shape = emit
        .split("    fn emit_shape(")
        .nth(1)
        .unwrap()
        .split("    fn emit_const_tensor(")
        .next()
        .unwrap();
    let observe = shape
        .find("chelis_tensor_shape(t{a}, {axis})")
        .expect("checked extent observation");
    let submit = shape.find("self.emit_slot_wrapper(id, ty)").unwrap();
    assert!(
        observe < submit,
        "capture the extent before destination repurpose"
    );
}

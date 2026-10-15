//! C2.2: shape queries consume checked metadata before any storage repurpose.
fn method<'a>(source: &'a str, name: &str) -> Result<&'a str, &'static str> {
    let start = source
        .find(&format!("    fn {name}("))
        .ok_or("missing checked owner")?;
    let rest = &source[start..];
    let end = rest.find("\n    }\n").ok_or("missing method end")?;
    Ok(&rest[..end])
}
fn validate(emit: &str, host: &str) -> Result<(), &'static str> {
    for raw in [
        "int64_t t{id}_shape[t{id}_rank",
        "int64_t t{id}_strides[t{id}_rank",
    ] {
        if emit.contains(raw) {
            return Err("rank-sized scratch");
        }
    }
    for raw in [
        "int64_t shape[rank > 0 ? rank : 1]",
        "int64_t shape[rank + 1]",
    ] {
        if host.contains(raw) {
            return Err("host rank-sized scratch");
        }
    }
    if !host.contains("chelis_tensor_alloc_like(") {
        return Err("missing checked host allocation");
    }
    let shape = method(emit, "emit_shape")?;
    let observe = shape
        .find("self.tensor_extent(NodeId(a), axis)")
        .ok_or("checked extent observation")?;
    let submit = shape
        .find("self.emit_slot_wrapper(id, ty)")
        .ok_or("missing checked submission")?;
    if observe >= submit {
        return Err("capture before repurpose");
    }
    let extent = method(emit, "tensor_extent")?;
    for checked in [
        "self.emitted_metadata.contains(&node)",
        "chelis_metadata_plan_shape(t{}_metadata)[{axis}]",
        "chelis_tensor_shape(t{}, {axis})",
    ] {
        if !extent.contains(checked) {
            return Err("missing checked extent route");
        }
    }
    Ok(())
}
#[test]
fn generated_metadata_uses_runtime_queries_without_rank_sized_stack_scratch() {
    validate(
        include_str!("../src/emit.rs"),
        include_str!("../src/host_emit.rs"),
    )
    .unwrap();
}
#[test]
fn bypassing_shape_capture_or_checked_metadata_is_rejected() {
    let emit = include_str!("../src/emit.rs");
    let host = include_str!("../src/host_emit.rs");
    validate(emit, host).unwrap();
    for (owner, anchor) in [
        ("emit_shape", "self.tensor_extent(NodeId(a), axis)"),
        ("tensor_extent", "self.emitted_metadata.contains(&node)"),
        (
            "tensor_extent",
            "chelis_metadata_plan_shape(t{}_metadata)[{axis}]",
        ),
        ("tensor_extent", "chelis_tensor_shape(t{}, {axis})"),
    ] {
        let body = method(emit, owner).unwrap();
        assert_eq!(body.matches(anchor).count(), 1, "unique mutation {anchor}");
        assert!(
            validate(&emit.replace(body, &body.replace(anchor, "removed")), host).is_err(),
            "{anchor}"
        );
    }
    let shape = method(emit, "emit_shape").unwrap();
    let reordered = shape
        .replace("self.emit_slot_wrapper(id, ty)", "removed")
        .replacen(") {", ") { self.emit_slot_wrapper(id, ty);", 1);
    assert!(validate(&emit.replace(shape, &reordered), host).is_err());
    assert!(validate(&format!("{emit}\nint64_t t{{id}}_shape[t{{id}}_rank"), host).is_err());
}

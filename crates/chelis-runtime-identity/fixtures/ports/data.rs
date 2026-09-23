use chelis_runtime_identity::*;
pub fn accept(recipe: &RuntimeRecipe, captured: &[CapturedInput], roots: &[InventoryRoot]) {
    let _ = plan_inputs(roots);
    let descriptor = derive_descriptor(recipe, captured).unwrap();
    let encoded = encode_record(&descriptor, RecordKind::Runtime).unwrap();
    assert_eq!(decode_record(&encoded, RecordKind::Runtime).unwrap(), descriptor);
    let mut hasher = ContentHasher::new();
    hasher.update(b"captured bytes");
    assert_eq!(hasher.finish(), hash_bytes(b"captured bytes"));
}

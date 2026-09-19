//! Deserialization is a construction route into the key-sorted annotation
//! storage that no other test walks: every hand-built wire fixture in this
//! crate is single-entry or a mutation of the encoder's own output, and the
//! encoder sorts by key spelling, so nothing hands the decoder a scrambled
//! multi-entry sequence and checks the order that comes back.
//!
//! From the #2210 review round (rt-2210-round1), landed at its request as
//! the one probe of its wider set that the merged five do not overlap. The
//! assertions are the reviewer's; one debug `eprintln!` of the wire shape
//! was dropped.
//!
//! It is a regression test, not a disposition lock. Mutating
//! `Metadata::insert`'s `core.insert(index, value)` to `core.push(value)`
//! fails it with `deserialized order: [SurfPath, PropertySourceId, Doc,
//! ChelisRole]`, the reversed wire sequence surviving into storage.
use chelis_deep::Span;
use chelis_deep::annotations::{Metadata, MetadataKey, MetadataValue, Spanned};

fn span() -> Span {
    Span::new(0, 0)
}
fn text(v: &str) -> Spanned<String> {
    Spanned::new(v.to_string(), span())
}
fn keys(m: &Metadata) -> Vec<MetadataKey> {
    m.values().map(|v| v.key()).collect()
}
fn is_sorted(m: &Metadata) -> bool {
    keys(m).windows(2).all(|p| p[0] < p[1])
}

/// Deserialization is a construction route the order lock does not walk.
/// Feed wire entries in reverse key order and require sorted storage back.
#[test]
fn deserialization_yields_sorted_storage_from_scrambled_wire() {
    let sample = Metadata::try_from_values([
        MetadataValue::ChelisRole(text("r")),
        MetadataValue::PropertySourceId(text("s")),
        MetadataValue::SurfPath(text("p")),
        MetadataValue::Doc(text("d")),
    ])
    .unwrap();
    let mut wire = serde_json::to_value(&sample).unwrap();
    // Reverse the entry order the encoder produced, so the decoder is handed
    // a scrambled sequence.
    wire["entries"]
        .as_array_mut()
        .expect("entries is an array")
        .reverse();
    let decoded: Metadata = match serde_json::from_value(wire.clone()) {
        Ok(m) => m,
        Err(e) => panic!("probe could not reach the decoder: {e}; wire = {wire}"),
    };
    assert_eq!(decoded.values().count(), 4, "probe decoded nothing");
    assert!(
        is_sorted(&decoded),
        "deserialized order: {:?}",
        keys(&decoded)
    );

    // Round-trip equality, which PartialEq decides by pairwise values().
    let built = Metadata::try_from_values([
        MetadataValue::ChelisRole(text("r")),
        MetadataValue::PropertySourceId(text("s")),
        MetadataValue::SurfPath(text("p")),
        MetadataValue::Doc(text("d")),
    ])
    .unwrap();
    assert_eq!(decoded, built);
    let reencoded: Metadata =
        serde_json::from_slice(&serde_json::to_vec(&decoded).unwrap()).unwrap();
    assert_eq!(reencoded, built);
    assert!(is_sorted(&reencoded));
}

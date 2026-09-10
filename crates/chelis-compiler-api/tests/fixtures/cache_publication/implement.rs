#[derive(serde::Serialize, serde::Deserialize)]
struct NewPayload {
    value: f64,
}
impl cache_envelope::CachePayload for NewPayload {
    const FORMAT_VERSION: u32 = 1;
    const KEY_DOMAIN: &'static [u8] = b"new";
}

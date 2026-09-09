#[derive(serde::Serialize, serde::Deserialize)]
struct NewPayload {
    value: f64,
}
impl cache_envelope::sealed::Sealed for NewPayload {}

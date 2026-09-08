use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
#[serde(tag = "span", rename_all = "snake_case")]
pub enum DiagnosticSpan {
    Range { offset: u64, len: u64 },
    Point { offset: u64 },
}
#[derive(Serialize, Deserialize)]
pub struct Wrapped {
    #[serde(rename = "where")]
    pub location: DiagnosticSpan,
}
pub struct Manual(pub f64);
impl Serialize for Manual {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

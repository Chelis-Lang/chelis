use std::collections::BTreeMap;
use std::sync::OnceLock;

fn rows() -> &'static BTreeMap<String, String> {
    static ROWS: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    ROWS.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../snapshots/issue_1870__reviewed__unsupported_wording.snap"
        ))
        .expect("generated reviewed unsupported wording snapshot")
    })
}

pub fn stderr(name: &str) -> &'static str {
    rows()
        .get(name)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("missing reviewed unsupported wording row `{name}`"))
}

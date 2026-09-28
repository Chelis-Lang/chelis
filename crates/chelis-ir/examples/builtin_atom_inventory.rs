//! Read the compiled declarations, independently of normative registrations.
use chelis_ir::dag::RiscAtomIdentity;

fn main() {
    let builtins = chelis_types::builtin_discovery::builtin_semantic_identities()
        .expect("complete typed builtin discovery");
    let risc: Vec<_> = RiscAtomIdentity::ALL
        .iter()
        .map(|id| format!("Numeric:{}:TableA", id.as_str()))
        .collect();
    let value = serde_json::json!({"builtins": builtins, "risc": risc});
    println!("{value}");
}

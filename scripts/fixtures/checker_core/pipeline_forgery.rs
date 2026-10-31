use chelis_compiler_api::pipeline::{
    CheckedLibrary, LoweredParts, NamedRoots, SemanticRejection, lower_library,
};

fn expose_checked(rejection: SemanticRejection) {
    let _ = rejection.checked();
}

fn consume_declared_roots(_: NamedRoots) {}

fn swap_root_map_roles(parts: LoweredParts) {
    consume_declared_roots(parts.forward_node_index);
}

fn lower_unbound_checked_program(library: &CheckedLibrary) {
    let _ = lower_library(library.program());
}

fn replace_lowered_payload(library: &CheckedLibrary) {
    let mut lowered = lower_library(library).expect("library must lower");
    lowered.dag = lowered.dag().clone();
}

fn transplant_lowered_identity(library: &CheckedLibrary) {
    let mut lowered = lower_library(library).expect("library must lower");
    let replacement = lower_library(library).expect("library must lower");
    lowered.library_proof_id = replacement.library_proof_id();
}

fn main() {
    let _ = expose_checked;
    let _ = swap_root_map_roles;
    let _ = lower_unbound_checked_program;
    let _ = replace_lowered_payload;
    let _ = transplant_lowered_identity;
}

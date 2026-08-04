use chelis_compiler_api::pipeline::{
    LoweredParts, NamedRoots, SemanticRejection,
};

fn expose_checked(rejection: SemanticRejection) {
    let _ = rejection.checked();
}

fn consume_declared_roots(_: NamedRoots) {}

fn swap_root_map_roles(parts: LoweredParts) {
    consume_declared_roots(parts.forward_node_index);
}

fn main() {
    let _ = expose_checked;
    let _ = swap_root_map_roles;
}

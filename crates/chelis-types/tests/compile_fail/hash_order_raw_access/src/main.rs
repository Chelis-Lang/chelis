use chelis_types::unify::Subst;

fn raw_iteration_is_rejected(subst: &Subst) {
    let _ = subst.deferred_expand_store_compile_probe().keys();
    let _ = subst.deferred_reshape_store_compile_probe().iter_mut();
}

fn main() {}

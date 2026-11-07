use super::*;

#[test]
fn provider_scope_restores_after_expected_failure() {
    let state = OperationState::default();
    let result = state
        .with_native_provider_call(Some(("provider".into(), "symbol".into())), || {
            Err::<(), _>("refusal")
        });
    assert_eq!(result, Err("refusal"));
    assert_eq!(state.active_native_provider_call(), None);
}

#[test]
fn nested_provider_scope_restores_outer_binding() {
    let state = OperationState::default();
    state.with_native_provider_call(Some(("outer".into(), "outer_symbol".into())), || {
        let result = state
            .with_native_provider_call(Some(("inner".into(), "inner_symbol".into())), || {
                Err::<(), _>("refusal")
            });
        assert_eq!(result, Err("refusal"));
        assert_eq!(
            state.active_native_provider_call(),
            Some(("outer".into(), "outer_symbol".into())),
        );
    });
    assert_eq!(state.active_native_provider_call(), None);
}

#[test]
fn independent_provider_states_do_not_share_bindings() {
    let first = OperationState::default();
    let second = OperationState::default();
    first.with_native_provider_call(Some(("first".into(), "symbol".into())), || {
        assert_eq!(second.active_native_provider_call(), None);
    });
}

#[test]
fn synchronized_fact_cell_retries_after_initializer_panic() {
    let shared = OnceLock::<Arc<String>>::new();
    let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        shared.get_or_init(|| panic!("interrupt fact initialization"));
    }));
    assert!(interrupted.is_err());
    assert!(shared.get().is_none());
    let published = shared.get_or_init(|| Arc::new("fact".into()));
    let reused = shared.get_or_init(|| panic!("published fact must be reused"));
    assert!(core::ptr::eq(published, reused));
    assert!(Arc::ptr_eq(published, reused));
}

#[test]
fn program_fact_caches_errors_before_later_queries() {
    let local = RefCell::new(None);
    let shared = OnceLock::new();
    let first = program_fact(&local, Some(&shared), || Err::<usize, _>("refusal"));
    let second = program_fact(&local, Some(&shared), || {
        panic!("cached error must be reused")
    });
    assert_eq!(first, Err("refusal"));
    assert_eq!(second, first);
    assert_eq!(shared.get(), Some(&first));
}

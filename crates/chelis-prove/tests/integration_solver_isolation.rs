//! #1740: ordinary-dependency libtest binaries contain every cvc5 solve.

mod support;

use chelis_prove::solver::SmtExpr;
use chelis_prove::tier_b::{SmtProperty, TierBResult, solve_property};
use chelis_prove::worker::test_support;

fn property(value: bool) -> SmtProperty {
    SmtProperty {
        variables: vec![],
        preconditions: vec![],
        postcondition: SmtExpr::BoolLit(value),
    }
}

#[cfg(feature = "smt")]
#[test]
fn integration_solves_observe_the_child_route_and_real_verdicts() {
    crate::support::isolate();
    let (proved, spawned) = test_support::observe_worker(|| solve_property(&property(true), 5_000));
    assert!(
        spawned,
        "observe an actual worker spawn, not only configuration"
    );
    assert_eq!(proved, TierBResult::Proved);
    let (disproved, spawned) =
        test_support::observe_worker(|| solve_property(&property(false), 5_000));
    assert!(spawned);
    assert!(
        matches!(disproved, TierBResult::Disproved(_)),
        "{disproved:?}"
    );
}

#[cfg(feature = "smt")]
#[test]
fn child_abort_panic_and_post_result_abort_fail_closed_then_next_solve_survives() {
    crate::support::isolate();
    use test_support::WorkerFault;
    for fault in [
        WorkerFault::Abort,
        WorkerFault::Panic,
        WorkerFault::AbortAfterResult,
    ] {
        let (result, spawned) = test_support::observe_worker(|| {
            test_support::with_worker_fault(fault, || solve_property(&property(true), 5_000))
        });
        assert!(spawned);
        let TierBResult::Error(reason) = result else {
            panic!("{fault:?}: {result:?}")
        };
        assert!(
            reason.contains("worker exited abnormally") && reason.contains("routes to Tier C"),
            "{reason}"
        );
        assert_eq!(solve_property(&property(true), 5_000), TierBResult::Proved);
    }
}

#[cfg(feature = "smt")]
#[test]
fn hung_child_is_killed_then_next_solve_survives() {
    crate::support::isolate();
    let (result, spawned) = test_support::observe_worker(|| {
        test_support::with_worker_fault(test_support::WorkerFault::Hang, || {
            solve_property(&property(true), 1)
        })
    });
    assert!(spawned);
    assert_eq!(result, TierBResult::Unknown);
    assert_eq!(solve_property(&property(true), 5_000), TierBResult::Proved);
}

#[cfg(feature = "smt")]
#[test]
fn parallel_engine_calls_use_independent_solver_children() {
    crate::support::isolate();
    use chelis_prove::{Cvc5Engine, DischargeEngine, Goal};
    std::thread::scope(|scope| {
        for expected in [true, false, true, false] {
            scope.spawn(move || {
                let (discharge, spawned) = test_support::observe_worker(|| {
                    Cvc5Engine::new().discharge(&Goal::smt(property(expected)), 5_000)
                });
                assert!(
                    spawned,
                    "engine-mediated solves must use the registered route"
                );
                if expected {
                    assert_eq!(discharge.result(), &TierBResult::Proved);
                } else {
                    assert!(matches!(discharge.result(), TierBResult::Disproved(_)));
                }
            });
        }
    });
}

#[cfg(feature = "smt")]
#[test]
fn conflicting_worker_registration_cannot_replace_the_binary_entry() {
    crate::support::isolate();
    for entry in ["", "another::worker"] {
        assert!(std::panic::catch_unwind(|| test_support::enable_isolation(entry)).is_err());
    }
    let (result, spawned) = test_support::observe_worker(|| solve_property(&property(true), 5_000));
    assert!(spawned);
    assert_eq!(result, TierBResult::Proved);
}

#[cfg(feature = "smt")]
#[test]
fn fault_scope_is_thread_local_and_restored_after_parent_panic() {
    crate::support::isolate();
    use test_support::WorkerFault;
    test_support::with_worker_fault(WorkerFault::Abort, || {
        let sibling = std::thread::spawn(|| solve_property(&property(true), 5_000));
        assert_eq!(
            sibling.join().expect("sibling survives"),
            TierBResult::Proved
        );
        assert!(matches!(
            solve_property(&property(true), 5_000),
            TierBResult::Error(_)
        ));
    });
    assert!(
        std::panic::catch_unwind(|| {
            test_support::with_worker_fault(WorkerFault::Abort, || panic!("parent test panic"));
        })
        .is_err()
    );
    assert_eq!(solve_property(&property(true), 5_000), TierBResult::Proved);
}

#[cfg(not(feature = "smt"))]
#[test]
fn solver_free_build_never_spawns_a_worker_or_claims_a_proof() {
    crate::support::isolate();
    let (result, spawned) = test_support::observe_worker(|| solve_property(&property(true), 5_000));
    assert!(!spawned);
    assert_eq!(result, TierBResult::Timeout);
}

fn check_setup(items: &[syn::Item]) -> Result<(), String> {
    for item in items {
        match item {
            syn::Item::Fn(function)
                if function
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident("test")) =>
            {
                let Some(syn::Stmt::Expr(syn::Expr::Call(call), Some(_))) =
                    function.block.stmts.first()
                else {
                    return Err(format!(
                        "{} must install isolation first",
                        function.sig.ident
                    ));
                };
                let syn::Expr::Path(path) = call.func.as_ref() else {
                    return Err("setup must be explicit".into());
                };
                let segments: Vec<_> = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                if segments != ["crate", "support", "isolate"] || !call.args.is_empty() {
                    return Err(format!(
                        "{} must call crate::support::isolate() first",
                        function.sig.ident
                    ));
                }
            }
            syn::Item::Mod(module) => {
                if let Some((_, items)) = &module.content {
                    check_setup(items)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[test]
fn every_integration_test_installs_its_single_worker_entry_before_test_work() {
    crate::support::isolate();
    for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/tests")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let file = syn::parse_file(&source).unwrap();
        let entries = file.items.iter().filter(|item| matches!(item, syn::Item::Mod(module) if module.ident == "support" && module.content.is_none())).count();
        assert_eq!(
            entries,
            1,
            "{} needs exactly one shared worker entry",
            path.display()
        );
        check_setup(&file.items).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }
}

#[test]
fn setup_guard_rejects_missing_late_and_nested_missing_activation() {
    crate::support::isolate();
    for source in [
        "#[test] fn missing() {}",
        "#[test] fn late() { solve(); crate::support::isolate(); }",
        "mod nested { #[test] fn missing() { solve(); } }",
    ] {
        assert!(check_setup(&syn::parse_file(source).unwrap().items).is_err());
    }
    assert!(
        check_setup(
            &syn::parse_file(
                "mod nested { #[test] fn ready() { crate::support::isolate(); solve(); } }"
            )
            .unwrap()
            .items
        )
        .is_ok()
    );
}

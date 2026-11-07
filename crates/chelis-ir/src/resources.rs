//! Native cancellation, profile clocks, and work observations for IR operations.

use chelis_ir_core::resources::{Observer, Operation, Resources};
use chelis_syntax_host::NativeStack;
use std::cell::Cell;
use std::time::Instant;

thread_local! {
    static PROGRAM_CONTEXT_PREPARATIONS: Cell<u64> = const { Cell::new(0) };
    static DECLARATION_BODY_VISITS: Cell<u64> = const { Cell::new(0) };
    static CONTRACT_NORMALIZATIONS: Cell<u64> = const { Cell::new(0) };
}

pub fn program_context_preparations() -> u64 {
    PROGRAM_CONTEXT_PREPARATIONS.with(Cell::get)
}

pub fn reset_program_context_preparations() {
    PROGRAM_CONTEXT_PREPARATIONS.with(|counter| counter.set(0));
}

pub fn declaration_body_visits() -> u64 {
    DECLARATION_BODY_VISITS.with(Cell::get)
}

pub fn contract_normalizations() -> u64 {
    CONTRACT_NORMALIZATIONS.with(Cell::get)
}

pub fn reset_contract_normalizations() {
    CONTRACT_NORMALIZATIONS.with(|counter| counter.set(0));
}

struct Profile {
    phase_start: Cell<Instant>,
    declaration_start: Cell<Option<Instant>>,
    previous_nodes: Cell<usize>,
}

struct NativeObserver {
    profile: Option<Profile>,
}

impl Observer for NativeObserver {
    fn ownership_reference_checks_enabled(&self) -> bool {
        cfg!(test)
    }

    fn phase(&self, label: &str) {
        if let Some(profile) = &self.profile {
            eprintln!(
                "lower_sub: {:>8.4}s {}",
                profile.phase_start.get().elapsed().as_secs_f64(),
                label
            );
            profile.phase_start.set(Instant::now());
        }
    }

    fn declaration_begin(&self, _name: &str) {
        if let Some(profile) = &self.profile {
            profile.declaration_start.set(Some(Instant::now()));
        }
    }

    fn declaration_end(&self, name: &str, nodes: usize) {
        if let Some(profile) = &self.profile
            && let Some(start) = profile.declaration_start.take()
        {
            let added = nodes.saturating_sub(profile.previous_nodes.replace(nodes));
            eprintln!(
                "lower_decl: {:>8.4}s nodes_added={:>5} dag_total={:>6} {}",
                start.elapsed().as_secs_f64(),
                added,
                nodes,
                name
            );
        }
    }

    fn program_context_prepared(&self) {
        PROGRAM_CONTEXT_PREPARATIONS.with(|counter| counter.set(counter.get() + 1));
    }

    fn declaration_body_visit(&self) {
        DECLARATION_BODY_VISITS.with(|counter| counter.set(counter.get() + 1));
    }

    fn contract_normalized(&self) {
        CONTRACT_NORMALIZATIONS.with(|counter| counter.set(counter.get() + 1));
    }

    fn record_hoist_nodes_enabled(&self) -> bool {
        crate::host::record_hoist_nodes_enabled()
    }

    fn record_hoist_nodes(&self, nodes: u64) {
        crate::host::observe_record_hoist_nodes(nodes);
    }

    fn refine_scope_entries(&self, entries: u64) {
        crate::host::observe_refine_scope_entries(entries);
    }

    fn terminal_name_comparisons(&self, comparisons: u64) {
        crate::host::observe_terminal_name_comparisons(comparisons);
    }

    fn host_summary_probe_build(&self) {
        crate::host::observe_host_summary_probe_build();
    }

    fn host_lowering_session(&self) {
        crate::host::observe_host_lowering_session();
    }

    fn def_effect_row_derived(&self) {
        crate::host::observe_def_effect_row_derived();
    }

    fn site_lookup_entries(&self, entries: u64) {
        crate::ownership::observe_site_lookup_entries(entries);
    }

    fn ownership_verification(&self) {
        crate::ownership::observe_ownership_verification();
    }

    fn ownership_lookup_entries(&self, entries: u64) {
        crate::ownership::observe_ownership_lookup_entries(entries);
    }

    fn ownership_reference_scan_entries(&self, entries: u64) {
        crate::ownership::observe_ownership_reference_scan_entries(entries);
    }

    fn sum_lowering_fallback(&self, operand: chelis_numeric_core::prim::Prim, message: &str) {
        eprintln!(
            "internal: sum lowering fallback (operand `{}`): {message}",
            operand.name()
        );
    }
}

#[cfg(test)]
struct TestOperationGuard<'a, 'r> {
    operation: &'a Operation<'r, NativeStack>,
}

#[cfg(test)]
impl Drop for TestOperationGuard<'_, '_> {
    fn drop(&mut self) {
        crate::lower::finish_test_operation(self.operation);
    }
}

/// Execute a complete native IR session with one borrowed operation.
/// The operation cannot escape its caller resources.
pub fn with_operation<R>(run: impl FnOnce(&Operation<'_, NativeStack>) -> R) -> R {
    chelis_types::linearity::with_free_variable_walk_work(|work| {
        chelis_types::infer::with_resources_on(&NativeStack, |checker| {
            let observer = NativeObserver {
                profile: std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
                    .filter(|value| value == "1")
                    .map(|_| Profile {
                        phase_start: Cell::new(Instant::now()),
                        declaration_start: Cell::new(None),
                        previous_nodes: Cell::new(0),
                    }),
            };
            let mut resources = Resources::new(checker.stack);
            resources.cancellation = checker.cancellation;
            resources.checker_policy = checker.policy;
            resources.checker_stack_bytes = checker.stack_bytes;
            resources.checker_generalize_sweep_oracle = checker.generalize_sweep_oracle;
            resources.observer = Some(&observer);
            resources.free_variable_walk_work = work;
            resources.run(|operation| {
                #[cfg(test)]
                crate::lower::configure_test_operation(operation);
                #[cfg(test)]
                let _guard = TestOperationGuard { operation };
                run(operation)
            })
        })
    })
}

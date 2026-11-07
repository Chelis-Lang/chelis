//! Caller services and isolated state for one IR operation.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use chelis_checker_core::linearity::FreeVariableWalkWork;
use chelis_checker_core::{Cancellation, CheckPolicy};
use chelis_deep::Expr;
use chelis_stack_services::StackServices;

/// Observations at existing profile and work-counter boundaries.
pub trait Observer {
    fn phase(&self, _label: &str) {}
    fn declaration_begin(&self, _name: &str) {}
    fn declaration_end(&self, _name: &str, _nodes: usize) {}
    fn program_context_prepared(&self) {}
    fn declaration_body_visit(&self) {}
    fn contract_normalized(&self) {}
    fn record_hoist_nodes_enabled(&self) -> bool {
        false
    }
    fn record_hoist_nodes(&self, _nodes: u64) {}
    fn refine_scope_entries(&self, _entries: u64) {}
    fn terminal_name_comparisons(&self, _comparisons: u64) {}
    fn host_summary_probe_build(&self) {}
    fn host_lowering_session(&self) {}
    fn def_effect_row_derived(&self) {}
    fn ownership_reference_checks_enabled(&self) -> bool {
        false
    }
    fn ownership_lookup_entries(&self, _entries: u64) {}
    fn ownership_reference_scan_entries(&self, _entries: u64) {}
    fn site_lookup_entries(&self, _entries: u64) {}
    fn ownership_verification(&self) {}
    fn sum_lowering_fallback(&self, _operand: chelis_numeric_core::prim::Prim, _message: &str) {}
}

/// Borrowed caller services. No field grants native checked-product authority.
pub struct Resources<'a, S: StackServices> {
    pub stack: &'a S,
    pub cancellation: Option<&'a dyn Cancellation>,
    pub observer: Option<&'a dyn Observer>,
    pub free_variable_walk_work: Option<&'a FreeVariableWalkWork>,
    pub checker_policy: CheckPolicy,
    pub checker_stack_bytes: usize,
    pub checker_generalize_sweep_oracle: bool,
}

impl<'a, S: StackServices> Resources<'a, S> {
    pub const fn new(stack: &'a S) -> Self {
        Self {
            stack,
            cancellation: None,
            observer: None,
            free_variable_walk_work: None,
            checker_policy: CheckPolicy {
                linked_program: false,
            },
            checker_stack_bytes: 512 * 1024 * 1024,
            checker_generalize_sweep_oracle: true,
        }
    }

    pub fn run<R>(&self, run: impl FnOnce(&Operation<'_, S>) -> R) -> R {
        run(&Operation::new(self))
    }
}

/// Mutable analysis state belongs to this operation, not a thread or process.
pub struct Operation<'a, S: StackServices> {
    pub resources: &'a Resources<'a, S>,
    pub lower: crate::lower::OperationState,
    pub host: crate::host::OperationState,
}

impl<'a, S: StackServices> Operation<'a, S> {
    pub fn new(resources: &'a Resources<'a, S>) -> Self {
        Self {
            resources,
            lower: Default::default(),
            host: Default::default(),
        }
    }

    pub fn cancelled(&self) -> bool {
        self.resources
            .cancellation
            .is_some_and(Cancellation::is_cancelled)
    }

    pub fn free_runtime_variables(&self, expr: &Expr) -> Vec<String> {
        match self.resources.free_variable_walk_work {
            Some(work) => {
                chelis_checker_core::linearity::free_runtime_variables_with_work(expr, work)
            }
            None => chelis_checker_core::linearity::free_runtime_variables(expr),
        }
    }

    pub fn rename_free_runtime_variables(
        &self,
        expr: &Expr,
        renames: &BTreeMap<String, String>,
    ) -> Expr {
        match self.resources.free_variable_walk_work {
            Some(work) => chelis_checker_core::linearity::rename_free_runtime_variables_with_work(
                expr, renames, work,
            ),
            None => chelis_checker_core::linearity::rename_free_runtime_variables(expr, renames),
        }
    }

    pub fn checker_resources(&self) -> chelis_checker_core::Resources<'_, S> {
        let mut resources = chelis_checker_core::Resources::new(self.resources.stack);
        resources.cancellation = self.resources.cancellation;
        resources.policy = self.resources.checker_policy;
        resources.stack_bytes = self.resources.checker_stack_bytes;
        resources.generalize_sweep_oracle = self.resources.checker_generalize_sweep_oracle;
        resources
    }

    pub fn phase(&self, label: &str) {
        if let Some(observer) = self.resources.observer {
            observer.phase(label);
        }
    }

    pub fn declaration_begin(&self, name: &str) {
        if let Some(observer) = self.resources.observer {
            observer.declaration_begin(name);
        }
    }

    pub fn declaration_end(&self, name: &str, nodes: usize) {
        if let Some(observer) = self.resources.observer {
            observer.declaration_end(name, nodes);
        }
    }

    pub fn program_context_prepared(&self) {
        if let Some(observer) = self.resources.observer {
            observer.program_context_prepared();
        }
    }

    pub fn declaration_body_visit(&self) {
        if let Some(observer) = self.resources.observer {
            observer.declaration_body_visit();
        }
    }

    pub fn contract_normalized(&self) {
        if let Some(observer) = self.resources.observer {
            observer.contract_normalized();
        }
    }

    pub fn record_hoist_nodes_enabled(&self) -> bool {
        self.resources
            .observer
            .is_some_and(Observer::record_hoist_nodes_enabled)
    }

    pub fn record_hoist_nodes(&self, nodes: u64) {
        if let Some(observer) = self.resources.observer {
            observer.record_hoist_nodes(nodes);
        }
    }

    pub fn refine_scope_entries(&self, entries: u64) {
        if let Some(observer) = self.resources.observer {
            observer.refine_scope_entries(entries);
        }
    }

    pub fn terminal_name_comparisons(&self, comparisons: u64) {
        if let Some(observer) = self.resources.observer {
            observer.terminal_name_comparisons(comparisons);
        }
    }

    pub fn host_summary_probe_build(&self) {
        if let Some(observer) = self.resources.observer {
            observer.host_summary_probe_build();
        }
    }

    pub fn host_lowering_session(&self) {
        if let Some(observer) = self.resources.observer {
            observer.host_lowering_session();
        }
    }

    pub fn site_lookup_entries(&self, entries: u64) {
        if let Some(observer) = self.resources.observer {
            observer.site_lookup_entries(entries);
        }
    }

    pub fn ownership_verification(&self) {
        if let Some(observer) = self.resources.observer {
            observer.ownership_verification();
        }
    }

    pub fn def_effect_row_derived(&self) {
        if let Some(observer) = self.resources.observer {
            observer.def_effect_row_derived();
        }
    }

    pub fn ownership_lookup_entries(&self, entries: u64) {
        if let Some(observer) = self.resources.observer {
            observer.ownership_lookup_entries(entries);
        }
    }

    pub fn ownership_reference_scan_entries(&self, entries: u64) {
        if let Some(observer) = self.resources.observer {
            observer.ownership_reference_scan_entries(entries);
        }
    }
}

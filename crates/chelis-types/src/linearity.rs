//! Native ownership operations and full-host test observation scopes.

use crate::infer::{CheckedProgram, with_resources};
use chelis_checker_core::errors::CheckError;
use chelis_checker_core::linearity::CopyRepair;
use chelis_deep::ast::Expr;
use std::collections::{BTreeMap, BTreeSet};

pub fn check_linearity(program: &CheckedProgram) -> Result<CheckedProgram, Vec<CheckError>> {
    with_resources(|resources| chelis_checker_core::check_linearity(program.facts(), resources))
        .map(|facts| program.with_facts(facts))
}
pub fn check_linearity_with_context(
    library: &CheckedProgram,
    new_code: &CheckedProgram,
) -> Result<CheckedProgram, Vec<CheckError>> {
    with_resources(|resources| {
        chelis_checker_core::check_linearity_with_context(
            library.facts(),
            new_code.facts(),
            resources,
        )
    })
    .map(|facts| new_code.with_facts(facts))
}
pub fn copy_repairs(
    program: &CheckedProgram,
    roots: Option<&BTreeSet<String>>,
) -> Result<Vec<CopyRepair>, Vec<CheckError>> {
    with_resources(|resources| chelis_checker_core::copy_repairs(program.facts(), roots, resources))
}

#[cfg(feature = "test-instrumentation")]
thread_local! {
    static WALK_WORK: chelis_checker_core::linearity::FreeVariableWalkWork = const {
        chelis_checker_core::linearity::FreeVariableWalkWork::inactive()
    };
}

pub fn free_runtime_variables(expr: &Expr) -> Vec<String> {
    #[cfg(feature = "test-instrumentation")]
    {
        WALK_WORK.with(|work| {
            chelis_checker_core::linearity::free_runtime_variables_with_work(expr, work)
        })
    }
    #[cfg(not(feature = "test-instrumentation"))]
    {
        chelis_checker_core::linearity::free_runtime_variables(expr)
    }
}
pub fn rename_free_runtime_variables(expr: &Expr, renames: &BTreeMap<String, String>) -> Expr {
    #[cfg(feature = "test-instrumentation")]
    {
        WALK_WORK.with(|work| {
            chelis_checker_core::linearity::rename_free_runtime_variables_with_work(
                expr, renames, work,
            )
        })
    }
    #[cfg(not(feature = "test-instrumentation"))]
    {
        chelis_checker_core::linearity::rename_free_runtime_variables(expr, renames)
    }
}

#[cfg(feature = "test-instrumentation")]
#[doc(hidden)]
pub fn counting_free_variable_walk_work<R>(work: impl FnOnce() -> R) -> (R, u64) {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            WALK_WORK.with(|counter| {
                counter.replace_count(self.0);
            });
        }
    }
    let previous = WALK_WORK.with(|counter| counter.replace_count(Some(0)));
    let restore = Restore(previous);
    let result = work();
    let counted = WALK_WORK
        .with(|counter| counter.count())
        .expect("the native counter scope stays armed");
    drop(restore);
    (result, counted)
}

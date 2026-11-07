//! Native checked-input admission and host execution products.

use chelis_ir_core::host::{
    CompiledProgram, ConcreteHostBinding, ConcreteHostFunction, ConcreteHostProgram,
    HelperExecutionProduct, HostExecutionMetadata, HostLoweringSession, ManifestEntry,
    ManifestView, PreparedHostFacts,
};
use chelis_ir_core::lower::LowerDiagnostic;
use chelis_ir_core::resources::Operation;
use chelis_stack_services::StackServices;
use chelis_types::CheckedProgram;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    static NATIVE_PROVIDER_CALL: RefCell<Option<(String, String)>> = const { RefCell::new(None) };
    static HOST_SUMMARY_PROBE_BUILDS: Cell<u64> = const { Cell::new(0) };
    static RECORD_HOIST_NODES: Cell<Option<u64>> = const { Cell::new(None) };
    static REFINE_SCOPE_ENTRIES: Cell<Option<u64>> = const { Cell::new(None) };
    static TERMINAL_NAME_COMPARISONS: Cell<Option<u64>> = const { Cell::new(None) };
}
static HOST_LOWERING_SESSIONS: AtomicU64 = AtomicU64::new(0);
static DEF_EFFECT_ROW_DERIVATIONS: AtomicU64 = AtomicU64::new(0);

struct NativeManifest<'a>(&'a chelis_types::manifest::RootManifest);
impl ManifestView for NativeManifest<'_> {
    fn len(&self) -> usize {
        self.0.entries.len()
    }
    fn entry(&self, index: usize) -> ManifestEntry<'_> {
        let entry = &self.0.entries[index];
        ManifestEntry {
            def_name: &entry.def_name,
            tensor_lane: entry.lane == chelis_types::types::Lane::Tensor,
        }
    }
}

/// A native checked source and its canonical shared host facts.
/// Raw core facts cannot construct this native owner.
pub struct PreparedHostProgram {
    prepared: PreparedHostFacts<CheckedProgram>,
}
impl PreparedHostProgram {
    pub fn new(program: CheckedProgram) -> Self {
        Self {
            prepared: PreparedHostFacts::new(program),
        }
    }
    pub fn program(&self) -> &CheckedProgram {
        self.prepared.program()
    }
    pub fn session<'a, S: StackServices>(
        &'a self,
        operation: &'a Operation<'a, S>,
    ) -> HostLoweringSession<'a, S> {
        self.prepared.session(operation)
    }
}

/// Scope one provider binding that the native caller admits.
/// The caller must validate the linked declaration and provider artifact first.
/// Source names alone do not grant provider authority.
pub fn with_native_provider_call<R>(
    linked_name: &str,
    c_symbol: &str,
    lower: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<(String, String)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            NATIVE_PROVIDER_CALL.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let old = NATIVE_PROVIDER_CALL
        .with(|slot| slot.replace(Some((linked_name.to_owned(), c_symbol.to_owned()))));
    let _restore = Restore(old);
    lower()
}
pub(crate) fn active_native_provider_call() -> Option<(String, String)> {
    NATIVE_PROVIDER_CALL.with(|slot| slot.borrow().clone())
}
pub fn host_summary_probe_builds() -> u64 {
    HOST_SUMMARY_PROBE_BUILDS.with(Cell::get)
}

pub fn reset_host_summary_probe_builds() {
    HOST_SUMMARY_PROBE_BUILDS.with(|builds| builds.set(0));
}

pub fn counting_record_hoist_nodes<R>(work: impl FnOnce() -> R) -> (R, u64) {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            RECORD_HOIST_NODES.with(|count| count.set(self.0));
        }
    }
    let restore = Restore(RECORD_HOIST_NODES.with(|count| count.replace(Some(0))));
    let result = work();
    let counted = RECORD_HOIST_NODES
        .with(Cell::get)
        .expect("the count stays armed while the work runs");
    drop(restore);
    (result, counted)
}

pub fn counting_refine_scope_entries<R>(work: impl FnOnce() -> R) -> (R, u64) {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFINE_SCOPE_ENTRIES.with(|count| count.set(self.0));
        }
    }
    let restore = Restore(REFINE_SCOPE_ENTRIES.with(|count| count.replace(Some(0))));
    let result = work();
    let counted = REFINE_SCOPE_ENTRIES
        .with(Cell::get)
        .expect("the count stays armed while the work runs");
    drop(restore);
    (result, counted)
}

pub fn counting_terminal_name_comparisons<R>(work: impl FnOnce() -> R) -> (R, u64) {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TERMINAL_NAME_COMPARISONS.with(|count| count.set(self.0));
        }
    }
    let restore = Restore(TERMINAL_NAME_COMPARISONS.with(|count| count.replace(Some(0))));
    let result = work();
    let counted = TERMINAL_NAME_COMPARISONS
        .with(Cell::get)
        .expect("the count stays armed while the work runs");
    drop(restore);
    (result, counted)
}

pub fn host_lowering_sessions() -> u64 {
    HOST_LOWERING_SESSIONS.load(Ordering::Relaxed)
}

pub(crate) fn observe_record_hoist_nodes(nodes: u64) {
    RECORD_HOIST_NODES.with(|count| {
        if let Some(previous) = count.get() {
            count.set(Some(previous + nodes));
        }
    });
}
pub(crate) fn record_hoist_nodes_enabled() -> bool {
    RECORD_HOIST_NODES.with(|count| count.get().is_some())
}
pub(crate) fn observe_refine_scope_entries(entries: u64) {
    REFINE_SCOPE_ENTRIES.with(|count| {
        if let Some(previous) = count.get() {
            count.set(Some(previous + entries));
        }
    });
}
pub(crate) fn observe_terminal_name_comparisons(entries: u64) {
    TERMINAL_NAME_COMPARISONS.with(|count| {
        if let Some(previous) = count.get() {
            count.set(Some(previous + entries));
        }
    });
}
pub(crate) fn observe_host_summary_probe_build() {
    HOST_SUMMARY_PROBE_BUILDS.with(|count| {
        let previous = count.get();
        count.set(if previous == u64::MAX {
            previous
        } else {
            previous + 1
        });
    });
}
pub(crate) fn observe_host_lowering_session() {
    HOST_LOWERING_SESSIONS.fetch_add(1, Ordering::Relaxed);
}

/// Native helper products stay bound to their original host payload.
/// ```compile_fail
/// use chelis_ir::host::HostExecutionPlan;
/// fn replace_helper(mut plan: HostExecutionPlan) {
///     plan.program().global_tensor_helpers.clear();
/// }
/// ```
/// ```compile_fail
/// use chelis_ir::host::HostExecutionPlan;
/// use chelis_ir_core::host::ConcreteHostProgram;
/// fn substitute(plan: HostExecutionPlan, forged: ConcreteHostProgram) {
///     let _ = plan.project(forged);
/// }
/// ```
/// Raw core products cannot create native execution authority.
/// ```compile_fail
/// use chelis_ir::host::HostExecutionPlan;
/// use chelis_ir_core::host::{ConcreteHostProgram, HostExecutionMetadata};
/// fn adopt(program: ConcreteHostProgram, metadata: HostExecutionMetadata) -> HostExecutionPlan {
///     HostExecutionPlan {
///         program,
///         global: metadata.global,
///         functions: metadata.functions,
///     }
/// }
/// ```
#[derive(Debug)]
pub struct HostExecutionPlan {
    program: ConcreteHostProgram,
    global: Vec<HelperExecutionProduct>,
    functions: Vec<Vec<HelperExecutionProduct>>,
}
impl HostExecutionPlan {
    pub fn program(&self) -> &ConcreteHostProgram {
        &self.program
    }

    /// Recover the ordinary public host payload only when no helper trace
    /// would be discarded.
    pub fn into_ordinary(self) -> Result<ConcreteHostProgram, String> {
        #[cfg(feature = "lowering-trace")]
        let retains_private_metadata = self.has_helper_traces();
        #[cfg(not(feature = "lowering-trace"))]
        let retains_private_metadata = false;
        if retains_private_metadata {
            Err("cannot discard retained host trace metadata".into())
        } else {
            Ok(self.program)
        }
    }

    /// Apply a manifest-owned rewrite to global observation metadata. Tensor
    /// helpers and function bodies are not exposed through this capability.
    pub fn try_transform_globals<E>(
        mut self,
        transform: impl FnOnce(&mut Vec<ConcreteHostBinding>, &[ConcreteHostFunction]) -> Result<(), E>,
    ) -> Result<Self, E> {
        transform(&mut self.program.globals, &self.program.functions)?;
        Ok(self)
    }

    /// Select compiler-proved reachable functions by name, moving the exact
    /// original bodies, helper graphs, and metadata slots together. Entry
    /// projection drops globals just like the ordinary compiler projection.
    pub fn project_functions(self, names: &[String]) -> Result<Self, String> {
        let mut wanted = names
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if wanted.len() != names.len() {
            return Err("host execution projection repeats a function name".into());
        }
        let Self {
            program,
            global: _,
            functions: execution,
        } = self;
        let mut functions = Vec::new();
        let mut selected_execution = Vec::new();
        for (function, metadata) in program.functions.into_iter().zip(execution) {
            if wanted.remove(&function.name) {
                functions.push(function);
                selected_execution.push(metadata);
            }
        }
        if let Some(missing) = wanted.into_iter().next() {
            return Err(format!(
                "projected function `{missing}` has no execution-plan origin"
            ));
        }
        let summary_rejections = functions
            .iter()
            .flat_map(|function| function.summary_rejections.iter().cloned())
            .collect();
        let projected = Self {
            program: ConcreteHostProgram {
                globals: Vec::new(),
                global_tensor_helpers: Vec::new(),
                functions,
                summary_rejections,
                adt_layouts: program.adt_layouts,
            },
            global: Vec::new(),
            functions: selected_execution,
        };
        projected.validate()?;
        Ok(projected)
    }

    #[cfg(feature = "lowering-trace")]
    pub fn has_helper_traces(&self) -> bool {
        self.global.iter().any(|product| product.trace.is_some())
            || self
                .functions
                .iter()
                .flatten()
                .any(|product| product.trace.is_some())
    }

    /// Explicitly discard only opt-in helper observations. Execution
    /// metadata and exact helper graphs remain attached; callers still cannot
    /// recover an ordinary host program while either is retained.
    #[cfg(feature = "lowering-trace")]
    pub fn discard_helper_traces(mut self) -> Self {
        for product in self
            .global
            .iter_mut()
            .chain(self.functions.iter_mut().flatten())
        {
            product.trace = None;
        }
        self
    }

    #[cfg(feature = "lowering-trace")]
    pub fn global_helper_trace(
        &self,
        helper: usize,
    ) -> Result<Option<&chelis_ir_core::lowering_trace::HelperLoweringTrace>, String> {
        self.global
            .get(helper)
            .map(|product| product.trace.as_ref())
            .ok_or_else(|| format!("global helper {helper} has no execution-plan origin"))
    }

    #[cfg(feature = "lowering-trace")]
    pub fn function_helper_trace(
        &self,
        function: &str,
        helper: usize,
    ) -> Result<Option<&chelis_ir_core::lowering_trace::HelperLoweringTrace>, String> {
        let index = self
            .program
            .functions
            .iter()
            .position(|candidate| candidate.name == function)
            .ok_or_else(|| format!("function `{function}` has no execution-plan origin"))?;
        self.functions[index]
            .get(helper)
            .map(|product| product.trace.as_ref())
            .ok_or_else(|| {
                format!("function `{function}` helper {helper} has no execution-plan origin")
            })
    }

    fn validate(&self) -> Result<(), String> {
        if self.global.len() != self.program.global_tensor_helpers.len()
            || self.functions.len() != self.program.functions.len()
            || self
                .functions
                .iter()
                .zip(&self.program.functions)
                .any(|(products, function)| products.len() != function.tensor_helpers.len())
        {
            return Err("host helper products do not match helper topology".into());
        }
        Ok(())
    }

    /// The host payload, once the caller has taken any helper traces.
    pub(crate) fn into_program(self) -> ConcreteHostProgram {
        self.program
    }
}

fn lower_checked(
    program: &CheckedProgram,
    manifest: Option<&chelis_types::manifest::RootManifest>,
    c_execution_lane: bool,
    collect_trace: bool,
) -> Result<(CompiledProgram, HostExecutionMetadata), LowerDiagnostic> {
    crate::resources::with_operation(|operation| {
        operation
            .host
            .with_native_provider_call(active_native_provider_call(), || {
                let manifest = manifest.map(NativeManifest);
                let view = manifest.as_ref().map(|view| view as &dyn ManifestView);
                crate::lower::catch_lowering_external(
                    chelis_ir_core::lower::UnrollBudget::Unbounded,
                    std::panic::AssertUnwindSafe(|| {
                        chelis_ir_core::host::try_lower_compiled_program_with_options(
                            program.facts(),
                            view,
                            c_execution_lane,
                            collect_trace,
                            operation,
                        )
                    }),
                )?
            })
    })
}

pub fn try_lower_compiled_program(
    program: &CheckedProgram,
) -> Result<CompiledProgram, LowerDiagnostic> {
    lower_checked(program, None, false, false).map(|(program, _)| program)
}
pub fn try_lower_compiled_program_with_manifest(
    program: &CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
) -> Result<CompiledProgram, LowerDiagnostic> {
    lower_checked(program, Some(manifest), false, false).map(|(program, _)| program)
}
pub fn try_lower_manifested_program(
    program: &chelis_types::manifest::ManifestedProgram,
) -> Result<CompiledProgram, LowerDiagnostic> {
    try_lower_compiled_program_with_manifest(program.checked(), program.manifest())
}
fn lower_execution_program(
    program: &CheckedProgram,
    manifest: Option<&chelis_types::manifest::RootManifest>,
    collect_trace: bool,
) -> Result<(Option<chelis_dag_core::Dag>, Option<HostExecutionPlan>), LowerDiagnostic> {
    let (compiled, products) = lower_checked(program, manifest, true, collect_trace)?;
    Ok((
        compiled.dag,
        compiled.host.map(|program| HostExecutionPlan {
            program,
            global: products.global,
            functions: products.functions,
        }),
    ))
}
pub fn try_lower_manifested_execution_program(
    program: &chelis_types::manifest::ManifestedProgram,
) -> Result<(Option<chelis_dag_core::Dag>, Option<HostExecutionPlan>), LowerDiagnostic> {
    lower_execution_program(program.checked(), Some(program.manifest()), false)
}
pub fn try_lower_execution_program_with_manifest(
    program: &CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
) -> Result<(Option<chelis_dag_core::Dag>, Option<HostExecutionPlan>), LowerDiagnostic> {
    lower_execution_program(program, Some(manifest), false)
}
#[cfg(feature = "lowering-trace")]
pub fn try_lower_manifested_execution_program_with_trace(
    program: &chelis_types::manifest::ManifestedProgram,
) -> Result<(Option<chelis_dag_core::Dag>, Option<HostExecutionPlan>), LowerDiagnostic> {
    lower_execution_program(program.checked(), Some(program.manifest()), true)
}
#[cfg(feature = "lowering-trace")]
pub fn try_lower_execution_program_with_manifest_and_trace(
    program: &CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
) -> Result<(Option<chelis_dag_core::Dag>, Option<HostExecutionPlan>), LowerDiagnostic> {
    lower_execution_program(program, Some(manifest), true)
}

pub fn def_effect_row_derivations() -> u64 {
    DEF_EFFECT_ROW_DERIVATIONS.load(Ordering::Relaxed)
}
pub(crate) fn observe_def_effect_row_derived() {
    DEF_EFFECT_ROW_DERIVATIONS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod native_boundary_tests {
    use super::*;

    #[test]
    fn provider_binding_restores_outer_scope_after_panic() {
        let original = active_native_provider_call();
        with_native_provider_call("outer", "outer_symbol", || {
            let caught = std::panic::catch_unwind(|| {
                with_native_provider_call("inner", "inner_symbol", || {
                    assert_eq!(
                        active_native_provider_call(),
                        Some(("inner".into(), "inner_symbol".into())),
                    );
                    panic!("interrupt provider admission");
                });
            });
            assert!(caught.is_err());
            assert_eq!(
                active_native_provider_call(),
                Some(("outer".into(), "outer_symbol".into())),
            );
        });
        assert_eq!(active_native_provider_call(), original);
    }

    #[test]
    fn record_count_scope_restores_outer_window_after_panic() {
        let ((), nodes) = counting_record_hoist_nodes(|| {
            observe_record_hoist_nodes(3);
            let caught = std::panic::catch_unwind(|| {
                counting_record_hoist_nodes(|| {
                    observe_record_hoist_nodes(7);
                    panic!("interrupt record count");
                });
            });
            assert!(caught.is_err());
            assert!(record_hoist_nodes_enabled());
            observe_record_hoist_nodes(5);
        });
        assert_eq!(nodes, 8);
        assert!(!record_hoist_nodes_enabled());
    }
}

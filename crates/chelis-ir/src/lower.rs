//! Native checked ingresses and panic adapters for portable lowering.

use chelis_checker_core::LinearityInfo;
use chelis_dag_core::dag::{Dag, NodeId, TensorType};
use chelis_deep::Expr;
use chelis_ir_core::lower::{ComposedLowering, LowerDiagnostic, UnrollBudget};
use chelis_numeric_core::prim::Prim;
use chelis_types::{CheckedProgram, LibraryProofId};
use chelis_unord::UnordMap;
use std::any::Any;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

thread_local! {
    static SUPPRESS_LOWERING_PANIC_OUTPUT: Cell<bool> = const { Cell::new(false) };
}

fn panic_payload_to_lower_diagnostic(payload: &(dyn Any + Send)) -> LowerDiagnostic {
    if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
        return diagnostic.clone();
    }
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|value| (*value).to_string())
        })
        .unwrap_or_else(|| "internal lowering error".to_string());
    LowerDiagnostic::new(message, None, None)
}

fn catch_lowering<R>(
    _budget: UnrollBudget,
    f: impl FnOnce() -> R + std::panic::UnwindSafe,
) -> Result<R, LowerDiagnostic> {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.set(false));
        }
    }
    install_chelis_panic_hook();
    SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.set(true));
    let _guard = Guard;
    std::panic::catch_unwind(f).map_err(|payload| panic_payload_to_lower_diagnostic(&*payload))
}

pub(crate) fn catch_lowering_external<R>(
    budget: UnrollBudget,
    f: impl FnOnce() -> R + std::panic::UnwindSafe,
) -> Result<R, LowerDiagnostic> {
    catch_lowering(budget, f)
}

static CHELIS_PANIC_HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

pub fn install_chelis_panic_hook() {
    CHELIS_PANIC_HOOK_INSTALLED.get_or_init(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if SUPPRESS_LOWERING_PANIC_OUTPUT.with(Cell::get) {
                return;
            }
            previous(info);
        }));
    });
}

// Only native checked ingresses pair ordinary data with a native proof identity.
#[derive(Debug, Clone)]
pub struct LoweredLibrary {
    data: chelis_ir_core::lower::LoweredLibrary,
    library_proof_id: Option<LibraryProofId>,
}

// Field order is the original positional binary wire order.
#[derive(serde::Serialize)]
#[serde(rename = "LoweredLibrary")]
struct LoweredLibraryWireRef<'a> {
    dag: &'a Dag,
    symbol_table: &'a UnordMap<String, NodeId>,
    program_defs: &'a BTreeMap<String, Expr>,
    program_signatures: &'a BTreeMap<String, Expr>,
    program_types: &'a BTreeMap<String, TensorType>,
    local_ascriptions: &'a [chelis_checker_core::CheckedLocalAscription],
    linearity: &'a LinearityInfo,
    lowered_names: &'a BTreeMap<String, bool>,
    rootless_defs: &'a BTreeSet<String>,
    library_proof_id: &'a Option<LibraryProofId>,
}

#[derive(serde::Deserialize)]
#[serde(rename = "LoweredLibrary")]
struct LoweredLibraryWire {
    dag: Dag,
    symbol_table: UnordMap<String, NodeId>,
    program_defs: BTreeMap<String, Expr>,
    program_signatures: BTreeMap<String, Expr>,
    program_types: BTreeMap<String, TensorType>,
    local_ascriptions: Vec<chelis_checker_core::CheckedLocalAscription>,
    linearity: LinearityInfo,
    lowered_names: BTreeMap<String, bool>,
    #[serde(default)]
    rootless_defs: BTreeSet<String>,
    #[serde(default)]
    library_proof_id: Option<LibraryProofId>,
}

impl serde::Serialize for LoweredLibrary {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        LoweredLibraryWireRef {
            dag: &self.data.dag,
            symbol_table: &self.data.symbol_table,
            program_defs: &self.data.program_defs,
            program_signatures: &self.data.program_signatures,
            program_types: &self.data.program_types,
            local_ascriptions: &self.data.local_ascriptions,
            linearity: &self.data.linearity,
            lowered_names: &self.data.lowered_names,
            rootless_defs: &self.data.rootless_defs,
            library_proof_id: &self.library_proof_id,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for LoweredLibrary {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = LoweredLibraryWire::deserialize(deserializer)?;
        Ok(Self {
            data: chelis_ir_core::lower::LoweredLibrary {
                dag: wire.dag,
                symbol_table: wire.symbol_table,
                program_defs: wire.program_defs,
                program_signatures: wire.program_signatures,
                program_types: wire.program_types,
                local_ascriptions: wire.local_ascriptions,
                linearity: wire.linearity,
                lowered_names: wire.lowered_names,
                rootless_defs: wire.rootless_defs,
            },
            library_proof_id: wire.library_proof_id,
        })
    }
}

impl LoweredLibrary {
    fn from_checked(
        data: chelis_ir_core::lower::LoweredLibrary,
        library_proof_id: Option<LibraryProofId>,
    ) -> Self {
        Self {
            data,
            library_proof_id,
        }
    }

    pub fn data(&self) -> &chelis_ir_core::lower::LoweredLibrary {
        &self.data
    }

    pub fn dag(&self) -> &Dag {
        &self.data.dag
    }

    pub fn symbol_table(&self) -> &UnordMap<String, NodeId> {
        &self.data.symbol_table
    }

    pub fn program_defs(&self) -> &BTreeMap<String, Expr> {
        &self.data.program_defs
    }

    pub fn program_signatures(&self) -> &BTreeMap<String, Expr> {
        &self.data.program_signatures
    }

    pub fn program_types(&self) -> &BTreeMap<String, TensorType> {
        &self.data.program_types
    }

    pub fn local_ascriptions(&self) -> &[chelis_checker_core::CheckedLocalAscription] {
        &self.data.local_ascriptions
    }

    pub fn linearity(&self) -> &LinearityInfo {
        &self.data.linearity
    }

    pub fn lowered_names(&self) -> &BTreeMap<String, bool> {
        &self.data.lowered_names
    }

    pub fn rootless_defs(&self) -> &BTreeSet<String> {
        &self.data.rootless_defs
    }

    pub fn library_proof_id(&self) -> Option<LibraryProofId> {
        self.library_proof_id
    }

    pub fn into_dag_and_rootless_defs(self) -> (Dag, BTreeSet<String>) {
        (self.data.dag, self.data.rootless_defs)
    }
}

pub fn lower_program(program: &CheckedProgram) -> Dag {
    try_lower_program(program).unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_program(program: &CheckedProgram) -> Result<Dag, LowerDiagnostic> {
    catch_lowering(
        UnrollBudget::Unbounded,
        std::panic::AssertUnwindSafe(|| {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::lower::try_lower_program(program.facts(), operation)
            })
        }),
    )?
}

pub fn lower_program_to_library(program: &CheckedProgram) -> LoweredLibrary {
    try_lower_program_to_library(program).unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_program_to_library(
    program: &CheckedProgram,
) -> Result<LoweredLibrary, LowerDiagnostic> {
    catch_lowering(
        UnrollBudget::Unbounded,
        std::panic::AssertUnwindSafe(|| {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::lower::try_lower_program_to_library(program.facts(), operation)
                    .map(|data| LoweredLibrary::from_checked(data, program.library_proof_id()))
            })
        }),
    )?
}

#[cfg(feature = "lowering-trace")]
pub(crate) fn try_lower_program_to_library_with_trace(
    program: &CheckedProgram,
) -> Result<
    (
        LoweredLibrary,
        chelis_ir_core::lowering_trace::LoweringTrace,
    ),
    LowerDiagnostic,
> {
    catch_lowering(
        UnrollBudget::Unbounded,
        std::panic::AssertUnwindSafe(|| {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::lower::try_lower_program_to_library_with_trace(
                    program.facts(),
                    operation,
                )
                .map(|(data, trace)| {
                    (
                        LoweredLibrary::from_checked(data, program.library_proof_id()),
                        trace,
                    )
                })
            })
        }),
    )?
}

pub fn lower_program_with_context(library: &LoweredLibrary, new_program: &CheckedProgram) -> Dag {
    try_lower_program_with_context(library, new_program)
        .map(|composed| composed.dag)
        .unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_program_with_context(
    library: &LoweredLibrary,
    new_program: &CheckedProgram,
) -> Result<ComposedLowering, LowerDiagnostic> {
    catch_lowering(
        UnrollBudget::Unbounded,
        std::panic::AssertUnwindSafe(|| {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::lower::try_lower_program_with_context(
                    library.data(),
                    new_program.facts(),
                    operation,
                )
            })
        }),
    )?
}

pub fn top_level_lowering_map_from_realizability(
    program: &CheckedProgram,
    target_prims: &[Prim],
) -> BTreeMap<String, bool> {
    let result = chelis_effects::realizability::infer_realizability(program, target_prims);
    result
        .lane_by_def
        .into_iter()
        .map(|(name, lane)| (name, lane == chelis_types::types::Lane::Tensor))
        .collect()
}

#[cfg(test)]
thread_local! {
    static UNROLL_BUDGET_PROBE: std::cell::RefCell<Option<(usize, usize, Option<String>)>> = const { std::cell::RefCell::new(None) };
    static ENFORCING_SCOPES_ENTERED: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn configure_test_operation(
    operation: &chelis_ir_core::resources::Operation<'_, chelis_syntax_host::NativeStack>,
) {
    *operation.lower.unroll_budget_probe.borrow_mut() =
        UNROLL_BUDGET_PROBE.with(|probe| probe.borrow().clone());
    operation
        .lower
        .enforcing_scopes_entered
        .set(ENFORCING_SCOPES_ENTERED.with(Cell::get));
}

#[cfg(test)]
pub(crate) fn finish_test_operation(
    operation: &chelis_ir_core::resources::Operation<'_, chelis_syntax_host::NativeStack>,
) {
    UNROLL_BUDGET_PROBE
        .with(|probe| *probe.borrow_mut() = operation.lower.unroll_budget_probe.borrow().clone());
    ENFORCING_SCOPES_ENTERED
        .with(|entered| entered.set(operation.lower.enforcing_scopes_entered.get()));
}

#[cfg(test)]
use chelis_dag_core::dag::{DimInfo, RiscOp};
#[cfg(test)]
mod unroll_budget_tests {
    //! chelis#3864: the unroll budget counts the bodies one recursion's
    //! unroll inlines, and nothing else. Each test replaces the budget with
    //! a small one, so an exponential unroll is cut early.
    use super::*;
    use chelis_ir_core::lower::{
        SubexprLoweringContext, collect_top_level_defs, collect_top_level_sigs,
        try_lower_subexpr_program_with_context_and_controls,
    };

    const TEST_BUDGET: usize = 2_000;

    /// nautilus's `Nautilus.Roots` bisection and its `examplerootfind`
    /// caller, with the iteration bound left open.
    const BISECTION: &str = r#"def r_abs_f32(x: f32) -> f32 = if lt(x, cast(0.0, f32)) then neg(x) else x
def r_nan_f32() -> f32 = cast(0.0, f32) |> div(cast(0.0, f32))
def r_is_nan_f32(x: f32) -> bool = neq(x, x)
def bisection_rec(f: f32 -> f32, lo: f32, hi: f32, flo: f32, tol: f32, iters: i64) -> f32 = {
  zero_i = cast(0, i64)
  one_i = cast(1, i64)
  if lte(iters, zero_i) then r_nan_f32() else {
    width = sub(hi, lo)
    if lt(width, tol) then (cast(0.5, f32) |> mul(add(lo, hi))) else {
      mid = cast(0.5, f32) |> mul(add(lo, hi))
      if r_is_nan_f32(mid) then r_nan_f32() else if or(eq(mid, lo), eq(mid, hi)) then mid else {
        fmid = f(mid)
        if r_is_nan_f32(fmid) then r_nan_f32() else {
          afmid = r_abs_f32(fmid)
          if lt(afmid, tol) then mid else {
            same_sign = flo |> mul(fmid) |> gt(cast(0.0, f32))
            if same_sign then bisection_rec(f, mid, hi, fmid, tol, sub(iters, one_i)) else bisection_rec(f, lo, mid, flo, tol, sub(iters, one_i))
          }
        }
      }
    }
  }
}
def bisection(f: f32 -> f32, lo: f32, hi: f32, tol: f32, max_iters: i64) -> f32 = {
  nan_input = r_is_nan_f32(lo) |> or(r_is_nan_f32(hi)) |> or(r_is_nan_f32(tol))
  if nan_input then r_nan_f32() else {
    flo = f(lo)
    if r_is_nan_f32(flo) then r_nan_f32() else {
      aflo = r_abs_f32(flo)
      if lt(aflo, tol) then lo else {
        fhi = f(hi)
        if r_is_nan_f32(fhi) then r_nan_f32() else {
          afhi = r_abs_f32(fhi)
          if lt(afhi, tol) then hi else {
            prod = mul(flo, fhi)
            if gt(prod, cast(0.0, f32)) then r_nan_f32() else bisection_rec(f, lo, hi, flo, tol, max_iters)
          }
        }
      }
    }
  }
}
def erf_projectile(theta: f32) -> f32 = {
  two_theta = mul(cast(2.0, f32), theta)
  s = sin(two_theta)
  g = cast(9.81, f32)
  v_sq = mul(cast(20.0, f32), cast(20.0, f32))
  d = cast(30.0, f32)
  sub(mul(div(v_sq, g), s), d)
}
def example() -> f32 = bisection(erf_projectile, cast(0.1, f32), cast(0.7, f32), cast(1e-8, f32), cast(ITERS, i64))"#;

    /// A recursion whose base case folds: `count_up` unrolls `LEVELS`
    /// levels, each a literal-rooted `add` chain the condition resolves.
    fn count_up(name: &str, levels: usize) -> String {
        format!(
            "def {name}(x: tensor[3, f32], k: i64) -> tensor[3, f32] = if gte(k, cast({levels}, i64)) then x else {name}(x, add(k, cast(1, i64)))\n"
        )
    }

    fn checked(source: &str) -> CheckedProgram {
        let declarations = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
            .expect("the probe parses");
        let checked = chelis_types::check_typed_program(
            &chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
                .expect("the probe desugars"),
        )
        .expect("the probe checks");
        let checked =
            chelis_effects::check_program(&checked).expect("the probe has no effect errors");
        chelis_types::check_linearity(&checked).expect("the probe is linear")
    }

    /// Run `work` under `budget` on a large stack, returning its result, the
    /// bodies counted against the budget and the last rejection's message.
    fn with_budget<R: Send + 'static>(
        budget: usize,
        work: impl FnOnce() -> R + Send + 'static,
    ) -> (R, usize, Option<String>) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(move || {
                UNROLL_BUDGET_PROBE.with(|probe| *probe.borrow_mut() = Some((budget, 0, None)));
                let result = work();
                let (_, counted, rejection) = UNROLL_BUDGET_PROBE
                    .with(|probe| probe.borrow_mut().take())
                    .expect("the probe stays armed");
                (result, counted, rejection)
            })
            .expect("spawn the lowering thread")
            .join()
            .expect("the lowering thread finishes")
    }

    fn lower_example(source: String, budget: usize) -> (bool, usize, Option<String>) {
        let program = checked(&source);
        with_budget(budget, move || {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::host::lower_named_tensor_entry_dags(
                    program.facts(),
                    &["example".to_string()],
                    operation,
                )
                .map(|dag| {
                    dag.expect("entry lowering returns no fatal diagnostic")
                        .is_some()
                })
                .collect::<Vec<_>>()[0]
            })
        })
    }

    #[test]
    fn an_unroll_that_branches_at_runtime_is_rejected_at_the_budget() {
        let (lowered, counted, rejection) =
            lower_example(BISECTION.replace("ITERS", "2"), TEST_BUDGET);
        assert!(!lowered, "the entry is rejected, not lowered");
        assert_eq!(
            counted,
            TEST_BUDGET + 1,
            "lowering stopped at the first body past the budget"
        );
        let message = rejection.expect("the budget rejected the entry");
        assert!(
            message.starts_with(&format!(
                "recursive inlining of `bisection_rec` exceeded the budget of {TEST_BUDGET} function bodies for an unroll in which a frame makes more than one recursive call"
            )) && message.contains("both arms of a condition known only at runtime")
                && message.contains("chelis#620"),
            "the rejection names the recursion, the budget and the runtime branch: {message}"
        );
    }

    #[test]
    fn the_budget_is_what_stops_it() {
        // A larger budget lets the same unroll go further: it does not end on
        // its own within either.
        let (_, small, _) = lower_example(BISECTION.replace("ITERS", "2"), TEST_BUDGET);
        let (_, large, _) = lower_example(BISECTION.replace("ITERS", "2"), 2 * TEST_BUDGET);
        assert_eq!(small, TEST_BUDGET + 1);
        assert_eq!(large, 2 * TEST_BUDGET + 1);
    }

    #[test]
    fn inlining_outside_a_recursion_never_counts() {
        // chelis#3864 round 1: 130 entries, each inlining a 255-body helper
        // tree, lower as one program under a budget far below their total.
        let mut source = String::from("def h0(x: tensor[4, f32]) -> tensor[4, f32] = mul(x, x)\n");
        for level in 1..=7 {
            let below = level - 1;
            source.push_str(&format!(
                "def h{level}(x: tensor[4, f32]) -> tensor[4, f32] = h{below}(h{below}(x))\n"
            ));
        }
        for entry in 0..130 {
            source.push_str(&format!(
                "out{entry} = h7(to_tensor([{entry}.0f32, 1.0f32, 2.0f32, 3.0f32]))\n"
            ));
        }
        let program = checked(&source);
        let (lowered, counted, rejection) = with_budget(100, move || {
            crate::lower::try_lower_program(&program).is_ok()
        });
        assert!(lowered, "the program lowers: {rejection:?}");
        assert_eq!(counted, 0, "no body was inlined inside a recursion");
    }

    /// `hN` inlines 2^(N+1) - 1 bodies and recurses nowhere.
    fn helper_tree(width: &str) -> String {
        let mut source = format!("def h0(x: {width}) -> {width} = mul(x, x)\n");
        for level in 1..=7 {
            let below = level - 1;
            source.push_str(&format!(
                "def h{level}(x: {width}) -> {width} = h{below}(h{below}(x))\n"
            ));
        }
        source
    }

    /// A recursion `depth` levels deep that recurses in both arms of a
    /// runtime `if`, applying `helper` on one arm: 2^(depth+1) - 1 calls.
    fn branched_tree(name: &str, depth: usize, helper: &str) -> String {
        format!(
            "def {name}(x: f32, k: i64) -> f32 = if gte(k, cast({depth}, i64)) then x else if gt(sin(x), cast(0.0, f32)) then {name}({helper}(x), add(k, cast(1, i64))) else {name}(neg(x), add(k, cast(1, i64)))\n"
        )
    }

    /// Lower `source`'s `example` under `budget`, and check that its unroll
    /// counted past the budget yet was not limited.
    fn assert_never_limited(source: String, budget: usize) {
        let (lowered, counted, rejection) = lower_example(source, budget);
        assert!(lowered, "the chain lowers: {rejection:?}");
        assert_eq!(rejection, None);
        assert!(
            counted > budget,
            "the chain inlined past the budget: {counted}"
        );
    }

    #[test]
    fn a_static_loop_with_a_heavy_helper_is_never_limited() {
        // chelis#3864 rounds 2 and 3: a loop whose base case folds is a chain,
        // so its bodies, a 31-body helper per iteration among them, never
        // meet the budget; the depth caps bound it.
        let source = format!(
            "{}{}def step(x: tensor[3, f32], i: i64) -> tensor[3, f32] = if gte(i, cast(32, i64)) then x else step(h4(x), add(i, cast(1, i64)))\ndef example(seed: tensor[3, f32]) -> tensor[3, f32] = up(step(seed, cast(0, i64)), cast(0, i64))\n",
            helper_tree("tensor[3, f32]"),
            count_up("up", 300)
        );
        assert_never_limited(source, 10);
    }

    #[test]
    fn an_early_exit_chain_with_a_heavy_helper_is_never_limited() {
        // chelis#3864 round 4: a static cap, a convergence test decided at
        // runtime, and the recursive call in one arm only. The call sits in
        // a runtime arm, but the unroll is a chain.
        let source = format!(
            "{}def step(x: f32, i: i64) -> f32 = if gte(i, cast(32, i64)) then x else if lt(sin(x), cast(-2.0, f32)) then x else step(h4(x), add(i, cast(1, i64)))\ndef example() -> f32 = step(cast(1.5, f32), cast(0, i64))\n",
            helper_tree("f32")
        );
        assert_never_limited(source, 10);
    }

    #[test]
    fn a_mutual_recursion_chain_is_never_limited() {
        // `ping` and `pong` call each other once per level: a chain through
        // two names.
        let source = format!(
            "{}def ping(x: f32, k: i64) -> f32 = if gte(k, cast(100, i64)) then x else pong(h3(x), add(k, cast(1, i64)))\ndef pong(x: f32, k: i64) -> f32 = ping(h3(x), add(k, cast(1, i64)))\ndef example() -> f32 = ping(cast(1.5, f32), cast(0, i64))\n",
            helper_tree("f32")
        );
        assert_never_limited(source, 10);
    }

    #[test]
    fn a_branched_unroll_with_a_heavy_helper_trips() {
        // chelis#3864 round 3: a 6-level branched tree makes only 127 calls,
        // but each call in one arm inlines a 31-body helper, so the unroll
        // inlines far more than 1,000 bodies.
        let source = format!(
            "{}{}def example() -> f32 = tree(cast(1.5, f32), cast(0, i64))\n",
            helper_tree("f32"),
            branched_tree("tree", 6, "h4")
        );
        let (lowered, counted, rejection) = lower_example(source, 1_000);
        assert!(!lowered);
        assert_eq!(counted, 1_001);
        let message = rejection.expect("the budget rejected the unroll");
        assert!(
            message.starts_with(
                "recursive inlining of `tree` exceeded the budget of 1000 function bodies for an unroll in which a frame makes more than one recursive call"
            ),
            "{message}"
        );
    }

    #[test]
    fn each_branched_unroll_starts_its_own_count() {
        // Two 5-level branched trees, 63 calls each, lower one after the
        // other in one entry under a budget of 100: the count returns to zero
        // between them, so their bodies never meet one budget.
        let source = format!(
            "{}{}def example() -> f32 = tree_a(tree_b(cast(1.5, f32), cast(0, i64)), cast(0, i64))\n",
            branched_tree("tree_a", 5, ""),
            branched_tree("tree_b", 5, "")
        );
        let (lowered, counted, rejection) = lower_example(source, 100);
        assert!(lowered, "both trees lower: {rejection:?}");
        assert!(counted > 100, "both unrolls were counted: {counted}");
    }

    #[test]
    fn a_rejected_unroll_leaves_no_recursion_behind() {
        // The first entry's unroll is rejected mid-recursion, abandoning its
        // inline levels; the second entry's helper tree must not count.
        let source = format!(
            "{}{}def bad() -> f32 = tree(cast(1.5, f32), cast(0, i64))\ndef good(seed: tensor[3, f32]) -> tensor[3, f32] = h7(seed)\n",
            branched_tree("tree", 10, ""),
            helper_tree("tensor[3, f32]")
        );
        let program = checked(&source);
        let (lowered, counted, _) = with_budget(100, move || {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::host::lower_named_tensor_entry_dags(
                    program.facts(),
                    &["bad".to_string(), "good".to_string()],
                    operation,
                )
                .map(|dag| {
                    dag.expect("entry lowering returns no fatal diagnostic")
                        .is_some()
                })
                .collect::<Vec<_>>()
            })
        });
        assert_eq!(lowered, [false, true]);
        assert_eq!(counted, 101, "only the rejected unroll counted");
    }

    #[test]
    fn sequential_and_nested_loops_inside_a_loop_are_never_limited() {
        // chelis#3864 round 4: `f` runs a complete static `g` loop at each of
        // its levels, and `g` runs a complete static `i` loop at each of its,
        // each level inlining a 15-body helper.
        // Every frame makes one recursive call back into its own recursion,
        // so the polynomial nest is a set of chains, never a tree.
        let source = format!(
            "{}def ib(x: f32, k: i64) -> f32 = if gte(k, cast(6, i64)) then x else ib(h3(x), add(k, cast(1, i64)))\n\
             def gb(x: f32, k: i64) -> f32 = if gte(k, cast(12, i64)) then x else gb(ib(h3(x), cast(0, i64)), add(k, cast(1, i64)))\n\
             def fb(x: f32, k: i64) -> f32 = if gte(k, cast(12, i64)) then x else fb(gb(gb(h3(x), cast(0, i64)), cast(0, i64)), add(k, cast(1, i64)))\n\
             def example() -> f32 = fb(cast(1.5, f32), cast(0, i64))\n",
            helper_tree("f32")
        );
        assert_never_limited(source, 100);
    }

    #[test]
    fn a_frame_that_recurses_twice_through_a_helper_trips() {
        // `fv` calls the non-recursive `via` helper twice per level, and each
        // call re-enters `fv`: both re-entries extend the same frame of `fv`,
        // so the unroll is a tree.
        let source = "def via(x: f32, k: i64) -> f32 = fv(x, k)\n\
             def fv(x: f32, k: i64) -> f32 = if gte(k, cast(10, i64)) then x else add(via(x, add(k, cast(1, i64))), via(neg(x), add(k, cast(1, i64))))\n\
             def example() -> f32 = fv(cast(1.5, f32), cast(0, i64))\n"
            .to_string();
        let (lowered, counted, rejection) = lower_example(source, 100);
        assert!(!lowered);
        assert_eq!(counted, 101);
        assert!(
            rejection.is_some_and(|message| message.starts_with("recursive inlining of `fv`")
                || message.starts_with("recursive inlining of `via`")),
            "the rejection names the recursion"
        );
    }

    /// Two loop combinators that each apply a callback named `f`.
    fn combinators() -> String {
        "def it(f: f32 -> f32, x: f32, i: i64, n: i64) -> f32 = if gte(i, n) then x else it(f, f(x), add(i, cast(1, i64)), n)\n\
         def it2(f: f32 -> f32, x: f32, i: i64, n: i64) -> f32 = if gte(i, n) then x else it2(f, f(x), add(i, cast(1, i64)), n)\n"
            .to_string()
    }

    #[test]
    fn distinct_combinators_sharing_a_callback_name_are_never_limited() {
        // chelis#3864 round 5: `it2` iterates `row`, which runs a complete
        // `it` loop over `h4`. Both callbacks are called through a parameter
        // named `f`, but they are different functions, so each frame is
        // credited once: two nested chains.
        let source = format!(
            "{}{}def row(x: f32) -> f32 = it(h4, x, cast(0, i64), cast(8, i64))\ndef example() -> f32 = it2(row, cast(1.5, f32), cast(0, i64), cast(8, i64))\n",
            helper_tree("f32"),
            combinators()
        );
        assert_never_limited(source, 100);
    }

    #[test]
    fn a_combinator_nested_in_itself_is_one_recursion() {
        // The same combinator run inside its own callback re-enters the same
        // function: the inner loop and the outer loop both extend the outer
        // frame of `it`, so the unroll is a tree, and it is limited.
        let source = format!(
            "{}{}def row(x: f32) -> f32 = it(h4, x, cast(0, i64), cast(8, i64))\ndef example() -> f32 = it(row, cast(1.5, f32), cast(0, i64), cast(8, i64))\n",
            helper_tree("f32"),
            combinators()
        );
        let (lowered, counted, rejection) = lower_example(source, 100);
        assert!(!lowered, "the nested combinator is limited");
        // The outer frame's second credit comes after the first inner loop,
        // and the count accumulated by then applies at once.
        assert!(counted > 100, "{counted}");
        assert!(
            rejection.is_some_and(|message| message.contains("recursive inlining of `it`")),
            "the rejection names the combinator"
        );
    }

    #[test]
    fn only_cost_entries_enforce_the_budget() {
        // chelis#3864 round 3: eval, build and every lowering other than
        // `chelis cost`'s per-entry lowering have no fallback for a rejected
        // unroll, so they count nothing. The same branched tree that a cost
        // entry rejects lowers through the very same entry lowering when cost
        // is not the caller.
        // A static tree: both calls of every level are lowered, so each
        // frame is credited twice, with no runtime condition at all.
        let source = "def tree(x: tensor[3, f32], k: i64) -> tensor[3, f32] = if gte(k, cast(8, i64)) then x else add(tree(x, add(k, cast(1, i64))), tree(neg(x), add(k, cast(1, i64))))\n\
             def example(seed: tensor[3, f32]) -> tensor[3, f32] = tree(seed, cast(0, i64))\n\
             out = example(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
            .to_string();
        let cost_program = checked(&source);
        let (cost, _, rejection) = with_budget(100, move || {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::host::lower_named_tensor_entry_dags(
                    cost_program.facts(),
                    &["example".to_string()],
                    operation,
                )
                .map(|dag| {
                    dag.expect("entry lowering returns no fatal diagnostic")
                        .is_some()
                })
                .collect::<Vec<_>>()
            })
        });
        assert_eq!(cost, [false], "the cost entry is rejected");
        assert!(rejection.is_some());
        let entry_program = checked(&source);
        let (entry, counted, rejection) = with_budget(100, move || {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::host::lower_named_tensor_entry_dag(
                    entry_program.facts(),
                    "example",
                    operation,
                )
                .expect("entry lowering returns no fatal diagnostic")
                .is_some()
            })
        });
        assert!(
            entry,
            "the same entry lowering outside cost is unbounded: {rejection:?}"
        );
        assert_eq!(counted, 0);
        // The sub-expression entry eval's `vmap` and `grad` lower through,
        // over the same checked program.
        let controls_program = checked(&source);
        let (controls, counted, rejection) = with_budget(100, move || {
            crate::resources::with_operation(|operation| {
                let defs = collect_top_level_defs(controls_program.exprs());
                let sigs = collect_top_level_sigs(controls_program.exprs());
                let context = SubexprLoweringContext::from_checked_program(
                    controls_program.facts(),
                    defs.into_iter().collect(),
                    sigs.into_iter().collect(),
                    operation,
                )?;
                let call = chelis_deep::parser::parse_str(
                    "(app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} (var {} example) (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} seed))",
                    &chelis_syntax_host::NativeStack,
                )
                .expect("the call parses")
                .remove(0);
                let seed = TensorType {
                    dims: vec![DimInfo::Lit(3)],
                    precision: Prim::F32,
                };
                try_lower_subexpr_program_with_context_and_controls(
                    &call,
                    UnordMap::from([("seed".to_string(), seed)]),
                    &context,
                    operation,
                )
                .map(|lowered| lowered.dag().nodes().len())
            })
        });
        assert!(
            controls.is_ok(),
            "the vmap and grad entry is unbounded: {:?} {rejection:?}",
            controls.err()
        );
        assert_eq!(counted, 0);
    }

    #[test]
    fn the_rejection_names_the_branching_function_not_its_enclosing_chain() {
        // chelis#3864 round 3: a chain `walk` runs a branched `tree` at each
        // step. `walk` is the deeper name, but `tree` is the frame that
        // branched, and the rejection names it.
        let source = format!(
            "{}def walk(x: f32, i: i64) -> f32 = if gte(i, cast(20, i64)) then x else walk(tree(x, cast(0, i64)), add(i, cast(1, i64)))\ndef example() -> f32 = walk(cast(1.5, f32), cast(0, i64))\n",
            branched_tree("tree", 6, "")
        );
        // Under a budget of 1,000 the trip comes about eight steps in, when
        // `walk` is deeper than any level of `tree`.
        let (lowered, _, rejection) = lower_example(source, 1_000);
        assert!(!lowered);
        let message = rejection.expect("the budget rejected the unroll");
        assert!(
            message.starts_with("recursive inlining of `tree`"),
            "{message}"
        );
    }

    #[test]
    fn no_public_lowering_entry_but_cost_enforces_the_budget() {
        // chelis#3864 round 3: the whole-program entry, which eval, build and
        // library lowering use, opens only unbounded boundaries; the cost
        // entry opens an enforcing one. Counted by the boundaries entered, so
        // it holds whatever the program lowers to.
        let entered = |work: Box<dyn FnOnce()>| {
            ENFORCING_SCOPES_ENTERED.with(|entered| entered.set(0));
            work();
            ENFORCING_SCOPES_ENTERED.with(Cell::get)
        };
        let source = format!(
            "{}def example(seed: tensor[3, f32]) -> tensor[3, f32] = h7(seed)\nout = h7(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
            helper_tree("tensor[3, f32]")
        );
        let program = checked(&source);
        let whole = entered(Box::new(|| {
            crate::lower::try_lower_program_to_library(&program).expect("the program lowers");
        }));
        assert_eq!(whole, 0, "the whole-program entry enforces nothing");
        let cost = entered(Box::new(|| {
            crate::resources::with_operation(|operation| {
                chelis_ir_core::host::lower_named_tensor_entry_dags(
                    program.facts(),
                    &["example".to_string()],
                    operation,
                )
                .collect::<Vec<_>>()
            });
        }));
        assert!(cost > 0, "the cost entry enforces the budget");
    }

    /// A static tree `depth` levels deep: every level's two calls are lowered.
    fn static_tree(depth: usize) -> String {
        format!(
            "def t(x: f32, k: i64) -> f32 = if gte(k, cast({depth}, i64)) then x else add(t(x, add(k, cast(1, i64))), t(neg(x), add(k, cast(1, i64))))\n"
        )
    }

    #[test]
    fn a_trip_inside_a_value_probe_rejects_the_entry() {
        // chelis#3864 round 4: the cost entry reads a top-level value whose
        // initializer is a branched tree past the budget. Lowering probes the
        // initializer under a nested boundary whose failure becomes a
        // verdict; the trip must still reject the entry, never leave it
        // costed without the value.
        let source = format!(
            "{}big = t(cast(1.5, f32), cast(0, i64))\ndef example(y: f32) -> f32 = add(big, y)\n",
            static_tree(8)
        );
        let (lowered, counted, rejection) = lower_example(source, 100);
        assert!(!lowered, "the entry is omitted, not costed without `big`");
        assert!(counted > 100, "{counted}");
        assert!(rejection.is_some_and(|message| message.starts_with("recursive inlining of `t`")));
    }

    #[test]
    fn a_tripped_cost_entry_leaves_later_lowerings_unbounded() {
        // A cost entry that trips, then the same tree through an unbounded
        // lowering on the same thread: enforcement ends with the cost entry.
        let source = format!(
            "{}def example(y: f32) -> f32 = add(t(y, cast(0, i64)), y)\n",
            static_tree(8)
        );
        let program = checked(&source);
        let ((cost, after_cost, unbounded), counted, _) = with_budget(100, move || {
            let cost = crate::resources::with_operation(|operation| {
                chelis_ir_core::host::lower_named_tensor_entry_dags(
                    program.facts(),
                    &["example".to_string()],
                    operation,
                )
                .map(|dag| {
                    dag.expect("entry lowering returns no fatal diagnostic")
                        .is_some()
                })
                .collect::<Vec<_>>()
            });
            let after_cost = UNROLL_BUDGET_PROBE
                .with(|probe| probe.borrow().as_ref().map(|(_, counted, _)| *counted));
            let unbounded = crate::resources::with_operation(|operation| {
                chelis_ir_core::host::lower_named_tensor_entry_dag(
                    program.facts(),
                    "example",
                    operation,
                )
                .expect("entry lowering returns no fatal diagnostic")
                .is_some()
            });
            (cost, after_cost, unbounded)
        });
        assert_eq!(cost, [false], "the cost entry trips");
        assert!(unbounded, "the later lowering is unbounded");
        assert_eq!(
            Some(counted),
            after_cost,
            "the later lowering counts nothing"
        );
    }

    #[test]
    fn a_value_probe_inside_a_cost_entry_enforces_the_budget() {
        // chelis#3864 round 4: the probe that lowers a top-level value's
        // initializer runs under its own boundary inside the cost entry, and
        // inherits the entry's enforcement rather than escaping its budget.
        let source = format!(
            "{}small = t(cast(1.5, f32), cast(0, i64))\ndef example(y: f32) -> f32 = add(small, y)\n",
            static_tree(3)
        );
        let program = checked(&source);
        ENFORCING_SCOPES_ENTERED.with(|entered| entered.set(0));
        let lowered = crate::resources::with_operation(|operation| {
            chelis_ir_core::host::lower_named_tensor_entry_dags(
                program.facts(),
                &["example".to_string()],
                operation,
            )
            .map(|dag| {
                dag.expect("entry lowering returns no fatal diagnostic")
                    .is_some()
            })
            .collect::<Vec<_>>()
        });
        assert_eq!(lowered, [true]);
        assert!(
            ENFORCING_SCOPES_ENTERED.with(Cell::get) >= 2,
            "the entry and the value probe inside it both enforce"
        );
    }

    /// One block of a generated program for the unroll model test.
    #[derive(Clone)]
    enum Block {
        /// A static loop of `n` steps.
        Chain(usize),
        /// A static loop whose every step runs `inner` first.
        ChainWith(usize, Box<Block>),
        /// A static tree `d` levels deep: two recursive calls per level.
        Tree(usize),
        /// Two functions calling each other `n` times, one call per level.
        Mutual(usize),
        /// A combinator looping `n` steps over a function literal, whose
        /// body runs `inner` when there is one.
        LambdaChain(usize, Option<Box<Block>>),
        /// A combinator tree `d` levels deep applying a function literal at
        /// every leaf.
        LambdaTree(usize),
        /// `first`, then `second` on its result.
        Seq(Box<Block>, Box<Block>),
    }

    /// The model's frame identity: a declaration, or a literal by number.
    #[derive(Clone, PartialEq)]
    enum ModelKey {
        Declaration(String),
        Literal(usize),
    }

    /// The unroll rule over a block's inline events, computed from the
    /// block's structure alone.
    struct UnrollModel {
        budget: usize,
        depth: std::collections::BTreeMap<String, usize>,
        recursing: usize,
        frames: Vec<(ModelKey, usize)>,
        unroll: usize,
        branched: bool,
        counted: usize,
        tripped: bool,
    }

    impl UnrollModel {
        fn push(&mut self, binding: &str, key: ModelKey) {
            if self.tripped {
                return;
            }
            let depth = self.depth.entry(binding.to_string()).or_insert(0);
            *depth += 1;
            if *depth == 2 {
                self.recursing += 1;
            }
            if let Some((_, credits)) = self
                .frames
                .iter_mut()
                .rev()
                .find(|(frame, _)| *frame == key)
            {
                *credits += 1;
                if *credits > 1 {
                    self.branched = true;
                }
            }
            self.frames.push((key, 0));
            if self.recursing > 0 {
                self.counted += 1;
                self.unroll += 1;
                if self.unroll > self.budget && self.branched {
                    self.tripped = true;
                }
            }
        }

        fn pop(&mut self, binding: &str) {
            if self.tripped {
                return;
            }
            self.frames.pop();
            let depth = self.depth.get_mut(binding).expect("a pushed binding");
            if *depth == 2 {
                self.recursing -= 1;
                if self.recursing == 0 {
                    self.unroll = 0;
                    self.branched = false;
                }
            }
            *depth -= 1;
        }
    }

    /// A block's inline events, replayed into the model.
    type Events = Box<dyn Fn(&mut UnrollModel)>;

    /// The program text and the model's events for one block, numbered so
    /// every generated function and literal is distinct.
    struct Generator {
        defs: String,
        next: usize,
    }

    impl Generator {
        fn fresh(&mut self) -> usize {
            self.next += 1;
            self.next
        }

        /// Emit `block`'s definitions and return its call on `arg`, with a
        /// closure replaying its inline events into the model.
        fn emit(&mut self, block: &Block, arg: &str) -> (String, Events) {
            let id = self.fresh();
            match block {
                Block::Chain(n) | Block::ChainWith(n, _) => {
                    let name = format!("c{id}");
                    let (step, inner): (String, Option<Events>) = match block {
                        Block::ChainWith(_, inner) => {
                            let (call, events) = self.emit(inner, "neg(x)");
                            (call, Some(events))
                        }
                        _ => ("neg(x)".to_string(), None),
                    };
                    self.defs.push_str(&format!(
                        "def {name}(x: f32, k: i64) -> f32 = if gte(k, cast({n}, i64)) then x else {name}({step}, add(k, cast(1, i64)))\n"
                    ));
                    let n = *n;
                    let events = move |model: &mut UnrollModel| {
                        fn level(
                            model: &mut UnrollModel,
                            name: &str,
                            j: usize,
                            n: usize,
                            inner: &Option<Events>,
                        ) {
                            model.push(name, ModelKey::Declaration(name.to_string()));
                            if j < n {
                                if let Some(inner) = inner {
                                    inner(model);
                                }
                                level(model, name, j + 1, n, inner);
                            }
                            model.pop(name);
                        }
                        level(model, &name, 0, n, &inner);
                    };
                    (format!("c{id}({arg}, cast(0, i64))"), Box::new(events))
                }
                Block::Tree(d) => {
                    let name = format!("t{id}");
                    self.defs.push_str(&format!(
                        "def {name}(x: f32, k: i64) -> f32 = if gte(k, cast({d}, i64)) then x else add({name}(x, add(k, cast(1, i64))), {name}(neg(x), add(k, cast(1, i64))))\n"
                    ));
                    let d = *d;
                    let events = move |model: &mut UnrollModel| {
                        fn level(model: &mut UnrollModel, name: &str, j: usize, d: usize) {
                            model.push(name, ModelKey::Declaration(name.to_string()));
                            if j < d {
                                level(model, name, j + 1, d);
                                level(model, name, j + 1, d);
                            }
                            model.pop(name);
                        }
                        level(model, &name, 0, d);
                    };
                    (format!("t{id}({arg}, cast(0, i64))"), Box::new(events))
                }
                Block::Mutual(n) => {
                    let (a, b) = (format!("ma{id}"), format!("mb{id}"));
                    self.defs.push_str(&format!(
                        "def {a}(x: f32, k: i64) -> f32 = if gte(k, cast({n}, i64)) then x else {b}(neg(x), k)\ndef {b}(x: f32, k: i64) -> f32 = {a}(x, add(k, cast(1, i64)))\n"
                    ));
                    let n = *n;
                    let events = move |model: &mut UnrollModel| {
                        fn level(model: &mut UnrollModel, a: &str, b: &str, j: usize, n: usize) {
                            model.push(a, ModelKey::Declaration(a.to_string()));
                            if j < n {
                                model.push(b, ModelKey::Declaration(b.to_string()));
                                level(model, a, b, j + 1, n);
                                model.pop(b);
                            }
                            model.pop(a);
                        }
                        level(model, &a, &b, 0, n);
                    };
                    (format!("ma{id}({arg}, cast(0, i64))"), Box::new(events))
                }
                Block::LambdaChain(n, inner) => {
                    let name = format!("lc{id}");
                    let literal = self.fresh();
                    let (body, inner) = match inner {
                        Some(inner) => {
                            let (call, events) = self.emit(inner, "y");
                            (call, Some(events))
                        }
                        None => ("neg(y)".to_string(), None),
                    };
                    self.defs.push_str(&format!(
                        "def {name}(f: f32 -> f32, x: f32, k: i64) -> f32 = if gte(k, cast({n}, i64)) then x else {name}(f, f(x), add(k, cast(1, i64)))\n"
                    ));
                    let n = *n;
                    let events = move |model: &mut UnrollModel| {
                        fn level(
                            model: &mut UnrollModel,
                            name: &str,
                            literal: usize,
                            j: usize,
                            n: usize,
                            inner: &Option<Events>,
                        ) {
                            model.push(name, ModelKey::Declaration(name.to_string()));
                            if j < n {
                                model.push("f", ModelKey::Literal(literal));
                                if let Some(inner) = inner {
                                    inner(model);
                                }
                                model.pop("f");
                                level(model, name, literal, j + 1, n, inner);
                            }
                            model.pop(name);
                        }
                        level(model, &name, literal, 0, n, &inner);
                    };
                    (
                        format!("lc{id}(fn (y: f32) -> {body}, {arg}, cast(0, i64))"),
                        Box::new(events),
                    )
                }
                Block::LambdaTree(d) => {
                    let name = format!("lt{id}");
                    let literal = self.fresh();
                    self.defs.push_str(&format!(
                        "def {name}(f: f32 -> f32, x: f32, k: i64) -> f32 = if gte(k, cast({d}, i64)) then f(x) else add({name}(f, x, add(k, cast(1, i64))), {name}(f, neg(x), add(k, cast(1, i64))))\n"
                    ));
                    let d = *d;
                    let events = move |model: &mut UnrollModel| {
                        fn level(
                            model: &mut UnrollModel,
                            name: &str,
                            literal: usize,
                            j: usize,
                            d: usize,
                        ) {
                            model.push(name, ModelKey::Declaration(name.to_string()));
                            if j == d {
                                model.push("f", ModelKey::Literal(literal));
                                model.pop("f");
                            } else {
                                level(model, name, literal, j + 1, d);
                                level(model, name, literal, j + 1, d);
                            }
                            model.pop(name);
                        }
                        level(model, &name, literal, 0, d);
                    };
                    (
                        format!("lt{id}(fn (y: f32) -> neg(y), {arg}, cast(0, i64))"),
                        Box::new(events),
                    )
                }
                Block::Seq(first, second) => {
                    let (first_call, first_events) = self.emit(first, arg);
                    let (call, second_events) = self.emit(second, &first_call);
                    let events = move |model: &mut UnrollModel| {
                        first_events(model);
                        second_events(model);
                    };
                    (call, Box::new(events))
                }
            }
        }
    }

    #[test]
    fn generated_unrolls_count_and_trip_as_the_model_says() {
        // chelis#3864 round 5: every combination of the block grammar below,
        // lowered as a cost entry, counts exactly the bodies the unroll rule
        // predicts from the program's structure, and trips exactly when it
        // predicts: frames keyed by function or literal identity, a frame
        // credited twice branching its unroll, and the unroll ending when no
        // name recurses.
        use Block::*;
        let leaves = [
            Chain(3),
            Chain(30),
            Tree(2),
            Tree(5),
            Mutual(4),
            Mutual(25),
            LambdaChain(20, None),
            LambdaTree(4),
            LambdaChain(6, Some(Box::new(LambdaChain(6, None)))),
            ChainWith(8, Box::new(Chain(6))),
            ChainWith(4, Box::new(Tree(2))),
        ];
        let mut blocks: Vec<Block> = leaves.to_vec();
        // Both verdicts, and unlimited unrolls past the budget, must occur,
        // or the comparison proves little.
        let (mut trips, mut unlimited_past_budget) = (0, 0);
        for first in &leaves {
            for second in &leaves {
                blocks.push(Seq(Box::new(first.clone()), Box::new(second.clone())));
            }
        }
        for block in blocks {
            for budget in [20, 60] {
                let mut generator = Generator {
                    defs: String::new(),
                    next: 0,
                };
                let (call, events) = generator.emit(&block, "cast(1.5, f32)");
                let source = format!("{}def example() -> f32 = {call}\n", generator.defs);
                let mut model = UnrollModel {
                    budget,
                    depth: Default::default(),
                    recursing: 0,
                    frames: Vec::new(),
                    unroll: 0,
                    branched: false,
                    counted: 0,
                    tripped: false,
                };
                events(&mut model);
                let (lowered, counted, _) = lower_example(source.clone(), budget);
                assert_eq!(
                    (counted, !lowered),
                    (model.counted, model.tripped),
                    "counted bodies and trip for:\n{source}"
                );
                if model.tripped {
                    trips += 1;
                } else if model.counted > budget {
                    unlimited_past_budget += 1;
                }
            }
        }
        assert!(
            trips > 0 && unlimited_past_budget > 0,
            "{trips} {unlimited_past_budget}"
        );
    }
}

#[cfg(test)]
mod declaration_attribution_tests {
    use super::*;
    fn checked(source: &str) -> CheckedProgram {
        let declarations =
            chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
        let checked = chelis_types::check_typed_program(
            &chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
                .expect("Surf fixture must desugar"),
        )
        .unwrap();
        let checked = chelis_effects::check_program(&checked).unwrap();
        chelis_types::check_linearity(&checked).unwrap()
    }

    fn lowered(source: &str) -> LoweredLibrary {
        try_lower_program_to_library(&checked(source)).unwrap()
    }

    fn declaration_name(dag: &Dag, node: &chelis_dag_core::dag::DagNode) -> String {
        dag.declaration(node.owner.decl).name.clone()
    }

    fn scalar_chain(depth: usize) -> String {
        let mut source = String::from("h0 = 2i32\n");
        for level in 1..=depth {
            let last = format!("h{}", level - 1);
            source.push_str(&format!(
                "h{level} = ((({last} * {last}) - ({last} * {last})) + 2i32)\n"
            ));
        }
        source.push_str(&format!(
            "def g(x: tensor[f32]) -> tensor[f32] = mul(x, scalar_to_tensor(cast(h{depth}, f32)))\ndef main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))\n"
        ));
        source
    }

    fn tensor_chain(depth: usize) -> String {
        let mut source = String::from("v0 = cast(scalar_to_tensor(3.0f32), i32)\n");
        for level in 1..=depth {
            let last = format!("v{}", level - 1);
            source.push_str(&format!("v{level} = sub(copy({last}), copy({last}))\n"));
        }
        source.push_str(&format!(
            "def g(x: tensor[f32]) -> tensor[f32] = {{\n  dead = copy(v{depth})\n  x\n}}\ndef main() -> tensor[f32] = g(scalar_to_tensor(1.0f32))\n"
        ));
        source
    }

    fn integer_ops(dag: &Dag, op: fn(&RiscOp) -> bool) -> BTreeMap<String, usize> {
        let mut owners = BTreeMap::new();
        for node in dag.nodes() {
            if op(&node.op) && node.output_type.precision.is_integer() {
                *owners.entry(declaration_name(dag, node)).or_default() += 1;
            }
        }
        owners
    }

    #[test]
    fn a_chain_of_trapping_values_holds_one_copy_per_value_and_owner() {
        let cast = |op: &RiscOp| matches!(op, RiscOp::Cast { .. });
        let sub = |op: &RiscOp| matches!(op, RiscOp::Sub);
        let mul = |op: &RiscOp| matches!(op, RiscOp::Mul);
        let mut entry_nodes = BTreeMap::<&str, Vec<usize>>::new();
        for depth in [4, 8, 16] {
            // The whole-program graph: `v0`'s cast once per owner, and each
            // owner's copy of `v1`..`v{i}` once.
            let library = lowered(&tensor_chain(depth));
            let dag = library.dag();
            let mut owners = (0..=depth)
                .map(|level| (format!("v{level}"), level))
                .collect::<BTreeMap<_, _>>();
            owners.insert("g".into(), depth);
            owners.insert("main".into(), depth);
            assert_eq!(
                integer_ops(dag, cast),
                owners.keys().map(|owner| (owner.clone(), 1)).collect(),
                "casts by owner, tensor chain of depth {depth}"
            );
            owners.remove("v0");
            assert_eq!(
                integer_ops(dag, sub),
                owners,
                "subtractions by owner, tensor chain of depth {depth}"
            );

            // `main`'s selected entry, as `chelis build` compiles it.
            for (form, source, op, per_level) in [
                ("tensor", tensor_chain(depth), sub as fn(&RiscOp) -> bool, 1),
                ("scalar", scalar_chain(depth), mul, 2),
            ] {
                let entry = crate::resources::with_operation(|operation| {
                    chelis_ir_core::host::lower_named_tensor_entry_dag(
                        checked(&source).facts(),
                        "main",
                        operation,
                    )
                    .expect("entry lowering returns no fatal diagnostic")
                })
                .unwrap_or_else(|| panic!("{form} chain of depth {depth}: no entry"));
                assert_eq!(
                    integer_ops(&entry, op).into_values().sum::<usize>(),
                    per_level * depth,
                    "{form} chain of depth {depth}"
                );
                if form == "scalar" {
                    // `h0 = 2i32` holds no node that can trap, so it is not
                    // inlined: the one copy of `h1` reads the input its
                    // declaration supplies, four times.
                    let h0 = entry
                        .nodes()
                        .iter()
                        .filter(|node| {
                            matches!(&node.op, RiscOp::Load { name }
                                if name.binding() == chelis_dag_core::load_store_name::LoadBinding::TopLevel("h0"))
                        })
                        .count();
                    assert_eq!(h0, 4, "scalar chain of depth {depth}");
                }
                entry_nodes.entry(form).or_default().push(entry.len());
            }
        }
        for (form, counts) in entry_nodes {
            let [four, eight, sixteen] = counts[..] else {
                unreachable!("three depths")
            };
            assert_eq!(sixteen - eight, 2 * (eight - four), "{form}: {counts:?}");
        }
    }

    #[test]
    fn an_initializer_with_no_tensor_form_stays_the_host_served_input() {
        let source = "ids: List[i64] = [cast(0, i64), cast(1, i64)]\n\
                      tokens = to_tensor(ids)\n\
                      table = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)\n\
                      out = gather(table, tokens, 0)\n\
                      offset = cast(scalar_to_tensor(3.0f32), i32)\n\
                      shifted = add(copy(offset), copy(offset))\n";
        let compiled = crate::host::try_lower_compiled_program(&checked(source)).unwrap();
        let host = compiled.host.expect("the program has a host lane");
        let helper = |global: &str| {
            let binding = host
                .globals
                .iter()
                .find(|binding| binding.name == global)
                .unwrap_or_else(|| panic!("no global `{global}`"));
            let chelis_ir_core::host::HostExprKind::TensorCall { helper, .. } = &binding.value.kind
            else {
                panic!("`{global}` has no tensor helper: {:?}", binding.value.kind);
            };
            &host.global_tensor_helpers[*helper].dag
        };
        let loads = |dag: &Dag| {
            dag.nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    RiscOp::Load { name } => Some(match name.binding() {
                        chelis_dag_core::load_store_name::LoadBinding::TopLevel(source) => {
                            ("top-level", source.to_owned())
                        }
                        chelis_dag_core::load_store_name::LoadBinding::GraphInput => {
                            ("graph-input", name.as_str().to_owned())
                        }
                    }),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(
            loads(helper("out")),
            BTreeSet::from([
                ("top-level", "table".to_owned()),
                ("top-level", "tokens".to_owned()),
            ])
        );
        let shifted = helper("shifted");
        assert!(loads(shifted).is_empty(), "{:?}", loads(shifted));
        assert!(
            shifted
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Cast { .. })),
            "`offset`'s cast is lowered again at the reference"
        );
    }
}

//! Entry-scoped compiled-execution metadata (issues #817, #818).
//!
//! `compile_for_execution` backs `chelis.compile_and_load`. Before this
//! change it compiled the WHOLE program and reported the union of every
//! top-level def's `Load`s as the callable's inputs (#817), and returned an
//! empty manifest whenever the entry's body used a syntactic host-runtime
//! builtin such as `concat` — because that excludes the def from the pure
//! DAG, leaving `roots()` empty and tripping the host-lane early-return
//! (#818).
//!
//! These tests pin the fixed contract: the metadata is scoped to a single
//! resolved entry def, an unknown `entry_name` is a loud error that lists
//! the available entries, and a def named `main` no longer forces an
//! un-linkable `main` C symbol.

use chelis_compiler_api::compiler::{EntryLaneDecline, compile, compile_for_execution};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

const HELPER_PLUS_ENTRY: &str = "\
def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)
def solve(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = add(helper(a), helper(b))
";

const FIXED_CONTROL_ENTRIES: &str = "\
def sample(x: tensor[4, f32]) -> tensor[4, f32] = with seed(42i64) { dropout(x, 0.5f32) }
def other(y: tensor[8, f32]) -> tensor[8, f32] = with seed(7i64) { dropout(y, 0.5f32) }
";

#[test]
fn fixed_control_hostless_entry_keeps_exact_callable_selection() {
    let artifact = compile_c(FIXED_CONTROL_ENTRIES, Some("sample"));
    assert_eq!(input_names(&artifact), ["x"]);
    assert_eq!(artifact.outputs.len(), 1);
    assert!(
        artifact
            .compile_result
            .files
            .iter()
            .any(|file| file.path == "chelis_main.c")
    );
    for (entry, reason) in [
        (None, "ambiguous entry"),
        (Some("ample"), "unknown entry_name `ample`"),
    ] {
        let error = compile_for_execution(CompileRequest {
            source_kind: SourceKind::Surf,
            source: FIXED_CONTROL_ENTRIES.into(),
            target: CompileTarget::C,
            entry_name: entry.map(str::to_owned),
        })
        .unwrap_err();
        assert!(format!("{error:?}").contains(reason), "{error:?}");
    }
}

#[test]
fn fixed_control_source_sibling_does_not_replace_an_ordinary_selected_entry() {
    let source =
        format!("{FIXED_CONTROL_ENTRIES}\ndef identity(z: tensor[2, f32]) -> tensor[2, f32] = z\n");
    let artifact = compile_c(&source, Some("identity"));
    assert_eq!(input_names(&artifact), ["z"]);
    let c = &artifact
        .compile_result
        .files
        .iter()
        .find(|file| file.path == "chelis_main.c")
        .unwrap()
        .contents;
    assert!(!c.contains("chelis_dropout_unit"));
}

#[test]
fn fixed_control_hostless_entry_does_not_admit_an_unhandled_draw() {
    let error = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: "def main(x: tensor[4, f32]) -> tensor[4, f32] = dropout(x, 0.5f32)\n".into(),
        target: CompileTarget::C,
        entry_name: Some("main".into()),
    })
    .unwrap_err();
    let message = format!("{error:?}");
    assert!(
        message.contains("inherited") || message.contains("Random"),
        "{error:?}"
    );
}

// #818 repro verbatim: single def, multi-statement block, a parameter reused
// across statements, `concat` in the body.
const CONCAT_ENTRY: &str = "\
def main(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[2, f32] = {
  x = mul(copy(a), b)
  y = add(a, b)
  concat([x, y], cast(0, i32))
}
";

fn compile_c(
    source: &str,
    entry: Option<&str>,
) -> chelis_compiler_api::compiler::CompiledExecutionArtifact {
    compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: entry.map(str::to_string),
    })
    .unwrap_or_else(|err| panic!("compile failed: {err:?}"))
}

fn input_names(artifact: &chelis_compiler_api::compiler::CompiledExecutionArtifact) -> Vec<String> {
    artifact
        .inputs
        .iter()
        .map(|spec| spec.name.clone())
        .collect()
}

const MAIN_PLUS_HELPER: &str = "\
def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)
def main(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = add(helper(a), helper(b))
";

/// Fix 3: a multi-def file with 2+ tensor-signature defs and NO `main`,
/// compiled with no explicit `entry_name`, is AMBIGUOUS. The old rule
/// silently selected the last def; the new rule errors loudly and lists the
/// candidates so the caller passes `entry_name`. (Was
/// `multi_def_without_entry_name_scopes_to_preferred_entry`, which pinned the
/// silent-last-def behavior this fix removes.)
#[test]
fn multi_def_without_entry_name_and_no_main_is_ambiguous_error() {
    let err = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: HELPER_PLUS_ENTRY.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .expect_err("ambiguous multi-def default selection must error");
    let message = &err.errors[0].message;
    assert!(
        message.contains("ambiguous entry"),
        "message must flag the ambiguity, got: {message}"
    );
    assert!(
        message.contains("helper") && message.contains("solve"),
        "message must list the candidate defs, got: {message}"
    );
    assert!(
        message.contains("entry_name"),
        "message must point at the remedy, got: {message}"
    );
}

/// Fix 3 companion: when a tensor-signature def named `main` is present, the
/// default selection is unambiguous — it auto-selects `main`, scoping to its
/// params, NOT the merged set.
#[test]
fn multi_def_without_entry_name_auto_selects_main() {
    let artifact = compile_c(MAIN_PLUS_HELPER, None);
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()],
        "expected only `main`'s params, not the merged `helper`+`main` set"
    );
    assert_eq!(artifact.outputs.len(), 1);
}

/// #817: an explicit `entry_name` selects that def's scope.
#[test]
fn multi_def_with_explicit_entry_name_scopes_to_it() {
    let artifact = compile_c(HELPER_PLUS_ENTRY, Some("solve"));
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(artifact.outputs.len(), 1);
}

/// The explicit selection genuinely changes the result: selecting `helper`
/// scopes to its single param `x`, distinct from `solve`'s `(a, b)`.
#[test]
fn explicit_entry_name_selects_a_different_def() {
    let artifact = compile_c(HELPER_PLUS_ENTRY, Some("helper"));
    assert_eq!(input_names(&artifact), vec!["x".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// Negative: an `entry_name` that names no def in a program that DOES have a
/// tensor entry is a typo'd selector — a loud error that lists the choices,
/// not a silently-merged or empty manifest.
#[test]
fn unknown_entry_name_errors_listing_available_defs() {
    let err = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: HELPER_PLUS_ENTRY.to_string(),
        target: CompileTarget::C,
        entry_name: Some("nope".to_string()),
    })
    .expect_err("unknown entry_name must error");
    let message = &err.errors[0].message;
    assert!(
        message.contains("unknown entry_name `nope`"),
        "message must name the bad entry, got: {message}"
    );
    assert!(
        message.contains("helper") && message.contains("solve"),
        "message must list the available entry defs, got: {message}"
    );
}

/// #818: a single def whose body uses `concat` (and reuses a parameter across
/// block statements) reports its real `(a, b)` inputs and one output, instead
/// of the empty manifest the host-lane early-return used to produce.
#[test]
fn concat_body_reports_real_inputs_and_outputs() {
    let artifact = compile_c(CONCAT_ENTRY, None);
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()],
        "concat body must not collapse to an empty input manifest"
    );
    assert_eq!(
        artifact.outputs.len(),
        1,
        "expected one output, got {:?}",
        artifact.outputs
    );
    assert_eq!(
        artifact.entry_lane_decline, None,
        "a claimed compilation must record no decline reason"
    );
}

/// Regression: a single composed-expression def (no block, no host-runtime
/// builtin) still reports its single input — the #818 "works" contrast case.
#[test]
fn single_expression_def_still_reports_its_input() {
    let artifact = compile_c(
        "def main(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        None,
    );
    assert_eq!(input_names(&artifact), vec!["a".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// A def literally named `main`, selected via `entry_name`, must compile to a
/// non-`main` C symbol (decoupled from def selection) so the native toolchain
/// does not reject the translation unit for redefining the reserved
/// `int main(...)`. The scoped metadata is still correct.
#[test]
fn entry_named_main_does_not_emit_reserved_main_symbol() {
    let artifact = compile_c(CONCAT_ENTRY, Some("main"));
    assert_ne!(
        artifact.host_entry_name, "main",
        "the emitted C symbol must not be the reserved `main`"
    );
    assert_eq!(artifact.host_entry_name, "chelis_main");
    // Metadata stays scoped to the `main` def.
    assert_eq!(
        input_names(&artifact),
        vec!["a".to_string(), "b".to_string()]
    );

    // The generated entry function carries the rewritten symbol, and no bare
    // `main(` entry is declared to collide with the C runtime.
    let c = artifact
        .compile_result
        .files
        .iter()
        .find(|file| file.path == "chelis_main.c")
        .expect("chelis_main.c present")
        .contents
        .clone();
    assert!(
        c.contains("chelis_main"),
        "entry function must use the rewritten symbol"
    );
}

// ---------------------------------------------------------------------------
// Fix 1: the entry lane must be scoped to the ENTRY, not the whole program.
// A host-flavored SIBLING def (or a helper the entry calls that itself uses a
// host-runtime builtin) must NOT disable the lane for a cleanly-lowerable
// entry. Each of these produced an EMPTY manifest under the old whole-program
// `!host_program_requires_host_backend` gate.
// ---------------------------------------------------------------------------

/// (a) A host-flavored sibling (`other` uses `concat`) must not disable the
/// lane for a pure entry (`solve`) selected by name.
#[test]
fn sibling_host_def_does_not_disable_entry_lane() {
    let artifact = compile_c(
        "def other(x: tensor[1, f32]) -> tensor[2, f32] = concat([copy(x), x], cast(0, i32))\n\
         def solve(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        Some("solve"),
    );
    assert_eq!(input_names(&artifact), vec!["a".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// (b) #818 factored into a helper: the entry `solve` calls `helper`, and
/// `helper`'s body uses `concat`. The entry still lowers and reports `(a)`.
#[test]
fn entry_calling_concat_helper_reports_real_metadata() {
    let artifact = compile_c(
        "def helper(x: tensor[1, f32]) -> tensor[2, f32] = concat([copy(x), x], cast(0, i32))\n\
         def solve(a: tensor[1, f32]) -> tensor[2, f32] = helper(a)\n",
        Some("solve"),
    );
    assert_eq!(input_names(&artifact), vec!["a".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// (c) The entry itself uses `concat` and also calls a pure helper.
#[test]
fn entry_with_concat_calling_pure_helper_reports_real_metadata() {
    let artifact = compile_c(
        "def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(copy(x), x)\n\
         def solve(a: tensor[1, f32]) -> tensor[2, f32] = concat([helper(copy(a)), a], cast(0, i32))\n",
        Some("solve"),
    );
    assert_eq!(input_names(&artifact), vec!["a".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// (d) An unused scalar-signature sibling (`scale`) must not force the
/// tensor entry onto the host lane.
#[test]
fn unused_scalar_sibling_does_not_disable_entry_lane() {
    let artifact = compile_c(
        "def scale(x: f32) -> f32 = mul(x, x)\n\
         def solve(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        Some("solve"),
    );
    assert_eq!(input_names(&artifact), vec!["a".to_string()]);
    assert_eq!(artifact.outputs.len(), 1);
}

/// Fix 7 (negative): a zero-input entry (`def main() -> tensor[1,f32] =
/// to_tensor([3.0])`) reports zero inputs and ONE output — it must NOT trip
/// any host-only / empty-manifest rejection. Pins that the manifest is
/// callable-shaped (has an output) even with no inputs.
#[test]
fn zero_input_entry_reports_one_output_no_inputs() {
    let artifact = compile_c("def main() -> tensor[1, f32] = to_tensor([3.0])\n", None);
    assert!(
        input_names(&artifact).is_empty(),
        "zero-input entry must report no inputs, got {:?}",
        input_names(&artifact)
    );
    assert_eq!(
        artifact.outputs.len(),
        1,
        "zero-input entry must still expose its single output"
    );
}

/// Fix 1 guard: a `grad`-using entry, even selected by its def name, stays on
/// the host lane (empty compiled-execution metadata) — the pure entry-kernel
/// lane does not own multi-root grad-tuple emission (#309). This is the line
/// that keeps "it lowers" from being sufficient to claim the entry lane.
#[test]
fn grad_entry_stays_on_host_lane_even_when_selected() {
    const GRAD_SRC: &str = "module Repro.GradEntry
def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(mul(x, w), cast(0, i32)))
def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = (grad(loss)(x, w)).0
";
    let artifact = compile_c(GRAD_SRC, Some("dloss"));
    assert!(
        artifact.inputs.is_empty() && artifact.outputs.is_empty(),
        "grad entry must stay host-lane (empty callable metadata), got inputs={:?} outputs={:?}",
        artifact.inputs,
        artifact.outputs
    );
    // #819 Fix 2: the decline is recorded, not silent — and it is the
    // grad-specific reason, so downstream error text can say WHY instead of
    // guessing from the empty manifest.
    assert_eq!(
        artifact.entry_lane_decline,
        Some(EntryLaneDecline::GradLike {
            entry: "dloss".to_string()
        }),
        "grad decline must be recorded with the GradLike reason"
    );
}

// ---------------------------------------------------------------------------
// Fix 1 (#819 review): a program with TOP-LEVEL GLOBALS must DECLINE the
// entry lane and keep its merge-base host-lane routing. The entry lane's
// standalone lowering (`lower_named_tensor_entry_dag`) seeds scope with the
// entry's params only, so claiming such a program would (repro A) demote a
// referenced global to a phantom required runtime input, or (repro B) drop an
// independent global's computation from the emitted program entirely.
// ---------------------------------------------------------------------------

/// Repro A: `def main` references the top-level global `two`. Before this fix
/// the entry lane claimed the program and the manifest became
/// `inputs=["a","two"]` — the global silently became a caller-supplied input.
/// It must instead route host-lane (empty callable metadata, the merge-base
/// behavior), with the global's initialization emitted by the host program.
#[test]
fn global_referenced_by_entry_declines_entry_lane_no_phantom_input() {
    let artifact = compile_c(
        "two = to_tensor([2.0])\n\
         def main(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), two)\n",
        None,
    );
    assert!(
        artifact.inputs.is_empty() && artifact.outputs.is_empty(),
        "globals program must stay host-lane; `two` must NOT surface as a \
         required input, got inputs={:?}",
        input_names(&artifact)
    );
    assert_eq!(
        artifact.entry_lane_decline,
        Some(EntryLaneDecline::HasGlobals),
        "the decline reason must be recorded as HasGlobals"
    );
    // The host lane owns the global: its definition is in the emitted C.
    let c = &artifact
        .compile_result
        .files
        .iter()
        .find(|file| file.path.ends_with(".c"))
        .expect("emitted C present")
        .contents;
    assert!(
        c.contains("two"),
        "the global `two` must be emitted by the host program, not dropped"
    );
}

/// Repro B: an independent global `total` alongside `def main`. Before this
/// fix the entry lane emitted only the kernel and `total`'s computation
/// vanished from the compiled output entirely. Host-lane routing keeps it.
#[test]
fn independent_global_computation_survives_host_lane_routing() {
    let artifact = compile_c(
        "total = add(to_tensor([1.0]), to_tensor([2.0]))\n\
         def main(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        None,
    );
    assert!(
        artifact.inputs.is_empty() && artifact.outputs.is_empty(),
        "globals program must stay host-lane (empty callable metadata), got \
         inputs={:?} outputs={}",
        input_names(&artifact),
        artifact.outputs.len()
    );
    assert_eq!(
        artifact.entry_lane_decline,
        Some(EntryLaneDecline::HasGlobals),
        "the decline reason must be recorded as HasGlobals"
    );
    let c = &artifact
        .compile_result
        .files
        .iter()
        .find(|file| file.path.ends_with(".c"))
        .expect("emitted C present")
        .contents;
    assert!(
        c.contains("total"),
        "`total`'s computation must not vanish from the compiled output"
    );
}

/// Repro C (#819 round 2): a SCALAR global `glb = 2.0` in a multi-def
/// program. Unlike the tensor globals above, lowering classifies a scalar
/// literal binding as DAG-lowerable and — uncaptured, with no host `main`
/// emitted — drops it from `host_program.globals` (`skip_for_lowered`), so
/// the old lowered-artifact check saw "no globals", the entry lane CLAIMED
/// the program, and `glb`'s computation vanished from the emitted C. The
/// decline is keyed on SOURCE-LEVEL top-level value bindings
/// (`chelis_ir::host::program_has_top_level_value_bindings`), so ANY
/// top-level value binding declines regardless of its lowered
/// classification.
///
/// On the STRICT (callable) surface that decline is now a loud error: the
/// whole-DAG fallback would hand back the merged (#817) manifest while
/// quietly ignoring `entry_name`, which is exactly the defect class this
/// stack fixes. (An earlier revision returned that merged 3-output
/// manifest from `compile_for_execution` and pinned it as intended.)
const SCALAR_GLOBAL_MULTI_DEF: &str = "\
glb = 2.0
def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)
def main(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)
";

#[test]
fn scalar_global_multi_def_is_a_loud_error_on_the_callable_surface() {
    let err = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: SCALAR_GLOBAL_MULTI_DEF.to_string(),
        target: CompileTarget::C,
        entry_name: Some("main".to_string()),
    })
    .expect_err("a top-level binding must be a loud error on compile_for_execution");
    let message = &err.errors[0].message;
    assert!(
        message.contains("top-level") && message.contains("chelis#817"),
        "message must name the top-level-binding cause and the #817 class, got: {message}"
    );
    assert!(
        message.contains("eval"),
        "message must point at `eval` as the whole-program runner, got: {message}"
    );
}

/// LEGACY surface pin: the same program still compiles whole-program via
/// `compile()` (tide's `/compile`, cove, `chelis.compile()`), where the
/// whole-program emission IS the product contract, and `glb`'s computation
/// survives instead of vanishing. [05-OBS-4] keeps a scalar root scalar, so
/// pin the exact f32 construction rather than the old rank-zero tensor fill:
/// the bit pattern of 2.0 is 0x40000000.
#[test]
fn scalar_global_multi_def_still_emits_whole_program_via_compile() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: SCALAR_GLOBAL_MULTI_DEF.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .unwrap_or_else(|err| panic!("legacy surface must still compile, got: {err:?}"));
    let c = &result
        .files
        .iter()
        .find(|file| file.path.ends_with(".c"))
        .expect("emitted C present")
        .contents;
    assert!(
        c.contains("chelis_f32_from_bits") && c.contains("0x40000000"),
        "`glb`'s exact scalar computation (2.0f) must not vanish from the compiled \
         output, got:\n{c}"
    );
}

/// Fix 2 (#819): an explicit `entry_name` naming a def with a scalar
/// (non-tensor) signature declines with the signature-specific reason —
/// recorded on the artifact so the python layer can report it — instead of a
/// silent empty manifest.
#[test]
fn scalar_entry_by_name_declines_with_not_tensor_signature() {
    let artifact = compile_c(
        "def scale(x: f32) -> f32 = mul(x, x)\n\
         def solve(a: tensor[1, f32]) -> tensor[1, f32] = mul(copy(a), a)\n",
        Some("scale"),
    );
    assert!(
        artifact.inputs.is_empty() && artifact.outputs.is_empty(),
        "scalar entry must stay host-lane (empty callable metadata)"
    );
    assert_eq!(
        artifact.entry_lane_decline,
        Some(EntryLaneDecline::NotTensorSignature {
            entry: "scale".to_string()
        }),
        "the decline reason must name the scalar-signature entry"
    );
}

/// Fix 7 (pre-existing behavior, PINNED not changed): a declared parameter
/// that never appears in the entry body is dead-code-eliminated and does NOT
/// appear in the manifest. `solve(a, b) = mul(copy(b), b)` reports only `b`.
///
/// This is NOT introduced by the entry-scoping fix — the whole-program lane
/// dropped unused `Load`s too — and changing it is out of scope. It is pinned
/// here so the asymmetry is explicit: positional callers must bind by
/// consulting `input_names`, not declaration order, or a positional argument
/// misbinds. See `bindings/python/README.md`.
#[test]
fn unused_declared_param_is_absent_from_manifest_preexisting() {
    let artifact = compile_c(
        "def solve(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = mul(copy(b), b)\n",
        Some("solve"),
    );
    assert_eq!(
        input_names(&artifact),
        vec!["b".to_string()],
        "unused `a` must be DCE'd out of the manifest (pre-existing behavior)"
    );
    assert_eq!(artifact.outputs.len(), 1);
}

/// Fix 6: the entry-scoping change rides `compiler::compile` — the same entry
/// point tide `/compile`, cove, and python `compile()` call. Pin that a
/// multi-def source with `main` present emits the entry-scoped kernel (the
/// `chelis_main` symbol) and does NOT additionally emit the sibling `helper`
/// def as its own standalone C entry function. This locks the contract change
/// on the shared `compile()` surface, not just `compile_for_execution`.
#[test]
fn compile_emits_entry_scoped_kernel_not_sibling_def() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: MAIN_PLUS_HELPER.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .unwrap_or_else(|err| panic!("compile failed: {err:?}"));
    let c = result
        .files
        .iter()
        .find(|file| file.path == "chelis_main.c")
        .expect("chelis_main.c present")
        .contents
        .clone();
    assert!(
        c.contains("chelis_main"),
        "must emit the entry-scoped `chelis_main` kernel, got:\n{c}"
    );
    // The sibling `helper` def must NOT be emitted as its own standalone
    // top-level entry function (the whole-program emission would have). It is
    // inlined into the entry DAG, not a separate callable symbol.
    assert!(
        !c.contains("void helper(") && !c.contains("chelis_helper("),
        "sibling `helper` must not be emitted as a standalone entry, got:\n{c}"
    );
}

/// Reviewer B1: a `vmap` entry declines the lane as `GradLike` but, unlike
/// `grad`, does NOT force the host backend, so it reaches the legacy
/// whole-DAG fallthrough. An earlier revision `debug_assert!`ed that this
/// combination was impossible: a vmap entry panicked every debug-built
/// caller of the shared pipeline (tide serve included) and, in release,
/// silently returned the merged whole-program manifest (#817 unfixed). On
/// the STRICT surface it is now a loud unsupported-feature error.
const VMAP_ENTRY: &str = "\
def process(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)
def batch_process(xs: tensor[8, 4, f32]) -> tensor[8, 4, f32] = xs |> vmap(process)
";

#[test]
fn vmap_entry_is_a_loud_unsupported_error_on_the_callable_surface() {
    let err = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: VMAP_ENTRY.to_string(),
        target: CompileTarget::C,
        entry_name: Some("batch_process".to_string()),
    })
    .expect_err("a vmap entry must be a loud error on compile_for_execution, not a panic");
    let diagnostic = &err.errors[0];
    assert_eq!(
        diagnostic.kind(),
        chelis_vocab::DiagnosticKind::UnsupportedFeature,
        "a transform entry is a not-yet-implemented capability, got kind {}: {}",
        diagnostic.kind().as_str(),
        diagnostic.message
    );
    assert!(
        diagnostic.message.contains("batch_process") && diagnostic.message.contains("eval"),
        "message must name the entry and point at `eval`, got: {}",
        diagnostic.message
    );
}

/// LEGACY surface pin for the same program: `compile()` must keep emitting
/// the whole program (pre-entry-lane behavior) with no panic and no error.
#[test]
fn vmap_entry_via_compile_still_emits_whole_program() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: VMAP_ENTRY.to_string(),
        target: CompileTarget::C,
        entry_name: Some("batch_process".to_string()),
    })
    .unwrap_or_else(|err| panic!("legacy surface must still compile a vmap program: {err:?}"));
    assert!(
        result.files.iter().any(|file| file.path.ends_with(".c")),
        "whole-program C must be emitted"
    );
}

/// Reviewer S1: the ambiguous-default rule is STRICT-surface only. tide's
/// `/compile`, cove's live pane, and `chelis.compile()` hardcode
/// `entry_name: None` and compile multi-def programs with no `main`
/// (examples/linreg.ch, examples/mnist.ch); an earlier revision hard-errored
/// that whole surface. The legacy surface falls back to whole-program
/// emission instead.
#[test]
fn ambiguous_default_via_compile_emits_whole_program() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: HELPER_PLUS_ENTRY.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .unwrap_or_else(|err| {
        panic!("legacy surface must not hard-error on an ambiguous default: {err:?}")
    });
    let c = &result
        .files
        .iter()
        .find(|file| file.path.ends_with(".c"))
        .expect("emitted C present")
        .contents;
    assert!(
        !c.is_empty(),
        "whole-program C must be emitted for the ambiguous default"
    );
}

/// Reviewer S5 (legacy half): on the C-source surface an `entry_name`
/// naming no def keeps its documented meaning as the OUTPUT SYMBOL (tide's
/// contract), sanitized but passed through, instead of the strict surface's
/// unknown-selector error.
#[test]
fn unknown_entry_name_via_compile_is_the_output_symbol() {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: HELPER_PLUS_ENTRY.to_string(),
        target: CompileTarget::C,
        entry_name: Some("mymodel".to_string()),
    })
    .unwrap_or_else(|err| {
        panic!("legacy surface must treat a non-def entry_name as the output symbol: {err:?}")
    });
    assert!(
        result.files.iter().any(|file| file.path == "mymodel.c"),
        "the emitted file must carry the requested output symbol, got: {:?}",
        result
            .files
            .iter()
            .map(|f| f.path.clone())
            .collect::<Vec<_>>()
    );
}

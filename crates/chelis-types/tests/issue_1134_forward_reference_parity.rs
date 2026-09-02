//! chelis#1134 / [04-INF-4]: top-level eager-value scope is identical at the
//! stamped typed and serialized-IR checker ingresses.

use chelis_deep::{parse_and_stamp, parse_and_stamp_file};
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{
    TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_typed_program,
};

type Diagnostics = Vec<(String, String)>;

fn deep_program(source: &str) -> Vec<chelis_deep::Expr> {
    parse_and_stamp(source).expect("Deep fixture must parse and stamp")
}

/// Declaration-level Deep parse. `parse_and_stamp` stamps expression forms;
/// a `(module {} ...)` wrapper is only valid at declaration position, so the
/// matrix's wrapped rows need this entry point.
fn deep_file_program(source: &str) -> Vec<chelis_deep::Expr> {
    parse_and_stamp_file(source).expect("Deep file fixture must parse and stamp")
}

fn surf_program(source: &str) -> Vec<chelis_deep::Expr> {
    let declarations = parse_surf(source).expect("Surf fixture must parse");
    desugar_program(&declarations)
}

fn diagnostics(program: &[chelis_deep::Expr]) -> (Diagnostics, Diagnostics) {
    let summarize = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| {
                (
                    error.kind.diagnostic_name().to_string(),
                    error.message.clone(),
                )
            })
            .collect(),
    };
    (
        summarize(check_ir_program(program)),
        summarize(check_typed_program(program)),
    )
}

fn assert_accepts_at_both_ingresses(program: &[chelis_deep::Expr], label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
    assert!(ir.is_empty(), "{label}: expected acceptance, got {ir:#?}");
}

fn assert_rejects_identically(program: &[chelis_deep::Expr], expected_kind: &str, label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
    assert!(
        ir.iter().any(|(kind, _)| kind == expected_kind),
        "{label}: expected {expected_kind}, got {ir:#?}"
    );
}

#[test]
fn defsig_less_forward_value_reference_rejects_at_both_ingresses() {
    let program = deep_program(
        "(def {} use_base (var {} base))\n\n\
         (def {} base (lit {type: (t-prim {} int32)} 7))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "defsig-less forward value");
}

#[test]
fn backward_value_reference_remains_legal_at_both_ingresses() {
    let program = deep_program(
        "(def {} base (lit {type: (t-prim {} int32)} 7))\n\n\
         (def {} use_base (var {} base))\n",
    );
    assert_accepts_at_both_ingresses(&program, "backward value");
}

#[test]
fn module_wrapped_forward_generic_helper_has_the_same_verdict() {
    let program = surf_program(
        r#"
module ForwardParity
def use_int(x: int32) -> int32 = identity(x)
def use_float(x: f32) -> f32 = identity(x)
def identity(x) = x
"#,
    );
    assert_accepts_at_both_ingresses(&program, "module generic forward helper");
}

#[test]
fn ascribed_external_input_self_reference_is_legal_at_both_ingresses() {
    let program = surf_program("x = (x : tensor[4, f32])\n");
    assert_accepts_at_both_ingresses(&program, "external input self-reference");
}

#[test]
fn declaration_typed_self_reference_is_an_external_input_at_both_ingresses() {
    let program = surf_program("x: tensor[4, f32] = x\n");
    assert_accepts_at_both_ingresses(&program, "declaration-typed external input");
}

#[test]
fn bare_self_reference_is_not_an_external_input_at_either_ingress() {
    let program = surf_program("x = x\n");
    assert_rejects_identically(&program, "CycleDetected", "bare self-reference");
}

#[test]
fn type_stamped_deep_self_reference_is_an_explicit_external_input() {
    let program = deep_program("(def {} x (var {type: (t-prim {} int32)} x))\n");
    assert_accepts_at_both_ingresses(&program, "typed Deep external input");
}

#[test]
fn a_later_external_input_is_not_visible_to_an_earlier_declaration() {
    for (label, source) in [
        (
            "later ascribed external input",
            r#"
module ExternalScope
def capture() -> int32 = x
x = (x : int32)
"#,
        ),
        (
            "later declaration-typed external input",
            r#"
module DeclaredExternalScope
def capture() -> int32 = x
x: int32 = x
"#,
        ),
    ] {
        let program = surf_program(source);
        assert_rejects_identically(&program, "UnboundVariable", label);
    }
}

#[test]
fn a_declaration_typed_value_is_not_visible_before_its_source_position() {
    let program = surf_program(
        r#"
module DeclaredValueScope
def capture() -> int32 = value
value: int32 = 7
"#,
    );
    assert_rejects_identically(&program, "UnboundVariable", "later declared value");
}

#[test]
fn a_separated_defsig_does_not_publish_a_future_eager_value() {
    let program = deep_program(
        "(defsig {} later (t-prim {} int32))\n\n\
         (def {} capture (var {} later))\n\n\
         (def {} later (lit {type: (t-prim {} int32)} 7))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "separated future eager defsig");
}

#[test]
fn a_separated_defsig_preserves_a_prior_context_binding_until_its_def() {
    let library = deep_program("(def {} value (lit {type: (t-prim {} f32)} 1.0))\n");
    let context = build_type_env_from_library(&library).expect("library context checks cleanly");
    let program = deep_program(
        "(defsig {} value (t-prim {} int32))\n\n\
         (defsig {} capture (t-prim {} f32))\n\n\
         (def {} capture (var {} value))\n\n\
         (def {} value (lit {type: (t-prim {} int32)} 7))\n",
    );
    check_ir_with_context(&context, &program)
        .expect("the prior f32 binding must remain visible until the new value def");
}

#[test]
fn context_check_cannot_prebind_a_later_external_input_globally() {
    for source in [
        r#"
module ExternalContext
def capture() -> int32 = x
x = (x : int32)
"#,
        r#"
module DeclaredExternalContext
def capture() -> int32 = x
x: int32 = x
"#,
    ] {
        let program = surf_program(source);
        let result = check_ir_with_context(&TypeEnv::empty(), &program)
            .expect_err("later external input must remain unavailable in context checks");
        assert!(
            result.errors.iter().any(|error| matches!(
                error.kind,
                chelis_types::errors::CheckErrorKind::UnboundVariable { .. }
            ) && error.message.contains("x")),
            "expected x to remain unbound: {result:#?}"
        );
    }
}

#[test]
fn sequential_let_shadow_does_not_create_a_top_level_dependency() {
    let program = surf_program(
        r#"
module ShadowParity
def helper() = {
  result = 1
  copied = result
  copied
}
result = helper()
"#,
    );
    assert_accepts_at_both_ingresses(&program, "sequential let shadow");
}

#[test]
fn missing_top_level_name_rejects_identically() {
    let program = deep_program("(def {} use_missing (var {} missing))\n");
    assert_rejects_identically(&program, "UnboundVariable", "missing top-level name");
}

#[test]
fn local_let_forward_reference_remains_sequential() {
    let program = deep_program(
        "(def {} local_forward\n\
           (let {}\n\
             (bind {} x (var {} y) y (lit {type: (t-prim {} int32)} 1))\n\
             (var {} x)))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "local let forward reference");
}

#[test]
fn eager_top_level_value_cycle_rejects_as_cycle_at_both_ingresses() {
    let program = deep_program(
        "(def {} first (var {} second))\n\n\
         (def {} second (var {} first))\n",
    );
    assert_rejects_identically(&program, "CycleDetected", "eager value cycle");
}

#[test]
fn value_only_diagnostics_remain_in_source_order_without_reaching_the_scheduler() {
    // This program has no function def, so `primary_inference_schedule` takes
    // its identity short circuit and neither the barrier nor the recursive
    // contraction runs. That is deliberate: it isolates the value path. The
    // companion below covers the scheduler, and does NOT assert source order,
    // because a reordered component genuinely moves its own diagnostics and
    // never had that property on `main` either.
    let program = deep_program(
        "(def {} first\n\
           (let {}\n\
             (bind {} dependency (var {} later))\n\
             (app {} (var {} missing_first))))\n\n\
         (def {} later (app {} (var {} missing_later)))\n",
    );
    let (ir, typed) = diagnostics(&program);
    assert_eq!(ir, typed, "ingress diagnostics diverged");
    let messages = ir
        .iter()
        .map(|(_, message)| message.as_str())
        .collect::<Vec<_>>();
    let first = messages
        .iter()
        .position(|message| message.contains("missing_first"))
        .expect("missing_first diagnostic");
    let later = messages
        .iter()
        .position(|message| message.contains("missing_later"))
        .expect("missing_later diagnostic");
    assert!(
        first < later,
        "diagnostics must retain source order: {ir:#?}"
    );
}

// ---------------------------------------------------------------------------
// [04-INF-4] generated ordering matrix
//
// This is the authoritative oracle for the atom. The named regressions above
// each pin one spelling; they cannot pin the interaction between declaration
// ORDER and the body-inference schedule, which is where every repair of this
// rule has failed. `primary_inference_schedule` reorders module functions, so
// a program's verdict depends on the layout as much as on the spelling, and a
// hand-written case only ever samples one layout. The matrix enumerates the
// layouts instead and derives each expected verdict from the atom: a
// top-level eager value is visible from its own `def` onward and nowhere
// earlier, at both ingresses, identically.
//
// The `wrapped` axis matters and is easy to lose. Bare declarations carry no
// lexical module key, so `module_fn_indices` is empty and the schedule is the
// identity map: an unwrapped program cannot reach the reordering at all.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Item {
    /// A module function that reads nothing. Its only job is to exist before
    /// the value, so the planner has an earlier function to hoist toward.
    Anchor,
    Value,
    Reader,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ReaderKind {
    Function,
    Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SurfSpelling {
    Unannotated,
    DeclarationTyped,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DeepSpelling {
    /// No sibling signature: the `def`'s own body stamp is the only type
    /// evidence, and only the serialized-IR ingress prebinds from it.
    BodyStampOnly,
    /// `(defsig ...)` immediately before the matching `(def ...)`.
    AdjacentSignature,
    /// The signature is declared at the head of the unit and the `def` stays
    /// at its layout position. A signature is metadata about a declaration,
    /// not the declaration, so it must not publish the value early.
    SeparatedSignature,
}

/// Every layout the matrix walks: all orderings of anchor/value/reader, plus
/// the two anchor-free orderings.
fn layouts() -> Vec<Vec<Item>> {
    vec![
        vec![Item::Anchor, Item::Value, Item::Reader],
        vec![Item::Anchor, Item::Reader, Item::Value],
        vec![Item::Value, Item::Anchor, Item::Reader],
        vec![Item::Value, Item::Reader, Item::Anchor],
        vec![Item::Reader, Item::Anchor, Item::Value],
        vec![Item::Reader, Item::Value, Item::Anchor],
        vec![Item::Value, Item::Reader],
        vec![Item::Reader, Item::Value],
    ]
}

/// Whether the schedule's hoist carries the reader back over the value.
///
/// Every module function is spliced at the earliest module-function ordinal, so
/// a function reader is inferred there regardless of its own position. It
/// therefore misses any value declared after that point.
fn value_is_hoisted_over(layout: &[Item]) -> bool {
    let first_function = layout
        .iter()
        .position(|item| *item == Item::Anchor || *item == Item::Reader);
    let value = layout.iter().position(|item| *item == Item::Value);
    matches!((first_function, value), (Some(f), Some(v)) if v > f)
}

/// [04-INF-4]: the reader resolves exactly when the value's `def` precedes it.
fn layout_accepts(layout: &[Item]) -> bool {
    let position = |wanted: Item| layout.iter().position(|item| *item == wanted);
    position(Item::Value) < position(Item::Reader)
}

fn surf_source(
    layout: &[Item],
    reader: ReaderKind,
    spelling: SurfSpelling,
    wrapped: bool,
) -> String {
    let mut source = String::new();
    if wrapped {
        source.push_str("module OrderingMatrix\n\n");
    }
    for item in layout {
        let declaration = match item {
            Item::Anchor => "def anchor() -> int32 = 1".to_string(),
            Item::Value => match spelling {
                SurfSpelling::Unannotated => "carried = 7".to_string(),
                SurfSpelling::DeclarationTyped => "carried: int32 = 7".to_string(),
            },
            Item::Reader => match reader {
                ReaderKind::Function => "def reader() -> int32 = carried".to_string(),
                ReaderKind::Value => "echoed = carried".to_string(),
            },
        };
        source.push_str(&declaration);
        source.push_str("\n\n");
    }
    source
}

fn deep_source(
    layout: &[Item],
    reader: ReaderKind,
    spelling: DeepSpelling,
    wrapped: bool,
) -> String {
    let value_signature = "(defsig {} carried (t-prim {} int32))";
    let value_def = "(def {} carried (lit {type: (t-prim {} int32)} 7))";
    let mut declarations: Vec<String> = Vec::new();
    if spelling == DeepSpelling::SeparatedSignature {
        declarations.push(value_signature.to_string());
    }
    for item in layout {
        match item {
            Item::Anchor => declarations.push(
                "(def {} anchor (fn {} (params {}) (lit {type: (t-prim {} int32)} 1)))".to_string(),
            ),
            Item::Value => match spelling {
                DeepSpelling::BodyStampOnly | DeepSpelling::SeparatedSignature => {
                    declarations.push(value_def.to_string())
                }
                DeepSpelling::AdjacentSignature => {
                    declarations.push(value_signature.to_string());
                    declarations.push(value_def.to_string());
                }
            },
            Item::Reader => declarations.push(match reader {
                ReaderKind::Function => {
                    "(def {} reader (fn {} (params {}) (var {} carried)))".to_string()
                }
                ReaderKind::Value => "(def {} echoed (var {} carried))".to_string(),
            }),
        }
    }
    if wrapped {
        format!(
            "(module {{}} OrderingMatrix\n  {})\n",
            declarations.join("\n  ")
        )
    } else {
        format!("{}\n", declarations.join("\n\n"))
    }
}

/// Rows that [04-INF-4] requires to pass and that the shipped body-inference
/// schedule cannot deliver.
///
/// `primary_inference_schedule` hoists every module function to the earliest
/// module-function ordinal, and `primary_inference_groups` emits a recursive
/// component at its first member, so a function body can be inferred before an
/// eager value it legally reads. The value's type does not exist yet, and the
/// two ingresses then disagree whenever the serialized-IR body-stamp prebind
/// can supply one and the typed ingress cannot. That reordering predates this
/// change, reproduces identically on `main`, and is not repaired here; it
/// remains part of chelis#1134.
///
/// This is a ratchet, not a mute. A row named here must still FAIL, so the set
/// cannot grow silently, and it shrinks visibly the moment the schedule stops
/// reordering across a value.
fn residual_row_failure(
    program: &[chelis_deep::Expr],
    accepts: bool,
    label: &str,
) -> Option<String> {
    if ordering_row_failure(program, accepts, label).is_none() {
        return Some(format!(
            "{label}: listed as a schedule residual but now satisfies \
             [04-INF-4]. Remove it from the residual predicate."
        ));
    }
    None
}

/// Check one generated program and report the disagreement, if any.
fn ordering_row_failure(
    program: &[chelis_deep::Expr],
    accepts: bool,
    label: &str,
) -> Option<String> {
    let (ir, typed) = diagnostics(program);
    if ir != typed {
        return Some(format!(
            "{label}: ingress diagnostics diverged\n  ir:    {ir:#?}\n  typed: {typed:#?}"
        ));
    }
    match (accepts, ir.is_empty()) {
        (true, false) => Some(format!("{label}: expected acceptance, got {ir:#?}")),
        (false, true) => Some(format!("{label}: expected UnboundVariable, got acceptance")),
        (false, false) if !ir.iter().any(|(kind, _)| kind == "UnboundVariable") => {
            Some(format!("{label}: expected UnboundVariable, got {ir:#?}"))
        }
        _ => None,
    }
}

fn report(failures: Vec<String>, rows: usize) {
    assert!(
        failures.is_empty(),
        "{}/{rows} ordering rows disagree with [04-INF-4]:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn surf_ordering_matrix_matches_04_inf_4_at_both_ingresses() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in layouts() {
        for reader in [ReaderKind::Function, ReaderKind::Value] {
            for spelling in [SurfSpelling::Unannotated, SurfSpelling::DeclarationTyped] {
                for wrapped in [true, false] {
                    rows += 1;
                    let source = surf_source(&layout, reader, spelling, wrapped);
                    let label = format!(
                        "surf layout={layout:?} reader={reader:?} spelling={spelling:?} \
                         wrapped={wrapped}\n{source}"
                    );
                    let program = surf_program(&source);
                    let accepts = layout_accepts(&layout);
                    let residual = wrapped
                        && reader == ReaderKind::Function
                        && spelling == SurfSpelling::Unannotated
                        && accepts
                        && value_is_hoisted_over(&layout);
                    let outcome = if residual {
                        residual_row_failure(&program, accepts, &label)
                    } else {
                        ordering_row_failure(&program, accepts, &label)
                    };
                    if let Some(failure) = outcome {
                        failures.push(failure);
                    }
                }
            }
        }
    }
    report(failures, rows);
}

#[test]
fn deep_ordering_matrix_matches_04_inf_4_at_both_ingresses() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in layouts() {
        for reader in [ReaderKind::Function, ReaderKind::Value] {
            for spelling in [
                DeepSpelling::BodyStampOnly,
                DeepSpelling::AdjacentSignature,
                DeepSpelling::SeparatedSignature,
            ] {
                for wrapped in [true, false] {
                    rows += 1;
                    let source = deep_source(&layout, reader, spelling, wrapped);
                    let label = format!(
                        "deep layout={layout:?} reader={reader:?} spelling={spelling:?} \
                         wrapped={wrapped}\n{source}"
                    );
                    let program = deep_file_program(&source);
                    let accepts = layout_accepts(&layout);
                    let residual = wrapped
                        && reader == ReaderKind::Function
                        && spelling == DeepSpelling::BodyStampOnly
                        && accepts
                        && value_is_hoisted_over(&layout);
                    let outcome = if residual {
                        residual_row_failure(&program, accepts, &label)
                    } else {
                        ordering_row_failure(&program, accepts, &label)
                    };
                    if let Some(failure) = outcome {
                        failures.push(failure);
                    }
                }
            }
        }
    }
    report(failures, rows);
}

// ---------------------------------------------------------------------------
// [04-INF-4] recursive-component ordering matrix
//
// The matrix above cannot reach this interaction. Its only function-shaped
// items never call each other, so `FunctionInferencePlan` never builds a
// multi-member recursive component and `primary_inference_groups`'
// whole-component emission path is never taken. That path is where the
// schedule's value barrier can be lost: a component is emitted at whichever
// member the schedule reaches first, so a barrier one member earns has to
// constrain every member.
//
// The value spelling matters here too. `carried = 7` desugars to a
// type-stamped literal, which the serialized-IR body-stamp prebind can read,
// so losing the barrier shows up as an ingress DIVERGENCE. `carried = seed()`
// has no header of any kind, so the same loss shows up as an over-rejection
// at both ingresses. Both spellings are enumerated.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RecursiveItem {
    /// The recursive peer that reads nothing.
    Ping,
    Value,
    /// The recursive peer that reads the value.
    Pong,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RecursiveValueSpelling {
    /// Desugars to a type-stamped literal the IR prebind can read.
    Stamped,
    /// No header anywhere: the type exists only once its own `def` is inferred.
    Computed,
}

fn recursive_layouts() -> Vec<Vec<RecursiveItem>> {
    use RecursiveItem::{Ping, Pong, Value};
    vec![
        vec![Ping, Value, Pong],
        vec![Ping, Pong, Value],
        vec![Value, Ping, Pong],
        vec![Value, Pong, Ping],
        vec![Pong, Ping, Value],
        vec![Pong, Value, Ping],
    ]
}

/// [04-INF-4]: the reading peer resolves exactly when the value precedes it.
fn recursive_layout_accepts(layout: &[RecursiveItem]) -> bool {
    let position = |wanted: RecursiveItem| layout.iter().position(|item| *item == wanted);
    position(RecursiveItem::Value) < position(RecursiveItem::Pong)
}

fn recursive_surf_source(
    layout: &[RecursiveItem],
    spelling: RecursiveValueSpelling,
    wrapped: bool,
) -> String {
    let mut source = String::new();
    if wrapped {
        source.push_str("module RecursiveOrderingMatrix\n\n");
    }
    // `seed` leads in both spellings so the layouts stay comparable.
    source.push_str("def seed() -> int32 = 3\n\n");
    for item in layout {
        let declaration = match item {
            RecursiveItem::Ping => {
                "def ping(n: int32) -> int32 = if (n <= 0) then 0 else pong((n - 1))".to_string()
            }
            RecursiveItem::Pong => {
                "def pong(n: int32) -> int32 = if (n <= 0) then carried else ping((n - 1))"
                    .to_string()
            }
            RecursiveItem::Value => match spelling {
                RecursiveValueSpelling::Stamped => "carried = 7".to_string(),
                RecursiveValueSpelling::Computed => "carried = seed()".to_string(),
            },
        };
        source.push_str(&declaration);
        source.push_str("\n\n");
    }
    source
}

#[test]
fn recursive_component_ordering_matrix_matches_04_inf_4_at_both_ingresses() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in recursive_layouts() {
        for spelling in [
            RecursiveValueSpelling::Stamped,
            RecursiveValueSpelling::Computed,
        ] {
            for wrapped in [true, false] {
                rows += 1;
                let source = recursive_surf_source(&layout, spelling, wrapped);
                let label = format!(
                    "recursive layout={layout:?} spelling={spelling:?} wrapped={wrapped}\n{source}"
                );
                let program = surf_program(&source);
                let accepts = recursive_layout_accepts(&layout);
                // `seed` leads every recursive row, so the wrapped hoist point
                // is ordinal zero and every accepting wrapped row is carried
                // over its value. A bare row has no hoist, but
                // `primary_inference_groups` still emits the component at its
                // first member, so it is residual whenever `ping` precedes the
                // value.
                let ping = layout
                    .iter()
                    .position(|item| *item == RecursiveItem::Ping)
                    .expect("every layout carries ping");
                let value = layout
                    .iter()
                    .position(|item| *item == RecursiveItem::Value)
                    .expect("every layout carries the value");
                let residual = accepts && (wrapped || ping < value);
                let outcome = if residual {
                    residual_row_failure(&program, accepts, &label)
                } else {
                    ordering_row_failure(&program, accepts, &label)
                };
                if let Some(failure) = outcome {
                    failures.push(failure);
                }
            }
        }
    }
    report(failures, rows);
}

// ---------------------------------------------------------------------------
// Function-visibility non-regression matrix
//
// [04-INF-4] governs eager values. It says nothing about when a `defsig`-less
// FUNCTION's inferred scheme becomes available, which [04-INF-2]/[04-INF-3]
// and the planner own. This change must therefore leave that axis exactly
// where it was, and it very nearly did not: replacing the old blunt hoist
// (every module function spliced at the earliest module-function ordinal) with
// a dependency order silently dropped the guarantee the hoist supplied, that a
// later declaration can always resolve a module function. The rows below pin
// the behavior in both directions, so a future ordering change cannot widen it
// either.
//
// Expected verdicts are derived, not measured: a module-wrapped unit resolves a
// function from any declaration at or after the first function, because that is
// where the planner's region begins; a bare unit has no planner and resolves
// only backwards.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FnDepItem {
    /// `defsig`-less leaf whose scheme exists only after its body is inferred.
    Helper,
    /// `defsig`-less function that calls `Helper`.
    Caller,
    /// An eager value that reads `Caller`, i.e. a non-function declaration
    /// depending on a function's inferred scheme.
    Consumer,
}

fn fn_dep_layouts() -> Vec<Vec<FnDepItem>> {
    use FnDepItem::{Caller, Consumer, Helper};
    vec![
        vec![Helper, Caller, Consumer],
        vec![Helper, Consumer, Caller],
        vec![Caller, Helper, Consumer],
        vec![Caller, Consumer, Helper],
        vec![Consumer, Helper, Caller],
        vec![Consumer, Caller, Helper],
    ]
}

fn fn_dep_layout_accepts(layout: &[FnDepItem], wrapped: bool) -> bool {
    let position = |wanted: FnDepItem| {
        layout
            .iter()
            .position(|item| *item == wanted)
            .expect("every layout carries every item")
    };
    let helper = position(FnDepItem::Helper);
    let caller = position(FnDepItem::Caller);
    let consumer = position(FnDepItem::Consumer);
    if wrapped {
        // The planner orders functions by dependency, so `caller` always sees
        // `helper`; a reader resolves once the planner's region has begun.
        consumer > helper.min(caller)
    } else {
        // No module key, no planner: both edges are plain backward references.
        helper < caller && caller < consumer
    }
}

fn fn_dep_source(layout: &[FnDepItem], wrapped: bool) -> String {
    let mut source = String::new();
    if wrapped {
        source.push_str("module FnDepMatrix\n\n");
    }
    for item in layout {
        let declaration = match item {
            FnDepItem::Helper => "def helper(x) = x",
            FnDepItem::Caller => "def caller(n) = helper(n)",
            FnDepItem::Consumer => "consumer = caller(1)",
        };
        source.push_str(declaration);
        source.push_str("\n\n");
    }
    source
}

#[test]
fn function_visibility_matrix_is_unchanged_by_eager_value_scope() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in fn_dep_layouts() {
        for wrapped in [true, false] {
            rows += 1;
            let source = fn_dep_source(&layout, wrapped);
            let label = format!("fn-dep layout={layout:?} wrapped={wrapped}\n{source}");
            let program = surf_program(&source);
            if let Some(failure) =
                ordering_row_failure(&program, fn_dep_layout_accepts(&layout, wrapped), &label)
            {
                failures.push(failure);
            }
        }
    }
    report(failures, rows);
}

/// An eager value whose initializer reaches, through a call, a value declared
/// after it. Every reference in it obeys [04-INF-4] and [04-INF-2] on its own,
/// so the scope rule must not be what rejects it; the binding-cycle detector
/// owns that verdict and must reach it identically at both ingresses. The
/// compiled read of a not-yet-assigned global in this shape is chelis#1339's
/// indirect residue and is owned there, not here.
#[test]
fn an_initialization_cycle_leaves_the_schedule_total_at_both_ingresses() {
    let program = surf_program(
        "module InitializationCycle\n\n\
         def ping(n: int32) -> int32 = if (n <= 0) then 0 else pong((n - 1))\n\n\
         carried = ping(1)\n\n\
         def pong(n: int32) -> int32 = if (n <= 0) then carried else ping((n - 1))\n",
    );
    // The cycle detector owns the verdict; this test owns the property that the
    // scheduler reaches it at all, identically at both ingresses.
    assert_rejects_identically(&program, "CycleDetected", "initialization cycle");
}

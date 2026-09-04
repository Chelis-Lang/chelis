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

/// An external input whose declared type does not resolve must report the
/// same diagnostics at both ingresses. The serialized-IR ingress resolves the
/// body stamp in its prebind pass and must not resolve it a second time when
/// the declaration's own external-input binding is installed.
#[test]
fn a_malformed_external_input_type_reports_identically_at_both_ingresses() {
    for (label, source) in [
        ("ascribed", "x = (x : Unknown)\n"),
        ("declaration-typed", "x: Unknown = x\n"),
    ] {
        let program = surf_program(source);
        let (ir, typed) = diagnostics(&program);
        assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
        assert!(!ir.is_empty(), "{label}: an unknown type must be rejected");
        assert_eq!(
            ir.len(),
            ir.iter().collect::<std::collections::BTreeSet<_>>().len(),
            "{label}: a diagnostic must be reported once: {ir:#?}"
        );
    }
}

/// chelis#1485, still recorded rather than repaired, restated for the narrowed
/// mirror edge.
///
/// A value that names a function which reads the value back. Narrowing the
/// mirror edge to `defsig`-less module functions (chelis#1486) removes the
/// back edge for the two SIGNED spellings, so their schedules are acyclic and
/// they now check clean; the `defsig`-less spelling keeps its mirror edge,
/// still stalls, and still reports the backward read as unbound. The stamped
/// spelling still splits between the two ingresses.
///
/// These acceptances are an intermediate state, not the disposition. Under
/// [04-INF-7] every one of these three programs is an eager value cycle, and
/// the cycle detector does not yet see a reference nested in a lambda body or
/// a call edge onto a value under evaluation (chelis#1487). When that lands
/// each row becomes `CycleDetected`; the rows are spelled out one per line so
/// that transition is visible rather than hidden in a loop.
///
/// Disposition lock. Every row's expectation moved with this change or moves
/// with the next one, and the test reddens whenever a row's verdict does.
#[test]
fn a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall() {
    // `None` means the program is accepted at both ingresses today.
    for (label, expected, source) in [
        (
            "signed reader",
            None,
            "module MirrorEscape\n\n\
             def anchor() -> int32 = 1\n\n\
             carried = wrap(f)\n\n\
             def wrap(g) = g\n\n\
             def f(n: int32) -> int32 = if (n <= 0) then 0 else carried((n - 1))\n",
        ),
        (
            "lambda naming a signed reader",
            None,
            "module PickEscape\n\n\
             def anchor() -> int32 = 1\n\n\
             carried = pick(fn (x: int32) -> f(x))\n\n\
             def pick(g) = 5\n\n\
             def f(n: int32) -> int32 = add(n, carried)\n",
        ),
        (
            "defsig-less reader",
            Some("UnboundVariable"),
            "module WrapEscape\n\n\
             def anchor() -> int32 = 1\n\n\
             carried = wrap(g)\n\n\
             def wrap(h) = h\n\n\
             def g(n) = if (n <= 0) then 0 else carried((n - 1))\n",
        ),
    ] {
        let program = surf_program(source);
        let (ir, typed) = diagnostics(&program);
        assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
        match expected {
            Some(kind) => assert!(
                ir.iter().any(|(reported, _)| reported == kind),
                "{label}: expected {kind}, got {ir:#?}"
            ),
            None => assert!(
                ir.is_empty(),
                "{label}: this spelling is accepted at this stage, got {ir:#?}"
            ),
        }
    }
    let stamped = deep_file_program(
        "(module {} MirrorEscapeStamped\n  \
           (defsig {} anchor (t-fn {} (t-prim {} int32)))\n  \
           (def {} anchor (fn {} (params {}) (lit {type: (t-prim {} int32)} 1)))\n  \
           (def {} carried (app {type: (t-fn {} (t-prim {} int32) (t-prim {} int32))} (var {} wrap) (var {} f)))\n  \
           (def {} wrap (fn {} (params {} g) (var {} g)))\n  \
           (defsig {} f (t-fn {} (t-prim {} int32) (t-prim {} int32)))\n  \
           (def {} f (fn {} (params {} (n {type: (t-prim {} int32)}))\n    \
             (if {} (app {} (var {} lte) (var {} n) (lit {type: (t-prim {} int32)} 0))\n      \
               (lit {type: (t-prim {} int32)} 0)\n      \
               (app {} (var {} carried) (app {} (var {} sub) (var {} n) (lit {type: (t-prim {} int32)} 1)))))))\n",
    );
    // The stamped spelling declares `f` with an explicit `defsig`, so it loses
    // its mirror edge with the Surf signed spelling and the two ingresses now
    // agree. chelis#1485's ingress SPLIT is therefore closed here; the
    // remaining half of its disposition is the `CycleDetected` verdict
    // [04-INF-7] owes this program, which chelis#1487 delivers.
    let (ir, typed) = diagnostics(&stamped);
    assert_eq!(ir, typed, "stamped reader: ingress diagnostics diverged");
    assert!(
        ir.is_empty(),
        "stamped reader: this spelling is accepted at this stage, got {ir:#?}"
    );
}

/// Assemble one unit from ordered declarations, with or without the module
/// wrapper that gives the planner a region.
fn layout_source(module: Option<&str>, declarations: &[&str]) -> String {
    let mut source = String::new();
    if let Some(name) = module {
        source.push_str(&format!("module {name}\n\n"));
    }
    for declaration in declarations {
        source.push_str(declaration);
        source.push_str("\n\n");
    }
    source
}

/// A partial or generic header must not be instantiated before its body
/// narrows it (chelis#1486). `def f(n: int32) = ...` synthesizes a `defsig`
/// with a wildcard result, which [04-INF-5] makes an inference hole whose type
/// is whatever the body determines; `def f(x: a) -> a` declares an authored
/// binder, which [04-INF-6] makes rigid. The two atoms close the defect at
/// opposite ends: the hole edge defers the READER past the body, and the rigid
/// check rejects the DECLARATION.
///
/// The three layouts matter because the mirror edge that used to defer the
/// reader was bounded to the planner's region. Below the hoist floor and in a
/// bare unit it never applied, so the same spelling was accepted or rejected
/// by position alone. The hole edge carries no such bound, so every layout now
/// rejects, identically at both ingresses.
///
/// Regression test. Before this change the two below-floor layouts and the two
/// bare layouts of the partial program were ACCEPTED, and the generic program
/// rejected at the reader rather than at the declaration.
#[test]
fn a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it() {
    let partial_reader = "r: f32 = f(2)";
    let partial_fn = "def f(n: int32) = add(v, n)";
    let generic_reader = "r: f32 = f(1.5)";
    // [04-INF-6]: the body pins the authored `a` to `int32`, so `f` itself is
    // the rejection and the reader never gets to matter.
    let generic_fn = "def f(x: a) -> a = add(x, v)";
    let anchor = "def anchor() -> int32 = 1";
    let carried = "v: int32 = 1";

    for (program, reader, function) in [
        ("PartialHeader", partial_reader, partial_fn),
        ("GenericHeader", generic_reader, generic_fn),
    ] {
        for (layout, declarations) in [
            // Today's layout: the reader sits at or after the hoist floor.
            ("after the floor", vec![anchor, reader, carried, function]),
            // Below the floor: no mirror edge ever reached this reader.
            ("below the floor", vec![reader, carried, anchor, function]),
        ] {
            let source = layout_source(Some(program), &declarations);
            assert_rejects_identically(
                &surf_program(&source),
                "TypeMismatch",
                &format!("{program} {layout}\n{source}"),
            );
        }
        // A bare unit has no planner region at all.
        let source = layout_source(None, &[reader, carried, function]);
        assert_rejects_identically(
            &surf_program(&source),
            "TypeMismatch",
            &format!("{program} bare unit\n{source}"),
        );
    }
}

/// [04-INF-6]: an authored type binder is rigid in the body, so a body that
/// pins it to a concrete type or collapses it onto a sibling binder is a type
/// error AT THE DECLARATION. No reader is present in any of these programs:
/// the rejection is a property of the declaration alone, which is what
/// separates this atom from the scheduling half of chelis#1486.
///
/// Both spellings of an authored binder are covered: the explicit binder list
/// (`def f[a](..)`) and §5.8.1's implicit quantification (`def f(x: a) -> a`),
/// which the resolver records identically.
///
/// Regression test. Every row was ACCEPTED before this change, and the first
/// two compiled a body typed at `int32` behind a signature promising `forall
/// a. (a) -> a`.
#[test]
fn a_body_that_narrows_an_authored_type_binder_rejects_at_the_declaration() {
    for (label, source) in [
        (
            "explicit binder pinned to int32",
            "module ExplicitRigid\n\n\
             def f[a](x: a) -> a = add(x, 1)\n",
        ),
        (
            "implicit binder pinned to int32",
            "module ImplicitRigid\n\n\
             def f(x: a) -> a = add(x, 1)\n",
        ),
        (
            "two binders collapsed onto each other",
            "module CollapsedBinders\n\n\
             def g[a, b](x: a, y: b) -> a = y\n",
        ),
        (
            "bounded binder pinned by an unsuffixed literal",
            "module BoundedRigid\n\n\
             def scale[p: Float](x: p) -> p = mul(x, 0.0)\n",
        ),
    ] {
        assert_rejects_identically(&surf_program(source), "TypeMismatch", label);
    }
}

/// The failure twin of the rigid-binder rejections: a body that leaves every
/// authored binder unconstrained, or constrains it only through the §P10
/// `cast(<literal>, <binder>)` override, stays accepted.
///
/// The bounded row is the shape the ten stdlib repairs took. Without it the
/// rejection above could be satisfied by rejecting every bounded binder, which
/// would be a different and much worse rule.
///
/// Disposition lock. Every row was green before this change too; the job is to
/// bound the rejection so it cannot grow into the polymorphic bodies the
/// language is for.
#[test]
fn a_body_that_keeps_its_authored_type_binders_polymorphic_stays_accepted() {
    for (label, source) in [
        (
            "identity with an explicit result",
            "module PolyIdentity\n\n\
             def id(x: a) -> a = x\n",
        ),
        (
            "identity whose result slot is a hole",
            "module PolyHoleResult\n\n\
             def k(x: a) = x\n",
        ),
        (
            "bounded binder written with the cast override",
            "module PolyBounded\n\n\
             def scale[p: Float](x: p) -> p = mul(x, cast(0.0, p))\n",
        ),
        (
            "two binders kept distinct",
            "module DistinctBinders\n\n\
             def pick[a, b](x: a, y: b) -> a = x\n",
        ),
    ] {
        assert_accepts_at_both_ingresses(&surf_program(source), label);
    }
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
    // This program has no function def, so the schedule has no module
    // function to move and no reference edge to honor: it is the identity map.
    // That is deliberate: it isolates the value path. A program that reaches
    // the scheduler is NOT held to source-ordered diagnostics, because a moved
    // function or component genuinely moves its own diagnostics, and no
    // version of the checker ever ordered those by source position.
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
// This is the verdict oracle for the atom. The named regressions above each
// pin one spelling; they cannot pin the interaction between declaration ORDER
// and the body-inference schedule. `primary_inference_schedule` reorders
// module functions by dependency, so a program's verdict depends on the
// layout as much as on the spelling, and a hand-written case only ever
// samples one layout. The matrix enumerates the layouts instead and derives
// each expected verdict from the atom: a top-level eager value is visible
// from its own `def` onward and nowhere earlier, at both ingresses,
// identically.
//
// A verdict matrix cannot see the emitted ORDER itself, which is where every
// schedule defect has lived. The order invariants are asserted directly on
// the schedule by `crates/chelis-types/src/infer/tests/schedule_invariants.rs`;
// this matrix pins that the resulting verdicts are the atom's.
//
// The `wrapped` axis matters and is easy to lose. Bare declarations carry no
// lexical module key, so no function is hoisted and only a recursive
// component straddling a value can move: an unwrapped program reaches the
// schedule only through that path.
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
                    if let Some(failure) = ordering_row_failure(&program, accepts, &label) {
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
                    if let Some(failure) = ordering_row_failure(&program, accepts, &label) {
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
// whole-component emission path is never taken. A component is inferred as
// one unit at the first member the schedule reaches, so the schedule
// contracts it to one vertex and an edge that one member earns constrains
// every member. This matrix pins the verdicts that contraction produces.
//
// The value spelling matters here too. `carried = 7` desugars to a
// type-stamped literal, which the serialized-IR body-stamp prebind can read,
// so a lost edge shows up as an ingress DIVERGENCE. `carried = seed()` has no
// header of any kind, so the same loss shows up as an over-rejection at both
// ingresses. Both spellings are enumerated.
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
                if let Some(failure) = ordering_row_failure(&program, accepts, &label) {
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
// where it was, and an earlier attempt did not: replacing the hoist order
// (every module function spliced at the earliest module-function ordinal) with
// a bare dependency order silently dropped the guarantee the hoist supplied,
// that a later declaration can always resolve a module function. The schedule
// now keeps the hoist order as its priority and adds only reference edges, so
// these rows hold by construction; they stay as the pin in both directions, so
// a future ordering change cannot widen the axis either.
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

//! chelis#874 / chelis#887 / chelis#1525 -- PP8: source-coverage totality.
//!
//! Normative authority: `spec/04-type-system.md` §10 [04-TOT-4], which extends
//! [04-TOT-3] from the form down to each of the form's slots. Design and probe
//! table: `spec/design/checker_totality.md` §PP8.
//!
//! The class. §C4.1's coverage invariant quantifies over the nodes inference
//! VISITS, so a role-slot child the checker never reads satisfies [04-TOT-2]
//! vacuously: the error vector is empty, no `Type::Error` exists anywhere, and
//! the program scores a perfect 1.0 while a node the author submitted was
//! silently discarded. Three spellings produce it, all in the same four
//! inference functions: `unwrap_or(default)`, `if let Some(..)` with no
//! `else`, and `else { continue }`. Each converts "I could not read this
//! child" into "there was no child", and [04-TOT-4] forbids that conversion:
//! an omitted optional child and a present unreadable one are distinct
//! inputs, and only the omission may take the form's declared default.
//!
//! Ingress. Every row here runs through `check_ir_program`, which is what
//! `chelis check` runs beneath (`chelis_types::fitness::analyze_ir_program`
//! calls it, and `chelis-compiler-api`'s `check` calls that). Nothing in this
//! file claims that the two checker ingresses agree on these programs; that is
//! [04-TOT-5]'s axis and chelis#1125 (PP7) owns it. The repairs sit inside
//! `infer_vmap`, `infer_grad`, `infer_record`, `infer_record_update`, and
//! `pattern_bindings`, which `infer_expr` dispatches to from both carrier
//! arms, but reaching them from the second carrier is not asserted here.
//!
//! Evidentiary status is labelled per assertion. A "regression test" was red
//! on the pre-fix tree; a "disposition lock" is green in both states and says
//! what job it does. The negative rows below need no revert step to be proven
//! red: the probe transcript recorded on `3b701e54b` shows each of them
//! accepting at score 1.0, or rejecting with a diagnostic that blames the
//! wrong node, before the seam landed.
//!
//! Scope. What is claimed is exactly the slots named in the tests below: the
//! `vmap` axis, the `pat-ctor` and `pat-record` constructor heads, the `grad`
//! operand, the `kv` key in record construction, record update, and record
//! patterns, the `pat-var` and `pat-as` binder names, and the `pat-lit` value.
//! The `Selector` role is enumerable and enumerated (`child_stamp_role` is
//! total over `DeepTag` and yields `Selector` at eight tag-and-index slots);
//! the `Binder`, `Syntax`, `EffectHandler`, and `Type` roles are NOT
//! enumerated here, and the `pat-var` / `pat-as` / `pat-lit` rows are named
//! instances rather than a statement about their roles.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_types::check_ir_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

// ── Harness ────────────────────────────────────────────────────────────────

/// Declarations every fixture shares: a scalar function, a rank-1 tensor
/// function to `vmap`, a two-variant record ADT, a single-variant record ADT
/// for `record-update`, and a nullary-plus-record ADT for constructor
/// patterns.
const PRELUDE: &str = r#"
  (deftype {} Shape ()
    (variant {} Circle (field {} r (t-prim {} f32)))
    (variant {} Square (field {} a (t-prim {} f32))))
  (deftype {} Box () (variant {} Box (field {} r (t-prim {} f32))))
  (deftype {} Opt () (variant {} Non) (variant {} Som (field {} value (t-prim {} f32))))
  (defsig {} double (t-fn {} (t-prim {} f32) (t-prim {} f32)))
  (def {} double
    (fn {} (params {} (x {type: (t-prim {} f32)}))
      (app {} (var {} mul) (var {} x) (lit {type: (t-prim {} f32)} 2.0))))
  (defsig {} process
    (t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32))
             (t-tensor {} (d-name {} features) (t-prim {} f32))))
  (def {} process
    (fn {} (params {} (x {type: (t-tensor {} (d-name {} features) (t-prim {} f32))}))
      (app {} (var {} relu) (var {} x))))
"#;

fn program(body: &str) -> String {
    format!("(module {{}} m.main{PRELUDE}  {body})\n")
}

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).unwrap_or_else(|e| panic!("fixture must stamp:\n{source}\n{e}"))
}

/// Diagnostics from the ingress `chelis check` runs.
fn diagnostics(body: &str) -> Vec<CheckError> {
    let source = program(body);
    match check_ir_program(&stamped(&source)) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

fn rendered(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect::<Vec<_>>()
        .join("\n  ")
}

/// The unreadable-child rejection: a `MalformedForm` naming the form and the
/// shape the form expected at that slot ([04-TOT-4]).
#[track_caller]
fn assert_slot_rejection(body: &str, form: &str, shape: &str, label: &str) {
    let errors = diagnostics(body);
    let malformed: Vec<&CheckError> = errors
        .iter()
        .filter(|e| matches!(e.kind, CheckErrorKind::MalformedForm))
        .collect();
    assert!(
        !malformed.is_empty(),
        "{label}: an unreadable child must push a MalformedForm ([04-TOT-4]); \
         got:\n  {}",
        rendered(&errors)
    );
    assert!(
        malformed
            .iter()
            .any(|e| e.message.contains(form) && e.message.contains(shape)),
        "{label}: the diagnostic must name the form `{form}` and the expected \
         shape \"{shape}\" so the author can see what to write; got:\n  {}",
        rendered(&errors)
    );
}

/// The seam's rejection: a `MalformedForm` whose message contains `expected`.
/// Whitespace is normalized so a wrapped expectation in the test source
/// matches the single-line diagnostic.
#[track_caller]
fn assert_seam_message(errors: &[CheckError], expected: &str, label: &str) {
    let want: String = expected.split_whitespace().collect::<Vec<_>>().join(" ");
    let hit = errors.iter().any(|e| {
        matches!(e.kind, CheckErrorKind::MalformedForm)
            && e.message
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .contains(&want)
    });
    assert!(
        hit,
        "{label}: expected a MalformedForm containing\n  {want}\ngot:\n  {}",
        rendered(errors)
    );
}

/// No diagnostic may blame the §C4.1 owner-stamp tripwire. That message is an
/// `internal:` invariant violation naming a node the author did not write
/// wrongly, and reporting it instead of the real cause is the R5 defect.
#[track_caller]
fn assert_not_misattributed(body: &str, label: &str) {
    let errors = diagnostics(body);
    assert!(
        !errors.iter().any(|e| e
            .message
            .contains("annotation owner-stamp invariant violated")),
        "{label}: the unreadable child must be diagnosed at its own slot, not \
         via the collateral owner-stamp tripwire; got:\n  {}",
        rendered(&errors)
    );
}

#[track_caller]
fn assert_accepted(body: &str, label: &str) {
    let errors = diagnostics(body);
    assert!(
        errors.is_empty(),
        "{label}: this program is well formed and must keep checking cleanly; \
         got:\n  {}",
        rendered(&errors)
    );
}

// ── Fixture builders ───────────────────────────────────────────────────────

fn vmap_axis(axis: &str) -> String {
    let child = if axis.is_empty() {
        String::new()
    } else {
        format!(" {axis}")
    };
    format!(
        "(defsig {{}} batched
    (t-fn {{}} (t-tensor {{}} (d-name {{}} batch) (d-name {{}} features) (t-prim {{}} f32))
             (t-tensor {{}} (d-name {{}} batch) (d-name {{}} features) (t-prim {{}} f32))))
  (def {{}} batched
    (fn {{}} (params {{}} (xs {{type: (t-tensor {{}} (d-name {{}} batch) \
(d-name {{}} features) (t-prim {{}} f32))}}))
      (app {{}} (vmap {{}} (var {{}} process){child}) (var {{}} xs))))"
    )
}

fn pat_ctor_head(head: &str) -> String {
    format!(
        "(defsig {{}} gg (t-fn {{}} (t-adt {{}} Opt) (t-prim {{}} f32)))
  (def {{}} gg (fn {{}} (params {{}} (s {{type: (t-adt {{}} Opt)}}))
    (match {{}} (var {{}} s)
      (arm {{}} (pat-ctor {{}} {head}) () (lit {{type: (t-prim {{}} f32)}} 1.0))
      (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    )
}

fn pat_record_head(head: &str) -> String {
    format!(
        "(defsig {{}} hh (t-fn {{}} (t-adt {{}} Shape) (t-prim {{}} f32)))
  (def {{}} hh (fn {{}} (params {{}} (s {{type: (t-adt {{}} Shape)}}))
    (match {{}} (var {{}} s)
      (arm {{}} (pat-record {{}} {head}) () (lit {{type: (t-prim {{}} f32)}} 1.0))
      (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    )
}

fn pat_record_kv_key(key: &str) -> String {
    format!(
        "(defsig {{}} pr (t-fn {{}} (t-adt {{}} Shape) (t-prim {{}} f32)))
  (def {{}} pr (fn {{}} (params {{}} (s {{type: (t-adt {{}} Shape)}}))
    (match {{}} (var {{}} s)
      (arm {{}} (pat-record {{}} Circle (kv {{}} {key} (pat-wild {{}}))) () \
(lit {{type: (t-prim {{}} f32)}} 1.0))
      (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    )
}

fn record_kv_key(key: &str) -> String {
    format!(
        "(defsig {{}} mk (t-fn {{}} (t-adt {{}} Shape)))
  (def {{}} mk (fn {{}} (params {{}})
    (record {{}} Circle (kv {{}} {key} (lit {{type: (t-prim {{}} f32)}} 1.0)))))"
    )
}

fn record_update_kv_key(key: &str) -> String {
    format!(
        "(defsig {{}} ru (t-fn {{}} (t-adt {{}} Box) (t-adt {{}} Box)))
  (def {{}} ru (fn {{}} (params {{}} (s {{type: (t-adt {{}} Box)}}))
    (record-update {{}} (var {{}} s) \
(kv {{}} {key} (lit {{type: (t-prim {{}} f32)}} 3.0)))))"
    )
}

fn pat_lit_value(value: &str) -> String {
    format!(
        "(defsig {{}} pl (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} f32)))
  (def {{}} pl (fn {{}} (params {{}} (n {{type: (t-prim {{}} i32)}}))
    (match {{}} (var {{}} n)
      (arm {{}} (pat-lit {{}} {value}) () (lit {{type: (t-prim {{}} f32)}} 1.0))
      (arm {{}} (pat-wild {{}}) () (lit {{type: (t-prim {{}} f32)}} 2.0)))))"
    )
}

fn pat_var_name(name: &str) -> String {
    format!(
        "(defsig {{}} pv (t-fn {{}} (t-prim {{}} f32) (t-prim {{}} f32)))
  (def {{}} pv (fn {{}} (params {{}} (q {{type: (t-prim {{}} f32)}}))
    (match {{}} (var {{}} q)
      (arm {{}} (pat-var {{}} {name}) () (lit {{type: (t-prim {{}} f32)}} 1.0)))))"
    )
}

fn pat_as_name(name: &str) -> String {
    format!(
        "(defsig {{}} pa (t-fn {{}} (t-prim {{}} f32) (t-prim {{}} f32)))
  (def {{}} pa (fn {{}} (params {{}} (q {{type: (t-prim {{}} f32)}}))
    (match {{}} (var {{}} q)
      (arm {{}} (pat-as {{}} {name}) () \
(lit {{type: (t-prim {{}} f32)}} 1.0)))))"
    )
}

/// The four unreadable children the §PP8 probe table drives through every
/// selector slot: a name reference, a literal of the wrong family, a type node
/// in a value slot, and an application.
const UNREADABLE: [(&str, &str); 4] = [
    ("var", "(var {} nonexistent_name_zzz)"),
    ("float literal", "(lit {type: (t-prim {} f32)} 1.5)"),
    ("type node", "(t-prim {} f32)"),
    (
        "application",
        "(app {} (var {} missing_fn_qqq) (var {} missing_arg_www))",
    ),
];

// ── R1: the `vmap` axis ────────────────────────────────────────────────────

/// REGRESSION. `expr_transform.rs`'s
/// `kids.get(1).and_then(extract_int_for_dim).unwrap_or(0)` let one
/// `unwrap_or` serve two different inputs. On `3b701e54b` all four of these
/// scored 1.0 with an empty error vector.
#[test]
fn vmap_axis_that_is_present_and_unreadable_is_rejected() {
    for (label, child) in UNREADABLE {
        assert_slot_rejection(
            &vmap_axis(child),
            "vmap",
            "integer axis",
            &format!("vmap axis given a {label}"),
        );
    }
}

/// DISPOSITION LOCK, green in both states. `vmap(f)` with no axis child at all
/// is spec/02 §0.1's spelling of the zero axis, so the omission keeps taking
/// the form's declared default. This is the control that stops the cheapest
/// wrong fix -- deleting the default -- from passing the negatives above.
#[test]
fn vmap_with_no_axis_child_keeps_the_declared_default() {
    assert_accepted(&vmap_axis(""), "vmap with the axis child omitted");
}

/// DISPOSITION LOCK. A readable axis still reads, on both the bare-literal and
/// the cast-wrapped spellings chelis#216 added.
#[test]
fn vmap_with_a_readable_axis_still_checks() {
    assert_accepted(
        &vmap_axis("(lit {type: (t-prim {} i32)} 0)"),
        "vmap with an explicit axis 0",
    );
    assert_accepted(
        &vmap_axis("(cast {} (lit {type: (t-prim {} i64)} 0) (t-prim {} i32))"),
        "vmap with a cast-wrapped axis literal (chelis#216)",
    );
}

/// DISPOSITION LOCK. The two axis rejections that already worked keep their
/// own diagnostics: the seam must not swallow a readable-but-wrong axis into
/// the generic malformed message.
#[test]
fn vmap_axis_range_diagnostics_are_unchanged() {
    let errors = diagnostics(&vmap_axis("(lit {type: (t-prim {} i32)} -7)"));
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("vmap axis must be non-negative, got -7")),
        "a readable negative axis keeps its own diagnostic; got:\n  {}",
        rendered(&errors)
    );
    let errors = diagnostics(&vmap_axis("(lit {type: (t-prim {} i32)} 2)"));
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("out of bounds for rank 1 tensor")),
        "a readable out-of-range axis keeps its own diagnostic; got:\n  {}",
        rendered(&errors)
    );
}

// ── R2: the `pat-ctor` constructor head ────────────────────────────────────

/// REGRESSION. `if let Some(ctor_name) = kids.first().and_then(symbol_name)`
/// with no `else`: a non-symbol head fell off the binding and the whole
/// pattern was skipped, so the arm neither bound nor rejected. Both rows
/// scored 1.0 on `3b701e54b`.
#[test]
fn pat_ctor_head_that_is_present_and_unreadable_is_rejected() {
    for (label, child) in UNREADABLE {
        assert_slot_rejection(
            &pat_ctor_head(child),
            "pat-ctor",
            "constructor name",
            &format!("pat-ctor head given a {label}"),
        );
    }
}

/// REGRESSION, and the row that keeps the repair from creating a second
/// coverage hole one level down. Rejecting the head must not abandon the
/// sub-patterns: they are nodes the author submitted, and leaving them
/// unvisited makes §C4.1's owner-stamp tripwire report an unstamped `pat-var`
/// instead of the head that was actually wrong. Exactly one diagnostic, and it
/// names the head's slot.
#[test]
fn rejecting_a_pattern_head_still_visits_its_sub_patterns() {
    let body = pat_ctor_head("(var {} nonexistent_name_zzz) (pat-var {} y) (pat-wild {})");
    assert_slot_rejection(
        &body,
        "pat-ctor",
        "constructor name",
        "pat-ctor with an unreadable head and two sub-patterns",
    );
    assert_not_misattributed(
        &body,
        "pat-ctor with an unreadable head and two sub-patterns",
    );

    let body = pat_record_head("(var {} nonexistent_name_zzz) (kv {} r (pat-var {} y))");
    assert_slot_rejection(
        &body,
        "pat-record",
        "constructor name",
        "pat-record with an unreadable head and a field pattern",
    );
    assert_not_misattributed(
        &body,
        "pat-record with an unreadable head and a field pattern",
    );

    let body = pat_as_name("(var {} nonexistent_name_zzz) (pat-var {} y)");
    assert_slot_rejection(
        &body,
        "pat-as",
        "binding name",
        "pat-as with an unreadable name and an inner pattern",
    );
    assert_not_misattributed(&body, "pat-as with an unreadable name and an inner pattern");
}

/// DISPOSITION LOCK. The seam rejects unreadability, not non-tagged-ness: a
/// bare symbol head is exactly the readable case and must still resolve.
#[test]
fn pat_ctor_with_a_real_constructor_head_still_checks() {
    assert_accepted(
        &pat_ctor_head("Non"),
        "pat-ctor with an in-scope constructor",
    );
}

/// DISPOSITION LOCK. A readable head that names no constructor keeps its own
/// `UnknownConstructor` diagnostic rather than degrading to malformed.
#[test]
fn pat_ctor_unknown_constructor_diagnostic_is_unchanged() {
    let errors = diagnostics(&pat_ctor_head("NoSuchCtorZZZ"));
    assert!(
        errors.iter().any(
            |e| matches!(e.kind, CheckErrorKind::UnknownConstructor { .. })
                && e.message.contains("unknown constructor: NoSuchCtorZZZ")
        ),
        "a readable out-of-scope constructor head keeps its own diagnostic; \
         got:\n  {}",
        rendered(&errors)
    );
}

// ── R3: the `pat-record` constructor head ──────────────────────────────────

/// REGRESSION. The `DeepTag::PatRecord` arm's sibling of the R2 read. Red on
/// `3b701e54b`: score 1.0, empty error vector.
#[test]
fn pat_record_head_that_is_present_and_unreadable_is_rejected() {
    for (label, child) in UNREADABLE {
        assert_slot_rejection(
            &pat_record_head(child),
            "pat-record",
            "constructor name",
            &format!("pat-record head given a {label}"),
        );
    }
}

/// DISPOSITION LOCK.
#[test]
fn pat_record_with_a_real_constructor_head_still_checks() {
    assert_accepted(
        &pat_record_head("Circle"),
        "pat-record with an in-scope constructor",
    );
    assert_accepted(
        &pat_record_head("Circle (kv {} r (pat-var {} v))"),
        "pat-record with an in-scope constructor and a field pattern",
    );
}

// ── R4: the `grad` operand ─────────────────────────────────────────────────

/// REGRESSION. `infer_grad`'s `_ => vg.fresh_type()` arm, commented "Can't
/// determine function structure, return fresh var". The sibling `infer_vmap`
/// rejects the identical input; both were written to the same template and
/// only one kept a disposition. Red on `3b701e54b`: score 1.0.
#[test]
fn grad_applied_to_a_non_function_is_rejected() {
    let scalar = "(def {} d2 (fn {} (params {} (x {type: (t-prim {} f32)}))
    (grad {} (var {} x))))";
    let errors = diagnostics(scalar);
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("grad expects a function, got f32")),
        "an f32-typed grad operand must be rejected the way the sibling vmap \
         rejects it ([04-TOT-1]); got:\n  {}",
        rendered(&errors)
    );

    let tensor = "(def {} d5 (fn {} (params {} (x {type: (t-tensor {} (d-name {} k) \
(t-prim {} f32))}))
    (grad {} (var {} x))))";
    let errors = diagnostics(tensor);
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("grad expects a function, got")),
        "a tensor-typed grad operand must be rejected too; got:\n  {}",
        rendered(&errors)
    );
}

/// DISPOSITION LOCK. `grad` of a real function keeps working, and the `wrt`
/// slot's VALUE checks keep their own diagnostics after Slice 2 moved the
/// slot's READ onto the seam.
///
/// The split is the one `vmap`'s axis already keeps: an unreadable child is a
/// malformed slot, covered by `migrated_selector_reads_reject_through_the_seam`
/// above, while a readable index that is out of range is a value error and
/// stays a `DimensionMismatch`. Before Slice 2 this test asserted the
/// unreadable case's old text, "grad `wrt` must be an integer parameter index
/// or tuple of indices"; that message is gone by design and its row moved.
#[test]
fn grad_of_a_function_still_checks_and_wrt_value_checks_are_unchanged() {
    assert_accepted(
        "(def {} d4 (grad {} (var {} double)))",
        "grad of a function",
    );
    let errors = diagnostics("(def {} d6 (grad {} (var {} double) -1))");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::DimensionMismatch)
                && e.message
                    .contains("grad `wrt` index must be non-negative, got -1")),
        "a readable negative `wrt` index keeps its own value diagnostic; got:\n  {}",
        rendered(&errors)
    );
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::MalformedForm)),
        "a readable index is not a malformed slot; got:\n  {}",
        rendered(&errors)
    );
}

// ── R5: the `kv` key ───────────────────────────────────────────────────────

/// REGRESSION on the diagnostic's IDENTITY. On `3b701e54b` this program WAS
/// rejected, but by §C4.1's owner-stamp tripwire firing on the unstamped
/// `lit` in the value slot -- an `internal:` invariant violation blaming a
/// node the author did not write wrongly. The `else { continue }` skipped the
/// `infer_expr(value, ..)` two lines below, which is why the value went
/// unstamped. A coarser "is it rejected" assertion passes in both states, so
/// this row asserts the identity instead.
#[test]
fn record_construction_with_an_unreadable_key_names_the_kv_slot() {
    for (label, child) in UNREADABLE {
        let body = record_kv_key(child);
        let label = format!("record construction whose kv key is a {label}");
        assert_slot_rejection(&body, "kv", "symbol field name", &label);
        assert_not_misattributed(&body, &label);
    }
}

/// DISPOSITION LOCK. The two readable keys keep their existing verdicts: a
/// declared field passes, an undeclared one gets the field-name diagnostic.
#[test]
fn record_construction_readable_key_verdicts_are_unchanged() {
    assert_accepted(
        &record_kv_key("r"),
        "record construction with a declared field",
    );
    let errors = diagnostics(&record_kv_key("nosuchfield"));
    assert!(
        errors.iter().any(|e| e
            .message
            .contains("unknown record field 'nosuchfield' in construction of Circle")),
        "a readable unknown field keeps its own diagnostic; got:\n  {}",
        rendered(&errors)
    );
}

/// REGRESSION. `infer_record_update` carries the identical `else { continue }`
/// and produced the identical misattribution.
#[test]
fn record_update_with_an_unreadable_key_names_the_kv_slot() {
    for (label, child) in UNREADABLE {
        let body = record_update_kv_key(child);
        let label = format!("record update whose kv key is a {label}");
        assert_slot_rejection(&body, "kv", "symbol field name", &label);
        assert_not_misattributed(&body, &label);
    }
}

/// DISPOSITION LOCK.
#[test]
fn record_update_with_a_readable_key_still_checks() {
    assert_accepted(
        &record_update_kv_key("r"),
        "record update with a declared field",
    );
}

/// REGRESSION. The third `kv` key read, inside `pattern_bindings`'
/// `DeepTag::PatRecord` arm, defaulted an unreadable key to `vg.fresh_type()`
/// and recursed into the sub-pattern anyway. This one was fully vacuous on
/// `3b701e54b`: score 1.0, empty error vector.
#[test]
fn record_pattern_with_an_unreadable_key_names_the_kv_slot() {
    for (label, child) in UNREADABLE {
        let body = pat_record_kv_key(child);
        let label = format!("record pattern whose kv key is a {label}");
        assert_slot_rejection(&body, "kv", "symbol field name", &label);
    }
}

/// DISPOSITION LOCK.
#[test]
fn record_pattern_with_a_readable_key_still_checks() {
    assert_accepted(
        &pat_record_kv_key("r"),
        "record pattern with a declared field",
    );
}

// ── The two binder-slot siblings in the same four-read arm ─────────────────

/// REGRESSION on the diagnostic's identity. `pat-var`'s name read is the same
/// `if let Some(..)` with no `else`; an unreadable name bound nothing and the
/// owner-stamp tripwire fired on the pattern node itself. `pat-var` and
/// `pat-as` are `Binder` slots, and this file makes no claim about the
/// `Binder` role as a whole -- these are two named instances.
#[test]
fn pattern_binder_names_that_are_unreadable_name_their_own_slot() {
    for (label, child) in UNREADABLE {
        let body = pat_var_name(child);
        let label_var = format!("pat-var whose name child is a {label}");
        assert_slot_rejection(&body, "pat-var", "binding name", &label_var);
        assert_not_misattributed(&body, &label_var);

        let body = pat_as_name(&format!("{child} (pat-wild {{}})"));
        let label_as = format!("pat-as whose name child is a {label}");
        assert_slot_rejection(&body, "pat-as", "binding name", &label_as);
        assert_not_misattributed(&body, &label_as);
    }
}

/// DISPOSITION LOCK.
#[test]
fn readable_pattern_binder_names_still_bind() {
    assert_accepted(&pat_var_name("v"), "pat-var with a symbol name");
    assert_accepted(&pat_as_name("v (pat-wild {})"), "pat-as with a symbol name");
}

// ── chelis#1525: the `pat-lit` value ───────────────────────────────────────

/// REGRESSION. `literal_pattern_atom` returns `None` for any non-atom child
/// and `check_literal_pattern` declines, so `(pat-lit {} (lit {} 1))` scored
/// 1.0 with [04-PAT-1] never applied. spec/03-deep-syntax.md §6.3 settles the
/// shape: "patterns do not contain expression nodes", so a `lit` node in a
/// `pat-lit` value slot is malformed Deep rather than a typing question.
///
/// The float row is the one that shows the vacuity biting: `1.5` as a bare
/// atom against an `i32` scrutinee is rejected under [04-PAT-1] today, and
/// wrapping it in a `lit` node made the same program check clean.
#[test]
fn pat_lit_value_that_is_not_a_literal_atom_is_rejected() {
    for value in [
        "(lit {} 1)",
        "(lit {type: (t-prim {} f32)} 1.5)",
        "(var {} zz)",
        "(app {} (var {} missing_fn_qqq) (var {} missing_arg_www))",
    ] {
        assert_slot_rejection(
            &pat_lit_value(value),
            "pat-lit",
            "literal value",
            &format!("pat-lit whose value child is `{value}`"),
        );
    }
}

/// DISPOSITION LOCK. A bare literal atom is the readable case: the agreeing
/// family still passes and the disagreeing family still gets [04-PAT-1].
#[test]
fn pat_lit_with_a_literal_atom_keeps_its_existing_verdicts() {
    assert_accepted(
        &pat_lit_value("1"),
        "pat-lit with an integer atom against an i32 scrutinee",
    );
    let errors = diagnostics(&pat_lit_value("1.5"));
    assert!(
        errors.iter().any(|e| e.message.contains("[04-PAT-1]")),
        "a readable float atom against an i32 scrutinee keeps its \
         [04-PAT-1] rejection; got:\n  {}",
        rendered(&errors)
    );
}

// ── Over-rejection controls: the sanctioned bare lists ─────────────────────

/// DISPOSITION LOCK, and the most important control in the file. The seam must
/// reject UNREADABILITY, not non-tagged-ness. Every slot below is a structural
/// bare list that a well-formed program legitimately contains, enumerated from
/// `crates/chelis-surf/src/desugar.rs`'s `bare_list()` (defined at `:481`) and
/// its call sites: `fn` parameter lists (`:1200`, `:1545`), an import name
/// list (`:1287`), a qualified import's empty name list (`:1297`), an absent
/// match guard (`:1804`), and the unit literal's empty list (`:1836`). Without
/// this test the cheapest wrong fix -- rejecting every bare list -- passes
/// every negative row above.
#[test]
fn sanctioned_bare_list_slots_keep_checking() {
    assert_accepted(
        "(def {} p1 (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
        "an annotated fn parameter list (desugar.rs:1200)",
    );
    assert_accepted(
        "(def {} p2 (fn {} (params {} x) (var {} x)))",
        "a bare fn parameter list (desugar.rs:1545)",
    );
    assert_accepted(
        "(def {} p3 (fn {} (params {}) (lit {type: (t-prim {} f32)} 1.0)))",
        "an empty fn parameter list",
    );
    assert_accepted(
        "(def {} p4 (fn {} (params {} (x {type: (t-prim {} f32)}))
      (match {} (var {} x) (arm {} (pat-wild {}) () (var {} x)))))",
        "a match arm with an absent guard (desugar.rs:1804)",
    );
    assert_accepted(
        "(def {} p5 (lit {type: (t-unit {})} ()))",
        "the unit literal's empty list (desugar.rs:1836)",
    );
    assert_accepted(
        "(deftype {} Solo () (variant {} Solo (field {} r (t-prim {} f32))))
  (def {} p6 (record {} Solo (kv {} r (lit {type: (t-prim {} f32)} 1.0))))",
        "an empty deftype type-parameter list",
    );
    assert_accepted(
        "(deftype {} Wrap (a) (variant {} Wrap (field {} v (t-var {} a))))
  (def {} p7 (record {} Wrap (kv {} v (lit {type: (t-prim {} f32)} 1.0))))",
        "a non-empty deftype type-parameter list",
    );
}

/// DISPOSITION LOCK. The import name lists are declaration-level bare lists in
/// the same `bare_list()` family; they are exercised as their own program
/// because an import of a module this fixture does not provide is a name
/// question rather than a slot-read one.
#[test]
fn import_name_lists_are_not_slot_reads() {
    let source = "(module {} m.main
  (import {} m.other (copy fill))
  (import {} m.qual ())
  (import-all {} m.every)
  (def {} q1 (lit {type: (t-prim {} f32)} 1.0)))\n";
    let errors = match check_ir_program(&stamped(source)) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    };
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::MalformedForm)),
        "import name lists (desugar.rs:1287/:1297) are sanctioned bare lists \
         and must not be read as malformed slots; got:\n  {}",
        rendered(&errors)
    );
}

/// DISPOSITION LOCK. Metadata-carrying children are the ordinary shape of
/// compiler-produced Deep; the seam reads the child, not its metadata.
#[test]
fn metadata_carrying_children_still_check() {
    assert_accepted(
        "(def {span: \"surf:1..2\"} m1
    (fn {} (params {} (x {type: (t-prim {} f32)}))
      (app {span: \"surf:3..4\"} (var {span: \"surf:5..6\"} mul)
        (var {} x) (lit {span: \"surf:7..8\", type: (t-prim {} f32)} 2.0))))",
        "a program whose nodes carry span metadata",
    );
}

// ── The four correctly-rejecting slots stay correct ────────────────────────

/// MIGRATION LOCKS (Slice 2). These four slots already rejected an unreadable
/// child correctly before PP8; Slice 2 moves them onto the same seam, so one
/// mechanism owns every role-slot read and chelis#874's condition is met.
///
/// Each row is red on `1b685d8c4` (the pre-migration text) and green after,
/// which makes them regression tests for the migration and locks for the
/// verdict: the slot still rejects, and the diagnostic still names the form
/// and what it found. What changes is the text and the kind, deliberately --
/// a seam that let each caller keep its own message would guarantee nothing
/// (`spec/design/checker_totality.md` §PP8, "What we deliver", the
/// implementation note).
///
/// `tuple-get` is the one slot whose owner says more than the shared
/// describer can: `describe_tuple_index` peels a `lit` wrapper to name the
/// payload atom's family and value, and chelis#1107's PP7 row exists because
/// the two ingresses once disagreed on exactly that wording. So the seam's
/// caller-detail affordance returns here with `describe_tuple_index` as its
/// only consumer, and the detail is suppressed when the generic describer
/// already names the atom.
#[test]
fn migrated_selector_reads_reject_through_the_seam() {
    let errors = diagnostics(
        "(deftype {} Solo () (variant {} Solo (field {} r (t-prim {} f32))))
  (def {} a1 (access {} (record {} Solo (kv {} r (lit {type: (t-prim {} f32)} 1.0)))
    (app {} (var {} missing_fn_qqq) (var {} missing_arg_www))))",
    );
    assert_seam_message(
        &errors,
        "malformed `access`: expected a symbol field name as child 1, \
         found a `app` form",
        "`access` rejects through the seam",
    );

    let errors = diagnostics(
        "(def {} t1 (tuple-get {} (tuple {} (lit {type: (t-prim {} f32)} 1.0)
      (lit {type: (t-prim {} f32)} 2.0)) (var {} nonexistent_name_zzz)))",
    );
    assert_seam_message(
        &errors,
        "malformed `tuple-get`: expected a non-negative integer index as child 1, \
         found a `var` form",
        "`tuple-get` rejects through the seam",
    );

    let errors = diagnostics(
        "(def {} t3 (tuple-get {} (tuple {} (lit {type: (t-prim {} f32)} 1.0)
      (lit {type: (t-prim {} f32)} 2.0)) zz))",
    );
    assert_seam_message(
        &errors,
        "malformed `tuple-get`: expected a non-negative integer index as child 1, \
         found symbol `zz`",
        "`tuple-get` keeps the generic describer when it already names the atom",
    );

    let errors = diagnostics(
        "(def {} c1 (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} i32)
    (var {} nonexistent_name_zzz)))",
    );
    assert_seam_message(
        &errors,
        "malformed `cast`: expected a symbol mode selector as child 2, \
         found a `var` form",
        "`cast`'s mode selector rejects through the seam",
    );

    let errors = diagnostics("(def {} g1 (grad {} (var {} double) (var {} nonexistent_name_zzz)))");
    assert_seam_message(
        &errors,
        "malformed `grad`: expected an integer parameter index or a tuple of \
         integer parameter indices as child 1, found a `var` form",
        "`grad`'s `wrt` selector rejects through the seam",
    );
}

/// MIGRATION LOCK. `tuple-get`'s caller detail is what chelis#1107's PP7 row
/// protects: a `lit`-wrapped index must be described by its payload atom, not
/// as "a `lit` form". This is the row that decides the `detail` affordance
/// earns its place.
#[test]
fn tuple_get_detail_names_the_payload_atom_behind_a_lit_wrapper() {
    let errors = diagnostics(
        "(def {} t4 (tuple-get {} (tuple {} (lit {type: (t-prim {} f32)} 1.0)
      (lit {type: (t-prim {} f32)} 2.0)) (lit {type: (t-prim {} i32)} -1)))",
    );
    assert_seam_message(
        &errors,
        "malformed `tuple-get`: expected a non-negative integer index as child 1, \
         found a `lit` form (integer literal -1)",
        "a `lit`-wrapped negative index names its payload atom",
    );
}

/// DISPOSITION LOCK, green in both states. A genuinely out-of-bounds index is
/// NOT malformed and keeps `TupleIndexOutOfBounds`. Its job is the
/// over-rejection half of the migration: before, `MalformedForm` was
/// unreachable from `tuple-get` at all, so the second assertion only becomes
/// capable of failing once the migration lands. Migrating the unreadable-index
/// branch off `TupleIndexOutOfBounds` is also what lets that kind mean only
/// what it says.
#[test]
fn a_readable_out_of_bounds_tuple_index_is_not_malformed() {
    let errors = diagnostics(
        "(def {} t5 (tuple-get {} (tuple {} (lit {type: (t-prim {} f32)} 1.0))
      (lit {type: (t-prim {} i32)} 7)))",
    );
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::TupleIndexOutOfBounds)
                && e.message.contains("out of bounds for tuple of size 1")),
        "a readable but out-of-range index keeps TupleIndexOutOfBounds; got:\n  {}",
        rendered(&errors)
    );
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::MalformedForm)),
        "a readable index is not a malformed form; got:\n  {}",
        rendered(&errors)
    );
}

/// DISPOSITION LOCK. The readable forms of the same three slots.
#[test]
fn already_total_selector_reads_still_accept_readable_children() {
    assert_accepted(
        "(deftype {} Solo () (variant {} Solo (field {} r (t-prim {} f32))))
  (def {} a2 (access {} (record {} Solo (kv {} r (lit {type: (t-prim {} f32)} 1.0))) r))",
        "`access` with a symbol field name",
    );
    assert_accepted(
        "(def {} t2 (tuple-get {} (tuple {} (lit {type: (t-prim {} f32)} 1.0)
      (lit {type: (t-prim {} f32)} 2.0)) (lit {type: (t-prim {} i32)} 0)))",
        "`tuple-get` with an integer index",
    );
    assert_accepted(
        "(def {} c2 (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} i32) trunc))",
        "`cast` with the `trunc` mode selector",
    );
    assert_accepted(
        "(def {} c3 (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} i32)))",
        "`cast` with the mode selector omitted",
    );
}

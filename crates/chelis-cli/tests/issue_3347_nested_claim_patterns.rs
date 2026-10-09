//! chelis#3347 and chelis#2644, spec/04 §4.7 and [04-ADT-4]
//! (runtime_extents.md C6.5): a declared extent stays an obligation when the
//! tensor it claims sits inside a tuple, an `Option`, a record or a nominal
//! value, including one a nominal dimension argument supplies. Every runtime
//! row reads its extent from a file, so no literal reaches the producer
//! through inlining, and runs on Eval and on compiled, linked and executed C
//! with the matching-extent control executing unchanged.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

use std::fs;

const PRELUDE: &str = r#"type Box[n] =
  | Box { v: tensor[n, f32] }
type Pair[n] =
  | Leaf { v: tensor[n, f32] }
  | Node { l: Pair[n], r: Pair[n] }
type Two[n] =
  | Has { v: tensor[n, f32] }
  | Lacks { w: i64 }
type Holder =
  | Holder { v: tensor[3, f32] }
def size_from(path: string) -> i64 ! { IO } = string_len(read_file(path))
def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)
def make[n](size: i64) -> Box[n] = Box { v: produce(size) }
def width[n](b: Box[n]) -> i64 =
  match b with {
    | Box { v } => shape(v, 0i32)
  }
def leaf_width[n](p: Pair[n]) -> i64 =
  match p with {
    | Leaf { v } => shape(v, 0i32)
    | Node { l, r } => -1i64
  }
def right_width[n](p: Pair[n]) -> i64 =
  match p with {
    | Node { l, r } => leaf_width(r)
    | Leaf { v } => -1i64
  }
"#;

const DIRECT: &str = r#"def three(size: i64) -> Box[3] = Box { v: insert(scalar_to_tensor(0.0f32), 0i32, size) }
out = width(three(size_from("PATH")))
"#;

const GENERIC: &str = r#"def three(size: i64) -> Box[3] = make(size)
out = width(three(size_from("PATH")))
"#;

const LOCAL: &str = r#"def three(size: i64) -> i64 = {
  b: Box[3] = Box { v: insert(scalar_to_tensor(0.0f32), 0i32, size) }
  match b with {
    | Box { v } => shape(v, 0i32)
  }
}
out = three(size_from("PATH"))
"#;

const FORMAL: &str = r#"def claimed(b: Box[3]) -> i64 = width(b)
out = claimed(make(size_from("PATH")))
"#;

const RECURSIVE: &str = r#"def leaf[n](size: i64) -> Pair[n] ! { IO } = {
  _ = print("leaf")
  Leaf { v: insert(scalar_to_tensor(0.0f32), 0i32, size) }
}
def two(size: i64) -> Pair[3] ! { IO } = Node { l: leaf(3i64), r: leaf(size) }
out = right_width(two(size_from("PATH")))
"#;

const NAMED: &str = r#"def same[k](x: tensor[k, f32], size: i64) -> Box[k] = Box { v: produce(size) }
out = width(same(to_tensor([1.0f32, 2.0f32, 3.0f32]), size_from("PATH")))
"#;

const TUPLE_LITERAL: &str = r#"def three(size: i64) -> (tensor[3, f32], i64) = (produce(size), 1i64)
out = shape(three(size_from("PATH")).0, 0i32)
"#;

const TUPLE_NAMED: &str = r#"def same[k](x: tensor[k, f32], size: i64) -> (i64, tensor[k, f32]) = (1i64, produce(size))
out = shape(same(to_tensor([1.0f32, 2.0f32, 3.0f32]), size_from("PATH")).1, 0i32)
"#;

const OPTION_LITERAL: &str = r#"def three(size: i64) -> Option[tensor[3, f32]] = Some(produce(size))
out = match three(size_from("PATH")) with {
  | Some(v) => shape(v, 0i32)
  | None => -1i64
}
"#;

const OPTION_NAMED: &str = r#"def same[k](x: tensor[k, f32], size: i64) -> Option[tensor[k, f32]] = Some(produce(size))
out = match same(to_tensor([1.0f32, 2.0f32, 3.0f32]), size_from("PATH")) with {
  | Some(v) => shape(v, 0i32)
  | None => -1i64
}
"#;

const RECORD_LITERAL: &str = r#"def three(size: i64) -> Holder = Holder { v: produce(size) }
out = match three(size_from("PATH")) with {
  | Holder { v } => shape(v, 0i32)
}
"#;

const TUPLE_BINDER: &str = r#"def pair(size: i64) -> (tensor[*, f32], i64) = (produce(size), 1i64)
def both[k](p: (tensor[k, f32], i64), y: tensor[k, f32]) -> i64 = shape(y, 0i32)
out = both(pair(size_from("PATH")), to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#;

const BOX_BINDER: &str = r#"def both[k](b: Box[k], y: tensor[k, f32]) -> i64 = shape(y, 0i32)
out = both(make(size_from("PATH")), to_tensor([1.0f32, 2.0f32, 3.0f32]))
"#;

const ABSENT: &str = r#"def lacks(size: i64) -> Two[3] = Lacks { w: size }
def maybe(size: i64) -> Option[tensor[3, f32]] = if gt(size, 4i64) then None else Some(produce(size))
def has_width(t: Two[3]) -> i64 =
  match t with {
    | Has { v } => shape(v, 0i32)
    | Lacks { w } => w
  }
def maybe_width(o: Option[tensor[3, f32]]) -> i64 =
  match o with {
    | Some(v) => shape(v, 0i32)
    | None => -1i64
  }
out = (has_width(lacks(size_from("PATH"))), maybe_width(maybe(size_from("PATH"))))
"#;

const EFFECTS: &str = r#"def three(size: i64) -> Box[3] ! { IO } = {
  _ = print("before-producer")
  b = Box { v: produce(size) }
  _ = print("after-producer")
  b
}
out = width(three(size_from("PATH")))
"#;

const UNCLAIMED: &str = r#"out = width(make(size_from("PATH")))
"#;

const STATIC_LITERAL: &str = r#"def bad() -> Box[3] = Box { v: to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]) }
out = width(bad())
"#;

const LITERAL_VISIBLE: &str = r#"def three() -> Box[3] = make(5i64)
out = width(three())
"#;

const NONREGULAR: &str = r#"type Nest[a] =
  | Done { h: a }
  | More { t: Nest[List[a]] }
def deep(size: i64) -> Nest[tensor[3, f32]] = Done { h: produce(size) }
def depth(n: Nest[tensor[3, f32]]) -> i64 =
  match n with {
    | Done { h } => shape(h, 0i32)
    | More { t } => -1i64
  }
out = depth(deep(size_from("PATH")))
"#;

/// Run `case` after the shared declarations with `size` read from a file.
fn run(case: &str, size: usize, native: bool) -> (bool, String) {
    let inputs = tempfile::tempdir().expect("runtime inputs");
    let path = inputs.path().join("size.txt");
    fs::write(&path, "x".repeat(size)).expect("size input");
    let source = format!("{PRELUDE}{case}").replace("PATH", &path.display().to_string());
    result_claims::run(&source, native)
}

fn lane(native: bool) -> &'static str {
    if native { "compiled C" } else { "Eval" }
}

/// The extent-5 run traps once, with exactly `context` and the Domain line
/// of `op`; the extent-3 control prints `control`.
fn assert_trap_and_control(case: &str, native: bool, context: &str, op: &str, control: &str) {
    let (ok, output) = run(case, 5, native);
    let lane = lane(native);
    assert!(!ok, "{lane}: the disagreeing extent must trap\n{output}");
    assert!(
        output
            .lines()
            .any(|line| line.trim_start_matches("error: ") == context),
        "{lane}: expected `{context}`\n{output}"
    );
    assert!(
        output
            .lines()
            .any(|line| line == format!("numeric trap: domain in {op} at i64")),
        "{lane}: expected the Domain trap in {op}\n{output}"
    );
    assert_eq!(
        output.matches("numeric trap:").count(),
        1,
        "{lane}\n{output}"
    );
    assert!(!output.contains("out ="), "{lane}\n{output}");
    let (ok, output) = run(case, 3, native);
    assert!(ok, "{lane}: the agreeing extent must run\n{output}");
    assert!(
        output.contains(control),
        "{lane}: expected `{control}`\n{output}"
    );
}

fn insert_literal(native: bool, case: &str) {
    assert_trap_and_control(
        case,
        native,
        "extent `3`: claimed = 3, insert axis 0 = 5",
        "insert",
        "out = 3",
    );
}

fn insert_named(native: bool, case: &str) {
    assert_trap_and_control(
        case,
        native,
        "extent `k`: x axis 0 = 3, insert axis 0 = 5",
        "insert",
        "out = 3",
    );
}

#[test]
fn eval_nongeneric_box_result_traps_at_insert() {
    insert_literal(false, DIRECT);
}

#[test]
fn c_nongeneric_box_result_traps_at_insert() {
    insert_literal(true, DIRECT);
}

#[test]
fn eval_generic_callee_inherits_box_result_claim() {
    insert_literal(false, GENERIC);
}

#[test]
fn c_generic_callee_inherits_box_result_claim() {
    insert_literal(true, GENERIC);
}

#[test]
fn eval_local_box_ascription_traps_at_its_initializer() {
    insert_literal(false, LOCAL);
}

#[test]
fn c_local_box_ascription_traps_at_its_initializer() {
    insert_literal(true, LOCAL);
}

/// Each lane renders the entry literal as it renders a bare tensor formal's.
#[test]
fn eval_box_formal_traps_at_entry() {
    assert_trap_and_control(
        FORMAL,
        false,
        "extent `3`: claimed = 3, b.v axis 0 = 5",
        "load",
        "out = 3",
    );
}

#[test]
fn c_box_formal_traps_at_entry() {
    assert_trap_and_control(
        FORMAL,
        true,
        "input `b.v` axis 0 expected 3, got 5",
        "load",
        "out = 3",
    );
}

/// The `l` leaf completes before the `r` leaf's producer traps.
fn recursive_pair(native: bool) {
    insert_literal(native, RECURSIVE);
    let (_, output) = run(RECURSIVE, 5, native);
    assert_eq!(
        output.matches("leaf").count(),
        2,
        "{}\n{output}",
        lane(native)
    );
}

#[test]
fn eval_recursive_pair_traps_at_the_right_leaf() {
    recursive_pair(false);
}

#[test]
fn c_recursive_pair_traps_at_the_right_leaf() {
    recursive_pair(true);
}

#[test]
fn eval_named_box_result_names_its_witness() {
    insert_named(false, NAMED);
}

#[test]
fn c_named_box_result_names_its_witness() {
    insert_named(true, NAMED);
}

fn tuple_option_record(native: bool) {
    insert_literal(native, TUPLE_LITERAL);
    insert_named(native, TUPLE_NAMED);
    insert_literal(native, OPTION_LITERAL);
    insert_named(native, OPTION_NAMED);
    insert_literal(native, RECORD_LITERAL);
}

#[test]
fn eval_tuple_option_and_record_results_trap_at_their_producer() {
    tuple_option_record(false);
}

#[test]
fn c_tuple_option_and_record_results_trap_at_their_producer() {
    tuple_option_record(true);
}

/// A binder first witnessed inside a formal's tuple or constructor field is
/// the canonical witness a later direct formal is compared with.
fn nested_binder_witness(native: bool) {
    for (case, path) in [(TUPLE_BINDER, "p.0"), (BOX_BINDER, "b.v")] {
        assert_trap_and_control(
            case,
            native,
            &format!("extent `k`: {path} axis 0 = 5, y axis 0 = 3"),
            "load",
            "out = 3",
        );
    }
}

#[test]
fn eval_nested_binder_witness_guards_a_later_formal() {
    nested_binder_witness(false);
}

#[test]
fn c_nested_binder_witness_guards_a_later_formal() {
    nested_binder_witness(true);
}

/// An untaken constructor or `None` owes nothing, whatever its absent field
/// would have held.
fn absent_variant(native: bool) {
    for (size, first, second) in [(5, 5, -1), (3, 3, 3)] {
        let (ok, output) = run(ABSENT, size, native);
        assert!(ok, "{}\n{output}", lane(native));
        assert!(
            output.contains(&format!("out.0 = {first}"))
                && output.contains(&format!("out.1 = {second}")),
            "{}\n{output}",
            lane(native)
        );
    }
}

#[test]
fn eval_absent_variant_owes_nothing() {
    absent_variant(false);
}

#[test]
fn c_absent_variant_owes_nothing() {
    absent_variant(true);
}

/// The producer checks before the effect that follows it.
fn effect_order(native: bool) {
    insert_literal(native, EFFECTS);
    let (_, output) = run(EFFECTS, 5, native);
    assert!(output.contains("before-producer"), "{output}");
    assert!(!output.contains("after-producer"), "{output}");
    let (_, output) = run(EFFECTS, 3, native);
    assert!(
        output.find("before-producer") < output.find("after-producer"),
        "{output}"
    );
}

#[test]
fn eval_effects_surround_the_nested_producer_guard() {
    effect_order(false);
}

#[test]
fn c_effects_surround_the_nested_producer_guard() {
    effect_order(true);
}

/// A generic call whose result meets no claim checks nothing.
fn unclaimed(native: bool) {
    for size in [5, 3] {
        let (ok, output) = run(UNCLAIMED, size, native);
        assert!(ok, "{}\n{output}", lane(native));
        assert!(output.contains(&format!("out = {size}")), "{output}");
    }
}

#[test]
fn eval_unclaimed_generic_call_checks_nothing() {
    unclaimed(false);
}

#[test]
fn c_unclaimed_generic_call_checks_nothing() {
    unclaimed(true);
}

/// A five-element literal field under `Box[3]` stays the checker's verdict.
#[test]
fn static_literal_field_is_a_checker_rejection_on_both_lanes() {
    for native in [false, true] {
        let dir = tempfile::tempdir().expect("source dir");
        let path = dir.path().join("completion.ch");
        fs::write(&path, format!("{PRELUDE}{STATIC_LITERAL}")).expect("fixture");
        let mut command = assert_cmd::Command::cargo_bin("chelis").expect("chelis");
        command.env("CHELIS_STYLE_GATE_DISABLE", "1");
        if native {
            command
                .args(["build", "--emit-c", "--allow-style-violations"])
                .arg(&path)
                .args(["--target", "c", "-o"])
                .arg(dir.path().join("c"));
        } else {
            command
                .args(["eval", "--allow-style-violations", "--file"])
                .arg(&path);
        }
        let output = command.output().expect("run");
        let text = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{}\n{text}", lane(native));
        assert!(
            text.contains("DimensionMismatch") && text.contains("expected `() -> Box 3`"),
            "{}\n{text}",
            lane(native)
        );
    }
}

/// A literal extent visible at the call never executes successfully under a
/// disagreeing nested claim.
#[test]
fn literal_visible_generic_call_never_succeeds() {
    for native in [false, true] {
        let (ok, output) = run(LITERAL_VISIBLE, 0, native);
        assert!(!ok, "{}\n{output}", lane(native));
        assert!(!output.contains("out ="), "{output}");
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in insert at i64"),
            "{}\n{output}",
            lane(native)
        );
    }
}

/// A claim through recursion whose arguments never close has no finite
/// pattern; both lanes refuse it by name instead of skipping it.
#[test]
fn non_regular_recursion_claim_is_refused_on_both_lanes() {
    for native in [false, true] {
        let dir = tempfile::tempdir().expect("source dir");
        let source = dir.path().join("completion.ch");
        let input = dir.path().join("size.txt");
        fs::write(&input, "xxx").expect("size input");
        fs::write(
            &source,
            format!("{PRELUDE}{NONREGULAR}").replace("PATH", &input.display().to_string()),
        )
        .expect("fixture");
        let mut command = assert_cmd::Command::cargo_bin("chelis").expect("chelis");
        command.env("CHELIS_STYLE_GATE_DISABLE", "1");
        if native {
            command
                .args(["build", "--emit-c", "--allow-style-violations"])
                .arg(&source)
                .args(["--target", "c", "-o"])
                .arg(dir.path().join("c"));
        } else {
            command
                .args(["eval", "--allow-style-violations", "--file"])
                .arg(&source);
        }
        let output = command.output().expect("run");
        let text = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{}\n{text}", lane(native));
        assert!(
            text.contains("an extent claim passes through `Nest`")
                && text.contains("no finite pattern"),
            "{}\n{text}",
            lane(native)
        );
    }
}

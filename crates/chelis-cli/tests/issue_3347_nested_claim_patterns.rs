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

const MAP: &str = r#"def three(sizes: List[i64]) -> List[Box[3]] = map(fn (s: i64) -> make(s), sizes)
out = fold(fn (acc: i64, b: Box[3]) -> add(acc, width(b)), 0i64, three([3i64, size_from("PATH")]))
"#;

const FLAT_MAP: &str = r#"def three(sizes: List[i64]) -> List[Box[3]] = flat_map(fn (s: i64) -> [make(s)], sizes)
out = fold(fn (acc: i64, b: Box[3]) -> add(acc, width(b)), 0i64, three([3i64, size_from("PATH")]))
"#;

const SCAN: &str = r#"def three(sizes: List[i64]) -> List[(tensor[3, f32], i64)] = scan(fn (acc: (tensor[*, f32], i64), s: i64) -> (produce(s), s), (produce(3i64), 0i64), sizes)
out = fold(fn (acc: i64, t: (tensor[*, f32], i64)) -> add(acc, shape(t.0, 0i32)), 0i64, three([3i64, size_from("PATH")]))
"#;

const FILTER: &str = r#"def three(sizes: List[i64]) -> List[tensor[3, f32]] = filter(fn (t: tensor[*, f32]) -> gt(shape(t, 0i32), 0i64), map(fn (s: i64) -> produce(s), sizes))
out = fold(fn (acc: i64, t: tensor[*, f32]) -> add(acc, shape(t, 0i32)), 0i64, three([3i64, size_from("PATH")]))
"#;

const PARTITION: &str = r#"def three(sizes: List[i64]) -> (List[tensor[3, f32]], List[tensor[3, f32]]) = partition(fn (t: tensor[*, f32]) -> gt(shape(t, 0i32), 0i64), map(fn (s: i64) -> produce(s), sizes))
out = fold(fn (acc: i64, t: tensor[*, f32]) -> add(acc, shape(t, 0i32)), 0i64, three([3i64, size_from("PATH")]).0)
"#;

const ALIAS_RESULT: &str = r#"type B3 = Box[3]
def three(size: i64) -> B3 = make(size)
out = width(three(size_from("PATH")))
"#;

const ALIAS_GENERIC: &str = r#"type BB[m] = Box[m]
def three(size: i64) -> BB[3] = make(size)
out = width(three(size_from("PATH")))
"#;

const ALIAS_TUPLE: &str = r#"type B3 = Box[3]
def three(size: i64) -> (B3, i64) = (make(size), 1i64)
out = width(three(size_from("PATH")).0)
"#;

const ALIAS_FORMAL: &str = r#"type B3 = Box[3]
def claimed(b: B3) -> i64 = width(b)
out = claimed(make(size_from("PATH")))
"#;

const ALIAS_ARGUMENT_BOX: &str = r#"type Wrap[a] =
  | Wrap { x: a }
type B3 = Box[3]
def mk(size: i64) -> Wrap[B3] = Wrap { x: make(size) }
out = match mk(size_from("PATH")) with {
  | Wrap { x } => width(x)
}
"#;

const ALIAS_ARGUMENT_TENSOR: &str = r#"type Wrap[a] =
  | Wrap { x: a }
type V3 = tensor[3, f32]
def mk(size: i64) -> Wrap[V3] = Wrap { x: produce(size) }
out = match mk(size_from("PATH")) with {
  | Wrap { x } => shape(x, 0i32)
}
"#;

const ALIAS_ARGUMENT_FORMAL: &str = r#"type Wrap[a] =
  | Wrap { x: a }
type B3 = Box[3]
def claimed(w: Wrap[B3]) -> i64 =
  match w with {
    | Wrap { x } => width(x)
  }
out = claimed(Wrap { x: make(size_from("PATH")) })
"#;

const ALIAS_ARGUMENT_LIST_FORMAL: &str = r#"type Wrap[a] =
  | Wrap { x: a }
type V3 = tensor[3, f32]
def total(ws: List[Wrap[V3]]) -> i64 =
  fold(fn (acc: i64, w: Wrap[V3]) -> match w with {
    | Wrap { x } => add(acc, shape(x, 0i32))
  }, 0i64, ws)
out = total([Wrap { x: produce(3i64) }, Wrap { x: produce(size_from("PATH")) }])
"#;

const RECURSIVE_ALIAS: &str = r#"type Wrap[a] =
  | Wrap { x: a }
type R = Wrap[R]
def f(r: R) -> i64 = 1i64
out = 1i64
"#;

const RECURSIVE_ALIAS_OPTION: &str = r#"type Duo[a, b] =
  | Duo { l: a, r: b }
type S = Option[Duo[S, S]]
def g(s: S) -> i64 = 2i64
out = g(None)
"#;

const CONTAINER_ALIAS: &str = r#"type V = Option[(tensor[3, f32], V)]
def f(v: V) -> i64 = 1i64
out = f(None)
"#;

const GROWING_ALIAS: &str = r#"type G[a] = Option[(a, G[(a, a)])]
def f(g: G[i64]) -> i64 = 1i64
out = f(None)
"#;

const GROWING_ALIAS_TENSOR: &str = r#"type G[a] = Option[(a, G[(a, a)])]
def f(g: G[tensor[3, f32]]) -> i64 = 1i64
out = f(None)
"#;

const DEEP_CLAIMED: &str = r#"type Wrap[n] =
  | W { inner: Wrap[n] }
  | B { v: tensor[n, f32] }
def rec(depth: i64, size: i64) -> Wrap[3] = if gt(depth, 0i64) then W { inner: rec(sub(depth, 1i64), size) } else B { v: produce(size) }
def leaf(w: Wrap[3]) -> i64 =
  match w with {
    | W { inner } => 1i64
    | B { v } => shape(v, 0i32)
  }
out = leaf(rec(size_from("PATH"), 3i64))
"#;

const DEEP_UNCLAIMED: &str = r#"type Wrap =
  | W { inner: Wrap }
  | B { v: tensor[*, f32] }
def rec(depth: i64, size: i64) -> Wrap = if gt(depth, 0i64) then W { inner: rec(sub(depth, 1i64), size) } else B { v: produce(size) }
def leaf(w: Wrap) -> i64 =
  match w with {
    | W { inner } => 1i64
    | B { v } => shape(v, 0i32)
  }
out = leaf(rec(size_from("PATH"), 3i64))
"#;

const MUTUAL_RECURSION: &str = r#"def f[m](u: tensor[m, f32], depth: i64, size: i64) -> Box[m] = if gt(depth, 0i64) then g(u, sub(depth, 1i64), size) else make(size)
def g[k](v: tensor[k, f32], depth: i64, size: i64) -> Box[k] = f(v, depth, size)
out = width(f(produce(3i64), 1i64, size_from("PATH")))
"#;

const SELF_RECURSION_VARYING_WITNESS: &str = r#"def f[m](u: tensor[m, f32], depth: i64, size: i64) -> Box[m] = if gt(depth, 0i64) then f(produce(add(depth, 3i64)), sub(depth, 1i64), size) else make(size)
out = width(f(produce(3i64), 1i64, size_from("PATH")))
"#;

const DEEP_MUTUAL_CLAIMED: &str = r#"type Wrap[n] =
  | W { inner: Wrap[n] }
  | B { v: tensor[n, f32] }
def f[m](u: tensor[m, f32], depth: i64) -> Wrap[m] = if gt(depth, 0i64) then W { inner: g(u, sub(depth, 1i64)) } else B { v: produce(shape(u, 0i32)) }
def g[k](v: tensor[k, f32], depth: i64) -> Wrap[k] = if gt(depth, 0i64) then W { inner: f(v, sub(depth, 1i64)) } else B { v: produce(shape(v, 0i32)) }
def leaf[n](w: Wrap[n]) -> i64 =
  match w with {
    | W { inner } => 1i64
    | B { v } => shape(v, 0i32)
  }
out = leaf(f(produce(3i64), size_from("PATH")))
"#;

const DEEP_MUTUAL_UNCLAIMED: &str = r#"type Wrap =
  | W { inner: Wrap }
  | B { v: tensor[*, f32] }
def f[m](u: tensor[m, f32], depth: i64) -> Wrap = if gt(depth, 0i64) then W { inner: g(u, sub(depth, 1i64)) } else B { v: produce(shape(u, 0i32)) }
def g[k](v: tensor[k, f32], depth: i64) -> Wrap = if gt(depth, 0i64) then W { inner: f(v, sub(depth, 1i64)) } else B { v: produce(shape(v, 0i32)) }
def leaf(w: Wrap) -> i64 =
  match w with {
    | W { inner } => 1i64
    | B { v } => shape(v, 0i32)
  }
out = leaf(f(produce(3i64), size_from("PATH")))
"#;

const LAMBDA_FOLD_BOX: &str = r#"out = width(fold(fn (acc: Box[3], i: i64) -> make(size_from("PATH")), make(3i64), range(0i64, 2i64)))
"#;

const LAMBDA_FOLD_CHAIN: &str = r#"type Chain[n] =
  | End { v: tensor[n, f32] }
  | Link { v: tensor[n, f32], next: Chain[n] }
def grow[m](size: i64, k: i64) -> Chain[m] = fold(fn (acc: Chain[m], i: i64) -> Link { v: produce(3i64), next: acc }, End { v: produce(size) }, range(0i64, k))
def head[n](c: Chain[n]) -> i64 =
  match c with {
    | End { v } => shape(v, 0i32)
    | Link { v, next } => shape(v, 0i32)
  }
out = head(grow(size_from("PATH"), 4i64))
"#;

const LAMBDA_SIBLING_BINDER: &str = r#"def run[n](size: i64) -> i64 = width(fold(fn (acc: Box[n], t: tensor[n, f32]) -> acc, make(size), [produce(3i64), produce(3i64)]))
out = run(size_from("PATH"))
"#;

const LAMBDA_CALLBACKS: [&str; 6] = [
    r#"out = fold(fn (a: i64, w: i64) -> add(a, w), 0i64, map(fn (b: Box[3]) -> width(b), [make(3i64), make(size_from("PATH"))]))
"#,
    r#"out = fold(fn (a: i64, w: i64) -> add(a, w), 0i64, scan(fn (acc: i64, b: Box[3]) -> add(acc, width(b)), 0i64, [make(3i64), make(size_from("PATH"))]))
"#,
    r#"out = fold(fn (a: i64, w: i64) -> add(a, w), 0i64, map(fn (b: Box[3]) -> width(b), filter(fn (b: Box[3]) -> gt(width(b), 0i64), [make(3i64), make(size_from("PATH"))])))
"#,
    r#"out = fold(fn (a: i64, w: i64) -> add(a, w), 0i64, map(fn (b: Box[3]) -> width(b), partition(fn (b: Box[3]) -> gt(width(b), 0i64), [make(3i64), make(size_from("PATH"))]).0))
"#,
    r#"out = fold(fn (a: i64, w: i64) -> add(a, w), 0i64, flat_map(fn (b: Box[3]) -> [width(b)], [make(3i64), make(size_from("PATH"))]))
"#,
    r#"out = (fn (b: Box[3]) -> width(b))(make(size_from("PATH")))
"#,
];

const LAMBDA_TUPLES: [&str; 2] = [
    r#"out = fold(fn (a: i64, w: i64) -> add(a, w), 0i64, map(fn (p: (tensor[3, f32], i64)) -> shape(p.0, 0i32), [(produce(3i64), 1i64), (produce(size_from("PATH")), 2i64)]))
"#,
    r#"out = (fn (p: (tensor[3, f32], i64)) -> shape(p.0, 0i32))((produce(size_from("PATH")), 1i64))
"#,
];

const LAMBDA_OUTER_BINDER: &str = r#"def run[n](w: tensor[n, f32], size: i64) -> i64 = width(fold(fn (acc: Box[n], i: i64) -> acc, make(size), range(0i64, 2i64)))
out = run(produce(4i64), size_from("PATH"))
"#;

const CALLABLE_FORMAL: &str = r#"def apply(f: (Box[3]) -> i64, size: i64) -> i64 = f(make(size))
out = apply(width, size_from("PATH"))
"#;

/// A chain whose element type is a type argument, so the builder's
/// `Chain[tensor[*, f32]]` claims nothing and only the claim under test walks
/// the value. `DEPTH` links of width 3 hang above an `End` whose width the size
/// file gives: a walk meets the disagreement at the chain's deepest position.
const DEEP_CHAIN: &str = r#"type Chain[t] =
  | End { v: t }
  | Link { v: t, next: Chain[t] }
def grow(size: i64, k: i64) -> Chain[tensor[*, f32]] = fold(fn (acc: Chain[tensor[*, f32]], i: i64) -> Link { v: produce(3i64), next: acc }, End { v: produce(size) }, range(0i64, k))
def head(c: Chain[tensor[*, f32]]) -> i64 =
  match c with {
    | End { v } => shape(v, 0i32)
    | Link { v, next } => shape(v, 0i32)
  }
"#;

/// The entry walk of a claimed formal.
const DEEP_ENTRY: &str = r#"def head3(c: Chain[tensor[3, f32]]) -> i64 = head(c)
out = head3(grow(size_from("PATH"), DEPTH))
"#;

/// The boundary walk of a claimed result whose value arrives from a formal.
const DEEP_RESULT: &str = r#"def keep(c: Chain[tensor[*, f32]]) -> Chain[tensor[3, f32]] = c
out = head(keep(grow(size_from("PATH"), DEPTH)))
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

/// A list combinator produces every tensor its result holds (spec/04 section
/// 4.7), so a nested result claim on that result traps at the combinator.
fn combinator(native: bool) {
    for (case, op) in [
        (MAP, "map"),
        (FLAT_MAP, "flat_map"),
        (SCAN, "scan"),
        (FILTER, "filter"),
        (PARTITION, "partition"),
    ] {
        assert_trap_and_control(
            case,
            native,
            &format!("extent `3`: claimed = 3, {op} axis 0 = 5"),
            op,
            "out = 6",
        );
    }
}

#[test]
fn eval_list_combinator_results_trap_at_the_combinator() {
    combinator(false);
}

#[test]
fn c_list_combinator_results_trap_at_the_combinator() {
    combinator(true);
}

/// An alias is transparent: its nominal dimension argument claims exactly as
/// the expanded spelling does, in a result, a tuple and a formal.
fn alias(native: bool) {
    insert_literal(native, ALIAS_RESULT);
    insert_literal(native, ALIAS_GENERIC);
    insert_literal(native, ALIAS_TUPLE);
    let context = if native {
        "input `b.v` axis 0 expected 3, got 5"
    } else {
        "extent `3`: claimed = 3, b.v axis 0 = 5"
    };
    assert_trap_and_control(ALIAS_FORMAL, native, context, "load", "out = 3");
}

#[test]
fn eval_aliased_nominal_claims_like_its_expansion() {
    alias(false);
}

#[test]
fn c_aliased_nominal_claims_like_its_expansion() {
    alias(true);
}

/// An alias in a type argument, at any depth, claims exactly as its
/// expansion: `Wrap[B3]` is `Wrap[Box[3]]` and `Wrap[V3]` is
/// `Wrap[tensor[3, f32]]`, as results and as formals.
fn alias_argument(native: bool) {
    insert_literal(native, ALIAS_ARGUMENT_BOX);
    insert_literal(native, ALIAS_ARGUMENT_TENSOR);
    for (case, path, control) in [
        (ALIAS_ARGUMENT_FORMAL, "w.x.v", "out = 3"),
        (ALIAS_ARGUMENT_LIST_FORMAL, "ws[1].x", "out = 6"),
    ] {
        let context = if native {
            format!("input `{path}` axis 0 expected 3, got 5")
        } else {
            format!("extent `3`: claimed = 3, {path} axis 0 = 5")
        };
        assert_trap_and_control(case, native, &context, "load", control);
    }
}

#[test]
fn eval_alias_type_argument_claims_like_its_expansion() {
    alias_argument(false);
}

#[test]
fn c_alias_type_argument_claims_like_its_expansion() {
    alias_argument(true);
}

/// The checker admits a recursive alias. One that can hold no tensor owes no
/// claim, so mentioning it in a signature refuses nothing: Eval runs the
/// program, and compiled C stops only at its established entry-layout
/// refusal for such a formal, never at a claim refusal.
#[test]
fn tensor_free_recursive_alias_owes_no_claim() {
    for (case, value) in [
        (RECURSIVE_ALIAS, "out = 1"),
        (RECURSIVE_ALIAS_OPTION, "out = 2"),
    ] {
        let (ok, output) = run(case, 3, false);
        assert!(ok && output.contains(value), "Eval\n{output}");
        let dir = tempfile::tempdir().expect("source dir");
        let path = dir.path().join("completion.ch");
        fs::write(&path, format!("{PRELUDE}{case}")).expect("fixture");
        let built = assert_cmd::Command::cargo_bin("chelis")
            .expect("chelis")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["build", "--emit-c", "--allow-style-violations"])
            .arg(&path)
            .args(["--target", "c", "-o"])
            .arg(dir.path().join("c"))
            .output()
            .expect("build");
        let text = String::from_utf8_lossy(&built.stderr);
        assert!(!text.contains("extent claim"), "compiled C\n{text}");
    }
}

/// A recursive alias whose recursion runs only through built-in containers
/// closes at the alias, so a claim on it derives a finite pattern instead of
/// a refusal: Eval runs the program, and compiled C stops only at its
/// established entry-layout refusal for such a formal. The checker admits no
/// value of this alias other than `None`.
#[test]
fn container_recursive_alias_closes_at_the_alias() {
    let (ok, output) = run(CONTAINER_ALIAS, 3, false);
    assert!(ok && output.contains("out = 1"), "Eval\n{output}");
    let dir = tempfile::tempdir().expect("source dir");
    let path = dir.path().join("completion.ch");
    fs::write(&path, format!("{PRELUDE}{CONTAINER_ALIAS}")).expect("fixture");
    let built = assert_cmd::Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", "--allow-style-violations"])
        .arg(&path)
        .args(["--target", "c", "-o"])
        .arg(dir.path().join("c"))
        .output()
        .expect("build");
    let text = String::from_utf8_lossy(&built.stderr);
    assert!(!text.contains("extent claim"), "compiled C\n{text}");
}

/// Run `case` on one lane under a time bound: `chelis eval`, or the compiled
/// C build. A run that exceeds the bound fails the test.
fn bounded(case: &str, native: bool) -> (bool, String) {
    let dir = tempfile::tempdir().expect("source dir");
    let path = dir.path().join("completion.ch");
    fs::write(&path, format!("{PRELUDE}{case}")).expect("fixture");
    let mut command = assert_cmd::Command::cargo_bin("chelis").expect("chelis");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .timeout(std::time::Duration::from_secs(120));
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
    let output = command.output().expect("bounded run");
    assert!(
        output.status.code().is_some(),
        "{} did not terminate within its bound",
        lane(native)
    );
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// A recursive alias whose arguments grow never closes and is never
/// unfolded. Without a tensor it owes nothing: Eval runs the program, and
/// compiled C stops only at its established entry-layout refusal. With one
/// it is refused by name on both lanes. Every run terminates.
#[test]
fn non_closing_alias_recursion_terminates_on_both_lanes() {
    let (ok, output) = bounded(GROWING_ALIAS, false);
    assert!(ok && output.contains("out = 1"), "Eval\n{output}");
    let (_, output) = bounded(GROWING_ALIAS, true);
    assert!(!output.contains("extent claim"), "compiled C\n{output}");
    for native in [false, true] {
        let (ok, output) = bounded(GROWING_ALIAS_TENSOR, native);
        assert!(
            !ok && output.contains("an extent claim passes through `G`")
                && output.contains("no finite pattern"),
            "{}\n{output}",
            lane(native)
        );
    }
}

/// The host program a claim walk is emitted into is compiled as C++ for the
/// Metal (`.mm`) and HIP (`.cpp`) targets. C++ zeroes no union bytes past the
/// member it initializes, and the runtime rejects a value whose unused
/// payload bytes are not zero, so every borrowed value a walk builds must be
/// zeroed whole. `program` is emitted for the HIP target, and its C++ host is
/// compiled with the platform's C++ compiler, unoptimized (an optimizer can
/// zero the unused bytes by accident and hide a partial value), linked
/// against the runtime archive that build staged, and run.
fn host_as_cxx(program: &str) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cxxhost.ch");
    fs::write(&path, program).expect("fixture");
    let out_dir = dir.path().join("hip");
    let built = assert_cmd::Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", "--allow-style-violations"])
        .arg(&path)
        .args(["--target", "hip", "-o"])
        .arg(&out_dir)
        .output()
        .expect("build");
    let report = format!(
        "{}{}",
        String::from_utf8_lossy(&built.stdout),
        String::from_utf8_lossy(&built.stderr)
    );
    assert!(built.status.success(), "{report}");
    // The exact archive the build staged, as it reports it.
    let archive = report
        .lines()
        .find_map(|line| line.strip_prefix("Staged runtime "))
        .and_then(|rest| rest.split(" (sha256").next())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| panic!("the build reports the runtime it staged:\n{report}"));
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let compiled = std::process::Command::new(&compiler)
        .current_dir(&out_dir)
        .args(["-x", "c++", "-O0", "-c", "cxxhost_hip.cpp", "-o", "host.o"])
        .output()
        .expect("C++ compiler runs");
    assert!(
        compiled.status.success(),
        "the HIP host source compiles as C++:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: false,
            needs_blas: false,
        },
    );
    let linked = std::process::Command::new(&compiler)
        .current_dir(&out_dir)
        .arg("host.o")
        .arg(&archive)
        .args(&toolchain.link_flags)
        .args(["-o", "host"])
        .output()
        .expect("C++ linker runs");
    assert!(
        linked.status.success(),
        "{}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = std::process::Command::new(out_dir.join("host"))
        .output()
        .expect("host runs");
    format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    )
}

/// Agreeing claims on both walk sites: the entry walk of a nominal formal,
/// and the result-value walk of a let-bound claimed aggregate the activation
/// did not construct (`pass3`). Its tensors come from a host producer, so
/// the HIP host holds only code the claims add to an ordinary host program.
const CXX_HOST_CLAIMS: &str = r#"type Box[n] =
  | Box { v: tensor[n, f32] }
def size_from(path: string) -> i64 ! { IO } = string_len(read_file(path))
def fill(size: i64) -> tensor[*, f32] = to_tensor(map(fn (i: i64) -> 0.0f32, range(0i64, size)))
def make[n](size: i64) -> Box[n] = Box { v: fill(size) }
def width[n](b: Box[n]) -> i64 =
  match b with {
    | Box { v } => shape(v, 0i32)
  }
def three(size: i64) -> Box[3] = Box { v: fill(size) }
def claimed(b: Box[3]) -> i64 = width(b)
def pass3(size: i64) -> (Box[3], i64) = {
  b = make(size)
  (b, 1i64)
}
def ident3(size: i64) -> Box[3] = {
  b = make(size)
  b
}
out = add(add(width(three(size_from("PATH"))), claimed(make(size_from("PATH")))), add(width(pass3(size_from("PATH")).0), width(ident3(size_from("PATH")))))
"#;

const KINDED_NOMINAL_DIMENSIONS: &str =
    include_str!("../../../examples/kinded_nominal_dimensions.ch");

/// Agreeing nested result and entry claims, and the shipped example whose
/// generic formal now walks its claim, run unchanged when the host is C++.
#[test]
fn agreeing_claim_walks_run_when_the_host_is_cxx() {
    let input = tempfile::tempdir().expect("size input");
    let size = input.path().join("size.txt");
    fs::write(&size, "xxx").expect("size");
    let program = CXX_HOST_CLAIMS.replace("PATH", &size.display().to_string());
    let output = host_as_cxx(&program);
    assert!(output.contains("out = 12"), "{output}");
    let output = host_as_cxx(KINDED_NOMINAL_DIMENSIONS);
    assert!(output.contains("out = ()"), "{output}");
}

/// Run `command` to completion and return its stdout and peak resident set
/// size, in the platform's `ru_maxrss` unit (only compared as a ratio).
#[expect(
    clippy::zombie_processes,
    reason = "the child is reaped by `wait4`, which also reports its peak memory"
)]
fn peak_resident(mut command: std::process::Command) -> (String, i64) {
    let child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("measured process starts");
    let pid = child.id() as libc::pid_t;
    let mut stdout = String::new();
    std::io::Read::read_to_string(&mut child.stdout.expect("stdout"), &mut stdout).expect("stdout");
    let mut status = 0;
    // SAFETY: `rusage` is plain data the call fills; `pid` is our child and
    // is reaped exactly once, here.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let reaped = unsafe { libc::wait4(pid, &mut status, 0, &mut usage) };
    assert_eq!(reaped, pid, "the measured process is reaped");
    (stdout, usage.ru_maxrss as i64)
}

/// A recursive builder of a claimed nominal holds one copy of its claim, not
/// one per level: its peak memory stays within a small factor of the same
/// builder whose field claims nothing, on compiled C and on Eval. Copying the
/// claim chain at every level made it quadratic in the depth.
fn deep_recursion_peak(native: bool, depth: usize) -> (i64, i64) {
    deep_peak(native, depth, DEEP_CLAIMED, DEEP_UNCLAIMED)
}

fn deep_peak(native: bool, depth: usize, claimed: &str, unclaimed: &str) -> (i64, i64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let size = dir.path().join("depth.txt");
    fs::write(&size, "x".repeat(depth)).expect("depth");
    let mut peaks = Vec::new();
    for (stem, case) in [("claimed", claimed), ("unclaimed", unclaimed)] {
        let path = dir.path().join(format!("{stem}.ch"));
        fs::write(
            &path,
            format!("{PRELUDE}{case}").replace("PATH", &size.display().to_string()),
        )
        .expect("fixture");
        let chelis = assert_cmd::cargo::cargo_bin("chelis");
        let command = if native {
            let out_dir = dir.path().join(stem);
            assert_cmd::Command::new(&chelis)
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["build", "--allow-style-violations"])
                .arg(&path)
                .args(["--target", "c", "-o"])
                .arg(&out_dir)
                .assert()
                .success();
            std::process::Command::new(out_dir.join(stem))
        } else {
            let mut command = std::process::Command::new(&chelis);
            command
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["eval", "--allow-style-violations", "--file"])
                .arg(&path);
            command
        };
        let (stdout, peak) = peak_resident(command);
        assert!(stdout.contains("out = 1"), "{stem}: {stdout}");
        peaks.push(peak);
    }
    (peaks[0], peaks[1])
}

#[test]
fn c_recursive_claimed_builder_memory_stays_linear() {
    let (claimed, unclaimed) = deep_recursion_peak(true, 4000);
    assert!(
        claimed <= 3 * unclaimed,
        "claimed peak {claimed} against unclaimed {unclaimed}"
    );
}

#[test]
fn eval_recursive_claimed_builder_memory_stays_linear() {
    let (claimed, unclaimed) = deep_recursion_peak(false, 2000);
    assert!(
        claimed <= 2 * unclaimed,
        "claimed peak {claimed} against unclaimed {unclaimed}"
    );
}

/// Mutual recursion f, g, f holds each distinct claim once as well: the
/// declared claim keeps its place at the head and only a later copy drops,
/// so memory stays linear on both lanes.
#[test]
fn mutual_recursive_claimed_builder_memory_stays_linear() {
    let (claimed, unclaimed) = deep_peak(true, 4000, DEEP_MUTUAL_CLAIMED, DEEP_MUTUAL_UNCLAIMED);
    assert!(claimed <= 3 * unclaimed, "C: {claimed} against {unclaimed}");
    let (claimed, unclaimed) = deep_peak(false, 2000, DEEP_MUTUAL_CLAIMED, DEEP_MUTUAL_UNCLAIMED);
    assert!(
        claimed <= 2 * unclaimed,
        "Eval: {claimed} against {unclaimed}"
    );
}

/// In mutual recursion f, g, f the innermost activation's claim names the
/// trap, as for the bare tensor: removing a repeated claim must not let an
/// outer, different one name it first.
fn mutual_recursion_attribution(native: bool) {
    assert_trap_and_control(
        MUTUAL_RECURSION,
        native,
        "extent `m`: u axis 0 = 3, insert axis 0 = 5",
        "insert",
        "out = 3",
    );
}

#[test]
fn eval_mutual_recursion_names_the_innermost_claim() {
    mutual_recursion_attribution(false);
}

#[test]
fn c_mutual_recursion_names_the_innermost_claim() {
    mutual_recursion_attribution(true);
}

/// A self-recursive activation with its own witness, 4 wide where its caller
/// had 3, checks its result against its own witness on both lanes.
fn self_recursion_varying_witness(native: bool) {
    for size in [3, 5] {
        let (ok, output) = run(SELF_RECURSION_VARYING_WITNESS, size, native);
        let lane = lane(native);
        assert!(!ok, "{lane}: the inner activation claims 4\n{output}");
        assert!(
            output.lines().any(|line| line.trim_start_matches("error: ")
                == format!("extent `m`: u axis 0 = 4, insert axis 0 = {size}")),
            "{lane}\n{output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in insert at i64"),
            "{lane}\n{output}"
        );
    }
}

#[test]
fn eval_self_recursion_checks_its_own_witness() {
    self_recursion_varying_witness(false);
}

#[test]
fn c_self_recursion_checks_its_own_witness() {
    self_recursion_varying_witness(true);
}

/// The native stack each lane's deep rows run on. A compiled program's main
/// thread gets 512 KiB. `chelis eval` gets 2 MiB, the least its own parser
/// and checker run on; it evaluates inside a grown stack segment, so its
/// claim walks' stack independence is pinned by the evaluator's own unit
/// tests, and these rows pin the walks' cost and path text.
const C_STACK_KIB: u32 = 512;
const EVAL_STACK_KIB: u32 = 2048;

/// The links of a deep chain: chelis#3460's 60,000 on compiled C, and
/// 20,000 on Eval, where a walk that built every position's whole path would
/// write gigabytes of path text.
fn deep_links(native: bool) -> usize {
    if native { 60_000 } else { 20_000 }
}

/// Run `case` after the shared and the deep chain's declarations, with
/// `size` read from a file and the program's native stack limited: the
/// compiled executable on C, the evaluating `chelis` process on Eval.
fn run_deep(case: &str, size: usize, native: bool) -> (bool, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = dir.path().join("size.txt");
    fs::write(&input, "x".repeat(size)).expect("size input");
    let path = dir.path().join("deep.ch");
    let source = format!("{PRELUDE}{DEEP_CHAIN}{case}")
        .replace("PATH", &input.display().to_string())
        .replace("DEPTH", &format!("{}i64", deep_links(native)));
    fs::write(&path, source).expect("fixture");
    let chelis = assert_cmd::cargo::cargo_bin("chelis");
    // The shell lowers its own limit before it becomes the program, so the
    // program's main thread starts on the bounded stack.
    let mut command = std::process::Command::new("/bin/sh");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    if native {
        let out_dir = dir.path().join("deep");
        assert_cmd::Command::new(&chelis)
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["build", "--allow-style-violations"])
            .arg(&path)
            .args(["--target", "c", "-o"])
            .arg(&out_dir)
            .assert()
            .success();
        command
            .args([
                "-c",
                &format!("ulimit -s {C_STACK_KIB} && exec \"$@\""),
                "sh",
            ])
            .arg(out_dir.join("deep"));
    } else {
        command
            .args([
                "-c",
                &format!("ulimit -s {EVAL_STACK_KIB} && exec \"$@\""),
                "sh",
            ])
            .arg(&chelis)
            .args(["eval", "--allow-style-violations", "--file"])
            .arg(&path);
    }
    let output = command.output().expect("the deep program runs");
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// The extent-3 chain runs unchanged; the extent-5 one traps once, with
/// exactly `context` and the Domain line of `op`.
fn assert_deep_trap_and_control(case: &str, native: bool, context: &str, op: &str) {
    let lane = lane(native);
    let (ok, output) = run_deep(case, 3, native);
    assert!(
        ok && output.contains("out = 3"),
        "{lane}: the agreeing chain must run\n{output}"
    );
    let (ok, output) = run_deep(case, 5, native);
    assert!(!ok, "{lane}: the deepest link disagrees\n{output}");
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
        "{lane}\n{output}"
    );
    assert_eq!(
        output.matches("numeric trap:").count(),
        1,
        "{lane}\n{output}"
    );
    assert!(!output.contains("out ="), "{lane}\n{output}");
}

/// A formal's claim on a deep chain is walked to its last link on a bounded
/// native stack, and the trap names the deepest link's path as each lane
/// renders an entry path: its first 497 bytes and a truncation marker.
fn deep_entry_walk(native: bool) {
    let path = format!("c{}.End.v", ".Link.next".repeat(deep_links(native)));
    let shown = format!("{}...(truncated)", &path[..497]);
    let context = if native {
        format!("input `{shown}` axis 0 expected 3, got 5")
    } else {
        format!("extent `3`: claimed = 3, {shown} axis 0 = 5")
    };
    assert_deep_trap_and_control(DEEP_ENTRY, native, &context, "load");
}

#[test]
fn eval_deep_entry_walk_reaches_the_last_link() {
    deep_entry_walk(false);
}

#[test]
fn c_deep_entry_walk_reaches_the_last_link() {
    deep_entry_walk(true);
}

/// A claimed result's boundary walk reaches the last link of a deep chain
/// on a bounded native stack.
fn deep_result_walk(native: bool) {
    assert_deep_trap_and_control(
        DEEP_RESULT,
        native,
        "extent `3`: claimed = 3, load axis 0 = 5",
        "load",
    );
}

#[test]
fn eval_deep_result_walk_reaches_the_last_link() {
    deep_result_walk(false);
}

#[test]
fn c_deep_result_walk_reaches_the_last_link() {
    deep_result_walk(true);
}

/// A literal entry claim at `path`, rendered as each lane renders a bare
/// tensor formal's: Eval as an extent claim, C as an input check.
fn literal_entry_context(native: bool, path: &str) -> String {
    if native {
        format!("input `{path}` axis 0 expected 3, got 5")
    } else {
        format!("extent `3`: claimed = 3, {path} axis 0 = 5")
    }
}

/// A lambda formal's declared nominal type claims its nested tensors on
/// every invocation, as a bare tensor or List lambda formal does: the second
/// fold step receives a 5-wide `Box` under `acc: Box[3]`.
fn lambda_formal_literal(native: bool) {
    assert_trap_and_control(
        LAMBDA_FOLD_BOX,
        native,
        &literal_entry_context(native, "acc.v"),
        "load",
        "out = 3",
    );
}

#[test]
fn eval_lambda_formal_claims_its_nested_tensor() {
    lambda_formal_literal(false);
}

#[test]
fn c_lambda_formal_claims_its_nested_tensor() {
    lambda_formal_literal(true);
}

/// A fold accumulator `acc: Chain[m]` binds `m` at its first link and
/// compares every later link with it, on each step, with one text on both
/// lanes: the second step's chain ends in the 5-wide `End`.
fn lambda_chain_accumulator(native: bool) {
    assert_trap_and_control(
        LAMBDA_FOLD_CHAIN,
        native,
        "extent `m`: acc.Link.v axis 0 = 3, acc.Link.next.End.v axis 0 = 5",
        "load",
        "out = 3",
    );
}

#[test]
fn eval_lambda_chain_accumulator_checks_every_link() {
    lambda_chain_accumulator(false);
}

#[test]
fn c_lambda_chain_accumulator_checks_every_link() {
    lambda_chain_accumulator(true);
}

/// A binder a lambda's nested formal witnesses first is compared with a
/// later tensor formal of the same lambda, with one text on both lanes.
fn lambda_sibling_binder(native: bool) {
    assert_trap_and_control(
        LAMBDA_SIBLING_BINDER,
        native,
        "extent `n`: acc.v axis 0 = 5, t axis 0 = 3",
        "load",
        "out = 3",
    );
}

#[test]
fn eval_lambda_nested_binder_guards_a_later_formal() {
    lambda_sibling_binder(false);
}

#[test]
fn c_lambda_nested_binder_guards_a_later_formal() {
    lambda_sibling_binder(true);
}

/// Every list callback and an immediately applied lambda check a `Box[3]`
/// formal on the 5-wide element.
fn lambda_callbacks(native: bool) {
    for (case, control) in LAMBDA_CALLBACKS.iter().zip([
        "out = 6", "out = 9", "out = 6", "out = 6", "out = 6", "out = 3",
    ]) {
        assert_trap_and_control(
            case,
            native,
            &literal_entry_context(native, "b.v"),
            "load",
            control,
        );
    }
}

#[test]
fn eval_lambda_callbacks_claim_their_nested_formals() {
    lambda_callbacks(false);
}

#[test]
fn c_lambda_callbacks_claim_their_nested_formals() {
    lambda_callbacks(true);
}

/// A function-typed formal's parameter type claims the argument every call
/// through it passes, whatever function the caller supplies.
fn callable_formal(native: bool) {
    assert_trap_and_control(
        CALLABLE_FORMAL,
        native,
        &literal_entry_context(native, "arg0.v"),
        "load",
        "out = 3",
    );
}

#[test]
fn eval_callable_formal_claims_its_nested_parameter() {
    callable_formal(false);
}

#[test]
fn c_callable_formal_claims_its_nested_parameter() {
    callable_formal(true);
}

/// A tuple of tensors in a lambda formal is claimed at its fixed positions,
/// as a named formal's tuple is: no signature entry observes them otherwise.
fn lambda_tuples(native: bool) {
    for (case, control) in LAMBDA_TUPLES.iter().zip(["out = 6", "out = 3"]) {
        assert_trap_and_control(
            case,
            native,
            &literal_entry_context(native, "p.0"),
            "load",
            control,
        );
    }
}

#[test]
fn eval_lambda_tuple_formals_claim_their_positions() {
    lambda_tuples(false);
}

#[test]
fn c_lambda_tuple_formals_claim_their_positions() {
    lambda_tuples(true);
}

/// DISPOSITION LOCK pending chelis#3517, not a regression test of correct
/// behaviour. A lambda formal's binder is witnessed by the lambda's own
/// formals at each invocation, never by the enclosing declaration's formal
/// that the checker resolves it to; both lanes agree, for bare, List and
/// nested lambda formals alike. `w` is 4 wide and `acc` is 3 or 5 wide, and
/// neither lane traps. When #3517 binds the lambda's `n` to `w`, both runs
/// trap and this lock is replaced by the regression test.
fn lambda_outer_binder_lock(native: bool) {
    for size in [3, 5] {
        let (ok, output) = run(LAMBDA_OUTER_BINDER, size, native);
        assert!(ok, "{}\n{output}", lane(native));
        assert!(
            output.contains(&format!("out = {size}")),
            "{}\n{output}",
            lane(native)
        );
    }
}

#[test]
fn eval_lambda_outer_binder_disposition_lock() {
    lambda_outer_binder_lock(false);
}

#[test]
fn c_lambda_outer_binder_disposition_lock() {
    lambda_outer_binder_lock(true);
}

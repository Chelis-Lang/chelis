//! chelis#1158 (successor to chelis#941): a recursive generic host call
//! lowers to a bounded, memoized monomorphic C symbol.
//!
//! Before this change a type-polymorphic function that participated in a
//! call cycle had no standalone C ABI (`lower_host_program` skips generic
//! defs) and could not be inlined either (the inline guard refuses re-entry),
//! so every such call site was rejected under `[05-UNS-1]`. That rejection
//! blocked the whole `Coral.Frame` API in the build lane
//! (Chelis-Lang/coral#26), whose `Hamt.from_pairs_rec` is exactly this shape.
//!
//! The positive coverage below locks the outlined-specialization contract:
//! direct recursion, mutual recursion, one generic used at two payload types
//! (with a symbol-count assertion so a duplicate-symbol regression is loud),
//! and the in-repo mini-Hamt that reproduces the coral shape — a recursive
//! trie generic in `a` instantiated at a payload carrying `tensor[n, f32]`.
//! Each runnable case asserts eval/C value parity, because a specialization
//! that compiles but computes a different value is the failure mode a
//! compile-only assertion would miss.
//!
//! The negative coverage keeps the narrowed `[05-UNS-1]` boundary honest:
//! polymorphic recursion mints a fresh type at every level and therefore has
//! no finite monomorphization, so it must fail loudly and promptly rather
//! than hang or exhaust memory.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command as StdCommand;
use std::time::Duration;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// chelis#941's own reproducer, verbatim in shape: `loop` recurses on
/// `Box[a]` and the entry point instantiates it at `f32`.
const DIRECT_RECURSION: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def loop[a](box: Box[a]) -> bool =
  match box with {
    | Empty => true
    | Full { value: item } => loop(Empty)
  }
out = print(loop(Full { value: cast(1.0, f32) }))
";

/// A generic call cycle spanning two defs. Memoized interning resolves this
/// without any SCC analysis: each distinct `(callee, signature)` pair is
/// interned once and the second visit hits the memo.
const MUTUAL_RECURSION: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def ping[a](box: Box[a], again: bool) -> bool = if again then pong(box, false) else true
def pong[a](box: Box[a], again: bool) -> bool = if again then ping(box, false) else true
out = print(ping(Full { value: cast(1.0, f32) }, true))
";

/// One recursive generic used at two payload types in the same program.
const MULTI_INSTANTIATION: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def count[a](box: Box[a], acc: int64) -> int64 =
  match box with {
    | Empty => acc
    | Full { value: item } => count(Empty, add(acc, cast(1, int64)))
  }
def both() -> int64 = add(count(Full { value: cast(1.0, f32) }, cast(0, int64)), count(Full { value: \"x\" }, cast(0, int64)))
out = print(both())
";

/// Polymorphic recursion: `poly[a]` calls itself at `(a, a)`, so the
/// instantiation family is infinite. There is no finite set of monomorphic
/// symbols; the boundary must stay.
const POLYMORPHIC_RECURSION: &str = "\
def poly[a](x: a, depth: int64) -> int64 = if lt(depth, cast(1, int64)) then cast(0, int64) else poly((x, x), sub(depth, cast(1, int64)))
out = print(poly(cast(1.0, f32), cast(3, int64)))
";

/// The other residue of the narrowed boundary: the entry point instantiates
/// `loop` at a nullary constructor that nothing constrains, so the call has
/// no concrete signature to intern. `eval` runs this (its values are not
/// typed at an ABI); the build lane must not invent one.
const NON_CONCRETIZABLE_CALL: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def loop[a](box: Box[a]) -> bool =
  match box with {
    | Empty => true
    | Full { value: item } => loop(Empty)
  }
out = print(loop(Empty))
";

/// The coral#26 front door without vendoring coral: a recursive trie generic
/// in `a` (`insert`/`from_pairs_rec`/`find`) instantiated at `Column[2]`,
/// whose payload is `tensor[2, f32]`. `from_pairs_rec` is the exact
/// `Coral.Internal.Hamt.from_pairs_rec` shape that `Coral.Frame` builds on.
const MINI_HAMT: &str = "\
type Column[n] =
  | FloatCol(tensor[n, f32])
type Hamt[a] =
  | Leaf
  | Node { key: int64, value: a, left: Hamt[a], right: Hamt[a] }
def insert[a](tree: Hamt[a], key: int64, value: a) -> Hamt[a] =
  match tree with {
    | Leaf => Node { key, value, left: Leaf, right: Leaf }
    | Node { key: k, value: v, left, right } => if lt(key, k) then Node { key: k, value: v, left: insert(left, key, value), right } else Node { key: k, value: v, left, right: insert(right, key, value) }
  }
def from_pairs_rec[a](tree: Hamt[a], keys: List[int64], values: List[a], at: int64) -> Hamt[a] = if lt(at, cast(0, int64)) then tree else from_pairs_rec(insert(tree, index(keys, at), index(values, at)), keys, values, sub(at, cast(1, int64)))
def find[a](tree: Hamt[a], key: int64) -> Option[a] =
  match tree with {
    | Leaf => None
    | Node { key: k, value: v, left, right } => if eq(key, k) then Some(v) else if lt(key, k) then find(left, key) else find(right, key)
  }
def column_sum(column: Column[2]) -> f32 =
  match column with {
    | FloatCol(values) => fold(fn (acc: f32, value: f32) -> add(acc, value), cast(0.0, f32), to_list(values))
  }
def built() -> Hamt[Column[2]] = from_pairs_rec(Leaf, [cast(1, int64), cast(2, int64)], [FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])), FloatCol(to_tensor([cast(10.0, f32), cast(20.0, f32)]))], cast(1, int64))
def total() -> f32 =
  match find(built(), cast(2, int64)) with {
    | Some(column) => column_sum(column)
    | None => cast(0.0, f32)
  }
out = print(total())
";

/// The adversarial-review trigger for emitted-C nondeterminism.
///
/// `driver` is lowered FIRST and its body calls both wrappers, so the
/// summary-rejection probe runs over a two-element `HashSet` before either
/// wrapper is really lowered. Each probe fully lowers a wrapper, and each
/// wrapper interns a different `count` specialization, so pre-fix whichever
/// probe ran first won the FIFO slot and the two specializations swapped
/// places between runs. Measured on the unfixed tree: 25 builds produced two
/// distinct `.c` hashes, 11 and 14.
///
/// The source ORDER is the trigger and must not be "tidied": moving `driver`
/// below the wrappers makes them intern in deterministic source order and the
/// probe becomes a memo hit, which is exactly why the original determinism
/// test missed this.
const PROBE_ORDER_DETERMINISM: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def count[a](box: Box[a], acc: int64) -> int64 =
  match box with {
    | Empty => acc
    | Full { value: item } => count(Empty, add(acc, cast(1, int64)))
  }
def driver(t: tensor[2, f32]) -> tensor[2, f32] = mul(wa(copy(t)), wb(t))
def wa(t: tensor[2, f32]) -> tensor[2, f32] = if eq(count(Full { value: cast(1.0, f32) }, cast(0, int64)), cast(1, int64)) then t else mul(t, t)
def wb(t: tensor[2, f32]) -> tensor[2, f32] = if eq(count(Full { value: \"x\" }, cast(0, int64)), cast(1, int64)) then t else mul(t, t)
r = driver(to_tensor([cast(1.0, f32), cast(2.0, f32)]))
";

/// A recursive generic mixing a type variable with a SYMBOLIC tensor dim.
///
/// Monomorphization substitutes type variables, not dimension names, so the
/// specialization interns with the declared `n` while the caller passes
/// `tensor[2, f32]`. Pre-fix that pair reached C ABI projection and aborted the
/// build with an internal compiler error `[04-TOT-2]`; it must be a clean
/// `[05-UNS-1]` call-site rejection instead.
const MIXED_TYPEVAR_AND_SYMBOLIC_DIM: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def mixed[a](box: Box[a], t: tensor[n, f32], c: int64) -> bool = if lt(c, cast(1, int64)) then true else mixed(box, t, sub(c, cast(1, int64)))
out = print(mixed(Full { value: cast(1, int64) }, to_tensor([cast(1.0, f32), cast(2.0, f32)]), cast(2, int64)))
";

/// The same signature with a LITERAL dim. Dims agree with the caller, so this
/// is a faithful monomorphization and must keep building — the control that
/// stops the fix above from degenerating into "reject anything with a tensor".
const MIXED_TYPEVAR_AND_LITERAL_DIM: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def mixed[a](box: Box[a], t: tensor[2, f32], c: int64) -> bool = if lt(c, cast(1, int64)) then true else mixed(box, t, sub(c, cast(1, int64)))
out = print(mixed(Full { value: cast(1, int64) }, to_tensor([cast(1.0, f32), cast(2.0, f32)]), cast(2, int64)))
";

/// A generic whose specialization has a pure-tensor signature, so it competes
/// with the authored entry. `run` is the author's entry; `spin[a]` at
/// `a := tensor[2, f32]` produces `spin__mono_*(chelis_tensor*) ->
/// chelis_tensor*`, which the drain appends AFTER `run`. Entry selection takes
/// the LAST pure-tensor function, so pre-fix the specialization displaced
/// `run` and the build silently fell back to the whole-program kernel — the
/// HIP entry went from 1-in/1-out to 1-in/4-out.
const ENTRY_DISPLACEMENT: &str = "\
def stop() -> bool = true
def spin[a](x: a, n: int64) -> a = if stop() then x else spin(x, n)
def run(flag: bool) -> tensor[2, f32] = spin(to_tensor([cast(1.0, f32), cast(2.0, f32)]), cast(0, int64))
def scale(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)
";

/// Build `source` to C under a fresh temp dir. Returns the temp dir (kept
/// alive by the caller) and the output directory.
fn build(source: &str, stem: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join("out");
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    (dir, out_dir)
}

fn eval_first_line(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let stdout = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8_lossy(&stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

fn run_generated(out_dir: &Path, stem: &str) -> String {
    let source_file = format!("{stem}.c");
    let linked = common::link_generated(out_dir, &source_file, "run");
    assert!(linked.success(), "generated C must link: {linked}");
    let output = StdCommand::new(out_dir.join("run"))
        .output()
        .expect("run generated binary");
    assert!(
        output.status.success(),
        "generated binary failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Build + link + run `source`, and assert the compiled first output line
/// equals `chelis eval`'s. Value parity is the point: an outlined
/// specialization that compiles but computes something else is exactly the
/// regression a compile-only assertion cannot see.
fn assert_build_lane_matches_eval(source: &str, stem: &str, expected: &str) {
    let eval = eval_first_line(source, stem);
    assert_eq!(eval, expected, "eval lane value");
    let (_dir, out_dir) = build(source, stem);
    let compiled = run_generated(&out_dir, stem);
    assert_eq!(compiled, eval, "eval/C value parity for `{stem}`");
}

/// Every distinct monomorphized symbol for `callee` that appears in the
/// emitted C. Scans for the `<callee>__mono_` marker and takes the run of
/// identifier characters that follows, so a second specialization at a
/// different payload type is a second set member rather than a second
/// occurrence of the same one.
fn mono_symbols(c_source: &str, callee: &str) -> BTreeSet<String> {
    let marker = format!("{callee}__mono_");
    let mut out = BTreeSet::new();
    let bytes = c_source.as_bytes();
    let mut search = 0usize;
    while let Some(offset) = c_source[search..].find(&marker) {
        let start = search + offset;
        let mut end = start + marker.len();
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        out.insert(c_source[start..end].to_string());
        search = start + marker.len();
    }
    out
}

#[test]
fn direct_recursive_generic_builds_and_runs() {
    assert_build_lane_matches_eval(DIRECT_RECURSION, "direct_recursion", "true");
}

#[test]
fn mutually_recursive_generics_build_and_run() {
    assert_build_lane_matches_eval(MUTUAL_RECURSION, "mutual_recursion", "true");
}

#[test]
fn multi_instantiation_emits_one_symbol_per_payload_type() {
    let stem = "multi_instantiation";
    assert_build_lane_matches_eval(MULTI_INSTANTIATION, stem, "2");
    let (_dir, out_dir) = build(MULTI_INSTANTIATION, stem);
    let emitted = std::fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("emitted C");
    let symbols = mono_symbols(&emitted, "count");
    assert_eq!(
        symbols.len(),
        2,
        "one specialization per payload type, no duplicates; found {symbols:?}"
    );
}

#[test]
fn polymorphic_recursion_stays_on_the_unsupported_boundary() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("polymorphic_recursion.ch");
    write_file(&path, POLYMORPHIC_RECURSION);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .timeout(Duration::from_secs(120))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("[05-UNS-1]"))
        .stderr(predicates::str::contains("chelis#1158"))
        .stderr(predicates::str::contains("polymorphic recursion"));
}

#[test]
fn non_concretizable_generic_call_reports_the_other_cause() {
    // The narrowed boundary has exactly two residues, and the message must
    // say which one fired: a reader who sees "polymorphic recursion" for an
    // under-constrained call site will go looking for a recursion problem
    // that is not there.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("non_concretizable.ch");
    write_file(&path, NON_CONCRETIZABLE_CALL);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .timeout(Duration::from_secs(120))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("[05-UNS-1]"))
        .stderr(predicates::str::contains("chelis#1158"))
        .stderr(predicates::str::contains("not fully concrete"))
        .stderr(predicates::str::contains("still carries a free variable"))
        .stderr(predicates::str::contains("polymorphic recursion").not());
}

#[test]
fn mini_hamt_reproduces_the_coral_frame_shape() {
    assert_build_lane_matches_eval(MINI_HAMT, "mini_hamt", "30.0");
}

#[test]
fn probe_order_does_not_reach_the_emitted_c() {
    // chelis#1002 determinism, at the trigger the original test lacked. Eight
    // runs is enough: the unfixed tree split 25 runs 11/14, so the chance of
    // eight agreeing by luck is under 1%.
    let stem = "probe_order_determinism";
    let mut digests = BTreeSet::new();
    for _ in 0..8 {
        let (_dir, out_dir) = build(PROBE_ORDER_DETERMINISM, stem);
        let emitted = std::fs::read(out_dir.join(format!("{stem}.c"))).expect("emitted C");
        digests.insert(emitted);
    }
    assert_eq!(
        digests.len(),
        1,
        "probe iteration order must not reach the emitted C; got {} distinct outputs",
        digests.len()
    );
    // Guard the precondition: if the program ever stops minting two
    // specializations, this test still passes but no longer tests anything.
    let (_dir, out_dir) = build(PROBE_ORDER_DETERMINISM, stem);
    let emitted = std::fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("emitted C");
    assert_eq!(
        mono_symbols(&emitted, "count").len(),
        2,
        "trigger precondition: two specializations must be in flight"
    );
}

#[test]
fn mixed_type_var_and_symbolic_dim_rejects_instead_of_ice() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mixed_symbolic_dim.ch");
    write_file(&path, MIXED_TYPEVAR_AND_SYMBOLIC_DIM);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .assert()
        .failure()
        // The clean call-site boundary, naming the disagreement...
        .stderr(predicates::str::contains("[05-UNS-1]"))
        .stderr(predicates::str::contains("chelis#1158"))
        .stderr(predicates::str::contains("not tensor dimension names"))
        // ...and NOT the internal compiler error it used to be.
        .stderr(predicates::str::contains("[04-TOT-2]").not())
        .stderr(predicates::str::contains("internal compiler error").not());
}

#[test]
fn mixed_type_var_and_literal_dim_still_builds() {
    assert_build_lane_matches_eval(MIXED_TYPEVAR_AND_LITERAL_DIM, "mixed_literal_dim", "true");
}

#[test]
fn a_specialization_never_displaces_the_authored_entry() {
    // The symptom is in the entry ABI, which the HIP target makes visible as
    // an explicit arity check in the emitted entry.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("entry_displacement.ch");
    let out_dir = dir.path().join("out");
    write_file(&path, ENTRY_DISPLACEMENT);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let emitted = std::fs::read_to_string(out_dir.join("entry_displacement_hip.cpp"))
        .expect("emitted HIP source");
    assert!(
        emitted.contains("expected %d outputs, got %d\\n\", 1,"),
        "the authored single-output entry must survive; a displaced entry falls \
         back to the 4-output whole-program kernel:\n{}",
        emitted
            .lines()
            .filter(|line| line.contains("expected %d"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    // The compiler-owned specialization must not appear in the published
    // header either.
    let header = std::fs::read_to_string(out_dir.join("entry_displacement_hip.h"))
        .expect("emitted HIP header");
    assert!(
        !header.contains("__mono_"),
        "published header must not declare specializations:\n{header}"
    );
}

#[test]
fn lowering_the_same_program_twice_is_byte_identical() {
    // chelis#1002 made emitted-C determinism a repo-level sensitivity: a
    // specialization worklist drained in hash-map order would reorder
    // functions between runs and silently break reproducible builds.
    let stem = "mini_hamt";
    let (_first_dir, first_out) = build(MINI_HAMT, stem);
    let (_second_dir, second_out) = build(MINI_HAMT, stem);
    let first = std::fs::read(first_out.join(format!("{stem}.c"))).expect("first emitted C");
    let second = std::fs::read(second_out.join(format!("{stem}.c"))).expect("second emitted C");
    assert_eq!(
        first, second,
        "repeated lowering of the same program must be byte-identical"
    );
}

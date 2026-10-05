//! chelis#2205: a dictionary whose last use is `dict_insert`, `dict_merge` or
//! `dict_remove` moves into the builtin instead of being cloned, and never
//! changes what a program means.
//!
//! The same two halves as `issue_2205_container_last_use.rs`, which covers
//! the list rows.
//!
//! The alias controls are a sample, not a quantifier. Each pins one shape in
//! which the rewritten dictionary stays observable somewhere else, and pins
//! that this program keeps its value on both lanes and does not abort. Ten
//! shapes: a shared seed grown twice through recursion, a tuple-held alias,
//! a list holding the dictionary twice, an `Option`-held source read after
//! the insert, a branch, an `Option`-held source read after a removal, a
//! `dict_merge` whose right-hand side aliases the consumed left-hand side
//! through a branch, the same through a match arm, the same through two
//! list indices, and a string-keyed program that runs the release paths on
//! refcounted handles. They are the shapes on which a verified Move meets a
//! strong-owner count above one, which is why the consuming entry points
//! check uniqueness at run time. Nothing here proves the property for a
//! shape not in that list; the run-time check is what covers the rest, and
//! `dict_owned_mutators.rs` is where its arms are pinned directly.
//!
//! The counted receipts are the number of cloning `chelis_dict_insert(`,
//! `chelis_dict_merge(` and `chelis_dict_remove(` calls the C lane emits for
//! an N-step chain, asserted as a ratio so they carry no machine budget.
//!
//! Evidentiary status: REGRESSION TESTS for the receipts, proven failing
//! first on the same emitter without the three `CONTAINER_CONSUMERS` rows.
//! The value-parity rows are LOCKS: they pass without the rows too, which is
//! the point.
#![allow(clippy::uninlined_format_args)]
use assert_cmd::Command;
use tempfile::tempdir;

mod common;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn write_file(path: &std::path::Path, contents: &str) {
    std::fs::write(path, contents).expect("write fixture");
}

fn eval_main(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find(|line| line.starts_with("main = "))
        .unwrap_or_else(|| panic!("no `main = ` line in:\n{stdout}"))
        .trim()
        .to_string()
}

/// Build through the C lane; returns the generated C and the binary's
/// `main = ` line. A runtime abort (the in-place guard firing on a shared
/// dictionary) surfaces as a non-zero exit and fails here.
fn c_main(program: &str, name: &str) -> (String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    assert!(
        built.status.success(),
        "build failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let generated =
        std::fs::read_to_string(out_dir.join(format!("{name}.c"))).expect("read generated C");
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled `{name}` exited with {}; a fired exclusivity guard is a bug, not a lock:\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    let line = stdout
        .lines()
        .find(|line| line.starts_with("main = "))
        .unwrap_or_else(|| panic!("no `main = ` line in:\n{stdout}"))
        .trim()
        .to_string();
    (generated, line)
}

const EMPTY: &str = "dict_of([] : List[(i64, i64)])";

const SHARED_SEED: &str = "module Dc.Main\n\
export (main)\n\
def grow(seed: Dict[i64, i64], n: i64) -> Dict[i64, i64] = if lte(n, cast(0, i64)) then seed else grow(dict_insert(seed, n, n), sub(n, cast(1, i64)))\n\
def main() -> i64 = {\n\
  seed = dict_insert(dict_of([] : List[(i64, i64)]), cast(99, i64), cast(7, i64))\n\
  a = grow(seed, cast(3, i64))\n\
  b = grow(seed, cast(2, i64))\n\
  add(add(mul(len(a), cast(100, i64)), mul(len(b), cast(10, i64))), len(seed))\n\
}\n";

const TUPLE_HELD: &str = "module Dc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64))\n\
  held = (d, cast(9, i64))\n\
  grown = dict_insert(d, cast(2, i64), cast(20, i64))\n\
  add(mul(len(grown), cast(10, i64)), len(held.0))\n\
}\n";

const LIST_HELD: &str = "module Dc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64))\n\
  outer = [d, d]\n\
  grown = dict_insert(d, cast(2, i64), cast(20, i64))\n\
  add(mul(len(grown), cast(100, i64)), add(mul(len(index(outer, cast(0, i64))), cast(10, i64)), len(index(outer, cast(1, i64)))))\n\
}\n";

const OBSERVABLE_SOURCE: &str = "module Dc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  source = dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64))\n\
  keep = Some(source)\n\
  grown = dict_insert(source, cast(2, i64), cast(20, i64))\n\
  match keep with {\n\
    | Some(original) => add(mul(len(grown), cast(10, i64)), len(original))\n\
    | None => cast(0, i64)\n\
  }\n\
}\n";

const BRANCHED_ALIAS: &str = "module Dc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64))\n\
  held = (d, cast(0, i64))\n\
  grown = if gt(len(d), cast(0, i64)) then dict_insert(d, cast(2, i64), cast(20, i64)) else d\n\
  add(mul(len(grown), cast(10, i64)), len(held.0))\n\
}\n";

const REMOVE_HELD: &str = "module Dc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64)), cast(2, i64), cast(20, i64))\n\
  keep = Some(d)\n\
  shrunk = dict_remove(d, cast(1, i64))\n\
  match keep with {\n\
    | Some(original) => add(mul(len(shrunk), cast(10, i64)), len(original))\n\
    | None => cast(0, i64)\n\
  }\n\
}\n";

/// String keys and string values, so the compiled lane runs the consuming
/// entries' release paths on refcounted handles rather than on scalars. The
/// integer fixtures above never execute one line of that work.
const STRING_KEYED: &str = "module Dc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_of([] : List[(string, string)]), \"alpha\", \"one\")\n\
  held = Some(d)\n\
  grown = dict_insert(d, \"beta\", \"two\")\n\
  replaced = dict_insert(grown, \"alpha\", \"three\")\n\
  shrunk = dict_remove(replaced, \"beta\")\n\
  match held with {\n\
    | Some(original) => add(mul(len(shrunk), cast(10, i64)), len(original))\n\
  | None => cast(0, i64)\n\
  }\n\
}\n";

/// A `dict_merge` whose right-hand side aliases the consumed left-hand side
/// through a branch: two operand identities, one runtime dictionary. The
/// owned entry point must clone, never abort.
const BRANCH_MERGE: &str = "module Dm.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64))\n\
  other = if gt(len(d), cast(0, i64)) then d else dict_of([] : List[(i64, i64)])\n\
  merged = dict_merge(d, other)\n\
  len(merged)\n\
}\n";

const MATCH_ARM_MERGE: &str = "module Dm.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64))\n\
  held = Some(d)\n\
  out = match held with {\n\
    | Some(kept) => dict_merge(d, kept)\n\
    | None => d\n\
  }\n\
  len(out)\n\
}\n";

const LIST_INDEX_MERGE: &str = "module Dm.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  d = dict_insert(dict_of([] : List[(i64, i64)]), cast(1, i64), cast(10, i64))\n\
  outer = [d, d]\n\
  a = index(outer, cast(0, i64))\n\
  b = index(outer, cast(1, i64))\n\
  merged = dict_merge(a, b)\n\
  len(merged)\n\
}\n";

/// An N-step let-bound `dict_insert` chain: every step's input is dead after
/// it, and every key is distinct so the dictionary grows by one per step.
fn insert_chain(steps: usize) -> String {
    let mut source =
        format!("module Dc.Main\nexport (main)\ndef main() -> i64 = {{\n  d0 = {EMPTY}\n");
    for step in 1..=steps {
        source.push_str(&format!(
            "  d{step} = dict_insert(d{}, cast({step}, i64), cast({step}, i64))\n",
            step - 1
        ));
    }
    source.push_str(&format!("  len(d{steps})\n}}\n"));
    source
}

/// An N-step let-bound `dict_merge` chain: each step folds a fresh
/// single-entry dictionary into the accumulator, which is dead afterwards.
fn merge_chain(steps: usize) -> String {
    let mut source =
        format!("module Dm.Main\nexport (main)\ndef main() -> i64 = {{\n  d0 = {EMPTY}\n");
    for step in 1..=steps {
        source.push_str(&format!(
            "  d{step} = dict_merge(d{}, dict_insert({EMPTY}, cast({step}, i64), cast({step}, i64)))\n",
            step - 1
        ));
    }
    source.push_str(&format!("  len(d{steps})\n}}\n"));
    source
}

/// An N-step let-bound `dict_remove` chain over an N-entry dictionary.
fn remove_chain(steps: usize) -> String {
    let mut source =
        format!("module Dr.Main\nexport (main)\ndef main() -> i64 = {{\n  s0 = {EMPTY}\n");
    for step in 1..=steps {
        source.push_str(&format!(
            "  s{step} = dict_insert(s{}, cast({step}, i64), cast({step}, i64))\n",
            step - 1
        ));
    }
    source.push_str(&format!("  d0 = s{steps}\n"));
    for step in 1..=steps {
        source.push_str(&format!(
            "  d{step} = dict_remove(d{}, cast({step}, i64))\n",
            step - 1
        ));
    }
    source.push_str(&format!("  len(d{steps})\n}}\n"));
    source
}

#[test]
fn dict_alias_controls_keep_their_values_on_both_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    for (name, program, expected) in [
        ("dict_shared_seed", SHARED_SEED, "main = 431"),
        ("dict_tuple_held", TUPLE_HELD, "main = 21"),
        ("dict_list_held", LIST_HELD, "main = 211"),
        ("dict_observable_source", OBSERVABLE_SOURCE, "main = 21"),
        ("dict_branched_alias", BRANCHED_ALIAS, "main = 21"),
        ("dict_remove_held", REMOVE_HELD, "main = 12"),
        ("dict_branch_merge", BRANCH_MERGE, "main = 1"),
        ("dict_match_arm_merge", MATCH_ARM_MERGE, "main = 1"),
        ("dict_list_index_merge", LIST_INDEX_MERGE, "main = 1"),
        ("dict_string_keyed", STRING_KEYED, "main = 11"),
    ] {
        let evaluated = eval_main(program);
        assert_eq!(evaluated, expected, "{name}: eval value");
        let (_, compiled) = c_main(program, name);
        assert_eq!(compiled, expected, "{name}: compiled value");
    }
}

#[test]
fn let_bound_dict_chains_do_not_clone_per_step() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    fn cloning_calls(source: &str, entry: &str, name: &str, expected_len: usize) -> usize {
        let (generated, value) = c_main(source, name);
        assert_eq!(value, format!("main = {expected_len}"), "{name}");
        generated.matches(&format!(" = {entry}(")).count()
    }
    fn receipt(
        prefix: &str,
        entry: &str,
        build: impl Fn(usize) -> String,
        expected_len: impl Fn(usize) -> usize,
    ) {
        let small = cloning_calls(&build(8), entry, &format!("{prefix}8"), expected_len(8));
        let large = cloning_calls(&build(16), entry, &format!("{prefix}16"), expected_len(16));
        eprintln!(
            "#2205 dict receipt: 8-step {prefix} chain emits {small} cloning `{entry}` calls, \
             16-step chain emits {large}"
        );
        assert!(
            large <= small,
            "#2205: cloning `{entry}` calls must not grow with the chain; an 8-step chain \
             emitted {small} and a 16-step chain emitted {large}"
        );
        assert_eq!(
            (small, large),
            (0, 0),
            "#2205: a let-bound {prefix} chain never clones; the exact count is a count of \
             emitted calls and carries no machine budget"
        );
    }

    receipt("dictins", "chelis_dict_insert", insert_chain, |steps| steps);
    receipt("dictmrg", "chelis_dict_merge", merge_chain, |steps| steps);
    receipt("dictrem", "chelis_dict_remove", remove_chain, |_| 0);
}

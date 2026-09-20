//! chelis#2205: a string whose last use is `string_concat` moves into the
//! builtin instead of being cloned, and never changes what a program means.
//!
//! The third and last kind, after the list rows in
//! `issue_2205_container_last_use.rs` and the dictionary rows in
//! `issue_2205_dict_last_use.rs`.
//!
//! The alias controls are a sample, not a quantifier. Each pins one shape in
//! which the extended string stays observable somewhere else, and pins that
//! this program keeps its value on both lanes and does not abort. The five
//! shapes chelis#943 named come first: a shared seed grown twice through
//! recursion, a branched alias, a nested holder, an empty seed grown twice,
//! and an observable source read after the concat. Three more follow: a
//! tuple-held alias, a `string_concat` whose right-hand side aliases the
//! consumed left-hand side through a match arm, and a multibyte program whose
//! character-indexed length and slicing would both go wrong if the in-place
//! arm left `char_count` stale. Nothing here proves the property for a shape
//! outside that list; the run-time uniqueness check covers the rest, and
//! `string_owned_mutators.rs` is where its arms are pinned directly.
//!
//! The counted receipt is the number of cloning `chelis_string_concat(` calls
//! the C lane emits for an N-step chain, asserted as a ratio so it carries no
//! machine budget.
//!
//! Evidentiary status: REGRESSION TEST for the receipt, proven failing first
//! on the same emitter without the `builtin:string_concat` row, where an
//! 8-step chain emits 8 cloning calls and a 16-step chain 16. The
//! value-parity rows are LOCKS: they pass without the row too, which is the
//! point.
//!
//! These tests link a runtime built without the `ownership-ledger` feature,
//! so they prove printed values, cross-lane agreement, the absence of an
//! abort, and which entry point the emitter **selected**. They do not prove
//! which arm that entry point then **took**: it re-checks the strong-owner
//! count and clones silently above one, so a receipt of zero cloning call
//! sites is consistent with every step still copying. Nor do they prove
//! ownership balance. Both need a ledger-enabled runtime, which chelis#2252
//! owns giving a committed home.
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
/// `main = ` line. A runtime abort surfaces as a non-zero exit and fails
/// here.
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

/// chelis#943 shape one: one seed grown twice through recursion, so the seed
/// is retained across both calls.
const SHARED_SEED: &str = "module Sc.Main\n\
export (main)\n\
def grow(seed: string, n: i64) -> string = if lte(n, cast(0, i64)) then seed else grow(string_concat(seed, \"x\"), sub(n, cast(1, i64)))\n\
def main() -> i64 = {\n\
  seed = string_concat(\"s\", \"\")\n\
  a = grow(seed, cast(3, i64))\n\
  b = grow(seed, cast(2, i64))\n\
  add(add(mul(string_len(a), cast(100, i64)), mul(string_len(b), cast(10, i64))), string_len(seed))\n\
}\n";

/// chelis#943 shape two: a branch, where only one arm concatenates.
const BRANCHED_ALIAS: &str = "module Sc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  s = string_concat(\"ab\", \"\")\n\
  held = (s, cast(0, i64))\n\
  grown = if gt(string_len(s), cast(1, i64)) then string_concat(s, \"c\") else s\n\
  add(mul(string_len(grown), cast(10, i64)), string_len(held.0))\n\
}\n";

/// chelis#943 shape three: a nested holder, here a list holding the string
/// twice while it is concatenated.
const NESTED_HOLDER: &str = "module Sc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  s = string_concat(\"ab\", \"\")\n\
  outer = [s, s]\n\
  grown = string_concat(s, \"c\")\n\
  add(mul(string_len(grown), cast(100, i64)), add(mul(string_len(index(outer, cast(0, i64))), cast(10, i64)), string_len(index(outer, cast(1, i64)))))\n\
}\n";

/// chelis#943 shape four: an empty seed grown twice.
const EMPTY_SEED: &str = "module Sc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  empty = \"\"\n\
  held = (empty, cast(0, i64))\n\
  one = string_concat(empty, \"a\")\n\
  two = string_concat(one, \"b\")\n\
  add(mul(string_len(two), cast(10, i64)), string_len(held.0))\n\
}\n";

/// chelis#943 shape five: a source read after the concat, through an option.
const OBSERVABLE_SOURCE: &str = "module Sc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  source = string_concat(\"ab\", \"\")\n\
  keep = Some(source)\n\
  grown = string_concat(source, \"c\")\n\
  match keep with {\n\
    | Some(original) => add(mul(string_len(grown), cast(10, i64)), string_len(original))\n\
    | None => cast(0, i64)\n\
  }\n\
}\n";

const TUPLE_HELD: &str = "module Sc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  s = string_concat(\"ab\", \"\")\n\
  held = (s, cast(9, i64))\n\
  grown = string_concat(s, \"cd\")\n\
  add(string_len(grown), string_len(held.0))\n\
}\n";

/// A right-hand side that aliases the consumed left-hand side: two operand
/// identities, one runtime handle. The owned entry point must clone.
const MATCH_ARM_SELF: &str = "module Sc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  s = string_concat(\"ab\", \"\")\n\
  held = Some(s)\n\
  out = match held with {\n\
    | Some(kept) => string_concat(s, kept)\n\
    | None => s\n\
  }\n\
  string_len(out)\n\
}\n";

/// The in-place arm maintains `char_count`, which the character-indexed
/// `string_len` reads and which decides `string_slice`'s strategy. A stale
/// count changes this program's answer on the C lane only.
const MULTIBYTE: &str = "module Sc.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  a = string_concat(\"héllo\", \" wörld\")\n\
  piece = string_slice(a, cast(6, i64), cast(5, i64))\n\
  add(mul(string_len(a), cast(10, i64)), string_len(piece))\n\
}\n";

/// An N-step let-bound concat chain: every step's input is dead after it.
fn concat_chain(steps: usize) -> String {
    let mut source =
        String::from("module Sc.Main\nexport (main)\ndef main() -> i64 = {\n  s0 = \"\"\n");
    for step in 1..=steps {
        source.push_str(&format!(
            "  s{step} = string_concat(s{}, \"x\")\n",
            step - 1
        ));
    }
    source.push_str(&format!("  string_len(s{steps})\n}}\n"));
    source
}

#[test]
fn string_alias_controls_keep_their_values_on_both_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    for (name, program, expected) in [
        ("string_shared_seed", SHARED_SEED, "main = 431"),
        ("string_branched_alias", BRANCHED_ALIAS, "main = 32"),
        ("string_nested_holder", NESTED_HOLDER, "main = 322"),
        ("string_empty_seed", EMPTY_SEED, "main = 20"),
        ("string_observable_source", OBSERVABLE_SOURCE, "main = 32"),
        ("string_tuple_held", TUPLE_HELD, "main = 6"),
        ("string_match_arm_self", MATCH_ARM_SELF, "main = 4"),
        ("string_multibyte", MULTIBYTE, "main = 115"),
    ] {
        let evaluated = eval_main(program);
        assert_eq!(evaluated, expected, "{name}: eval value");
        let (_, compiled) = c_main(program, name);
        assert_eq!(compiled, expected, "{name}: compiled value");
    }
}

#[test]
fn let_bound_string_concat_chain_does_not_clone_per_step() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    fn cloning_calls(steps: usize) -> usize {
        let (generated, value) = c_main(&concat_chain(steps), &format!("strchain{steps}"));
        assert_eq!(value, format!("main = {steps}"), "chain of {steps} concats");
        generated.matches(" = chelis_string_concat(").count()
    }
    let small = cloning_calls(8);
    let large = cloning_calls(16);
    eprintln!(
        "#2205 string receipt: 8-step chain emits {small} cloning concats, \
         16-step chain emits {large}"
    );
    assert!(
        large <= small,
        "#2205: cloning concats must not grow with the chain; an 8-step chain emitted \
         {small} `chelis_string_concat(` calls and a 16-step chain emitted {large}"
    );
    assert_eq!(
        (small, large),
        (0, 0),
        "#2205: a let-bound chain never clones; the exact count is a count of emitted calls \
         and carries no machine budget"
    );
}

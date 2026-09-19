//! chelis#2205: a container whose last use is `append`/`concat` moves into the
//! builtin instead of being cloned, and never changes what a program means.
//!
//! Two halves. The alias controls (written before the lowering change, as
//! #943 required) pin that every program in which the appended list is still
//! observable somewhere else keeps its value on both lanes and never aborts:
//! a shared seed grown twice, a tuple-held alias appended in one branch, a
//! nested list whose inner list is appended, a typed empty seed appended
//! twice, and an `Option`-held source read after the append. The three
//! witnesses are the shapes on which a verified Move meets a strong-owner
//! count above one (aggregate construction retains the list), which is why
//! the consuming entry points check uniqueness at run time.
//!
//! The counted receipt is the number of cloning `chelis_list_append(` calls the
//! C lane emits for an N-step let-bound append chain, asserted as a ratio so it
//! carries no machine budget.
//!
//! Evidentiary status: REGRESSION TEST for the receipt, proven failing first:
//! on `main` (`e0250b290`, the same emitter) the 8-step chain emits 8 cloning
//! calls and the 16-step chain 16; after the fix both emit 0. The
//! value-parity rows are LOCKS: they pass on `main` too, which is the point.
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
/// list) surfaces as a non-zero exit and fails here.
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

const SHARED_SEED: &str = "module Ac.Main\n\
export (main)\n\
def grow(seed: List[i64], n: i64) -> List[i64] = if lte(n, cast(0, i64)) then seed else grow(append(seed, n), sub(n, cast(1, i64)))\n\
def main() -> i64 = {\n\
  seed = [cast(7, i64)]\n\
  a = grow(seed, cast(3, i64))\n\
  b = grow(seed, cast(2, i64))\n\
  add(add(mul(len(a), cast(100, i64)), mul(len(b), cast(10, i64))), len(seed))\n\
}\n";

const BRANCHED_ALIAS: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64)]\n\
  held = (xs, cast(0, i64))\n\
  ys = if gt(len(xs), cast(1, i64)) then append(xs, cast(3, i64)) else xs\n\
  add(mul(len(ys), cast(10, i64)), len(held.0))\n\
}\n";

const NESTED_LIST: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  inner = [cast(1, i64), cast(2, i64)]\n\
  outer = [inner, inner]\n\
  grown = append(inner, cast(3, i64))\n\
  add(mul(len(grown), cast(100, i64)), add(mul(len(index(outer, cast(0, i64))), cast(10, i64)), len(index(outer, cast(1, i64)))))\n\
}\n";

const EMPTY_LIST: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  empty: List[i64] = []\n\
  held = (empty, cast(0, i64))\n\
  one = append(empty, cast(1, i64))\n\
  two = append(one, cast(2, i64))\n\
  add(mul(len(two), cast(10, i64)), len(held.0))\n\
}\n";

const OBSERVABLE_SOURCE: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  source = [cast(1, i64), cast(2, i64)]\n\
  keep = Some(source)\n\
  grown = append(source, cast(3, i64))\n\
  match keep with {\n\
    | Some(original) => add(mul(len(grown), cast(10, i64)), index(original, cast(1, i64)))\n\
    | None => cast(0, i64)\n\
  }\n\
}\n";

const TUPLE_HELD: &str = "module Aw.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64)]\n\
  held = (xs, cast(9, i64))\n\
  zs = append(xs, cast(3, i64))\n\
  add(len(zs), len(held.0))\n\
}\n";

const RETURNED_PAIR: &str = "module Aw.Main\n\
export (main)\n\
def pair(xs: List[i64]) -> (List[i64], List[i64]) = (xs, xs)\n\
def main() -> i64 = {\n\
  both = pair([cast(1, i64), cast(2, i64)])\n\
  zs = append(both.0, cast(3, i64))\n\
  add(len(zs), len(both.1))\n\
}\n";

const CONCAT_HELD: &str = "module Aw.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64)]\n\
  held = Some(xs)\n\
  joined = concat(xs, [cast(3, i64), cast(4, i64)])\n\
  match held with {\n\
    | Some(kept) => add(mul(len(joined), cast(10, i64)), len(kept))\n\
    | None => cast(0, i64)\n\
  }\n\
}\n";

/// RT-2225 round 1 P0 witnesses: `concat` whose rhs aliases the consumed lhs
/// through a branch, an index, an ADT or a match arm. Two operand identities,
/// one runtime list; the owned entry point must clone, never abort.
const BRANCH_KEEP: &str = "module Az.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64)]\n\
  ys = if gt(len(xs), cast(0, i64)) then xs else [cast(9, i64)]\n\
  zs = concat(xs, ys)\n\
  len(zs)\n\
}\n";

const INDEX_CONCAT: &str = "module Az.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  inner = [cast(1, i64), cast(2, i64)]\n\
  outer = [inner, inner]\n\
  a = index(outer, cast(0, i64))\n\
  b = index(outer, cast(1, i64))\n\
  joined = concat(a, b)\n\
  len(joined)\n\
}\n";

const ADT_CONCAT: &str = "module Az.Main\n\
export (main)\n\
type Pair2 = | Both(List[i64], List[i64])\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64)]\n\
  p = Both(xs, xs)\n\
  match p with {\n\
    | Both(l, r) => len(concat(l, r))\n\
  }\n\
}\n";

const MATCH_ARM_CONCAT: &str = "module Az.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64)]\n\
  held = Some(xs)\n\
  out = match held with {\n\
    | Some(kept) => concat(xs, kept)\n\
    | None => xs\n\
  }\n\
  len(out)\n\
}\n";

/// An N-step let-bound append chain: every step's input is dead after it.
fn append_chain(steps: usize) -> String {
    let mut source = String::from(
        "module Ac.Main\nexport (main)\ndef main() -> i64 = {\n  x0: List[i64] = []\n",
    );
    for step in 1..=steps {
        source.push_str(&format!(
            "  x{step} = append(x{}, cast({step}, i64))\n",
            step - 1
        ));
    }
    source.push_str(&format!("  len(x{steps})\n}}\n"));
    source
}

#[test]
fn alias_controls_and_witnesses_keep_their_values_on_both_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    for (name, program, expected) in [
        ("shared_seed", SHARED_SEED, "main = 431"),
        ("branched_alias", BRANCHED_ALIAS, "main = 32"),
        ("nested_list", NESTED_LIST, "main = 322"),
        ("empty_list", EMPTY_LIST, "main = 20"),
        ("observable_source", OBSERVABLE_SOURCE, "main = 32"),
        ("tuple_held", TUPLE_HELD, "main = 5"),
        ("returned_pair", RETURNED_PAIR, "main = 5"),
        ("concat_held", CONCAT_HELD, "main = 42"),
        ("branch_keep", BRANCH_KEEP, "main = 4"),
        ("index_concat", INDEX_CONCAT, "main = 4"),
        ("adt_concat", ADT_CONCAT, "main = 4"),
        ("match_arm_concat", MATCH_ARM_CONCAT, "main = 4"),
    ] {
        let evaluated = eval_main(program);
        assert_eq!(evaluated, expected, "{name}: eval value");
        let (_, compiled) = c_main(program, name);
        assert_eq!(compiled, expected, "{name}: compiled value");
    }
}

#[test]
fn let_bound_append_chain_does_not_clone_per_step() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    fn cloning_calls(steps: usize) -> usize {
        let (generated, value) = c_main(&append_chain(steps), &format!("chain{steps}"));
        assert_eq!(value, format!("main = {steps}"), "chain of {steps} appends");
        generated.matches(" = chelis_list_append(").count()
    }
    let small = cloning_calls(8);
    let large = cloning_calls(16);
    eprintln!(
        "#2205 receipt: 8-step chain emits {small} cloning appends, 16-step chain emits {large}"
    );
    assert!(
        large <= small,
        "#2205: cloning appends must not grow with the chain; an 8-step chain emitted {small} \
         `chelis_list_append(` calls and a 16-step chain emitted {large}"
    );
    assert_eq!(
        (small, large),
        (0, 0),
        "#2205: a let-bound chain never clones; the exact count is a count of emitted calls \
         and carries no machine budget"
    );
}

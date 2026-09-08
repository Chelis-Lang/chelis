//! Every compiler-source string literal naming `expand` or `insert`, with both
//! counts, per file.
//!
//! chelis#1277 splits one primitive into two. The sibling inventory,
//! `expand_call_site_inventory`, pins the surviving `expand` PROGRAM sites --
//! Chelis text a user could run. This one pins something different and, so
//! far, more dangerous: the compiler's own literals. A dispatch key, a
//! `matches!` arm, a registry row, or a callee spelled inside a diagnostic
//! message is not program text, so the sibling never sees it, and three
//! separate review rounds each found a site of this kind that the migration
//! had missed.
//!
//! # Why both counts, and why a per-file count of one name is not enough
//!
//! Counting `expand` alone catches a NEW literal appearing. It cannot catch
//! the defect that actually bit, four times: an EXISTING literal that never
//! got its counterpart. `vmap_extent.rs` listed `"expand"` in the §3.7
//! movement guard and gained no `"insert"`, so the guard went silent for the
//! spelling the migration had just made canonical, and its `expand` count
//! never moved. The same shape appeared in `validate.rs`'s rank-derivation
//! table, in the chelis#469 sourceless-size diagnostic, and in the
//! host-runtime evaluator's negative-extent errors.
//!
//! The asymmetry between the two counts is exactly that signal. Three and
//! three is a file where every mention was paired. Three and zero is either a
//! deliberate old-name-only site or the next `vmap_extent.rs`, and the row has
//! to say which. So a written reason is REQUIRED whenever the counts differ,
//! and the test fails without one.
//!
//! # What the counts do not prove
//!
//! Two asymmetries in one file cancel. `app_post.rs` reads four and four, and
//! one of those four `expand`s is `shape_override_operand_error`'s
//! parallel-list entry, which still has no `insert` beside it. Round 2 probed
//! that path and found it inert today, and the row below says so. The count
//! pair is a tripwire on MOVEMENT; the reason column is where the file records
//! what a human established. A row whose counts are equal may still carry a
//! reason, and several do.
//!
//! # The needle
//!
//! The word `expand` or `insert`, with word boundaries, anywhere inside a
//! string literal under `crates/*/src`. Not the exact token `"expand"`: that
//! narrower needle finds only the dispatch keys, and it would have been blind
//! to every diagnostic-message instance, which is the majority of them. Word
//! boundaries keep `expand_dims` and `fallback_expand_type` out.
//!
//! The English words are in scope too, and they are, deliberately, most of the
//! noise: "insert whitespace", "desugar+expand ok", "should not insert a
//! redundant cast". They cost a row and a reason each, once. The alternative
//! is a cleverer needle that decides for itself which mentions matter, which
//! is the decision this file exists to take away from a scanner.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `(path, expand literals, insert literals, reason)`.
///
/// The reason is required when the two counts differ and optional when they
/// agree. Sorted by path, which is also the order the failure message prints.
const ALLOWED: &[(&str, usize, usize, &str)] = &[
    (
        "crates/chelis-backend-metal/src/kernels.rs",
        0,
        1,
        "the English verb, in a test message about a redundant cast",
    ),
    (
        "crates/chelis-compiler-api/src/compiler.rs",
        6,
        0,
        "backend messages and their tests, all about the IR's single `RiscOp::Expand` node, which both spellings lower to",
    ),
    (
        "crates/chelis-compiler-api/src/fragment.rs",
        1,
        0,
        "`macro expand`, the English word for macro expansion",
    ),
    (
        "crates/chelis-compiler-api/src/runtime/eval.rs",
        2,
        3,
        "the host dispatch arm and the named-axis test name the two spellings together; the extra `insert` is the dispatch's own branch, which routes the two evaluators by callee now that each operation has one result shape",
    ),
    (
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        1,
        1,
        "one each: `tensor_expand_host`'s `NumericTrap::Domain` names `expand` as the operation whose precondition it guards, and `insert position` is the English phrase in `tensor_insert_host`'s bounds message. Neither is a dispatch key; both functions take the callee from the evaluator",
    ),
    (
        "crates/chelis-compiler-api/src/runtime/named_axis.rs",
        1,
        1,
        "",
    ),
    (
        "crates/chelis-compiler-api/src/runtime/tests.rs",
        0,
        10,
        "these host-runtime tests exercise `insert` programs only",
    ),
    (
        "crates/chelis-compiler-api/src/runtime/transforms.rs",
        1,
        0,
        "an example list of tensor-lane primitives (`range`, `map`, `expand`), not a dispatch; `expand` is still one of them",
    ),
    (
        "crates/chelis-conformance/src/expect.rs",
        1,
        0,
        "sample error text in a unit test of the expectation classifier, which is about substring matching and not about any operation",
    ),
    (
        "crates/chelis-deep/src/lexer.rs",
        0,
        2,
        "`insert whitespace`, the English verb, in a literal-suffix diagnostic",
    ),
    (
        "crates/chelis-deep/src/tag.rs",
        1,
        0,
        "the Deep macro-expansion tag, which is not the builtin and does not split",
    ),
    (
        "crates/chelis-effects/src/lib.rs",
        0,
        1,
        "a test program, renamed with the rest",
    ),
    (
        "crates/chelis-effects/src/realizability.rs",
        0,
        1,
        "a test program, renamed with the rest",
    ),
    (
        "crates/chelis-ir/src/axis_sources.rs",
        1,
        0,
        "one `expand`, in `UnitExtentClaim::trap_op`: the `<op>` slot section 4.7 gives a locally placed unit-extent claim. It is the rendered operation name, not a dispatch key, and there is no `insert` counterpart because `insert` makes no such claim",
    ),
    (
        "crates/chelis-ir/src/dag.rs",
        2,
        0,
        "guidance naming the `expand`+`mul` lowering of an integer inner product, and a test reading an IR error; both are IR-level",
    ),
    (
        "crates/chelis-ir/src/eval.rs",
        3,
        0,
        "DAG evaluator messages for `RiscOp::Expand`, the one IR op both spellings lower to",
    ),
    (
        "crates/chelis-ir/src/grad.rs",
        1,
        0,
        "the IR op's own printed name, produced by the `RiscOp::Expand` arm of the \
         op-name mapping",
    ),
    (
        "crates/chelis-ir/src/host.rs",
        8,
        5,
        "two movement dispatch lists carry both spellings; the surplus is the IR op's printed name and assertions about Expand nodes",
    ),
    (
        "crates/chelis-ir/src/lower.rs",
        22,
        10,
        "the lowering dispatch carries both, and the shared movement arm now names its callee through `{callee}` rather than hard-coding `expand`: the named-collision raise, the operand-desync raise, the non-literal-axis construct, and both size diagnostics moved with the arm split, which is what took this row from 28/3 to 22/10, the last move being the chelis#318 extent-recovery fixture, whose program raises the rank and so spells `insert`. The surplus 23 are embedded Deep fixtures spelling `(var {} expand)` and prose in comments, neither of which a program can reach. Re-checked at S2b as the previous reason asked; B2b's widening of the sourceless-size rejection is still outstanding",
    ),
    (
        "crates/chelis-ir/src/tier2.rs",
        1,
        0,
        "a test message about softmax's Expand nodes, IR-level",
    ),
    (
        "crates/chelis-ir/src/verify.rs",
        15,
        0,
        "the IR verifier's C10 rules for `RiscOp::Expand`. The verifier sees one op, and its same-rank and rank-plus-one cases are the two shapes that op has",
    ),
    (
        "crates/chelis-macros/src/lib.rs",
        0,
        1,
        "the `linear_layer` bias, renamed to the rank-increasing spelling",
    ),
    (
        "crates/chelis-reef/src/lib.rs",
        8,
        0,
        "`desugar+expand ok`, the English word for macro expansion, in eight sibling tests",
    ),
    (
        "crates/chelis-surf/src/desugar.rs",
        1,
        0,
        "the Deep macro-expansion tag, which is not the builtin and does not split",
    ),
    (
        "crates/chelis-surf/src/lexer.rs",
        0,
        2,
        "`insert whitespace`, the English verb, in a literal-suffix diagnostic",
    ),
    (
        "crates/chelis-types/src/builtins.rs",
        7,
        7,
        "the registry. Every list here carries both spellings, and the pairing is the point",
    ),
    (
        "crates/chelis-types/src/infer/app.rs",
        4,
        5,
        "three routing arms name both spellings; the extra `insert` is the two-arm selection that picks which callee to report",
    ),
    (
        "crates/chelis-types/src/infer/app_post.rs",
        4,
        4,
        "equal, but not fully paired: `shape_override_operand_error`'s list still names only `expand`. Round 2 probed the chelis#731 cascade through both spellings and got one identical witness, because the Error-propagating arm covers it first, so the gap is inert today. Two asymmetries in one file cancel, which is why the reason and not the count carries this",
    ),
    (
        "crates/chelis-types/src/infer/app_tensor.rs",
        1,
        10,
        "the one `expand` is guidance about the `expand`+`mul` lowering of an integer inner product. The `insert`s are the one-shape flag plus the diagnostics that name `insert` as the fix when a program asks `expand` to raise a rank: the named-axis rejection, the four-argument rejection, the axis-range message, and the rank message. Every one of them names its own callee through `{builtin}` and names `insert` as a literal because it is telling the user which other operation to write",
    ),
    ("crates/chelis-types/src/infer/common.rs", 1, 1, ""),
    (
        "crates/chelis-types/src/infer/shape_honesty.rs",
        0,
        1,
        "the rank-change suggestion, which after the split is `insert` and cannot be `expand`",
    ),
    (
        "crates/chelis-types/src/infer/tests/more.rs",
        0,
        1,
        "a test program, renamed with the rest",
    ),
    (
        "crates/chelis-types/src/infer/validate.rs",
        3,
        2,
        "the movement allowlist and the rank-derivation table each carry both; the extra `expand` is the same integer-inner-product guidance sentence",
    ),
    (
        "crates/chelis-types/src/infer/vmap_extent.rs",
        2,
        2,
        "both §3.7 guard arms carry both spellings. This file is why the inventory records two counts: it listed only `expand` and its `expand` count never moved",
    ),
    ("crates/chelis-types/src/linearity.rs", 1, 1, ""),
    (
        "crates/chelis-types/src/unify.rs",
        18,
        2,
        "the deferred two-shape constraint machinery and its tests. `insert` has one legal shape and records no such constraint, so these messages can only ever name `expand`; a counterpart here would be unreachable code",
    ),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate sits two levels below the repository root")
        .to_path_buf()
}

/// Does `text[at..]` start with `needle` standing as a whole word?
fn word_at(chars: &[char], at: usize, needle: &[char]) -> bool {
    if !chars[at..].starts_with(needle) {
        return false;
    }
    let before_ok = at == 0 || !is_word_char(chars[at - 1]);
    let after = at + needle.len();
    let after_ok = after >= chars.len() || !is_word_char(chars[after]);
    before_ok && after_ok
}

fn is_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

/// Count whole-word `expand` and `insert` occurrences inside string literals.
///
/// A character state machine, not a regular expression. A regex for Rust
/// string literals cannot see a `//` comment, a nested `/* */` block, or a
/// `'"'` char literal, and one stray quote then swallows the rest of the file:
/// the first instrument built for this migration reported eleven Rust call
/// sites in `chelis-ir/src/host.rs` as program text for exactly that reason.
///
/// Raw strings matter here in a way they do not for the sibling inventory,
/// because `crates/*/src` writes diagnostics and embedded Deep fixtures with
/// `r"..."` and `r#"..."#`.
fn literal_word_counts(text: &str) -> (usize, usize) {
    let chars: Vec<char> = text.chars().collect();
    let expand: Vec<char> = "expand".chars().collect();
    let insert: Vec<char> = "insert".chars().collect();
    let n = chars.len();
    let (mut expands, mut inserts) = (0usize, 0usize);
    let mut i = 0usize;

    // Count the needles inside one already-delimited literal body.
    let tally = |chars: &[char], from: usize, to: usize, e: &mut usize, s: &mut usize| {
        let mut k = from;
        while k < to {
            if word_at(chars, k, &expand) {
                *e += 1;
                k += expand.len();
            } else if word_at(chars, k, &insert) {
                *s += 1;
                k += insert.len();
            } else {
                k += 1;
            }
        }
    };

    while i < n {
        let ch = chars[i];

        // Line comment.
        if ch == '/' && chars.get(i + 1) == Some(&'/') {
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // Block comment; Rust nests them.
        if ch == '/' && chars.get(i + 1) == Some(&'*') {
            let mut depth = 1usize;
            i += 2;
            while i < n && depth > 0 {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        // Raw string, with any number of hashes. `b`-prefixed forms reach here
        // with the `b` already consumed as an ordinary code character.
        if ch == 'r' && matches!(chars.get(i + 1), Some('#') | Some('"')) {
            let mut j = i + 1;
            let mut hashes = 0usize;
            while j < n && chars[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if chars.get(j) == Some(&'"') {
                let body_start = j + 1;
                let mut k = body_start;
                let mut end = n;
                while k < n {
                    if chars[k] == '"'
                        && chars[k + 1..]
                            .iter()
                            .take(hashes)
                            .filter(|c| **c == '#')
                            .count()
                            == hashes
                    {
                        end = k;
                        break;
                    }
                    k += 1;
                }
                tally(&chars, body_start, end, &mut expands, &mut inserts);
                i = if end == n { n } else { end + 1 + hashes };
                continue;
            }
        }
        // Char literal or lifetime. `'"'` is why this arm exists.
        if ch == '\'' {
            if chars.get(i + 1) == Some(&'\\') {
                let mut j = i + 2;
                while j < n && chars[j] != '\'' {
                    j += 1;
                }
                i = j + 1;
                continue;
            }
            if chars.get(i + 2) == Some(&'\'') {
                i += 3;
                continue;
            }
            i += 1;
            continue;
        }
        // Normal string.
        if ch == '"' {
            let body_start = i + 1;
            let mut j = body_start;
            while j < n {
                if chars[j] == '\\' {
                    j += 2;
                    continue;
                }
                if chars[j] == '"' {
                    break;
                }
                j += 1;
            }
            tally(&chars, body_start, j.min(n), &mut expands, &mut inserts);
            i = j + 1;
            continue;
        }
        i += 1;
    }
    (expands, inserts)
}

fn measure(root: &Path) -> Vec<(String, usize, usize)> {
    let listed = Command::new("git")
        .arg("ls-files")
        .arg("crates")
        .current_dir(root)
        .output()
        .expect("git ls-files runs in the repository");
    assert!(listed.status.success(), "git ls-files failed");
    let listed = String::from_utf8(listed.stdout).expect("git ls-files emits utf-8");

    let mut measured = Vec::new();
    for rel in listed.lines() {
        if !rel.ends_with(".rs") || !rel.contains("/src/") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        if !text.contains("expand") && !text.contains("insert") {
            continue;
        }
        let (expands, inserts) = literal_word_counts(&text);
        if expands == 0 && inserts == 0 {
            continue;
        }
        measured.push((rel.to_string(), expands, inserts));
    }
    measured.sort();
    measured
}

#[test]
fn every_expand_or_insert_source_literal_is_inventoried() {
    let root = repo_root();
    let measured = measure(&root);

    let mut expected: Vec<(String, usize, usize)> = ALLOWED
        .iter()
        .map(|(path, expands, inserts, _)| ((*path).to_string(), *expands, *inserts))
        .collect();
    expected.sort();

    if measured != expected {
        let rows: String = measured
            .iter()
            .map(|(path, expands, inserts)| {
                format!("    (\"{path}\", {expands}, {inserts}, \"\"),\n")
            })
            .collect();
        panic!(
            "the `expand`/`insert` source literals moved.\n\n\
             A count that rose is a new literal: check that its counterpart \
             exists. A count that fell without its partner falling is a \
             literal that lost its pair, which is the defect this inventory \
             exists to catch. Update the row, and where the two counts differ \
             write the reason.\n\n\
             measured rows:\n{rows}\n\
             expected rows: {expected:#?}"
        );
    }
}

#[test]
fn every_asymmetric_row_states_a_reason() {
    let missing: Vec<&str> = ALLOWED
        .iter()
        .filter(|(_, expands, inserts, reason)| expands != inserts && reason.trim().is_empty())
        .map(|(path, _, _, _)| *path)
        .collect();
    assert!(
        missing.is_empty(),
        "these rows count the two names differently and say nothing about why. \
         An unequal pair is either a deliberate old-name-only site or a literal \
         whose counterpart was never written; the row has to say which: {missing:?}"
    );
}

#[test]
fn no_path_is_listed_twice() {
    let mut paths: Vec<&str> = ALLOWED.iter().map(|(path, _, _, _)| *path).collect();
    paths.sort_unstable();
    let before = paths.len();
    paths.dedup();
    assert_eq!(
        before,
        paths.len(),
        "a duplicated path lets one row's counts hide another's"
    );
}

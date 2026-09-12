//! chelis#1678: `chelis check <dir>` emits one typed envelope (spec/04
//! § Directory mode, [04-FIT-19] through [04-FIT-25]).
//!
//! Before this change directory mode was assembled by `format!`: a populated
//! directory printed `{"files":[...]}`, an empty one printed a different
//! shape, `{"files":[],"errors":[]}`, and a directory the walk could not read
//! printed NOTHING and exited 1 -- one unreadable subdirectory discarded
//! every readable file's report. Each atom below has its own section.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use chelis_compiler_api::schema::WireCheckDirectoryReport;
use serde_json::Value;
use tempfile::tempdir;

const CLEAN: &str = "def f(x: f32) -> f32 = add(x, x)\n";
const BROKEN: &str = "def g(x: f32) -> f32 = add(x, nope)\n";

struct Run {
    code: Option<i32>,
    envelope: Value,
    stdout: String,
}

impl Run {
    fn files(&self) -> Vec<String> {
        self.envelope["files"]
            .as_array()
            .expect("files array")
            .iter()
            .map(|entry| entry["file"].as_str().expect("file string").to_string())
            .collect()
    }

    fn error_kinds(&self) -> Vec<String> {
        self.envelope["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .map(|error| error["kind"].as_str().expect("kind string").to_string())
            .collect()
    }

    fn error_messages(&self) -> Vec<String> {
        self.envelope["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .map(|error| error["message"].as_str().expect("message").to_string())
            .collect()
    }

    fn entry(&self, name: &str) -> &Value {
        self.envelope["files"]
            .as_array()
            .expect("files array")
            .iter()
            .find(|entry| entry["file"] == name)
            .unwrap_or_else(|| panic!("no entry for {name}: {}", self.stdout))
    }
}

fn check(target: &Path) -> Run {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(target)
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let envelope: Value = serde_json::from_str(&stdout).unwrap_or_else(|error| {
        panic!(
            "directory mode must print one JSON document ({error}); stdout={stdout} stderr={stderr}"
        )
    });
    Run {
        code: output.status.code(),
        envelope,
        stdout,
    }
}

fn write(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, text).expect("write");
}

/// Makes `path` unreadable, and restores it on drop so the tempdir can be
/// removed. `None` when permissions are not enforced (running as root), in
/// which case the caller skips: asserting a failure the host cannot produce
/// would test nothing.
struct Unreadable(PathBuf);

impl Unreadable {
    fn new(path: &Path) -> Option<Self> {
        fs::set_permissions(path, fs::Permissions::from_mode(0o000)).expect("chmod");
        let guard = Self(path.to_path_buf());
        if fs::read_dir(path).is_ok() {
            eprintln!("skipped: permissions are not enforced for this user");
            return None;
        }
        Some(guard)
    }
}

impl Drop for Unreadable {
    fn drop(&mut self) {
        let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
    }
}

// ---------------------------------------------------------------------------
// [04-FIT-19]: one typed value, `files` and `errors` always present.
// ---------------------------------------------------------------------------

/// Sorted, so the set is compared independently of how `serde_json` orders a
/// map. Member order is asserted separately, on the bytes.
fn member_names(envelope: &Value) -> Vec<String> {
    let mut names: Vec<String> = envelope
        .as_object()
        .expect("the envelope is an object")
        .keys()
        .cloned()
        .collect();
    names.sort();
    names
}

/// The same two members in every outcome. The old populated document had no
/// `errors` at all, so a consumer reading it found "no errors" by absence.
#[test]
fn every_outcome_has_exactly_files_then_errors() {
    let populated = tempdir().unwrap();
    write(&populated.path().join("a.ch"), CLEAN);
    let empty = tempdir().unwrap();
    let failing = tempdir().unwrap();
    write(&failing.path().join("a.ch"), BROKEN);

    for (label, dir) in [
        ("populated", populated.path()),
        ("empty", empty.path()),
        ("failing", failing.path()),
    ] {
        let run = check(dir);
        assert_eq!(
            member_names(&run.envelope),
            ["errors", "files"],
            "{label}: {}",
            run.stdout
        );
        // Member ORDER is part of the bytes: `files` first.
        assert!(
            run.stdout.starts_with("{\n  \"files\": ["),
            "{label}: {}",
            run.stdout
        );
    }
}

/// The document reads back through the consumer type, and each entry's
/// `report` is the same report `chelis check <file>` prints for that file:
/// the envelope carries reports, it does not re-shape them.
#[test]
fn each_entry_carries_the_single_file_report_unchanged() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    write(&dir.path().join("b.ch"), BROKEN);
    let run = check(dir.path());

    let typed: WireCheckDirectoryReport =
        serde_json::from_str(&run.stdout).expect("the consumer type reads the envelope");
    assert_eq!(typed.files.len(), 2);
    assert!(typed.errors.is_empty());

    for name in ["a.ch", "b.ch"] {
        let single = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("check")
            .arg(dir.path().join(name))
            .output()
            .unwrap();
        let single: Value = serde_json::from_slice(&single.stdout).expect("single-file report");
        assert_eq!(run.entry(name)["report"], single, "{name}");
    }
}

/// One serialization: the whole envelope is one line per top-level member,
/// and no report inside it is the pretty, multi-line single-file rendering
/// spliced in (the old `format!` shape).
#[test]
fn reports_are_serialized_in_place_not_spliced() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    let run = check(dir.path());
    assert_eq!(
        run.stdout.trim_end().lines().count(),
        4,
        "`{{`, `files`, `errors`, `}}`: {}",
        run.stdout
    );
    assert!(!run.stdout.contains("\"report\":{\n"), "{}", run.stdout);
}

// ---------------------------------------------------------------------------
// [04-FIT-20]: the corpus and the walk.
// ---------------------------------------------------------------------------

#[test]
fn exclusions_apply_below_the_target_by_name() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    write(&dir.path().join("s.dp"), "(lit {} 1)\n");
    write(&dir.path().join(".hidden.ch"), BROKEN);
    write(&dir.path().join(".git/x.ch"), BROKEN);
    write(&dir.path().join("target/built.ch"), BROKEN);
    write(&dir.path().join("sub/target/built.ch"), BROKEN);
    // A FILE named like the excluded directory is not excluded.
    write(&dir.path().join("target.ch"), CLEAN);
    write(&dir.path().join("README.md"), "# not chelis\n");
    let run = check(dir.path());
    assert_eq!(run.files(), ["a.ch", "s.dp", "target.ch"], "{}", run.stdout);
}

/// The target itself is never excluded, whatever its name.
#[test]
fn the_target_itself_is_never_excluded() {
    let root = tempdir().unwrap();
    for name in [".dotted", "target"] {
        let dir = root.path().join(name);
        write(&dir.join("a.ch"), CLEAN);
        let run = check(&dir);
        assert_eq!(run.files(), ["a.ch"], "{name}: {}", run.stdout);
        assert_eq!(run.code, Some(0), "{name}");
    }
}

/// #1827 red-team F1: a dot-named or `target` link that fails to resolve
/// aborted the whole walk, because `walkdir` resolved it before the name
/// filter saw it. The name test now runs first, so an excluded link
/// contributes nothing, whatever it points at.
#[test]
fn an_excluded_link_contributes_nothing_whatever_it_resolves_to() {
    let root = tempdir().unwrap();
    let dir = root.path().join("checked");
    write(&dir.join("a.ch"), CLEAN);
    let locked = root.path().join("locked");
    write(&locked.join("secret.ch"), BROKEN);
    symlink("../locked", dir.join(".cache")).unwrap();
    symlink("../locked", dir.join("target")).unwrap();
    symlink(".self", dir.join(".self")).unwrap();
    let Some(_guard) = Unreadable::new(&locked) else {
        return;
    };
    let run = check(&dir);
    assert_eq!(run.files(), ["a.ch"], "{}", run.stdout);
    assert!(run.error_kinds().is_empty(), "{}", run.stdout);
    assert_eq!(run.code, Some(0));
}

/// Each file is visited once, by canonical path, under the first path in
/// walk order. #1827 reported one type error three times through two
/// aliases.
#[test]
fn a_file_reached_by_several_paths_is_one_entry() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("lib/bad.ch"), BROKEN);
    symlink("lib/bad.ch", dir.path().join("alias.ch")).unwrap();
    symlink("lib", dir.path().join("lib2")).unwrap();
    let run = check(dir.path());
    // `alias.ch` sorts first, so it names the file.
    assert_eq!(run.files(), ["alias.ch"], "{}", run.stdout);
    assert_eq!(run.code, Some(2));
}

/// #1827 red-team F4: sixteen links doubling back over eight levels gave
/// 511 entries for one file. This pins the entry count; that the walk also
/// stops RE-ENTERING each directory is pinned by
/// `a_link_out_of_the_target_is_followed_once`, whose expected paths a
/// directory revisit would lengthen.
#[test]
fn doubling_links_do_not_amplify_the_walk() {
    let dir = tempdir().unwrap();
    let mut level = dir.path().join("l0");
    write(&level.join("only.ch"), CLEAN);
    for depth in 1..=8 {
        let next = dir.path().join(format!("l{depth}"));
        fs::create_dir_all(&next).unwrap();
        symlink(&level, next.join("a")).unwrap();
        symlink(&level, next.join("b")).unwrap();
        level = next;
    }
    let run = check(dir.path());
    assert_eq!(run.files().len(), 1, "{}", run.stdout);
}

/// Links are followed wherever they resolve, including out of the target;
/// a link back to the target itself adds nothing, because it is visited.
#[test]
fn a_link_out_of_the_target_is_followed_once() {
    let root = tempdir().unwrap();
    let target = root.path().join("checked");
    write(&target.join("in.ch"), CLEAN);
    write(&root.path().join("outside.ch"), BROKEN);
    symlink("..", target.join("up")).unwrap();
    let run = check(&target);
    assert_eq!(run.files(), ["in.ch", "up/outside.ch"], "{}", run.stdout);
    assert_eq!(run.code, Some(2));
}

/// An entry that cannot be resolved: an entry when its name is a source
/// name, nothing when its referent does not exist, and a walk failure
/// otherwise. These three pin the distinction the dangling-link check made
/// on #1827, which red-team F2 found unpinned.
#[test]
fn an_unresolvable_source_name_is_an_entry_with_the_read_failure() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    symlink("nowhere.ch", dir.path().join("gone.ch")).unwrap();
    symlink("loop.ch", dir.path().join("loop.ch")).unwrap();
    let run = check(dir.path());
    assert_eq!(
        run.files(),
        ["a.ch", "gone.ch", "loop.ch"],
        "{}",
        run.stdout
    );
    for name in ["gone.ch", "loop.ch"] {
        let message = run.entry(name)["report"]["errors"][0]["message"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert!(message.contains("failed to read"), "{name}: {message}");
    }
    assert!(run.error_kinds().is_empty(), "{}", run.stdout);
    assert_eq!(run.code, Some(2));
}

#[test]
fn a_dangling_non_source_link_contributes_nothing() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    symlink("nowhere", dir.path().join("vendored")).unwrap();
    let run = check(dir.path());
    assert_eq!(run.files(), ["a.ch"], "{}", run.stdout);
    assert!(run.error_kinds().is_empty(), "{}", run.stdout);
    assert_eq!(run.code, Some(0));
}

/// A self-referencing link is not dangling (its referent is not absent, it
/// cannot be resolved at all), so it is a walk failure -- and the walk goes on.
#[test]
fn an_unresolvable_non_source_link_is_a_walk_failure_not_an_abort() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    symlink("selfloop", dir.path().join("selfloop")).unwrap();
    write(&dir.path().join("z.ch"), CLEAN);
    let run = check(dir.path());
    assert_eq!(run.files(), ["a.ch", "z.ch"], "{}", run.stdout);
    assert_eq!(
        run.error_kinds(),
        ["directory_walk_error"],
        "{}",
        run.stdout
    );
    assert!(
        run.error_messages()[0].contains("selfloop"),
        "the diagnostic names the entry: {}",
        run.stdout
    );
    assert_eq!(run.code, Some(2));
}

// ---------------------------------------------------------------------------
// [04-FIT-21]: `file` is `/`-joined, relative, and never substituted.
// ---------------------------------------------------------------------------

#[test]
fn a_nested_entry_is_relative_and_slash_joined() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("one/two/three.ch"), CLEAN);
    let run = check(dir.path());
    assert_eq!(run.files(), ["one/two/three.ch"], "{}", run.stdout);
}

/// APFS and HFS+ refuse a name that is not UTF-8, so only Linux can build
/// the fixture.
#[cfg(target_os = "linux")]
#[test]
fn a_non_utf8_path_is_a_walk_failure_not_a_lossy_entry() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    let name = OsStr::from_bytes(b"\xffbad.ch");
    fs::write(dir.path().join(name), BROKEN).unwrap();
    let run = check(dir.path());
    assert_eq!(run.files(), ["a.ch"], "{}", run.stdout);
    assert!(!run.stdout.contains('\u{fffd}'), "{}", run.stdout);
    assert_eq!(
        run.error_kinds(),
        ["directory_walk_error"],
        "{}",
        run.stdout
    );
    let message = &run.error_messages()[0];
    assert!(message.contains("not valid UTF-8"), "{message}");
    // [05-HOST-4]'s rendering, which [04-FIT-23] adopts: the offending byte
    // comes back out of the message instead of being replaced.
    assert!(message.contains("\\xffbad.ch"), "{message}");
    assert_eq!(run.code, Some(2));
}

// ---------------------------------------------------------------------------
// [04-FIT-22]: walk order, not path order.
// ---------------------------------------------------------------------------

#[test]
fn files_are_in_walk_order_with_sibling_byte_order() {
    let dir = tempdir().unwrap();
    for name in [
        "a.ch", "a/b.ch", "a-b.ch", "B.ch", "_u.ch", "z/y/x.ch", "m.dp",
    ] {
        let text = if name.ends_with(".dp") {
            "(lit {} 1)\n"
        } else {
            CLEAN
        };
        write(&dir.path().join(name), text);
    }
    let run = check(dir.path());
    // `B` (0x42) < `_` (0x5F) < `a` (0x61). The directory `a` precedes the
    // file `a-b.ch` and `a.ch`, although both paths sort before `a/b.ch`.
    assert_eq!(
        run.files(),
        [
            "B.ch", "_u.ch", "a/b.ch", "a-b.ch", "a.ch", "m.dp", "z/y/x.ch"
        ],
        "{}",
        run.stdout
    );
    let mut sorted = run.files();
    sorted.sort();
    assert_ne!(
        run.files(),
        sorted,
        "the pin must discriminate from a path sort"
    );
}

// ---------------------------------------------------------------------------
// [04-FIT-23]: a failure of the walk is a diagnostic, and the walk goes on.
// ---------------------------------------------------------------------------

/// The headline defect: one unreadable subdirectory used to discard every
/// readable file's report and exit 1 with nothing on stdout.
#[test]
fn an_unreadable_subdirectory_is_one_diagnostic_beside_the_readable_files() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    write(&dir.path().join("locked/hidden_error.ch"), BROKEN);
    write(&dir.path().join("z.ch"), CLEAN);
    let Some(_guard) = Unreadable::new(&dir.path().join("locked")) else {
        return;
    };
    let run = check(dir.path());
    assert_eq!(run.files(), ["a.ch", "z.ch"], "{}", run.stdout);
    assert_eq!(
        run.error_kinds(),
        ["directory_walk_error"],
        "{}",
        run.stdout
    );
    assert!(run.error_messages()[0].contains("locked"), "{}", run.stdout);
    assert_eq!(run.code, Some(2), "the envelope still fails");
}

#[test]
fn an_unreadable_target_is_an_envelope_with_one_diagnostic() {
    let root = tempdir().unwrap();
    let dir = root.path().join("locked");
    write(&dir.join("a.ch"), CLEAN);
    let Some(_guard) = Unreadable::new(&dir) else {
        return;
    };
    let run = check(&dir);
    assert!(run.files().is_empty(), "{}", run.stdout);
    assert_eq!(
        run.error_kinds(),
        ["directory_walk_error"],
        "{}",
        run.stdout
    );
    assert_eq!(run.code, Some(2));
}

/// [04-FIT-23]: the diagnostics are in walk order too, so a second run over
/// an unchanged tree emits the same document.
#[test]
fn walk_failures_appear_in_walk_order() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("m.ch"), CLEAN);
    // Two entries that cannot be resolved, sorting either side of `m.ch`.
    symlink("a_loop", dir.path().join("a_loop")).unwrap();
    symlink("z_loop", dir.path().join("z_loop")).unwrap();
    let run = check(dir.path());
    let messages = run.error_messages();
    assert_eq!(messages.len(), 2, "{}", run.stdout);
    assert!(messages[0].contains("a_loop"), "{}", run.stdout);
    assert!(messages[1].contains("z_loop"), "{}", run.stdout);
    assert_eq!(
        check(dir.path()).stdout,
        run.stdout,
        "the document is stable"
    );
}

// ---------------------------------------------------------------------------
// [04-FIT-24]: an empty corpus is a failure, judged once, over a completed walk.
// ---------------------------------------------------------------------------

#[test]
fn an_empty_directory_is_one_empty_corpus_error() {
    let dir = tempdir().unwrap();
    let run = check(dir.path());
    assert!(run.files().is_empty());
    assert_eq!(run.error_kinds(), ["empty_corpus"], "{}", run.stdout);
    let message = &run.error_messages()[0];
    assert!(
        message.contains(&dir.path().display().to_string()),
        "the message names the target: {message}"
    );
    assert!(message.contains(" 0 excluded"), "{message}");
    assert_eq!(run.code, Some(2));
}

/// "Empty" is after the exclusions, so a directory of Markdown is empty
/// too, and one whose only sources sit under `target/` or a dot-directory
/// says how many the exclusions removed.
#[test]
fn the_empty_corpus_message_counts_what_the_exclusions_removed() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("README.md"), "# docs\n");
    write(&dir.path().join("target/a.ch"), CLEAN);
    write(&dir.path().join("target/deep/b.dp"), "(lit {} 1)\n");
    write(&dir.path().join(".scratch/c.ch"), CLEAN);
    write(&dir.path().join(".d.ch"), CLEAN);
    let run = check(dir.path());
    assert_eq!(run.error_kinds(), ["empty_corpus"], "{}", run.stdout);
    assert!(
        run.error_messages()[0].contains(" 4 excluded"),
        "{}",
        run.stdout
    );
    assert_eq!(run.code, Some(2));
}

/// Judged once, on the whole walk: an empty subdirectory beside a source is
/// not an empty corpus.
#[test]
fn an_empty_subdirectory_in_a_populated_tree_contributes_nothing() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    fs::create_dir_all(dir.path().join("empty/deeper")).unwrap();
    let run = check(dir.path());
    assert!(run.error_kinds().is_empty(), "{}", run.stdout);
    assert_eq!(run.code, Some(0));
}

/// A walk that failed has not established emptiness, and its diagnostic
/// already explains the result: never both.
#[test]
fn a_failed_walk_is_not_also_an_empty_corpus() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("locked/a.ch"), CLEAN);
    let Some(_guard) = Unreadable::new(&dir.path().join("locked")) else {
        return;
    };
    let run = check(dir.path());
    assert_eq!(
        run.error_kinds(),
        ["directory_walk_error"],
        "{}",
        run.stdout
    );
}

/// The count is a lower bound when the second walk -- the one that counts what
/// the exclusions removed -- cannot read part of an excluded subtree, and the
/// message says so rather than presenting a partial count as exact.
#[test]
fn the_excluded_count_is_marked_a_lower_bound_when_it_is_one() {
    let dir = tempdir().unwrap();
    write(&dir.path().join(".hidden/a.ch"), CLEAN);
    fs::create_dir_all(dir.path().join(".hidden/locked")).unwrap();
    write(&dir.path().join(".hidden/locked/b.ch"), CLEAN);
    let exact = check(dir.path());
    assert!(
        exact.error_messages()[0].contains(" 2 excluded"),
        "a readable excluded subtree counts exactly: {}",
        exact.stdout
    );

    let Some(_guard) = Unreadable::new(&dir.path().join(".hidden/locked")) else {
        return;
    };
    let run = check(dir.path());
    assert_eq!(run.error_kinds(), ["empty_corpus"], "{}", run.stdout);
    assert!(
        run.error_messages()[0].contains("at least 1 excluded"),
        "{}",
        run.stdout
    );
    assert_eq!(run.code, Some(2));
}

// ---------------------------------------------------------------------------
// [04-FIT-25]: exit 0 iff every error list is empty, otherwise 2.
// ---------------------------------------------------------------------------

#[test]
fn a_clean_corpus_exits_zero_with_no_errors_anywhere() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    write(&dir.path().join("sub/b.ch"), CLEAN);
    let run = check(dir.path());
    assert!(run.error_kinds().is_empty(), "{}", run.stdout);
    for name in ["a.ch", "sub/b.ch"] {
        assert_eq!(run.entry(name)["report"]["errors"], serde_json::json!([]));
    }
    assert_eq!(run.code, Some(0));
}

/// An error inside one report fails the envelope even though the envelope's
/// own `errors` is empty: the rule reads every list.
#[test]
fn an_error_in_one_report_exits_two() {
    let dir = tempdir().unwrap();
    write(&dir.path().join("a.ch"), CLEAN);
    write(&dir.path().join("b.ch"), BROKEN);
    let run = check(dir.path());
    assert!(run.error_kinds().is_empty(), "{}", run.stdout);
    assert_eq!(run.code, Some(2));
}

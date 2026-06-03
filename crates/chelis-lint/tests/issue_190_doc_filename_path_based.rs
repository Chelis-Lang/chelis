//! Regression tests for chelis#190.
//!
//! `doc-filename-convention` previously classified any `.md` under
//! `docs/` into one of two disjoint slots based solely on `book.toml`
//! ancestry: snake_case (§8.3) when no `book.toml` ancestor existed,
//! kebab-case (§8.5) when one did. The discriminator was retroactive:
//! dropping a `book.toml` into `docs/` flipped every narrative
//! `docs/foo_bar.md` from accepted to rejected (and vice versa when
//! the file was removed), even though no doc filename had changed.
//!
//! The fix moves §8.5 to a path-based opt-in: the rule applies only
//! to paths under a directory literally named `book/` (most commonly
//! `docs/book/` in the chelis layout). §8.3 applies to everything
//! else under `docs/`. The opt-in is the path itself, not ancestor
//! file inference. Adding or removing `book.toml` cannot flip any
//! verdict.
//!
//! These tests pin:
//!   * `docs/foo_bar.md` (snake) is accepted whether or not a
//!     `book.toml` exists at `.` or at `docs/`;
//!   * `docs/foo-bar.md` (kebab) is rejected whether or not a
//!     `book.toml` exists at `.` or at `docs/`;
//!   * paths under `docs/book/` enforce kebab-case;
//!   * paths under a top-level `book/` enforce kebab-case;
//!   * `SUMMARY.md` and `README.md` remain exempt under `book/`.

use chelis_lint::rules::doc_filename_convention::DocFilenameConvention;
use chelis_lint::{Context, Rule, Surface, Violation};
use std::fs;
use std::path::Path;

fn check(repo: &Path, target: &Path) -> Vec<Violation> {
    let ctx = Context {
        root: repo,
        path: target,
        source: None,
        surface: Surface::DocFile,
    };
    DocFilenameConvention.check(&ctx)
}

fn write_empty(path: &Path) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir parent");
    }
    fs::write(path, "").expect("write empty file");
}

fn write_book_toml(dir: &Path) {
    fs::create_dir_all(dir).expect("mkdir for book.toml");
    fs::write(
        dir.join("book.toml"),
        "[book]\ntitle = \"x\"\nlanguage = \"en\"\n",
    )
    .expect("write book.toml");
}

#[test]
fn snake_docs_accepted_without_any_book_toml() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let target = repo.join("docs").join("foo_bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert!(
        v.is_empty(),
        "snake_case docs/foo_bar.md should pass without book.toml; got: {v:?}"
    );
}

#[test]
fn snake_docs_still_accepted_when_book_toml_added_at_repo_root() {
    // Invariant: dropping book.toml into the repo root does not flip
    // the verdict for docs/foo_bar.md.
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    write_book_toml(repo);
    let target = repo.join("docs").join("foo_bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert!(
        v.is_empty(),
        "snake_case docs/foo_bar.md must still pass with book.toml at repo root; got: {v:?}"
    );
}

#[test]
fn snake_docs_still_accepted_when_book_toml_added_at_docs_root() {
    // Invariant: dropping book.toml directly into docs/ does not flip
    // the verdict for docs/foo_bar.md. This is the exact retroactive
    // failure mode reported in issue #190.
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    write_book_toml(&repo.join("docs"));
    let target = repo.join("docs").join("foo_bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert!(
        v.is_empty(),
        "snake_case docs/foo_bar.md must still pass with book.toml at docs/ root; got: {v:?}"
    );
}

#[test]
fn kebab_docs_rejected_without_any_book_toml() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let target = repo.join("docs").join("foo-bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert_eq!(
        v.len(),
        1,
        "kebab-case docs/foo-bar.md should be rejected; got: {v:?}"
    );
    assert_eq!(v[0].spec_ref, "§8.3");
}

#[test]
fn kebab_docs_still_rejected_when_book_toml_added_at_repo_root() {
    // Invariant: dropping book.toml at repo root does not retroactively
    // accept kebab-case docs/foo-bar.md.
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    write_book_toml(repo);
    let target = repo.join("docs").join("foo-bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert_eq!(
        v.len(),
        1,
        "kebab-case docs/foo-bar.md must still be rejected with book.toml at repo root; got: {v:?}"
    );
    assert_eq!(v[0].spec_ref, "§8.3");
}

#[test]
fn kebab_docs_still_rejected_when_book_toml_added_at_docs_root() {
    // Invariant: dropping book.toml at docs/ does not retroactively
    // accept kebab-case docs/foo-bar.md. This is the inverse of the
    // retroactive-flip footgun.
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    write_book_toml(&repo.join("docs"));
    let target = repo.join("docs").join("foo-bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert_eq!(
        v.len(),
        1,
        "kebab-case docs/foo-bar.md must still be rejected with book.toml at docs/ root; got: {v:?}"
    );
    assert_eq!(v[0].spec_ref, "§8.3");
}

#[test]
fn kebab_accepted_under_docs_book() {
    // §8.5: paths under docs/book/ are mdBook chapters; kebab-case.
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let target = repo.join("docs").join("book").join("foo-bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert!(
        v.is_empty(),
        "kebab docs/book/foo-bar.md should pass under §8.5; got: {v:?}"
    );
}

#[test]
fn kebab_accepted_under_docs_book_src() {
    // §8.5: the legacy chelis layout docs/book/src/ is still kebab.
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let target = repo
        .join("docs")
        .join("book")
        .join("src")
        .join("first-program.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert!(
        v.is_empty(),
        "kebab docs/book/src/first-program.md should pass under §8.5; got: {v:?}"
    );
}

#[test]
fn kebab_accepted_under_top_level_book() {
    // §8.5: top-level book/ (no docs/ prefix) is also a recognized
    // mdBook opt-in path.
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let target = repo.join("book").join("foo-bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert!(
        v.is_empty(),
        "kebab book/foo-bar.md should pass under §8.5; got: {v:?}"
    );
}

#[test]
fn snake_rejected_under_docs_book() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let target = repo.join("docs").join("book").join("foo_bar.md");
    write_empty(&target);

    let v = check(repo, &target);
    assert_eq!(
        v.len(),
        1,
        "snake docs/book/foo_bar.md must be rejected under §8.5; got: {v:?}"
    );
    assert_eq!(v[0].spec_ref, "§8.5");
}

#[test]
fn mdbook_summary_and_readme_remain_exempt_under_docs_book() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let repo = tmp.path();
    let summary = repo.join("docs").join("book").join("SUMMARY.md");
    let readme = repo.join("docs").join("book").join("README.md");
    write_empty(&summary);
    write_empty(&readme);

    assert!(check(repo, &summary).is_empty());
    assert!(check(repo, &readme).is_empty());
}

#[test]
fn book_toml_position_does_not_change_verdict_for_kebab_under_docs_book() {
    // Sanity: regardless of whether book.toml sits at docs/, docs/book/,
    // or doesn't exist at all, docs/book/foo-bar.md must remain kebab-OK.
    for book_toml_dir in [None, Some("docs"), Some("docs/book")] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path();
        if let Some(rel) = book_toml_dir {
            write_book_toml(&repo.join(rel));
        }
        let target = repo.join("docs").join("book").join("foo-bar.md");
        write_empty(&target);

        let v = check(repo, &target);
        assert!(
            v.is_empty(),
            "docs/book/foo-bar.md should pass regardless of book.toml position {book_toml_dir:?}; got: {v:?}"
        );
    }
}

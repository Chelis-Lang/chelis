# Source guard directory-read correctness addendum

**Verdict: real fail-open gap in the source guard's stated PR-time scope.** Current `origin/main` is `5e2e8630db704a0e9ac64d384ce39d16f763ad2e`. `guarded_sources` visits five named source roots, but `collect_rust_files` returns success-shaped empty output on `fs::read_dir` failure and drops `ReadDir` entry errors with `entries.flatten()` (`crates/chelis-compiler-api/src/source_arch.rs:2712-2745`). `actual_workspace_findings` parses *only* enumerated files, and the production unit passes on an empty findings vector (`:2781-2801,4781-4791`). A selected file's `read_to_string` failure does panic; the directory-level omissions do not.

**Executable probe.** I extracted the exact `origin/main` root list and `guarded_sources`/`collect_rust_files` functions into a temporary standalone Rust program, compiled with `rustc --edition=2024` (no Cargo build), and invoked it on a fixture under `target/fast-gt10-audit/source-guard-probe/`. Five roots held benign `probe.rs`; `chelis-reef/src/hidden/duplicate.rs` held the repo red fixture's checker → effects → linearity sequence. Run `.venv/bin/python target/fast-gt10-audit/source_guard_probe.py`; captured output is `target/fast-gt10-audit/source-guard-probe-output.txt`:

| Fixture state | Selected `.rs` | Duplicate selected |
|---|---:|---|
| All readable | 6 | yes |
| Reef root missing | 4 | no |
| Reef root mode `000` | 4 | no |
| Nested `hidden/` mode `000` | 5 | no |
| Permissions restored | 6 | yes |

The probe verified `PermissionError` and restored permissions. It executed the production *enumerator*, not the full `syn` inspector. The full inspector's existing nested Reef fixture plants this exact sequence and asserts a finding (`source_arch.rs:2833-2862`). Therefore omission removes a known finding from the production guard's input; the remaining benign files yield no pipeline finding, so its final empty-findings assertion can pass falsely. `entries.flatten()` has the same fail-open mechanism for per-entry errors, though this probe did not inject one.

**Operational boundary.** A readable PR is unaffected. A missing compiled crate root may make Cargo fail first; this does **not** show overall green CI. The unreadable nested file need not be a Rust module: the guard deliberately scans such `.rs` files, while Cargo need not compile them. That makes the nested case a correctness hole in the guard's own all-scoped-sources assertion; its occurrence in hosted CI remains unmeasured. Make root, nested, entry and metadata read errors fail the unit. Test missing roots and injected nested/entry errors, retaining a positive unreferenced-source control.

**Worktree:** no tracked edits; all probe output is under the assigned ignored `target/` directory.

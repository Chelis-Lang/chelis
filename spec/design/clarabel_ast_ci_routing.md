# Clarabel validation for Surf AST representation changes

## Purpose

The ideal QP property lowering in `chelis-prove` consumes
`chelis_surf::ast::Expr`. Its `qp_ideal` module is compiled only when both
`clarabel-provider` and `smt` are enabled. A change to the shared Surf AST can
therefore break this module while default-feature checks still compile. A
change that adds `Drop` to `Expr`, for example, can make an existing move out
of an `Expr` variant illegal.

The CI path selector must run Clarabel feature validation for changes to the
Surf AST representation. It must continue to skip that validation for paths
that neither own Clarabel nor change the representation it consumes. This is a
CI dependency rule; it does not alter the Clarabel runtime or proof contract
in [Clarabel as an external optimizer](clarabel_external_optimizer.md).

## Selection rule

The existing `clarabel_changed` output in `scripts/ci_detect_docs_only.py`
selects the `ci.yml` `clarabel-provider` job. Extend its owner set with:

| Changed path | Match | Reason |
|---|---|---|
| `crates/chelis-surf/src/ast.rs` | Exact | Defines the shared Surf AST consumed by the ideal QP lowering. |
| `crates/chelis-surf/src/ast/` | Directory prefix | Covers AST definitions if the module is split into files. |

Use repository-relative paths and a slash-terminated directory prefix, so
similarly named files do not match. Keep the existing Clarabel package, proof,
and native-call owners in the same selector. A candidate touching any owner
selects the job, even when it does not edit a Clarabel-owned file. An empty or
unreadable changed-path set retains the existing conservative selection.

Do not route the whole `chelis-surf` crate. Changes confined to its parser,
formatter, resugaring, tests, or other files outside the AST owner paths do
not select this job unless another existing owner path also changed. A move of
AST definitions outside the paths above must update the selector and its
tests in the same change.

The existing workflow conditions remain: candidate preflight must succeed;
documentation-only candidates and the accepted docs rebase lane skip the
feature job; non-PR main pushes use their existing cadence. On a selected PR,
`clarabel-provider` runs beside `ci-fast`, and the required Integration
context incorporates its result. The scheduled `smt-full-prove.yml` coverage
continues independently. This design adds no job to unrelated PRs and no
serial dependency ahead of `ci-fast`.

## Implementation and regression checks

1. Add the exact file and module-directory prefix to the selector's reviewed
   owner sets. Keep the selector's output name and the workflow's job
   condition unchanged.
2. Extend `scripts/test_ci_detect_docs_only.py` with positive cases for the
   exact AST file, a file under `ast/`, and an AST file mixed with an unrelated
   path. Add negative cases for `parser.rs`, `resugar.rs`, a similarly named
   `ast.rs.bak`, and an unrelated crate path. Retain the empty-input positive
   case and the existing Clarabel-owner positives.
3. Verify the workflow still consumes `clarabel_changed` and that the
   required Integration context reads the selected job result. Update the
   `clarabel-provider` cadence row in `docs/ci_validation.md` to name Surf AST
   representation changes.

Run `.venv/bin/python scripts/test_ci_detect_docs_only.py` for the selector
cases, then `python3 scripts/gate.py --fast` before pushing the implementation.
An AST-only changed-path probe must produce `clarabel_changed=true`; a
parser-only or unrelated-path probe must produce `clarabel_changed=false`.
The implementation PR must also complete its selected
`Clarabel Feature Checks (Linux)` job on the exact candidate. Its own selector
edit selects that job independently of the AST rule, so the AST-only probe is
the evidence for the new route. The feature job's
provider-plus-SMT Clippy step is the compile witness for `qp_ideal`; its
adapter and integration steps retain their existing coverage.

This routing increases required CI time for PRs that edit the Surf AST
representation. It does not increase Clarabel work for unrelated PRs. A
solver-dependency download failure is a failed selected job to diagnose or
retry on the same candidate, not a reason to weaken the owner set.

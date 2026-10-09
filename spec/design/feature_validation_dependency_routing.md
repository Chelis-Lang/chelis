# Feature validation dependencies for shared source changes

## Purpose and boundary

Default-feature builds do not compile every feature-gated consumer of a shared
source representation. The Surf AST provides a concrete example:
`chelis-prove` consumes `chelis_surf::ast::Expr` in its `qp_ideal` module, but
that module is compiled only with both `clarabel-provider` and `smt`. A change
to `Expr` can therefore break the ideal QP lowering while default and
SMT-only builds compile.

The CI router records **validation dependencies**: a changed source path
requires a named, already defined feature job. It does not claim to infer
every affected consumer from a path. Cargo's reverse package dependency graph
identifies packages that can consume `chelis-surf`, but it does not identify
which feature combinations compile a particular consumer. Running every
optional configuration for every AST edit would make ordinary PR CI needlessly
expensive. The full configuration matrix keeps its scheduled owner in
`smt-full-prove.yml` and the other nightly workflows.

This document concerns CI selection. It changes neither Surf semantics nor
the Clarabel runtime or proof contract in
[Clarabel as an external optimizer](clarabel_external_optimizer.md).

## Routing model

Keep the reviewed path-to-package obligations in
`.config/ci-test-targets.toml`. Add a generic `required_feature_job_rule` row
kind in the same versioned manifest. Each row names one exact repository path
or slash-terminated directory prefix, a nonempty set of existing PR feature
job identities, and a reason for the dependency. Several changed paths may
select the same job; run it once. A path may also match an existing
`required_package_rule`: the two obligations are additive, not competing
dispositions. The feature rule never substitutes for the package tests.

Only registered jobs with a fixed feature configuration and a required PR
result may be named. The parser rejects unknown jobs, malformed paths,
duplicate or conflicting rules, and stale paths. A directory rule is added
when tracked source exists below it, rather than preauthoring a dormant rule.
The candidate detector evaluates the rules with the exact changed-path set;
both sides of a rename or move count. An empty changed-path set selects every
registered per-PR feature job. A diff or manifest that cannot be read or
validated fails candidate preflight. Neither case silently becomes an
unselected feature set.

The rule evaluator returns selected job identities. The existing
`clarabel_changed` workflow output becomes a projection of that set, rather
than a second hard-coded owner list. The `ci.yml` `clarabel-provider` job and
the required Integration context continue to consume that output. A new
feature owner adds its own static job and required-context wiring, then uses
the same rule evaluator; a row cannot name a job whose result is ignored by
the required context. The job runs beside `ci-fast`, with no serial dependency
ahead of it. CI-policy and rule-manifest edits trigger the existing
workflow-native contract preflight, including tests of the selected job's
feature command and required-context wiring.

For trusted targeted rebases, selection uses the verifier's complete
synthetic-candidate delta. If the targeted lane cannot execute a selected
feature owner on that delta, it falls back to full CI instead of reusing a
receipt that lacks the obligation. The ordinary PR selector and targeted
rebase selector must agree on the same rule table. The selected jobs and
rule-table digest enter the candidate plan and receipt, so evidence from a
different routing table cannot authorize reuse.

## Surf AST example

The exact path `crates/chelis-surf/src/ast.rs` adds the existing
`ci.yml:clarabel-provider` feature job to its obligations. That job compiles
the provider-plus-SMT configuration containing `qp_ideal`, and runs its
adapter and integration checks. The existing `required_package_rule` for the
same path still makes the `chelis-types` integration owner set required.
Default and SMT feature checks already run on code PRs. No other optional
feature job is selected for this path without its own reviewed dependency
rule. The coverage for an AST-only PR is:

| Obligation | Source |
|---|---|
| Default build and focused SMT check | Existing every-code-PR jobs |
| `chelis-types` integration owner set | Existing `required_package_rule` |
| Clarabel provider-plus-SMT build and tests | New `required_feature_job_rule` |
| Other optional feature configurations | Scheduled matrix, unless another reviewed PR rule applies |

A split into `crates/chelis-surf/src/ast/` adds a directory rule in the same
change that moves the definitions; the candidate's old and new paths both
participate in selection. Changes confined to `parser.rs`, `resugar.rs`,
other Surf files, or unrelated crates do not select the Clarabel job unless
they match another reviewed rule. A similarly named `ast.rs.bak` does not
match the exact AST path.

## Implementation and acceptance

1. Add the strict feature-rule schema and parser to the existing ownership
   manifest and version its plan representation. Put bootstrap-light rule
   parsing in a shared standard-library module consumed by candidate
   detection and the change-owned/targeted-rebase
   planner; route that new module through the CI-contract preflight. Replace
   the hard-coded `CLARABEL_PATHS` and `CLARABEL_PREFIXES` with the table's
   selected-job projection; retain the existing `clarabel_changed` output
   and job wiring.
2. Add the Surf AST rule. Keep its package rule intact. Validate that the
   selected job exists, runs the provider-plus-SMT Clippy command, and is
   required by Integration when selected. Update the Clarabel cadence row in
   `docs/ci_validation.md` to include the shared AST dependency.
3. Add selector tests before implementation: AST-only and AST-plus-unrelated
   changes select Clarabel; parser-only, resugaring-only, similarly named,
   and unrelated changes do not; repeated matching paths select the job once.
   Reject an unknown job, an untracked path, and a rule or workflow edit that
   removes required-context wiring. Include a rename/move whose old AST path
   must still select the job. Test the targeted-rebase projection against the
   ordinary PR projection.

Run `.venv/bin/python scripts/test_ci_detect_docs_only.py` and the ownership
planner's focused tests, then `python3 scripts/gate.py --fast` before the
implementation push. An AST-only changed-path probe must emit
`clarabel_changed=true`; parser-only and unrelated probes must emit `false`.
The implementation PR also runs the selected `Clarabel Feature Checks
(Linux)` job on its exact candidate, but its own CI-policy edits may select
that job independently, so that run alone cannot prove the AST edge. The
provider-plus-SMT Clippy step is the compile witness for `qp_ideal`.

Only PRs that match a reviewed feature dependency gain its job. AST-changing
PRs may take longer because the Clarabel feature job becomes required;
unrelated PRs retain their current fast path. A solver-dependency download
failure is diagnosed or retried on the same candidate, not treated as a
reason to remove the dependency rule.

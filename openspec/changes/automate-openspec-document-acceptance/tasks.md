## 1. Add executable contracts

- [x] 1.1 Add positive tests for every OpenSpec document path that may land automatically.
- [x] 1.2 Add negative tests for code paths, mixed change sets, and acceptance-policy paths.
- [x] 1.3 Add negative tests for symlink, executable, submodule, and directory file modes.
- [x] 1.4 Add negative tests for type-change, copy, unmerged, and unknown diff statuses.
- [x] 1.5 Add negative tests for specification deletion and for renames that cross the document boundary.
- [x] 1.6 Add negative tests for an empty change set, an unreadable record, and a combined merge diff.
- [x] 1.7 Add tests proving an invalid revision never reaches Git and a Git failure stays operational.
- [x] 1.8 Add workflow contract tests, each with the mutation that proves it can fail.

## 2. Implement the acceptance boundary

- [x] 2.1 Add `scripts/openspec_acceptance.py` with the document classification rules.
- [x] 2.2 Parse `git diff --raw -z -M` records, raising rather than skipping an unreadable one.
- [x] 2.3 Refuse every file mode that is not a regular file, on both sides of each record.
- [x] 2.4 Exclude `openspec/config.yaml`, the workflow, the classifier, the submission command, and the consumer checker.
- [x] 2.5 Emit `verdict` and `reason` to the GitHub step output and to stdout.
- [x] 2.6 Separate the classified, review-required, and operational exit codes.

## 3. Implement the submission command

- [x] 3.1 Add `scripts/openspec_submit.py` with a plan built before any action.
- [x] 3.2 Classify tracked changes and untracked files through the same classifier.
- [x] 3.3 Run strict structural validation and separate findings from operational failures.
- [x] 3.4 Create the branch, commit, push, and open the pull request.
- [x] 3.5 State the limits of the evidence in the pull request body.
- [x] 3.6 Add `--dry-run`, which prints the plan and creates nothing.

## 4. Wire the hosted path

- [x] 4.1 Add the `openspec-autoland` workflow with a read-only boundary job and a merge job.
- [x] 4.2 Run it on `pull_request_target` so it executes the base branch's copy.
- [x] 4.3 Confine the write permission to the merge job, and re-derive the verdict there from base code.
- [x] 4.4 Keep pull-request content out of the working tree: one base checkout per job, head fetched as objects.
- [x] 4.5 Record that the workflow must not become a required status check.
- [x] 4.6 Add `openspec-autoland-validate.yml`: strict enforce-mode validation of the head, read-only, published as a named check.

## 5. Record the policy in its owning locations

- [x] 5.1 Record the maintainer authorization in `spec/design/spec_provenance.md`.
- [x] 5.2 Describe the submission command and the acceptance boundary in `AGENTS.md`.
- [x] 5.3 Capture the requirements as a delta on the `openspec-validation` capability.

## 6. Validate

- [x] 6.1 Run `scripts/test_openspec_acceptance.py`, `scripts/test_openspec_submit.py`, and `scripts/test_openspec_autoland_workflow.py`.
- [x] 6.2 Run `scripts/test_openspec_validation.py` and confirm the advisory workflow is unchanged.
- [x] 6.3 Run `openspec validate --all --strict --no-interactive`.
- [x] 6.4 Run adversarial validation in a fresh context against the boundary.
- [ ] 6.5 Hosted acceptance oracle for section 9: a document submission merges on the hosted repository with no human approval, without any code path becoming automatically mergeable.

## 7. Repair the adversarial findings

- [x] 7.1 Classify the index as well as the working tree, and commit only the classified paths.
- [x] 7.2 Pass classified paths to Git as literal filenames rather than pathspec patterns.
- [x] 7.3 Refuse a working tree that holds an unresolved merge.
- [x] 7.4 Start every comparison at the merge base instead of the base tip.
- [x] 7.5 Write only a closed vocabulary to the machine-readable step output, and escape reported paths.
- [x] 7.6 Remove the path filter. (The revocation job added here was removed by 8.4.)
- [x] 7.7 Add the workflow mutations the first contract test failed to catch, including `--admin`, a dropped `--auto`, a self-diff re-derivation, and an extra command in a write-permission job.
- [x] 7.8 Pin each path-safety rule and each status/mode pair with a case only that rule rejects.

## 8. Repair the second review's findings

- [x] 8.1 Move the workflow to `pull_request_target`: a `pull_request` workflow is supplied by the change it judges, so a privileged one can delete its own boundary job.
- [x] 8.2 Remove every write permission, merge, approval, and comment from the workflow.
- [x] 8.3 Remove the auto-merge request from the submission command, and the option that selected its merge method.
- [x] 8.4 Remove the revocation job: withdrawing a standing grant races the merge it is meant to prevent.
- [x] 8.5 Add the governance-identity check over `.github`, `scripts`, and `openspec/config.yaml`.
- [x] 8.6 Keep pull-request content out of the working tree in the `pull_request_target` job.
- [x] 8.7 Correct the branch-protection guidance: do not propose lowering the review requirement for code.
- [x] 8.8 Record the measured hosted state and the merge design's real prerequisites, including that `CODEOWNERS` cannot name a GitHub App and `GITHUB_TOKEN` cannot approve.
- [x] 8.9 Update the tests before the fixes, and add the mutations the earlier workflow test did not catch.

## 9. Implement automatic merging

- [x] 9.1 Add `scripts/openspec_merge.py` and its tests, written first against a fake API runner.
- [x] 9.2 Discover required contexts from branch protection; refuse an empty or unreadable set.
- [x] 9.3 Resolve each context by its most recent result, matching the app protection names.
- [x] 9.4 Treat missing, pending, queued, in-progress, neutral, cancelled, and failed as not passing. (`skipped` was corrected to passing by 12.9.)
- [x] 9.5 Require the strict validation context in addition to protection's set.
- [x] 9.6 Poll with a bounded timeout and print a re-run instruction on expiry.
- [x] 9.7 Re-read the pull request each round and immediately before merging; refuse a changed state.
- [x] 9.8 Bind the merge to the exact head commit and use no auto-merge, administrative, or approval path.
- [x] 9.9 Report a protection refusal as blocked rather than escalating.
- [x] 9.10 Verify the read and decision paths against the live API, read-only.

## 10. Repair the third review's findings

- [x] 10.1 Write the failing test for each finding before its fix, positive and negative, and prove every fix with a mutation only that test kills.
- [x] 10.2 Discover required checks through `GET .../branches/{branch}` (`Contents: read`); the protection endpoint needs `Administration`, which no `permissions:` block grants, so the worker could never have merged.
- [x] 10.3 Refuse an unprotected base branch by name, and name the permission each attempted source needs when discovery fails outright.
- [x] 10.4 Bind the merge to the authorized base ref, and accept a moved base sha only where it moved forward.
- [x] 10.5 Satisfy an app-pinned context only from that app; a commit status carries no app and never satisfies one.
- [x] 10.6 Pin the strict validation requirement to a successful `pull_request` run of the validation workflow FILE on the exact head sha, and grant the merge job `actions: read` for it.
- [x] 10.7 Raise rather than truncate a paged read, and request full pages on the combined-status endpoint.
- [x] 10.8 Replace the timeout re-run instruction: the workflow declares no `workflow_dispatch`, so `gh workflow run` could not work.
- [x] 10.9 Re-read a bounded number of times when GitHub has not finished computing mergeability.
- [x] 10.10 Refuse a rename that moves `openspec/project.md` out of its slot, as deletion already was.
- [x] 10.11 Stop classifying untracked strays outside the document set; the tree holds 37 of them and every submission was refused.
- [x] 10.12 Correct D9's inverted measurement: `base.sha` is ahead of the merge base, never behind it.
- [ ] 10.13 Hosted confirmation that the required-check discovery, the workflow-run pin, and the compare read all succeed under the Actions token. Verified read-only under a maintainer token only, which is a different principal.

## 11. Make the command usable

- [x] 11.1 Expose `openspec-submit` as a Devenv script, not shadowing the upstream `openspec` CLI.
- [x] 11.2 Report one line per stage (prepare, validate, submit, wait) and one final outcome.
- [x] 11.3 Wait for the real outcome by default; never report an opened pull request as acceptance.
- [x] 11.4 Add `--no-wait` (queued, with the URL) and `--watch NUMBER` (resume waiting).
- [x] 11.5 Give each stop its own summary and remedy instead of one "human review required".
- [x] 11.6 Report a validation finding as validation, never as approval.
- [x] 11.7 Preflight tools and `gh` authentication before any write.
- [x] 11.8 Reuse an existing submission for the same content instead of opening a second one.
- [x] 11.9 Make `--dry-run` write nothing and state the read-only fetch it performed.
- [x] 11.10 Document the command and its current activation blockers in `README.md`.

## 12. Make a plain push the entry point

- [x] 12.1 Write the controller tests first, positive and negative, against a fake API.
- [x] 12.2 Add `openspec-autoland-signal.yml`: `push`, `branches-ignore: [main]`, `permissions: {}`, no token, no action, no artifact.
- [x] 12.3 Add `openspec-autoland-controller.yml` on `workflow_run`, running default-branch code with `pull-requests: write` and `contents: read`.
- [x] 12.4 Add `scripts/openspec_controller.py`: re-derive event, repository, branch, and commit from the API; refuse a non-push event, a foreign repository, the default branch, a deleted branch, and a moved branch.
- [x] 12.5 Classify the exact pushed commit with default-branch code before any write.
- [x] 12.6 Open one internal pull request per branch and reuse an existing open one.
- [x] 12.7 Read nothing the signal run produced, including its conclusion.
- [x] 12.8 Measure, rather than assume, that a `GITHUB_TOKEN` pull request cannot start the required checks, and lock the measurement with a test over the real workflow files.
- [x] 12.9 Accept `skipped` as a passing required check: measured on merged docs-only pull request #1634, six of nine required contexts were `skipped`, so the previous rule refused every eligible pull request.
- [x] 12.10 Update `README.md` and `AGENTS.md` to the push-first flow, keeping `openspec-submit` as an optional helper.

## 13. Repair the fourth review's findings

- [x] 13.1 Require the strict validation context to have actually RUN: `skipped` satisfies protection's own contexts, but a skipped validation proves nothing about the documents being merged.
- [x] 13.2 Filter the push signal to `openspec/**` so an ordinary code push starts no runner at all.
- [x] 13.3 Fail the controller job only on an operational error; a push that is not a document change is an ordinary answer, reported in the step summary.
- [x] 13.4 Remove the stray mid-file `unittest.main()` that hid both workflow suites from a direct `python test_openspec_controller.py` run.
- [x] 13.5 Type-check API payload fields so a malformed shape is an operational failure, not an uncaught `AttributeError` reported as a refusal.
- [x] 13.6 Match the merge worker's branch validation: reject `..`, a trailing slash, and an unbounded length.
- [x] 13.7 Pin the signal run to the signal workflow FILE; the `workflows:` filter matches a display name any branch can claim.
- [x] 13.8 Assert the hostile branch name never reaches a `gh` argv, not merely that the status was a refusal.
- [x] 13.9 Pin the concurrency group, supersession, and timeout of both new workflows, and the controller's exit-status policy.
- [x] 13.10 Make the evidence test fail rather than pass vacuously when its producers disappear.
- [x] 13.11 Print full shas in the moved-branch refusal, so two commits sharing a short prefix do not read as identical.
- [x] 13.12 Correct the README headline and the controller docstring: the merge is built but not reachable without the credential.

## 14. Wire the submission credential

- [x] 14.1 Write the credential tests first: configured, missing, blank, reuse-without-token, and no-silent-fallback.
- [x] 14.2 Add the `OPENSPEC_SUBMISSION_TOKEN` secret input to the controller workflow, referenced in exactly one step.
- [x] 14.3 Use the credential for the pull-request creation call only; every read keeps the built-in token.
- [x] 14.4 Create the pull request through `POST /repos/{owner}/{repo}/pulls`, whose only fine-grained requirement is `Pull requests: write`, instead of `gh pr create`, which would additionally read repository metadata.
- [x] 14.5 Reduce the built-in token to `pull-requests: read`, so it structurally cannot open a pull request.
- [x] 14.6 Add `actions: read`, required by `GET /repos/{owner}/{repo}/actions/runs/{id}`, which the controller uses to re-derive the push identity.
- [x] 14.7 Report a missing or blank secret as blocked, with the setup recipe, before any write and without falling back.
- [x] 14.8 Check the credential immediately before the create call, so an ineligible push still reports the boundary rather than the secret.
- [x] 14.9 Assert the secret is absent from every head-supplied workflow and present in exactly one step of the trusted one.
- [x] 14.10 Document the one-time activation recipe with the exact least-privilege permission, and state plainly why a GitHub App installation token is not what this uses.

## Acceptance oracle

The authoritative completion oracle is task 6.5: one document submission merges on the hosted repository with no human approval, and no code path becomes automatically mergeable.

Until it runs, the local evidence is task 6.1's suites plus the live read-only verification in 9.10: required-context discovery returned all nine contexts from real branch protection, evaluation of a real head sha returned zero failures with `Changelog` correctly resolved to its latest run rather than an earlier cancellation, and a commit missing one context failed closed naming it. The merge call itself has not been executed.

Section 10 narrows what that evidence is worth. The 9.10 reads ran under a maintainer token holding the `repo` scope on an administered repository, which is a **different principal** from `GITHUB_TOKEN`. One of those reads -- branch protection -- is documented as needing `Administration`, a permission no workflow can be granted, so the original worker would have failed on its first substantive call. The endpoints it uses now (`branches/{branch}`, `compare`, `actions/workflows/{file}/runs`) are documented as `Contents: read`, `Contents: read`, and `Actions: read`, all grantable; each was verified read-only to return the data the worker needs. That the Actions token can reach them remains unproven here and is task 10.13.

## Context

Automatic acceptance moves a merge decision from a person to a program. The design question is not "how do we merge without review" -- `gh pr merge` already does that -- but "what makes it safe to hand a write token to a rule". Three facts shape every decision below.

First, a pull request controls its own content. Any check that runs code from the pull request head can be rewritten by that pull request. Second, GitHub's write token is job-scoped, so a workflow can hold both an untrusted step and a privileged step, and the boundary between them is the only thing preventing escalation. Third, a classifier that guesses is worse than no classifier: a wrong "review required" costs a click, a wrong "automatic" merges unreviewed code into `main`.

The existing repository controls that this design reuses: `scripts/ci_detect_docs_only.py` established the fail-closed diff-classification pattern, `scripts/check_openspec.py` established the finding/operational exit split, and the pinned `Chelis-Lang/ci` actions supply the locked OpenSpec runtime.

## Goals / Non-Goals

Goals:

- One command submits an OpenSpec document change, with the local verdict matching the one CI reports.
- The boundary between documents and everything else is decided by trusted code over trusted data, on a trigger the change cannot control.
- Normative `openspec/specs/**` text is inside that boundary, per explicit maintainer authorization.
- Every unclassifiable input fails toward human review.
- Branch protection, required checks, and the merge record stay exactly as they are for every other change.

Non-Goals:

- **Approving anything.** The worker never submits a review. It merges on the strength of the required checks, which is all `main` currently gates on.
- **Requesting auto-merge**, in any form. See D1.
- **Changing any repository or branch-protection setting.**
- Judging whether an OpenSpec document is correct, wanted, or consistent with `spec/**`. Structural validation does not do this and this change does not claim it.
- Automating implementation code, workflow, script, or configuration changes.
- Reducing the review requirement on any code path, now or as an activation step.
- Activating the Phase 0 provenance regime in `spec/design/spec_provenance.md`.
- Making autoland a required status check.

## Decisions

### D1: An immediate merge bound to one commit, with no standing grant

The merge worker polls the checks branch protection actually requires, for the exact head commit, then performs one ordinary merge bound to that commit. It never enables auto-merge, never uses an administrative option, never approves, and changes no setting.

This works on the current configuration without touching anything, and the reason is worth stating precisely: `main` requires **zero** approving reviews, so the only gate is its nine required status checks. An immediate merge is therefore exactly the call a person makes after seeing those checks go green. Nothing is bypassed and nothing is relaxed.

It also degrades correctly. If an approval requirement is ever added, the merge API refuses, the worker reports the pull request as blocked, and the change waits for a human. That is the desired failure: protection tightening must stop the robot, not be worked around by it.

The rejected alternative was auto-merge. It is a standing grant on a mutable branch -- GitHub keeps it enabled across later pushes by anyone with write access -- so a decision made on one commit would authorize every commit after it. A revocation job cannot fix that, because it races the merge it is trying to prevent.

### D14: Why the REST merge endpoint rather than `gh pr merge`

`gh pr merge` documents a fallback: "If required checks have not yet passed, auto-merge will be enabled." That is the standing grant D1 refuses, and it would be reached by a race rather than by a decision.

`PUT /repos/{owner}/{repo}/pulls/{n}/merge` has no such path. Its `sha` parameter binds the merge to one commit exactly as `--match-head-commit` does, and it returns a documented status for every outcome: `409` when the head moved, `405` when protection says no. Same semantics, strictly fewer ways to end up somewhere unintended.

### D2: The trigger must be `pull_request_target`, and this is not a detail

An earlier version of this workflow used `pull_request` and gave a job `contents: write`. The boundary job checked out the classifier from the base revision, and the design argued that this made the verdict trustworthy.

**It did not.** On a `pull_request` event the *workflow file itself* comes from the pull request. A pull request can rewrite `.github/workflows/openspec-autoland.yml` to delete the boundary job, keep the write permission, and do as it likes with the token. Checking out a trusted script does not help when the step that runs it is head-controlled. The classifier's own rule -- "a change to `.github/**` routes to review" -- was being enforced by a job the change had already deleted.

`pull_request_target` runs the base branch's copy of the workflow. That is the only trigger under which any in-workflow check constrains anything. It comes with one absolute rule: **never check out or execute pull-request code, configuration, or actions in a `pull_request_target` job.** So the workflow has exactly one checkout, of the base revision; the pull request enters as fetched Git objects; and nothing outside `base/` is executed.

The general lesson is worth more than the fix: *the trust boundary is the trigger, not the checkout.* Any reasoning of the form "we run a trusted script, so the job is trusted" is wrong on a head-controlled trigger.

### D3: Governance is compared by object identity, not by diff

`--require-identical-governance` compares Git object ids between head and base for `.github`, `scripts`, `openspec/config.yaml`, and the inputs that decide what CI executes inside: `devenv.nix`/`.yaml`/`.lock`, `devenv/`, `flake.nix`/`.lock`, `nix/`, `.cargo/`, `rust-toolchain.toml`, `.gitattributes`, and `.gitmodules`. Empty result means the head carries byte-identical governance content.

A diff answers a weaker question. It starts at a merge base, so a head that merely *predates* a change to `.github/` shows no difference there while actually carrying an older workflow. Verified: with `main` ahead by one workflow commit, the diff-only classifier reports `auto` for a head whose `.github` tree is stale, and the identity check reports `review`.

The Devenv and toolchain entries are there because identical `scripts/` content proves less than it looks like: every real CI step runs through `devenv-retry --profile ci shell`, so the head's Devenv inputs choose the interpreter that identical code runs under. Adversarial validation produced exactly that case -- a head with stale `devenv.nix`, `.cargo/config.toml`, and `rust-toolchain.toml` classifying `auto`. Cargo manifests and lockfiles are deliberately excluded: they select library versions rather than steering CI execution, and they change often enough to refuse most branches.

Two properties, stated precisely so neither is over-read. A path absent on both sides is identical, not different -- counting it otherwise would let one not-yet-existing entry refuse every branch forever. And if *no* governance path resolves at all, that is an operational error, not a pass: a comparison that never happened is not a clean bill of health.

This is what lets the merge worker trust a result produced by a head-side validation workflow: if these trees are identical, `openspec-autoland-validate.yml` *is* the base's workflow, running under the base's declared environment. It is the load-bearing part of D13.

**Its cost is real.** Sampling 25 recent pull requests, 18 had a differing `.github` or `scripts` tree against their base, because those directories churn constantly here. So a branch that has not been rebased recently reports `review` regardless of its content. `openspec_submit.py` applies the same rule locally and names the remedy, so the failure arrives as "rebase onto origin/main" before the push rather than as a confusing verdict after it.

### D4: Strict validation runs on the head, in a job that can never write

`openspec-autoland-validate.yml` runs on `pull_request` in `enforce` mode and publishes the check run `OpenSpec Strict Validation`. The merge worker requires that context by name, in addition to branch protection's own set.

Running it on `pull_request` means the file is supplied by the pull request. That is safe here for exactly two reasons, and it is worth being explicit because the reasoning is the same one that failed in D2 when a *privileged* job relied on it:

1. The job holds no write permission and no secret. A rewritten copy can lie about its own result and nothing else.
2. The merge worker does not take the result on trust. Before merging it verifies, from the base revision, that the head's governance trees are byte-identical (D3). If they are, that workflow file is the base's file. If they are not, the worker refuses and never consults the result.

The difference from D2 is that here the untrusted thing is *data being checked by a trusted checker*, not the checker itself.

Why it cannot simply be a branch-protection required check: a required context that never reports blocks every pull request forever (chelis#419), and this one only runs for changes touching `openspec/`. Requiring it in the worker rather than in protection keeps it mandatory for document merges and absent for everything else.

`openspec-validate` keeps its advisory disposition and is not modified. A finding there is a warning for a human already reading the change; a finding here is the difference between merging and not.

### D5: Fail closed on every diff shape the classifier does not fully understand

`git diff --raw` reports a status letter and both file modes. The classifier accepts exactly `A`, `M`, `D`, and `R` on regular-file mode `100644`, with the absent side `000000`. Everything else routes to review: `T` (type change), `C` (copy), `U` (unmerged), any unknown letter, mode `100755` (executable), `120000` (symlink), `160000` (submodule), and `040000` (tree).

A symlink matters specifically: `openspec/specs/x/spec.md` as a symlink is a path that satisfies every document rule while its content is somewhere the rules never inspected. `CLAUDE.md -> AGENTS.md` in this repository shows the pattern is in live use.

Renames carry both paths, and both must be documents. A rename out of `openspec/specs/**` is a specification removal wearing a move, so it routes to review. A deletion under `openspec/specs/**` routes to review for the same reason, while a deletion under `openspec/changes/**` is ordinary archiving and proceeds.

An empty change set routes to review. Nothing was proven, so nothing is accepted.

### D6: The classified set and the committed set are the same set

The command reads three streams and classifies their union: the working tree, the index, and untracked files rendered as synthetic addition records with the file mode taken from the filesystem. It then commits with `git commit --only -- <paths>`, naming exactly the classified paths as `:(literal)` pathspecs.

An earlier version classified only the working tree and committed with a plain `git commit`. Adversarial validation broke it in one step: stage a change to `scripts/gate.py`, restore the working-tree copy, and edit a document. The working-tree diff is empty for `gate.py`, so the verdict was `auto`, and the commit carried the staged file into an automatically merged pull request. The verified reproduction is in the change record.

Two rules follow, and they are the same rule twice:

1. **Classify everything a commit could pick up**, not the most convenient stream. The index is a commit input; a check that ignores it is checking the wrong object.
2. **Commit exactly what was classified.** `--only` makes the committed set structurally equal to the classified set, so the second rule holds even if the first is later weakened. `:(literal)` is part of it: a file named `openspec/specs/*.md` is a valid path, and as an ordinary pathspec it would expand across the tree.

Nothing is staged until the verdict is `auto` and validation has passed. A tree with an unresolved merge is refused outright, because `git diff` reports a conflicted path as an ordinary modification and the classifier cannot see the conflict.

### D9: Comparisons start at the merge base

The workflow and the command both compare `base...head`, not `base head`. A two-dot range against the tip of `main` also reports every commit `main` gained after the branch started -- changes the pull request did not make.

This is not a corner case here. Sampling 30 recent merged pull requests in this repository, 13 had `github.event.pull_request.base.sha` strictly **ahead** of the merge base, and for one of them two-dot would have attributed 136 extra paths to the change. Two-dot would have refused those document changes for other people's code, and the failure mode is invisible: the feature would simply stop working whenever `main` moved.

An earlier draft of this section stated the relation the other way round, as `base.sha` being *behind* the merge base. That spelling is not merely mismeasured, it is impossible: `merge-base(base, head)` is an ancestor-or-equal of `base` by construction, so `base.sha` can never be strictly behind it. The decision is unchanged and the measurement supports it; only the stated direction was wrong, and a wrong rationale is what a later simplification would reason from.

### D10: No standing merge grant, because a withdrawal is a race

Auto-merge is sticky: GitHub keeps it enabled across later pushes by anyone with write access, so a verdict computed on one commit authorizes every commit after it.

A previous version tried to fix this with a `revoke` job that disabled auto-merge whenever a verdict stopped being `auto`. That is a race, not a fix. Between the push that adds code and the completion of the revoke job, the required checks can go green and GitHub can merge. The withdrawal is competing with the event it is meant to prevent, and it loses whenever the checks happen to finish first.

Nothing in this change requests auto-merge. The local command opens the pull request and stops; the workflow has no write permission at all. A merge design that binds an exact head SHA is in D13; a standing grant on a mutable branch is not an acceptable substitute for it, and no amount of revocation logic makes it one.

### D11: The classifier's step output is a closed vocabulary

A Git filename may contain a newline, and `$GITHUB_OUTPUT` is a `key=value` file whose duplicate keys the runner resolves last-wins. Writing a reason built from a raw path let a pull request append its own `verdict=auto` line: adversarial validation produced exactly that, with a file named `openspec/a\nverdict=auto\nx=y.md` beside `scripts/backdoor.py`.

So the step output carries only `verdict` from a two-word vocabulary and an integer `reason_count`. Human-readable reasons go to stdout with every control character escaped. The general rule: a value derived from repository content is data, and it does not belong in a control channel.

The merge job's independent re-derivation had already blocked the merge in that attack. That is what defence in depth is for, and it is not a reason to leave the first control broken.

### D7: Forks and drafts would be excluded by the merge design

The maintainer authorized their own document changes to land automatically. A fork head is outside that authorization, and a draft is by definition not offered for landing.

Neither exclusion is implemented, because nothing implemented can merge. The reporting workflow runs for fork and draft pull requests and reports a verdict, which is harmless: it holds no write permission. The exclusions belong to D13 and are listed there.

### D8: Autoland is not a required status check

A required check that is filtered out by `paths:` never reports its status context, so branch protection waits forever and no pull request can merge (chelis#419). Autoland is a reporter rather than a gate, so it must stay optional. The workflow says so in its header comment and a test asserts the sentence is present.

### D12: The bot must never be able to land a code change

The requirement is not "merge documents". It is "merge documents **and nothing else**, including when the change is mixed, when governance moved, and when someone pushes after the verdict".

Three properties carry that, and all three must hold at once:

1. **The decision runs on a trusted trigger** (D2). Otherwise the change writes its own verdict.
2. **The decision covers governance by identity** (D3). Otherwise a stale or altered `.github`/`scripts` tree passes.
3. **The decision binds one exact commit.** A verdict authorizes a SHA, never a branch.

Measured state matters here: `main` currently requires **zero** approving reviews, so a merge capability is gated only by the nine required checks. That makes property 3 the load-bearing one, and it makes "just enable auto-merge" indefensible, since auto-merge deliberately outlives the commit it was granted on.

### D13: The merge worker, and the three API facts it encodes

Implemented in `scripts/openspec_merge.py`. Three properties of this repository's real API responses drive it, each measured rather than assumed:

1. **Check runs attach to the pull-request HEAD sha**, not to the test-merge sha. Binding the merge to the head sha therefore binds it to the commit the checks actually ran on. Verified against a real merged pull request: every check run's `head_sha` equalled the pull request head.
2. **One context appears many times on one commit.** `Changelog` was observed with three `success` runs and one `cancelled` run on a single sha. `all()` would refuse a healthy commit and `any()` would accept a broken one; only the most recent run per context is correct, which is what branch protection itself uses. Verified live: the worker resolves `Changelog` to the 22:18:55 success, not the earlier cancellation.
3. **The legacy combined-status endpoint is empty here and reports `state: "pending"`** even for commits whose checks all passed. Reading it as the answer would never merge. It is consulted only for contexts that post real statuses, and its own `state` field is ignored.

The job:

1. resolve the pull request and its head SHA from the triggering run;
2. run the base classifier with `--require-auto --require-identical-governance` for that exact SHA;
3. require the pull request to be open, not draft, not from a fork, and mergeable;
4. read the branch protection's required contexts and require **every** one to be `success` for that exact head SHA -- treating an empty or missing check set as failure, never as "nothing to wait for";
5. merge with the head SHA bound (`gh pr merge --match-head-commit <sha>`, which fails rather than merging a commit that arrived after the verdict);
6. never use `--admin`, `--auto`, or any bypass.

Step 4's failure modes are the ones that bite: no checks reported yet, a context that exists but is `neutral`/`skipped`, a required context whose app id does not match, and a re-run that supersedes an earlier conclusion. Each must be an explicit refusal.

**What is not proven locally.** The worker's read and decision paths were exercised against the live API read-only: required-context discovery returned all nine contexts, evaluation of a real head sha returned zero failures, and a commit missing one context failed closed naming it. The merge call itself has never run under the Actions token.

These files are on `main` (see Migration Plan), so `workflow_run` and `pull_request_target` take effect. Being live is not being proven: the merge call, the token mint, and the check discovery under the Actions token stay unverified until a hosted document push is observed end to end.

**No new credential or setting is needed.**

- `GITHUB_TOKEN` with `contents: write` and `pull-requests: write` on the merge job is sufficient, because `main` requires zero approvals.
- Repository setting `allow_auto_merge` stays `false` and is irrelevant: this design never asks for auto-merge.
- No branch protection change, no GitHub App, no machine user, no `CODEOWNERS` edit.

**If the owner later wants review restored on code.** Nothing here depends on approvals staying at zero; the worker simply reports blocked instead. To keep documents landing while code gains a review requirement: Set `required_approving_review_count: 1`, `require_code_owner_reviews: true`, and `dismiss_stale_reviews: true` on `main`, and give the OpenSpec document paths a code owner that is a dedicated bot identity. Then a document pull request is satisfied by the bot's approval, a code pull request needs a human code owner the bot is not, and a push of code after approval dismisses the approval automatically -- GitHub enforcing the property, not a workflow racing it.

Its real limitations, checked rather than assumed:

- **`CODEOWNERS` cannot name a GitHub App.** It accepts users and teams only. A dedicated machine *user* in a team is required, which is an organization decision with its own account, seat, and credential handling.
- **`GITHUB_TOKEN` cannot supply the approval.** Reviews from the Actions token do not satisfy required reviews, so a separate App or machine-user credential is needed regardless.
- The existing `.github/CODEOWNERS` covers only chelis#729/#730 guard artifacts and owns them to `@rlronan`. It has no `openspec/` entry, and `require_code_owner_reviews` is `false`, so it currently has no effect on merges. Adding `openspec/specs/` and `openspec/changes/` entries is part of activation; `openspec/config.yaml`, `.github/`, and `scripts/` must be left to human owners.
- Turning on `require_code_owner_reviews` affects **every** path with an owner, including the existing guard-artifact rows. That is a repository-wide review change and belongs to the owner, not to this change set.

### D15: Branch protection is read through the branch endpoint, not the protection endpoint

`GET /repos/{owner}/{repo}/branches/{branch}/protection` requires the **`Administration`** permission. A workflow `permissions:` block has no `administration` key -- the accepted keys are `actions`, `artifact-metadata`, `attestations`, `checks`, `code-quality`, `contents`, `deployments`, `discussions`, `id-token`, `issues`, `packages`, `pages`, `pull-requests`, `security-events`, `statuses`, and `vulnerability-alerts` -- so `GITHUB_TOKEN` can never hold it.

This was an outright activation blocker, not a hardening gap. `read_required_checks` called that endpoint first, so the merge job's first substantive API call would have failed with `403 Resource not accessible by integration`, the worker would have exited `2`, and no pull request would ever have merged. The read succeeded during verification only because a maintainer's personal token carries the `repo` scope on a repository they administer. An Actions token is a different principal, and a personal token proves nothing about it.

`GET /repos/{owner}/{repo}/branches/{branch}` requires only **`Contents: read`** and carries the same data. Verified live against `Chelis-Lang/chelis`: its `protection.required_status_checks.checks` returns the same nine `{context, app_id}` entries the dedicated endpoint returns, all pinned to app `15368`. It is now the primary source; the protection endpoint stays as a fallback for a caller running with an admin credential.

Two things this does not change. An unprotected base branch is refused by name rather than falling through to a confusing combined error, because a branch with no protection has no protection to read anywhere. And if both sources fail, the worker refuses with a message naming the exact permission each one needs, so the next reader is not left guessing which grant is missing.

`GET /repos/{owner}/{repo}/rules/branches/{branch}` was considered and rejected: it returns ruleset rules only, and this repository uses classic branch protection. It returns `[]` here.

### D16: A verdict authorizes a base, not a pull request number

The boundary verdict is computed over `base.sha...head.sha` for one base **branch**. The merge worker was given only the head sha, so it re-read the base from the live pull request and used whatever it found.

That is an authorization confusion, and `pull_request_target` makes it reachable: its trigger list is `opened`, `synchronize`, `reopened`, and `ready_for_review`. Retargeting a pull request fires `edited`, which is **not** in that list, so no new verdict is computed. A document pull request could therefore be approved against `main` and then retargeted to a branch it was never judged against, where a squash merge lands `merge-base(new base, head)...head` -- a diff that can contain arbitrary commits `main` has and the new base does not. Stacked pull requests onto non-`main` bases are in live use in this repository, so the retarget is an ordinary operation rather than an exotic one.

The worker now takes `--base-ref` and `--base-sha` from the event and refuses any base it did not decide about:

- A different base **ref** is refused outright, on every poll and immediately before the merge.
- The same ref at a different **sha** is compared with `GET /repos/{owner}/{repo}/compare/{authorized}...{live}`. `ahead` and `identical` are accepted; `behind`, `diverged`, and any unrecognized value are refused. Verified live that the endpoint returns exactly those spellings.

The asymmetry is deliberate. A busy `main` advances constantly and advancing does not change `merge-base(base, head)...head`, so refusing on it would refuse every submission. A rewound or rewritten base does change that set, and those commits were never classified.

### D17: A check-run name is not provenance

Requiring the context `OpenSpec Strict Validation` by name has two gaps. Any Actions workflow can publish a job with that name, and anyone with write access can `POST /repos/{owner}/{repo}/statuses/{sha}` with that context -- a pull request head must be a branch of this repository, so its author has exactly that access.

The name-based requirement also degraded silently. The app id for the extra context was inferred from branch protection's own set and fell back to `None` whenever protection did not agree on a single app. `None` matched anything, so a remote setting change could quietly turn a pinned requirement into an unpinned one.

Two independent repairs, and the requirements are conjunctive so neither weakens the other:

1. **App pinning is now exact.** A context protection pins to an app is satisfied only by a result from that app. A commit status carries no app id at all, so it can never satisfy one. The previous rule skipped the comparison when the observed app was unknown, which is precisely the forged-status case. GitHub itself applies the strict rule, so the old one also meant asking for merges protection would then refuse.
2. **The validation requirement is pinned to a workflow FILE.** `--require-workflow openspec-autoland-validate.yml` requires a `completed`/`success` run of that file, under the `pull_request` event, on the exact head sha. Verified live that `GET /repos/{owner}/{repo}/actions/workflows/{file}/runs?head_sha=...` filters to one workflow file and one commit, and that a `pull_request` run's `head_sha` is the pull request head. Where several runs exist the newest decides, by run id then attempt: a re-run keeps its id and increments the attempt.

This is what D3 was already buying and the worker was not spending. Because the head's `.github` tree is proven byte-identical to the base's, a run of that path **is** the base's workflow, so its conclusion is the base's conclusion. A name proves nothing of the kind.

It costs one read permission, `actions: read`, on the merge job. The job's write grants are unchanged.

### D18: A truncated read is not an answer

`read_check_state` stopped after `MAX_PAGES` and returned what it had. That silently converts "there are more results" into "these are all the results", and the newest run for a context is exactly what could be behind the boundary -- turning a superseded `success` into the verdict. The combined-status read was worse: it requested no page size at all, so it saw the default first thirty statuses. Both now request full pages and raise rather than decide on a partial read.

### D19: Untracked strays are not part of the change set

`openspec_submit.py` classified every untracked file in the tree. The maintainer's working tree holds 37 untracked, unignored non-document paths -- `result`, `.work/`, build output -- so the front-door command refused every document submission with `result: outside the OpenSpec document boundary`.

The refusal proved nothing. An untracked file enters a commit only through an explicit `git add`, and this command adds exactly the classified paths as `:(literal)` pathspecs and then commits with `--only` naming the same list. A stray cannot be in the commit CI classifies, so refusing on it reports a difference that does not exist -- and makes the local verdict disagree with the hosted one, which is the opposite of the goal in D6.

Untracked files **at document paths** are still classified, because those are committed and their filesystem mode is what catches a specification spelled as a symlink. Tracked streams are untouched: the index is a commit input, and a staged code path still stops the submission. That is D6's finding and it stands.

### D15: The command reports outcomes, not transport events

The first version printed a plan and ended with "opened a pull request". That is the wrong last word: it describes the transport, and a contributor reading it has no idea whether their change landed. So the default run waits and ends in one of four words -- ACCEPTED, QUEUED, BLOCKED, FAILED -- with `--no-wait` for the case where the queued URL is genuinely what you want.

Two smaller consequences follow from the same principle.

**Every stop names itself.** A submission can stop for five unrelated reasons: a non-document path, a branch stale against the base, an unresolved merge, a validation finding, or a failed check. They need five different actions, so collapsing them into "human review required" told the contributor nothing. Each now carries its own summary, the specific offending paths, and the exact next command. A validation finding in particular is reported as *validation*, because calling it an approval problem sends the reader to the wrong fix.

**Everything knowable is checked before anything is written.** Missing `gh`, missing `openspec`, and unauthenticated `gh` are all discoverable up front, and discovering them after a branch, a commit, and a push leaves the contributor to clean up. Preflight runs first; `--dry-run` skips only the authentication check, because it writes nothing to authenticate for.

### D16: A plain push is the entry point, and the push side holds nothing

The contributor pushes a branch. That is the whole interface.

The difficulty is that a `push` event runs the workflow file *from the pushed branch*, so anything privileged on that trigger is written by the person being judged -- the same mistake D2 corrects for `pull_request`. The answer has the same shape: split the event from the authority.

`openspec-autoland-signal.yml` runs on the push, declares `permissions: {}`, uses no action, holds no token, and does one `echo`. Its only product is the existence of a run. `openspec-autoland-controller.yml` reacts on `workflow_run`, which executes the **default branch's** copy, and re-derives everything from the API: event type, head repository, branch, commit, and whether the branch still points at that commit. It reads nothing the signal produced -- not its outputs, not its artifacts, not even its conclusion, which is why a signal run that *failed* still yields a correct decision and a signal run that lies yields nothing extra.

So a rewritten signal workflow has exactly one power: to not run, which stops autoland for that push. Failing closed is the right direction and is the entire attack surface.

Ordering and duplication are handled by the same fact rather than by bookkeeping. The controller refuses unless the branch still points at the commit the run named, so a late, repeated, or out-of-order event for a superseded commit simply declines; the newer push has its own run. One pull request per branch is kept by looking for an open one before creating.

### D17: The event-suppression problem, measured rather than assumed

A pull request opened with `GITHUB_TOKEN` raises no `pull_request` event. That is the load-bearing constraint on this whole design, so it was checked against the repository's own workflow files rather than asserted:

- `ci.yml` — `push: [main]`, `pull_request`, `workflow_dispatch`
- `conformance.yml` — `push: [main]`, `pull_request` (no dispatch)
- `changelog.yml` (on `main`) — `pull_request` only

Of the nine required contexts, several live in `ci.yml` and could in principle be produced by `workflow_dispatch`. But `Hull Conformance Gate (Linux)` and `Changelog` cannot be produced by anything except a `pull_request` event, and protection requires all nine. So a `GITHUB_TOKEN`-opened pull request can never go green here, no matter what else is built.

The design does not route around it: it opens the pull request with a different credential. See D20. `test_at_least_one_required_context_needs_a_pull_request_event` re-measures the constraint against the live workflow files, so it fails loudly if a future trigger change makes the prerequisite obsolete.

### D18: `skipped` is a passing required check

The merge worker refused `skipped`, on the reasoning that a skipped check proved nothing. Measured against reality, that rule refused *every* pull request the feature exists for.

On real docs-only pull request #1634 — which merged — six of the nine required contexts concluded `skipped`, because `ci.yml` skips its heavy jobs for a docs-only diff through a job-level `if` (chelis#419). Branch protection treats those as satisfied; that is the whole point of the design in `ci_detect_docs_only.py`. A worker stricter than protection is not safer here, it is merely broken: it would never merge anything while protection stood ready to.

`neutral` stays refused. Nothing in this repository produces it, so accepting it would widen the rule on speculation, and protection remains the final arbiter either way.

### D19: `skipped` passes protection's contexts, never our own

D18 made `skipped` a passing conclusion. Adversarial validation then showed that the same rule was being applied to `--require-context`, which names the strict OpenSpec validation — and drove the real worker to `MERGED` with `strict=skipped`, meaning the validation never ran and the documents merged anyway.

The two requirements only look alike. Branch protection's contexts are skipped *by design* on a docs-only diff, and protection accepts that. The context this worker adds itself exists precisely to have run. So `RequiredCheck` now carries `must_run`, set for every `--require-context` value, and those demand `success` exactly.

It was unreachable in practice today only by luck: the validation workflow has exactly one job, so a skipped job makes the whole run conclude `skipped` and the separate `--require-workflow` pin catches it. Adding a second job, a job-level `if`, or a matrix to that file would have silently removed the only thing holding it.

### D20: One credential, one call, and no fallback

The pull request is opened with the `OPENSPEC_SUBMISSION_TOKEN` repository secret. Every read — the workflow run, the branch heads, the existing-pull-request lookup — keeps the ordinary `GITHUB_TOKEN`, and the controller job's `permissions:` grants that token `pull-requests: read`, not `write`. The default token is therefore structurally unable to open a pull request, rather than merely asked not to.

**No fallback, deliberately.** An absent secret makes the controller report the push blocked, print the setup recipe, and write nothing. Falling back to `GITHUB_TOKEN` would open a pull request that can never merge (D17) and would hide the misconfiguration behind something that looks like success. A stalled pull request is a worse outcome than a loud refusal, and it is harder to diagnose.

**The credential is checked immediately before the create call, not at the top.** Checking first would make every ordinary code push report "missing secret" instead of "not a document change" — noise about the wrong thing. Reusing an existing pull request needs no credential at all, because it writes nothing.

**Fine-grained PAT, not a GitHub App.** `POST /repos/{owner}/{repo}/pulls` requires exactly one fine-grained permission, `Pull requests: write`, which is why the controller calls that endpoint rather than `gh pr create` — the CLI would additionally read repository and branch metadata and so need a wider grant. The **Workflows** permission is not needed: it governs writing repository content, and this credential never pushes.

An App installation token would be stronger, and it is honestly not what this is. Installation tokens expire after an hour, so one cannot be stored as a secret; using an App would mean holding an App ID and a private key and minting a token per run. That is more moving parts and is not implemented, so the documentation says PAT and means it.

**Exposure.** The secret is referenced by `openspec-autoland-controller.yml` only, which runs from the default branch. The push-side signal and the head-run validator are written by whoever pushed and reference no secret at all; `CredentialExposureTests` asserts that, and asserts the controller passes it to exactly one step.

## Risks / Trade-offs

**A wrong document lands with no reader.** This is the accepted cost, and it is bounded: the change is a document, the merge record is ordinary, and `git revert` undoes it. Structural validation catches schema breakage; it catches nothing about meaning. The pull request body states this limit rather than implying that a green check means the wording is right.

**The per-change `.openspec.yaml` steers its own validation.** It is inside the document boundary because `openspec new change` scaffolds it and excluding it would make new changes unsubmittable. A crafted one could select a weaker schema for that change. The blast radius is one planning document validating loosely -- it cannot widen the path boundary, which is decided by independent trusted code.

**The boundary is the whole control.** There is no second, differently-derived check. This is mitigated by keeping the classifier small, pure, and tested in both directions, and by excluding the classifier from its own automatic path.

**Hosted behavior is unproven here.** The classifier, the submission command, and the workflow contract are covered by local tests. Auto-merge interacting with real branch protection is not, and cannot be until the workflow runs on the hosted repository.

**Structural validation is tree-wide.** `openspec validate --all --strict` validates every specification and active change, so one pre-existing failure anywhere blocks every submission. That is the correct direction, and it is currently blocking: the tree fails today on two unrelated changes. Narrowing validation to the submitted change was rejected because it would diverge from what CI runs and let a submission pass locally and fail hosted.

**A workflow contract test only tests what it names.** The first version of the workflow test caught none of ten unsafe mutations, including `gh pr merge --admin`, which bypasses branch protection outright. Each of those mutations is now a named case with its own assertion. The general shape of that failure -- a test suite that looks thorough and pins nothing dangerous -- is worth more attention than any single mutation in it.

## Migration Plan

The change is additive. `openspec-validate` keeps running in advisory mode and is not modified.

Ordering:

1. Land the classifier, the controller, the merge worker, the submission command, their tests, and the four workflows.
2. Give the mechanism a credential that can actually open a pull request.
3. Observe one hosted document push end to end.

**Step 1 landed.** Pull request #1653 merged as `40266a0a` after every required check passed on its exact head `581955a1`, by an ordinary merge bound to that sha: no administrative option, no auto-merge grant, no protection changed. It carried `a658d655`, which repaired a pre-existing tree-wide validation failure in `harden-lint-traversal-edges` whose MODIFIED block had renamed two scenarios. That failure predated this work -- it reproduced on pristine `origin/main` -- and while it stood, strict validation would have refused every document submission.

**Step 2 landed, and only because the first hosted run refused.** The controller as first deployed minted from the shared `CI_APP_*` App, and its first real run failed: `POST /app/installations/.../access_tokens` returned HTTP 422, `The permissions requested are not granted to this installation`. The classify step never ran and no pull request was opened, which is the designed fail-closed path rather than a fallback to `GITHUB_TOKEN`. The remedy was a dedicated App, `chelis-openspec`, whose installation grants `contents: read`, `metadata: read`, and `pull requests: write` -- not widening the shared App, which several workflows already hold the key to. Pull request #1660 merged as `59bf8bda`.

This is the argument for a boundary that refuses by default, stated concretely: the mechanism's first act on real infrastructure was to be wrong about its own credential and write nothing.

**Step 3 is the open oracle.** Until a hosted push is observed producing a controller decision, an App-opened pull request, and a merge verdict, the token mint, the create call, the check discovery under the Actions token, and the merge call remain unverified.

Rollback is deleting the workflow files. It touches neither the submission command nor the document tree.

## Open Questions

- Whether `openspec/project.md` should stay in the document set. No such file exists today, but when one does it is agent-facing instruction text, which is closer to `openspec/config.yaml` -- excluded because it steers tooling -- than to a capability specification. Decide before creating it.
- `openspec/changes/.openspec.yaml`, directly under `changes/` rather than inside a change directory, currently classifies as a document. It is a scaffold path that does not correspond to any change; it is harmless today and worth tightening when the classifier is next touched.
- Whether an accepted document should notify a channel after the fact, so "no review" does not become "no awareness".

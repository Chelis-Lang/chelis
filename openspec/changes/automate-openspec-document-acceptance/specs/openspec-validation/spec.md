## ADDED Requirements

### Requirement: The OpenSpec document boundary is machine-decidable

Chelis SHALL classify a change as confined to OpenSpec documents, or as requiring human review. The document set SHALL be `openspec/project.md`, Markdown files under `openspec/specs/`, and Markdown or `.openspec.yaml` files under `openspec/changes/`. Normative capability specifications under `openspec/specs/` SHALL be included, by explicit maintainer authorization recorded in `spec/design/spec_provenance.md`.

The decision SHALL be made by one classifier over `git diff --raw` metadata. The comparison SHALL start at the merge base of the base and head revisions, so that commits the change did not make are not attributed to it.

The classifier SHALL additionally verify that the head carries byte-identical governance content, comparing `.github`, `scripts`, and `openspec/config.yaml` by Git object identity rather than by diff. A path that cannot be resolved on either side SHALL count as a difference.

The classifier SHALL write only a closed vocabulary to its machine-readable step output. It SHALL NOT write repository-derived text there, and it SHALL escape control characters in any path it reports.

#### Scenario: A capability specification edit is classified as a document change
- **WHEN** a pull request changes only `openspec/specs/<capability>/spec.md`
- **THEN** the classifier SHALL report the verdict `auto`

#### Scenario: A stale base revision does not refuse the change
- **WHEN** the base branch gained unrelated commits after the pull request branched
- **THEN** the classifier SHALL report only the paths the pull request changed

#### Scenario: A head carrying older governance content is refused
- **WHEN** the head changes no governance path but its `.github` tree differs from the base
- **THEN** the classifier SHALL report the verdict `review`
- **AND** the reason SHALL name the differing governance path

#### Scenario: A path cannot forge a step output
- **WHEN** a changed path contains a newline spelling a `verdict=auto` line
- **THEN** the machine-readable step output SHALL contain only the classifier's own verdict
- **AND** the reported reason SHALL escape the control characters

### Requirement: An ordinary branch push starts acceptance

Chelis SHALL start automatic acceptance from an ordinary push to a branch other than the default branch. No local command SHALL be required.

The push-side workflow SHALL hold no permissions and SHALL produce nothing that any later step consults. The controller SHALL run from the default branch and SHALL re-derive, from the API, the event type, the head repository, the branch, and the commit. It SHALL refuse an event that is not a push, a head repository that is not this repository, the default branch, a branch that no longer exists, and a branch that has moved off the commit it was told about.

The controller SHALL classify that exact commit with default-branch code before any write, and SHALL open exactly one internal pull request per branch, reusing an existing open one.

#### Scenario: A document-only push opens an internal pull request
- **WHEN** a branch whose diff is confined to OpenSpec documents is pushed
- **THEN** the controller SHALL classify the pushed commit
- **AND** it SHALL open one internal pull request for that branch

#### Scenario: A mixed push writes nothing
- **WHEN** a pushed branch changes an OpenSpec document and a source file
- **THEN** the controller SHALL report the push blocked
- **AND** it SHALL NOT open a pull request

#### Scenario: A superseded push is not acted on
- **WHEN** the branch has moved to a different commit since the push being processed
- **THEN** the controller SHALL refuse, leaving the newer push to decide

#### Scenario: A signal from another repository is refused
- **WHEN** the run's head repository is not this repository
- **THEN** the controller SHALL refuse before any write

#### Scenario: A repeated event does not duplicate the pull request
- **WHEN** the controller runs again for a branch that already has an open pull request
- **THEN** it SHALL reuse that pull request

#### Scenario: The push-side workflow confers no authorization
- **WHEN** the pushed branch supplies its own copy of the signal workflow
- **THEN** the controller SHALL read none of its outputs, artifacts, or conclusion
- **AND** the decision SHALL rest only on API facts and default-branch code

### Requirement: The pull request is opened by a dedicated credential

Chelis SHALL open the internal pull request with a short-lived installation token minted per run from the GitHub App this repository already configures, and SHALL NOT open it with the workflow's built-in token. The built-in token SHALL NOT be granted permission to write pull requests.

The minted token SHALL be scoped to the current owner and the current repository only, SHALL request `Pull requests: write` and no other permission, and SHALL be revoked when the job ends. No installation token SHALL be stored.

The submission credential SHALL be used for the pull-request creation call and for no other call. Every read SHALL use the built-in token.

When no submission credential reaches the controller, Chelis SHALL report the push blocked, naming the App configuration and the exact installation permission it requires, and SHALL NOT open a pull request. It SHALL NOT fall back to the built-in token.

The App private key SHALL be referenced only by the workflow that runs from the default branch. No workflow that runs a file supplied by the pushed branch SHALL reference any secret, App identifier, or token-minting action.

#### Scenario: The pull request is created by the submission credential
- **WHEN** the controller opens a pull request
- **THEN** the creation call SHALL use the submission credential
- **AND** every read call SHALL use the built-in token

#### Scenario: A missing credential blocks before any write
- **WHEN** no submission credential reaches the controller
- **THEN** the controller SHALL report the push blocked and name the App configuration
- **AND** it SHALL NOT open a pull request with any other credential

#### Scenario: The minted token is scoped to one repository and one permission
- **WHEN** the workflow mints the submission token
- **THEN** it SHALL request the current owner and the current repository only
- **AND** it SHALL request `Pull requests: write` and no other permission
- **AND** it SHALL leave automatic revocation enabled

#### Scenario: A missing credential does not mask an ineligible push
- **WHEN** the push is not a document change and the credential is also absent
- **THEN** the reported reason SHALL be the boundary, not the secret

#### Scenario: Reusing a pull request needs no credential
- **WHEN** an open pull request already exists for the branch
- **THEN** the controller SHALL reuse it without the submission credential

#### Scenario: A head-supplied workflow cannot mint a token
- **WHEN** the push-side signal workflow or the head-run validator runs
- **THEN** neither SHALL reference the App identifier, its private key, or a token-minting action

#### Scenario: A head-supplied workflow cannot reach the secret
- **WHEN** the push-side signal workflow or the head-run validator runs
- **THEN** neither SHALL reference any secret

### Requirement: The acceptance decision runs on a trusted trigger

The autoland workflow SHALL run on `pull_request_target`, so that it executes the base branch's copy of itself. It SHALL NOT run on `pull_request`, because that event runs the workflow file supplied by the change under judgement.

No job in that workflow SHALL check out or execute pull-request code, configuration, or actions. Pull-request content SHALL enter as Git objects only.

A job holding a write permission SHALL exist only for the merge, SHALL run only after the boundary classifier reported `auto`, and SHALL re-derive that verdict from base-revision code before it acts.

#### Scenario: The workflow cannot be rewritten by the change it judges
- **WHEN** a pull request modifies the autoland workflow file
- **THEN** the workflow that runs SHALL be the base branch's copy
- **AND** the classifier SHALL report the verdict `review` for that pull request

#### Scenario: The write token is not reachable from pull-request content
- **WHEN** the merge job runs
- **THEN** every command it runs SHALL come from the base revision checkout

#### Scenario: A complete change proposal is a document change
- **WHEN** a pull request adds `.openspec.yaml`, `proposal.md`, `design.md`, `tasks.md`, and delta specifications under one `openspec/changes/<change>/` directory
- **THEN** the boundary classifier SHALL report the verdict `auto`

#### Scenario: Archiving a change is a document change
- **WHEN** a pull request renames documents from `openspec/changes/<change>/` into `openspec/changes/archive/<change>/` and deletes the remaining change documents
- **THEN** the boundary classifier SHALL report the verdict `auto`

#### Scenario: A code change is never accepted automatically
- **WHEN** a pull request changes any path outside the OpenSpec document set
- **THEN** the boundary classifier SHALL report the verdict `review`

#### Scenario: A document change mixed with code is not accepted automatically
- **WHEN** one pull request changes an OpenSpec document and a source file together
- **THEN** the boundary classifier SHALL report the verdict `review`

### Requirement: The acceptance boundary fails closed

The classifier SHALL report `review` for every change set it cannot fully account for. It SHALL NOT skip, ignore, or infer past an input it does not recognize.

The classifier SHALL accept only the `A`, `M`, `D`, and `R` statuses on regular-file mode `100644`, with mode `000000` on the absent side of an addition or deletion. It SHALL report `review` for a symlink, a submodule pointer, an executable file, a directory entry, a type change, a copy record, an unmerged record, and any other status. The unmerged and combined-merge cases are defensive: the commands the callers run do not emit them.

The classifier SHALL report `review` for a deletion under `openspec/specs/`, for a rename whose source or target is not a document, and for a rename that moves a capability specification out of `openspec/specs/`. It SHALL likewise report `review` for a rename that moves `openspec/project.md` out of that path, because a move out of a normative slot removes the document from it as completely as a deletion does.

Automatic acceptance SHALL exclude every input that decides acceptance policy, including `openspec/config.yaml`, the autoland workflow, the classifier, the submission command, and the consumer checker.

#### Scenario: An empty change set proves nothing
- **WHEN** the diff between base and head contains no records
- **THEN** the classifier SHALL report the verdict `review`

#### Scenario: An unreadable diff record is not skipped
- **WHEN** a diff record cannot be parsed, or the diff is a combined merge diff
- **THEN** the classifier SHALL report the verdict `review`

#### Scenario: A document path spelled as a symlink is refused
- **WHEN** a change adds or modifies a path in the document set with file mode `120000`
- **THEN** the classifier SHALL report the verdict `review`

#### Scenario: Editing the acceptance policy requires review
- **WHEN** a pull request changes the classifier, the autoland workflow, the submission command, the consumer checker, or `openspec/config.yaml`
- **THEN** the classifier SHALL report the verdict `review`

#### Scenario: Deleting a capability specification requires review
- **WHEN** a pull request deletes a file under `openspec/specs/`
- **THEN** the classifier SHALL report the verdict `review`

#### Scenario: Moving the project document out of its slot requires review
- **WHEN** a pull request renames `openspec/project.md` to any other path
- **THEN** the classifier SHALL report the verdict `review`

### Requirement: An OpenSpec document change merges without human review

Chelis SHALL merge a pull request whose changed paths are all OpenSpec documents, once every required check has passed for its exact head commit. No human approval SHALL be required.

The merge SHALL require every context branch protection declares for the base branch, plus the strict OpenSpec validation context. A missing, pending, queued, in-progress, neutral, cancelled, or failed result SHALL be treated as not passing. A `skipped` result SHALL be treated as passing, because branch protection does, and because a docs-only diff skips this repository's heavy jobs by design. Where a context has several results on one commit, only the most recent SHALL be considered.

Where branch protection pins a context to an owning app, only a result from that app SHALL satisfy it. A result whose owning app is unknown SHALL NOT satisfy it, and a commit status -- which carries no app and which any account with write access may post -- SHALL NOT satisfy it.

The strict OpenSpec validation requirement SHALL additionally be pinned to its producer rather than to a check-run name. Chelis SHALL require a completed, successful run of the named validation workflow **file**, for the pull-request event, on the exact head commit. A run recorded against another commit, another event, or another workflow file SHALL NOT satisfy it, and where several runs exist only the most recent SHALL be considered.

Required-check discovery SHALL use an endpoint the Actions token can reach. `GET /repos/{owner}/{repo}/branches/{branch}` requires `Contents: read` and carries the required contexts with their owning apps; `GET /repos/{owner}/{repo}/branches/{branch}/protection` requires `Administration`, which no workflow `permissions:` block can grant. The first SHALL be the primary source and the second MAY be a fallback.

An empty, unreadable, or undiscoverable required-check set SHALL refuse the merge rather than be treated as satisfied, and the refusal SHALL name the permission each attempted source requires. A base branch that is not protected SHALL be refused by name.

A result set read across pages SHALL NOT be truncated. Where the pages available are exhausted before the list ends, the merge SHALL refuse rather than decide on a partial read.

The merge SHALL bind the exact head commit that was validated, so a commit pushed after the decision cannot be merged by it. It SHALL also bind the base the boundary verdict was computed against: a pull request whose base **ref** differs from the authorized one SHALL be refused, and a base at a different commit on that ref SHALL be accepted only where the authorized commit is an ancestor of, or identical to, the live one. The merge SHALL re-read the pull request immediately beforehand and refuse a state that has changed. Where GitHub has not yet computed mergeability, the worker SHALL re-read a bounded number of times before refusing, so an unfinished computation is not read as a refusal.

The merge SHALL NOT enable auto-merge, SHALL NOT use an administrative or protection-bypassing option, and SHALL NOT submit a review. Branch protection SHALL remain the final arbiter; where it refuses, the pull request SHALL be reported as blocked.

Waiting SHALL be bounded, and a run that times out SHALL report how to run the decision again.

#### Scenario: A validated document change merges
- **WHEN** every required context and the strict validation context have concluded `success` for the head commit
- **THEN** the pull request SHALL be merged with the squash method
- **AND** the merge SHALL name that exact head commit

#### Scenario: The latest result for a context decides
- **WHEN** a required context has an earlier `success` and a later `failure` on the head commit
- **THEN** the merge SHALL refuse

#### Scenario: A skipped required check satisfies it
- **WHEN** a required context concluded `skipped` for the head commit
- **THEN** that context SHALL count as passing

#### Scenario: An earlier cancellation does not block a later success
- **WHEN** a required context has an earlier `cancelled` run and a later `success` run on the head commit
- **THEN** that context SHALL count as passing

#### Scenario: A commit pushed after the decision is not merged by it
- **WHEN** the head advances after the required checks were evaluated
- **THEN** the merge SHALL refuse rather than merge the new commit

#### Scenario: Strict validation must pass for the same commit
- **WHEN** the strict OpenSpec validation context has not concluded `success` for the head commit
- **THEN** the merge SHALL refuse

#### Scenario: Protection refusal is reported, not worked around
- **WHEN** branch protection refuses the merge
- **THEN** the pull request SHALL be reported as blocked
- **AND** no administrative or bypass option SHALL be attempted

#### Scenario: An absent check result is not a pass
- **WHEN** a required status context has reported no conclusion for the head commit
- **THEN** the merge SHALL refuse naming that context

#### Scenario: A fork or draft pull request never reaches the write token
- **WHEN** the head repository is not the Chelis repository, or the pull request is a draft
- **THEN** the merge job SHALL NOT run

#### Scenario: A retargeted base is not the base that was authorized
- **WHEN** the pull request's base branch differs from the one the boundary verdict was computed against
- **THEN** the merge SHALL refuse naming the base

#### Scenario: A base that only advanced does not refuse the change
- **WHEN** the base branch has gained commits since the boundary verdict, and the authorized base commit is an ancestor of the live one
- **THEN** the merge SHALL proceed

#### Scenario: A rewound base is refused
- **WHEN** the base branch has been rewound or rewritten, so the authorized base commit is not an ancestor of the live one
- **THEN** the merge SHALL refuse

#### Scenario: A commit status does not satisfy an app-pinned context
- **WHEN** a required context that protection pins to an app has only a commit status reported for the head commit
- **THEN** the merge SHALL refuse naming that context

#### Scenario: A check-run name does not substitute for the validation workflow
- **WHEN** a check run named for the strict validation context has concluded `success`, but no successful pull-request run of the validation workflow file exists for the head commit
- **THEN** the merge SHALL refuse naming that workflow file

#### Scenario: Unreachable protection is a refusal, not an empty requirement
- **WHEN** no available endpoint reports the checks branch protection requires
- **THEN** the merge SHALL refuse
- **AND** the reason SHALL name the permission each attempted endpoint requires

### Requirement: Automatic acceptance authorizes structure, not meaning

A successful automatic acceptance SHALL authorize only that every changed path was an OpenSpec document and that the tree passed structural validation.

It SHALL NOT imply that the wording is correct, that the requirement is wanted, or that the document agrees with `spec/**` or an executable oracle. Where an accepted OpenSpec artifact contradicts an owning authority, the owning authority SHALL remain controlling and the artifact SHALL be corrected.

The submission SHALL state these limits in the pull request record.

#### Scenario: An accepted document conflicts with an owning authority
- **WHEN** an automatically accepted OpenSpec artifact contradicts `spec/**` or an executable oracle
- **THEN** the owning authority SHALL remain controlling
- **AND** the OpenSpec artifact SHALL be corrected through a further change

#### Scenario: The pull request record states the limit
- **WHEN** the submission command opens a pull request
- **THEN** the body SHALL state that structural validation proves schema validity only

### Requirement: One command submits an OpenSpec document change

Chelis SHALL provide `scripts/openspec_submit.py`, which classifies the change, runs strict structural validation, creates a branch, and opens a pull request. It SHALL NOT request auto-merge and SHALL NOT merge or approve anything.

The command SHALL classify the union of the working tree, the index, and untracked files at OpenSpec document paths, taking the file mode from the filesystem for untracked files. It SHALL NOT modify the index before the verdict is `auto` and validation has passed.

An untracked file outside the document set SHALL NOT be classified and SHALL NOT stop the submission. Such a file cannot reach the commit: untracked content enters a commit only through an explicit `git add`, and the command adds and commits exactly the classified paths. This exclusion SHALL NOT extend to tracked changes, whose index entries are commit inputs.

The command SHALL commit exactly the classified paths, passing each as a literal filename rather than a pathspec pattern. It SHALL NOT commit any other staged change.

The command SHALL refuse a working tree that holds an unresolved merge.

The command SHALL exit `0` on success, `1` when the change requires human review or validation reports a finding, and `2` for an operational failure.

#### Scenario: A staged code change stops the submission
- **WHEN** a code path is staged in the index while the working tree copy is unchanged
- **THEN** the command SHALL exit `1` naming that path
- **AND** it SHALL NOT create a branch, commit, or pull request

#### Scenario: An unrelated staged change is not committed
- **WHEN** the index holds a staged change outside the classified paths
- **THEN** the commit SHALL contain only the classified paths

#### Scenario: A path spelled as a glob is treated as a filename
- **WHEN** a classified document path contains a pathspec metacharacter
- **THEN** the command SHALL pass it to Git as a literal path
- **AND** the commit SHALL NOT include any other file matched by that pattern

#### Scenario: An unresolved merge stops the submission
- **WHEN** the working tree holds an unresolved merge
- **THEN** the command SHALL exit `1`

#### Scenario: An untracked stray does not stop the submission
- **WHEN** the working tree holds untracked files outside the document set alongside a document change
- **THEN** the command SHALL submit the document change
- **AND** the commit SHALL NOT contain any of those files

#### Scenario: An untracked document is still classified by its file mode
- **WHEN** an untracked file at a document path is a symlink or carries the executable bit
- **THEN** the command SHALL exit `1`

#### Scenario: A document change is submitted
- **WHEN** the working tree contains only OpenSpec document changes and validation passes
- **THEN** the command SHALL create a branch, commit, push, and open a pull request
- **AND** it SHALL NOT request auto-merge

#### Scenario: A code change stops the submission
- **WHEN** the working tree contains any non-document change
- **THEN** the command SHALL exit `1` naming the offending path
- **AND** it SHALL NOT create a branch, commit, or pull request

#### Scenario: A validation finding stops the submission
- **WHEN** strict validation reports a schema finding
- **THEN** the command SHALL exit `1`
- **AND** it SHALL NOT create a branch, commit, or pull request

#### Scenario: A dry run creates nothing
- **WHEN** the command runs with `--dry-run`
- **THEN** it SHALL print the plan
- **AND** it SHALL NOT create a branch, commit, or pull request

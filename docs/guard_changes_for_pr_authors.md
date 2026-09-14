# Changing tests, inventories and protected contracts

Read this guide before starting PR work and again before merging. Follow the
sections relevant to changes in tests, generated lists, error messages, build
workflows or protected project rules. The owning specification still decides
correct behavior; this guide explains how to make the change reviewable.

Run the focused checks for your change and the normal pre-push gate:

```sh
python3 scripts/gate.py --fast
```

CI selects additional checks automatically. A passing fast gate does not certify
all compiler behavior or replace a named acceptance command required by the
owning design.

## Adding, moving or removing tests

For an existing command whose selected tests are protected, adding another test
is allowed: every previously required test and every newly selected test must
remain enabled, execute and pass. Do not rewrite the required list merely
because the selected set grew.

Deleting, renaming, ignoring or filtering out a required test still fails. Restore
its coverage, or review an intentional replacement and update its owning
contract and required identities together. Regenerating the list to omit a
missing test is not a repair. Commented-out test declarations do not count as
present.

New or directly modified default-enabled Rust integration targets enter required
PR CI automatically. Use exact `package::target` identities when inspecting the
plan; equal target names can belong to different packages. A change to a helper
or implementation file does not imply that every affected integration test ran:
broader package execution is currently informational. Run your focused behavior
checks. Feature-specific, ignored or hardware-dependent coverage needs its
explicit owner; do not hide a failing target with an exclusion.

If planning reports an unknown shared path, add a reviewed mapping to its real
packages or existing automated owner in `.config/ci-test-targets.toml`. Do not
add an unrelated mapping just to satisfy the planner. The standing test list is
a reviewed coverage decision, not a list to extend for every new test. See
[CI ownership and execution](ci_validation.md).

For a new executable example under `examples/`, add its per-file parity test.
Pass its literal root path to the existing parity/check helper, directly or
through an immutable local. The completeness check derives membership from
those test inputs; there is no second filename list to update. The test still
owes its applicable execution and comparison checks.

## Updating an inventory

First determine what each row means.

- A generated list of what exists in source should be regenerated with its
  owning tool and the resulting diff reviewed.
- A reviewed classification, required-coverage list or exclusion policy records
  a decision. A generator cannot invent that decision or remove its obligation.

Moving a typed-vocabulary consumer within its owning crate does not require a
new file-location pin. Removing the crate's last real evidence still fails;
comments and literal text cannot supply it. Adding a valid `expand` use does not
require updating an occurrence count: the behavioral and dispatch-position tests
own the retained obligations.

A newly discovered numeric export still needs the exact authority required by
[Numeric Surface Discipline](../AGENTS.md#numeric-surface-discipline). Nested
published headers are included in discovery. Changing a baseline to include an
unclassified export does not authorize it.

## Changing a protected rule or test

Keep the existing digest, required-literal and negative-test protections. A
report naming a change is review evidence, not permission to weaken the check.
Review the underlying change, its consumers and its independent positive and
negative behavior evidence before making a deliberate freeze update.

After committing a contract change, obtain the exact required PR-description
lines with:

```sh
.venv/bin/python scripts/phase4b_change_report.py --base origin/main --output target/phase4b-contract-changes.json
```

The report compares committed revisions. Uncommitted edits are not its input.
Read `required_acknowledgements` in the JSON and put each listed line exactly
once in the PR body, outside a code fence. For example, a changed numeric rule
can require both:

```text
Frozen-contract-change: spec/05-risc-primitives.md
Frozen-contract-change: atom:05-OP-33
```

A changed protected region uses its exact label as a JSON string, for example
`Frozen-contract-change: region:"agent numeric surface discipline"`, plus its
changed-file line. Use only the lines your report requires: missing, duplicate,
stale and unknown acknowledgements fail. Registry edits can require an owning
atom's line even when the chapter itself was not edited. Removed declarations
remain review obligations.

The enforcing local check accepts the saved PR body:

```sh
.venv/bin/python scripts/dtype_phase4b_oracle.py --base origin/main --require-acknowledgement --acknowledgements-file target/pr-body.md
```

This command also checks compiler-produced metadata and can compile project
code; follow the build-concurrency instructions in AGENTS.md. CI supplies its
own validated PR comparison. Do not use its `--pr-head` mode on an ordinary
local branch.

For protected Rust tests, the Docs job also publishes
`phase3-test-changes.json`, identifying changed definitions and required
membership. Review those changes and retain the existing test-definition
freezes, comparator calls, receipt checks and negative controls. A changed
mutation witness still owes review of its implementation, command and expected
rejection; producing the same error alone does not prove the witness stayed
strong enough.

## Changing an error message or its issue reference

Preserve the rejection's subject, compiler stage, diagnostic kind, authority,
source location and supported alternative where applicable. Existing exact-text
checks and their negative controls remain until the owning contract replaces
them with complete structured checks. Some public error contracts deliberately
require exact bytes.

For a message already using a shared reviewed expectation, update that one
expectation and regenerate its artifact with the owning tool, then run its
consumer tests. The current softmax wording generator is
[`generate_reviewed_unsupported_wording_snapshot.py`](../scripts/generate_reviewed_unsupported_wording_snapshot.py).
It generates a reviewed expectation, not text inferred from the production
emitter; a changed emitter does not automatically make its new wording correct.
Do not add another parser for the rendered sentence.

After changing a production `unimplemented_rejection!` citation, regenerate:

```sh
.venv/bin/python scripts/generate_rejection_registries.py --write
```

Review that the issue is relevant to the missing implementation. PR validation
checks new or changed issue identities against GitHub and rejects closed,
missing or pull-request references. It still checks complete source/compiler
agreement on relevant runs. A scheduled canary owns later closure of unchanged
issue identities; do not re-cite unrelated code merely to quiet that report.

## Changing the guarded Nix workflow

The temporary event restrictions and byte freeze still apply. A recipe change
needs an explicit reviewed freeze update. Removing the freeze belongs in the
reviewed replacement-policy change, after its event/recipe controls and complete
native flake and Devenv smoke checks pass on both supported systems. Evaluation
alone does not meet that condition.

[The assessment](../spec/design/guard_artifact_proposal_assessment.md) records
these selected contracts and the work still required before further protections
can be retired.

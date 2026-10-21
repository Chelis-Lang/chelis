# Repository quality mechanisms

Repository rules work best when the construction boundary enforces them.
Use this order when choosing an enforcement mechanism:

1. A type or exhaustive enum makes an invalid state unrepresentable.
2. A compiler or lint error rejects a forbidden construction.
3. An executable test rejects a regression or detects drift.
4. Generated checklists and review instructions guide decisions that code
   cannot enforce.

A proposal that uses a weaker mechanism should explain why a stronger one
cannot express the rule. Prose documents the contract; it does not establish
that an implementation satisfies it.

## Reuse and completeness

Before adding a numeric helper, dispatch arm, formatter, or gate, search for
the existing owner. Reuse its typed interface or explain why a distinct
mechanism is necessary. Extend closed enums and generated tables through
their owning registration path so an omitted consumer fails visibly.

Intentional copies need an executable drift check. The conformance manifest
and managed-block checks enforce the copies distributed to shell repositories.
Do not keep a second static fixture when a committed regression test already
owns its source and expected behavior.

## Checks and evidence

The [PR author guide](guard_changes_for_pr_authors.md) defines review and
protected-contract checks. The [local gate](local_gate.md) documents the fast
checks; [CI validation](ci_validation.md) records the hosted owners and
[manual gates](manual_gates.md) name acceptance commands outside routine CI.

Run positive and negative cases. A passing comparison between two execution
paths establishes agreement for those cases; independent expected results
are needed to catch a shared wrong answer. Do not infer an untested dtype,
operation, layout, or backend from a nearby passing case.

The [red-team workflow](../agent-skills/redteam-exec/SKILL.md) requires command
execution against the candidate head and keeps the reviewer available to
verify repairs. Review source, tests, examples, and documentation against the
owning specification. Report the reviewed head, commands, findings, and
unvalidated scope together.

## Guidance and historical evidence

`AGENTS.md` is the repository entry point. Directory instructions and shared
skills add guidance where the work needs it; they do not create another
source of truth for language semantics. Numbered specifications own those
semantics, and tests must retain their positive and negative obligations.

The [architecture notes](archive/agent_quality_architecture_notes.md) preserve
the original audit, proposed mechanisms, and implementation backlog.
They are historical evidence rather than an inventory of installed tooling.

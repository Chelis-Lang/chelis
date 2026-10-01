`chelis reef conform sync` and `conform bump` now propagate the inherited agent
surface in full. A shell's `docs/CHELIS_SURFACE.md` carries the pinned
toolchain's complete capability surface guide in a `chelis-surface` managed
block, in place of the short `chelis-surface-header` pointer block, which sync
replaces where it stood. Shell-owned text outside the block is kept, and a
`shell-local:exclude` selector span outside it omits inherited sections, as in
`AGENTS.md`. The audit no longer asks for `@pin`/`@upstream` markers, and the
Pin Bump Checklist no longer has a manual surface refresh step. Sync also
restores the `CLAUDE.md -> AGENTS.md` symlink. Because the surface document is
generated, `sync` and `bump` create it when it is missing instead of refusing
the repo; they still refuse, before writing anything, when a directory or other
non-file occupies its path.

The `chelis-std` downstream-authoring skill, authored at
`packages/chelis-std/SKILL.md`, is now a shared skill: sync materializes it into
every shell's `agent-skills/`, where `.claude/skills` and `.codex/skills` expose
it, and `[conform] excluded_skills` can omit it. A shell that declared its own
copy in `[conform] local_skills` removes that entry; the audit reports it as
shadowing a shared skill until then.

Inherited text keeps working links. When sync materializes the `AGENTS.md` and
`docs/CHELIS_SURFACE.md` blocks and the shared skills, each repo-relative link
resolves against its source file in the chelis repository: a link to a file
sync also materializes stays relative, and any other becomes an absolute URL
pinned to the shell's release, such as
`https://github.com/Chelis-Lang/chelis/blob/v0.18.12/docs/local_gate.md`. The
audit expects the same rewritten text, so pinned links are not drift.

The shell contract no longer has an issue-drafts convention. Every upstream bug
or capability request is filed as an issue in the Chelis-Lang repository where
it originates and cited by number (`chelis#NNN`, `<repo>#NNN`). `conform audit`
drops the `docs/issue_drafts/` row (rows after it renumber down by one), no
longer accepts a draft path as a `docs/UPSTREAM_BUGS.md` citation, and fails a
`docs/UPSTREAM_BUGS.md` that still has a Parked section; move its entries to
Tracking with their issue numbers. `chelis test --expect blocked` no longer
accepts a draft path as a sidecar citation. Shells adopt these rules when they
bump their pin. See [#2831](https://github.com/Chelis-Lang/chelis/issues/2831).

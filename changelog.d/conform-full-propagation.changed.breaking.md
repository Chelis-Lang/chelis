`chelis reef conform sync` and `conform bump` now propagate the inherited agent
surface in full. A shell's `docs/CHELIS_SURFACE.md` carries the pinned
toolchain's complete capability surface guide in a `chelis-surface` managed
block, in place of the short `chelis-surface-header` pointer block, which sync
replaces where it stood. Shell-owned text outside the block is kept, and a
`shell-local:exclude` selector span outside it omits inherited sections, as in
`AGENTS.md`. The audit no longer asks for `@pin`/`@upstream` markers, and the
Pin Bump Checklist no longer has a manual surface refresh step. Sync also
restores the `CLAUDE.md -> AGENTS.md` symlink.

The shell contract no longer has an issue-drafts convention. Every upstream bug
or capability request is filed as an issue in the Chelis-Lang repository where
it originates and cited by number (`chelis#NNN`, `<repo>#NNN`). `conform audit`
drops the `docs/issue_drafts/` row (rows after it renumber down by one), no
longer accepts a draft path as a `docs/UPSTREAM_BUGS.md` citation, and fails a
`docs/UPSTREAM_BUGS.md` that still has a Parked section; move its entries to
Tracking with their issue numbers. `chelis test --expect blocked` no longer
accepts a draft path as a sidecar citation. Shells adopt these rules when they
bump their pin. See [#2831](https://github.com/Chelis-Lang/chelis/issues/2831).

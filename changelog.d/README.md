# Changelog fragments

Behavior-changing pull requests add a Markdown fragment here instead of editing
`CHANGELOG.md`. This convention takes effect after the 0.18.7 release PR (#1586)
merges and the remaining `[Unreleased]` notes are moved here. The implementation
PR stays draft until that cutover is complete.

## Author a fragment

Use `<pr-or-slug>.<category>[.breaking].md`, for example `1625.fixed.md` or
`parser_errors.changed.breaking.md`. Slugs contain lowercase letters, digits,
underscores, or hyphens. Categories are `added`, `changed`, and `fixed`.
The optional `breaking` suffix supplies the visible `BREAKING` marker.

Write one entry, beginning with a paragraph, without its outer list bullet:

```markdown
The checker reports the invalid argument's location. Previously it reported the
call's location. See [#1625](https://github.com/Chelis-Lang/chelis/pull/1625).
```

Paragraphs, nested lists, and fenced examples are supported. Put citations in
the text: the filename does not generate an issue or PR link. A PR can supply
multiple fragments when it needs distinct categories. Correct an existing
pending fragment when a follow-up changes its claim.

PR checks start **advisory**. Non-documentation changes and normative spec
changes should supply a new or substantively amended fragment. For internal
work with no release-note value, apply `no-changelog`; this suppresses only the
missing-fragment warning. Malformed fragments and direct changelog edits still
produce warnings. Advisory checks do not prove complete release coverage.

Only this README and correctly named regular fragment files belong here.

## Prepare a release

Use the checkout's uv-managed Python. First run the existing
`scripts/bump_compiler_pins.py` release bump, then preview and assemble notes:

```sh
.venv/bin/python scripts/changelog.py check
.venv/bin/python scripts/changelog.py build --version 0.18.8 --date 2026-09-15
.venv/bin/python scripts/changelog.py build --version 0.18.8 --date 2026-09-15 --write
```

Supply the actual version and release date. The version must match the workspace
version. Preview writes nothing. Assembly creates one release section, ordered
Added, Changed, Fixed, with breaking entries first and filenames sorted within
each group. It preserves the preamble and historical releases byte-for-byte,
then removes the consumed fragments. It never creates `[Unreleased]`.

All validation happens before writes. The changelog is replaced atomically
before any fragment is deleted. If a filesystem operation fails, the command
fails and names the operation; inspect the diff and finish or restore the
release edit before retrying. It never stages files, commits, or tags.

Commit the version bump, assembled changelog, and fragment deletions together
in the release PR. Direct historical edits are outside the assembly convention
and produce an advisory warning even with `no-changelog`.

The tag-publishing workflow extracts the committed version section into its
GitHub Release body. It refuses missing, duplicate, or empty notes, a workspace
version mismatch, `[Unreleased]`, and unconsumed fragments. Local extraction:

```sh
.venv/bin/python scripts/changelog.py extract --version v0.18.8 --output /tmp/chelis-release-notes.md
```

## Validation

The focused acceptance oracle is:

```sh
.venv/bin/python -m unittest scripts.test_changelog
```

It exercises the real CLI in temporary Git repositories, including negative
cases. PR policy can also be checked locally without GitHub access:

```sh
.venv/bin/python scripts/changelog.py check-pr --base origin/main --head HEAD --advisory
```

The workflow supplies its event JSON for label handling. Policy findings are
warnings in advisory mode; operational failures remain nonzero. The advisory
workflow is not a required merge check.

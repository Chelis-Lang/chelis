`chelis check <dir>` and `chelis test <dir>` now follow symbolic links. Both
used to skip every symlink while walking a directory, so a symlinked source
file, or a symlinked directory of them, was never checked or run -- with a
successful exit and no warning. A type error or a failing test could be
hidden just by putting it behind a link. **A directory that previously passed
can now fail**, because code it silently skipped is checked and run.

Following links handles three cases it introduces. A link that loops back to
a directory already being walked is skipped. A dangling link named like a
source (`.ch`, or `.dp` for `check`) is collected rather than skipped: `check`
reports it as an unreadable file, exactly as naming it directly would, and
`test` reports it as a failing file. A dangling link to anything else,
including one named like a dot-entry, is ignored.

A link to a source directory is followed wherever it points, including out of
the directory named on the command line. A link that fails to resolve for any
other reason -- a link to a directory that cannot be read, a self-referencing
link, or a link through a file -- stops `chelis test`, which skipped every
link before. `chelis check` reports it and carries on; the entry that follows
in this release describes its envelope. See
[#1678](https://github.com/Chelis-Lang/chelis/issues/1678).

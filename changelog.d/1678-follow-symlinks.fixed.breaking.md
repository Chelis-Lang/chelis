`chelis check <dir>` and `chelis test <dir>` now follow symbolic links. Both
used to skip every symlink while walking a directory, so a symlinked source
file, or a symlinked directory of them, was never checked or run -- with a
successful exit and no warning. A type error or a failing test could be
hidden just by putting it behind a link. **A directory that previously passed
can now fail**, because code it silently skipped is checked and run.

Following links is careful about the cases it introduces. A link that loops
back to a directory already being walked is skipped. A dangling link named
like a source (`.ch`, or `.dp` for `check`) is reported as an unreadable file,
exactly as naming it directly would be, rather than skipped. A dangling link
to anything else is ignored. The existing rules still apply to links by their
own name: a link named like a dot-directory or `target` is skipped. See
[#1678](https://github.com/Chelis-Lang/chelis/issues/1678).

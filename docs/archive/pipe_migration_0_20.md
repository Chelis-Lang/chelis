# Pipe migration for Deep 0.20

Historical migration instructions retained from the CLI guide.

Surf retains `|>`, and `fmt` preserves it. Deep 0.20 represents a pipe as ordinary
applications. `chelis deep` (with or without `--annotate`) and `chelis surf` therefore
emit calls. Stored Deep and compiled caches containing the previous pipe node must
be regenerated. Compiler caches and shell packages reject previous format versions.

Folding before literal typing makes `0.1 |> cast(f64)` equal to `cast(0.1, f64)`.
To retain an existing program's previous f32-rounded value, migrate using a compiler
from before this change:

```console
chelis migrate pipes --baseline-compiler /path/to/previous/chelis --inplace file.ch other.ch
chelis migrate pipes --baseline-compiler /path/to/previous/chelis --check file.ch other.ch
```

With neither flag, the command prints one migrated file. It uses the previous
compiler's expanded Deep to suffix unsuffixed numeric literals in pipe seeds and
adds grouping with the previous grammar's reading, such as `(2.0f32 * x) |> f`.
Every file must preserve normalized expanded Deep and reach a formatter fixed point.
The whole batch is checked before any file is replaced; missing dtype evidence,
changed Deep, or comments the formatter cannot preserve reject the migration.
Migrate shell and downstream sources and bump their compiler pins in the same
change. Run the migration with the retained pre-A1 compiler before using the new
pin to check, format, evaluate, or build the migrated sources.
A file rejected by the previous compiler needs separate diagnosis, rather than an
assumed default dtype.

Retain **both** compilers for this migration, even when A1 (#3164) lands in the
same release. The baseline must precede pipe normalization and A1; the compiler
running `migrate pipes` must include pipe normalization and precede A1. A later
compiler cannot prove an unchanged whole-file graph after unrelated A1 typing
changes. Pin source revisions and retain the corresponding binaries together.
Use baseline revision
`c5e4d116c3852c318bdb415fb052add73efbff6d` and migration tool revision
`a436d99235d9193482dc404565a1d02cf2a8be8d`; build each revision's `chelis-cli`
in its own checkout and preserve those executables rather than a moving default.
The historical reader is available only through the non-default
`pre-020-pipe-migration` feature; the CLI opts in for this explicit command.
Normal Deep parsing and stamping still reject retired pipe receipts.

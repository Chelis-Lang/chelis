# Wave 1 red-team follow-ups — diagnosis (M1/M2/L1/L2)

Diagnosis note covering the four findings from
`docs/investigations/wave1_red_team_report.md`. One paragraph per
finding, identifying the actual cause and the planned fix surface.

## M1 — #207: parse error returns exit 1, not the documented exit 2

`chelis check`'s exit-code contract (`crates/chelis-cli/src/main.rs`,
constant `CHECK_ERRORS_EXIT_CODE = 2`) promises
`exit != 0 iff json.errors.len() > 0`. The Ok arm of `cmd_check_one`
honors that promise by returning a populated JSON report with a
non-empty `errors` array. The Err arm in `main`'s dispatch (around
`crates/chelis-cli/src/main.rs:523`) does not: any
`Result::Err` from `cmd_check_one` — including the parse failure
that `chelis_surf::parser::parse_str` raises via the `?` operator
at the top of `cmd_check_one` — flows through the generic
"error: {err}" / exit 1 path. The JSON document never gets emitted,
the array does not exist, and exit 1 disagrees with the documented
exit 2. The cause is that the parse-error branch was never threaded
through the JSON-emit machinery; the brief contract was written
assuming all `errors[]`-shaped failures route through the Ok arm,
but parser failures bypass it. Fix is to catch the parse error in
`cmd_check` (single-file path) and the per-file dir-walk path,
synthesize a minimal JSON report with `errors: [{ kind: "Other",
message: <parse error text> }]`, print it on stdout, and return
`CHECK_ERRORS_EXIT_CODE`.

## M2 — #207: empty file silently passes both check and build

The red-team report described `chelis check` reporting `score=1,
errors=[]` (exit 0) for an empty file while `chelis build` rejected.
Direct reproduction in this PR's worktree shows that **`build`
also passes** on a zero-byte `.ch` and emits a no-op C function —
the two surfaces agree, but they both agree on the WRONG verdict.
The hazard is the same: a truncated or empty `.ch` produces a
"perfect score" check and a degenerate build artifact. The cause is
that `chelis_surf::parser::parse_str("")` returns
`Ok(vec![])` (empty input is a syntactically valid zero-decl
program), and neither `cmd_check_one` nor `cmd_build` guards against
that. The fix introduces a shared
`reject_zero_decl_program` helper in `crates/chelis-cli/src/main.rs`
that, when `decls` is empty after the prepared/parser stage,
surfaces the canonical message `"empty program: no declarations
found"` — as a `CheckErrorKind::Other` entry in `cmd_check_one`'s
JSON `errors[]` (exit 2) and as a boxed string error in `cmd_build`
(non-zero exit). The shared message keeps the two surfaces in lock
step without introducing a new `CheckErrorKind` variant (which
would require a `chelis-types` edit outside the named file set).

## L1 — #208 §4.7.5: error message direction is inverted from spec

> **Superseded 2026-08-03 (PR #1000):** the §4.7.5 narrative this diagnosis
> produced has been replaced. The intra-list unification mechanism described
> below is real, but it is not what rejects an all-`int32` list — the slot
> itself demands `List<int64>` (`unify` against `List<Int64>` in
> `infer/app_shape.rs`), so `reshape(x, [2, 2])` errors with no `int64`
> element present. chelis#916 tracks the inverted diagnostic; kept as the
> historical record of #208.

The spec narrative at `spec/04-type-system.md:706-715` describes
`reshape`'s shape list as `List<Int64>` and says the bare
`shape(x, k)` returns `int32`, "so a runtime axis size MUST be cast
to `int64`." The narrative implies the diagnostic should read
"expected int64, got int32". The actual diagnostic, raised by the
type-checker's PrecisionMismatch path when unifying the shape list's
element type with the inferred argument type, reads
`"precision mismatch: expected int32, got int64"`. The direction is
a unification-order artifact: the checker unifies the first element
of the list (the bare `shape(...)` result, type `int32`) with each
subsequent element (the `int64` literal `4`), so the "expected" side
is whichever element the unifier saw first. The diagnostic text is
the contract; rewriting the spec narrative to match
("expected int32, got int64", with a short note that "the bare
`shape(x, k)` element fixes the expected precision; later `int64`
entries trip the mismatch") is correct and preserves the spec's
underlying type-safety message. Spec-only change at §4.7.5.

## L2 — #188: actions/upload-artifact@v4 + download-artifact@v4 still node20

The #188 sweep enumerated `actions/checkout` and `softprops/action-
gh-release` but did not touch `actions/upload-artifact` or
`actions/download-artifact`. Three live references remain in
`.github/workflows/release.yml` (lines 70, 148, 162) pinning the
`@v4` major, whose action.yml still declares `using: 'node20'`.
`v7.0.1` of upload-artifact and `v7.x` of download-artifact both
declare `using: 'node24'`. Sweep over the rest of
`.github/workflows/*.yml` finds no other `actions/*@v4` pins
(verified by `grep -rn '@v[0-9]\+'`); only the three in release.yml
need to bump. Fix is a literal `v4 → v7` substitution at those three
lines.

# Performance diagnosis of the open `area:perf` issues (2026-10-16)

This record collects a diagnosis pass over every open issue labelled `area:perf`
that is about compiler or runtime cost, rather than CI runner cost. Each issue
received one diagnosis comment with its evidence; this document synthesizes those
comments into shared root causes and a suggested fix order. The comments carry the
commands, probe sources, scaling tables and profiles; this record carries the
cross-issue picture.

## Scope and method

All measurements were taken against `main` at `928f1a63a`, with one release build
of the CLI (`cargo build --release -p chelis-cli --bin chelis`) shared read-only by
five diagnosis agents, and with the released toolchains 0.16.0 through 0.20.0 for
version-to-version comparisons. Source line numbers in this record refer to
`928f1a63a`. The issues were split into five groups: the type
checker and front end, lowering and ownership verification, generated C and the
compiled lane's ownership code, runtime kernels, and the `chelis test`, `chelis
prove` and bindings harnesses.

The machine was a shared ten-core Apple Silicon workstation, so wall-clock time was
not the primary metric:

- Every run was timed with `/usr/bin/time -l`, reporting user plus system CPU
  seconds, which include every descendant process.
- `instructions retired` was cited only for single-process commands (one `chelis
  check` or `chelis eval`, or a produced executable run directly), because the
  counter covers only the direct child.
- Each headline number was replicated at least twice, and every scaling claim comes
  from a doubling sweep of at least four points.
- Profiles came from `sample` on the release binary, after confirming the sampled
  process's command line.

Free disk stayed below the 25 GB floor set for builds throughout the pass, so no
instrumented or candidate-fix compiler was built. Cost shares are therefore profile
shares, not before-and-after measurements, and each comment labels what is
confirmed by measurement or profile and what is a hypothesis from reading code.

Excluded from the pass: issues labelled `area:ci` and the CI runner-cost issues
#1501 to #1505; the tracking hubs #830 and #828; the tool-adoption requests #823
(Hyperfine) and #824 (Criterion); the documentation issue #2180; and the design
questions #1315 and #1696, which have no cost to measure.

## Per-issue results

| Issue | Status at `928f1a63a` | Mechanism | Diagnosis |
|---|---|---|---|
| #3707 | reproduces, 2^D in nesting depth | lowerability walk re-walks a callee's body at every call site with no memo, on two paths | [comment](https://github.com/Chelis-Lang/chelis/issues/3707#issuecomment-6223963100) |
| #2622 | reproduces, quadratic time and memory (6.3 GB at depth 8,000) | types are unshared trees rebuilt by substitution, and every node stores its own type | [comment](https://github.com/Chelis-Lang/chelis/issues/2622#issuecomment-6223997688) |
| #2536 | reproduces, 12% of samples | substitution cloned per replayed route combination; two per-declaration costs dominate | [comment](https://github.com/Chelis-Lang/chelis/issues/2536#issuecomment-6224089892) |
| #3512 | partially | accumulator re-validation is skipped only for a fixed list of syntactic shapes; others re-walk the list every step | [comment](https://github.com/Chelis-Lang/chelis/issues/3512#issuecomment-6224160565) |
| #1921 | the predicted shape exists at two production sites | `chelis cost` and `manifest_for_eval` build a lowering session per item | [comment](https://github.com/Chelis-Lang/chelis/issues/1921#issuecomment-6224206701) |
| #2641 | reproduces, for any integer chain | each top-level value re-lowers the trapping chain below it; emitted C grows about 3.8x per doubling | [comment](https://github.com/Chelis-Lang/chelis/issues/2641#issuecomment-6223985667) |
| #2330 | reproduces, 49% of an N=640 build, 69% at N=1280 | ownership `verify` rescans the live set per operation and runs three times per build | [comment](https://github.com/Chelis-Lang/chelis/issues/2330#issuecomment-6224054908) |
| #2207 | stated mechanism gone | pipe folding deleted and lowering context cached; a different quadratic is on its probe (#3721) | [comment](https://github.com/Chelis-Lang/chelis/issues/2207#issuecomment-6224213022) |
| #1935 | reproduces | an obligation-free gradient still emits and computes the forward | [comment](https://github.com/Chelis-Lang/chelis/issues/1935#issuecomment-6224252882) |
| #1177 | reproduces; macro expansion is about 10% | 93-97% of a warm build re-runs effect and linearity checks over the cached dependency | [comment](https://github.com/Chelis-Lang/chelis/issues/1177#issuecomment-6224393185) |
| #1329 | reproduces, twice per `check` | two unconditional `cargo locate-project` spawns, about 75% of `check` CPU on a small file | [comment](https://github.com/Chelis-Lang/chelis/issues/1329#issuecomment-6224422124) |
| #3692 | reproduces, 1.8x build CPU against 0.19.4 | #3657's origin bookkeeping, inlined by clang into one 39k-line `forward` | [comment](https://github.com/Chelis-Lang/chelis/issues/3692#issuecomment-6224430736) |
| #3704 | reproduces | every unit parses `arm_neon.h`, about 45 ms CPU per unit | [comment](https://github.com/Chelis-Lang/chelis/issues/3704#issuecomment-6224151558) |
| #2205 | mostly done | `append`, `concat` and `skip` are linear; the `take` cursor is quadratic | [comment](https://github.com/Chelis-Lang/chelis/issues/2205#issuecomment-6224040670) |
| #2206 | reproduces, more widely than titled | a value moves only in tail position; every other last use pays a retain and release | [comment](https://github.com/Chelis-Lang/chelis/issues/2206#issuecomment-6223978969) |
| #2252 | gap stands | no compiled fixture proves list or dictionary in-place reuse | [comment](https://github.com/Chelis-Lang/chelis/issues/2252#issuecomment-6224109383) |
| #2068 | scalar case fixed in 0.18.10; heap case remains (#3716) | each argument slot decides move or copy alone | [comment](https://github.com/Chelis-Lang/chelis/issues/2068#issuecomment-6221740698) |
| #3700 | reproduces, 92% of samples at K=3 | each canonical `Sum` emitter allocates a scratch tensor per output element or 8-column group | [comment](https://github.com/Chelis-Lang/chelis/issues/3700#issuecomment-6223988418) |
| #3701 | reproduces, 39% of samples in the #3700 probe | `new_tensor` formats ledger strings with the ledger disabled | [comment](https://github.com/Chelis-Lang/chelis/issues/3701#issuecomment-6223997984) |
| #3702 | reproduces | runtime `einsum` re-derives and checks indices per product: about 310 instructions per multiply-add against 21 for `matmul` | [comment](https://github.com/Chelis-Lang/chelis/issues/3702#issuecomment-6224047282) |
| #3703 | gap measured at 13-15x | host `einsum` always takes the runtime kernel, even with static shapes | [comment](https://github.com/Chelis-Lang/chelis/issues/3703#issuecomment-6224088929) |
| #3584 | partly moved | full ConvNeXt-Base forward compiles and runs (17.1 CPU s); the training step is held back by build cost | [comment](https://github.com/Chelis-Lang/chelis/issues/3584#issuecomment-6224368969) |
| #2301 | reproduces, n^2.0 | linear key scan per operation, re-validating both keys per comparison | [comment](https://github.com/Chelis-Lang/chelis/issues/2301#issuecomment-6224403032) |
| #3117 | partly reproduces | desugar copies a name table per library def on every sample; Deep lane recompiles per sample | [comment](https://github.com/Chelis-Lang/chelis/issues/3117#issuecomment-6224121254) |
| #2256 | reproduces, about 4% of a worker | the unused lowered DAG is still in the handoff; the desugar copy is 71% of the worker | [comment](https://github.com/Chelis-Lang/chelis/issues/2256#issuecomment-6224240911) |
| #2117 | regression against 0.18.6 gone | Nautilus suite 61 CPU s on main against 110 s on 0.18.6; residue is the desugar copy | [comment](https://github.com/Chelis-Lang/chelis/issues/2117#issuecomment-6224972024) |
| #1268 | regression gone | the School corpus never batches because every probe declares `test_probe` | [comment](https://github.com/Chelis-Lang/chelis/issues/1268#issuecomment-6224978767) |
| #3231 | exponential not reproducible through shipped entry points | lookup-by-id scans remain quadratic | [comment](https://github.com/Chelis-Lang/chelis/issues/3231#issuecomment-6225037161) |

## Shared root causes

### Module-sized state copied once per item

This is the largest class, and most of its fixes are small to medium.

- `GradSelectorResolver::seed_deep_origins` copies the global name table once per
  library definition (`crates/chelis-surf/src/desugar.rs:512`, with the same copy in
  `visit_decl` at lines 583, 611, 618 and 629). Every compile against a library pays
  it: `chelis test` workers, `chelis prove` samples, `chelis eval` inside a package,
  and layered library builds. It is 71% of the prove sampling loop, 71% of a test
  worker on a 3,000-definition package, and about 25% of a School probe worker. It
  is most of what remains of #3117, #2256, #2117 and #1268.
- The linearity checker copies the module-wide scope once per declaration
  (`crates/chelis-types/src/linearity.rs:1557`; #3719). Two groups found it
  independently: about 42% of `chelis check` on 1,600 definitions, most of #1177's
  warm-build cost, and about 70 CPU s for a cold `check` of 4,000 flat definitions.
- The signature-metadata fixed point repeats one identical sweep once per
  definition, with no convergence test
  (`crates/chelis-types/src/infer/declarations.rs:258-274`; #3717): about 54% of
  `check` on 1,600 one-line definitions.
- The macro expander copies a call's arguments before testing whether the call is a
  macro (`crates/chelis-macros/src/lib.rs:644`; #3721), so `check` is quadratic in
  the length of one list literal.
- #2330 rescans the live set per operation, #2536 clones the substitution per route
  combination, and #2622 rebuilds and stores unshared types.

The fix shape is shared: layered or persistent scopes (a shared parent plus a local
overlay), convergence tests, running totals, and interned types. Each wants a
counted guard, such as entries copied or visited, rather than a timing assertion.

### Work recomputed on every use

- #3707: the lowerability walk has no per-declaration memo; the older path for
  precision- and rank-polymorphic definitions is exponential too, on 0.20.0 as well.
- #2641: each top-level value re-lowers the trapping chain below it.
- #1921: `chelis cost` and `manifest_for_eval` construct a lowering session per item.
- #3117: the library is rebuilt per property.
- #3512: accumulator claims are re-walked every step outside the recognized shapes;
  five partial changes have each added one more shape, and the loop-invariant
  design in the issue body would end that cycle.
- `compose_live_byte_bound` finds call cycles by a pairwise search (#3723): about
  cubic in the number of functions, 73% of an 800-function build.

The compiler acceptance canary was also measured here at 1.1-1.3 CPU s per `chelis
build`. #3715 (closing #3712) has since reduced it to the program's own kernels; a
cache across processes is excluded by design (`spec/design/correctly_rounded_math.md`
§5), so no further issue was filed.

### Volume of emitted C, and the cost of each byte

- #3692: #3657's origin bookkeeping gives every local a root and every
  sub-expression a scope, and clang inlines the helpers into one function. In
  compile experiments, `noinline` on seven emitted helpers recovered 52% of the
  regression, eliding empty scopes 45%, and both together 71%.
- #3704: every generated unit parses `arm_neon.h`.
- The C sealing pass parses the whole emitted file twice, at roughly 4,100-4,800
  instructions per emitted byte (#3722). It is 88% of #2641's depth-128 build and
  21-39% of #2330's, so every issue that inflates C pays a second time here.
- #1935 and #2641 both keep copies so that traps stay observable; one tool that
  keeps only observable trap seeds could serve both.
- #3584: the scaled ConvNeXt training step takes 490 CPU s in `--emit-c` and writes
  147 MB of C.

### Runtime hot paths in compiled programs

- #3700 and #3701 are one allocation path: a runtime scratch tensor per output, of
  which roughly 40% is ledger-string formatting. The scratch-free carry tree already
  exists (`emit_deferred_sum`), so the fix routes the four allocating emitters
  through it, extended to 8-wide lanes and other accumulator types.
- #3702 and #3703: the kernel depends on the spelling. `matmul` gets the emitted
  fast kernel; `einsum` gets the runtime one. Routing static host `einsum` through
  the existing DAG lowering closes #3703 in general and leaves #3702 for integer and
  runtime-diagonal cases.
- #2301: a Dict is a vector searched linearly, re-validating both keys per
  comparison; dictionary accumulators in #2205 and #2252 are quadratic for the same
  reason.
- #3584: depthwise convolution costs about 440 instructions per multiply-add,
  because it materializes gathered windows and checks each index through a runtime
  call.

### Ownership move decisions

#2068's remaining heap case (#3716) and #2206 share one cause: `consume` in
`crates/chelis-ir/src/ownership/lower.rs` (lines 882-918) decides moves per argument slot,
keyed on tail position. Two slots can both move one heap value, which the verifier
rejects, and a last use outside tail position never moves. Deciding moves from the
scheduler's last-use placement, with a rule for one value appearing twice in a call,
addresses both. #3716 comes first, because it rejects a valid program: spec/04
(lines 3389-3400) gives such reuse inserted copies, the checker and `chelis eval`
accept it, and the test `heap_value_reused_in_tail_call_slots_is_still_rejected`
added under #2069 locks in the rejection, so the fix inverts that test.

## Issues filed from this pass

| Issue | Defect | Parent |
|---|---|---|
| #3716 | a `List` or `string` passed to two owned slots of a tail call is rejected by the C backend | #1286 |
| #3717 | the signature-metadata sweep repeats once per definition | #830 |
| #3719 | the linearity checker copies the module-wide scope per declaration | #830 |
| #3720 | eval's growing-List fold copies the whole list per `append` | #828 |
| #3721 | the macro expander copies call arguments before testing for a macro | #830 |
| #3722 | the C sealing pass parses the emitted file twice | #830 |
| #3723 | `compose_live_byte_bound` finds call cycles pairwise | #830 |

## Original form resolved

These issues no longer reproduce as filed; each diagnosis comment states what
remains: #2207, #2117, #1268, the exponential case of #3231, and the scalar case of
#2068.

## Suggested fix order

1. The desugar name-table copy: test, prove, eval inside a package, and library
   builds.
2. Linearity scope sharing (#3719) and the signature-sweep convergence test (#3717):
   `check` and warm builds on any large module.
3. #3701, then #3700 through `emit_deferred_sum`.
4. Fixed per-invocation costs: #1329's cargo probe and #3704.
5. `noinline` for #3692, then elision of empty origin bookkeeping.
6. A memo for #3707, running live-set totals for #2330, and the macro expander copy
   (#3721).
7. #3716, then #2206 from last-use placement, and the `take` cursor from #2205.
8. The larger structural items: #3703 then #3702, a hashed Dict for #2301, the
   rollback substitution for #2536, the loop invariant for #3512, interned types for
   #2622, the trap-seed tool for #2641 and #1935, and grouped convolution for #3584,
   which is a language decision.

The first two steps are small-to-medium changes that address about seven of the
diagnosed issues between them.

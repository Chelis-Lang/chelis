# Known core release issues

Snapshot: 2026-10-04 (UTC). This page lists the open issues tagged `scope:core`
and `launch:p2` at capture time. Each issue link supplies its reproducer,
current scope, and implementation status. This is an inventory of reported
limitations, not a claim that every affected program runs or that the release
has passed its gates.

The [launch ledger](https://github.com/Chelis-Lang/chelis/issues/1362) controls the
release boundary. Published P2 issues may remain open at release only when they
have no remaining `demo-path`, `freeze`, or `launch:required` obligation.
The literal-construction decision for #1416 landed in
[PR #3081](https://github.com/Chelis-Lang/chelis/pull/3081), clearing its decision
modifiers while retaining the implementation follow-up. The bounded 13-case
C Note/Sonar receipt passed in
[PR #3095](https://github.com/Chelis-Lang/chelis/pull/3095), clearing #2379's
`demo-path` modifier for that manifest. #2379's wider transform defect remains
open. Neither resolution claims acceptance of unavailable Voyage captures or
other programs outside the recorded manifest.

For supported operations and explicit C exclusions, see [C support](c_support.md).
The [live P2 query](https://github.com/Chelis-Lang/chelis/issues?q=is%3Aissue+is%3Aopen+label%3Ascope%3Acore+label%3Alaunch%3Ap2)
is current when issue state or labels differ from this dated snapshot.

| Issue | Reported limitation | Remaining gate at snapshot |
|---|---|---|
| [#409](https://github.com/Chelis-Lang/chelis/issues/409) | Chelis lowering stack overflow on Shoals pricer cost JSON | None of these modifiers |
| [#613](https://github.com/Chelis-Lang/chelis/issues/613) | C build: bare `out = grad(f)` export rejected ('grad is not supported by IR evaluation yet') unless the verb body contains a shape() read | None of these modifiers |
| [#679](https://github.com/Chelis-Lang/chelis/issues/679) | Surf grad lane: arange has no AD adjoint — computed index tensors re-materialize as F32 in backward (gather/scatter reject); Surf-lane sibling of #570/#601 | None of these modifiers |
| [#906](https://github.com/Chelis-Lang/chelis/issues/906) | eval: stack-overflow abort on large flat list literals (~2-4k elements) — kills the literal-baking data path | None of these modifiers |
| [#1225](https://github.com/Chelis-Lang/chelis/issues/1225) | Std.Io.Csv parse_line_chars recurses per character: a valid CSV line beyond the lane's stack budget crashes try_read_csv (eval ~4 KiB SIGABRT, compiled ~100 KiB SIGSEGV) | None of these modifiers |
| [#1416](https://github.com/Chelis-Lang/chelis/issues/1416) | Std.Tensor.Construct rank-polymorphic reshape exports reject concrete tensors | None of these modifiers |
| [#1489](https://github.com/Chelis-Lang/chelis/issues/1489) | `copy`, `cast`, `gather` and `concat` reject a not-yet-resolved type variable where ~71 other gates defer | None of these modifiers |
| [#1639](https://github.com/Chelis-Lang/chelis/issues/1639) | Builtin aliases lose mixed scalar/tensor diagnostic ownership | None of these modifiers |
| [#1748](https://github.com/Chelis-Lang/chelis/issues/1748) | C emission loses verified call authority for a named nullary callback | None of these modifiers |
| [#1760](https://github.com/Chelis-Lang/chelis/issues/1760) | Rank-polymorphic Bool results compile with a rank-zero iteration domain | None of these modifiers |
| [#1761](https://github.com/Chelis-Lang/chelis/issues/1761) | Generic where subtree mixes numeric and Bool precision during C lowering | None of these modifiers |
| [#1768](https://github.com/Chelis-Lang/chelis/issues/1768) | Deep checker accepts a negative literal dimension extent (d-lit {} -1) and scores it 1; the wire carrier rejects the same type | None of these modifiers |
| [#1891](https://github.com/Chelis-Lang/chelis/issues/1891) | Qualified exported record construction rejects at LBrace before module resolution | None of these modifiers |
| [#1901](https://github.com/Chelis-Lang/chelis/issues/1901) | eval reports `lowered root count mismatch` for a tuple match inside a lambda passed through a generic apply, where C reports a typed [04-TOT-3] receipt | None of these modifiers |
| [#1908](https://github.com/Chelis-Lang/chelis/issues/1908) | Eval renders runtime shrink-bound failures through private diagnostics where C prints the canonical Domain trap | None of these modifiers |
| [#1917](https://github.com/Chelis-Lang/chelis/issues/1917) | omitted_extent_claim_contract fails on main since #1895: eval loses the symbolic binding `left` at an inlined root over helper results | None of these modifiers |
| [#1921](https://github.com/Chelis-Lang/chelis/issues/1921) | A fourth host-lowering entry point can construct a session per ask and reintroduce chelis#1829's cost with no compile error and no guard | None of these modifiers |
| [#1977](https://github.com/Chelis-Lang/chelis/issues/1977) | Chained shape-sourced rank-4 insert helpers regress to conflicting synthetic dimension aliases | None of these modifiers |
| [#1978](https://github.com/Chelis-Lang/chelis/issues/1978) | Symbolic ReLU grad shim regresses to incompatible dimensions in verified backward DAG | None of these modifiers |
| [#1985](https://github.com/Chelis-Lang/chelis/issues/1985) | C build succeeds but omits executable entry for a closed tensor-gradient program | None of these modifiers |
| [#2103](https://github.com/Chelis-Lang/chelis/issues/2103) | Transforms eagerly evaluate an untaken scalar-if overflow branch | None of these modifiers |
| [#2152](https://github.com/Chelis-Lang/chelis/issues/2152) | `chelis build` rejects a cast to a `Float`-bounded binder in an imported generic def, even at a concrete f32 call | None of these modifiers |
| [#2185](https://github.com/Chelis-Lang/chelis/issues/2185) | Wire census Deftype invariant mutation loses the `future_form` rejection reason | None of these modifiers |
| [#2203](https://github.com/Chelis-Lang/chelis/issues/2203) | chelis build rejects a recursive function that takes a function-typed parameter; check and eval accept it | None of these modifiers |
| [#2307](https://github.com/Chelis-Lang/chelis/issues/2307) | Std.Io.Json recurses per character and per element: 1,209 bytes aborts chelis eval, and the compiled binary segfaults silently below a GPT-2 vocabulary | None of these modifiers |
| [#2374](https://github.com/Chelis-Lang/chelis/issues/2374) | eval: correct local ascription on pad_sequences_to panics at verified local ascription producer | None of these modifiers |
| [#2377](https://github.com/Chelis-Lang/chelis/issues/2377) | Eval/C disagree on insert-result guard ordering against an earlier trap | None of these modifiers |
| [#2379](https://github.com/Chelis-Lang/chelis/issues/2379) | C scalar grad rejects a callee with local bindings although the equivalent direct expression builds | None of these modifiers |
| [#2418](https://github.com/Chelis-Lang/chelis/issues/2418) | chelis build refuses sum over a recursive tensor function's result (builtin sum on host emission), while check and eval accept | None of these modifiers |
| [#2455](https://github.com/Chelis-Lang/chelis/issues/2455) | Exhaustiveness is not enforced for a nested wildcard (&#124; Some(_) alone) or for scalar matches: both check at score 1 and trap at run time | None of these modifiers |
| [#2525](https://github.com/Chelis-Lang/chelis/issues/2525) | vmap silently unmaps a mapped extent operand (Expand size, SplitN count) instead of refusing the ragged result | None of these modifiers |
| [#2546](https://github.com/Chelis-Lang/chelis/issues/2546) | C build of a chelis-std initialiser over a to_tensor(map(..., range(...))) template aborts: expected 1 inputs, got 2 (eval runs) | None of these modifiers |
| [#2565](https://github.com/Chelis-Lang/chelis/issues/2565) | check: a package module importing a sibling module's value is unbound when its file name sorts first | None of these modifiers |
| [#2605](https://github.com/Chelis-Lang/chelis/issues/2605) | C build of a function reading a rank-0 top-level tensor emits C that does not compile (float vs chelis_tensor*) | None of these modifiers |
| [#2606](https://github.com/Chelis-Lang/chelis/issues/2606) | eval_selected of a value that reads an external input reports a cyclic top-level definition | None of these modifiers |
| [#2610](https://github.com/Chelis-Lang/chelis/issues/2610) | chelis eval --file aborts with a stack overflow in resugar on a 1,000-deep Deep program that check accepts | None of these modifiers |
| [#2614](https://github.com/Chelis-Lang/chelis/issues/2614) | surf, build, fmt and cost abort on Deep nesting that check and prove accept: their front end runs on the main thread | None of these modifiers |
| [#2618](https://github.com/Chelis-Lang/chelis/issues/2618) | libchelis_runtime.a and the receipt archive_sha256 differ between two builds of the same head | None of these modifiers |
| [#2623](https://github.com/Chelis-Lang/chelis/issues/2623) | linearity: a top-level bare-variable binding (z = x) is accepted after x was consumed | None of these modifiers |
| [#2625](https://github.com/Chelis-Lang/chelis/issues/2625) | vmap over a generic helper that reshapes a local function's result fails: missing extent witness on eval, undeclared t17_data in C | None of these modifiers |
| [#2636](https://github.com/Chelis-Lang/chelis/issues/2636) | A result claim of a callee inlined into a tensor kernel is dropped; placing its guard needs per-node activation (#2586) | None of these modifiers |
| [#2638](https://github.com/Chelis-Lang/chelis/issues/2638) | lower_app's 'lower func and args, return last' fallback turns an unresolved call into its last argument (silent) | None of these modifiers |
| [#2644](https://github.com/Chelis-Lang/chelis/issues/2644) | Extent claims through a tuple are unchecked: a binder witnessed only by a tuple parameter, and a claim on a tensor inside a tuple result | None of these modifiers |
| [#2645](https://github.com/Chelis-Lang/chelis/issues/2645) | A strided extent restamped under another binder fails untyped and differently per lane (Surf form of #2512's symptom) | None of these modifiers |
| [#2648](https://github.com/Chelis-Lang/chelis/issues/2648) | grad of a recursive-group member checks but hangs eval and exhausts memory in the C build | None of these modifiers |
| [#2649](https://github.com/Chelis-Lang/chelis/issues/2649) | Compiler-API whole-program C releases a top-level value that a function still reads (heap handle has no live strong owner) | None of these modifiers |
| [#2651](https://github.com/Chelis-Lang/chelis/issues/2651) | vmap passes an unresolved result or parameter type through unbatched: order-dependent in recursive groups, unsound when the variable later binds to a tensor | None of these modifiers |
| [#2654](https://github.com/Chelis-Lang/chelis/issues/2654) | An unguarded restamp that is an independent kernel root beside loads declaring the same name: eval returns it silently, C fails untyped (lane divergence) | None of these modifiers |
| [#2662](https://github.com/Chelis-Lang/chelis/issues/2662) | reshape of a scalar returned by a function checks but fails in eval and C: a staged host value has no tensor representation | None of these modifiers |
| [#2664](https://github.com/Chelis-Lang/chelis/issues/2664) | dict_keys and dictionary enumeration use insertion order, not [05-OP-56]'s canonical sorted order, in eval and C | None of these modifiers |
| [#2668](https://github.com/Chelis-Lang/chelis/issues/2668) | eval: the #2583 join refusal rejects runtime ifs over independently sized * arms that main ran | None of these modifiers |
| [#2669](https://github.com/Chelis-Lang/chelis/issues/2669) | A runtime if joining a split_keys result with a runtime count hits the chelis#1482 extent-source refusal | None of these modifiers |
| [#2671](https://github.com/Chelis-Lang/chelis/issues/2671) | eval: under vmap, a nested runtime if shape-checks a branch that only inactive rows select | None of these modifiers |
| [#2672](https://github.com/Chelis-Lang/chelis/issues/2672) | Design: what a Reef package's compiled context is keyed on once only reached stdlib modules are linked (A1/A2/A3) | None of these modifiers |
| [#2691](https://github.com/Chelis-Lang/chelis/issues/2691) | Surf line comments consume CR-only following lines | None of these modifiers |
| [#2734](https://github.com/Chelis-Lang/chelis/issues/2734) | C host elementwise tensor emission omits admitted dtypes and uses f32-only unary functions | None of these modifiers |
| [#2740](https://github.com/Chelis-Lang/chelis/issues/2740) | List-grad C lowering substitutes a top-level List for a local argument | None of these modifiers |
| [#2746](https://github.com/Chelis-Lang/chelis/issues/2746) | Nested vmap checks but cannot lower in Eval or C | None of these modifiers |
| [#2773](https://github.com/Chelis-Lang/chelis/issues/2773) | A gather in an untaken if arm still checks its index and aborts, but only when the if is let-bound | None of these modifiers |
| [#2837](https://github.com/Chelis-Lang/chelis/issues/2837) | eval: a top-level dim declaration makes chelis eval fail with 'lowered root count mismatch' | None of these modifiers |
| [#2840](https://github.com/Chelis-Lang/chelis/issues/2840) | chelisup install and Reef GitHub fetches require a GitHub token even for public releases | None of these modifiers |
| [#2869](https://github.com/Chelis-Lang/chelis/issues/2869) | Package modules: a match-arm binder named like a top-level binding is captured by it (false binding cycle; #851 residual) | None of these modifiers |
| [#2870](https://github.com/Chelis-Lang/chelis/issues/2870) | to_float in the compiled C lane disagrees with [05-OP-59] on overflow, whitespace and non-finite spellings | None of these modifiers |
| [#2874](https://github.com/Chelis-Lang/chelis/issues/2874) | A comparison applied through a lambda gets no bool result type | None of these modifiers |
| [#2884](https://github.com/Chelis-Lang/chelis/issues/2884) | Compiled C program using Std.Sort's sort segfaults | None of these modifiers |
| [#2893](https://github.com/Chelis-Lang/chelis/issues/2893) | Compiled C rejects and of comparisons over two tensors that share a declared dimension | None of these modifiers |
| [#2916](https://github.com/Chelis-Lang/chelis/issues/2916) | chelis build --target c leaves a zoned period root with unrealized host parameters | None of these modifiers |
| [#2925](https://github.com/Chelis-Lang/chelis/issues/2925) | Compiled C releases a record argument read twice through an accessor in one tensor expression | None of these modifiers |
| [#2929](https://github.com/Chelis-Lang/chelis/issues/2929) | Compiled C rejects i8 and i16 sparse indices that [05-SPARSE-1] admits | None of these modifiers |
| [#2938](https://github.com/Chelis-Lang/chelis/issues/2938) | A tuple-element runtime extent fails on both lanes in any def with a declared tensor result | None of these modifiers |
| [#2939](https://github.com/Chelis-Lang/chelis/issues/2939) | chelis build refuses sum over a non-recursive helper's tensor result (builtin sum on host emission), while check and eval accept | None of these modifiers |
| [#2954](https://github.com/Chelis-Lang/chelis/issues/2954) | Inlined callees see the caller's dimension, precision and rank maps by name during lowering | None of these modifiers |
| [#2974](https://github.com/Chelis-Lang/chelis/issues/2974) | sum/mean/max_reduce/min_reduce/prod_reduce reject two positional axes on a concrete-rank tensor that spec/04 §4.5.3 admits (only count takes the multi-axis positional path) | None of these modifiers |
| [#2977](https://github.com/Chelis-Lang/chelis/issues/2977) | uniform_like rejects an f64 template with f64 bounds and accepts f32 bounds instead, against [05-OP-8] | None of these modifiers |
| [#2983](https://github.com/Chelis-Lang/chelis/issues/2983) | chelis eval rejects sum over an empty axis; [05-OP-30] and compiled C give exact zero | None of these modifiers |
| [#2985](https://github.com/Chelis-Lang/chelis/issues/2985) | sum's explicit accumulator argument does not parse: sum(x, 0i32, accumulator=f64) is "expected RParen, found Eq" | None of these modifiers |
| [#2986](https://github.com/Chelis-Lang/chelis/issues/2986) | to_list rejects every tensor of rank above one, contrary to [05-OP-57] | None of these modifiers |
| [#2987](https://github.com/Chelis-Lang/chelis/issues/2987) | linear is unbound and cross_entropy's signature and result differ from spec/05 §3.4 and §4.3 | None of these modifiers |
| [#2994](https://github.com/Chelis-Lang/chelis/issues/2994) | grad through cumsum is rejected in eval and C ("cumsum has no numeric IR lowering") at every float dtype | None of these modifiers |
| [#2999](https://github.com/Chelis-Lang/chelis/issues/2999) | tensor_scan rejects a tensor state ("element type must be a scalar primitive") that [05-OP-38] admits | None of these modifiers |
| [#3000](https://github.com/Chelis-Lang/chelis/issues/3000) | Borrowed comparisons: where(lt(&x, &y), x, y) fails "operand type never determined", and lt(&x, &y) has no C tensor emission arm | None of these modifiers |
| [#3012](https://github.com/Chelis-Lang/chelis/issues/3012) | Opaque-invariant validation depends on the entry module's import set, and misreports an in-module constant as not one | None of these modifiers |
| [#3015](https://github.com/Chelis-Lang/chelis/issues/3015) | Compiled C: a function literal with a local tensor ascription in a host arm fails with an ownership-emission internal error | None of these modifiers |
| [#3019](https://github.com/Chelis-Lang/chelis/issues/3019) | chelis eval resolves a call to a selectively imported name instead of the local def that shadows it | None of these modifiers |
| [#3025](https://github.com/Chelis-Lang/chelis/issues/3025) | reef: export-bundle records a stale lock's chelis-std hashes for the runtime it extracts | None of these modifiers |
| [#3032](https://github.com/Chelis-Lang/chelis/issues/3032) | Compiled C leaks both tensors of every sort result | None of these modifiers |
| [#3038](https://github.com/Chelis-Lang/chelis/issues/3038) | chelis eval rejects a builtin-named definition in a package module that check and build accept (spec/04 §8.6) | None of these modifiers |
| [#3044](https://github.com/Chelis-Lang/chelis/issues/3044) | sort on a bool tensor passes the checker: eval sorts it, compiled C fails at run time ([05-OP-53] admits arithmetic dtypes only) | None of these modifiers |
| [#3079](https://github.com/Chelis-Lang/chelis/issues/3079) | Surf signed minimum literals reject the inner positive magnitude at their declared width | None of these modifiers |
| [#3092](https://github.com/Chelis-Lang/chelis/issues/3092) | eval prints the first site value where compiled C traps a disagreeing later ascription under a rank-polymorphic (..r) signature | None of these modifiers |
| [#3097](https://github.com/Chelis-Lang/chelis/issues/3097) | runtime_bundle_oracle runtime-mutation restore check depends on the ambient RUSTC_WRAPPER/incremental settings: same head fails without kache, passes with it | None of these modifiers |

# Source scanner fork optimization

Reviewed primary-checkout source at `af5dd14c7ac6187a63e435b73551bbe3376865ce`, verified to equal `origin/main`. Static analysis only: no builds, timing probes, repository edits, or inspection of the author's performance worktree. The supplied timings are accepted evidence, not rederived here.

## Recommendation

In `crates/chelis-compiler-api/src/source_arch.rs:536`, make `CallCollector::fork()` initialize `paths: Vec::new()` instead of `paths: self.paths.clone()`. Add a short comment stating that callers install the intended execution frontier before visiting. This is the smallest justified change: every current caller overwrites that field before reading it.

An explicit `fork_with_paths(paths: Vec<CallPath>)` is a stronger API if preferred: initialize the field from the supplied argument and replace the assignments below with arguments. It mechanically prevents a future caller from assuming that `fork` copied the execution paths, but changes more lines and requires moving closure-path construction before the fork. The empty-vector change is sufficient for the present implementation.

This removes redundant work proportional to the ambient frontier at each fork; it does not reduce the frontier or change the number of abstract paths. The semantic equivalence is established by the use-before-overwrite audit below. A substantial timing improvement is a hypothesis until measured.

## Complete call-site audit

All line numbers refer to the reviewed head. There are exactly nine calls to `self.fork()`.

| Site | Installed paths | Operations between fork and overwrite |
| --- | --- | --- |
| `closure_target`, line 654 | `collector.paths = vec![closure_path]` at 683 | Construct a new `CallPath` from the explicit `path` argument's bindings, increment collector depth, and bind closure parameters into that new path. None reads `collector.paths`. |
| `evaluate_expression_value`, line 746 | `collector.paths = vec![path]` immediately | None. |
| `evaluate_match_value`, line 865 | `branch.paths = vec![scrutinee_path.clone()]` at 868 | Increment binding depth and save it. |
| `evaluate_block_value`, line 908 | `branch.paths = vec![path]` at 914 | Clone import and type-alias maps, push block identity, increment and save depth. |
| `visit_local` let-else divergence, line 1482 | `branch.paths = vec![evaluated_path.clone()]` immediately | None. |
| `visit_expr_while`, line 1675 | `body.paths = inputs` immediately | None. |
| `visit_expr_for_loop`, known iterable, line 1735 | `body.paths = iteration_paths` immediately | None. |
| `visit_expr_for_loop`, unknown iterable, line 1790 | `body.paths = inputs` immediately | None. |
| `visit_expr_loop`, line 1845 | `body.paths = inputs` immediately | None. |

For the explicit-argument API, each right-hand side in this table is the corresponding `fork_with_paths(...)` argument. At the closure site, construct the closure path first, use `let closure_depth = self.binding_depth + 1` while binding its parameters, then initialize the collector with that path and set its depth to `closure_depth`.

## Why the guard cannot lose a two-stage pipeline

The old clone is never an input to a transfer operation. Immediately before each branch's first analysis operation, all collector fields have the same values with either implementation: the metadata is copied identically, and the execution frontier comes from the same explicit argument or loop-input vector. From that point onward, identical visitors execute over identical paths. Therefore the collected `FunctionFacts`, helper propagation inputs, and final `Finding` values remain identical, including call multiplicities, stage histories, aliases, scope depths, flow labels, and branch correlations.

There is no aliasing change. The old frontier is an owned vector of cloned plain data and is discarded on assignment. Its `Clone`/drop operations do not execute source-program code. In the author's separate `Rc` experiments, omitting an unused clone also leaves final ownership relationships equivalent before analysis starts; it only avoids temporary reference-count increments and decrements.

This argument establishes equivalence to the current scanner, rather than a new claim that the scanner completely models Rust execution.

## Where cost can disappear

Many visitor methods first `mem::take` the current paths, so some existing forks clone an empty vector already. The redundant copy is still real where the parent retains paths: `evaluate_block_value` evaluates its tail through a collector retaining the entire block frontier; match-arm guard/body evaluation retains branch paths; and `visit_expr_loop` retains its original frontier while forking loop iterations. Nested expression evaluation may repeat those copies. The parent's measurements establish that path collection is expensive, but do not yet quantify this particular cause. Preserve the measured `apply_transform` path count and compare wall time before claiming an improvement.

## Existing validation

Run the existing `source_arch` tests and the production `guarded_upper_consumers_do_not_recreate_the_semantic_pipeline` test on the candidate. Relevant rejecting witnesses include `helper_composed_stage_sequence_is_rejected`, `repeated_conditional_helper_calls_preserve_call_multiplicity`, `branch_result_callable_aliases_cannot_bypass_the_guard`, `invoked_closure_contributes_its_stages`, `higher_order_closure_flow_cannot_bypass_the_guard`, `repeated_while_iterations_can_compose_stages`, `repeated_for_iterations_can_compose_stages`, and `a_break_path_continues_after_the_loop`.

Corresponding allowed-path controls include `mutually_exclusive_dispatch_paths_do_not_compose_stages`, `branch_result_callable_aliases_preserve_path_correlation`, `uninvoked_callable_bodies_are_not_execution_paths`, `lexical_shadow_does_not_reuse_an_outer_function_alias`, `early_return_keeps_execution_paths_separate`, `invariant_loop_conditions_do_not_create_impossible_stage_paths`, and `known_finite_for_loops_preserve_iteration_count_and_values`. The macro-scope and typed-receiver tests cover copied metadata remaining intact. No new semantic test is necessary for a discarded-copy removal; identical full findings and path-count instrumentation, if already available, are stronger targeted evidence than a new test duplicating the implementation.

No broader state pruning is recommended in this bounded change. Removing call histories, joining unequal paths, or filtering nominally harmless helpers would require additional proofs about higher-order arguments and correlation; none is needed for the fork fix.

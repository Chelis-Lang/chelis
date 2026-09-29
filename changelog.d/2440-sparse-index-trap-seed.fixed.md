A discarded `gather`, `scatter`, `scatter_replace`, `scatter_elements` or
`one_hot` that sits under no activation is no longer eliminated, so its
out-of-bounds index reports in the evaluator and in compiled C instead of
vanishing. `spec/06-transformations.md` §5.2 makes a potentially trapping node
an observable root, and a sparse index is data, so no static fact rules the
failure out for a non-literal one. These five operations now form their own
run-time check class, which the one seed predicate reads, and each member is
pinned individually against reclassification.

The same §5.2 also says a node whose activation is false checks nothing, and
this is the one checking class with no activation gate in any lane. An
activated sparse node is therefore left to ordinary value reachability rather
than seeded: seeding it would make a discarded out-of-bounds index in an
untaken `if` arm abort a correct program. Closing that half needs an
inactive-value contract for the class in the evaluator and the C emitter, which
chelis#2440 still tracks, along with an activated sparse node's index check,
which a discarded node still does not run.

Retaining a sparse node also makes the value declaration holding it
re-lowered at each reference rather than shared, which §5.2 requires of a
declaration whose initializer can trap. Results are unchanged and the public C
declaration is unchanged, but emitted C grows for a program that references
such a value repeatedly.

The checker rejects an out-of-range literal index only when the base tensor is
also a literal, so a provably in-range index is retained too and its plan is
emitted. Over-retention is the safe direction; a static index-range refinement
is not attempted here.

Because the node is now retained, a lane that cannot lower it refuses a program
whose sparse operation is dead, where before it refused only one whose sparse
operation was live. `chelis build --target metal` rejects any `gather` or
`scatter` this way ([#1383](https://github.com/Chelis-Lang/chelis/issues/1383)),
and a `vmap`ped sparse operation fails to lower in the C lane and misreports its
index in the evaluator
([#2772](https://github.com/Chelis-Lang/chelis/issues/2772)). Both are
pre-existing lane limits now reachable from dead code; the default C target and
HIP are unaffected. See
[#2440](https://github.com/Chelis-Lang/chelis/issues/2440).

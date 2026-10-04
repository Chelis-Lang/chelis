# Frozen nominal-import test identity

#3150 preserves the executable acceptance selection after #3043 changed the
import collision fixtures to follow spec/02 §P2. The Phase 1 frozen manifest
still requires `stdlib_closure_tests::local_nominal_names_shadow_imported_names`.
Its deletion made the unchanged Phase 2 oracle stop before executing the shared
capacity-census target.

The restored historical identity executes both existing descriptive tests:
qualified-only imports leave the local nominal type nonnumeric, and an
unqualified imported/local collision rejects with the named collision diagnostic.
The name does not restore precedence semantics. Both descriptive identities and
their original assertions remain enabled. No frozen manifest, oracle selection,
classifier, production code or normative rule changes.

Acceptance is the complete unchanged
`scripts/runtime_representation_oracle.py --phase 2`, plus focused execution of
all three nominal-name tests. The negative diagnostic and positive capacity
assertions must execute through the frozen identity as well as independently.

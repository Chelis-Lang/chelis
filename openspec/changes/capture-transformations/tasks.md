## 1. Capture requirements

- [ ] 1.1 Record the DAG-to-DAG model and `grad` signature/`wrt`/algorithm/accumulation
- [ ] 1.2 Record non-differentiable handling, symbolic-dim adjoints, and higher-order derivatives
- [ ] 1.3 Record static match/if and field-wise ADT gradients
- [ ] 1.4 Record `vmap`, `jit`, and the optimization passes
- [ ] 1.5 Record composition/ordering and the transform error contract

## 2. Validate and sync

- [ ] 2.1 Run `openspec validate --all --strict --no-interactive` and fix until green
- [ ] 2.2 Sync the delta into `openspec/specs/transformations/spec.md`

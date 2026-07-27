## 1. Capture requirements

- [ ] 1.1 Record RISC philosophy, no-broadcasting, primitives-as-functions, and two tiers
- [ ] 1.2 Record division semantics, elementwise unary precision/adjoints, and reductions
- [ ] 1.3 Record windowed reductions, movement ops/runtime bounds, and memory/shape primitives
- [ ] 1.4 Record effectful primitives, seed determinism, scatter determinism, and host-only builtins
- [ ] 1.5 Record standard lowerings, AD completeness/oracle, unsupported-case, and observation contracts

## 2. Validate and sync

- [ ] 2.1 Run `openspec validate --all --strict --no-interactive` and fix until green
- [ ] 2.2 Sync the delta into `openspec/specs/risc-primitives/spec.md`

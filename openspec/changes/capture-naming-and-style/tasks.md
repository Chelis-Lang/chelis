## 1. Capture requirements

- [ ] 1.1 Record hard language constraints (case-split, single-letter override, module-path lowering, keywords, Deep tags/symbols, formatter/backend fidelity)
- [ ] 1.2 Record filesystem/manifest and Surf/Rust/Python identifier conventions
- [ ] 1.3 Record module ladder, function prefix/suffix, documentation, and test naming conventions
- [ ] 1.4 Record the `chelis lint` severity model, style-gate contract, and opaque-domain-construction discipline
- [ ] 1.5 Link §12.2 to the existing `lint-traversal-policy` capability without duplicate requirements

## 2. Validate and sync

- [ ] 2.1 Run `openspec validate --all --strict --no-interactive` and fix until green
- [ ] 2.2 Sync the delta into `openspec/specs/naming-and-style/spec.md`

## 1. Contract

- [ ] 1.1 Add the `spec-authority-migration` capability delta defining per-chapter transfer, capture provenance, recorded divergence, supersession marking, and the blocked-by-in-force-document rule.
- [ ] 1.2 Record the atom-ID and `spec/design/` questions as open rather than deciding them here.

## 2. Unblock transfer

- [ ] 2.1 Amend `spec/design/spec_provenance.md` § OpenSpec boundary, which currently denies OpenSpec authority outright and blocks every transfer under requirement 5.
- [x] 2.2 Amend the AGENTS.md documentation hierarchy so it describes authority per subject rather than ranking `spec/00-11*.md` above `openspec/` unconditionally.
- [x] 2.3 Reconcile the AGENTS.md OpenSpec section, which states `spec/**` is controlling without qualification.

## 3. Pilot transfer

- [ ] 3.1 Transfer `spec/11-ffi.md` to the `ffi` capability: review the capability against the chapter, record divergence, mark the chapter superseded.
- [ ] 3.2 Transfer `spec/10-serialization.md` to the `serialization` capability the same way.
- [ ] 3.3 Both are the smallest captured chapters (4 requirements each) and are deliberately the pilot. Do not transfer a further chapter until both have landed and the process has been reviewed.

## 4. Validation

- [ ] 4.1 `openspec validate --all --strict --no-interactive` is green.
- [ ] 4.2 Every captured capability carries a source citation, so no capability is silently ineligible.
- [ ] 4.3 After the pilot, confirm a reader can answer "which tree controls this subject?" for a transferred chapter and an untransferred one without reading both trees.

## 1. Capture requirements

- [ ] 1.1 Record backend strategy/separation and the C reference backend
- [ ] 1.2 Record the shared ABI, source-to-source GPU model, and per-backend dtype gating
- [ ] 1.3 Record backend-boundary effect checks, backend selection/escalation, and all-backend invariants

## 2. Validate and sync

- [ ] 2.1 Run `openspec validate --all --strict --no-interactive` and fix until green
- [ ] 2.2 Sync the delta into `openspec/specs/backends/spec.md`

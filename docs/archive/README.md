# Docs Archive

Point-in-time documents live here so top-level `docs/` stays focused on active runbooks
and current status. Everything under this directory is a historical record: it describes
the project as it was when written, not current behavior. The numbered chapters in
[`spec/`](../../spec/) and the active documents in `docs/` own current behavior.

- [Verification stack handover](verification_stack_handover.md) — July 2026 stack snapshot; current proof design is in [`chelis_property_spec.md`](../../spec/design/chelis_property_spec.md).
- [Compiler cleanup lock](../../spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md) — 0.7.8 workstream design; current linearity rule is in [`spec/04-type-system.md`](../../spec/04-type-system.md).
- May compiler gap reports: [findings](reports/identified_gaps.md), [synthesis](reports/gap_synthesis.md), and [closure analysis](reports/initial_gap_report_closure_analysis.md).
- Historical plans: [July execution](chelis_plan_execution.md) and [completed maintenance schedule](maintenance_schedule.md); current sequence is in [`chelis_project_plan.md`](../../spec/design/chelis_project_plan.md).
- [Phase 1e benchmark capture](perf/phase1e/RESULTS.md) — point-in-time comparison.
- [Retired mascot](mascot/README.md): replaced by the line art in [`docs/assets/brand/`](../assets/brand/).
- [Deep 0.20 pipe migration](pipe_migration_0_20.md) — version-specific source conversion notes.

Other archives: [`investigations/`](investigations/README.md) (resolved bug diagnoses and review notes), `perf/` (dated performance reports), `rca/` (incident analysis),
`red-team/` (review records), `reports/` (one-off reports), `snapshots/`
(repository archaeology), and `mascot/` (retired art).

# Fitness `UnitInterval` API Review

## Scope

This review covers `FitnessReport::score` and its current Rust and wire boundaries. This review does not change fitness code.

The review also records constraints for a separate `UnitInterval` proposal. That proposal must define its complete API before implementation.

## Current Construction

`FitnessReport` is public in `chelis-types`. The public fields include `score: f64`.

The repository contains two `FitnessReport` struct expressions. Both expressions are in `crates/chelis-types/src/fitness.rs`.

The main constructors are:

- `FitnessReport::from_infer_result`
- `FitnessReport::from_infer_result_with_structure`
- `clean_fitness_from_stats`
- `rejected_ir_fitness`

External crates can still construct any report state because each field is public. The current repository does not use that capability.

## Current Mutation

`rejected_ir_fitness` updates node counts, the type component, and `score`. These updates keep one rejected report internally consistent.

`assemble_check_json` in `crates/chelis-cli/src/main.rs` subtracts effect and linearity penalties from `score`. Each subtraction clamps the result at zero.

No public method owns these mutations. A private score field will require typed update operations before callers can migrate.

## Current Read and Wire Boundaries

The compiler API copies `FitnessReport::score` into `schema::CheckResult::score`. The schema field remains `f64` and supports serialization.

The CLI writes the score into its existing JSON report. The E2E snippet tool also writes the score into JSON.

The LSP and Cove surfaces read schema or analysis scores as `f64`. These machine-facing and display boundaries must keep their current values.

`FitnessReport` does not derive `Serialize` or `Deserialize`. The compiler API schema remains the machine-facing conversion boundary.

## Invalid States

The field documentation requires a value from `0.0` through `1.0`. The `f64` type also permits these invalid states:

- a value below zero
- a value above one
- negative zero
- positive or negative infinity
- NaN

The current constructors calculate bounded finite values from bounded components. Public construction and mutation do not preserve that fact.

`FitnessComponents` also uses public `f64` fields with the same documented range. A score-only type will not make the complete fitness product valid.

## Numeric Surface Constraints

A future public numeric type must obey the repository numeric surface rules. The proposal must review each public constructor, accessor, and wire conversion.

The type must not create a new machine-facing schema. Existing schema fields must remain `f64` at their current boundaries.

The type must not accept a raw number and repeat range checks in downstream code. It must parse the number once at the earliest boundary.

## Recommendation

Create a separate OpenSpec proposal for `UnitInterval`. Do not add the type as part of the pipeline artifact change.

The proposal must include these decisions:

1. Define an opaque `UnitInterval` with private storage.
2. Parse finite values from `0.0` through `1.0` at the first raw-number boundary.
3. Return `UnitInterval` from the parse operation.
4. Add typed operations for weighted sums and clamped penalty subtraction.
5. Decide whether all `FitnessComponents` fields also use `UnitInterval`.
6. Keep `f64` conversion inside existing compiler API and JSON adapters.
7. Add positive, range-edge, nonfinite, and compile-fail tests.
8. Review all public report fields before private storage replaces them.

The proposal must preserve current JSON bytes and score calculations. It must not change fitness weights or semantic penalties.

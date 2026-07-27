# Dim Canon Properties: add-dim-canon-property-tests

## ADDED Requirements

### Requirement: Equal keys imply equal values
The property suite SHALL check the soundness direction of the contract published in the `DimExprKey` rustdoc (`crates/chelis-ir/src/dag.rs:147-161`): for any two generated `DimExpr` values and any generated binding under which both evaluate successfully, equal normalized keys SHALL imply equal evaluated sizes. The suite SHALL NOT assert the converse. Unequal keys carry no requirement, because `normalized_key` is deliberately incomplete and keeps a quotient structural when divisibility of a symbol is unknown. A failure of the sound direction SHALL name both source expressions, the binding, the shared key, and the two differing values.

#### Scenario: Equal keys agree on value
- **WHEN** two generated expressions produce equal normalized keys and both evaluate to `Ok` under the same binding
- **THEN** the two evaluated sizes are equal and the case passes

#### Scenario: Colliding keys with differing values fail loudly
- **WHEN** two generated expressions produce equal normalized keys but evaluate to different sizes under the same binding
- **THEN** the suite fails and reports both expressions, the binding, the shared key, and both values

#### Scenario: Unequal keys are not a failure
- **WHEN** two generated expressions denote the same size under a binding but produce different normalized keys
- **THEN** the case passes, because completeness is not asserted

#### Scenario: A case where either side does not evaluate is not counted
- **WHEN** either expression evaluates to `Err` under the drawn binding
- **THEN** the case is discarded rather than passed, and is excluded from the usable-case count

### Requirement: The key is invariant over a closed family of value-preserving rewrites
The suite SHALL assert that applying any sequence of rewrites drawn from a closed, named family leaves the normalized key unchanged. The family SHALL be exactly: commuting the operands of a `Mul`; reassociating nested `Mul` nodes; multiplying a subterm by `Concrete(1)`; and replacing a subterm `x` with `Div(Mul(x, k), k)` for a concrete `k >= 1`. The family SHALL be enumerated in the test source. No rewrite outside the family SHALL be required to preserve the key, and the suite SHALL NOT assert key equality for value-equal expressions produced by any other means.

#### Scenario: Reordering and reassociating a product preserves the key
- **WHEN** a generated expression is rewritten by commuting and reassociating its `Mul` nodes
- **THEN** the rewritten expression produces a key equal to the original's

#### Scenario: Multiplying by one preserves the key
- **WHEN** a generated expression has `Concrete(1)` introduced as a factor at any position
- **THEN** the rewritten expression produces a key equal to the original's

#### Scenario: Cancelling round trip preserves the key
- **WHEN** a subterm `x` in a generated expression is replaced by `Div(Mul(x, k), k)` for a concrete `k` of at least 1
- **THEN** the rewritten expression produces a key equal to the original's

#### Scenario: A rewrite outside the family carries no key requirement
- **WHEN** two expressions are value-equal because of a divisibility fact not expressible by the listed rewrites
- **THEN** the suite makes no assertion about their keys and does not fail

### Requirement: Generation stays inside the evaluable, non-overflowing domain
The generator SHALL draw symbol bindings of at least 1, because the published contract covers positive-integer-valued dimension expressions only and a zero binding is outside it. The generator SHALL construct divisible quotients by assembling a numerator from a factor multiset and a denominator from a sub-multiset of those factors, rather than by drawing two independent subtrees and discarding the inexact results. The generator SHALL bound concrete values and constructed products so that no evaluated product can overflow `usize`, because `DimExpr::evaluate` multiplies with an unchecked operator (`dag.rs:178`) and would panic in a debug build. That bound SHALL be documented in the test source as a deliberate exclusion, naming the overflow regime as out of scope for this suite.

#### Scenario: Bindings are positive
- **WHEN** the generator draws a binding for any symbol
- **THEN** the drawn value is at least 1

#### Scenario: Quotients are exact by construction
- **WHEN** the generator emits a `Div` node through the constructive path
- **THEN** the numerator evaluates to an exact multiple of the denominator under every drawn binding, and `evaluate` returns `Ok` for both sides

#### Scenario: The overflow regime is not entered
- **WHEN** the generator draws concrete values and assembles products
- **THEN** every evaluated product remains below the documented ceiling and no case panics inside `evaluate`

#### Scenario: A zero binding is never drawn
- **WHEN** the generator produces a binding map for an expression containing symbols
- **THEN** no symbol is bound to zero

### Requirement: A vacuous run fails rather than passing
The suite SHALL count the cases in which both expressions evaluated successfully, and SHALL fail when that count falls below a declared floor expressed as a fraction of cases attempted. The failure SHALL name the observed fraction and the floor. The suite SHALL NOT report success on the basis of cases that were all discarded, because a run that checked nothing is indistinguishable in its exit code from a run that checked everything and found no defect. The floor SHALL be declared in the test source alongside the generator it governs.

#### Scenario: Too many discarded cases fails the run
- **WHEN** the fraction of cases in which both expressions evaluated successfully falls below the declared floor
- **THEN** the suite fails and reports the observed fraction together with the floor

#### Scenario: A productive run passes
- **WHEN** the usable fraction meets or exceeds the floor and every usable case satisfies the properties
- **THEN** the suite passes

#### Scenario: A productive run with a real counterexample still fails
- **WHEN** the usable fraction meets the floor and one usable case violates the soundness property
- **THEN** the suite fails on the counterexample, and the health check does not mask it

### Requirement: Gate runs are deterministic and failures are reproducible
The suite SHALL use a fixed, committed default seed so that a run inside the required `integration` stage is deterministic and cannot introduce an intermittent failure on an unrelated pull request. The seed SHALL be overridable by an environment variable for local and maintainer use. On failure the suite SHALL print the effective seed and the exact invocation needed to reproduce the run. The suite SHALL NOT draw a fresh random seed by default.

#### Scenario: Default invocation is deterministic
- **WHEN** the suite runs twice with no seed override on the same commit and toolchain
- **THEN** both runs exercise the same cases and reach the same result

#### Scenario: Failure output is reproducible
- **WHEN** the suite fails under a seed supplied by the override variable
- **THEN** the output names that seed and the invocation that reproduces the failure

#### Scenario: No fresh randomness in the gate
- **WHEN** the suite runs without the override variable set
- **THEN** it uses the committed default seed and does not draw a new one

### Requirement: Counterexamples become committed example tests
A counterexample discovered by the property suite SHALL be added to `crates/chelis-ir/tests/dim_canon_adversarial.rs` as a named, deterministic test that constructs the offending expressions directly and depends on no property-testing library. The promoted test SHALL be committed in the same change that records the finding. The property suite alone SHALL NOT be treated as the regression guard for a known defect, because a seed change removes the case from the run.

#### Scenario: A found counterexample is pinned deterministically
- **WHEN** the property suite reports a soundness counterexample
- **THEN** a named test constructing those expressions is added to `dim_canon_adversarial.rs` and fails for the same reason without the property library

#### Scenario: Promoted tests survive removal of the property suite
- **WHEN** the property suite and its dev-dependency are removed
- **THEN** every promoted counterexample test still compiles and still runs

#### Scenario: No finding is left only in a seed
- **WHEN** a defect is known and unfixed
- **THEN** its reproduction exists as a committed test rather than as a recorded seed value alone

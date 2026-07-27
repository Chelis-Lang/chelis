# Dim Arith Overflow Policy: prove-dim-canon-overflow-safety

## ADDED Requirements

### Requirement: Concrete-factor folding never creates a false key equality
The canonicalizer SHALL NOT fold a product of concrete dimension factors into a single constant when that product exceeds the range of `usize`. It SHALL instead leave the factors unfolded and structural, so that expressions with different factor multisets retain different keys. Clamping an overflowing product to a saturated maximum SHALL NOT occur, because two products with different true values clamp to the same maximum and the canonicalizer then reports two different sizes as equal. The key MAY become less complete above the overflow threshold; it SHALL NOT become unsound.

#### Scenario: Two distinct overflowing products do not collide
- **WHEN** two expressions whose concrete factors each overflow `usize` are normalized, and their true products differ
- **THEN** their normalized keys are different

#### Scenario: An overflowing product stays structural
- **WHEN** an expression's concrete factors multiply beyond the range of `usize`
- **THEN** the resulting key retains those factors rather than presenting a single folded constant

#### Scenario: Incompleteness above the threshold is acceptable
- **WHEN** two expressions denote the same size and their concrete factors overflow `usize`
- **THEN** differing keys are a permitted outcome and are not a failure

#### Scenario: Overflow does not panic
- **WHEN** the canonicalizer folds concrete factors that overflow
- **THEN** it returns a key and does not panic or abort

### Requirement: Key output is unchanged for every non-overflowing input
For every input whose concrete factor products fit within `usize`, `DimExpr::normalized_key` SHALL return exactly the key it returned before this change. Evidence SHALL be a direct comparison against the pre-change implementation over the existing example corpus and a bounded generator, not an argument from inspection. The pre-change implementation SHALL be retained only for the duration of that comparison and SHALL be removed before the change is complete.

#### Scenario: Existing corpus is unchanged
- **WHEN** the existing canonicalization example tests and unit tests run against the changed implementation
- **THEN** every one passes without modification to its expected values

#### Scenario: Bounded generator agrees exactly
- **WHEN** the changed and pre-change implementations are both applied to generated expressions whose products fit in `usize`
- **THEN** the two keys are equal for every such expression

#### Scenario: No second canonicalizer is left behind
- **WHEN** the change is complete
- **THEN** the pre-change implementation is absent from the source tree

### Requirement: Key equality remains an equivalence relation and a total order
The key type SHALL continue to satisfy the contracts its derived traits require. Equality SHALL be reflexive, symmetric, and transitive, and the ordering SHALL remain a total order. No sentinel or poison key value that compares unequal to itself SHALL be introduced. The type is sorted during atom assembly and during quotient cancellation, and is hashed and stored by the backends, so a violated comparator contract would corrupt cancellation rather than surface as a comparison error.

#### Scenario: Every key equals itself
- **WHEN** any key produced by the canonicalizer, including one from an overflowing product, is compared to itself
- **THEN** the comparison reports equality

#### Scenario: Sorting stays well defined
- **WHEN** a vector of keys including overflow-derived keys is sorted
- **THEN** the sort produces a consistent total order and quotient cancellation over those atoms behaves as it does for ordinary keys

#### Scenario: No poison variant exists
- **WHEN** the key type is inspected
- **THEN** it contains no variant whose equality is defined to fail against itself

### Requirement: The overflow policy of every dimension traversal is documented
The `DimExprKey` rustdoc SHALL state the canonicalizer's overflow policy and SHALL note that `DimExpr::evaluate` and `DimExpr::as_concrete` multiply without checking, so the three traversals of the same tree do not share one policy. The documented contract SHALL NOT continue to assert unconditional mathematical equivalence without naming the bound under which it holds.

#### Scenario: The canonicalizer policy is stated
- **WHEN** a reader consults the `DimExprKey` documentation
- **THEN** it states that overflowing concrete products remain unfolded and that the key is incomplete above that threshold

#### Scenario: The divergence between traversals is visible
- **WHEN** a reader consults the same documentation
- **THEN** it records that the evaluating and concrete-folding traversals use unchecked multiplication and therefore differ from the canonicalizer

#### Scenario: The contract no longer claims an unbounded guarantee
- **WHEN** the documented equivalence claim is read
- **THEN** it names the range within which it holds rather than asserting it for all positive integers

### Requirement: Mechanically verified arithmetic is adopted only under declared conditions
Adoption of a mechanically verified implementation for the dimension arithmetic primitives SHALL require that the verification toolchain install reproducibly through the project's development environment, that it not alter the toolchain the gate jobs install, that the named proof obligations be discharged, and that building and testing this repository not require the verification tool to be present. If any condition fails, the outcome SHALL be recorded as declined with its evidence, and the tested fix SHALL ship regardless. The correctness of the shipped fix SHALL NOT depend on any proof being completed.

#### Scenario: Adoption proceeds when every condition holds
- **WHEN** the toolchain installs through the development environment, the gate toolchain is unaffected, both obligations are discharged, and an ordinary build needs no verification tool
- **THEN** the verified implementation is adopted and the decision is recorded with its evidence

#### Scenario: A failed condition results in a recorded decline
- **WHEN** any one of the declared conditions does not hold
- **THEN** the outcome is recorded as declined together with the condition that failed, and no partial verification lane is left in the tree

#### Scenario: An ordinary build never requires the verification tool
- **WHEN** a contributor without the verification tool builds and tests the workspace
- **THEN** the build and the test suite succeed

#### Scenario: The fix does not depend on the proof
- **WHEN** the verification effort is declined
- **THEN** the overflow fix and its regression tests are still present and still pass

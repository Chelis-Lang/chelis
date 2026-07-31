## 1. Baseline and Failing Tests

- [x] 1.1 Record PR #964 commit `5e81c966` and the seven remaining runtime consumers.
- [x] 1.2 Record that the base already corrects int32 tensor formatting.
- [x] 1.3 Add failing native-int32 `cmplt` tests for negative and non-negative values.
- [x] 1.4 Add failing native-int32 tests for `where` and scatter-add.
- [x] 1.5 Add failing native-int32 tests for `cumsum` and `trace`.
- [x] 1.6 Add failing native-int32 tests for `clamp` and `einsum`.
- [x] 1.7 Add emitter tests for `int32_t`, the f32 control, and the forbidden int32 `float` pointer.
- [x] 1.8 Add debug-boundary tests that accept f32 and bool tensors but reject an int32 tensor.
- [x] 1.9 Restore each old f32-view class with a temporary local mutation.
- [x] 1.10 Record the predicted failure from each temporary mutation, then remove the mutation.

## 2. Runtime Corrections

- [x] 2.1 Split int32 from bool in `chelis_tensor_cmplt` and use the generic `i32` comparison loop.
- [x] 2.2 Give `chelis_tensor_where` a native `i32` condition arm.
- [x] 2.3 Route int32 scatter-add through typed `i32` pointers.
- [x] 2.4 Route int32 `cumsum` and `trace` through their existing generic `i32` loops.
- [x] 2.5 Route int32 `clamp` and `einsum` through their existing generic `i32` loops.
- [x] 2.6 Add debug assertions for valid `data_as_f32` and `data_as_f32_const` representations.

## 3. Generated C Correction

- [x] 3.1 Map `DtypeArm::I32` to `int32_t` in `DtypeArm::elem_t()`.
- [x] 3.2 Keep `DtypeArm::F32` and `DtypeArm::Bool` on their current `float` mapping.
- [x] 3.3 Emit integer comparisons for int32 max and min operations.
- [x] 3.4 Compile a generated exactness probe with int32 values above 2²⁴.
- [x] 3.5 Run the emitter tests against generated elementwise int32 and f32 arms.

## 4. Documentation and Release Note

- [x] 4.1 Replace stale int32 f32-storage comments in the runtime and C host emitter.
- [x] 4.2 Update section C6.2 with physical representations and representation-compatible element access.
- [x] 4.3 Correct `CRuntime-I32Storage-F1` to state native storage and close the listed decode sites.
- [x] 4.4 Add an Unreleased changelog entry for the corrected int32 runtime and generated C results.
- [x] 4.5 Make sure that the numbered specifications retain their existing language semantics.

## 5. Validation

- [x] 5.1 Run `cargo fmt --all -- --check`.
- [x] 5.2 Run the authoritative oracle: `cargo nextest run -p chelis-runtime -p chelis-backend-c`.
- [x] 5.3 Run `python3 scripts/gate.py --local` as supporting repository evidence.
- [x] 5.4 Run `openspec validate --all --strict --no-interactive`.
- [x] 5.5 Run `git diff --check`.
- [x] 5.6 Audit all `RuntimeDType::I32` branches near `data_as_f32` calls.

## 6. Adversarial and Hosted Evidence

- [x] 6.1 Run a fresh-context adversarial review against the proposal, specification, code, and tests.
- [x] 6.2 Correct each valid high-severity finding and rerun the authoritative oracle.
- [ ] 6.3 After pull request creation, require green hosted Linux C tests as GNU C evidence.

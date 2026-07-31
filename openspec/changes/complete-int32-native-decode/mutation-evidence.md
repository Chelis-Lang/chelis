# Mutation Evidence

## Method

A reverse source patch restored all old f32-view paths from commit `5e81c966`.
The tests and OpenSpec artifacts stayed unchanged during this mutation.

The runtime command was:

```text
cargo nextest run -p chelis-runtime --test i32_native_decode
```

The emitter command was:

```text
cargo nextest run -p chelis-backend-c --test host_emit_dtype_dispatch
```

The source patch then restored the corrected implementation. `git diff --check` returned success after the restoration.

## Results

| Old path | Predicted failure | Actual failure |
|---|---|---|
| `cmplt` int32 through `f32` | `-1 < 0` returns false | `[false, false, false]` instead of `[true, false, false]` |
| `where` int32 condition through `f32` | `i32::MIN` selects the else value | `[-10, -20]` instead of `[10, -20]` |
| scatter-add int32 through `f32` | The exact result is not `1` | `[-1000000076]` instead of `[1]` |
| `cumsum` int32 through `f32` | The second prefix is not `1` | `[1000000000, -1000000076]` instead of `[1000000000, 1]` |
| `trace` int32 through `f32` | The diagonal sum is not `1` | `[-1000000076]` instead of `[1]` |
| `clamp` int32 through `f32` | The negative value does not clamp to `-8` | `[-1000000000, 7, 8]` instead of `[-8, 7, 8]` |
| `einsum` int32 through `f32` | `2 * 3` produces zero | `[0]` instead of `[6]` |
| f32 boundary without assertions | Int32 access does not fail | Both panic tests reported `test did not panic as expected` |
| generated int32 arm through `float` | The type and negative tests fail | The int32 arm contained three `float` pointers |

The runtime mutation result was 1 passed and 9 failed. The positive F32 and Bool boundary test remained green.

The emitter mutation result was 12 passed and 2 failed. The F32 control test remained green.

## Red-team regression

A fresh local review found an additional binary32 conversion in generated int32 max and min operations. The pointer type was `int32_t`, but the expression called `fmaxf` or `fminf`.

The review compiled the generated expression. `max(16_777_217, 0)` returned `16_777_216` instead of `16_777_217`.

Two tests then compiled the generated max and min expressions. Before the correction, the focused suite reported 15 passed and 2 failed. After the correction, all 17 tests passed. A later I32 unary-rejection test increased the suite to 18 tests.

A second fresh local review found no remaining issues. Its oracle reported 439 passed and one skipped. The focused emitter and runtime suites reported 18 passed and 10 passed.

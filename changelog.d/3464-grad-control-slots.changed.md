`grad` no longer rejects a piecewise-constant conversion, or another operation
whose atom structurally rejects differentiation, when its result reaches the
differentiated output only through control slots: a comparison operand, a
`where`, `if`, or `match` condition, an index, a movement bound or size, or a
`shape` read. Such an operation runs forward with its traps and contributes
nothing to the gradient, so `where(lt(cast(x, i32), w), x, mul(x, x))` and
`where(lt(floor(x), t), x, mul(x, x))` differentiate like the equivalent float
comparison, on the evaluator and the C lane. Previously each rejected with
`PiecewiseConstant`. A conversion with a data path to the output, including a
float-to-integer round trip used as a value, still rejects. An operation
read only through control slots also no longer contaminates the gradient
through its adjoint: `if lt(mul(x, y), 1.0f32) then x else mul(x, x)` at
`y = inf` returned a NaN gradient and now returns `2x`. See
[#3464](https://github.com/Chelis-Lang/chelis/issues/3464).

The same rejections also apply only to an active operation, one whose operands
depend on a differentiated parameter. A conversion of data that is not
differentiated is a constant, so `grad(f, wrt=x)` of `sum(mul(x, floor(m)))`
returns `floor(m)` instead of rejecting. Dependence passes through every
operand slot, indices included, so a conversion of a value gathered by an
index computed from `x` is active and rejects when used as data. A guard
predicate carries no dependence, and a metadata read such as `shape` carries
only the dependence of its operand's extent, never of its element values; an
extent computed from a value that depends on `x` is not yet tracked
([#3531](https://github.com/Chelis-Lang/chelis/issues/3531)). See
[#3487](https://github.com/Chelis-Lang/chelis/issues/3487).

The structural `grad` rejections (piecewise constant, integer index or
reduction output, signed-integer arithmetic, truncated-quotient jump, logical
operation, and replace-scatter) now say that the operation's result has a data
path to the differentiated output and how to keep it off one, instead of
recommending a `stop_gradient` that programs cannot yet call.

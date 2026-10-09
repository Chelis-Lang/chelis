`grad` no longer rejects a piecewise-constant conversion, or another operation
whose atom structurally rejects differentiation, when its result reaches the
differentiated output only through control slots: a comparison operand, a
`where`, `if`, or `match` condition, an index, a movement bound or size, or a
`shape` read. Such an operation runs forward with its traps and contributes an
exact zero, so `where(lt(cast(x, i32), w), x, mul(x, x))` and
`where(lt(floor(x), t), x, mul(x, x))` differentiate like the equivalent float
comparison, on the evaluator and the C lane. Previously each rejected with
`PiecewiseConstant`. A conversion with a data path to the output, including a
float-to-integer round trip used as a value, still rejects. An operation
read only through control slots also no longer contaminates the gradient
through its adjoint: `if lt(mul(x, y), 1.0f32) then x else mul(x, x)` at
`y = inf` returned a NaN gradient and now returns `2x`. See
[#3464](https://github.com/Chelis-Lang/chelis/issues/3464).

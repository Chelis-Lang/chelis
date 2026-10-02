# First Program

Create `app.ch` with a function that applies ReLU and softmax to a tensor, then
calls that function with three values:

```chelis-surf
def relu_then_softmax[n](x: tensor[n, f32]) -> tensor[n, f32] = x |> relu |> softmax(0)
result = [-1.0, 0.0, 1.0] |> to_tensor |> relu_then_softmax
```

The dimension variable `n` lets the function accept a vector of any length.
`tensor[n, f32]` says that its elements have `f32` precision. The `|>` operator
passes its left value as the first argument of the next call: `x |> softmax(0)`
means `softmax(x, 0)`.

Run these commands in the directory containing `app.ch`:

```sh
chelis fmt --inplace app.ch
chelis check app.ch
chelis eval --file app.ch
```

The evaluator prints `result` as a tensor with shape `[3]`. Its values are
approximately `0.212`, `0.212`, and `0.576`.

To inspect the compiler's Deep representation and build a native executable, run:

```sh
chelis deep app.ch
chelis build app.ch --output out/
./out/app
```

`chelis build` invokes the system C compiler and links its carried runtime.
Use `--emit-c` when you only want generated sources and runtime artifacts.
See [CLI Workflow](cli.md) for the other commands and their output.

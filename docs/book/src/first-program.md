# First program

Create `app.ch` with a function that applies ReLU and softmax to a tensor, then
calls that function with three values:

```chelis-surf
def relu_then_softmax[n](x: tensor[n, f32]) -> tensor[n, f32] = x |> relu |> softmax(0)
result = [-1.0, 0.0, 1.0] |> to_tensor(f32) |> relu_then_softmax
```

The dimension variable `n` lets the function accept a vector of any length.
`tensor[n, f32]` says that its elements have `f32` precision. The `|>` operator
passes its left value as the first argument of the next call: `x |> softmax(0)`
means `softmax(x, 0)`.

`to_tensor(f32)` states the element dtype for the bracket literal. Numeric
literals inside `to_tensor` need either a dtype argument or explicit suffixes;
they do not take a default dtype.

Run these commands in the directory containing `app.ch`:

```sh
chelis fmt --inplace app.ch
chelis check app.ch
chelis eval --file app.ch
```

The evaluator prints each top-level value:

```text
result = tensor(shape=[3], data=[0.21194156, 0.21194156, 0.57611686])
```

ReLU maps `-1.0` to `0.0`, so the first two inputs are equal and receive equal
probability.

`chelis deep app.ch` prints the Deep form the compiler works on. The first two
forms it prints for this file are the signature and body of
`relu_then_softmax`; the pipeline has become nested calls:

```text
(defsig {span: "surf:0..87"}
  relu_then_softmax
  (n)
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))

(def {span: "surf:0..87"}
  relu_then_softmax
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (app {span: "surf:77..87"}
      (var {span: "surf:77..84"} softmax)
      (app {span: "surf:69..73"}
        (var {span: "surf:69..73"} relu)
        (var {span: "surf:64..65"} x))
      (lit {span: "surf:85..86", type: (t-prim {} i32)} 0))))
```

Each `span` is the byte range of the Surf source the node came from. To build
a native executable and run it:

```sh
chelis build app.ch --output out/
./out/app
```

The executable prints the same line as the evaluator:

```text
result = tensor(shape=[3], data=[0.21194156, 0.21194156, 0.57611686])
```

`chelis build` invokes the system C compiler and links its carried runtime.
Use `--emit-c` when you only want generated sources and runtime artifacts.
See [CLI Workflow](cli.md) for the other commands and their output.

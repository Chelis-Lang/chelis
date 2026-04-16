# First Program

Start with a small Surf function and let the compiler show you the canonical Deep form.

## Surf

```chelis-surf
def relu_then_softmax(x: tensor[n, f32]) -> tensor[n, f32] =
  softmax(relu(x), 0)
```

## Deep

```chelis-deep
(defsig {}
  relu_then_softmax
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))

(def {}
  relu_then_softmax
  (fn {}
    (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (app {}
      (var {} softmax)
      (app {} (var {} relu) (var {} x))
      (lit {type: (t-prim {} int32)} 0))))
```

## Useful Commands

Check a program:

```sh
chelis check app.ch
```

Print canonical Deep:

```sh
chelis deep app.ch
```

Convert Deep back to Surf:

```sh
chelis surf app.dp
```

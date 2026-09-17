# First Program

Start with a small Surf function, format it, check it, then ask the compiler to show the
canonical Deep form.

Create `app.ch`:

## Surf

```chelis-surf
def relu_then_softmax[n](x: tensor[n, f32]) -> tensor[n, f32] =
  softmax(relu(x), 0)
```

Then run:

```sh
chelis fmt --inplace app.ch
chelis check app.ch
chelis deep app.ch > app.dp
chelis surf app.dp
```

`check`, `deep`, and `surf` are useful together: Surf stays readable for humans, while
Deep is the stable machine form that shell tooling can inspect.

## Deep

```chelis-deep
(defsig {}
  relu_then_softmax
  (n)
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
      (lit {type: (t-prim {} i32)} 0))))
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

Build source artifacts without invoking a native compiler:

```sh
chelis build app.ch --target c --output out/
```

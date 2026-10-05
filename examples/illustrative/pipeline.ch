def normalize[features](x: tensor[features, f32]) -> tensor[features, f32] = {
  total =
    x
    |> exp
    |> sum(0i32)
    |> insert(0, shape(x, 0i32))
  x |> exp |> div(total)
}
def process[features](x: tensor[features, f32]) -> tensor[features, f32] = x |> relu |> normalize
def batch_process(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] = vmap(process)(xs)

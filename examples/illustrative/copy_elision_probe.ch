-- Keep explicit copy/drop calls: crates/chelis-cli/tests/copy_elision.rs
-- checks their Copy IR and the resulting storage plan.
def fanout(x: tensor[1024, 1024, f32]) -> tensor[1024, 1024, f32] = {
  a = x |> copy |> exp
  b = x |> copy |> log
  c = x |> copy |> sin
  d = x |> copy |> neg
  e = x |> copy |> sqrt
  out = add(add(add(a, b), add(c, d)), e)
  _ = drop(a)
  _ = drop(b)
  _ = drop(c)
  _ = drop(d)
  _ = drop(e)
  out
}

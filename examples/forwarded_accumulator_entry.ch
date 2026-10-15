// The named callback checks its own tensor extent after the wrapper forwards
// the original accumulator and item through immutable local bindings.
def step(acc: List[tensor[3, f32]], item: i64) -> List[tensor[*, f32]] =
  append(acc, to_tensor([1.0f32, 2.0f32, 3.0f32]))

def run(seed: List[tensor[3, f32]]) -> i64 =
  len(fold(fn (acc: List[tensor[*, f32]], item: i64) -> {
    old = acc
    forwarded = old
    index = item
    step(forwarded, index)
  }, seed, range(0i64, 4i64)))

out = run([to_tensor([1.0f32, 2.0f32, 3.0f32])])

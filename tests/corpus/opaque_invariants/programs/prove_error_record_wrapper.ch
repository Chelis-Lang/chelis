module M
export (make_wrapped)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type T = | T { value: f32 }
type Wrapper = | Wrapper { inner: T }
def make_wrapped(x: f32) -> Wrapper = Wrapper { inner: T { value: 99.0 } }

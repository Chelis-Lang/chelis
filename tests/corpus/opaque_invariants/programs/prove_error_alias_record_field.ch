module M
export (make_w)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type T =
  | T { value: f32 }
type TA = T
type Wrapper =
  | Wrapper { inner: TA }
def make_w(x: f32) -> Wrapper = Wrapper { inner: T { value: 99.0 } }

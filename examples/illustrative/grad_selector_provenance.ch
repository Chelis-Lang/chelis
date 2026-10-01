-- Illustrative evaluator/checker coverage: named grad selectors retain the
-- callable's formal names through immutable aliases and nested ADT
-- constructor/record pattern projection. A C build rejects differentiation
-- through functions extracted from constructor or record payloads.
type FnBox =
  | FnBox(f32 -> f32 -> f32)
type Outer =
  | Outer(FnBox)
type FnRecord =
  | FnRecord { callback: f32 -> f32 -> f32 }
type OuterRecord =
  | OuterRecord { payload: FnRecord }
def pair(x: f32, w: f32) -> f32 = mul(x, w)
def call_constructor_payload(unused: bool) -> f32 = {
  _ = unused
  alias = pair
  inner = FnBox(alias)
  boxed = Outer(inner)
  match boxed with {
    | Outer(FnBox(chosen)) => chosen(2.0f32, 3.0f32)
  }
}
def grad_constructor_payload(unused: bool) -> f32 = {
  _ = unused
  alias = pair
  inner = FnBox(alias)
  boxed = Outer(inner)
  match boxed with {
    | Outer(FnBox(chosen)) => grad(chosen, wrt=w)(2.0f32, 3.0f32)
  }
}
def call_record_payload(unused: bool) -> f32 = {
  _ = unused
  alias = pair
  recorded = OuterRecord { payload: FnRecord { callback: alias } }
  match recorded with {
    | OuterRecord { payload: FnRecord { callback: chosen } } => chosen(2.0f32, 3.0f32)
  }
}
def grad_record_payload(unused: bool) -> f32 = {
  _ = unused
  alias = pair
  recorded = OuterRecord { payload: FnRecord { callback: alias } }
  match recorded with {
    | OuterRecord { payload: FnRecord { callback: chosen } } => grad(chosen, wrt=w)(2.0f32, 3.0f32)
  }
}
constructor_direct = call_constructor_payload(true)
constructor_grad = grad_constructor_payload(true)
record_direct = call_record_payload(true)
record_grad = grad_record_payload(true)

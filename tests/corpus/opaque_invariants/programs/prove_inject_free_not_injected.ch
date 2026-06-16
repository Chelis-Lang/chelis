module M.Plain
export (mk)
@opaque
type Token =
  | Token { value: f32 }
def mk(x: f32) -> Token = Token { value: x }
def token_value(t: Token) -> f32 = t.value
@property tok_bounded forall(t: Token):
  token_value(t) <= 1.0

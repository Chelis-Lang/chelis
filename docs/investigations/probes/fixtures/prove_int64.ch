module Audit.ProveInt
export (shift_a, shift_b)
def shift_a(x: int64) -> int64 = add(x, 9007199254740993i64)
def shift_b(x: int64) -> int64 = add(x, 9007199254740992i64)
@property distinct_offsets_stay_distinct forall(x: int64):
  (shift_a(x) != shift_b(x))

module Audit.ProveIntCtl
export (increment_by_one, advance_by_two)
def increment_by_one(x: int64) -> int64 = add(x, 1i64)
def advance_by_two(x: int64) -> int64 = add(x, 2i64)
@property small_offsets_stay_distinct forall(x: int64):
  (increment_by_one(x) != advance_by_two(x))

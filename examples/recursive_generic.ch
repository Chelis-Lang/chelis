def choose[a](x: a, n: i32) -> a = if (n <= 0) then x else choose(x, (n - 1))
def concrete() -> i32 = if choose(true, 2) then choose(5, 3) else 0
out = print(concrete())
type Column[n] =
  | FloatCol(tensor[n, f32])
  | BoolCol(tensor[n, bool])
def join_pairs[n](xs: List[(string, Column[n])], ys: List[(string, Column[n])]) -> List[(string, Column[n])] = if eq(len(ys), 0i64) then xs else join_pairs(append(xs, index(ys, 0i64)), skip(ys, 1i64))
def extract[n](pairs: List[(string, Column[n])], target: string, acc: List[(string, Column[n])]) -> (Column[n], List[(string, Column[n])]) =
  if eq(len(pairs), 0i64) then fail("missing") else {
    hp = index(pairs, 0i64)
    if eq(hp.0, target) then (hp.1, join_pairs(acc, skip(pairs, 1i64))) else extract(skip(pairs, 1i64), target, append(acc, hp))
  }
def reorder[n](pairs: List[(string, Column[n])], order: List[string], acc: List[(string, Column[n])]) -> List[(string, Column[n])] =
  if eq(len(order), 0i64) then acc else {
    name = index(order, 0i64)
    matched = extract(pairs, name, [])
    reorder(matched.1, skip(order, 1i64), append(acc, (name, matched.0)))
  }
def first_value[n](col: Column[n]) -> i64 =
  match col with {
    | FloatCol(xs) => cast(index(to_list(xs), 0i64), i64)
    | BoolCol(xs) => if index(to_list(xs), 0i64) then 1i64 else 0i64
  }
def result() -> i64 = {
  pairs = [("a", FloatCol(to_tensor([1.0f32, 2.0f32]))), ("b", FloatCol(to_tensor([9.0f32, 8.0f32])))]
  ordered = reorder(pairs, ["b", "a"], [])
  add(mul(first_value(index(ordered, 0i64).1), 10i64), first_value(index(ordered, 1i64).1))
}
reordered = print(result())

module KindedNominalDimensions
type Column[n] =
  | Column { items: tensor[n, f32] }
type Pair[n] = Column[n]
def total(value: Pair[2]) -> f32 = add(index(to_list(value.items), 0i64), index(to_list(value.items), 1i64))
def main() -> f32 = total(Column { items: to_tensor([1.0f32, 2.0f32]) })
out = print(main())

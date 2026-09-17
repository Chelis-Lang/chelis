module KindedNominalDimensions
type Column[n] =
  | Column { items: tensor[n, f32] }
type Pair[n] = Column[n]
def total(value: Pair[2]) -> f32 = add(index(to_list(value.items), 0i64), index(to_list(value.items), 1i64))
def main() -> f32 = total(Column { items: to_tensor([cast(1.0, f32), cast(2.0, f32)]) })
out = print(main())

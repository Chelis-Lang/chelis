x = to_tensor([[1.0f32, -2.0f32], [3.0f32, -4.0f32]])
negative = vmap(neg)(x)
absolute = abs
positive = vmap(absolute, axis=1)(x)
finite = vmap(is_finite)(x)
def classify_rows(h: &tensor[*, f32] -> tensor[2, bool], rows: tensor[*, 2, f32]) -> tensor[*, 2, bool] = vmap(h, axis=1)(rows)
checked_rows = classify_rows(is_finite, x)

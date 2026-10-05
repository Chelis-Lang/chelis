def route(expert_weights: tensor[3, 2, f32], x: tensor[3, 2, f32]) -> tensor[3, 2, f32] = {
  ids_list: List[i64] = [0i64, 0i64, 0i64]
  expert_indices = to_tensor(ids_list)
  weights_for_tokens = gather(expert_weights, expert_indices, 0)
  mul(x, weights_for_tokens)
}

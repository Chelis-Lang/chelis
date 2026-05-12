module Std.Nn.Attention
export (scaled_dot_product_attention, multi_head_attention, grouped_query_attention, gqa_broadcast_kv)
sig scaled_dot_product_attention: &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> tensor[4, 4, p]
def scaled_dot_product_attention(q, k, v, scale) = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  scaled = mul(scores, scale)
  weights = softmax(scaled, -1)
  out = matmul(weights, v)
  _ = drop(kt)
  _ = drop(scores)
  _ = drop(scaled)
  _ = drop(weights)
  out
}
sig multi_head_attention: &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> tensor[4, 4, p]
def multi_head_attention(q_head, k_head, v_head, scale) = scaled_dot_product_attention(q_head, k_head, v_head, scale)
sig gqa_broadcast_kv: &tensor[1, 4, 4, p] -> &tensor[2, int64] -> tensor[2, 4, 4, p]
def gqa_broadcast_kv(kv_pool, group_map) = gather(kv_pool, group_map, 0)
sig grouped_query_attention: &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> tensor[4, 4, p]
def grouped_query_attention(q_head, k_head, v_head, scale) = scaled_dot_product_attention(q_head, k_head, v_head, scale)

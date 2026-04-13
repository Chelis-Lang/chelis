module Std.Nn.Attention
export (scaled_dot_product_attention, multi_head_attention, grouped_query_attention, gqa_broadcast_kv)
def scaled_dot_product_attention(q: tensor[4, 4, f32], k: tensor[4, 4, f32], v: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = {
  kt = (permute(k, 1, 0) : tensor[4, 4, f32])
  scores = (matmul(q, kt) : tensor[4, 4, f32])
  scaled = (mul(scores, scale) : tensor[4, 4, f32])
  weights = (softmax(scaled, -1) : tensor[4, 4, f32])
  (matmul(weights, v) : tensor[4, 4, f32])
}
def multi_head_attention(q_head: tensor[4, 4, f32], k_head: tensor[4, 4, f32], v_head: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = scaled_dot_product_attention(q_head, k_head, v_head, scale)
def gqa_broadcast_kv(kv_pool: tensor[1, 4, 4, f32], group_map: tensor[2, int64]) -> tensor[2, 4, 4, f32] = (gather(kv_pool, group_map, 0) : tensor[2, 4, 4, f32])
def grouped_query_attention(q_head: tensor[4, 4, f32], k_head: tensor[4, 4, f32], v_head: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = scaled_dot_product_attention(q_head, k_head, v_head, scale)

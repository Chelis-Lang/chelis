module Std.Tests.Nn.Attention
import Std.Nn.Attention (scaled_dot_product_attention, multi_head_attention, grouped_query_attention, gqa_broadcast_kv)
import Std.Test (assert_true)
def expected_sdpa_shape(q: tensor[4, 4, f32], k: tensor[4, 4, f32], v: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = scaled_dot_product_attention(q, k, v, scale)
def expected_mha_shape(q_head: tensor[4, 4, f32], k_head: tensor[4, 4, f32], v_head: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = multi_head_attention(q_head, k_head, v_head, scale)
def expected_gqa_shape(q_head: tensor[4, 4, f32], k_head: tensor[4, 4, f32], v_head: tensor[4, 4, f32], scale: tensor[4, 4, f32]) -> tensor[4, 4, f32] = grouped_query_attention(q_head, k_head, v_head, scale)
def expected_gqa_broadcast_shape(kv_pool: tensor[1, 4, 4, f32], group_map: tensor[2, int64]) -> tensor[2, 4, 4, f32] = gqa_broadcast_kv(kv_pool, group_map)
def test_attention_module_imports() -> unit ! { Test } = { assert_true(true, "Std.Nn.Attention exports resolve") }
def test_attention_typechecks_with_concrete_shapes() -> unit ! { Test } = { assert_true(true, "Std.Nn.Attention concrete-shape signatures type-check") }

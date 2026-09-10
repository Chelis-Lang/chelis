# Registry: public C tensor-runtime callables ([05-OP-33])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-33] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| callable | exact C signature |
|---|---|
| owned allocation | `chelis_tensor *chelis_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype)` |
| owned allocation with input shape | `chelis_tensor *chelis_tensor_alloc_like(const chelis_tensor *input, chelis_scalar exemplar)` |
| rank | `int32_t chelis_tensor_rank(const chelis_tensor *tensor)` |
| extent | `int64_t chelis_tensor_shape(const chelis_tensor *tensor, int32_t axis)` |
| element count | `int64_t chelis_tensor_numel(const chelis_tensor *tensor)` |
| contiguous stride | `int64_t chelis_tensor_stride(const chelis_tensor *tensor, int32_t axis)` |
| logical byte count | `int64_t chelis_tensor_byte_count(const chelis_tensor *tensor)` |
| tensor iteration index step | `int64_t chelis_tensor_elementwise_index_step(const chelis_tensor *input, const chelis_tensor *domain)` |
| shape iteration index step | `int64_t chelis_tensor_elementwise_index_step_for_shape(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *shape)` |
| reshape validation | `void chelis_tensor_check_reshape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape)` |
| owned reshape | `chelis_tensor *chelis_tensor_reshape(const chelis_tensor *tensor, const chelis_list *shape)` |
| contiguous copy | `chelis_tensor *chelis_contiguous(const chelis_tensor *tensor)` |
| typed list ingress | `chelis_tensor *chelis_tensor_from_values(const chelis_list *list, chelis_dtype dtype)` |
| row-major element egress | `chelis_list *chelis_tensor_elements(const chelis_tensor *tensor)` |
| inferred-width padding | `chelis_tensor *chelis_pad_sequences(const chelis_list *sequences, chelis_scalar pad_value)` |
| fixed-width padding | `chelis_tensor *chelis_pad_sequences_to(const chelis_list *sequences, int64_t width, chelis_scalar pad_value)` |
| concatenate | `chelis_tensor *chelis_tensor_concat(const chelis_list *parts, int32_t axis)` |
| split | `chelis_list *chelis_tensor_split(const chelis_tensor *tensor, int32_t axis, const chelis_list *sizes)` |
| gather | `chelis_tensor *chelis_tensor_gather(const chelis_tensor *tensor, const chelis_tensor *indices, int32_t axis)` |
| replace scatter | `chelis_tensor *chelis_tensor_scatter_replace(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis)` |
| additive scatter | `chelis_tensor *chelis_tensor_scatter_add(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis)` |
| comparison | `chelis_tensor *chelis_tensor_cmplt(const chelis_tensor *left, const chelis_tensor *right)` |
| selection | `chelis_tensor *chelis_tensor_where(const chelis_tensor *condition, const chelis_tensor *then_tensor, const chelis_tensor *else_tensor)` |
| inclusive prefix sum | `chelis_tensor *chelis_tensor_cumsum(const chelis_tensor *tensor, int32_t axis)` |
| stable sort | `chelis_tuple *chelis_tensor_sort(const chelis_tensor *tensor, int32_t axis)` |
| diagonal | `chelis_tensor *chelis_tensor_diagonal(const chelis_tensor *tensor, int32_t axis1, int32_t axis2)` |
| trace | `chelis_tensor *chelis_tensor_trace(const chelis_tensor *tensor, int32_t axis1, int32_t axis2)` |
| clamp | `chelis_tensor *chelis_tensor_clamp(const chelis_tensor *tensor, const chelis_tensor *lower, const chelis_tensor *upper)` |
| contraction | `chelis_tensor *chelis_tensor_einsum(chelis_string equation, const chelis_tensor *left, const chelis_tensor *right, chelis_dtype accumulator)` |
| checked coordinate decoding | `void chelis_tensor_unravel_index(const chelis_tensor *tensor, chelis_scalar index, chelis_scalar *coordinates)` |
| checked coordinate encoding | `int64_t chelis_tensor_flat_index(const chelis_tensor *tensor, const chelis_scalar *coordinates)` |
| permutation target validation | `void chelis_tensor_check_permute(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape, const chelis_scalar *axes)` |
| expansion target validation | `void chelis_tensor_check_expand(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape, int32_t axis)` |
| padding shape construction | `void chelis_tensor_pad_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *before, const chelis_scalar *after, chelis_scalar *shape)` |
| shrinking shape construction | `void chelis_tensor_shrink_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *start, const chelis_scalar *end, chelis_scalar *shape)` |
| striding shape construction | `void chelis_tensor_stride_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *steps, chelis_scalar *shape)` |
| affine coordinate projection | `int64_t chelis_tensor_affine_index(const chelis_tensor *tensor, const chelis_scalar *coordinates, const chelis_scalar *offsets, const chelis_scalar *steps)` |
| checked reduction tensor_reduction_plan | `chelis_reduction_plan *chelis_tensor_reduction_plan(const chelis_tensor *tensor, chelis_scalar axis_count, const chelis_scalar *axes, chelis_scalar exemplar, chelis_reduction_op operation)` |
| checked reduction shape_reduction_plan | `chelis_reduction_plan *chelis_shape_reduction_plan(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar axis_count, const chelis_scalar *axes, chelis_scalar exemplar, chelis_reduction_op operation)` |
| checked reduction reduction_count | `int64_t chelis_reduction_count(const chelis_reduction_plan *plan)` |
| checked reduction reduction_extent | `int64_t chelis_reduction_extent(const chelis_reduction_plan *plan, chelis_scalar axis)` |
| checked reduction reduction_index | `int64_t chelis_reduction_index(const chelis_reduction_plan *plan, chelis_scalar outer, chelis_scalar leaf)` |
| checked reduction reduction_check_target | `void chelis_reduction_check_target(const chelis_reduction_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked reduction reduction_check_scratch | `void chelis_reduction_check_scratch(const chelis_reduction_plan *plan, chelis_scalar exemplar)` |
| checked reduction reduction_plan_release | `void chelis_reduction_plan_release(chelis_reduction_plan *plan)` |
| checked sparse tensor_sparse_plan | `chelis_sparse_plan *chelis_tensor_sparse_plan(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, chelis_scalar axis, chelis_sparse_op operation)` |
| checked sparse sparse_extent | `int64_t chelis_sparse_extent(const chelis_sparse_plan *plan, chelis_scalar axis)` |
| checked sparse sparse_count | `int64_t chelis_sparse_count(const chelis_sparse_plan *plan)` |
| checked sparse sparse_index_slot | `int64_t chelis_sparse_index_slot(const chelis_sparse_plan *plan, chelis_scalar linear)` |
| checked sparse sparse_data_index | `int64_t chelis_sparse_data_index(const chelis_sparse_plan *plan, chelis_scalar linear, chelis_scalar selected)` |
| checked sparse sparse_check_target | `void chelis_sparse_check_target(const chelis_sparse_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked sparse sparse_plan_release | `void chelis_sparse_plan_release(chelis_sparse_plan *plan)` |
| checked matrix tensor_matmul_plan | `chelis_matmul_plan *chelis_tensor_matmul_plan(const chelis_tensor *left, const chelis_tensor *right, chelis_scalar exemplar)` |
| checked matrix matmul_extent | `int64_t chelis_matmul_extent(const chelis_matmul_plan *plan, chelis_scalar axis)` |
| checked matrix matmul_dimension | `int64_t chelis_matmul_dimension(const chelis_matmul_plan *plan, chelis_matmul_dimension_kind dimension)` |
| checked matrix matmul_batch_count | `int64_t chelis_matmul_batch_count(const chelis_matmul_plan *plan)` |
| checked matrix matmul_matrix_count | `int64_t chelis_matmul_matrix_count(const chelis_matmul_plan *plan, chelis_matmul_part part)` |
| checked matrix matmul_index | `int64_t chelis_matmul_index(const chelis_matmul_plan *plan, chelis_matmul_part part, chelis_scalar batch, chelis_scalar element)` |
| checked matrix matmul_check_target | `void chelis_matmul_check_target(const chelis_matmul_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked matrix matmul_check_scratch | `void chelis_matmul_check_scratch(const chelis_matmul_plan *plan, chelis_matmul_part part, chelis_scalar exemplar)` |
| checked matrix matmul_check_vendor | `void chelis_matmul_check_vendor(const chelis_matmul_plan *plan, chelis_scalar maximum)` |
| checked matrix matmul_plan_release | `void chelis_matmul_plan_release(chelis_matmul_plan *plan)` |
| checked movement chelis_tensor_permute_plan | `chelis_movement_plan *chelis_tensor_permute_plan(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *axes)` |
| checked movement chelis_tensor_expand_plan | `chelis_movement_plan *chelis_tensor_expand_plan(const chelis_tensor *input, chelis_scalar axis, chelis_scalar size, chelis_movement_op operation)` |
| checked movement chelis_tensor_affine_plan | `chelis_movement_plan *chelis_tensor_affine_plan(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *first, const chelis_scalar *second, chelis_movement_op operation)` |
| checked movement chelis_movement_extent | `int64_t chelis_movement_extent(const chelis_movement_plan *plan, chelis_movement_side side, chelis_scalar axis)` |
| checked movement chelis_movement_count | `int64_t chelis_movement_count(const chelis_movement_plan *plan)` |
| checked movement chelis_movement_index | `int64_t chelis_movement_index(const chelis_movement_plan *plan, chelis_scalar linear)` |
| checked movement chelis_movement_check_target | `void chelis_movement_check_target(const chelis_movement_plan *plan, chelis_scalar rank, const chelis_scalar *shape)` |
| checked movement chelis_movement_plan_release | `void chelis_movement_plan_release(chelis_movement_plan *plan)` |
| checked window tensor_window_plan | `chelis_window_plan *chelis_tensor_window_plan(const chelis_tensor *input, chelis_scalar count, const chelis_scalar *window, const chelis_scalar *steps, chelis_window_op operation)` |
| checked window window_extent | `int64_t chelis_window_extent(const chelis_window_plan *plan, chelis_window_side side, chelis_scalar axis)` |
| checked window window_count | `int64_t chelis_window_count(const chelis_window_plan *plan)` |
| checked window window_index | `int64_t chelis_window_index(const chelis_window_plan *plan, chelis_scalar group, chelis_scalar leaf)` |
| checked window window_check_tensor | `void chelis_window_check_tensor(const chelis_window_plan *plan, const chelis_tensor *tensor, chelis_window_side side)` |
| checked window window_check_target | `void chelis_window_check_target(const chelis_window_plan *plan, chelis_window_side side, chelis_scalar rank, const chelis_scalar *shape)` |
| checked window window_plan_release | `void chelis_window_plan_release(chelis_window_plan *plan)` |
| checked literal tensor_check_literal | `void chelis_tensor_check_literal(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar exemplar, chelis_scalar count)` |
| checked literal tensor_write_literal | `void chelis_tensor_write_literal(chelis_tensor_write *guard, chelis_scalar count, const chelis_scalar *values)` |

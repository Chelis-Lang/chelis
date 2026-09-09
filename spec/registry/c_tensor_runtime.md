# Registry: public C tensor-runtime callables ([05-OP-33])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-33] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| callable | exact C signature |
|---|---|
| owned allocation | `chelis_tensor *chelis_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype)` |
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

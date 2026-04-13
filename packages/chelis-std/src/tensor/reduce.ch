module Std.Tensor.Reduce
export (min, prod, argmax, argmin)
-- Thin wrappers over the Phase 3j-pre Batch 1 IR reductions
-- (commit d79b9ec: MinReduce, ProdReduce, Argmax, Argmin).
-- Each wrapper removes the reduced axis from the input tensor type.
--
-- min: reduce along axis taking the minimum value. The autodiff rule is
-- a first-class equality-mask adjoint mirroring max_reduce; gradient flows
-- only to the minimum element. See crates/chelis-ir/src/grad.rs.
def min[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = min_reduce(x, axis)
-- prod: reduce along axis taking the product. Adjoint uses prefix*suffix
-- products so the gradient stays finite and correct when the reduced slice
-- contains a zero element.
def prod[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = prod_reduce(x, axis)
-- argmax: index of the maximum element along axis.
--
-- CAVEAT: the returned tensor stores indices as integer-valued F32 floats,
-- not Int64 -- the C runtime does not yet carry Int64 tensors. See
-- RiscOp::Argmax in crates/chelis-ir/src/dag.rs and the Batch 1 pin-down
-- test adv_argmax_output_stores_integer_valued_floats. Callers that need
-- proper Int64 indices must cast downstream once backend Int64 tensors
-- land. argmax is non-differentiable; grad through argmax errors
-- cleanly with a named op message instead of returning a silent zero.
def argmax[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = argmax_reduce(x, axis)
-- argmin: index of the minimum element along axis. Same integer-as-F32
-- caveat as argmax; same non-differentiability rule.
def argmin[a, b](x: tensor[a, b, f32], axis: int32) -> tensor[b, f32] = argmin_reduce(x, axis)

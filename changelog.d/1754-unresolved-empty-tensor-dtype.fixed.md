Reject unconstrained `to_tensor([])` before compiled code generation instead of
silently choosing `f32`. Explicitly and contextually typed empty tensors retain
their exact checked dtype, as required by [05-OP-57] (#1754).

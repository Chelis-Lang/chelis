Compile path-sensitive random gradients without differentiating their boolean
activation controls, and preserve runtime `List[tensor]` selection through
compiled recursive adjoint calls.

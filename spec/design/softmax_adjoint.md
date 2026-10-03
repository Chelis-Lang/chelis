# Softmax adjoint implementation

Owning contract: [05-OP-48], with [05-OP-30] reduction finalization. Issue #2990.

Retain a one-input `Softmax { axis }` RISC identity through differentiation and batching. Its exact semantic registration is the existing [05-OP-48]. The adjoint consumes the retained forward output and builds `y * (g - sum(g*y, axis))`, explicitly finalizing the reduction into the operand storage width before subtraction. This prevents a max-shift cotangent path from changing the decided bits. Higher derivatives see the same identity.

A shared post-AD decomposition recreates the existing stable forward graph. Eval decomposes before execution; backend preparation decomposes before ownership binds the emitted graph. Reconstruction preserves declaration ownership, activations, source provenance, shape dependencies and result-claim dependencies. GPU lanes must either use that preparation or reject an undecomposed identity explicitly; they may not silently emit a different graph. No source-name or graph-pattern recognizer selects the adjoint.

Acceptance oracle: `cargo test -p chelis-cli --test issue_2990_softmax_adjoint`. It compares every observed gradient bit against the declared-width formula computed independently from the lane's own forward result, for f16/bf16/f32/f64, tied maxima and rank-two inputs on both axes, in eval and compiled C. Named `vmap(grad(loss))` also compares exact rows, and an independent f64 finite-difference control checks the weighted mathematical loss. Wrong dtype and invalid-axis controls remain rejections. Supporting IR tests cover retained identity, verifier invariants, batching and preservation during decomposition.

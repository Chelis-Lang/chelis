`grad` again differentiates a stack of runtime-sized windows. A reshape whose
target extent is computed at run time names that extent after the scalar node
that produces it, so two reshapes sized from one scalar agree on the extent.
`concat` over such rows stays on its differentiable padding cascade instead of
falling back to the host-runtime lane, where the reduction that followed it
collapsed to rank zero and the backward graph could not be built.

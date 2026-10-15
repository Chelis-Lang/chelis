Preserve caller-declared tensor extent claims through host `grad` and `vmap`
applications, including nested gradient results and multi-output C helpers.
Disagreeing runtime extents trap at their primitive producers before later
effects on the supported result spine.

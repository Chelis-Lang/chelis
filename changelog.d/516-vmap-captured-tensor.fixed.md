`vmap` and `vmap(grad(...))` now keep lexical tensor captures at their
declared rank and explicitly repeat them across the mapped batch axis. Mapped
tensor formals remain batch-varying, and ordinary elementwise operations still
reject mismatched shapes. See [#516](https://github.com/Chelis-Lang/chelis/issues/516).

The C backend emits sequential host bindings with bounded C block depth,
preserving shadowed names and ownership actions. Large host functions no longer
fail solely because their binding count exceeds the native compiler's bracket
limit. See [#3372](https://github.com/Chelis-Lang/chelis/issues/3372).

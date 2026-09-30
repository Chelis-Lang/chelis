`vmap` now keeps gather and scatter indices within each batch element, including
under `grad` and nested batching. Previously a vmapped gather could silently
read the batch axis and return a wrong result shape. See
[#2772](https://github.com/Chelis-Lang/chelis/issues/2772).

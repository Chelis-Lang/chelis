Generic helper and vectorized-call lowering now split rank spreads through the
operational callable shape while preserving the authored binder identity used
by runtime extent witnesses and result claims. Anonymous wildcard axes remain
non-binders, caller labels cannot merge unrelated extents, and direct `vmap`
and `vmap(grad)` can actualize multi-spread signatures at every legal mapped
axis. This repairs the bounded
regressions introduced by #2070; #1889's callable-alias and worker/disk
residuals remain open. See [#1889](https://github.com/Chelis-Lang/chelis/issues/1889).

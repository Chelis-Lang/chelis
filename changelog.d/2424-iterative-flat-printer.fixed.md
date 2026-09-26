The checker no longer overflows the native stack while it prints a deeply
nested program for grad-selector identity. The canonical flat printer, the
Deep-to-wire conversion beneath it, and the release of the converted tree each
recursed once per nesting level outside the checker's stack guard, so a deep
enough program crashed instead of receiving the located `stack budget
exhausted` diagnostic. Builds with debug info crashed this way at the
4,000-level depth-guard fixtures. All three now work from heap stacks, and
printing an unknown form no longer copies its children at every level. The
printed text is unchanged. See
[#2424](https://github.com/Chelis-Lang/chelis/issues/2424).

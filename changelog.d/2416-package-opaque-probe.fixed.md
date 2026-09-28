`chelis prove` in a Reef package now runs a property whose binder is an
invariant-carrying opaque type, as it does for the same file outside a package.
Previously the property's evaluation probe was placed outside the type's
defining module once the program was package-linked, so every such property
failed with ``record construction of opaque type ... outside its defining
module``. The probe now takes its module from the type's own linked name, and
the injected assumption carries the source spelling
(`invariant:Probability:binder:p`). When `prove` rejects a genuine
construction outside the defining module, it now lists the module's exported
producers as `chelis check` does, instead of `none` when the entry module did
not import them. See [#2416](https://github.com/Chelis-Lang/chelis/issues/2416).

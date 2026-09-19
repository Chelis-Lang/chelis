Deep AST annotation storage is now shared behind a reference count and held in
a key-sorted vector rather than a map, so cloning a Deep tree copies one
pointer per node instead of deep-copying every node's annotations, and a
one-to-four-entry annotation map costs one 736-byte allocation rather than the
map's 2048-byte leaf node. A write still copies through `Arc::make_mut`, and
an operation that writes nothing does not copy at all, so the sharing is
invisible. On a 300-definition reef package this cuts `chelis test` CPU by
about a third and recovers just over half the regression measured from
[#1604](https://github.com/Chelis-Lang/chelis/pull/1604). See
[#2117](https://github.com/Chelis-Lang/chelis/issues/2117).
